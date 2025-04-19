# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, streaming: bool):
    """Load C4 dataset with default configuration."""
    logger.info(f"Loading c4 dataset with streaming = {streaming}")
    return load_dataset(dataset_path, name="en", split="train", streaming=streaming, trust_remote_code=True)


def _process_c4_text(sample: Dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]

def _load_bm_dataset(dataset_path: str, streaming: bool, ext: str):
    logger.info(f"Loading benchmark dataset with streaming = {streaming} and ext = {ext}")
    return load_dataset(dataset_path, streaming=streaming, data_files=[f"*.{ext}"], split="train")

def _process_bm_text(sample: Dict[str, Any]) -> str:
    return sample["text"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
# Path gets only used if dataset_path is not set!
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=_load_c4_dataset,
        text_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path, streaming: load_dataset(path, split="train", streaming=streaming),
        text_processor=_process_c4_text,
    ),
    "benchmark_jsonl": DatasetConfig(
        path="/iopsstor/scratch/cscs/mbther/benchmark_data/jsonl",
        loader=lambda path, streaming: _load_bm_dataset(path, streaming, "jsonl"),
        text_processor=_process_bm_text,
    ),
    "benchmark_parquet": DatasetConfig(
        path="/iopsstor/scratch/cscs/mbther/benchmark_data/parquet",
        loader=lambda path, streaming: _load_bm_dataset(path, streaming, "parquet"),
        text_processor=_process_bm_text,
    ),
    "benchmark_jsonlzst": DatasetConfig(
        path="/iopsstor/scratch/cscs/mbther/benchmark_data/jsonl.zst",
        loader=lambda path, streaming: _load_bm_dataset(path, streaming, "jsonl.zst"),
        text_processor=_process_bm_text,
    ),
    "benchmark_webdatasets": DatasetConfig(
        path="/iopsstor/scratch/cscs/mbther/benchmark_data/tar",
        loader=lambda path, streaming: _load_bm_dataset(path, streaming, "tar"),
        text_processor=_process_bm_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        add_bos: bool = True,
        add_eos: bool = True,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path, True)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self._add_bos = add_bos
        self._add_eos = add_eos

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)

                sample_tokens = self._tokenizer.encode(sample_text, bos=self._add_bos, eos=self._add_eos)

                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}

class MappedHuggingFaceDataset(Dataset, Stateful):
    """Mapped dataset using torch.utils.data.Dataset for preprocessed data."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        add_bos: bool = True,
        add_eos: bool = True,
        infinite: bool = False
    ) -> None:
        if infinite:
            raise RuntimeError("Cannot have inifinite mapped dataset.")
        
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path, False)

        self.dataset_name = dataset_name
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self._text_processor = text_processor
        self._add_bos = add_bos
        self._add_eos = add_eos

        # Use split_dataset_by_node to partition the dataset
        self._data = split_dataset_by_node(ds, rank, world_size)

        # Preprocess data: tokenize and flatten
        def tokenize_function(examples):
            sample_text = self._text_processor(examples)
            sample_tokens = self._tokenizer.encode(sample_text, bos=self._add_bos, eos=self._add_eos)
            return {"tokens": sample_tokens}

        self._data = self._data.map(tokenize_function, batched=False, remove_columns=self._data.column_names)

        from itertools import chain

        all_tokens = list(chain.from_iterable(self._data["tokens"]))

        # Truncate to a multiple of seq_len
        total_len = (len(all_tokens) // seq_len) * seq_len
        all_tokens = all_tokens[:total_len]

        # Split into input and label sequences
        inputs = torch.tensor(all_tokens).view(-1, seq_len)
        labels = torch.tensor(all_tokens[1:] + [self._tokenizer.eos_token_id]).view(-1, seq_len)

        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        label_ids = self.labels[idx]
        return {"input": input_ids}, label_ids

    def load_state_dict(self, state_dict):
        # Mapped datasets are stateless in this context
        pass

    def state_dict(self):
        # Mapped datasets are stateless in this context
        return {}

class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(
        self, dp_rank: int, hf_ds: IterableDataset, batch_size: int, world_size: int, num_workers: int
    ):
        super().__init__(hf_ds, batch_size, num_workers=num_workers)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"
        # Data loader resharding is not yet supported, so we need to store the world size to compare during loading
        # raise error if dp_word_size does not match.
        self._world_size = world_size

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self._world_size,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}"
            )
            return
        assert (
            self._world_size == state_dict["world_size"]
        ), "dp_degree is inconsistent before and after checkpoint, dataloader resharding is not supported yet."
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    num_workers: int,
    streaming: bool,
    add_bos: bool,
    add_eos: bool,
    infinite: bool = False,
):
    """Build a data loader for HuggingFace datasets."""
    logger.info(f"building a hf data loader with {num_workers} workers, batch size {batch_size}, seq len {seq_len}, world size {world_size}, rank {rank}, streaming {streaming}, add_bos {add_bos}, add_eos {add_eos}")
    hf_class = HuggingFaceDataset if streaming else MappedHuggingFaceDataset
    hf_ds = hf_class(
        dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, add_bos, add_eos, infinite
    )
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size, world_size=world_size, num_workers=num_workers)

build_hf_dataloader = build_hf_data_loader