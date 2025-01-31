import os
from typing import Any
import webdataset as wds
import torch
from torchtitan.datasets.tokenizer import Tokenizer
from loguru import logger
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

class WebDatasetLoader(IterableDataset):
    def __init__(
        self,
        url: str,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        infinite: bool = False,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.url = url
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.add_bos = add_bos
        self.add_eos = add_eos
        self._all_tokens: list[int] = []

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        logger.info(self.url)
        dataset = wds.DataPipeline(
            wds.SimpleShardList(self.url),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.to_tuple("txt"),
        )

        for idx, sample in enumerate(dataset):
            sample = sample[0].decode("utf-8")
            if idx < 3:
                logger.info(sample)
            sample_tokens = self.tokenizer.encode(sample, bos=self.add_bos, eos=self.add_eos)

            self._all_tokens.extend(sample_tokens)

            while len(self._all_tokens) >= max_buffer_token_len:
                x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                self._all_tokens = self._all_tokens[max_buffer_token_len:]
                input = x[:-1]
                label = x[1:]
                yield input, label

        if not self.infinite:
            logger.warning(f"WebDataset has run out of data")
        else:
            logger.warning(f"WebDataset is being re-looped")
            self._all_tokens = []

    def load_state_dict(self, state_dict):
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens}

class DPAwareWebDatasetLoader(StatefulDataLoader):
    def __init__(self, web_ds: WebDatasetLoader, batch_size: int, num_workers: int):
        super().__init__(web_ds, batch_size, num_workers=num_workers)

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)

def construct_webdataset_urls(directory: str) -> str:
    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")
    
    tar_files = sorted([f for f in os.listdir(directory) if f.endswith('.tar')])
    
    if not tar_files:
        raise ValueError(f"No tar files found in the directory '{directory}'.")

    base_url = f"file://{os.path.abspath(directory)}/"
    url_pattern = f"{base_url}{{{','.join(tar_files)}}}"

    return url_pattern


def build_web_data_loader(
    tar_directory: str,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    num_workers: int,
    add_bos: bool,
    add_eos: bool,
    infinite: bool = False,
):
    logger.info(f"Building a WebDataset data loader with {num_workers} workers, batch size {batch_size}, seq len {seq_len}")
    url = construct_webdataset_urls(tar_directory)
    web_ds = WebDatasetLoader(url, tokenizer, seq_len, infinite, add_bos, add_eos)
    return DPAwareWebDatasetLoader(web_ds, batch_size=batch_size, num_workers=num_workers)

