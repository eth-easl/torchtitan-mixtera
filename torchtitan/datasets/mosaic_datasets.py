import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from streaming import StreamingDataset
from torchtitan import utils
from torchtitan.datasets.tokenizer import Tokenizer
from filelock import FileLock

from loguru import logger

class MosaicStreamingDataset(IterableDataset):
    def __init__(
        self,
        jsonl_directory: str,
        tokenizer: Tokenizer,
        batch_size: int,
        seq_len: int = 2048,
        infinite: bool = False,
        shuffle: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
        local_cache: Path = Path('/tmp/mosaic')
    ) -> None:
        self.jsonl_directory = jsonl_directory
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.shuffle = shuffle
        self.local_cache = local_cache
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.batch_size = batch_size

        if not local_cache.exists():
            local_cache.mkdir()

        self.index_file_path = os.path.join(jsonl_directory, 'index.json')
        self.index_file_lock_path = os.path.join(jsonl_directory, 'index.json')

        lock = FileLock(self.index_file_lock_path)
        with lock:
            if os.path.exists(self.index_file_path):
                last_modified_time = os.path.getmtime(self.index_file_path)
                current_time = time.time()
                if current_time - last_modified_time > 300:
                    os.remove(self.index_file_path)

            if not os.path.exists(self.index_file_path):
                self.create_index_json()           

        self.dataset = StreamingDataset(local=jsonl_directory, remote=None, download_retry=0, batch_size=batch_size, shuffle=shuffle, replication=None)

        self._sample_idx = 0
        self._all_tokens: list[int] = []

    def create_index_json(self):
        shards = []
        
        jsonl_files = [f for f in os.listdir(self.jsonl_directory) if f.endswith('.jsonl') and "index.json" not in f]
        logger.info(jsonl_files)
        for filename in tqdm(sorted(jsonl_files), desc="Creating Mosaic meta/index files"):
            file_path = os.path.join(self.jsonl_directory, filename)
            file_size = os.path.getsize(file_path)

            samples = [sample for sample in open(file_path)]
            sizes = [len(sample.encode("utf-8")) for sample in samples]
            num_samples = np.uint32(len(samples))
            offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
            meta = {"columns": {"text": "str"}, "compression": None, "format": "json", "hashes": [], "newline": "\n",  "size_limit": 67108864, "version": 2}
            text = json.dumps(meta, sort_keys=True)
            meta_path = f"{file_path}.meta"

            with open(meta_path, 'wb') as meta_file:
                meta_file.write(num_samples.tobytes())
                meta_file.write(offsets.tobytes())
                meta_file.write(text.encode("utf-8"))

            num_samples = len(samples)

            raw_data = {
                "basename": filename,
                "bytes": file_size,
                "hashes": {}
            }

            raw_meta = {
                "basename": f"{filename}.meta",
                "bytes": os.path.getsize(meta_path),
                "hashes": {}
            }
            
            shard_info = {
                "columns": {"text": "str"},
                "compression": None,
                "hashes": [],
                "newline": "\n",
                "raw_data": raw_data,
                "raw_meta": raw_meta,
                "zip_data": None,
                "zip_meta": None,
                "samples": num_samples,
                "size_limit": 67108864,  # 64MB chunk size
                "version": 2,
                "format": "json"
            }
            shards.append(shard_info)

        index_data = {
            "shards": shards,
            "version": 2
        }
        
        with open(self.index_file_path, 'w') as index_file:
            json.dump(index_data, index_file)
        
        logger.info(f"Index file created at {self.index_file_path}")

    def __iter__(self):
        # Identical to huggingface dataset.
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self.dataset:
                sample_text = sample["text"]
                sample_tokens = self.tokenizer.encode(sample_text, bos=self.add_bos, eos=self.add_eos)

                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label

            if not self.infinite:
                logger.warning(f"Dataset has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}

class DPAwareMosaicDataLoader(StatefulDataLoader):
    def __init__(self, mosaic_ds: MosaicStreamingDataset, batch_size: int, num_workers: int):
        super().__init__(mosaic_ds, batch_size, num_workers=num_workers)

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)

def build_mosaic_data_loader(
    jsonl_directory: str,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    num_workers: int,
    add_bos: bool,
    add_eos: bool,
    infinite: bool = False,
    shuffle: bool = False,
):
    logger.info(f"Building a Mosaic data loader with {num_workers} workers, batch size {batch_size}, seq len {seq_len}")
    mosaic_ds = MosaicStreamingDataset(jsonl_directory, tokenizer, batch_size, seq_len, infinite, shuffle, add_bos=add_bos, add_eos=add_eos)
    logger.info("Global barrier!")
    utils.global_barrier()
    return DPAwareMosaicDataLoader(mosaic_ds, batch_size=batch_size, num_workers=num_workers)
