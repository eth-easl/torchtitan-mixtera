# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.datasets.hf_datasets import build_hf_data_loader
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer


class TestDatasetCheckpointing:
    def test_c4_resumption(self):
        dataset_name = "c4_test"
        batch_size = 1
        seq_len = 1024
        world_size = 4
        rank = 0

        dl = self._build_dataloader(dataset_name, batch_size, seq_len, world_size, rank)

        it = iter(dl)
        for _ in range(250):
            next(it)
        state = dl.state_dict()
        expected_input_ids, expected_labels = next(it)

        # Create new dataloader, restore checkpoint, and check if next data yielded is the same as above
        dl = self._build_dataloader(dataset_name, batch_size, seq_len, world_size, rank)
        dl.load_state_dict(state)
        input_ids, labels = next(iter(dl))

        assert torch.equal(input_ids["input"], expected_input_ids["input"])
        assert torch.equal(labels, expected_labels)

    def _build_dataloader(
        self, dataset_name, dataset_path, batch_size, seq_len, world_size, rank
    ):
        tokenizer = build_tokenizer("tiktoken", "./tests/assets/test_tiktoken.model")
        return build_hf_data_loader(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=1,
            seq_len=1024,
            world_size=4,
            rank=0,
        )
