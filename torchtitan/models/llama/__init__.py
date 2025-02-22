# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.components.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer import TikTokenizer
from torchtitan.models.llama.model import Transformer, TransformerModelArgs
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .parallelize_llama import parallelize_llama
from .pipeline_llama import pipeline_llama

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_configs",
]


llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=8, n_heads=16, rope_theta=500000
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "ado": TransformerModelArgs(dim=768, n_layers=12, n_heads=12, max_seq_len=1024),
    "ado1b": TransformerModelArgs(dim=2048, n_layers=24, n_heads=16, max_seq_len=1024, multiple_of=8, ffn_dim_multiplier=None),
    "doremi500": TransformerModelArgs(dim=1024, n_layers=12, n_heads=16, max_seq_len=1024, ffn_dim_multiplier=8192 / 2730), # enforces hidden dim 8192
    "3b": TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        multiple_of=1024,
        rope_theta=500000, # no ffn_dim_multiplier gives intermediate size 8192
    ),
}

llama3_configs["smollm162m"] = llama3_configs["ado"]

register_train_spec(
    TrainSpec(
        name="llama3",
        cls=Transformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        tokenizer_cls=TikTokenizer,
        loss_fn=cross_entropy_loss,
    )
)
