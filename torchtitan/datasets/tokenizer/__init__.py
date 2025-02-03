# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer
from torchtitan.datasets.tokenizer.tokenizer import Tokenizer
from torchtitan.datasets.tokenizer.huggingface import HuggingFaceTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential, after_log, before_log

from torchtitan.logging import logger
import logging

@retry( # instantiating huggingface in CSCS is flaky.
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=2, max=60),
    before=before_log(logger, logging.ERROR),
    after=after_log(logger, logging.ERROR),
    reraise=True,
)
def build_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    if tokenizer_type == "tiktoken":
        logger.info(f"Building {tokenizer_type} tokenizer locally from {tokenizer_path}")
        return TikTokenizer(tokenizer_path)
    else:
        logger.info(f"Building {tokenizer_type} tokenizer using huggingface")
        return HuggingFaceTokenizer(tokenizer_type)
