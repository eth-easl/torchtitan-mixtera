from transformers import AutoTokenizer
from typing import List, Optional, Union

from torchtitan.datasets.tokenizer.tokenizer import Tokenizer
from torchtitan.logging import logger


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self._n_words = max(self.tokenizer.vocab_size, len(self.tokenizer)) + 100
        # BOS / EOS token IDs
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        # Initialize stop_tokens if needed
        self.stop_tokens = set()
        if self.eos_id is not None:
            self.stop_tokens.add(self.eos_id)
        logger.info(
            f"HuggingFaceTokenizer built: #words {self.n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}, PAD ID {self.pad_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool = False,
        eos: bool = False,
        max_length: Optional[int] = None,
        truncation: Union[bool, str] = False,
        padding: Union[bool, str] = False,
        **kwargs,
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            max_length (int, optional): If set, will truncate the sequence to the max length.
            truncation (bool | str): Activates and controls truncation. Default is False.
            padding (bool | str): Activates and controls padding. Default is False.
            **kwargs: Additional keyword arguments passed to the tokenizer's encode method.

        Returns:
            list[int]: A list of token IDs.
        """
        tokens = self.tokenizer.encode(
            s,
            add_special_tokens=False,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            **kwargs
        )
        if bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            tokens (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
    
    @property
    def n_words(self) -> int:
        return self._n_words
