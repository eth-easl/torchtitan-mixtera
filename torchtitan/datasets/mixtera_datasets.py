
from mixtera.torch import MixteraTorchDataset

import torch
from torch.utils.data import DataLoader
from torchtitan import utils
from torchtitan.logging import logger


class MixteraWrapper(torch.utils.data.IterableDataset):
    def __init__(self, torch_ds: MixteraTorchDataset, return_key_id: bool):
        self.torch_ds = torch_ds
        self.return_key_id = return_key_id

    def __iter__(self):
        for item in self.torch_ds:
            assert (self.return_key_id and isinstance(item, tuple) and len(item) == 2) or (not isinstance(item, tuple)), f"Inconsistent state:\n self.return_key_id = {self.return_key_id}\n item = {item}\n type(item)={type(item)}"

            if self.return_key_id:
                key_id = item[0]
                sample = item[1]
            else:
                sample = item
                key_id = None

            del item
            assert isinstance(key_id, int) or (key_id is None and not self.return_key_id), f"key id = {key_id} sample = {sample} item = {item} return_key_id = {self.return_key_id}"
            assert isinstance(sample, list)
            assert isinstance(sample[0], int)

            x = torch.LongTensor(sample)
            input = x[:-1]
            label = x[1:]
            seq_len = len(input)
            
            if not self.return_key_id:
                yield input, label
            else:
                key_ids = torch.full((seq_len,), key_id, dtype=torch.long) if self.return_key_id else None
                yield input, label, key_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        if "torch_ds" in state: # Not pickable, and is pickled on checkpoint.
            del state["torch_ds"]
        return state

def build_mixtera_data_loader(
    mixtera_ds: MixteraTorchDataset,
    batch_size: int,
    num_workers: int,
    return_key_id: bool
):
    logger.info("Global barrier before building Mixtera dataloader")
    utils.global_barrier()
    logger.info("Global barrier before building Mixtera datalaoder done.")
    # TODO: This currently assumes we tokenize in Mixtera. Full support for tokenization/no tokenization we should implement later.
    return DataLoader(
        MixteraWrapper(mixtera_ds, return_key_id), # should be put into mixtera
        batch_size=batch_size,
        num_workers=num_workers,
    )
