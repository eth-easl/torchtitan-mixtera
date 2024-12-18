
from mixtera.torch import MixteraTorchDataset

import torch
from torch.utils.data import DataLoader



class MixteraWrapper(torch.utils.data.IterableDataset):
    def __init__(self, torch_ds: MixteraTorchDataset):
        self.torch_ds = torch_ds

    def __iter__(self):
        for key_id, sample in self.torch_ds:
            assert isinstance(key_id, int)
            assert isinstance(sample, list)
            assert isinstance(sample[0], int)
            x = torch.LongTensor(sample)
            input = x[:-1]
            label = x[1:]
            seq_len = len(input)
            key_ids = torch.full((seq_len,), key_id, dtype=torch.long)

            yield input, label, key_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        if "torch_ds" in state:
            del state["torch_ds"]
        return state

def build_mixtera_data_loader(
    mixtera_ds: MixteraTorchDataset,
    batch_size: int,
    num_workers: int,
):
    # TODO this currently assumes we tokenize in Mixtera. Full support for tokenization/no tokenization we will implement later.
    return DataLoader(
        MixteraWrapper(mixtera_ds), # should be put into mixtera
        batch_size=batch_size,
        num_workers=num_workers,
    )
