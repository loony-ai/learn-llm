"""Chapter 11.8: PyTorch's Dataset and DataLoader.

Run from `code/`:  python examples/ch11/dataloader.py
"""

import torch
from torch.utils.data import DataLoader

from llmfp.data import TokenWindowDataset

dataset = TokenWindowDataset(list(range(100)), context_length=8)
print("windows:", len(dataset))


def first_tokens(loader):
    return [inputs[:, 0].tolist() for inputs, _ in loader]


# batch_size groups windows; shuffle reorders them each epoch; the generator makes shuffling reproducible.
inputs, targets = next(iter(DataLoader(dataset, batch_size=4)))
print("one batch:", tuple(inputs.shape), tuple(targets.shape), inputs.dtype)

loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
print("epoch 1 first tokens:", first_tokens(loader))
print("epoch 2 first tokens:", first_tokens(loader), "(a new order each epoch)")

same_seed = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
print("fresh loader, same seed, same first epoch:", first_tokens(same_seed)[0] == first_tokens(
    DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0)))[0])

print("drop_last=True batch sizes:", [len(x) for x, _ in DataLoader(dataset, batch_size=5, drop_last=True)])
print("drop_last=False batch sizes:", [len(x) for x, _ in DataLoader(dataset, batch_size=5)])
