"""Chapter 11.6: padding, attention masks, and keeping padding out of the loss.

Run from `code/`:  python examples/ch11/padding_masks.py
"""

import torch
from torch import nn

from llmfp.data import IGNORE_INDEX, pad_batch

sequences = [[5, 9, 2, 7, 3], [5, 1, 4], [8, 6]]
batch = pad_batch(sequences, pad_id=0)
print("input_ids:\n", batch.input_ids)
print("targets (IGNORE_INDEX = -100 where there is nothing to predict):\n", batch.targets)
print("attention_mask:\n", batch.attention_mask)
print("left padding:\n", pad_batch(sequences, pad_id=0, side="left").input_ids)

# Logits from some model, for every position of the padded batch.
torch.manual_seed(0)
vocab_size = 10
logits = torch.randn(3, 4, vocab_size)                     # (batch, length, vocab)
loss_fn = nn.CrossEntropyLoss()                            # ignore_index defaults to -100

# Cross-entropy expects (examples, vocab) and (examples,): flatten batch and positions together.
padded_loss = loss_fn(logits.reshape(-1, vocab_size), batch.targets.reshape(-1))

# The same loss computed only over real positions, without any padding.
real = batch.attention_mask.reshape(-1)
real_loss = loss_fn(logits.reshape(-1, vocab_size)[real], batch.targets.reshape(-1)[real])
print(f"\nloss with ignored padding: {padded_loss:.4f}   loss on real positions only: {real_loss:.4f}")

# The bug: treating padding as a real target (here, "predict the pad ID").
bugged_targets = batch.targets.clone()
bugged_targets[bugged_targets == IGNORE_INDEX] = 0
bugged = loss_fn(logits.reshape(-1, vocab_size), bugged_targets.reshape(-1))
print(f"loss if padding were a target:  {bugged:.4f}   ({int((~real).sum())} of {real.numel()} positions are padding)")
