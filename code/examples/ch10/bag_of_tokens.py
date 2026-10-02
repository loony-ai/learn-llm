"""Chapter 10.6-10.7: averaging embeddings loses order; position embeddings, used correctly, restore it.

Run from `code/`:  python examples/ch10/bag_of_tokens.py
"""

import torch
from torch import nn

from llmfp.tokenizers import load_tokenizer

torch.manual_seed(0)
tokenizer = load_tokenizer("data/tokenizer/harbor-bpe-2048.json")
first = tokenizer.encode("The keeper fed the gulls")
second = tokenizer.encode("The gulls fed the keeper")
print("same tokens, different order:", sorted(first) == sorted(second), first != second)

tokens = nn.Embedding(tokenizer.vocab_size, 4)
positions = nn.Embedding(8, 4)
layer = nn.Linear(4, 4)


def bag(ids):                       # average of token embeddings
    return tokens(torch.tensor(ids)).mean(dim=0)


def bag_with_positions_averaged_first(ids):   # average of (token + position)
    ids = torch.tensor(ids)
    return (tokens(ids) + positions(torch.arange(len(ids)))).mean(dim=0)


def bag_with_positions_transformed(ids):      # a nonlinear step per token, then average
    ids = torch.tensor(ids)
    return torch.relu(layer(tokens(ids) + positions(torch.arange(len(ids))))).mean(dim=0)


for name, function in [("bag", bag), ("bag + positions, averaged directly", bag_with_positions_averaged_first),
                       ("bag + positions, ReLU layer first", bag_with_positions_transformed)]:
    same = torch.allclose(function(first), function(second))
    print(f"{name:<36} the two sentences look identical: {same}")
