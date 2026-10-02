## Chapter 10: Embeddings and Position

[Back to index](../../README.md) · Previous: [Chapter 9](ch09-byte-pair-encoding.md) · Next: Chapter 11 (planned)

A tokenizer turns text into token IDs. A network needs numbers it can compute with. Chapter 7 bridged the gap with one-hot vectors, and section 7.4 listed their costs: they grow with the vocabulary, they share nothing between related tokens, and they are expensive. Every modern language model uses a different bridge, the **embedding table**: a learned list of numbers for each token. This chapter builds embeddings, trains them, looks at what they learn (honestly, including where they learn less than you might hope), and then addresses a problem that embeddings alone do not solve: **word order**.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain why token IDs cannot be used as numeric inputs, and what one-hot vectors cost at realistic vocabulary sizes.
2. Describe an embedding table as a learned lookup, and show it is equivalent to one-hot times a table.
3. Explain vectors, and what cosine similarity reports, from observed behavior.
4. Train models with embeddings and inspect what the learned tables encode, without over-interpreting them.
5. Show, with a measurement, what a model loses when it cannot see word order.
6. Add learned position embeddings, use them correctly, and state their limitations.

#### Prerequisites

- [Chapter 3.5](../part-1-foundations/ch03-tensors.md): indexing a table with a tensor of IDs (`table[ids]`).
- [Chapter 5](../part-1-foundations/ch05-neural-networks.md): linear layers, activations, output layer.
- [Chapter 7.4](../part-1-foundations/ch07-project-0-char-model.md): one-hot inputs and their costs.
- [Chapter 9](ch09-byte-pair-encoding.md): the Project 1 BPE tokenizer, saved at `data/tokenizer/harbor-bpe-2048.json`.

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Vector | A list of numbers treated as a single item, such as one row of a table | 10.3 |
| Embedding table | A learned table with one row (vector) per token ID | 10.3 |
| Embedding (of a token) | The row of the embedding table for that token | 10.3 |
| Embedding dimension (`d_model`) | How many numbers each embedding has | 10.3 |
| Cosine similarity | A score of how closely two vectors point in the same direction: 1 same, 0 unrelated, -1 opposite | 10.4 |
| Bag of tokens | Treating a context as an unordered collection, losing word order | 10.6 |
| Positional embedding | A learned vector for each position, added to the token embedding so position is visible | 10.7 |
| Context length (of positions) | The number of positions a model has embeddings for; the longest sequence it can take | 10.7 |

---

### 10.1 The problem: token IDs are labels, not measurements

File: [`code/examples/ch10/ids_are_labels.py`](../../code/examples/ch10/ids_are_labels.py)

```python
"""Chapter 10.1: token IDs are labels, not measurements.

Run from `code/`:  python examples/ch10/ids_are_labels.py
"""

from llmfp.tokenizers import load_tokenizer

tokenizer = load_tokenizer("data/tokenizer/harbor-bpe-2048.json")
for word in [" dawn", " dusk", " noon", " night", " keeper", " lamp"]:
    print(f"{word!r:<9} -> ID {tokenizer.encode(word)}")

# IDs reflect the ORDER IN WHICH MERGES WERE LEARNED, which depends on frequency
# in the training corpus. A different corpus, or one more merge early on, renumbers
# everything. Arithmetic on IDs ("dusk minus dawn") therefore means nothing.
dawn, dusk = tokenizer.encode(" dawn")[0], tokenizer.encode(" dusk")[0]
print(f"\nID of ' dusk' minus ID of ' dawn' = {dusk - dawn}: a number with no meaning")
print("The token with the ID halfway between them:", repr(tokenizer.token_text((dawn + dusk) // 2)))
```

Observed output:

```text
' dawn'   -> ID [422]
' dusk'   -> ID [442]
' noon'   -> ID [459]
' night'  -> ID [446]
' keeper' -> ID [381]
' lamp'   -> ID [596]

ID of ' dusk' minus ID of ' dawn' = 20: a number with no meaning
The token with the ID halfway between them: ' fog'
```

The IDs of `' dawn'`, `' dusk'`, `' noon'`, and `' night'` happen to be close together, because the Project 1 tokenizer learned those merges at a similar point in training (Chapter 9.4): they are similarly frequent in the corpus. That is an accident of the merge order, not a property of the words. The difference between two IDs, or the token "halfway" between them, means nothing. Retrain the tokenizer on slightly different text and every ID changes.

A network multiplies inputs by weights (Chapter 5.2). Fed raw IDs, it would treat token 596 as "more" than token 422. So token IDs must be used as *labels*: to look something up, never as quantities. Chapter 7 looked up a one-hot vector. This chapter looks up something much better.

---

### 10.2 One-hot inputs revisited, and their cost

A one-hot vector for a vocabulary of size V is V numbers with a single 1. The first linear layer then has one weight per (input position, vocabulary entry, hidden unit). With the Project 1 tokenizer's 2,048 tokens, a context of 8 tokens, and 128 hidden units, the milestone script in section 10.5 reports what that layer would need: **2,097,152 weights**, before learning anything else. With a 50,000-token vocabulary and a 1,000-token context, the same design would need billions of weights in the first layer alone.

There is a second, subtler cost. Every vocabulary entry has its own separate weights, and every one-hot vector is equally different from every other. Nothing lets what the model learns about `' dawn'` help with `' dusk'`. Each token must be learned from its own examples.

---

### 10.3 Embedding tables: a learned lookup

An **embedding table** gives every token ID one row of numbers, `d_model` numbers long (the **embedding dimension**). A list of numbers treated as one item like this is called a **vector**, and a token's row is its **embedding**. Using the table is a lookup: the IDs select rows. The numbers in the rows are parameters, set randomly at first and adjusted by training like any others.

File: [`code/examples/ch10/embedding_lookup.py`](../../code/examples/ch10/embedding_lookup.py)

```python
"""Chapter 10.3: an embedding table is a learned lookup, equivalent to one-hot times a table.

Run from `code/`:  python examples/ch10/embedding_lookup.py
"""

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
table = nn.Embedding(num_embeddings=6, embedding_dim=3)     # 6 tokens, 3 numbers each
print("table.weight shape:", tuple(table.weight.shape), "-> one row per token ID")

ids = torch.tensor([[4, 1, 4]])                             # (batch=1, seq=3)
vectors = table(ids)
print("ids shape", tuple(ids.shape), "-> vectors shape", tuple(vectors.shape))
print("row for ID 4 appears twice:", torch.equal(vectors[0, 0], vectors[0, 2]), "and equals table row 4:",
      torch.equal(vectors[0, 0], table.weight[4]))

# The same result via Chapter 7's one-hot route: one-hot (1, 3, 6) times the table (6, 3).
one_hot = F.one_hot(ids, num_classes=6).float()
print("one-hot @ table gives the same vectors:", torch.allclose(one_hot @ table.weight, vectors))

# Training updates only the rows that were looked up.
vectors.sum().backward()
used = [i for i in range(6) if table.weight.grad[i].abs().sum() > 0]
print("rows that received a gradient:", used, "(IDs 1 and 4; the rest are untouched)")

# An ID outside the table is an error, a common symptom of a tokenizer/model mismatch.
try:
    table(torch.tensor([6]))
except IndexError as error:
    print("ID 6 in a 6-row table ->", type(error).__name__ + ":", error)
```

Observed output:

```text
table.weight shape: (6, 3) -> one row per token ID
ids shape (1, 3) -> vectors shape (1, 3, 3)
row for ID 4 appears twice: True and equals table row 4: True
one-hot @ table gives the same vectors: True
rows that received a gradient: [1, 4] (IDs 1 and 4; the rest are untouched)
ID 6 in a 6-row table -> IndexError: index out of range in self
```

What to take from it:

- **Shapes.** `nn.Embedding(6, 3)` has a weight of shape `(6, 3)`: one row per token. IDs of shape `(batch, seq)` become vectors of shape `(batch, seq, 3)`. Every token position now carries a vector, which is the `(batch, sequence, features)` layout Chapter 3.3 said transformers use.
- **Equivalence with one-hot.** One-hot vectors multiplied by the table give exactly the same result. An embedding layer is a linear layer applied to one-hot inputs, implemented as a lookup so the one-hot vectors never have to exist. That is why it is cheap: the cost depends on `d_model`, not on the vocabulary size.
- **Only looked-up rows learn.** The backward pass gave gradients only to rows 1 and 4, the IDs that were used. A token that never appears in the training data keeps its random starting row forever. Section 10.5 has to account for this when inspecting embeddings, and Part 4 discusses why a tokenizer trained on different text than the model can leave many such untrained rows.
- **IDs must fit the table.** An ID equal to or beyond the number of rows is an `IndexError`. When you see this error in model code, the first suspect is a tokenizer whose vocabulary is larger than the model's embedding table (Chapter 8.6).

The book's embedding module, which Part 3's transformer will use:

File: [`code/llmfp/model/embeddings.py`](../../code/llmfp/model/embeddings.py)

```python
"""Token embeddings, learned position embeddings, and similarity helpers (Chapter 10).

An embedding table is a learned lookup: row i is the list of numbers that
represents token i. Looking up a row is equivalent to multiplying a one-hot
vector by the table (Chapter 7.4), but without building the one-hot vector.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TokenAndPositionEmbedding(nn.Module):
    """token IDs (batch, seq) -> vectors (batch, seq, d_model).

    Each position's vector is its token's embedding plus its position's embedding,
    so the same token at different positions gets different vectors.
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context_length, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence of {sequence_length} tokens is longer than the {self.context_length} positions this model has"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)   # 0, 1, ..., seq - 1
        return self.token(token_ids) + self.position(positions)              # (batch, seq, d) + (seq, d): broadcast


def cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between every pair of rows: 1 = same direction, 0 = unrelated, -1 = opposite."""
    unit = F.normalize(vectors, dim=-1)   # rescale each row to length 1, keeping its direction
    return unit @ unit.T


def nearest_neighbors(table: torch.Tensor, index: int, k: int = 5) -> list[tuple[int, float]]:
    """The k rows most similar (by cosine similarity) to row `index`, excluding itself."""
    similarities = F.cosine_similarity(table[index].unsqueeze(0), table, dim=-1)
    similarities[index] = -2.0   # below any possible similarity, so it is never chosen
    top = similarities.topk(k)
    return list(zip(top.indices.tolist(), top.values.tolist()))
```

`TokenAndPositionEmbedding` is explained in section 10.7, and the two similarity helpers in section 10.4. The package's `__init__.py` lists what it holds:

File: [`code/llmfp/model/__init__.py`](../../code/llmfp/model/__init__.py)

```python
"""Model components, built up from Chapter 10 to Chapter 17.

    embeddings.py     token and position embeddings, similarity helpers (Chapter 10)
    embedding_mlp.py  a next-token model used to study embeddings and order (Chapter 10)

Part 3 adds attention, transformer blocks, and the GPT model.
"""
```

---

### 10.4 Vectors as lists of learned features; similarity and cosine similarity

Why would a list of 32 numbers per token help a model more than an ID? Because training adjusts each token's numbers to whatever makes the model's predictions better. Tokens that call for similar predictions tend to be pushed toward similar numbers, and then everything the model computes from those numbers treats them similarly. In principle, what the model learns from `' dawn'` can then carry over to `' dusk'`.

It is tempting to think of each of the 32 numbers as a "feature" with a meaning, such as "is a time of day" or "is plural". **That picture is mostly a teaching analogy.** Individual dimensions of learned embeddings rarely correspond to a single human concept; whatever structure exists is usually spread across many dimensions at once. Research on interpreting learned representations is active, and its findings are partial (Chapter 1.12).

What we *can* measure reliably is whether two vectors point in a similar direction. **Cosine similarity** does that:

File: [`code/examples/ch10/cosine_similarity.py`](../../code/examples/ch10/cosine_similarity.py)

```python
"""Chapter 10.4: what cosine similarity reports, by observation.

Run from `code/`:  python examples/ch10/cosine_similarity.py
"""

import torch
from torch.nn import functional as F

a = torch.tensor([1.0, 2.0, 0.5])
cases = {
    "itself": a,
    "same direction, 10x longer": a * 10,
    "slightly different": torch.tensor([1.1, 1.9, 0.6]),
    "unrelated (at right angles)": torch.tensor([2.0, -1.0, 0.0]),
    "opposite direction": -a,
}
for label, b in cases.items():
    print(f"{label:<28} cosine similarity {F.cosine_similarity(a, b, dim=0).item():6.3f}")
```

Observed output:

```text
itself                       cosine similarity  1.000
same direction, 10x longer   cosine similarity  1.000
slightly different           cosine similarity  0.997
unrelated (at right angles)  cosine similarity  0.000
opposite direction           cosine similarity -1.000
```

Cosine similarity is 1 for vectors pointing the same way (even if one is ten times longer: length is ignored), close to 1 for slightly different directions, 0 for vectors at right angles, which in practice means "unrelated", and -1 for opposite directions. It is the standard way to compare embeddings, and you will use it again for search in Chapter 32. In `embeddings.py`, `cosine_similarity_matrix` first rescales every vector to length 1 (`F.normalize`), then compares all pairs with one matrix multiplication; `nearest_neighbors` finds the rows most similar to a given row.

---

### 10.5 Training embeddings and inspecting what they learn

The milestone model, `EmbeddingMLP`, is a next-token predictor over the Project 1 tokenizer's vocabulary. It looks up embeddings for the previous 8 tokens and combines them in one of three ways (`mode`), studied in sections 10.6 and 10.7:

File: [`code/llmfp/model/embedding_mlp.py`](../../code/llmfp/model/embedding_mlp.py)

```python
"""A next-token model for studying embeddings and word order (Chapter 10).

Three ways to combine a context of token embeddings before predicting:

    "concat"         place the embeddings side by side, then the hidden layer
                     (order is kept by each embedding's place in the list)
    "bag"            pass each embedding through the hidden layer on its own, then average
                     (order is lost: a "bag of tokens")
    "bag_position"   add a learned position embedding to each token first, then as "bag"

Why the hidden layer comes BEFORE averaging in the bag modes: averaging
"token + position" directly would equal (average of tokens) + (average of
positions), and the second part is the same for every context, so the order
information would cancel out. Exercise 10.5 explores that mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from llmfp.model.embeddings import TokenAndPositionEmbedding

MODES = ("concat", "bag", "bag_position")


@dataclass(frozen=True)
class EmbeddingMLPConfig:
    vocab_size: int
    context_size: int = 8
    d_model: int = 32
    hidden: int = 128
    mode: str = "concat"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


class EmbeddingMLP(nn.Module):
    def __init__(self, config: EmbeddingMLPConfig) -> None:
        super().__init__()
        self.config = config
        if config.mode == "bag_position":
            self.embed = TokenAndPositionEmbedding(config.vocab_size, config.context_size, config.d_model)
        else:
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
        hidden_inputs = config.d_model * (config.context_size if config.mode == "concat" else 1)
        self.hidden = nn.Linear(hidden_inputs, config.hidden)
        self.output = nn.Linear(config.hidden, config.vocab_size)

    @property
    def token_table(self) -> torch.Tensor:
        """The learned token embedding table, (vocab_size, d_model)."""
        module = self.embed.token if self.config.mode == "bag_position" else self.embed
        return module.weight

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        vectors = self.embed(contexts)                                  # (batch, context, d_model)
        if self.config.mode == "concat":
            hidden = torch.relu(self.hidden(vectors.flatten(start_dim=1)))  # (batch, hidden)
        else:
            per_token = torch.relu(self.hidden(vectors))                 # (batch, context, hidden)
            hidden = per_token.mean(dim=1)                               # (batch, hidden): averaged
        return self.output(hidden)
```

The script trains all three on harbor text, then examines the first model's learned tables:

File: [`code/configs/embedding-mlp-cpu.toml`](../../code/configs/embedding-mlp-cpu.toml)

```toml
# Chapter 10: next-token models with learned embeddings, on BPE tokens.
runs_root = "runs"
run_name = "ch10-embeddings"
data = "data/tiny/harbor_synth.txt"
tokenizer = "data/tokenizer/harbor-bpe-2048.json"
seed = 0
context_size = 8
d_model = 32
hidden = 128
learning_rate = 0.003
epochs = 10
batch_size = 128
modes = ["concat", "bag", "bag_position"]
probe_tokens = [" keeper", " dawn", " lit", " lamp"]
```

File: [`code/scripts/ch10_train_embedding_model.py`](../../code/scripts/ch10_train_embedding_model.py)

```python
"""Chapter 10 milestone: learned embeddings, and what is lost without word order.

Trains three next-token models on harbor text encoded with the Project 1 BPE
tokenizer, identical except for how they combine the context's embeddings.
Then, for the first model, compares how similar tokens with the same role
(actors, times, verbs) are in two learned tables: the input embedding table
and the output layer (one row per candidate next token).

Run from `code/`:
    python -m scripts.ch10_train_embedding_model
    python -m scripts.ch10_train_embedding_model --set "modes=[\\"concat\\"]" --set d_model=8
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from llmfp.char_model import make_examples
from llmfp.config import ConfigError, load_config
from llmfp.counting_lm import read_lines
from llmfp.devices import add_device_argument, pick_device
from llmfp.experiment import finish_run, set_seed, start_run
from llmfp.model.embedding_mlp import EmbeddingMLP, EmbeddingMLPConfig
from llmfp.model.embeddings import cosine_similarity_matrix, nearest_neighbors
from llmfp.nn_basics import count_parameters
from llmfp.splits import deduplicate, hash_split
from llmfp.tokenizers import load_tokenizer
from llmfp.training_basics import fit


# Harbor tokens grouped by the role they play in sentences (from the corpus templates).
ROLE_GROUPS = {
    "actors": [" keeper", " fishers", " children", " gulls", " boats", " master"],
    "times": [" dawn", " dusk", " noon", " night"],
    "verbs": [" lit", " rang", " mended", " sold", " wrote", " fed", " opened", " followed"],
}


def role_similarity(table: torch.Tensor, tokenizer) -> dict[str, float]:
    """Average cosine similarity within each role group, and between different groups."""
    ids = [tokenizer.encode(word)[0] for words in ROLE_GROUPS.values() for word in words]
    labels = [group for group, words in ROLE_GROUPS.items() for _ in words]
    similarity = cosine_similarity_matrix(table[ids])
    pairs = [(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    result = {}
    for group in ROLE_GROUPS:
        values = [similarity[i, j].item() for i, j in pairs if labels[i] == labels[j] == group]
        result[f"within {group}"] = sum(values) / len(values)
    values = [similarity[i, j].item() for i, j in pairs if labels[i] != labels[j]]
    result["between groups"] = sum(values) / len(values)
    return result


@dataclass(frozen=True)
class EmbeddingRunConfig:
    runs_root: str = "runs"
    run_name: str = "ch10-embeddings"
    data: str = "data/tiny/harbor_synth.txt"
    tokenizer: str = "data/tokenizer/harbor-bpe-2048.json"
    seed: int = 0
    context_size: int = 8
    d_model: int = 32
    hidden: int = 128
    learning_rate: float = 0.003
    epochs: int = 10
    batch_size: int = 128
    modes: list[str] = field(default_factory=lambda: ["concat", "bag", "bag_position"])
    probe_tokens: list[str] = field(default_factory=lambda: [" keeper", " dawn", " lit", " lamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/embedding-mlp-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    add_device_argument(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        config = load_config(EmbeddingRunConfig, args.config, args.overrides)
        device = pick_device(args.device)
    except (ConfigError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")

    set_seed(config.seed)
    run_dir = start_run(config.runs_root, config.run_name, config, data_files=[config.data, config.tokenizer])
    tokenizer = load_tokenizer(config.tokenizer)
    splits = hash_split(deduplicate(read_lines(config.data)), salt=str(config.seed))
    train_ids = tokenizer.encode("\n".join(splits.train) + "\n")
    val_ids = tokenizer.encode("\n".join(splits.validation) + "\n")
    train = tuple(t.to(device) for t in make_examples(train_ids, config.context_size))
    val = tuple(t.to(device) for t in make_examples(val_ids, config.context_size))
    seen = sorted(set(train_ids))
    print(f"Run: {run_dir}")
    print(f"Tokens: {len(train_ids):,} training, {len(val_ids):,} validation; "
          f"{len(seen)} of {tokenizer.vocab_size} vocabulary entries occur in training")
    one_hot_weights = config.context_size * tokenizer.vocab_size * config.hidden
    print(f"(A one-hot input layer for this context would need {one_hot_weights:,} weights.)\n")

    print(f"{'mode':<13} {'parameters':>10} | {'val loss':>8} | {'val acc':>7}")
    models = {}
    for mode in config.modes:
        set_seed(config.seed)
        model_config = EmbeddingMLPConfig(tokenizer.vocab_size, config.context_size, config.d_model, config.hidden, mode)
        model = EmbeddingMLP(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        history = fit(model, train, val, nn.CrossEntropyLoss(), optimizer, config.epochs, config.batch_size, config.seed)
        last = history[-1]
        print(f"{mode:<13} {count_parameters(model):>10,} | {last['val_loss']:>8.4f} | {last['val_accuracy']:>7.1%}")
        models[mode] = (model, history)

    # Where does role-like structure appear? Compare the input table with the output layer.
    mode = config.modes[0]
    model = models[mode][0]
    tables = {"input embeddings": model.token_table.detach().cpu(), "output layer rows": model.output.weight.detach().cpu()}
    print(f"\nAverage cosine similarity by role, '{mode}' model:")
    print(f"  {'':<18}" + "".join(f"{name:>19}" for name in tables))
    summaries = {name: role_similarity(table, tokenizer) for name, table in tables.items()}
    for key in summaries["input embeddings"]:
        print(f"  {key:<18}" + "".join(f"{summaries[name][key]:>19.2f}" for name in tables))

    # Nearest neighbors, among tokens that occurred in training (other rows were never updated).
    position_of = {token_id: i for i, token_id in enumerate(seen)}
    for name, table in tables.items():
        print(f"\nNearest neighbors by {name}:")
        for text in config.probe_tokens:
            ids = tokenizer.encode(text)
            if len(ids) != 1 or ids[0] not in position_of:
                print(f"  {text!r}: not a single token seen in training")
                continue
            neighbors = nearest_neighbors(table[seen], position_of[ids[0]], k=4)
            shown = ", ".join(f"{tokenizer.token_text(seen[i])!r} {s:.2f}" for i, s in neighbors)
            print(f"  {text!r:<10} -> {shown}")

    finish_run(run_dir, {mode: history for mode, (_, history) in models.items()})


if __name__ == "__main__":
    main()
```

```bash
python -m scripts.ch10_train_embedding_model
```

Observed output (about 15 seconds on the test machine):

```text
Run: runs/ch10-embeddings/20261002-235501
Tokens: 10,076 training, 1,362 validation; 68 of 2048 vocabulary entries occur in training
(A one-hot input layer for this context would need 2,097,152 weights.)

mode          parameters | val loss | val acc
concat           362,624 |   0.7517 |   71.0%
bag              333,952 |   1.1754 |   65.1%
bag_position     334,208 |   0.7340 |   72.2%

Average cosine similarity by role, 'concat' model:
                       input embeddings  output layer rows
  within actors                   -0.03               0.31
  within times                    -0.04               0.70
  within verbs                     0.01               0.19
  between groups                  -0.01              -0.01

Nearest neighbors by input embeddings:
  ' keeper'  -> ' warned' 0.45, ' fog' 0.37, ' after' 0.37, ' storm' 0.35
  ' dawn'    -> ' followed' 0.51, ' children' 0.42, ' ran' 0.40, ' opened' 0.32
  ' lit'     -> ' stairs' 0.39, ' for' 0.37, ' cleaned' 0.36, ' left' 0.27
  ' lamp'    -> ' watched' 0.41, ' fog' 0.36, ' keeper' 0.34, ' pier' 0.26

Nearest neighbors by output layer rows:
  ' keeper'  -> ' children' 0.63, ' boats' 0.60, ' fishers' 0.55, ' harbor' 0.53
  ' dawn'    -> ' night' 0.77, ' dusk' 0.76, ' noon' 0.68, ' children' 0.20
  ' lit'     -> ' climbed' 0.75, ' fixed' 0.72, ' rang' 0.64, ' cleaned' 0.62
  ' lamp'    -> ' oil' 0.46, ' glass' 0.41, ' stairs' 0.37, ' crates' 0.31
```

First, the cost. The `concat` model has about 363,000 parameters in total, while a one-hot input layer for the same context would need about 2.1 million weights by itself. Most of the remaining parameters are in the output layer, whose 2,048 rows are mostly for tokens that never occur in harbor text: only 68 of the 2,048 vocabulary entries appear in training. That is a cost of using a tokenizer trained on broader text than the model's data.

Now the inspection, which is more interesting for what it does *not* show.

**The input embeddings show essentially no grouping by role.** Average cosine similarity among actors (keeper, fishers, children, gulls, boats, master), among times of day, and among verbs is about zero, the same as between groups. The nearest neighbors of `' keeper'` in the input table are a jumble (`' warned'`, `' fog'`, `' after'`). In this small model, trained on templated text, the input embeddings did not organize themselves by meaning.

**The output layer's rows group strongly.** The output layer has one row per candidate next token (Chapter 5.6). Its rows are also learned vectors, one per token, and they *are* organized: times of day average 0.70 similarity with each other, and the nearest neighbors of `' dawn'` are `' night'`, `' dusk'`, and `' noon'`. Those of `' keeper'` are the other actors, and those of `' lit'` are other things the keeper does (climbed, fixed, rang, cleaned).

Why the difference? The output layer scores every candidate next token from the same hidden values. Tokens that are likely in the same situations ("after `at`, a time of day comes next") need similar rows, so that they score high together; training pushes them together directly. Input embeddings, in the `concat` model, feed into position-specific weights in the hidden layer, so there is more than one way to make predictions work, and nothing forced a tidy arrangement.

Three lessons, all of which recur later:

1. **Learned representations can encode useful structure, and where it appears depends on the architecture and the data.** Large models trained on huge corpora do show strong structure in their input embeddings; this tiny model on templated text did not. Do not generalize from either case to the other.
2. **Inspect, measure, and report what you find, including null results.** A neighbor list from one well-chosen word can be made to look meaningful. The averaged group comparison is harder to fool yourself with.
3. **Input and output tables are both "one vector per token".** Chapter 16.5 shows a common design that uses one table for both (weight tying).

---

### 10.6 Why order matters: what a bag of tokens loses

Embeddings give each token a vector, but a context is a *sequence* of tokens. How should a model combine them?

The `concat` mode places the vectors side by side, so the first token's numbers always go to the same hidden-layer weights: order is preserved by position in the list, as with one-hot inputs. This has the position-specific cost from Chapter 7.4 and works only for a fixed number of tokens.

A tempting alternative is to combine the vectors in a way that does not depend on how many there are, such as averaging them. That treats the context as a **bag of tokens**: an unordered collection. The problem is that order carries meaning:

File: [`code/examples/ch10/bag_of_tokens.py`](../../code/examples/ch10/bag_of_tokens.py)

```python
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
```

Observed output:

```text
same tokens, different order: True True
bag                                  the two sentences look identical: True
bag + positions, averaged directly   the two sentences look identical: True
bag + positions, ReLU layer first    the two sentences look identical: False
```

"The keeper fed the gulls" and "The gulls fed the keeper" contain the same tokens. To a bag model they are identical. The milestone measured the cost on prediction: the `bag` model reached about 65% validation accuracy against about 71% for `concat`, and a noticeably higher loss. It cannot tell "the keeper" (followed by a keeper action) from "keeper the".

The `bag` mode in `EmbeddingMLP` passes each token's embedding through the hidden layer separately and then averages. Each token is processed identically wherever it sits, so the result cannot depend on order.

---

### 10.7 Learned positional embeddings

The fix used by GPT-2 and many transformers: make position part of each token's vector. A second embedding table has one row per **position** (0, 1, 2, ... up to the maximum sequence length). Each token's vector becomes its token embedding *plus* its position's embedding. The same token at two positions now has two different vectors, and anything computed from them can depend on position.

`TokenAndPositionEmbedding` (section 10.3's listing) does exactly this. `torch.arange(sequence_length)` produces the position IDs 0, 1, 2, ...; the position table turns them into vectors of shape `(seq, d_model)`; broadcasting (Chapter 3.7) adds them to every sequence in the batch. The `bag_position` mode uses it, and it changed the result substantially: about 72% validation accuracy, from 65% for the plain bag. Position information restored what the bag lost, and here even slightly beat `concat`.

#### The averaging trap

Position embeddings only help if the model does something with them *before* the information is averaged away. The last two lines of the bag example show it:

- "bag + positions, averaged directly": the two sentences still look identical.
- "bag + positions, ReLU layer first": they now differ.

Averaging the sums of token and position vectors gives the average of the token vectors plus the average of the position vectors, and the second part is the same for every context of that length. It is a constant; the order information cancels out. A nonlinear step applied to each token's vector *before* combining keeps it, because the activation (Chapter 5.4) responds to the token and the position together. The author's first draft of the milestone made exactly this mistake; Exercise 5 has you reproduce it, and a test (`test_averaging_first_cancels_position_information`) guards against it. In a transformer, the nonlinear, position-sensitive step is attention itself (Chapter 13), which weighs tokens differently depending on their vectors, including the position part.

#### Limitations of learned positions

- **A fixed maximum length.** There are only `context_length` rows. A longer sequence has no position vectors, and `TokenAndPositionEmbedding` raises an error rather than failing obscurely. A model's maximum context is often set this way, at training time.
- **Rarely seen positions are poorly trained.** If most training sequences are short, embeddings for late positions get few updates (only looked-up rows learn, section 10.3).
- **Absolute, not relative.** "Position 7" means the same in every sequence, but what usually matters in language is relative position: the word just before, three words back. A model with absolute positions has to learn relative patterns separately for every position.

---

### 10.8 Preview of other positional methods

Learned absolute positions are one of several designs, an area where practice has changed over time:

- **Fixed (sinusoidal) position vectors**, used in the original transformer: the position vectors are computed from a fixed recipe instead of being learned, so they exist for any position, though models still tend to perform poorly on sequences much longer than those seen in training.
- **Rotary positional embeddings (RoPE)**, used by many recent open models: instead of adding a position vector, they rotate parts of the vectors used inside attention by an amount that depends on position, so that attention scores depend on *relative* distance. Chapter 15.7 implements RoPE and shows its behavior in code.
- **Relative position biases** in attention, and several other variants, discussed at an overview level in Chapter 43.4 together with techniques for extending context length.

Which works best is an empirical question that depends on model size, training data, and the lengths you need. This book builds learned absolute positions first, because they are the simplest to understand, then RoPE.

---

### 10.9 Common mistakes, recap, concept checks, exercises, answers, and checkpoint

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| `IndexError: index out of range in self` in an embedding | A token ID at or beyond the table size: tokenizer and model mismatched, or a padding/special ID not included | Check `tokenizer.vocab_size` against the embedding's number of rows |
| Error or nonsense for long inputs | Sequence longer than the position table | Truncate or window inputs (Chapter 11); know the model's context length |
| Model ignores word order | Combining token vectors with an order-insensitive operation, or adding positions and then averaging immediately | Concatenate, or apply a nonlinear (or attention) step to token-plus-position vectors before combining |
| Embedding inspection shows meaningful-looking neighbors for untrained tokens | Rows for tokens absent from training are still random | Restrict analysis to tokens seen in training |
| Strong claims from one nearest-neighbor list | Cherry-picking | Measure averaged similarities across groups and report null results |
| Raw IDs fed into a linear layer | Treating labels as quantities | Always look IDs up in an embedding table |

#### Recap

- Token IDs are labels; their numeric values carry no meaning.
- One-hot inputs are equivalent to an embedding lookup but cost a weight per vocabulary entry per input position; embedding tables cost `d_model` numbers per token and are looked up directly.
- Only rows that are looked up receive gradients; tokens absent from training keep random rows.
- Cosine similarity compares direction: 1 same, 0 unrelated, -1 opposite. Individual embedding dimensions rarely have a human meaning.
- In the milestone model, the output layer's rows grouped tokens by role strongly; the input embeddings did not. Where structure appears depends on the model and the data.
- A **bag of tokens** loses order and measurably worse predictions follow. **Positional embeddings** restore order, but only if a nonlinear step sees token and position together before combining.
- Learned absolute positions limit the maximum sequence length and treat positions absolutely; RoPE (Chapter 15.7) is a common alternative.

#### Concept checks

1. Why can't token IDs be fed to a linear layer as numbers?
2. What is the shape of `nn.Embedding(2048, 32)` applied to IDs of shape `(16, 8)`?
3. Explain why an embedding lookup gives the same result as one-hot vectors times the table, and why the lookup is cheaper.
4. After one training step on a batch containing tokens 5 and 9, which rows of the embedding table changed?
5. Two embedding vectors have cosine similarity 0.98. What does that tell you, and what does it not tell you?
6. In the milestone, why did the output layer's rows group times of day together?
7. Why were nearest neighbors computed only among tokens seen in training?
8. "The keeper fed the gulls" and "The gulls fed the keeper": which of the three modes can tell them apart?
9. Why does averaging token-plus-position vectors directly not preserve order?
10. A model has 256 position embeddings. What happens if you give it 300 tokens?
11. Why might "absolute" positions be less natural for language than relative ones?
12. What happens to a token's embedding if that token never appears in the training data?

#### Exercises

**Exercise 1 (shapes).** For `TokenAndPositionEmbedding(vocab_size=2048, context_length=64, d_model=48)`, predict: the number of parameters, the output shape for IDs of shape `(4, 64)`, and what happens for shape `(4, 65)`. Check with code.

**Exercise 2 (cosine by hand).** Without code, decide which pairs have cosine similarity 1, 0, or -1: `[1, 1]` and `[3, 3]`; `[1, 0]` and `[0, 5]`; `[2, -1]` and `[-4, 2]`; `[1, 2]` and `[2, 1]` (this one is none of them: is it closer to 1 or 0?). Check with `F.cosine_similarity`.

**Exercise 3 (embedding size).** Rerun the milestone with `d_model` 4, 16, 32, and 64 (`--set d_model=4`). Record validation accuracy and the role-similarity table for each. Does a larger embedding help accuracy here? Does it change where role structure appears?

**Exercise 4 (seen and unseen rows).** After training the `concat` model, compute the average size of the embedding rows (the length of each vector, `row.norm()`) for tokens that occurred in training and for those that did not. Explain the difference, and the role of weight decay in AdamW (Chapter 6.6) in it.

**Exercise 5 (the averaging trap).** Write a variant of the `bag_position` model that averages the token-plus-position vectors *before* the hidden layer. Train it with the same settings and compare its validation accuracy with `bag` and with the correct `bag_position`. Write a test showing it gives identical outputs for a context and its reverse.

**Exercise 6 (relative position, informally).** Train the `concat` model with `context_size` 2, 4, 8, and 16. Explain the trend in accuracy using what you know about the corpus, and say which positions in the context the model most needs for predicting the next token. (Optional: measure it by shuffling one position at a time in the validation contexts and recording the accuracy drop.)

#### Suggested answers and acceptance criteria

**Concept checks**

1. The layer multiplies inputs by weights, so it would treat a larger ID as "more" of something, while IDs are arbitrary labels.
2. `(16, 8, 32)`.
3. Multiplying a one-hot vector by the table picks out exactly the row at the 1; the lookup picks the same row directly, without building a vector of thousands of zeros or multiplying by them.
4. Only rows 5 and 9 (plus, through AdamW's weight decay, a tiny shrink of all rows if decay is applied to the table).
5. They point in almost the same direction, so the model will tend to treat them similarly in computations that depend on direction. It does not tell you the tokens mean the same thing, nor which dimensions are responsible.
6. Every candidate is scored from the same hidden values, and times of day are likely in the same situations (after "at"), so training pushes their rows toward similar directions so they score high together.
7. Rows for tokens never seen are never updated and stay random, so they would add meaningless neighbors.
8. `concat` and `bag_position`. `bag` cannot.
9. The average of sums is the sum of averages, and the average of the position vectors is the same for every context, a constant that carries no order information.
10. There is no position vector for positions 256 and beyond; the book's module raises an error (other code may fail differently or silently truncate).
11. Language patterns are mostly about relative distance ("the word before", "the subject a few words back"), which absolute positions must learn separately at every position.
12. It keeps its random initial values (shrunk slightly by weight decay, if applied), so it carries no learned information.

**Exercise 1.** Parameters: 2,048 rows of 48 plus 64 rows of 48, which `count_parameters` reports as 101,376. Output `(4, 64, 48)`; the 65-token input raises `ValueError` mentioning 64 positions. Acceptance: predictions checked in code.

**Exercise 2.** 1; 0; -1; the last is about 0.8, closer to 1. Acceptance: checked with code.

**Exercise 3.** Acceptance: a table of accuracy and role similarity per `d_model`. Expect accuracy to be similar from 16 upward (the harbor task is small), worse at 4; and the output-layer grouping to persist across sizes while the input-table grouping stays weak. Your conclusion should be stated for this data only.

**Exercise 4.** The author's run (10 epochs, default settings): rows of seen tokens grew on average from a length of 5.64 to 5.98, while rows of unseen tokens shrank from 5.61 to 5.48. Acceptance: unseen-token rows have a noticeably different average size from seen ones. Seen rows grew or changed under training; unseen rows were only shrunk by AdamW's default weight decay of 0.01 at every step, because weight decay applies to all parameters, used or not. You explain both effects and note that unseen rows carry no learned information either way.

**Exercise 5.** Solution: [`code/solutions/ch10_average_first.py`](../../code/solutions/ch10_average_first.py); the test is in [`code/tests/test_embeddings.py`](../../code/tests/test_embeddings.py). Observed output:

```text
bag (no positions)                 val loss 1.1754, val accuracy 65.1%
positions, averaged first          val loss 1.0573, val accuracy 66.1%
positions, layer before averaging  val loss 0.7340, val accuracy 72.2%
```

Acceptance: the averaged-first variant scores close to the plain bag and well below the correct `bag_position`; its outputs for a context and its reverse are identical; you can explain why with the "average of sums" argument. (Its numbers are not exactly the bag's, because it is a slightly different model, with the hidden layer after the average rather than before. Neither can see order.)

**Exercise 6.** Acceptance: accuracy rises from context 2 to 8 and then flattens or changes little, because most of what predicts the next token in this corpus is within the current sentence's last few tokens (the actor determines the possible actions; "at"/"in the" predicts a time). If you did the optional measurement: shuffling the most recent positions costs the most accuracy, and shuffling the earliest positions very little. That is a measured case for why relative position matters more than absolute.

#### Checkpoint: what you can now do independently

You can now:

- Replace one-hot inputs with embedding tables and explain the equivalence and the savings.
- Measure similarity between learned vectors and inspect a trained model's tables without over-interpreting them.
- Demonstrate, with a measurement, what a model loses without word order.
- Add learned positional embeddings correctly, avoid the averaging trap, and state their limits.

**Next:** Chapter 11 turns a stream of token IDs into training batches for a transformer: context windows, the input/target shift, padding, masks, packing, and PyTorch's `DataLoader`.
