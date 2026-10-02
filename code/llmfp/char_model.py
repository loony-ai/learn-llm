"""A neural next-character model and its supporting pieces (Chapter 7, Project 0).

    CharVocabulary     characters <-> integer IDs, saved as JSON
    make_examples      slide a window over text: (context of N characters, next character)
    CharMLP            one-hot inputs -> hidden layer -> one score per character
    counting_baseline  a character-level counting model, for comparison
    sample_text        generate text one character at a time
    save_checkpoint / load_checkpoint   model weights + vocabulary + configuration

Chapter 10 replaces the one-hot inputs with learned embeddings, and Part 3
replaces the fixed window with attention. The interfaces stay recognizable.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class CharVocabulary:
    """A fixed, sorted list of characters. A character's position in the list is its ID."""

    def __init__(self, characters: list[str]) -> None:
        if len(set(characters)) != len(characters):
            raise ValueError("characters must be unique")
        self.characters = list(characters)
        self.index = {character: i for i, character in enumerate(self.characters)}

    @classmethod
    def build(cls, text: str) -> "CharVocabulary":
        # sorted(): the same text always gives the same IDs (no set-order surprises, Chapter 4.6)
        return cls(sorted(set(text)))

    @property
    def size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.index[character] for character in text]
        except KeyError as error:
            raise ValueError(f"character {error.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: list[int]) -> str:
        return "".join(self.characters[i] for i in ids)

    def to_dict(self) -> dict:
        return {"characters": self.characters}

    @classmethod
    def from_dict(cls, data: dict) -> "CharVocabulary":
        return cls(data["characters"])


def make_examples(ids: list[int], context_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every window of `context_size` IDs, paired with the ID that follows it.

    Returns contexts of shape (N, context_size) and targets of shape (N,), both int64,
    where N = len(ids) - context_size.
    """
    if len(ids) <= context_size:
        raise ValueError(f"need more than {context_size} IDs, got {len(ids)}")
    sequence = torch.tensor(ids, dtype=torch.int64)
    contexts = sequence.unfold(0, context_size, 1)[:-1]   # windows starting at 0, 1, 2, ...
    targets = sequence[context_size:]                     # the ID right after each window
    return contexts.contiguous(), targets


@dataclass(frozen=True)
class CharModelConfig:
    vocab_size: int
    context_size: int = 8
    hidden: int = 128


class CharMLP(nn.Module):
    """Scores every character as the next one, given the previous `context_size` characters.

    contexts (batch, context_size) int64
      -> one-hot            (batch, context_size, vocab_size)
      -> flatten            (batch, context_size * vocab_size)
      -> Linear, ReLU       (batch, hidden)
      -> Linear             (batch, vocab_size)   logits
    """

    def __init__(self, config: CharModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden = nn.Linear(config.context_size * config.vocab_size, config.hidden)
        self.output = nn.Linear(config.hidden, config.vocab_size)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(contexts, num_classes=self.config.vocab_size).float()
        flat = one_hot.flatten(start_dim=1)   # keep the batch axis, join the rest
        return self.output(torch.relu(self.hidden(flat)))


def counting_baseline(
    train_contexts: torch.Tensor, train_targets: torch.Tensor, contexts: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor | float]:
    """A character counting model (Chapter 1's method) trained and evaluated on the same windows.

    Returns its accuracy and coverage, plus `seen` (bool, one per evaluated example):
    whether that context appeared in training, so other models can be compared on
    seen and unseen contexts separately.
    """
    table: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for context, target in zip(train_contexts.tolist(), train_targets.tolist()):
        table[tuple(context)][target] += 1
    seen = torch.zeros(len(targets), dtype=torch.bool)
    correct = torch.zeros(len(targets), dtype=torch.bool)
    for i, (context, target) in enumerate(zip(contexts.tolist(), targets.tolist())):
        followers = table.get(tuple(context))
        if followers:
            seen[i] = True
            best = min(followers.items(), key=lambda item: (-item[1], item[0]))[0]
            correct[i] = best == target
    return {
        "accuracy": correct.float().mean().item(),
        "coverage": seen.float().mean().item(),
        "seen": seen,
        "correct": correct,
    }


@torch.no_grad()
def sample_text(
    model: CharMLP,
    vocabulary: CharVocabulary,
    prompt: str,
    length: int,
    generator: torch.Generator | None = None,
    greedy: bool = False,
) -> str:
    """Extend `prompt` by `length` characters, one at a time (Chapter 1.7's loop).

    A prompt shorter than the context is padded on the left with newlines, the
    character that comes before every sentence in the training text, much as
    Chapter 1 padded with <start>. The padding is removed from the result.
    """
    size = model.config.context_size
    padding = max(0, size - len(prompt))
    model.eval()
    device = next(model.parameters()).device
    ids = vocabulary.encode("\n" * padding + prompt)
    for _ in range(length):
        context = torch.tensor([ids[-size:]], device=device)          # (1, context_size)
        logits = model(context)[0]                                    # (vocab_size,)
        if greedy:
            next_id = int(logits.argmax())
        else:
            shares = torch.softmax(logits, dim=-1).cpu()
            next_id = int(torch.multinomial(shares, 1, generator=generator))
        ids.append(next_id)
    return vocabulary.decode(ids[padding:])


def save_checkpoint(directory: str | Path, model: CharMLP, vocabulary: CharVocabulary, extra: dict | None = None) -> None:
    """Write model.pt (parameters), vocab.json, and model_config.json into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "vocab.json").write_text(json.dumps(vocabulary.to_dict(), ensure_ascii=False), encoding="utf-8")
    payload = {"model": asdict(model.config), **(extra or {})}
    (directory / "model_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(directory: str | Path, device: torch.device | str = "cpu") -> tuple[CharMLP, CharVocabulary]:
    """Rebuild the model from its configuration, then load the saved parameters into it."""
    directory = Path(directory)
    config = CharModelConfig(**json.loads((directory / "model_config.json").read_text(encoding="utf-8"))["model"])
    vocabulary = CharVocabulary.from_dict(json.loads((directory / "vocab.json").read_text(encoding="utf-8")))
    model = CharMLP(config)
    # weights_only=True: only tensors are loaded, never arbitrary Python objects
    state = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device), vocabulary
