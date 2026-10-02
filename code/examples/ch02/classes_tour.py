"""Chapter 2.6: classes, methods, properties, inheritance, and special methods.

The final class mimics the calling pattern PyTorch uses for neural network
layers (Chapter 5): you *call* the object, and the call runs its `forward`
method. This is a teaching imitation, not how PyTorch is implemented.

Run from `code/`:  python examples/ch02/classes_tour.py
"""

from __future__ import annotations


class Vocabulary:
    """Maps tokens to integer IDs and back."""

    def __init__(self, tokens: list[str]) -> None:
        # Attributes are stored on `self`, the object being created.
        self.id_to_token = sorted(set(tokens))
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}

    # A regular method: called on an instance, receives it as `self`.
    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

    # A property: computed on access, used like an attribute (no parentheses).
    @property
    def size(self) -> int:
        return len(self.id_to_token)

    # A class method: an alternative constructor. It receives the class, not an instance.
    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        return cls(text.split())

    # Special ("dunder", double-underscore) methods plug into Python syntax.
    def __len__(self) -> int:  # len(vocab)
        return self.size

    def __contains__(self, token: str) -> bool:  # "lamp" in vocab
        return token in self.token_to_id

    def __repr__(self) -> str:  # how the object prints
        return f"Vocabulary(size={self.size})"


vocab = Vocabulary.from_text("the keeper lit the lamp")
print(vocab, "| len:", len(vocab), "| size property:", vocab.size, "| 'lamp' in vocab:", "lamp" in vocab)
ids = vocab.encode(["the", "lamp"])
print("encode:", ids, "| decode:", vocab.decode(ids))


# Inheritance: a subclass reuses its parent's code and changes some of it.
class VocabularyWithUnknown(Vocabulary):
    UNKNOWN = "<unk>"

    def __init__(self, tokens: list[str]) -> None:
        super().__init__(tokens + [self.UNKNOWN])  # run the parent's __init__ first

    def encode(self, tokens: list[str]) -> list[int]:
        unknown_id = self.token_to_id[self.UNKNOWN]
        return [self.token_to_id.get(token, unknown_id) for token in tokens]


safe = VocabularyWithUnknown("the keeper lit the lamp".split())
print("with <unk>:", safe.encode(["the", "boats"]), "->", safe.decode(safe.encode(["the", "boats"])))
print("isinstance of parent:", isinstance(safe, Vocabulary))


# __call__ lets an object be called like a function. PyTorch layers work this way:
# `layer(x)` runs extra bookkeeping and then `layer.forward(x)`.
class Layer:
    def __call__(self, value: float) -> float:
        print(f"  [{type(self).__name__}] called with {value}")
        return self.forward(value)

    def forward(self, value: float) -> float:
        raise NotImplementedError("subclasses define forward")


class Scale(Layer):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def forward(self, value: float) -> float:
        return value * self.factor


class Shift(Layer):
    def __init__(self, amount: float) -> None:
        self.amount = amount

    def forward(self, value: float) -> float:
        return value + self.amount


pipeline = [Scale(2.0), Shift(1.0)]
value = 3.0
for layer in pipeline:
    value = layer(value)  # calls __call__, which calls forward
print("pipeline result:", value)
