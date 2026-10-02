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
