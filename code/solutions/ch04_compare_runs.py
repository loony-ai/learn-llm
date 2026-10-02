"""Chapter 4, Exercise 4 (suggested solution): show what differs between two runs.

Compares config.json, environment.json (selected fields), and metrics.json of two
run directories and prints only the differences.

Run from `code/`:
    python -m solutions.ch04_compare_runs runs/ch04-counting-eval/<run A> runs/ch04-counting-eval/<run B>
    python -m solutions.ch04_compare_runs --latest runs/ch04-counting-eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from llmfp.config import load_json

IGNORED_ENVIRONMENT_KEYS = {"started_utc", "command"}


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Turn nested dicts and lists into {"a.b.0.c": value} so they can be compared key by key."""
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((str(index), value) for index, value in enumerate(data))
    else:
        return {prefix: data}
    flat: dict[str, Any] = {}
    for key, value in items:
        flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def differences(a: Any, b: Any) -> list[tuple[str, Any, Any]]:
    flat_a, flat_b = flatten(a), flatten(b)
    return [
        (key, flat_a.get(key, "<missing>"), flat_b.get(key, "<missing>"))
        for key in sorted(set(flat_a) | set(flat_b))
        if flat_a.get(key, "<missing>") != flat_b.get(key, "<missing>")
    ]


def compare(run_a: Path, run_b: Path) -> dict[str, list[tuple[str, Any, Any]]]:
    report = {}
    for name in ("config.json", "environment.json", "metrics.json"):
        a, b = load_json(run_a / name), load_json(run_b / name)
        if name == "environment.json":
            a = {k: v for k, v in a.items() if k not in IGNORED_ENVIRONMENT_KEYS}
            b = {k: v for k, v in b.items() if k not in IGNORED_ENVIRONMENT_KEYS}
        report[name] = differences(a, b)
    return report


def show(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else repr(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", type=Path, help="two run directories")
    parser.add_argument("--latest", type=Path, help="compare the two most recent runs in this directory")
    parser.add_argument("--limit", type=int, default=12, help="maximum differences shown per file")
    args = parser.parse_args()

    if args.latest:
        runs = sorted(path for path in args.latest.iterdir() if path.is_dir())[-2:]
    else:
        runs = args.runs
    if len(runs) != 2:
        raise SystemExit("Give exactly two run directories, or --latest DIR with at least two runs in it.")

    print(f"A: {runs[0]}\nB: {runs[1]}")
    for name, diffs in compare(*runs).items():
        print(f"\n{name}: {len(diffs)} difference(s)")
        for key, value_a, value_b in diffs[: args.limit]:
            print(f"  {key}: {show(value_a)} -> {show(value_b)}")
        if len(diffs) > args.limit:
            print(f"  ... and {len(diffs) - args.limit} more")


if __name__ == "__main__":
    main()
