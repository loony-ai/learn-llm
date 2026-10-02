"""Seeds, run directories, and experiment records (Chapter 4).

Every experiment in the book from here on calls `start_run`, which creates a new
directory and writes what is needed to understand and repeat the run:

    runs/<name>/<timestamp>/
        config.json        the fully resolved configuration
        environment.json   Python, packages, hardware, git commit, command line
        metrics.json       results, written by `finish_run`
        log.txt            everything logged during the run
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llmfp.config import save_json, to_dict

TRACKED_PACKAGES = ("llmfp", "numpy", "torch")


def set_seed(seed: int) -> int:
    """Seed every random number generator the book's code uses: Python, NumPy, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every GPU device
    return seed


def file_sha256(path: str | Path) -> str:
    """Fingerprint of a file's exact contents. Any change to the file changes it."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def capture_environment() -> dict[str, Any]:
    """Describe the software and hardware a run used."""
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": commit,
        "git_uncommitted_changes": None if status is None else bool(status),
        "command": sys.argv,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def create_run_dir(root: str | Path, name: str) -> Path:
    """Create runs/<name>/<timestamp>, adding a suffix if that directory already exists."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(root) / name
    candidate, suffix = base / stamp, 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stamp}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def start_run(root: str | Path, name: str, config: Any, data_files: list[str | Path] = ()) -> Path:
    """Create a run directory, record config and environment, and log to log.txt in it."""
    run_dir = create_run_dir(root, name)
    save_json(to_dict(config), run_dir / "config.json")
    environment = capture_environment()
    environment["data_files"] = {str(path): file_sha256(path) for path in data_files}
    save_json(environment, run_dir / "environment.json")

    handler = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return run_dir


def finish_run(run_dir: Path, metrics: dict[str, Any]) -> None:
    """Write the run's results and stop logging to its log file."""
    save_json(metrics, run_dir / "metrics.json")
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).parent == run_dir.resolve():
            root_logger.removeHandler(handler)
            handler.close()
