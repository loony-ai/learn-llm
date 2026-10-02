"""Choose where tensors live and measure how much memory they use (Chapter 3).

Every script in the book that uses PyTorch accepts `--device`:

    auto  -> cuda if an NVIDIA GPU is usable, else mps (Apple GPU), else cpu
    cpu   -> always the CPU
    cuda  -> NVIDIA GPU (error if unavailable)
    mps   -> Apple GPU (error if unavailable)
"""

from __future__ import annotations

import argparse

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def available_devices() -> list[str]:
    """Names of the device types PyTorch can use on this machine, CPU always included."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def pick_device(preference: str = "auto") -> torch.device:
    """Turn a --device choice into a torch.device, failing clearly if it is unavailable."""
    if preference not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device {preference!r}; choose one of {DEVICE_CHOICES}")
    available = available_devices()
    if preference == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if candidate in available:
                return torch.device(candidate)
    if preference not in available:
        raise RuntimeError(
            f"Device {preference!r} was requested but is not available here. "
            f"Available: {available}. Use --device cpu or --device auto."
        )
    return torch.device(preference)


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add the book's standard --device option to a command-line parser."""
    parser.add_argument(
        "--device",
        default="auto",
        choices=DEVICE_CHOICES,
        help="where to run: auto (default), cpu, cuda, or mps",
    )


def describe_device(device: torch.device) -> str:
    """A one-line human-readable description of a device."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return (
            f"cuda:{index} {torch.cuda.get_device_name(index)} "
            f"({format_bytes(free)} free of {format_bytes(total)} VRAM)"
        )
    if device.type == "mps":
        return "mps (Apple GPU, shares system memory)"
    return f"cpu ({torch.get_num_threads()} threads used by PyTorch)"


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Memory used by a tensor's numbers: element count times bytes per element."""
    return tensor.nelement() * tensor.element_size()


def format_bytes(count: int | float) -> str:
    """Format a byte count with binary units: 1 KiB = 1024 bytes, 1 MiB = 1024 KiB, ..."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
