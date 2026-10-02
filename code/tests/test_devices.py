"""Tests for llmfp.devices (Chapter 3)."""

from __future__ import annotations

import argparse

import pytest
import torch

from llmfp.devices import (
    add_device_argument,
    available_devices,
    describe_device,
    format_bytes,
    pick_device,
    tensor_bytes,
)


def test_cpu_is_always_available():
    assert "cpu" in available_devices()
    assert pick_device("cpu") == torch.device("cpu")


def test_auto_picks_an_available_device():
    assert pick_device("auto").type in available_devices()


def test_unknown_device_name_rejected():
    with pytest.raises(ValueError):
        pick_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has CUDA")
def test_unavailable_device_fails_clearly():
    with pytest.raises(RuntimeError, match="not available"):
        pick_device("cuda")


def test_describe_cpu():
    assert describe_device(torch.device("cpu")).startswith("cpu (")


@pytest.mark.parametrize(
    ("shape", "dtype", "expected"),
    [
        ((10,), torch.float32, 40),
        ((2, 3), torch.float16, 12),
        ((4, 5), torch.int64, 160),
        ((0,), torch.float32, 0),
    ],
)
def test_tensor_bytes(shape, dtype, expected):
    assert tensor_bytes(torch.zeros(shape, dtype=dtype)) == expected


def test_tensor_bytes_of_a_view_counts_only_its_elements():
    big = torch.zeros(100, 100)
    assert tensor_bytes(big[:10]) == 10 * 100 * 4


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "0 B"), (1023, "1023 B"), (1024, "1.0 KiB"), (1536, "1.5 KiB"), (1024**2, "1.0 MiB"), (3 * 1024**3, "3.0 GiB")],
)
def test_format_bytes(count, expected):
    assert format_bytes(count) == expected


def test_device_argument():
    parser = argparse.ArgumentParser()
    add_device_argument(parser)
    assert parser.parse_args([]).device == "auto"
    assert parser.parse_args(["--device", "cpu"]).device == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "gpu"])
