"""Chapter 3 milestone: devices, memory use, and the cost of moving data.

Run from `code/`:
    python -m scripts.ch03_tensor_tour
    python -m scripts.ch03_tensor_tour --device cpu --size 1024
"""

from __future__ import annotations

import argparse
import time

import torch

from llmfp.devices import (
    add_device_argument,
    available_devices,
    describe_device,
    format_bytes,
    pick_device,
    tensor_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_device_argument(parser)
    parser.add_argument("--size", type=int, default=2048, help="side length of the square test tensors")
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def time_matmul(a: torch.Tensor, b: torch.Tensor, repeats: int) -> float:
    """Average seconds for one matrix multiplication, after one warm-up run."""
    (a @ b).sum().item()  # warm-up; .item() also waits for the device to finish
    start = time.perf_counter()
    for _ in range(repeats):
        (a @ b).sum().item()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    args = parse_args()
    try:
        device = pick_device(args.device)
    except RuntimeError as error:
        raise SystemExit(f"Error: {error}")
    print(f"Available devices: {available_devices()}")
    print(f"Using: {describe_device(device)}")

    # Memory: same shape, different data types.
    print("\nMemory for a tensor of shape (32, 256, 768), e.g. a batch of 32 sequences,")
    print("256 tokens each, 768 numbers per token:")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.empty(32, 256, 768, dtype=dtype)
        print(f"  {str(dtype):<15} {format_bytes(tensor_bytes(t)):>10}")

    # Parameters of a model: count times bytes per number.
    print("\nMemory just to hold the parameters of a model with 124 million parameters:")
    for dtype in (torch.float32, torch.bfloat16):
        bytes_needed = 124_000_000 * torch.empty(0, dtype=dtype).element_size()
        print(f"  {str(dtype):<15} {format_bytes(bytes_needed):>10}")

    # Speed: a large matrix multiplication on the chosen device, and on the CPU.
    a = torch.randn(args.size, args.size)
    b = torch.randn(args.size, args.size)
    cpu_seconds = time_matmul(a, b, args.repeats)
    print(f"\n{args.size}x{args.size} matrix multiplication on cpu: {cpu_seconds * 1000:.1f} ms")
    if device.type != "cpu":
        start = time.perf_counter()
        a_dev, b_dev = a.to(device), b.to(device)
        torch.ones(1, device=device).item()  # wait for the copy to finish
        copy_seconds = time.perf_counter() - start
        device_seconds = time_matmul(a_dev, b_dev, args.repeats)
        print(f"copying both tensors to {device.type}: {copy_seconds * 1000:.1f} ms")
        print(f"same multiplication on {device.type}: {device_seconds * 1000:.1f} ms")

    # Tensors on different devices cannot be combined.
    if device.type != "cpu":
        try:
            a + a.to(device)
        except RuntimeError as error:
            print(f"\nMixing devices fails: {str(error).splitlines()[0]}")
    else:
        print("\n(Only the CPU is available, so the GPU comparison is skipped.)")


if __name__ == "__main__":
    main()
