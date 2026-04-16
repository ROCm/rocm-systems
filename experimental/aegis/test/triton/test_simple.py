#!/usr/bin/env python3
"""
Minimal Triton test for AegisBit profiler.
Uses a simple vector-add kernel with known coalescing behavior.
"""
import torch
import triton
import triton.language as tl
import os
import sys


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"PID:     {os.getpid()}")

    n = 4096
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)

    print(f"\nLaunching add_kernel: n={n}, BLOCK_SIZE=256")
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    expected = x + y
    max_diff = (output - expected).abs().max().item()
    print(f"Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5
    print("PASS")


if __name__ == "__main__":
    main()
