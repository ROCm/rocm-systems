#!/usr/bin/env python3
"""
Triton vector-add test for AegisBit profiler validation.

Run without profiler:
    python3 test_vector_add.py

Run with AegisBit (memory coalescing profiler):
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY AEGISBIT_LOG=1 \
    LD_PRELOAD=<path>/libaegisbit.so \
    python3 test_vector_add.py
"""
import torch
import triton
import triton.language as tl
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


@triton.jit
def strided_load_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Deliberately strided access pattern — poor coalescing."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    strided_offsets = offsets * stride
    x = tl.load(x_ptr + strided_offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)


def test_coalesced_add():
    """Contiguous vector add — should show good coalescing."""
    print("=" * 60)
    print("Test 1: Coalesced vector add (n=65536, block=256)")
    print("=" * 60)

    n = 65536
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)

    torch.cuda.synchronize()

    expected = x + y
    max_diff = (output - expected).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


def test_strided_access():
    """Strided access — should show poor coalescing."""
    print("=" * 60)
    print("Test 2: Strided access (stride=16, n=4096, block=256)")
    print("=" * 60)

    n = 4096
    stride = 16
    x = torch.randn(n * stride, device="cuda", dtype=torch.float32)
    output = torch.empty(n, device="cuda", dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    strided_load_kernel[grid](x, output, n, stride=stride, BLOCK_SIZE=256)

    torch.cuda.synchronize()

    expected = x[::stride][:n]
    max_diff = (output - expected).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


def test_matmul():
    """Simple matrix multiply — mixed coalescing pattern."""
    print("=" * 60)
    print("Test 3: Naive matmul (128x128)")
    print("=" * 60)

    M, N, K = 128, 128, 128
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.matmul(a, b)

    torch.cuda.synchronize()

    expected = a @ b
    max_diff = (c - expected).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    print("  PASS\n")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print()

    test_coalesced_add()
    test_strided_access()
    test_matmul()

    print("All tests passed.")
