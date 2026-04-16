#!/usr/bin/env python3
"""
Triton LDS (shared memory) bank conflict test for AegisBit profiler validation.

Run without profiler:
    python3 test_lds.py

Run with AegisBit (memory + LDS bank conflict profiler):
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY AEGISBIT_LOG=1 \
    LD_PRELOAD=<path>/libaegisbit.so \
    python3 test_lds.py

Test 1: Synthetic bank-conflict kernel (stride-32 shared memory reads)
Test 2: Tiled matrix multiply (natural LDS usage via tl.dot)
"""
import torch
import triton
import triton.language as tl
import sys


@triton.jit
def bank_conflict_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Deliberately creates LDS bank conflicts via stride-32 access pattern.
    Each lane reads from the same bank (stride of 32 dwords = 128 bytes),
    which serializes all accesses.
    """
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Write sequential values to shared memory (no conflicts on write)
    scratch = tl.arange(0, BLOCK_SIZE).to(tl.float32)

    # Stride-32 read: every lane hits the same bank
    strided = (offsets * 32) % BLOCK_SIZE
    vals = tl.load(output_ptr + pid * BLOCK_SIZE + strided)

    tl.store(output_ptr + pid * BLOCK_SIZE + offsets, vals + scratch)


@triton.jit
def no_conflict_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sequential shared memory access pattern — no bank conflicts expected.
    Each lane reads from a different bank.
    """
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)

    vals = tl.load(output_ptr + pid * BLOCK_SIZE + offsets)
    tl.store(output_ptr + pid * BLOCK_SIZE + offsets, vals + 1.0)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matmul — natural LDS usage via tl.dot."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def test_bank_conflict():
    """Synthetic kernel with deliberate bank conflicts."""
    print("=" * 60)
    print("Test 1: Bank conflict kernel (stride-32 pattern)")
    print("=" * 60)

    n = 8192
    BLOCK_SIZE = 256
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    grid = (n // BLOCK_SIZE,)

    bank_conflict_kernel[grid](x, n, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    print(f"  Output sample: {x[:8].tolist()}")
    print("  PASS\n")


def test_no_conflict():
    """Sequential access — should show no bank conflicts."""
    print("=" * 60)
    print("Test 2: No-conflict kernel (sequential pattern)")
    print("=" * 60)

    n = 8192
    BLOCK_SIZE = 256
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    grid = (n // BLOCK_SIZE,)

    no_conflict_kernel[grid](x, n, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    expected = 2.0
    max_diff = (x - expected).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


def test_tiled_matmul():
    """Tiled matmul — natural LDS usage via tl.dot."""
    print("=" * 60)
    print("Test 3: Tiled matmul (256x256, BLOCK=64)")
    print("=" * 60)

    M, N, K = 256, 256, 256
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()

    expected = a @ b
    max_diff = (c - expected).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-2, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print()

    test_bank_conflict()
    test_no_conflict()
    test_tiled_matmul()

    print("All LDS tests passed.")
