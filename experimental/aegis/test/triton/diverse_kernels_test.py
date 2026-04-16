#!/usr/bin/env python3
"""
Diverse Triton kernel validation for AegisBit profiling.

Tests a variety of kernel patterns to ensure instrumentation is stable
and produces correct results across different memory access patterns,
register pressures, and LDS usage.

Run:
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY \
    AEGISBIT_STRATEGY=on_gpu_reduce AEGISBIT_MAX_SITES=200 \
    AEGISBIT_KERNELS="vector_add_kernel.kd,softmax_kernel.kd,matmul_kernel.kd,layernorm_kernel.kd,transpose_kernel.kd,fused_add_relu_kernel.kd,sum_reduction_kernel.kd,coalesced_copy_kernel.kd,scattered_copy_kernel.kd,strided_copy_kernel.kd" \
    LD_PRELOAD=build/src/libaegisbit.so \
    python3 test/triton/diverse_kernels_test.py

Note: The kernel filter must list Triton kernel names explicitly. Glob patterns
like "*_kernel.kd" also match PyTorch internal HIP kernels (e.g.
vectorized_elementwise_kernel, elementwise_kernel_manual_unroll) which can
cause hangs or incorrect results when instrumented.
"""
import torch
import triton
import triton.language as tl
import sys
import os
import time


# ---------------------------------------------------------------------------
# Kernel 1: Vector Add (simplest possible — 1 load + 1 load + 1 store)
# ---------------------------------------------------------------------------
@triton.jit
def vector_add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask)
    b = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a + b, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: Softmax (reduction with LDS, typical attention pattern)
# ---------------------------------------------------------------------------
@triton.jit
def softmax_kernel(
    Input_ptr, Output_ptr, N_cols,
    stride_row,
    BLOCK_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_COLS)
    mask = col_offs < N_cols

    row_start = Input_ptr + row_idx * stride_row
    x = tl.load(row_start + col_offs, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    out_start = Output_ptr + row_idx * stride_row
    tl.store(out_start + col_offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3: Matrix Multiply (classic GEMM with tiling)
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = B_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 4: Layer Norm (mean + variance reduction, then normalize)
# ---------------------------------------------------------------------------
@triton.jit
def layernorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    stride, N,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    w = tl.load(W_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask).to(tl.float32)
    y = x_norm * w + b
    tl.store(Y_ptr + row * stride + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 5: Matrix Transpose (strided reads, coalesced writes)
# ---------------------------------------------------------------------------
@triton.jit
def transpose_kernel(
    In_ptr, Out_ptr,
    M, N,
    stride_in_m, stride_in_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    in_ptrs = In_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tile = tl.load(in_ptrs, mask=mask)

    out_ptrs = Out_ptr + offs_n[None, :] * stride_out_m + offs_m[:, None] * stride_out_n
    tl.store(out_ptrs, tile, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 6: Fused Add + ReLU (elementwise, two loads, one store, branching)
# ---------------------------------------------------------------------------
@triton.jit
def fused_add_relu_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask)
    b = tl.load(B_ptr + offs, mask=mask)
    c = a + b
    c = tl.where(c > 0, c, 0.0)
    tl.store(C_ptr + offs, c, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 7: Reduction (sum along axis — high LDS usage)
# ---------------------------------------------------------------------------
@triton.jit
def sum_reduction_kernel(
    X_ptr, Out_ptr,
    M, N,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        idx = start + offs
        mask = idx < N
        x = tl.load(X_ptr + row * stride_row + idx, mask=mask, other=0.0)
        acc += x
    total = tl.sum(acc, axis=0)
    tl.store(Out_ptr + row, total)


# ===========================================================================
# Test runners
# ===========================================================================

def banner(name, desc):
    print(f"\n{'='*60}")
    print(f"  {name}: {desc}")
    print(f"{'='*60}")


def test_vector_add():
    banner("Vector Add", "simplest kernel — 2 loads, 1 store, stride-1")
    N = 131072
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.empty(N, device='cuda', dtype=torch.float32)
    grid = (triton.cdiv(N, 1024),)
    vector_add_kernel[grid](a, b, c, N, BLOCK=1024)
    torch.cuda.synchronize()
    ref = a + b
    diff = (c - ref).abs().max().item()
    assert diff < 1e-6, f"FAIL vector_add: {diff}"
    print(f"  max_diff={diff:.2e}  PASS")


def test_softmax():
    banner("Softmax", "row-wise reduction, LDS-heavy, attention-style")
    rows, cols = 1024, 256
    x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (rows,)
    softmax_kernel[grid](x, y, cols, x.stride(0), BLOCK_COLS=256)
    torch.cuda.synchronize()
    ref = torch.softmax(x, dim=-1)
    diff = (y - ref).abs().max().item()
    assert diff < 1e-5, f"FAIL softmax: {diff}"
    print(f"  max_diff={diff:.2e}  PASS")


def test_matmul():
    banner("Matmul", "tiled GEMM — high VMEM, many sites")
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    BM, BN, BK = 32, 32, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    torch.cuda.synchronize()
    ref = (a.float() @ b.float())
    diff = (c - ref).abs().max().item()
    assert diff < 1e-2, f"FAIL matmul: {diff}"
    print(f"  max_diff={diff:.2e}  PASS")


def test_layernorm():
    banner("LayerNorm", "mean+var reduction, normalize, affine — mixed LDS/VMEM")
    rows, cols = 512, 128
    x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
    w = torch.ones(cols, device='cuda', dtype=torch.float32)
    b = torch.zeros(cols, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (rows,)
    layernorm_kernel[grid](x, y, w, b, x.stride(0), cols, eps=1e-5, BLOCK=128)
    torch.cuda.synchronize()
    ref = torch.nn.functional.layer_norm(x, [cols], weight=w, bias=b, eps=1e-5)
    diff = (y - ref).abs().max().item()
    assert diff < 1e-4, f"FAIL layernorm: {diff}"
    print(f"  max_diff={diff:.2e}  PASS")


def test_transpose():
    banner("Transpose", "strided reads, coalesced writes — scattered pattern")
    M, N = 512, 512
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(N, M, device='cuda', dtype=torch.float32)
    BM, BN = 32, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    transpose_kernel[grid](
        x, y, M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    torch.cuda.synchronize()
    ref = x.T
    diff = (y - ref).abs().max().item()
    assert diff < 1e-6, f"FAIL transpose: {diff}"
    print(f"  max_diff={diff:.2e}  PASS")


def test_fused_add_relu():
    banner("Fused Add+ReLU", "elementwise with branch — simple but distinct pattern")
    N = 131072
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.empty(N, device='cuda', dtype=torch.float32)
    grid = (triton.cdiv(N, 1024),)
    fused_add_relu_kernel[grid](a, b, c, N, BLOCK=1024)
    torch.cuda.synchronize()
    ref = torch.relu(a + b)
    diff = (c - ref).abs().max().item()
    assert diff < 1e-6, f"FAIL fused_add_relu: {diff}"
    print(f"  max_diff={diff:.2e}  PASS")


def test_sum_reduction():
    banner("Sum Reduction", "row-sum with loop — high LDS, accumulation pattern")
    M, N = 256, 4096
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    out = torch.empty(M, device='cuda', dtype=torch.float32)
    grid = (M,)
    sum_reduction_kernel[grid](x, out, M, N, x.stride(0), BLOCK_N=256)
    torch.cuda.synchronize()
    ref = x.sum(dim=-1)
    diff = (out - ref).abs().max().item()
    assert diff < 0.1, f"FAIL sum_reduction: {diff}"
    print(f"  max_diff={diff:.2e}  PASS")


def main():
    profiled = os.environ.get("AEGISBIT_ENABLED")
    mode = "PROFILED" if profiled else "BASELINE"
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"Mode:    {mode}")

    tests = [
        ("vector_add",     test_vector_add),
        ("softmax",        test_softmax),
        ("matmul",         test_matmul),
        ("layernorm",      test_layernorm),
        ("transpose",      test_transpose),
        ("fused_add_relu", test_fused_add_relu),
        ("sum_reduction",  test_sum_reduction),
    ]

    passed = 0
    failed = 0
    t_start = time.perf_counter()

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed+failed} passed  ({elapsed:.1f}s)")
    print(f"{'='*60}")

    if failed:
        print(f"\n  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\n  All tests passed.")


if __name__ == "__main__":
    main()
