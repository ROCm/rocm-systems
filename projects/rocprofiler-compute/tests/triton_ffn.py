# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Pure-Triton LLaMA-style feed-forward block (RMSNorm + gated MLP).

Implements one transformer FFN entirely in Triton:

    h     = rmsnorm(x)
    gate  = h @ Wgate
    up    = h @ Wup
    out   = x + (silu(gate) * up) @ Wdown

This exercises five distinct Triton kernels (rmsnorm, matmul, silu, mul, add);
the matmul kernel is launched from three call sites (the gate, up, and down
projections).
"""

import sys

import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
ELT_BLOCK = 1024


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, x * rstd * w, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < K - k
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x), mask=mask)


@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * y, mask=mask)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    rows, cols = x.shape
    out = torch.empty_like(x)
    rmsnorm_kernel[(rows,)](
        x, weight, out, cols, eps, BLOCK=triton.next_power_of_2(cols)
    )
    return out


def matmul(a, b):
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def elementwise(kernel, *tensors):
    out = torch.empty_like(tensors[0])
    n = out.numel()
    grid = (triton.cdiv(n, ELT_BLOCK),)
    kernel[grid](*tensors, out, n, BLOCK_SIZE=ELT_BLOCK)
    return out


def ffn(x, w_norm, w_gate, w_up, w_down):
    h = rmsnorm(x, w_norm)
    gate = matmul(h, w_gate)
    up = matmul(h, w_up)
    act = elementwise(silu_kernel, gate)
    fused = elementwise(mul_kernel, act, up)
    down = matmul(fused, w_down)
    return elementwise(add_kernel, x, down)


def main():
    if not torch.cuda.is_available():
        print("GPU is required for this sample. Exiting.")
        sys.exit(1)

    tokens, hidden, inter = 512, 512, 2048
    dev = "cuda"
    x = torch.randn(tokens, hidden, device=dev)
    w_norm = torch.randn(hidden, device=dev)
    w_gate = torch.randn(hidden, inter, device=dev)
    w_up = torch.randn(hidden, inter, device=dev)
    w_down = torch.randn(inter, hidden, device=dev)

    for _ in range(3):
        out = ffn(x, w_norm, w_gate, w_up, w_down)

    torch.cuda.synchronize()
    print(f"FFN completed, output sum: {out.sum().item():.3f}")


if __name__ == "__main__":
    main()
