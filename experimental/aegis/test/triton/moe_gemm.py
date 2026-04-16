#!/usr/bin/env python3
"""
Triton MoE GEMM test — full per-token matmul with expert routing.

This kernel uses MFMA (matrix multiply-accumulate) instructions which
allocate AccVGPRs, pushing the total VGPR count above 256. Tests that the
profiler correctly handles the AccVGPR/regular VGPR split.

Run with AegisBit:
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY AEGISBIT_LOG=1 \
    AEGISBIT_KERNELS="*moe_gemm*" \
    LD_PRELOAD=<path>/libaegisbit.so \
    python3 moe_gemm.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def moe_gemm_kernel(
    X_ptr,            # (num_tokens, K)
    W_ptr,            # (num_experts, K, N) flattened row-major
    Out_ptr,          # (num_tokens, N)
    ExpertIds_ptr,    # (num_tokens,) int32
    num_tokens,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm,
    stride_wk,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < num_tokens

    expert_ids = tl.load(ExpertIds_ptr + offs_m, mask=mask_m, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
        mask_x = mask_m[:, None] & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_base = expert_ids[:, None, None] * (K * N)
        w_ptrs = (W_ptr
                  + w_base
                  + offs_k[None, :, None] * stride_wk
                  + offs_n[None, None, :])
        mask_w = mask_m[:, None, None] & (offs_k[None, :, None] < K) & (offs_n[None, None, :] < N)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc += tl.sum(x[:, :, None] * w, axis=1)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    mask_out = mask_m[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


def moe_gemm(X, W, expert_ids):
    num_tokens, K = X.shape
    num_experts, _, N = W.shape
    Out = torch.empty((num_tokens, N), device=X.device, dtype=X.dtype)

    BLOCK_M = 32
    BLOCK_N = min(64, N)
    BLOCK_K = min(32, K)
    grid = (triton.cdiv(num_tokens, BLOCK_M), triton.cdiv(N, BLOCK_N))

    moe_gemm_kernel[grid](
        X, W, Out, expert_ids,
        num_tokens, K, N,
        X.stride(0), W.stride(1), Out.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )
    return Out


def reference_moe(X, W, expert_ids):
    num_tokens = X.shape[0]
    N = W.shape[2]
    out = torch.empty((num_tokens, N), device=X.device, dtype=X.dtype)
    for t in range(num_tokens):
        out[t] = X[t] @ W[expert_ids[t]]
    return out


def test_uniform():
    print("=" * 60)
    print("MoE GEMM: Uniform routing (all tokens → expert 0)")
    print("=" * 60)

    num_tokens, K, N, num_experts = 256, 64, 64, 8
    X = torch.randn(num_tokens, K, device="cuda", dtype=torch.float32)
    W = torch.randn(num_experts, K, N, device="cuda", dtype=torch.float32)
    expert_ids = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)

    out = moe_gemm(X, W, expert_ids)
    torch.cuda.synchronize()

    ref = reference_moe(X, W, expert_ids)
    max_diff = (out - ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-2, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


def test_scattered():
    print("=" * 60)
    print("MoE GEMM: Scattered routing (random expert per token)")
    print("=" * 60)

    num_tokens, K, N, num_experts = 256, 64, 64, 8
    X = torch.randn(num_tokens, K, device="cuda", dtype=torch.float32)
    W = torch.randn(num_experts, K, N, device="cuda", dtype=torch.float32)
    expert_ids = torch.randint(0, num_experts, (num_tokens,),
                               device="cuda", dtype=torch.int32)

    out = moe_gemm(X, W, expert_ids)
    torch.cuda.synchronize()

    ref = reference_moe(X, W, expert_ids)
    max_diff = (out - ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-2, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print()

    test_uniform()
    test_scattered()
    print("All tests passed.")
