#!/usr/bin/env python3
"""
Triton MoE gather test for AegisBit profiler validation.

Demonstrates coalescing behavior of expert-routed memory access patterns
without using MFMA (avoids AccVGPR allocation that pushes VGPR count > 256).

Two scenarios:
  1. Uniform routing  — all tokens go to expert 0 → coalesced weight loads
  2. Scattered routing — random expert per token  → scattered weight loads

Run with AegisBit:
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY AEGISBIT_LOG=1 \
    AEGISBIT_KERNELS="*moe*" \
    LD_PRELOAD=<path>/libaegisbit.so \
    python3 moe.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def moe_gather_kernel(
    X_ptr,            # (num_tokens, D)
    W_ptr,            # (num_experts, D) — one weight row per expert
    Out_ptr,          # (num_tokens, D)
    ExpertIds_ptr,    # (num_tokens,) int32
    num_tokens,
    D: tl.constexpr,
    stride_x,
    stride_w,
    stride_o,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Per-token gather-and-multiply: out[t, d] = x[t, d] * w[expert_ids[t], d]

    The weight load w[expert_ids[t], d] is the interesting access:
    - Uniform routing: all lanes load from the same expert row → coalesced
    - Scattered routing: each lane loads from a different expert → scattered
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < num_tokens

    expert_ids = tl.load(ExpertIds_ptr + offs_m, mask=mask_m, other=0)

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load X[offs_m, offs_d] — contiguous per token, always coalesced
        x_ptrs = X_ptr + offs_m[:, None] * stride_x + offs_d[None, :]
        mask_xd = mask_m[:, None] & mask_d[None, :]
        x = tl.load(x_ptrs, mask=mask_xd, other=0.0)

        # Load W[expert_ids[m], offs_d] — THIS is the interesting access
        # When expert_ids differ across lanes → different base addresses → scattered
        w_ptrs = W_ptr + expert_ids[:, None] * stride_w + offs_d[None, :]
        w = tl.load(w_ptrs, mask=mask_xd, other=0.0)

        out = x * w

        out_ptrs = Out_ptr + offs_m[:, None] * stride_o + offs_d[None, :]
        tl.store(out_ptrs, out, mask=mask_xd)


def moe_gather(X, W, expert_ids):
    num_tokens, D = X.shape
    Out = torch.empty_like(X)

    BLOCK_M = 64
    BLOCK_D = min(64, D)
    grid = (triton.cdiv(num_tokens, BLOCK_M),)

    moe_gather_kernel[grid](
        X, W, Out, expert_ids,
        num_tokens, D,
        X.stride(0), W.stride(0), Out.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
    )
    return Out


def test_uniform_routing():
    """All tokens → same expert. Weight loads should be perfectly coalesced."""
    print("=" * 60)
    print("Test 1: Uniform routing (all tokens → expert 0)")
    print("  Expect: HIGH coalescing (all lanes load same weight row)")
    print("=" * 60)

    num_tokens, D, num_experts = 1024, 128, 8
    X = torch.randn(num_tokens, D, device="cuda", dtype=torch.float32)
    W = torch.randn(num_experts, D, device="cuda", dtype=torch.float32)
    expert_ids = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)

    out = moe_gather(X, W, expert_ids)
    torch.cuda.synchronize()

    ref = X * W[0].unsqueeze(0)
    max_diff = (out - ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


def test_scattered_routing():
    """Each token → random expert. Weight loads should show poor coalescing."""
    print("=" * 60)
    print("Test 2: Scattered routing (random expert per token)")
    print("  Expect: LOW coalescing (lanes load from different experts)")
    print("=" * 60)

    num_tokens, D, num_experts = 1024, 128, 8
    X = torch.randn(num_tokens, D, device="cuda", dtype=torch.float32)
    W = torch.randn(num_experts, D, device="cuda", dtype=torch.float32)
    expert_ids = torch.randint(0, num_experts, (num_tokens,),
                               device="cuda", dtype=torch.int32)

    out = moe_gather(X, W, expert_ids)
    torch.cuda.synchronize()

    ref = X * W[expert_ids.long()]
    max_diff = (out - ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"FAIL: max diff {max_diff}"
    print("  PASS\n")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print()

    test_uniform_routing()
    test_scattered_routing()

    print("All tests passed.")
