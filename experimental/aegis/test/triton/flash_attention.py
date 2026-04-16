#!/usr/bin/env python3
"""
Triton Flash Attention v2 test — causal self-attention.

Uses heavy MFMA, extensive LDS for Q/K/V tile staging, and complex
control flow (online softmax). Stress-tests the profiler with many
VMEM + LDS sites, AccVGPR allocation, and scratch spilling.

Run with AegisBit:
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY \
    AEGISBIT_STRATEGY=on_gpu_reduce AEGISBIT_MAX_SITES=200 \
    AEGISBIT_MAX_LDS=200 \
    AEGISBIT_KERNELS="flash_attention_kernel.kd" \
    LD_PRELOAD=<path>/libaegisbit.so \
    python3 flash_attention.py
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seqlen_q, seqlen_k,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = Q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    mask_m = offs_m < seqlen_q
    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < BLOCK_K), other=0.0)
    q = (q * scale).to(Q_ptr.dtype.element_ty)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum(seqlen_k, (pid_m + 1) * BLOCK_M)
    else:
        hi = seqlen_k

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        mask_n = offs_n < seqlen_k
        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < BLOCK_K), other=0.0)

        qk = tl.dot(q, tl.trans(k))

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < BLOCK_K), other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = Out_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & (offs_k[None, :] < BLOCK_K))


def flash_attention(Q, K, V, causal=True):
    B, H, M, K_dim = Q.shape
    _, _, N, _ = K.shape

    Out = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(K_dim)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = K_dim

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    flash_attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        M, N,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        IS_CAUSAL=causal,
    )
    return Out


def reference_attention(Q, K, V, causal=True):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    attn = (Q @ K.transpose(-2, -1)) * scale
    if causal:
        M, N = attn.shape[-2], attn.shape[-1]
        mask = torch.triu(torch.ones(M, N, device=Q.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return attn @ V


def test_causal():
    print("=" * 60)
    print("Flash Attention: causal self-attention")
    print("=" * 60)

    B, H, M, K = 2, 4, 128, 64
    Q = torch.randn(B, H, M, K, device="cuda", dtype=torch.float16)
    Kk = torch.randn(B, H, M, K, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, M, K, device="cuda", dtype=torch.float16)

    out = flash_attention(Q, Kk, V, causal=True)
    torch.cuda.synchronize()

    ref = reference_attention(Q.float(), Kk.float(), V.float(), causal=True).half()
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    ok = max_diff < 5e-2
    print(f"  {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_non_causal():
    print("=" * 60)
    print("Flash Attention: non-causal self-attention")
    print("=" * 60)

    B, H, M, K = 2, 4, 128, 64
    Q = torch.randn(B, H, M, K, device="cuda", dtype=torch.float16)
    Kk = torch.randn(B, H, M, K, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, M, K, device="cuda", dtype=torch.float16)

    out = flash_attention(Q, Kk, V, causal=False)
    torch.cuda.synchronize()

    ref = reference_attention(Q.float(), Kk.float(), V.float(), causal=False).half()
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    ok = max_diff < 5e-2
    print(f"  {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print()

    ok1 = test_causal()
    ok2 = test_non_causal()

    if ok1 and ok2:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED.")
        exit(1)
