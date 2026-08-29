#!/usr/bin/env python3
"""Standalone gpt-oss kernel correctness suite for the rocjitsu emulator.

Runs every distinct GPU kernel that vLLM dispatches for `openai/gpt-oss-20b`
and validates each one against a float64 CPU reference computed in this same
file.  It is deliberately a *single* file with no dependency on vLLM, no model
weights and no network access, so it can be dropped into any ROCm python
environment and run under the emulator:

    rocjitsu --config configs/gfx950_mi355x_kmd.json -- python gpt_oss_kernels.py
    rocjitsu --config configs/gfx1250_mi455x.json    -- python gpt_oss_kernels.py

Why a reimplementation rather than importing vLLM: the two targets need
different ROCm wheels (gfx950 and gfx1250 nightlies do not ship from one
index), and a suite that only runs where vLLM installs cannot be used to
compare the two architectures.  Every case therefore carries its own device
implementation -- torch eager and, where vLLM uses one, a Triton kernel written
to the same algorithm -- plus an independent CPU reference.  `--with-vllm`
additionally cross-checks against vLLM's own custom ops when they import.

The kernel inventory mirrors a gpt-oss forward pass under vLLM v0.28 on ROCm,
which selects the TRITON_ATTN attention backend and the `EMULATION` MXFP4 MoE
backend (`OCP_MXQuantizationEmulationTritonExperts`):

    embedding -> [RMSNorm -> QKV GEMM -> YaRN RoPE -> KV write ->
                  attention with sinks (sliding window on even layers) ->
                  O GEMM -> RMSNorm -> router GEMM -> top-k softmax ->
                  MXFP4 dequant -> MoE GEMM with SwiGLU-OAI] x L
              -> RMSNorm -> LM head -> sample

Exit status is 0 when every selected case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# --------------------------------------------------------------------------
# Model shape.  `model` is gpt-oss-20b as published; the smaller profiles keep
# every shape that changes kernel *behaviour* (head_dim, the MXFP4 block size,
# the sliding-window/full alternation, top-k routing) and shrink only the
# dimensions that cost emulator time.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    name: str
    num_tokens: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    num_experts: int
    top_k: int
    vocab_size: int
    sliding_window: int
    context_len: int

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim


SHAPES = {
    # Default.  ~10 MFLOP for the heaviest case; the whole suite is minutes
    # under functional emulation.
    "tiny": Shape("tiny", 4, 256, 4, 2, 64, 128, 4, 2, 512, 8, 16),
    "small": Shape("small", 16, 512, 8, 2, 64, 256, 8, 4, 2048, 16, 48),
    # Real gpt-oss-20b widths, one layer, four tokens.  Hours under emulation.
    "model": Shape("model", 4, 2880, 64, 8, 64, 2880, 32, 4, 201088, 128, 32),
}

# gpt-oss rope/activation constants, from the published config.json.
ROPE_THETA = 150000.0
ROPE_FACTOR = 32.0
ROPE_ORIGINAL_MAX_POSITION = 4096
ROPE_BETA_FAST = 32
ROPE_BETA_SLOW = 1
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
RMS_NORM_EPS = 1e-5
MXFP4_BLOCK_SIZE = 32


class SkipCase(Exception):
    """Raised by a case that cannot run in this environment."""


# --------------------------------------------------------------------------
# float64 CPU references.  These are the oracle: every one is written from the
# model definition rather than by calling the device path in another dtype, so
# a shared bug cannot cancel out.
# --------------------------------------------------------------------------


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.to(torch.float64)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight.to(torch.float64)


def ref_swiglu_oai(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """gpt-oss SwiGLU: interleaved gate/up, one-sided gate clamp, up + 1."""
    x = x.to(torch.float64)
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low, high, dim):
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=torch.float64) - low) / (high - low)
    return linear.clamp(0, 1)


def ref_yarn_cos_sin(
    positions: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN cos/sin cache rows for `positions`, in float64.

    Mirrors vLLM's YaRNScalingRotaryEmbedding with gpt-oss parameters,
    including the `mscale` attention factor that YaRN folds into cos/sin.
    """
    pos_freqs = ROPE_THETA ** (
        torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (ROPE_FACTOR * pos_freqs)
    low, high = _yarn_find_correction_range(
        ROPE_BETA_FAST, ROPE_BETA_SLOW, head_dim, ROPE_THETA, ROPE_ORIGINAL_MAX_POSITION
    )
    inv_freq_mask = 1 - _yarn_linear_ramp_mask(low, high, head_dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    mscale = 0.1 * math.log(ROPE_FACTOR) + 1.0 if ROPE_FACTOR > 1 else 1.0
    freqs = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    return freqs.cos() * mscale, freqs.sin() * mscale


def ref_rope_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """NeoX-style rotary: rotate the two halves of each head against each other.

    x: [num_tokens, num_heads, head_dim]; cos/sin: [num_tokens, head_dim // 2].
    """
    x = x.to(torch.float64)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def ref_attention_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    context_len: int,
    sliding_window: int | None,
) -> torch.Tensor:
    """Causal GQA attention with learned per-head softmax sinks.

    The sink is an extra logit that enters the softmax denominator but carries
    no value, so a head can attend to "nothing".  vLLM implements it by seeding
    the online-softmax running max with the sink and the running sum with 1;
    computed in one shot that is a softmax over `concat([sink, scores])` whose
    sink column is dropped before the value matmul.

    q: [num_q_tokens, num_heads, head_dim]
    k, v: [context_len, num_kv_heads, head_dim]   (the full cache)
    Query token i sits at absolute position `context_len - num_q_tokens + i`.
    """
    q = q.to(torch.float64)
    k = k.to(torch.float64)
    v = v.to(torch.float64)
    sinks = sinks.to(torch.float64)

    num_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    first_pos = context_len - num_q

    out = torch.zeros(num_q, num_heads, head_dim, dtype=torch.float64)
    for h in range(num_heads):
        kv_h = h // group
        # [num_q, context_len]
        scores = (q[:, h, :] @ k[:, kv_h, :].transpose(0, 1)) * scale
        qpos = torch.arange(num_q, dtype=torch.int64) + first_pos
        kpos = torch.arange(context_len, dtype=torch.int64)
        mask = kpos[None, :] <= qpos[:, None]
        if sliding_window is not None:
            mask &= kpos[None, :] > (qpos[:, None] - sliding_window)
        scores = scores.masked_fill(~mask, float("-inf"))
        # Softmax with the sink as an extra, value-less column.
        augmented = torch.cat([sinks[h].expand(num_q, 1), scores], dim=1)
        probs = torch.softmax(augmented, dim=-1)[:, 1:]
        out[:, h, :] = probs @ v[:, kv_h, :]
    return out


def ref_topk_softmax(
    router_logits: torch.Tensor, top_k: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """gpt-oss routing: softmax over *all* experts, then top-k, then renormalize."""
    logits = router_logits.to(torch.float64)
    probs = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids


# OCP MX FP4 (E2M1) value table, indexed by the 4-bit code.
_FP4_E2M1_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def ref_dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """OCP MXFP4 dequantisation: E2M1 elements with a shared E8M0 exponent.

    `packed` is uint8 with two 4-bit elements per byte, low nibble first.
    `scales` is uint8 holding the biased power-of-two exponent (E8M0); 255 is
    the NaN code.  One scale covers MXFP4_BLOCK_SIZE consecutive elements.
    """
    table = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float64)
    lo = table[(packed & 0x0F).long()]
    hi = table[(packed >> 4).long()]
    values = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    exponent = scales.to(torch.int64) - 127
    scale = torch.where(
        scales == 255,
        torch.full_like(exponent, 0, dtype=torch.float64).fill_(float("nan")),
        torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent.to(torch.float64)),
    )
    scale = scale.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return values * scale


def ref_moe(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    """Dense-per-token reference for the gpt-oss MoE block.

    w1: [num_experts, 2 * intermediate, hidden] with gate/up interleaved on
    dim 1 (the layout SwiGLU-OAI expects); w2: [num_experts, hidden,
    intermediate].
    """
    hidden = hidden.to(torch.float64)
    out = torch.zeros_like(hidden)
    num_tokens, top_k = ids.shape
    for t in range(num_tokens):
        for j in range(top_k):
            e = int(ids[t, j])
            h = hidden[t] @ w1[e].to(torch.float64).transpose(0, 1) + b1[e].to(
                torch.float64
            )
            a = ref_swiglu_oai(h, SWIGLU_ALPHA, SWIGLU_LIMIT)
            y = a @ w2[e].to(torch.float64).transpose(0, 1) + b2[e].to(torch.float64)
            out[t] += weights[t, j].to(torch.float64) * y
    return out


# --------------------------------------------------------------------------
# Triton kernels.  vLLM runs attention, the MoE GEMM and the MXFP4 dequant
# through Triton on ROCm, so the suite exercises Triton-compiled code as well
# as the library GEMMs that torch eager dispatches.
# --------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    HAVE_TRITON = False
    triton = None
    tl = None


if HAVE_TRITON:

    @triton.jit
    def _rms_norm_kernel(
        out_ptr, x_ptr, w_ptr, stride_row, n_cols, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + row * stride_row + cols,
            (x * rstd * w).to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _swiglu_oai_kernel(
        out_ptr, x_ptr, stride_x, stride_out, d, alpha, limit, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < d
        # gate/up are interleaved: gate at 2*i, up at 2*i + 1.
        gate = tl.load(x_ptr + row * stride_x + 2 * cols, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(x_ptr + row * stride_x + 2 * cols + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = tl.minimum(gate, limit)
        up = tl.maximum(tl.minimum(up, limit), -limit)
        glu = gate * tl.sigmoid(gate * alpha)
        y = (up + 1.0) * glu
        tl.store(
            out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask
        )

    @triton.jit
    def _attention_sinks_kernel(
        out_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        sink_ptr,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        stride_ot,
        stride_oh,
        scale,
        num_q,
        context_len,
        first_pos,
        group,
        sliding_window,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_SWA: tl.constexpr,
    ):
        """One program per (query token, query head).

        Online softmax seeded from the sink, exactly as vLLM's unified
        attention seeds ``M`` with the sink and ``L`` with 1.0.
        """
        t = tl.program_id(0)
        h = tl.program_id(1)
        kv_h = h // group

        d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + t * stride_qt + h * stride_qh + d).to(tl.float32)

        m_i = tl.load(sink_ptr + h).to(tl.float32)
        l_i = 1.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        qpos = first_pos + t
        for start in range(0, context_len, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            kmask = offs < context_len
            k = tl.load(
                k_ptr + offs[:, None] * stride_kt + kv_h * stride_kh + d[None, :],
                mask=kmask[:, None],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + offs[:, None] * stride_kt + kv_h * stride_kh + d[None, :],
                mask=kmask[:, None],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(q[None, :] * k, axis=1) * scale
            valid = kmask & (offs <= qpos)
            if USE_SWA:
                valid = valid & (offs > qpos - sliding_window)
            s = tl.where(valid, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            p = tl.where(valid, p, 0.0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(
            out_ptr + t * stride_ot + h * stride_oh + d,
            out.to(out_ptr.dtype.element_ty),
        )

    @triton.jit
    def _dequant_mxfp4_kernel(
        out_ptr,
        packed_ptr,
        scale_ptr,
        n_bytes,
        n_scales,
        BLOCK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """E2M1 nibbles x E8M0 shared exponent -> float.

        The E2M1 code is decoded arithmetically rather than through a lookup
        table: bit 3 is the sign, bits 2:1 the exponent, bit 0 the mantissa,
        with the subnormal case (exponent 0) handled separately.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_bytes
        byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)

        for half in tl.static_range(2):
            code = (byte >> (4 * half)) & 0xF
            sign = tl.where((code & 0x8) != 0, -1.0, 1.0)
            exp = (code >> 1) & 0x3
            mant = code & 0x1
            # exp == 0 -> subnormal: value = mant * 0.5
            normal = (1.0 + 0.5 * mant.to(tl.float32)) * tl.exp2(
                (exp - 1).to(tl.float32)
            )
            sub = 0.5 * mant.to(tl.float32)
            val = sign * tl.where(exp == 0, sub, normal)

            elem = offs * 2 + half
            sidx = elem // BLOCK_SIZE
            e8m0 = tl.load(
                scale_ptr + sidx, mask=mask & (sidx < n_scales), other=127
            ).to(tl.int32)
            val = val * tl.exp2((e8m0 - 127).to(tl.float32))
            tl.store(out_ptr + elem, val.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _attention_paged_kernel(
        out_ptr,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_table_ptr,
        sink_ptr,
        stride_qt,
        stride_qh,
        stride_blk,
        stride_slot,
        stride_kvh,
        stride_ot,
        stride_oh,
        scale,
        num_q,
        context_len,
        first_pos,
        group,
        sliding_window,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        USE_SWA: tl.constexpr,
    ):
        """Attention over a paged KV cache, one KV block per iteration.

        vLLM never reads a contiguous K/V; it walks a block table and gathers
        each page. The indirection is a different addressing pattern from the
        contiguous cases -- a scalar load feeds a vector address -- so it gets
        its own case.
        """
        t = tl.program_id(0)
        h = tl.program_id(1)
        kv_h = h // group

        d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + t * stride_qt + h * stride_qh + d).to(tl.float32)

        m_i = tl.load(sink_ptr + h).to(tl.float32)
        l_i = 1.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        qpos = first_pos + t

        num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(0, num_blocks):
            phys = tl.load(block_table_ptr + b)
            offs = tl.arange(0, BLOCK_SIZE)
            pos = b * BLOCK_SIZE + offs
            kmask = pos < context_len
            base = phys * stride_blk + offs[:, None] * stride_slot + kv_h * stride_kvh
            k = tl.load(
                k_cache_ptr + base + d[None, :], mask=kmask[:, None], other=0.0
            ).to(tl.float32)
            v = tl.load(
                v_cache_ptr + base + d[None, :], mask=kmask[:, None], other=0.0
            ).to(tl.float32)
            s = tl.sum(q[None, :] * k, axis=1) * scale
            valid = kmask & (pos <= qpos)
            if USE_SWA:
                valid = valid & (pos > qpos - sliding_window)
            s = tl.where(valid, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.where(valid, tl.exp(s - m_new), 0.0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        tl.store(
            out_ptr + t * stride_ot + h * stride_oh + d,
            (acc / l_i).to(out_ptr.dtype.element_ty),
        )

    @triton.jit
    def _dot_gemm_kernel(
        out_ptr,
        a_ptr,
        b_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_am,
        stride_bn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """out = a @ b.T + bias, through tl.dot.

        tl.dot is what lowers to the matrix cores -- MFMA on CDNA, WMMA on RDNA.
        The elementwise Triton kernels elsewhere in this file never reach them,
        so a wrong matrix-core result would go unseen without this.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None],
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                other=0.0,
            )
            acc += tl.dot(a, b)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(
                tl.float32
            )[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _attention_sinks_dot_kernel(
        out_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        sink_ptr,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        stride_ot,
        stride_oh,
        scale,
        num_q,
        context_len,
        first_pos,
        group,
        sliding_window,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_SWA: tl.constexpr,
    ):
        """Flash-attention tiling with sinks, both matmuls through tl.dot.

        One program per (query tile, head). The sink seeds the running max and
        the running sum starts at one, so it contributes exp(sink - m) to the
        denominator and nothing to the numerator -- the same formulation vLLM's
        unified attention kernel uses.
        """
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        kv_h = h // group

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d = tl.arange(0, HEAD_DIM)
        qmask = offs_m < num_q
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qt + h * stride_qh + d[None, :],
            mask=qmask[:, None],
            other=0.0,
        )

        m_i = tl.where(qmask, tl.load(sink_ptr + h).to(tl.float32), float("-inf"))
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        qpos = first_pos + offs_m

        for start in range(0, context_len, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            kmask = offs_n < context_len
            k = tl.load(
                k_ptr + offs_n[None, :] * stride_kt + kv_h * stride_kh + d[:, None],
                mask=kmask[None, :],
                other=0.0,
            )
            v = tl.load(
                v_ptr + offs_n[:, None] * stride_kt + kv_h * stride_kh + d[None, :],
                mask=kmask[:, None],
                other=0.0,
            )
            s = tl.dot(q, k) * scale
            valid = kmask[None, :] & (offs_n[None, :] <= qpos[:, None]) & qmask[:, None]
            if USE_SWA:
                valid = valid & (offs_n[None, :] > qpos[:, None] - sliding_window)
            s = tl.where(valid, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            # A row with no valid key yet keeps m_i; guard the exponent so an
            # all -inf row does not produce inf - inf.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
            alpha = tl.where(m_new == float("-inf"), 1.0, alpha)
            p = tl.where(valid, tl.exp(s - m_safe[:, None]), 0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            out_ptr + offs_m[:, None] * stride_ot + h * stride_oh + d[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=qmask[:, None],
        )

    @triton.jit
    def _moe_gemm_kernel(
        out_ptr,
        x_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        ids_ptr,
        wts_ptr,
        hidden,
        inter,
        top_k,
        stride_w1e,
        stride_w1r,
        stride_w2e,
        stride_w2r,
        alpha,
        limit,
        BLOCK_H: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """One program per (token, selected expert): w1 -> SwiGLU-OAI -> w2.

        Mirrors the shape of vLLM's fused MoE: gather the expert's rows, run
        the gate/up GEMM, apply the activation, run the down GEMM, and scale by
        the routing weight.  The caller sums the top-k partial results.
        """
        t = tl.program_id(0)
        j = tl.program_id(1)
        e = tl.load(ids_ptr + t * top_k + j)
        w = tl.load(wts_ptr + t * top_k + j).to(tl.float32)

        hoff = tl.arange(0, BLOCK_H)
        hmask = hoff < hidden
        x = tl.load(x_ptr + t * hidden + hoff, mask=hmask, other=0.0).to(tl.float32)

        ioff = tl.arange(0, BLOCK_I)
        imask = ioff < inter

        # Gate and up rows are interleaved in w1: row 2*i is gate, 2*i+1 is up.
        gate = tl.zeros([BLOCK_I], dtype=tl.float32)
        up = tl.zeros([BLOCK_I], dtype=tl.float32)
        wg = tl.load(
            w1_ptr + e * stride_w1e + (2 * ioff)[:, None] * stride_w1r + hoff[None, :],
            mask=imask[:, None] & hmask[None, :],
            other=0.0,
        ).to(tl.float32)
        wu = tl.load(
            w1_ptr
            + e * stride_w1e
            + (2 * ioff + 1)[:, None] * stride_w1r
            + hoff[None, :],
            mask=imask[:, None] & hmask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate = tl.sum(wg * x[None, :], axis=1) + tl.load(
            b1_ptr + e * 2 * inter + 2 * ioff, mask=imask, other=0.0
        ).to(tl.float32)
        up = tl.sum(wu * x[None, :], axis=1) + tl.load(
            b1_ptr + e * 2 * inter + 2 * ioff + 1, mask=imask, other=0.0
        ).to(tl.float32)

        gate = tl.minimum(gate, limit)
        up = tl.maximum(tl.minimum(up, limit), -limit)
        act = (up + 1.0) * (gate * tl.sigmoid(gate * alpha))
        act = tl.where(imask, act, 0.0)

        w2 = tl.load(
            w2_ptr + e * stride_w2e + hoff[:, None] * stride_w2r + ioff[None, :],
            mask=hmask[:, None] & imask[None, :],
            other=0.0,
        ).to(tl.float32)
        y = tl.sum(w2 * act[None, :], axis=1)
        y = y + tl.load(b2_ptr + e * hidden + hoff, mask=hmask, other=0.0).to(
            tl.float32
        )
        tl.store(
            out_ptr + (t * top_k + j) * hidden + hoff,
            (w * y).to(out_ptr.dtype.element_ty),
            mask=hmask,
        )


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

# The device the "device arm" of every case runs on.  `main` may point this at
# the CPU: running the identical code on CPU torch is how a failure is
# attributed -- a case that fails on CPU indicts this file's reference or the
# case itself, not the emulator.
DEV = "cuda"


@dataclass
class Result:
    name: str
    role: str
    impl: str
    dtype: str
    status: str  # pass | fail | error | skip
    max_abs_err: float = float("nan")
    max_rel_err: float = float("nan")
    tolerance: float = float("nan")
    seconds: float = float("nan")
    detail: str = ""
    digest: str = ""
    shapes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Case:
    name: str
    role: str
    impl: str
    dtype: torch.dtype
    tolerance: float
    fn: Callable[[Shape, torch.Generator], tuple[torch.Tensor, torch.Tensor]]
    note: str = ""


REGISTRY: list[Case] = []


def case(
    name: str,
    role: str,
    impl: str,
    dtype: torch.dtype,
    tolerance: float,
    note: str = "",
):
    def deco(fn):
        REGISTRY.append(Case(name, role, impl, dtype, tolerance, fn, note))
        return fn

    return deco


def require_triton() -> None:
    """Triton cases need a real GPU; the CPU self-check arm skips them."""
    if not HAVE_TRITON:
        raise SkipCase("triton unavailable")
    if DEV == "cpu":
        raise SkipCase("triton kernels do not run on the CPU self-check arm")


def randn(
    gen: torch.Generator, *shape, dtype=torch.float32, scale: float = 1.0
) -> torch.Tensor:
    """Deterministic CPU-side normal noise, so the reference and the device
    path see bit-identical inputs regardless of device RNG differences."""
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale).to(dtype)


# bfloat16 carries 8 significand bits, so one correctly-rounded bf16 store
# costs up to 2**-8 relative to the value's own magnitude.  Tolerances below
# are quoted as multiples of this, against the largest magnitude in the
# reference -- the scale an elementwise kernel's worst element actually has.
BF16_EPS = 2.0**-8


def compare(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """Max absolute error, and that error relative to the reference's peak.

    Normalising by ``max|want|`` rather than per element keeps a single
    near-zero output from manufacturing a huge relative error out of a result
    that is fine, while still being tight: for an elementwise bf16 kernel the
    quantity is bounded by a small multiple of ``BF16_EPS`` regardless of how
    the values are distributed.
    """
    got = got.detach().to(torch.float64).cpu()
    want = want.detach().to(torch.float64).cpu()
    if got.shape != want.shape:
        raise AssertionError(
            f"shape mismatch: got {tuple(got.shape)} want {tuple(want.shape)}"
        )
    if not torch.isfinite(got).all():
        raise AssertionError("device result contains non-finite values")
    diff = (got - want).abs()
    scale = want.abs().max().item()
    if scale == 0.0:
        scale = 1.0
    return diff.max().item(), (diff.max().item() / scale)


def digest_of(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's raw bytes, in its own dtype.

    This is what makes the suite a cross-architecture differential tester.
    Comparing per-case error magnitudes only catches a divergence that happens
    to change the worst element; comparing the bytes catches any divergence at
    all. Every case feeds both arms identical CPU-generated inputs, so two
    architectures running the same kernel should produce the same bytes, and a
    difference is a fact to explain rather than noise.
    """
    # Reinterpret as bytes rather than going through numpy: numpy has no
    # bfloat16, which is the dtype most of these cases produce.
    raw = t.detach().cpu().contiguous().flatten().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16]


# ---- 1. embedding --------------------------------------------------------


@case(
    "embedding_gather",
    "embedding",
    "torch",
    torch.bfloat16,
    0.0,
    "index_select over the vocab table; must be exact",
)
def _embedding(s: Shape, gen: torch.Generator):
    table = randn(gen, s.vocab_size, s.hidden_size, dtype=torch.bfloat16)
    ids = torch.randint(
        0, s.vocab_size, (s.num_tokens,), generator=gen, dtype=torch.int64
    )
    got = torch.nn.functional.embedding(ids.to(DEV), table.to(DEV))
    return got, table[ids].to(torch.float64)


# ---- 2. normalisation ----------------------------------------------------


@case("rms_norm", "norm", "torch", torch.bfloat16, 6e-3)
def _rms_norm(s: Shape, gen: torch.Generator):
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.5)
    xd, wd = x.to(DEV), w.to(DEV)
    var = xd.to(torch.float32).pow(2).mean(-1, keepdim=True)
    got = (xd.to(torch.float32) * torch.rsqrt(var + RMS_NORM_EPS)).to(
        torch.bfloat16
    ) * wd
    return got, ref_rms_norm(x, w, RMS_NORM_EPS)


@case("rms_norm", "norm", "triton", torch.bfloat16, 6e-3)
def _rms_norm_triton(s: Shape, gen: torch.Generator):
    require_triton()
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.5)
    xd, wd = x.to(DEV), w.to(DEV)
    out = torch.empty_like(xd)
    block = triton.next_power_of_2(s.hidden_size)
    _rms_norm_kernel[(s.num_tokens,)](
        out, xd, wd, xd.stride(0), s.hidden_size, RMS_NORM_EPS, BLOCK=block
    )
    return out, ref_rms_norm(x, w, RMS_NORM_EPS)


@case(
    "fused_add_rms_norm",
    "norm",
    "torch",
    torch.bfloat16,
    6e-3,
    "residual add fused into the norm, as vLLM's two-output RMSNorm does",
)
def _fused_add_rms_norm(s: Shape, gen: torch.Generator):
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    res = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.5)
    xd, resd, wd = x.to(DEV), res.to(DEV), w.to(DEV)
    summed = xd + resd
    var = summed.to(torch.float32).pow(2).mean(-1, keepdim=True)
    got = (summed.to(torch.float32) * torch.rsqrt(var + RMS_NORM_EPS)).to(
        torch.bfloat16
    ) * wd
    # The residual add happens in bf16 in vLLM too, so the reference adds in
    # bf16 and only then widens.
    ref_sum = x + res
    return got, ref_rms_norm(ref_sum, w, RMS_NORM_EPS)


# ---- 3. projections ------------------------------------------------------


@case(
    "qkv_proj_gemm",
    "gemm",
    "torch",
    torch.bfloat16,
    2e-2,
    "fused QKV projection with bias; attention weights are not quantized in gpt-oss",
)
def _qkv(s: Shape, gen: torch.Generator):
    out_dim = s.q_size + 2 * s.kv_size
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w = randn(gen, out_dim, s.hidden_size, dtype=torch.bfloat16, scale=0.05)
    b = randn(gen, out_dim, dtype=torch.bfloat16, scale=0.1)
    got = torch.nn.functional.linear(x.to(DEV), w.to(DEV), b.to(DEV))
    ref = x.to(torch.float64) @ w.to(torch.float64).T + b.to(torch.float64)
    return got, ref


@case(
    "qkv_proj_gemm",
    "gemm",
    "triton_dot",
    torch.bfloat16,
    2e-2,
    "the same projection through tl.dot, so the matrix cores are exercised",
)
def _qkv_dot(s: Shape, gen: torch.Generator):
    require_triton()
    out_dim = s.q_size + 2 * s.kv_size
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w = randn(gen, out_dim, s.hidden_size, dtype=torch.bfloat16, scale=0.05)
    b = randn(gen, out_dim, dtype=torch.bfloat16, scale=0.1)
    xd, wd, bd = x.to(DEV).contiguous(), w.to(DEV).contiguous(), b.to(DEV).contiguous()
    out = torch.empty(s.num_tokens, out_dim, dtype=torch.bfloat16, device=DEV)
    block_m, block_n, block_k = 16, 64, 32
    grid = (triton.cdiv(s.num_tokens, block_m), triton.cdiv(out_dim, block_n))
    _dot_gemm_kernel[grid](
        out,
        xd,
        wd,
        bd,
        s.num_tokens,
        out_dim,
        s.hidden_size,
        xd.stride(0),
        wd.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_BIAS=True,
    )
    ref = x.to(torch.float64) @ w.to(torch.float64).T + b.to(torch.float64)
    return out, ref


@case("o_proj_gemm", "gemm", "torch", torch.bfloat16, 2e-2)
def _o_proj(s: Shape, gen: torch.Generator):
    x = randn(gen, s.num_tokens, s.q_size, dtype=torch.bfloat16)
    w = randn(gen, s.hidden_size, s.q_size, dtype=torch.bfloat16, scale=0.05)
    b = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.1)
    got = torch.nn.functional.linear(x.to(DEV), w.to(DEV), b.to(DEV))
    ref = x.to(torch.float64) @ w.to(torch.float64).T + b.to(torch.float64)
    return got, ref


@case(
    "router_gemm",
    "gemm",
    "torch",
    torch.bfloat16,
    2e-2,
    "the router stays unquantized (modules_to_not_convert) and is tiny",
)
def _router(s: Shape, gen: torch.Generator):
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w = randn(gen, s.num_experts, s.hidden_size, dtype=torch.bfloat16, scale=0.05)
    b = randn(gen, s.num_experts, dtype=torch.bfloat16, scale=0.1)
    got = torch.nn.functional.linear(x.to(DEV), w.to(DEV), b.to(DEV))
    ref = x.to(torch.float64) @ w.to(torch.float64).T + b.to(torch.float64)
    return got, ref


@case("lm_head_gemm", "gemm", "torch", torch.bfloat16, 2e-2)
def _lm_head(s: Shape, gen: torch.Generator):
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w = randn(gen, s.vocab_size, s.hidden_size, dtype=torch.bfloat16, scale=0.05)
    got = torch.nn.functional.linear(x.to(DEV), w.to(DEV))
    ref = x.to(torch.float64) @ w.to(torch.float64).T
    return got, ref


# ---- 4. rotary embedding -------------------------------------------------


@case(
    "rope_yarn_neox",
    "rope",
    "torch",
    torch.bfloat16,
    3e-2,
    "YaRN-scaled NeoX rotary with the gpt-oss theta=150000, factor=32 schedule",
)
def _rope(s: Shape, gen: torch.Generator):
    positions = torch.arange(
        s.context_len - s.num_tokens, s.context_len, dtype=torch.int64
    )
    q = randn(gen, s.num_tokens, s.num_heads, s.head_dim, dtype=torch.bfloat16)
    cos, sin = ref_yarn_cos_sin(positions, s.head_dim)
    cos_d = cos.to(torch.float32).to(DEV)
    sin_d = sin.to(torch.float32).to(DEV)
    qd = q.to(DEV).to(torch.float32)
    half = s.head_dim // 2
    x1, x2 = qd[..., :half], qd[..., half:]
    c, sn = cos_d[:, None, :], sin_d[:, None, :]
    got = torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn], dim=-1).to(torch.bfloat16)
    return got, ref_rope_neox(q, cos, sin)


@case(
    "rope_cos_sin_cache",
    "rope",
    "torch",
    torch.float32,
    1e-5,
    "the YaRN ramp itself: correction range, linear ramp and mscale",
)
def _rope_cache(s: Shape, gen: torch.Generator):
    positions = torch.arange(0, s.context_len, dtype=torch.int64)
    cos_ref, sin_ref = ref_yarn_cos_sin(positions, s.head_dim)
    # Recompute the cache on the device in float32 (what vLLM stores).
    pos_freqs = ROPE_THETA ** (
        torch.arange(0, s.head_dim, 2, dtype=torch.float32, device=DEV) / s.head_dim
    )
    low, high = _yarn_find_correction_range(
        ROPE_BETA_FAST,
        ROPE_BETA_SLOW,
        s.head_dim,
        ROPE_THETA,
        ROPE_ORIGINAL_MAX_POSITION,
    )
    ramp = (
        (torch.arange(s.head_dim // 2, dtype=torch.float32, device=DEV) - low)
        / (high - low if high != low else 0.001)
    ).clamp(0, 1)
    inv_freq_mask = 1 - ramp
    inv_freq = (1.0 / (ROPE_FACTOR * pos_freqs)) * (1 - inv_freq_mask) + (
        1.0 / pos_freqs
    ) * inv_freq_mask
    mscale = 0.1 * math.log(ROPE_FACTOR) + 1.0
    freqs = positions.to(torch.float32).to(DEV)[:, None] * inv_freq[None, :]
    got = torch.cat([freqs.cos() * mscale, freqs.sin() * mscale], dim=-1)
    return got, torch.cat([cos_ref, sin_ref], dim=-1)


# ---- 5. KV cache ---------------------------------------------------------


@case(
    "kv_cache_store",
    "copy",
    "torch",
    torch.bfloat16,
    0.0,
    "scatter of new K/V into a paged cache; must be exact",
)
def _kv_cache(s: Shape, gen: torch.Generator):
    block_size = 16
    num_blocks = (s.context_len + block_size - 1) // block_size + 1
    cache = torch.zeros(
        num_blocks,
        block_size,
        s.num_kv_heads,
        s.head_dim,
        dtype=torch.bfloat16,
        device=DEV,
    )
    k = randn(gen, s.num_tokens, s.num_kv_heads, s.head_dim, dtype=torch.bfloat16)
    slots = torch.arange(s.context_len - s.num_tokens, s.context_len, dtype=torch.int64)
    flat = cache.view(-1, s.num_kv_heads, s.head_dim)
    flat.index_copy_(0, slots.to(DEV), k.to(DEV))
    ref = torch.zeros(
        num_blocks * block_size, s.num_kv_heads, s.head_dim, dtype=torch.float64
    )
    ref[slots] = k.to(torch.float64)
    return flat, ref


# ---- 6. attention --------------------------------------------------------


def _attention_inputs(s: Shape, gen: torch.Generator):
    q = randn(
        gen, s.num_tokens, s.num_heads, s.head_dim, dtype=torch.bfloat16, scale=0.5
    )
    k = randn(
        gen, s.context_len, s.num_kv_heads, s.head_dim, dtype=torch.bfloat16, scale=0.5
    )
    v = randn(
        gen, s.context_len, s.num_kv_heads, s.head_dim, dtype=torch.bfloat16, scale=0.5
    )
    sinks = randn(gen, s.num_heads, dtype=torch.float32, scale=1.0)
    return q, k, v, sinks


def _attention_torch(s, q, k, v, sinks, window):
    """Eager reproduction of the online-softmax-with-sink formulation."""
    scale = s.head_dim**-0.5
    qd = q.to(DEV).to(torch.float32)
    kd = k.to(DEV).to(torch.float32)
    vd = v.to(DEV).to(torch.float32)
    sd = sinks.to(DEV).to(torch.float32)
    group = s.num_heads // s.num_kv_heads
    first_pos = s.context_len - s.num_tokens
    qpos = torch.arange(s.num_tokens, device=DEV) + first_pos
    kpos = torch.arange(s.context_len, device=DEV)
    mask = kpos[None, :] <= qpos[:, None]
    if window is not None:
        mask &= kpos[None, :] > (qpos[:, None] - window)
    # [heads, num_q, context]
    kk = kd.repeat_interleave(group, dim=1)
    vv = vd.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", qd, kk) * scale
    scores = scores.masked_fill(~mask[None], float("-inf"))
    aug = torch.cat(
        [sd[:, None, None].expand(s.num_heads, s.num_tokens, 1), scores], dim=-1
    )
    probs = torch.softmax(aug, dim=-1)[..., 1:]
    out = torch.einsum("hqk,khd->qhd", probs, vv)
    return out.to(torch.bfloat16)


@case(
    "attention_prefill_full",
    "attention",
    "torch",
    torch.bfloat16,
    3e-2,
    "full causal attention with sinks (odd gpt-oss layers)",
)
def _attn_full(s: Shape, gen: torch.Generator):
    q, k, v, sinks = _attention_inputs(s, gen)
    got = _attention_torch(s, q, k, v, sinks, None)
    return got, ref_attention_sinks(
        q, k, v, sinks, s.head_dim**-0.5, s.context_len, None
    )


@case(
    "attention_prefill_swa",
    "attention",
    "torch",
    torch.bfloat16,
    3e-2,
    "sliding-window attention with sinks (even gpt-oss layers)",
)
def _attn_swa(s: Shape, gen: torch.Generator):
    q, k, v, sinks = _attention_inputs(s, gen)
    got = _attention_torch(s, q, k, v, sinks, s.sliding_window)
    return got, ref_attention_sinks(
        q, k, v, sinks, s.head_dim**-0.5, s.context_len, s.sliding_window
    )


@case("attention_prefill_full", "attention", "triton", torch.bfloat16, 3e-2)
def _attn_full_triton(s: Shape, gen: torch.Generator):
    require_triton()
    q, k, v, sinks = _attention_inputs(s, gen)
    qd, kd, vd = q.to(DEV), k.to(DEV), v.to(DEV)
    sd = sinks.to(DEV)
    out = torch.empty_like(qd)
    _attention_sinks_kernel[(s.num_tokens, s.num_heads)](
        out,
        qd,
        kd,
        vd,
        sd,
        qd.stride(0),
        qd.stride(1),
        kd.stride(0),
        kd.stride(1),
        out.stride(0),
        out.stride(1),
        s.head_dim**-0.5,
        s.num_tokens,
        s.context_len,
        s.context_len - s.num_tokens,
        s.num_heads // s.num_kv_heads,
        0,
        HEAD_DIM=s.head_dim,
        BLOCK_N=16,
        USE_SWA=False,
    )
    return out, ref_attention_sinks(
        q, k, v, sinks, s.head_dim**-0.5, s.context_len, None
    )


@case("attention_prefill_swa", "attention", "triton", torch.bfloat16, 3e-2)
def _attn_swa_triton(s: Shape, gen: torch.Generator):
    require_triton()
    q, k, v, sinks = _attention_inputs(s, gen)
    qd, kd, vd = q.to(DEV), k.to(DEV), v.to(DEV)
    sd = sinks.to(DEV)
    out = torch.empty_like(qd)
    _attention_sinks_kernel[(s.num_tokens, s.num_heads)](
        out,
        qd,
        kd,
        vd,
        sd,
        qd.stride(0),
        qd.stride(1),
        kd.stride(0),
        kd.stride(1),
        out.stride(0),
        out.stride(1),
        s.head_dim**-0.5,
        s.num_tokens,
        s.context_len,
        s.context_len - s.num_tokens,
        s.num_heads // s.num_kv_heads,
        s.sliding_window,
        HEAD_DIM=s.head_dim,
        BLOCK_N=16,
        USE_SWA=True,
    )
    return out, ref_attention_sinks(
        q, k, v, sinks, s.head_dim**-0.5, s.context_len, s.sliding_window
    )


@case(
    "attention_decode",
    "attention",
    "torch",
    torch.bfloat16,
    3e-2,
    "single-query decode step against a full cache",
)
def _attn_decode(s: Shape, gen: torch.Generator):
    one = dataclasses.replace(s, num_tokens=1)
    q, k, v, sinks = _attention_inputs(one, gen)
    got = _attention_torch(one, q, k, v, sinks, None)
    return got, ref_attention_sinks(
        q, k, v, sinks, one.head_dim**-0.5, one.context_len, None
    )


# ---- 7. routing ----------------------------------------------------------


def _attn_dot(s: Shape, gen: torch.Generator, window: int | None):
    require_triton()
    q, k, v, sinks = _attention_inputs(s, gen)
    qd, kd, vd = q.to(DEV).contiguous(), k.to(DEV).contiguous(), v.to(DEV).contiguous()
    sd = sinks.to(DEV).contiguous()
    out = torch.empty_like(qd)
    block_m, block_n = 16, 16
    _attention_sinks_dot_kernel[(triton.cdiv(s.num_tokens, block_m), s.num_heads)](
        out,
        qd,
        kd,
        vd,
        sd,
        qd.stride(0),
        qd.stride(1),
        kd.stride(0),
        kd.stride(1),
        out.stride(0),
        out.stride(1),
        s.head_dim**-0.5,
        s.num_tokens,
        s.context_len,
        s.context_len - s.num_tokens,
        s.num_heads // s.num_kv_heads,
        window if window is not None else 0,
        HEAD_DIM=s.head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        USE_SWA=window is not None,
    )
    ref = ref_attention_sinks(q, k, v, sinks, s.head_dim**-0.5, s.context_len, window)
    return out, ref


@case(
    "attention_prefill_full",
    "attention",
    "triton_dot",
    torch.bfloat16,
    3e-2,
    "flash tiling with both matmuls through tl.dot, the form vLLM's kernel uses",
)
def _attn_full_dot(s: Shape, gen: torch.Generator):
    return _attn_dot(s, gen, None)


@case("attention_prefill_swa", "attention", "triton_dot", torch.bfloat16, 3e-2)
def _attn_swa_dot(s: Shape, gen: torch.Generator):
    return _attn_dot(s, gen, s.sliding_window)


@case(
    "attention_paged",
    "attention",
    "triton",
    torch.bfloat16,
    3e-2,
    "attention over a paged KV cache walked through a block table, as vLLM does",
)
def _attn_paged(s: Shape, gen: torch.Generator):
    require_triton()
    q, k, v, sinks = _attention_inputs(s, gen)
    block_size = 16
    num_blocks = (s.context_len + block_size - 1) // block_size
    # Shuffle the physical blocks so a kernel that ignores the table and reads
    # the cache contiguously gets a different answer.
    perm = torch.randperm(num_blocks, generator=gen)
    k_cache = torch.zeros(
        num_blocks, block_size, s.num_kv_heads, s.head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)
    for logical in range(num_blocks):
        phys = int(perm[logical])
        lo = logical * block_size
        hi = min(lo + block_size, s.context_len)
        k_cache[phys, : hi - lo] = k[lo:hi]
        v_cache[phys, : hi - lo] = v[lo:hi]

    kd, vd = k_cache.to(DEV).contiguous(), v_cache.to(DEV).contiguous()
    qd, sd = q.to(DEV).contiguous(), sinks.to(DEV).contiguous()
    table = perm.to(torch.int32).to(DEV).contiguous()
    out = torch.empty_like(qd)
    _attention_paged_kernel[(s.num_tokens, s.num_heads)](
        out,
        qd,
        kd,
        vd,
        table,
        sd,
        qd.stride(0),
        qd.stride(1),
        kd.stride(0),
        kd.stride(1),
        kd.stride(2),
        out.stride(0),
        out.stride(1),
        s.head_dim**-0.5,
        s.num_tokens,
        s.context_len,
        s.context_len - s.num_tokens,
        s.num_heads // s.num_kv_heads,
        s.sliding_window,
        HEAD_DIM=s.head_dim,
        BLOCK_SIZE=block_size,
        USE_SWA=True,
    )
    ref = ref_attention_sinks(
        q, k, v, sinks, s.head_dim**-0.5, s.context_len, s.sliding_window
    )
    return out, ref


@case(
    "topk_softmax",
    "moe_route",
    "torch",
    torch.float32,
    1e-5,
    "softmax over all experts, then top-k and renormalize",
)
def _topk(s: Shape, gen: torch.Generator):
    logits = randn(gen, s.num_tokens, s.num_experts, dtype=torch.float32, scale=2.0)
    probs = torch.softmax(logits.to(DEV), dim=-1)
    weights, ids = torch.topk(probs, s.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ref_w, ref_ids = ref_topk_softmax(logits, s.top_k, True)
    if not torch.equal(ids.cpu().to(torch.int64), ref_ids):
        raise AssertionError(
            f"expert selection differs: {ids.cpu().tolist()} vs {ref_ids.tolist()}"
        )
    return weights, ref_w


# ---- 8. MXFP4 ------------------------------------------------------------


def _mxfp4_inputs(s: Shape, gen: torch.Generator, rows: int, cols: int):
    """Random MXFP4 payload: packed E2M1 nibbles plus one E8M0 exponent per
    block of 32 elements.  Exponent codes are kept away from 0 and 255 so the
    reference and the device agree on finite, normal values."""
    assert cols % MXFP4_BLOCK_SIZE == 0
    packed = torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8)
    scales = torch.randint(
        120, 135, (rows, cols // MXFP4_BLOCK_SIZE), generator=gen, dtype=torch.uint8
    )
    return packed, scales


@case(
    "mxfp4_dequant",
    "quant",
    "torch",
    torch.bfloat16,
    0.0,
    "E2M1 x E8M0 -> bf16; every representable value is exact in bf16",
)
def _mxfp4_torch(s: Shape, gen: torch.Generator):
    rows, cols = s.num_experts, s.hidden_size
    packed, scales = _mxfp4_inputs(s, gen, rows, cols)
    table = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=DEV)
    pd = packed.to(DEV)
    lo = table[(pd & 0x0F).long()]
    hi = table[(pd >> 4).long()]
    values = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    exponent = scales.to(DEV).to(torch.int32) - 127
    scale = torch.exp2(exponent.to(torch.float32)).repeat_interleave(
        MXFP4_BLOCK_SIZE, dim=-1
    )
    got = (values * scale).to(torch.bfloat16)
    return got, ref_dequant_mxfp4(packed, scales)


@case(
    "mxfp4_dequant",
    "quant",
    "triton",
    torch.bfloat16,
    0.0,
    "arithmetic E2M1 decode, the form a fused dequant kernel uses",
)
def _mxfp4_triton(s: Shape, gen: torch.Generator):
    require_triton()
    rows, cols = s.num_experts, s.hidden_size
    packed, scales = _mxfp4_inputs(s, gen, rows, cols)
    pd = packed.to(DEV).contiguous().view(-1)
    sd = scales.to(DEV).contiguous().view(-1)
    out = torch.empty(pd.numel() * 2, dtype=torch.bfloat16, device=DEV)
    block = 128
    grid = ((pd.numel() + block - 1) // block,)
    _dequant_mxfp4_kernel[grid](
        out, pd, sd, pd.numel(), sd.numel(), BLOCK=block, BLOCK_SIZE=MXFP4_BLOCK_SIZE
    )
    return out.view(rows, cols), ref_dequant_mxfp4(packed, scales)


# ---- 9. activation -------------------------------------------------------


@case(
    "swiglu_oai",
    "activation",
    "torch",
    torch.bfloat16,
    6e-3,
    "clamped SwiGLU with alpha=1.702, limit=7.0 and interleaved gate/up",
)
def _swiglu_torch(s: Shape, gen: torch.Generator):
    # Scale wide enough that the one-sided clamp is actually exercised.
    x = randn(
        gen, s.num_tokens, 2 * s.intermediate_size, dtype=torch.bfloat16, scale=6.0
    )
    xd = x.to(DEV).to(torch.float32)
    gate, up = xd[..., ::2], xd[..., 1::2]
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    got = ((up + 1) * (gate * torch.sigmoid(gate * SWIGLU_ALPHA))).to(torch.bfloat16)
    return got, ref_swiglu_oai(x, SWIGLU_ALPHA, SWIGLU_LIMIT)


@case("swiglu_oai", "activation", "triton", torch.bfloat16, 6e-3)
def _swiglu_triton(s: Shape, gen: torch.Generator):
    require_triton()
    x = randn(
        gen, s.num_tokens, 2 * s.intermediate_size, dtype=torch.bfloat16, scale=6.0
    )
    xd = x.to(DEV)
    out = torch.empty(
        s.num_tokens, s.intermediate_size, dtype=torch.bfloat16, device=DEV
    )
    block = triton.next_power_of_2(s.intermediate_size)
    _swiglu_oai_kernel[(s.num_tokens,)](
        out,
        xd,
        xd.stride(0),
        out.stride(0),
        s.intermediate_size,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        BLOCK=block,
    )
    return out, ref_swiglu_oai(x, SWIGLU_ALPHA, SWIGLU_LIMIT)


# ---- 10. MoE -------------------------------------------------------------


def _moe_inputs(s: Shape, gen: torch.Generator):
    hidden = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    w1 = randn(
        gen,
        s.num_experts,
        2 * s.intermediate_size,
        s.hidden_size,
        dtype=torch.bfloat16,
        scale=0.05,
    )
    b1 = randn(
        gen, s.num_experts, 2 * s.intermediate_size, dtype=torch.bfloat16, scale=0.1
    )
    w2 = randn(
        gen,
        s.num_experts,
        s.hidden_size,
        s.intermediate_size,
        dtype=torch.bfloat16,
        scale=0.05,
    )
    b2 = randn(gen, s.num_experts, s.hidden_size, dtype=torch.bfloat16, scale=0.1)
    logits = randn(gen, s.num_tokens, s.num_experts, dtype=torch.float32, scale=2.0)
    weights, ids = ref_topk_softmax(logits, s.top_k, True)
    return hidden, w1, b1, w2, b2, weights.to(torch.float32), ids.to(torch.int32)


@case(
    "moe_gemm_swiglu",
    "moe_gemm",
    "torch",
    torch.bfloat16,
    4e-2,
    "gate/up GEMM -> SwiGLU-OAI -> down GEMM, weighted by the routing scores",
)
def _moe_torch(s: Shape, gen: torch.Generator):
    hidden, w1, b1, w2, b2, weights, ids = _moe_inputs(s, gen)
    hd, w1d, b1d, w2d, b2d = (t.to(DEV) for t in (hidden, w1, b1, w2, b2))
    wts, idd = weights.to(DEV), ids.to(DEV).to(torch.int64)
    out = torch.zeros(s.num_tokens, s.hidden_size, dtype=torch.float32, device=DEV)
    for j in range(s.top_k):
        e = idd[:, j]
        h = torch.einsum(
            "th,tih->ti", hd.to(torch.float32), w1d[e].to(torch.float32)
        ) + b1d[e].to(torch.float32)
        gate, up = h[..., ::2], h[..., 1::2]
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        a = (up + 1) * (gate * torch.sigmoid(gate * SWIGLU_ALPHA))
        y = torch.einsum("ti,thi->th", a, w2d[e].to(torch.float32)) + b2d[e].to(
            torch.float32
        )
        out += wts[:, j : j + 1] * y
    ref = ref_moe(hidden, w1, b1, w2, b2, weights, ids)
    return out.to(torch.bfloat16), ref


@case("moe_gemm_swiglu", "moe_gemm", "triton", torch.bfloat16, 4e-2)
def _moe_triton(s: Shape, gen: torch.Generator):
    require_triton()
    hidden, w1, b1, w2, b2, weights, ids = _moe_inputs(s, gen)
    hd = hidden.to(DEV).contiguous()
    w1d, b1d = w1.to(DEV).contiguous(), b1.to(DEV).contiguous()
    w2d, b2d = w2.to(DEV).contiguous(), b2.to(DEV).contiguous()
    wts, idd = weights.to(DEV).contiguous(), ids.to(DEV).contiguous()
    partial = torch.zeros(
        s.num_tokens * s.top_k, s.hidden_size, dtype=torch.float32, device=DEV
    )
    _moe_gemm_kernel[(s.num_tokens, s.top_k)](
        partial,
        hd,
        w1d,
        b1d,
        w2d,
        b2d,
        idd,
        wts,
        s.hidden_size,
        s.intermediate_size,
        s.top_k,
        w1d.stride(0),
        w1d.stride(1),
        w2d.stride(0),
        w2d.stride(1),
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        BLOCK_H=triton.next_power_of_2(s.hidden_size),
        BLOCK_I=triton.next_power_of_2(s.intermediate_size),
    )
    got = (
        partial.view(s.num_tokens, s.top_k, s.hidden_size).sum(dim=1).to(torch.bfloat16)
    )
    ref = ref_moe(hidden, w1, b1, w2, b2, weights, ids)
    return got, ref


# ---- 11. sampling --------------------------------------------------------


@case(
    "greedy_sample",
    "sampling",
    "torch",
    torch.float32,
    0.0,
    "argmax over the vocabulary; the token id must match exactly",
)
def _sample(s: Shape, gen: torch.Generator):
    logits = randn(gen, s.num_tokens, s.vocab_size, dtype=torch.float32, scale=3.0)
    got = torch.argmax(logits.to(DEV), dim=-1)
    ref = torch.argmax(logits.to(torch.float64), dim=-1)
    return got.to(torch.float64), ref.to(torch.float64)


# ---- 12. integration -----------------------------------------------------


@case(
    "transformer_block",
    "integration",
    "torch",
    torch.bfloat16,
    6e-2,
    "one full gpt-oss layer: norm -> QKV -> RoPE -> attention -> O -> norm -> MoE",
)
def _block(s: Shape, gen: torch.Generator):
    out_dim = s.q_size + 2 * s.kv_size
    x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
    n1 = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.5)
    wqkv = randn(gen, out_dim, s.hidden_size, dtype=torch.bfloat16, scale=0.05)
    bqkv = randn(gen, out_dim, dtype=torch.bfloat16, scale=0.1)
    sinks = randn(gen, s.num_heads, dtype=torch.float32)
    wo = randn(gen, s.hidden_size, s.q_size, dtype=torch.bfloat16, scale=0.05)
    bo = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.1)
    n2 = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.5)
    wr = randn(gen, s.num_experts, s.hidden_size, dtype=torch.bfloat16, scale=0.05)
    br = randn(gen, s.num_experts, dtype=torch.bfloat16, scale=0.1)
    w1 = randn(
        gen,
        s.num_experts,
        2 * s.intermediate_size,
        s.hidden_size,
        dtype=torch.bfloat16,
        scale=0.05,
    )
    b1 = randn(
        gen, s.num_experts, 2 * s.intermediate_size, dtype=torch.bfloat16, scale=0.1
    )
    w2 = randn(
        gen,
        s.num_experts,
        s.hidden_size,
        s.intermediate_size,
        dtype=torch.bfloat16,
        scale=0.05,
    )
    b2 = randn(gen, s.num_experts, s.hidden_size, dtype=torch.bfloat16, scale=0.1)

    positions = torch.arange(
        s.context_len - s.num_tokens, s.context_len, dtype=torch.int64
    )
    cos, sin = ref_yarn_cos_sin(positions, s.head_dim)
    # Past K/V that the new tokens attend to.
    kv_past = s.context_len - s.num_tokens
    k_past = randn(
        gen, kv_past, s.num_kv_heads, s.head_dim, dtype=torch.bfloat16, scale=0.5
    )
    v_past = randn(
        gen, kv_past, s.num_kv_heads, s.head_dim, dtype=torch.bfloat16, scale=0.5
    )

    def run(dev: bool):
        """Both arms share this body; `dev` picks device+bf16 or CPU+float64."""
        if dev:
            to = lambda t: t.to(DEV)
            acc = torch.float32
            cast = lambda t: t.to(torch.bfloat16)
        else:
            to = lambda t: t.to(torch.float64)
            acc = torch.float64
            cast = lambda t: t

        h = to(x)
        var = h.to(acc).pow(2).mean(-1, keepdim=True)
        normed = cast(h.to(acc) * torch.rsqrt(var + RMS_NORM_EPS)) * to(n1)
        qkv = torch.nn.functional.linear(normed, to(wqkv), to(bqkv))
        q, k, v = qkv.split([s.q_size, s.kv_size, s.kv_size], dim=-1)
        q = q.view(s.num_tokens, s.num_heads, s.head_dim)
        k = k.view(s.num_tokens, s.num_kv_heads, s.head_dim)
        v = v.view(s.num_tokens, s.num_kv_heads, s.head_dim)

        c = to(cos.to(torch.float32) if dev else cos)[:, None, :]
        sn = to(sin.to(torch.float32) if dev else sin)[:, None, :]
        half = s.head_dim // 2
        rot = lambda t: torch.cat(
            [
                t[..., :half].to(acc) * c - t[..., half:].to(acc) * sn,
                t[..., half:].to(acc) * c + t[..., :half].to(acc) * sn,
            ],
            dim=-1,
        )
        q = cast(rot(q))
        k = cast(rot(k))

        k_all = torch.cat([to(k_past).to(k.dtype), k], dim=0)
        v_all = torch.cat([to(v_past).to(v.dtype), v], dim=0)

        if dev:
            attn = _attention_torch(s, q, k_all, v_all, sinks, s.sliding_window)
        else:
            attn = ref_attention_sinks(
                q,
                k_all,
                v_all,
                sinks,
                s.head_dim**-0.5,
                s.context_len,
                s.sliding_window,
            )
        attn = attn.reshape(s.num_tokens, s.q_size)
        h = torch.nn.functional.linear(cast(attn), to(wo), to(bo)) + to(x)

        var = h.to(acc).pow(2).mean(-1, keepdim=True)
        normed = cast(h.to(acc) * torch.rsqrt(var + RMS_NORM_EPS)) * to(n2)
        logits = torch.nn.functional.linear(normed, to(wr), to(br))
        probs = torch.softmax(logits.to(acc), dim=-1)
        wts, ids = torch.topk(probs, s.top_k, dim=-1)
        wts = wts / wts.sum(-1, keepdim=True)

        if dev:
            moe = torch.zeros(
                s.num_tokens, s.hidden_size, dtype=torch.float32, device=DEV
            )
            for j in range(s.top_k):
                e = ids[:, j]
                hh = torch.einsum(
                    "th,tih->ti", normed.to(torch.float32), to(w1)[e].to(torch.float32)
                ) + to(b1)[e].to(torch.float32)
                gate, up = hh[..., ::2], hh[..., 1::2]
                gate = gate.clamp(max=SWIGLU_LIMIT)
                up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
                a = (up + 1) * (gate * torch.sigmoid(gate * SWIGLU_ALPHA))
                y = torch.einsum("ti,thi->th", a, to(w2)[e].to(torch.float32)) + to(b2)[
                    e
                ].to(torch.float32)
                moe += wts[:, j : j + 1].to(torch.float32) * y
            moe = moe.to(torch.bfloat16)
        else:
            moe = ref_moe(normed, w1, b1, w2, b2, wts, ids)
        return (moe.to(acc) + h.to(acc)), ids

    got, got_ids = run(True)
    want, want_ids = run(False)
    if not torch.equal(got_ids.cpu().to(torch.int64), want_ids.to(torch.int64)):
        raise AssertionError(
            f"routing diverged between device and reference: "
            f"{got_ids.cpu().tolist()} vs {want_ids.tolist()}"
        )
    return got, want


# --------------------------------------------------------------------------
# vLLM cross-check (optional)
# --------------------------------------------------------------------------


def vllm_crosschecks(s: Shape, gen: torch.Generator) -> list[Result]:
    """Compare vLLM's own custom ops against this file's references.

    Only runs with --with-vllm and only reports what actually imported, so the
    suite stays usable in an environment where vLLM is not installed (which is
    the normal case on at least one of the two emulated targets).
    """
    results: list[Result] = []

    # vLLM's CustomOp subclasses resolve their dispatch (native versus the
    # compiled `_C` op) from the ambient VllmConfig, and assert rather than
    # guess when there is none. Constructing one outside an engine therefore
    # needs the config context an engine would otherwise have entered.
    try:
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

        # `custom_ops=["all"]` selects the compiled `_C` kernels over the
        # torch-native fallbacks. Those are what an engine dispatches, and the
        # only ones that put anything on the GPU worth emulating; without this
        # the cross-check silently compares two CPU-side torch expressions.
        # `custom_ops=["all"]` selects the compiled `_C` kernels over the
        # torch-native fallbacks, which is what an engine dispatches and the
        # only form that puts anything on the GPU. It does not reach RMSNorm:
        # that one is chosen from an IR op-priority list an engine populates
        # from the platform defaults, and outside an engine it resolves to the
        # native implementation. The rms_norm row therefore cross-checks this
        # file's reference against vLLM's torch expression, not against the
        # compiled kernel; the suite's own `rms_norm` cases cover the device.
        config_ctx = set_current_vllm_config(
            VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
        )
    except Exception as exc:  # noqa: BLE001
        return [
            Result(
                "vllm_crosscheck",
                "meta",
                "vllm",
                "n/a",
                "skip",
                detail=f"vLLM unavailable: {type(exc).__name__}: {exc}",
            )
        ]

    def record(name, role, fn, tol, impl="vllm", dtype="bfloat16"):
        t0 = time.time()
        try:
            got, want = fn()
            abs_err, rel_err = compare(got, want)
            status = "pass" if rel_err <= tol else "fail"
            results.append(
                Result(
                    name,
                    role,
                    "vllm",
                    dtype,
                    status,
                    abs_err,
                    rel_err,
                    tol,
                    time.time() - t0,
                )
            )
        except SkipCase as exc:
            results.append(Result(name, role, impl, dtype, "skip", detail=str(exc)))
        except Exception as exc:  # noqa: BLE001
            results.append(
                Result(
                    name,
                    role,
                    "vllm",
                    dtype,
                    "error",
                    detail=f"{type(exc).__name__}: {exc}",
                    seconds=time.time() - t0,
                )
            )

    def hf_attention():
        """Cross-check the sink-attention reference against HuggingFace's.

        This is the one reference in the file with no other independent
        witness: vLLM computes attention inside an attention backend that
        cannot be driven standalone, so `--with-vllm` reaches RMSNorm, SwiGLU
        and rotary but not this. `transformers` implements the same operator in
        eager PyTorch, which does give a second opinion on the formulation
        (concatenate the sink as an extra logit, softmax, drop the sink column)
        rather than on this file's arithmetic.
        """
        try:
            from transformers.models.gpt_oss.modeling_gpt_oss import (
                eager_attention_forward,
            )
        except ImportError as exc:
            raise SkipCase(f"transformers gpt_oss unavailable: {exc}") from exc

        q, k, v, sinks = _attention_inputs(s, gen)
        group = s.num_heads // s.num_kv_heads

        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_key_value_groups = group
                self.sinks = torch.nn.Parameter(
                    sinks.to(torch.float32), requires_grad=False
                )

        # HF lays tensors out as [batch, heads, seq, head_dim] and takes an
        # additive mask, where this file masks with -inf inside the reference.
        qh = q.to(torch.float32).permute(1, 0, 2).unsqueeze(0)
        kh = k.to(torch.float32).permute(1, 0, 2).unsqueeze(0)
        vh = v.to(torch.float32).permute(1, 0, 2).unsqueeze(0)
        first_pos = s.context_len - s.num_tokens
        qpos = torch.arange(s.num_tokens) + first_pos
        kpos = torch.arange(s.context_len)
        allowed = (kpos[None, :] <= qpos[:, None]) & (
            kpos[None, :] > qpos[:, None] - s.sliding_window
        )
        mask = torch.where(allowed, 0.0, float("-inf")).to(torch.float32)[None, None]

        out, _ = eager_attention_forward(Stub(), qh, kh, vh, mask, s.head_dim**-0.5)
        got = out[0].reshape(s.num_tokens, s.num_heads, s.head_dim)
        want = ref_attention_sinks(
            q, k, v, sinks, s.head_dim**-0.5, s.context_len, s.sliding_window
        )
        return got, want

    def rms():
        from vllm.model_executor.layers.layernorm import RMSNorm

        x = randn(gen, s.num_tokens, s.hidden_size, dtype=torch.bfloat16)
        w = randn(gen, s.hidden_size, dtype=torch.bfloat16, scale=0.5)
        layer = RMSNorm(s.hidden_size, eps=RMS_NORM_EPS).to(DEV).to(torch.bfloat16)
        with torch.no_grad():
            layer.weight.copy_(w.to(DEV))
        return layer(x.to(DEV)), ref_rms_norm(x, w, RMS_NORM_EPS)

    def swiglu():
        from vllm.model_executor.layers.activation import SwigluOAIAndMul

        x = randn(
            gen, s.num_tokens, 2 * s.intermediate_size, dtype=torch.bfloat16, scale=6.0
        )
        act = SwigluOAIAndMul(alpha=SWIGLU_ALPHA, limit=SWIGLU_LIMIT).to(DEV)
        return act(x.to(DEV)), ref_swiglu_oai(x, SWIGLU_ALPHA, SWIGLU_LIMIT)

    def rope():
        from vllm.model_executor.layers.rotary_embedding import get_rope

        rope_layer = get_rope(
            s.head_dim,
            max_position=131072,
            dtype=torch.float32,
            rope_parameters={
                "rope_theta": ROPE_THETA,
                "rope_type": "yarn",
                "factor": ROPE_FACTOR,
                "original_max_position_embeddings": ROPE_ORIGINAL_MAX_POSITION,
                "beta_fast": ROPE_BETA_FAST,
                "beta_slow": ROPE_BETA_SLOW,
                "truncate": True,
            },
            is_neox_style=True,
        ).to(DEV)
        positions = torch.arange(
            s.context_len - s.num_tokens, s.context_len, dtype=torch.int64
        )
        q = randn(gen, s.num_tokens, s.num_heads, s.head_dim, dtype=torch.bfloat16)
        k = randn(gen, s.num_tokens, s.num_kv_heads, s.head_dim, dtype=torch.bfloat16)
        qo, _ = rope_layer(
            positions.to(DEV),
            q.reshape(s.num_tokens, -1).to(DEV),
            k.reshape(s.num_tokens, -1).to(DEV),
        )
        cos, sin = ref_yarn_cos_sin(positions, s.head_dim)
        return qo.view(s.num_tokens, s.num_heads, s.head_dim), ref_rope_neox(
            q, cos, sin
        )

    with config_ctx:
        record("rms_norm", "norm", rms, 6e-3)
        record("swiglu_oai", "activation", swiglu, 6e-3)
        record("rope_yarn_neox", "rope", rope, 3e-2)
    record("attention_sinks", "attention", hf_attention, 3e-2, impl="hf")
    return results


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "triton": getattr(triton, "__version__", None) if HAVE_TRITON else None,
        "host": platform.node(),
    }
    if DEV != "cpu" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "device_name": props.name,
                "gcn_arch": getattr(props, "gcnArchName", None),
                "multi_processor_count": props.multi_processor_count,
                "total_memory_mib": props.total_memory // (1024 * 1024),
                "warp_size": getattr(props, "warp_size", None),
            }
        )
    return info


def run_case(c: Case, s: Shape, seed: int) -> Result:
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    try:
        got, want = c.fn(s, gen)
        abs_err, rel_err = compare(got, want)
        status = "pass" if rel_err <= c.tolerance else "fail"
        return Result(
            c.name,
            c.role,
            c.impl,
            str(c.dtype).replace("torch.", ""),
            status,
            abs_err,
            rel_err,
            c.tolerance,
            time.time() - t0,
            digest=digest_of(got),
            shapes={"profile": s.name},
        )
    except SkipCase as exc:
        return Result(
            c.name,
            c.role,
            c.impl,
            str(c.dtype).replace("torch.", ""),
            "skip",
            detail=str(exc),
            seconds=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            c.name,
            c.role,
            c.impl,
            str(c.dtype).replace("torch.", ""),
            "error",
            detail=f"{type(exc).__name__}: {exc}",
            seconds=time.time() - t0,
            shapes={"traceback": traceback.format_exc(limit=6)},
        )


def compare_reports(path_a: str, path_b: str) -> int:
    """Diff two runs of this suite case by case.

    A case that ran the same kernel on the same inputs should produce the same
    bytes on both architectures. Where it does not, the divergence is named
    rather than left to be inferred from two error columns that happen to
    differ.
    """
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)
    if a["profile"] != b["profile"] or a["seed"] != b["seed"]:
        print(
            "refusing to compare: the two runs used different shapes or seeds",
            file=sys.stderr,
        )
        return 2

    def key(r):
        return (r["name"], r["impl"], r["dtype"])

    ra = {key(r): r for r in a["results"]}
    rb = {key(r): r for r in b["results"]}
    label_a = a["environment"].get("gcn_arch") or a["environment"].get("device")
    label_b = b["environment"].get("gcn_arch") or b["environment"].get("device")
    print(
        f"comparing {label_a} against {label_b} | profile={a['profile']['name']} seed={a['seed']}"
    )

    diverged = 0
    for k in sorted(ra.keys() | rb.keys()):
        x, y = ra.get(k), rb.get(k)
        if x is None or y is None:
            print(f"  ONLY-IN-{'B' if x is None else 'A'} {k[0]}/{k[1]}")
            diverged += 1
            continue
        if x["status"] != y["status"]:
            print(f"  STATUS   {k[0]}/{k[1]}: {x['status']} vs {y['status']}")
            diverged += 1
        elif x["status"] == "pass" and x["digest"] != y["digest"]:
            print(
                f"  BYTES    {k[0]}/{k[1]}: {x['digest']} vs {y['digest']} "
                f"(rel {x['max_rel_err']:.3e} vs {y['max_rel_err']:.3e})"
            )
            diverged += 1
    print(f"\n{len(ra)} cases compared, {diverged} diverged")
    return 1 if diverged else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--size",
        choices=sorted(SHAPES),
        default="tiny",
        help="shape profile (default: tiny)",
    )
    ap.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="substring filter on the case name; repeatable",
    )
    ap.add_argument(
        "--impl",
        action="append",
        default=[],
        choices=["torch", "triton", "triton_dot"],
        help="restrict to one implementation; repeatable",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="device for the device arm (default: cuda). Use 'cpu' to "
        "self-check the suite against plain CPU torch.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", metavar="PATH", help="write the full report here")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.json", "B.json"),
        help="diff two reports (typically one per architecture) and exit; "
        "reports every case whose raw device output differs",
    )
    ap.add_argument(
        "--with-vllm",
        action="store_true",
        help="also cross-check vLLM's own custom ops",
    )
    args = ap.parse_args(argv)

    global DEV
    DEV = args.device

    if args.compare:
        return compare_reports(*args.compare)

    shape = SHAPES[args.size]
    cases = REGISTRY
    if args.kernel:
        cases = [c for c in cases if any(k in c.name for k in args.kernel)]
    if args.impl:
        cases = [c for c in cases if c.impl in args.impl]

    if args.list:
        for c in cases:
            print(f"{c.name:<24} {c.role:<12} {c.impl:<7} {c.note}")
        return 0

    if DEV != "cpu" and not torch.cuda.is_available():
        print(
            "no HIP device visible; run this under the emulator or on a GPU host",
            file=sys.stderr,
        )
        return 2

    info = device_info()
    info["device"] = DEV
    print(
        f"gpt-oss kernel suite | profile={shape.name} | {info.get('gcn_arch') or DEV} "
        f"| torch {info['torch']} | {len(cases)} cases",
        flush=True,
    )

    # Print each case as it finishes rather than batching at the end: at the
    # `model` profile a single case can run for minutes, and a suite that prints
    # nothing until the last one is indistinguishable from a hung one.
    results = []
    for c in cases:
        r = run_case(c, shape, args.seed)
        results.append(r)
        print(
            f"  {r.status.upper():<5} {r.name:<24} {r.impl:<10} "
            f"rel={r.max_rel_err:.3e} tol={r.tolerance:.1e} {r.seconds:6.2f}s"
            + (f"  {r.detail}" if r.detail else ""),
            flush=True,
        )

    if args.with_vllm:
        gen = torch.Generator().manual_seed(args.seed)
        extra = vllm_crosschecks(shape, gen)
        results.extend(extra)
        for r in extra:
            print(
                f"  {r.status.upper():<5} {r.name:<24} {r.impl:<10} "
                f"rel={r.max_rel_err:.3e}" + (f"  {r.detail}" if r.detail else ""),
                flush=True,
            )

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(
        f"\n{counts.get('pass', 0)} passed, {counts.get('fail', 0)} failed, "
        f"{counts.get('error', 0)} errored, {counts.get('skip', 0)} skipped"
    )

    if args.json:
        payload = {
            "schema": "rocjitsu/gpt-oss-kernels/v1",
            "profile": shape.__dict__,
            "seed": args.seed,
            "environment": info,
            "results": [r.__dict__ for r in results],
            "summary": counts,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"report written to {args.json}")

    return 0 if counts.get("fail", 0) == 0 and counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
