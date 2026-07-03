#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small PyTorch operator repro for the Qwen SDPA failure.

This avoids running the full model while keeping the same attention operator
shape that failed under rocjitsu.  The default shape is Qwen2.5-0.5B layer-0
prefill: q/k/v as (batch=1, heads=14, seq=5, head_dim=64), bf16, causal SDPA.
The alternate modes intentionally exercise the decomposed math so we can tell
whether a failure belongs to PyTorch SDPA/AOTriton or to a plain matmul.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Inputs:
    q_cpu: torch.Tensor
    k_cpu: torch.Tensor
    v_cpu: torch.Tensor
    p_cpu: torch.Tensor
    attn_mask_cpu: torch.Tensor | None
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    attn_mask: torch.Tensor | None


def parse_dtype(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def make_inputs(args: argparse.Namespace) -> Inputs:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)

    q_cpu = torch.randn(
        (args.batch, args.heads, args.seq, args.head_dim),
        generator=gen,
        dtype=torch.float32,
    )
    k_cpu = torch.randn(
        (args.batch, args.kv_heads, args.kv_seq, args.head_dim),
        generator=gen,
        dtype=torch.float32,
    )
    v_cpu = torch.randn(
        (args.batch, args.kv_heads, args.kv_seq, args.head_dim),
        generator=gen,
        dtype=torch.float32,
    )
    p_cpu = torch.rand(
        (args.batch, args.heads, args.seq, args.kv_seq),
        generator=gen,
        dtype=torch.float32,
    )
    if args.causal:
        mask = causal_mask(args.seq, args.kv_seq, device=p_cpu.device)
        p_cpu = p_cpu.masked_fill(~mask, 0.0)
    p_cpu = p_cpu / p_cpu.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)

    attn_mask_cpu = make_attention_mask(
        args, dtype=torch.float32, device=torch.device("cpu")
    )
    q = q_cpu.to(device="cuda", dtype=args.dtype)
    k = k_cpu.to(device="cuda", dtype=args.dtype)
    v = v_cpu.to(device="cuda", dtype=args.dtype)
    p = p_cpu.to(device="cuda", dtype=args.dtype)
    attn_mask = (
        attn_mask_cpu.to(device="cuda", dtype=args.dtype)
        if attn_mask_cpu is not None
        else None
    )
    return Inputs(
        q_cpu=q_cpu,
        k_cpu=k_cpu,
        v_cpu=v_cpu,
        p_cpu=p_cpu,
        attn_mask_cpu=attn_mask_cpu,
        q=q,
        k=k,
        v=v,
        p=p,
        attn_mask=attn_mask,
    )


def expand_gqa(
    k: torch.Tensor, v: torch.Tensor, heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if k.shape[1] == heads:
        return k, v
    if heads % k.shape[1] != 0:
        raise ValueError(f"heads={heads} is not divisible by kv_heads={k.shape[1]}")
    repeats = heads // k.shape[1]
    return k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)


def causal_mask(q_len: int, kv_len: int, *, device: torch.device) -> torch.Tensor:
    # Match PyTorch SDPA's upper-left causal bias for non-square masks.
    return torch.ones((q_len, kv_len), dtype=torch.bool, device=device).tril()


def make_attention_mask(
    args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Create the explicit SDPA mask variants used by Qwen diagnostics.

    Transformers' cached Qwen path passes an explicit additive mask even when a
    one-token decode step can attend to every cached key.  That mask disables
    the `enable_gqa` SDPA path and can select different backend kernels, so the
    repro needs to distinguish "no mask" from an all-zero additive mask.
    """

    shape = (args.batch, 1, args.seq, args.kv_seq)
    if args.mask_mode == "none":
        return None
    if args.mask_mode == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if args.mask_mode == "causal-bias":
        mask = torch.full(shape, float("-inf"), dtype=dtype, device=device)
        allowed = causal_mask(args.seq, args.kv_seq, device=device)
        return mask.masked_fill(allowed.view(1, 1, args.seq, args.kv_seq), 0.0)
    raise AssertionError(f"unhandled mask mode: {args.mask_mode}")


def attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float | None,
    causal: bool,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    actual_scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * actual_scale
    if attn_mask is not None:
        scores = scores + attn_mask
    if causal:
        mask = causal_mask(q.shape[-2], k.shape[-2], device=scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores


def run_operator(args: argparse.Namespace, inputs: Inputs) -> torch.Tensor:
    q, k, v = inputs.q, inputs.k, inputs.v

    if args.mode == "sdpa":
        kwargs = {
            "dropout_p": 0.0,
            "is_causal": args.causal,
        }
        if inputs.attn_mask is not None:
            kwargs["attn_mask"] = inputs.attn_mask
        if args.scale is not None:
            kwargs["scale"] = args.scale
        if args.enable_gqa:
            kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    if args.enable_gqa:
        k, v = expand_gqa(k, v, q.shape[1])

    if args.mode == "qk":
        return torch.matmul(q, k.transpose(-2, -1))

    if args.mode == "pv":
        return torch.matmul(inputs.p, v)

    scores = attention_scores(
        q, k, scale=args.scale, causal=args.causal, attn_mask=inputs.attn_mask
    )
    probs = torch.softmax(scores, dim=-1)
    if args.mode == "decomposed":
        return torch.matmul(probs, v)
    raise AssertionError(f"unhandled mode: {args.mode}")


def cpu_reference(args: argparse.Namespace, inputs: Inputs) -> torch.Tensor:
    q, k, v = inputs.q_cpu, inputs.k_cpu, inputs.v_cpu
    if args.enable_gqa:
        k, v = expand_gqa(k, v, q.shape[1])

    if args.mode == "qk":
        return torch.matmul(q, k.transpose(-2, -1))

    if args.mode == "pv":
        return torch.matmul(inputs.p_cpu, v)

    scores = attention_scores(
        q,
        k,
        scale=args.scale,
        causal=args.causal,
        attn_mask=inputs.attn_mask_cpu,
    )
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def print_tensor_summary(label: str, tensor: torch.Tensor) -> None:
    host = tensor.detach().float().cpu()
    flat = host.flatten()
    print(f"{label}_shape", tuple(tensor.shape))
    print(f"{label}_dtype", str(tensor.dtype))
    print(f"{label}_finite", bool(torch.isfinite(host).all().item()))
    print(f"{label}_sum", float(host.sum().item()))
    print(f"{label}_min", float(host.min().item()))
    print(f"{label}_max", float(host.max().item()))
    print(f"{label}_first8", flat[:8].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("sdpa", "qk", "pv", "decomposed"), default="sdpa"
    )
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=14)
    parser.add_argument("--seq", type=int, default=5)
    parser.add_argument("--kv-seq", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--enable-gqa", action="store_true")
    parser.add_argument(
        "--mask-mode",
        choices=("none", "zeros", "causal-bias"),
        default="none",
        help="Explicit additive SDPA attention mask to pass via attn_mask.",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--causal", dest="causal", action="store_true")
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.set_defaults(causal=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")

    print("torch", torch.__version__, "hip", torch.version.hip)
    print("device", torch.cuda.get_device_name(0))
    print(
        "config",
        {
            "mode": args.mode,
            "dtype": str(args.dtype),
            "batch": args.batch,
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "seq": args.seq,
            "kv_seq": args.kv_seq,
            "head_dim": args.head_dim,
            "causal": args.causal,
            "enable_gqa": args.enable_gqa,
            "mask_mode": args.mask_mode,
        },
    )

    inputs = make_inputs(args)
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = run_operator(args, inputs)
            torch.cuda.synchronize()

        start = time.time()
        out = None
        for _ in range(args.iters):
            out = run_operator(args, inputs)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    assert out is not None
    print("elapsed_sec", round(elapsed, 6))
    print("per_iter_sec", round(elapsed / args.iters, 6))
    print_tensor_summary("out", out)

    if args.check:
        ref = cpu_reference(args, inputs)
        diff = (out.detach().float().cpu() - ref.float()).abs()
        print("max_abs_diff", float(diff.max().item()))
        print("mean_abs_diff", float(diff.mean().item()))


if __name__ == "__main__":
    main()
