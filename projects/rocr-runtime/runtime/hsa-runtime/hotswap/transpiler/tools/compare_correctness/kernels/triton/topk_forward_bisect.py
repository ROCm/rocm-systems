"""Bisection probe — compile-time MODE knob around `_topk_forward`.

What it targets
===============

The silent miscompile on `topk_forward_bf16` surfaced by the sibling
recipe.  Root cause unknown; the single-primitive canaries for DPP /
permlane / ds_swizzle / ds_bpermute all match in isolation, so the
bug is composition-level.  This recipe narrows which COMPOSITION
triggers it via a `tl.constexpr` MODE knob that bisects the kernel
path top-down:

  MODE=0 — write zeros and return.  If WRONG: the bug is in
           basic store / grid math / kernarg plumbing.
  MODE=1 — load X row sums, store to Yv[:, 0].  If WRONG:
           basic load + masked store at this shape is wrong.
  MODE=2 — call `streaming_topk`, store raw top-k values (no
           softmax).  If WRONG while MODE=1 matches: the bug is
           in streaming_topk's composition (DPP + LDS-based
           reductions + sort).
  MODE=3 — streaming_topk + softmax, store softmax'd top-k (no
           sort, no bitmatrix).  If WRONG while MODE=2 matches:
           softmax is the culprit (v_div_scale_f32 chain + DPP
           reductions).
  MODE=4 — full _topk_forward (sort + bitmatrix).  If WRONG
           while MODE=3 matches: the tl.sort or bitmatrix
           construction is the culprit.

Whichever mode first diverges pins the obstruction class.  The
remaining modes' verdicts refine the signal: if MODE=2 is WRONG
but MODE=3 "accidentally matches", that's informative too (e.g.
softmax masks the underlying streaming_topk bug).

All other recipe parameters match `topk_forward_bf16`:
  * N_ROWS = 512, N_COLS = 128, N_EXPTS_ACT = 4
  * BLOCK_M = BLOCK_N = 32, num_warps = 4
"""

import triton
import triton.language as tl
from triton_kernels.topk_details._topk_forward \
    import streaming_topk


@triton.jit
def _topk_forward_bisect(
    X, stride_xm,
    Yv, stride_ym,
    n_rows, n_expts_tot,
    BLOCK_M: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows

    if MODE == 0:
        # Baseline: write zeros.  Exercises kernarg plumbing +
        # grid math + masked store shape; nothing else.
        y_values = tl.zeros([BLOCK_M, N_EXPTS_ACT], dtype=tl.float32).to(X.dtype.element_ty)
    elif MODE == 1:
        # Load row sum; broadcast to all k columns.  Exercises
        # basic global load + reduction-over-columns + masked
        # scatter, no topk / softmax / sort.
        x = tl.load(X + offs_m[:, None] * stride_xm + tl.arange(0, BLOCK_N)[None, :],
                    mask=(mask_m & (tl.arange(0, BLOCK_N)[None, :] < n_expts_tot)),
                    other=0.0)
        row_sum = tl.sum(x, axis=1, keep_dims=True)
        y_values = tl.broadcast_to(row_sum, [BLOCK_M, N_EXPTS_ACT]).to(X.dtype.element_ty)
    elif MODE == 2:
        # streaming_topk, no softmax.  Exercises DPP + LDS
        # reductions + bitonic_merge + tl.topk + tl.sort.
        y_values, _y_indices = streaming_topk(
            X, stride_xm, n_expts_tot, offs_m, mask_m,
            N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)
    elif MODE == 3:
        # streaming_topk + softmax.  Adds v_div_scale_f32 chain.
        y_values, _y_indices = streaming_topk(
            X, stride_xm, n_expts_tot, offs_m, mask_m,
            N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)
        y_values = tl.softmax(y_values.to(tl.float32), dim=1,
                              keep_dims=True).to(X.dtype.element_ty)
    else:
        # MODE=4: streaming_topk + softmax + sort.  Full pipeline
        # minus bitmatrix (which writes a different output buffer,
        # irrelevant to Yv correctness bisection).
        y_values, _y_indices = streaming_topk(
            X, stride_xm, n_expts_tot, offs_m, mask_m,
            N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)
        y_values = tl.softmax(y_values.to(tl.float32), dim=1,
                              keep_dims=True).to(X.dtype.element_ty)
        y_values = tl.sort(y_values, dim=1, descending=True)

    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)


# Shape constants match topk_forward_bf16.
_N_ROWS      = 512
_N_COLS      = 128
_N_EXPTS_ACT = 4
_N_EXPTS_PAD = _N_COLS
_BLOCK_M     = 32
_BLOCK_N     = 32


def _recipe(name: str, mode: int) -> dict:
    return {
        "name": name,
        "kernel_fn":     _topk_forward_bisect,
        "kernel_symbol": "_topk_forward_bisect",
        "signature": {
            "X":           "*bf16",
            "stride_xm":   "i32",
            "Yv":          "*bf16",
            "stride_ym":   "i32",
            "n_rows":      "i32",
            "n_expts_tot": "i32",
        },
        "constexprs": {
            "BLOCK_M":     _BLOCK_M,
            "N_EXPTS_PAD": _N_EXPTS_PAD,
            "N_EXPTS_ACT": _N_EXPTS_ACT,
            "BLOCK_N":     _BLOCK_N,
            "MODE":        mode,
        },
        "harness_constants": {
            "N_COLS":      _N_COLS,
            "N_EXPTS_ACT_HC": _N_EXPTS_ACT,
        },
        "scalar_args": {
            "stride_xm":   "N_COLS",
            "stride_ym":   "N_EXPTS_ACT_HC",
            "n_rows":      "N_ROWS",
            "n_expts_tot": "N_COLS",
        },
        "num_warps": 4,
        "shape_dim": "N_ROWS",
        "default_shapes": [_N_ROWS],
        "grid": {
            "x": "ceil_div(N_ROWS, BLOCK_M)",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "X", "dtype": "bf16",
             "elems": "N_ROWS * N_COLS",
             "range_lo": -4.0, "range_hi": 4.0},
        ],
        "outputs": [
            {"name": "Yv", "dtype": "bf16",
             "elems": "N_ROWS * N_EXPTS_ACT_HC"},
        ],
        "comparator": {"kind": "abs", "tol": 0.0},
    }


# Phase-1 AOT supports one recipe per file.  This file is imported
# by the make rule with the file stem = recipe name, so emit only
# the full-pipeline MODE=4 recipe here; the siblings
# (topk_forward_bisect_m0 / m1 / m2 / m3) live in their own files.
RECIPES = [_recipe("topk_forward_bisect", 4)]
