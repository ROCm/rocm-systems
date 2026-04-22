"""Minimal `tl.topk` probe on bf16 at (BLOCK_M=32, BLOCK_N=32), k=4.

Why this probe exists
=====================

`streaming_topk` (exercised by `topk_forward_bisect_m2_strict`) shows
75% of y_value slots diverging bit-exactly between native and salmon.
streaming_topk is a composition of:

  1. `tl.load` of X[r, BLOCK_N] per iteration  (verified OK by the
     _m_load_store / _m_laneprobe probes).
  2. `fpval_to_key` — pure per-lane bit manipulation on u16.
  3. `(value_key << 16) | index_key` — per-lane bit pack.
  4. `tl.topk(packed_u32, k=4, dim=1)` — first iteration.
  5. Loop of `tl.bitonic_merge(acc) + tl.topk(new_packed, k=4) +
     tl.maximum(acc, new_topk)`.
  6. `tl.sort(acc, dim=1, descending=True)`.
  7. Unpack (shift + bitcast).

This probe isolates step (4) alone on bf16: a 2D tile (BLOCK_M=32,
BLOCK_N=32) of random bf16 values, `tl.topk(x, k=4, dim=1)`, store.

Expected verdicts
=================

* salmon=match — `tl.topk` on bf16 at this shape is correct in
  isolation.  The bug is in the COMPOSITION (steps 5, 6, 7, or the
  u32 pack/unpack around sort keys).
* salmon=WRONG — `tl.topk` is broken in isolation.  That localises
  the miscompile to the Triton-lowered `tl.topk` implementation at
  this specific wave-projection shape; the next bisect would be
  `tl.sort` alone (which `tl.topk` presumably uses) vs the pure
  `tl.argmax` or similar.

Shape choices
=============

Matches production (N_ROWS=512 × N_COLS=32, k=4, num_warps=4,
BLOCK_M=32).  The 2D-tile shape is deliberate — a 1D topk would
not exercise the cross-lane reductions that the production
kernel's per-row tile-topk hits.  Input range [-4, 4] matches the
GPT-OSS MoE logit range `topk_forward_bf16` uses.

Comparator
==========

`rel-rms tol=0.02` on the values output: `tl.topk` on ties is
allowed to pick different indices under cross-widening (see the
topk_forward_bf16 open finding in learnings.md 2026-04-22), so
we tolerate the rel-rms drift of a pure-values comparison.  If
this probe drifts by MORE than that, the bug is in `tl.topk`
itself and the specific max|err| number will bound the drift class.
"""

import triton
import triton.language as tl


@triton.jit
def _canary_tl_topk_bf16(
    X, stride_xm,
    Yv, stride_ym,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, K)
    mask_m = offs_m[:, None] < n_rows
    offs_n = tl.arange(0, BLOCK_N)[None, :]

    x = tl.load(X + offs_m[:, None] * stride_xm + offs_n,
                mask=mask_m, other=float("-inf"))
    y = tl.topk(x, K, dim=1)

    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_k[None, :]
    tl.store(Yv_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_topk_bf16",
    "kernel_fn":     _canary_tl_topk_bf16,
    "kernel_symbol": "_canary_tl_topk_bf16",
    "signature": {
        "X":           "*bf16",
        "stride_xm":   "i32",
        "Yv":          "*bf16",
        "stride_ym":   "i32",
        "n_rows":      "i32",
    },
    "constexprs": {"BLOCK_M": 32, "BLOCK_N": 32, "K": 4},
    "harness_constants": {"N_COLS": 32, "K_HC": 4},
    "scalar_args": {
        "stride_xm": "N_COLS",
        "stride_ym": "K_HC",
        "n_rows":    "N_ROWS",
    },
    "num_warps": 4,
    "shape_dim": "N_ROWS",
    "default_shapes": [512],
    "grid": {
        "x": "ceil_div(N_ROWS, BLOCK_M)",
        "y": "1",
        "z": "1",
    },
    "inputs": [
        {"name": "X", "dtype": "bf16", "elems": "N_ROWS * N_COLS",
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "bf16", "elems": "N_ROWS * K_HC"},
    ],
    # rel-rms(0.02): tl.topk is allowed to pick a different
    # tie-break order under cross-widening (the top-k VALUES
    # should match; the INDICES may flip on near-ties).  Same
    # tolerance scale as MODE=1 of topk_forward_bisect (pure
    # bf16 reduction drift); anything bigger than that is
    # a structural bug we want to catch.
    "comparator": {"kind": "rel-rms", "tol": 0.02},
}]
