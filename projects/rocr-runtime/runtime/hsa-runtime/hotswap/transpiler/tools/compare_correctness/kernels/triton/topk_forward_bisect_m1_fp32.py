"""MODE=1 variant with fp32 input AND fp32 output — no bf16
anywhere.  Isolates whether the remaining MODE=1..4 miscompile
(post d16_hi-store fix) is in the PER-ROW axis=1 reduction
(which fp32 canaries DO exercise and they all match) or in
some bf16-specific path the other probes still touch.

If matches: the remaining bug is bf16-cast-specific — probably
in how the pre-store RNE-biased-sum → bf16 half-extraction is
emitted when `tl.sum(axis=1)` feeds a `.to(tl.bfloat16)` cast
(different lift path from the laneprobe case we fixed, since
this one goes through LDS rather than global_store_d16_hi_b16).

If WRONG: the bug is in the 2D per-row cross-lane reduction
itself at num_warps=4 — a different class from the canary grid
(which is all num_warps=1 1D reductions).
"""

import triton
import triton.language as tl


@triton.jit
def _topk_forward_bisect_m1_fp32(
    X, stride_xm,
    Yv, stride_ym,
    n_rows, n_expts_tot,
    BLOCK_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    x = tl.load(X + offs_m[:, None] * stride_xm + tl.arange(0, BLOCK_N)[None, :],
                mask=(mask_m & (tl.arange(0, BLOCK_N)[None, :] < n_expts_tot)),
                other=0.0)
    row_sum = tl.sum(x, axis=1, keep_dims=True)
    # Stay in fp32 everywhere — no bf16 cast.
    y_values = tl.broadcast_to(row_sum, [BLOCK_M, N_EXPTS_ACT])
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)


RECIPES = [{
    "name": "topk_forward_bisect_m1_fp32",
    "kernel_fn":     _topk_forward_bisect_m1_fp32,
    "kernel_symbol": "_topk_forward_bisect_m1_fp32",
    "signature": {
        "X":           "*fp32",
        "stride_xm":   "i32",
        "Yv":          "*fp32",
        "stride_ym":   "i32",
        "n_rows":      "i32",
        "n_expts_tot": "i32",
    },
    "constexprs": {"BLOCK_M": 32, "N_EXPTS_ACT": 4, "BLOCK_N": 32},
    "harness_constants": {"N_COLS": 128, "N_EXPTS_ACT_HC": 4},
    "scalar_args": {
        "stride_xm":   "N_COLS",
        "stride_ym":   "N_EXPTS_ACT_HC",
        "n_rows":      "N_ROWS",
        "n_expts_tot": "N_COLS",
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
        {"name": "X", "dtype": "fp32", "elems": "N_ROWS * N_COLS",
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "fp32", "elems": "N_ROWS * N_EXPTS_ACT_HC"},
    ],
    # Rel-rms comparator at 1e-5.  Motivation:
    #
    # With the cross-lane-divergent rewrite in place (update.dpp ->
    # ds_bpermute + select, ...), the per-row reduction tree shape
    # differs between native wave32 and salmon-wave64 lift (wave64
    # folds 32 lanes in one tree, wave32 folds 32 lanes across 2
    # waves and re-combines; the final sum traverses a different
    # associativity).  fp32 adds are NOT associative, so ULP-scale
    # drift is expected.  Empirical measurement on this recipe:
    #
    #   rms(gold) = 12.68  (RMS of per-slot fp32 sum output)
    #   rms(diff) = 1.0e-6 (~1 fp32 ULP at magnitude 7)
    #   rel-rms   = rms(diff) / rms(gold) = 8.7e-8
    #   max|err|  = 3.8e-6 (4x single-ULP, at the worst slot)
    #
    # rel-rms 8.7e-8 is the NORMALISED buffer-level metric; the
    # ~1-ULP absolute scale maps to 1.0e-6 / rms(gold) = ~1e-7.
    # The two numbers are both in the same small-drift regime but
    # describe different things, don't conflate them.
    #
    # Why tol=1e-5 and not 1e-7: gives ~115x safety margin on the
    # observed 8.7e-8 rel-rms drift, which covers variance in the
    # RNG-seeded input and across-seed shape of the error envelope
    # without masking a real structural bug.  A systematic fp32-
    # reduction miscompile (wrong accumulator width, dropped sum
    # lane, etc.) would drift by a significant fraction of
    # rms(gold) — several orders of magnitude above 1e-5.
    #
    # Caveat on seed-dependence: tolerance is calibrated for a
    # single (recipe, shape, seed) triple.  The harness seeds from
    # std::hash<std::string> which is libstdc++-version-specific;
    # if inputs shift under a libstdc++ change, drift may shift
    # proportionally.  The 115x margin is deliberately generous
    # for this reason.  See `hotswap/docs/learnings.md`
    # 2026-04-22 entry on principled tolerances for the full
    # argument.
    "comparator": {"kind": "rel-rms", "tol": 1e-5},
}]
