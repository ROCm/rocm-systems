"""`tl.sort` probe on fp32 at (BLOCK_M=32, BLOCK_N=32).

Both `canary_tl_topk_bf16` and `canary_tl_topk_fp32` are WRONG at
2048/2048 under cross-widening, ruling out the bf16 `fpval_to_key`
bit-magic as the root cause.  The bug is therefore in `tl.topk`'s
shared skeleton.  Triton's `tl.topk` implements as sort + slice,
so if this `tl.sort` probe also fails, the bug is in the sort
primitive (bitonic-style cross-lane shuffle network).  If
`tl.sort` matches while `tl.topk` fails, the bug is in the
"take first K" layer above sort.

Simple `tl.max` canaries (canary_dpp_reduce_fp32,
canary_permlanex16_rowmax_fp32) match — so atomic cross-lane
reductions are fine.  Multi-stage sort networks (DPP + LDS
compounds, often with wave-size-dependent tree heights) are
different.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32(
    X, stride_xm,
    Y, stride_ym,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m[:, None] < n_rows

    x = tl.load(X + offs_m[:, None] * stride_xm + offs_n[None, :],
                mask=mask_m, other=float("-inf"))
    # Sort each row descending.
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32",
    "kernel_fn":     _canary_tl_sort_fp32,
    "kernel_symbol": "_canary_tl_sort_fp32",
    "signature": {
        "X":           "*fp32",
        "stride_xm":   "i32",
        "Y":           "*fp32",
        "stride_ym":   "i32",
        "n_rows":      "i32",
    },
    "constexprs": {"BLOCK_M": 32, "BLOCK_N": 32},
    "harness_constants": {"N_COLS": 32},
    "scalar_args": {
        "stride_xm": "N_COLS",
        "stride_ym": "N_COLS",
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
        {"name": "X", "dtype": "fp32", "elems": "N_ROWS * N_COLS",
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        # Full-sort output is a permutation of the input; correct
        # cross-lane sort is bit-exact (no FP arithmetic, just
        # comparisons and moves).  tol=0.0 is the right comparator.
        {"name": "Y", "dtype": "fp32", "elems": "N_ROWS * N_COLS"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
