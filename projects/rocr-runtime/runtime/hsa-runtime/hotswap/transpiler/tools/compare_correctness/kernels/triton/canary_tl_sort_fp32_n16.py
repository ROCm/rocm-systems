"""`tl.sort` probe on fp32 at (BLOCK_M=32, BLOCK_N=16), all in one
16-lane row.

Companion to `canary_tl_sort_fp32`.  That probe at BLOCK_N=32 shows
salmon producing two SORTED HALVES (cols 0..15 desc, cols 16..31
desc) — the final bitonic merge step that crosses the 16-lane
boundary is broken under cross-widening.

Bitonic sort of N=16 elements needs only log2(16)=4 stages, the
largest of which has distance 8.  That stays within a single
16-lane DPP row, so it does NOT require the cross-16 swap
(permlane16 / cross-row DPP / wave-level swap) that N=32 does.

Expected verdicts
=================

* salmon=match — confirms the bug is isolated to the cross-16
  merge step (permlane16_swap emulation or cross-16 DPP).  The
  four smaller stages work correctly in isolation.
* salmon=WRONG — bug is in a smaller stage too; narrowing
  continues.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32_n16(
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
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32_n16",
    "kernel_fn":     _canary_tl_sort_fp32_n16,
    "kernel_symbol": "_canary_tl_sort_fp32_n16",
    "signature": {
        "X":           "*fp32",
        "stride_xm":   "i32",
        "Y":           "*fp32",
        "stride_ym":   "i32",
        "n_rows":      "i32",
    },
    "constexprs": {"BLOCK_M": 32, "BLOCK_N": 16},
    "harness_constants": {"N_COLS": 16},
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
        {"name": "Y", "dtype": "fp32", "elems": "N_ROWS * N_COLS"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
