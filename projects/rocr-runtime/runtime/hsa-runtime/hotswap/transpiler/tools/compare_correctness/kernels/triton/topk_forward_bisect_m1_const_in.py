"""MODE=1 with deterministic input: every X[r, c] = 1.0.  The
correct tl.sum over BLOCK_N=32 columns = 32.0 for every row.
If salmon writes anything other than 32.0, the deviation tells
us EXACTLY how many elements are being included in the sum.

Narrowing lemma for the remaining m1 miscompile: different
rows showed different wrong-to-right ratios (row 0: 3.09x, row 1:
4.53x, row 2: 0.09x), which is NOT a "subset sum of the 32
elements" pattern — partial sums would have stable ratios.  With
all-ones input, the ambiguity vanishes: every row's expected sum
is exactly 32.0, and any deviation is a VALUE-independent
artifact of the reduction tree / cross-lane plumbing itself.
"""

import triton
import triton.language as tl


@triton.jit
def _topk_forward_bisect_m1_const(
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
    y_values = tl.broadcast_to(row_sum, [BLOCK_M, N_EXPTS_ACT]).to(X.dtype.element_ty)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)


RECIPES = [{
    "name": "topk_forward_bisect_m1_const_in",
    "kernel_fn":     _topk_forward_bisect_m1_const,
    "kernel_symbol": "_topk_forward_bisect_m1_const",
    "signature": {
        "X":           "*bf16",
        "stride_xm":   "i32",
        "Yv":          "*bf16",
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
        # Narrow range — all inputs round to the same bf16 value 1.0
        # (range_lo == range_hi is treated as "fill with this value"
        # per the harness's `makeInput` bf16 path).  If the harness
        # doesn't support a single-value fill, the dynamic fallback
        # is to use a tight [1.0, 1.0+tiny] range.
        {"name": "X", "dtype": "bf16", "elems": "N_ROWS * N_COLS",
         "range_lo": 1.0, "range_hi": 1.001},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "bf16", "elems": "N_ROWS * N_EXPTS_ACT_HC"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
