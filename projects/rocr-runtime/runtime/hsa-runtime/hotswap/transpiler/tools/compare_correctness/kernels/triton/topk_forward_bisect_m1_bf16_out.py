"""fp32 input → tl.sum → bf16 output.  Companion to m1_fp32
(both-fp32) to bisect: if this MATCHES, the bug is strictly in
the bf16 INPUT path (load + upconvert inside `tl.sum`).  If this
is WRONG, the bug is in the bf16 OUTPUT cast from a reduction
result (distinct from the `global_store_d16_hi_b16` path the
laneprobe probe already covered, since that one matches).
"""

import triton
import triton.language as tl


@triton.jit
def _topk_forward_bisect_m1_bf16_out(
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
    # fp32 input, bf16 output.
    y_values = tl.broadcast_to(row_sum, [BLOCK_M, N_EXPTS_ACT]).to(tl.bfloat16)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)


RECIPES = [{
    "name": "topk_forward_bisect_m1_bf16_out",
    "kernel_fn":     _topk_forward_bisect_m1_bf16_out,
    "kernel_symbol": "_topk_forward_bisect_m1_bf16_out",
    "signature": {
        "X":           "*fp32",
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
        {"name": "X", "dtype": "fp32", "elems": "N_ROWS * N_COLS",
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "bf16", "elems": "N_ROWS * N_EXPTS_ACT_HC"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
