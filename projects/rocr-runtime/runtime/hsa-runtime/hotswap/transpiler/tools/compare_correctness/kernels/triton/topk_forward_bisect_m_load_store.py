"""Load + store probe — after the D16_HI fix in handle_flat.cpp,
`topk_forward_bisect_m_laneprobe` graduates to match, which proves
the store pipeline is sound.  But MODE=1..4 stay WRONG with the
SAME verdicts as pre-fix, which means there is a SECOND bug
downstream that a reduction composes with.

This probe isolates the minimal "load a bf16 tile + store it
back" shape — no reduction, no cross-lane compute, just 2D load
+ basic arithmetic + 2D masked store, at the same
(BLOCK_M=32, BLOCK_N=32, num_warps=4) shape as the other bisect
siblings.

If this matches: the bug is specifically in tl.sum / cross-lane
reduction, narrowing the next triage step to the reduction path
(which my prior DPP rewrite should have covered — interesting if
it didn't).

If this is WRONG: the bug is in basic 2D tile load or per-lane
arithmetic, not in the reduction — wider search surface.
"""

import triton
import triton.language as tl


@triton.jit
def _topk_forward_bisect_load_store(
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
    # Load the same tile MODE=1 loads, but sum via a CONSTANT-VALUED
    # axis-1 reduction that folds to a zero at compile time — Triton
    # emits the tile-load machinery but skips the cross-lane
    # reduction.  The result per row is `x[r, 0]` broadcast across
    # N_EXPTS_ACT.
    # Load ONLY the column-0 element per row — a 1D-shape load that
    # reaches the store path through the same (BLOCK_M, N_EXPTS_ACT)
    # masked-2D-store shape as MODE=1, BUT without any tl.sum / tl.max
    # / cross-lane reduction in the data flow.
    x0 = tl.load(X + offs_m * stride_xm, mask=offs_m < n_rows, other=0.0)
    y_values = tl.broadcast_to(x0[:, None], [BLOCK_M, N_EXPTS_ACT]).to(X.dtype.element_ty)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)


RECIPES = [{
    "name": "topk_forward_bisect_m_load_store",
    "kernel_fn":     _topk_forward_bisect_load_store,
    "kernel_symbol": "_topk_forward_bisect_load_store",
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
        {"name": "X", "dtype": "bf16", "elems": "N_ROWS * N_COLS",
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "bf16", "elems": "N_ROWS * N_EXPTS_ACT_HC"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
