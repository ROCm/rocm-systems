"""Lane-id probe — write the workitem.id (= program_id * BLOCK_M +
arange) to output.  No cross-lane ops.  Tests whether salmon and
native agree on the lane→output-index layout for the same (BLOCK_M,
N_EXPTS_ACT) shape the other bisect kernels use.

If salmon diverges here, the bug is NOT in the cross-lane reduction
(the probe has none) but in the basic load-mask-store layout, which
would reshape the entire triage: the DPP rewrite is a no-op for a
bug that lives in `tl.load` / `tl.store` scheduling rather than in
the reduction tree.
"""

import triton
import triton.language as tl


@triton.jit
def _topk_forward_bisect_laneprobe(
    X, Yv, stride_ym,
    n_rows,
    BLOCK_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
):
    # X is a no-op input kept to exercise the `inputs` shape of the
    # bisect sibling recipes (avoids tripping a harness edge case on
    # a zero-input recipe).  Triton's dead-argument elimination keeps
    # it out of the compiled kernel body.
    _ = X
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    # Per-thread value = offs_m (= the source-wave lane's absolute
    # row index).  Broadcast to all N_EXPTS_ACT cols.  Cast to bf16
    # so the value is encodable exactly (row indices up to 512 fit
    # in bf16 losslessly).
    y_values = tl.broadcast_to(offs_m[:, None].to(tl.float32),
                                [BLOCK_M, N_EXPTS_ACT]).to(tl.bfloat16)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)


RECIPES = [{
    "name": "topk_forward_bisect_m_laneprobe",
    "kernel_fn":     _topk_forward_bisect_laneprobe,
    "kernel_symbol": "_topk_forward_bisect_laneprobe",
    "signature": {
        "X":         "*bf16",
        "Yv":        "*bf16",
        "stride_ym": "i32",
        "n_rows":    "i32",
    },
    "constexprs": {"BLOCK_M": 32, "N_EXPTS_ACT": 4},
    "harness_constants": {"N_EXPTS_ACT_HC": 4, "N_COLS": 128},
    "scalar_args": {
        "stride_ym": "N_EXPTS_ACT_HC",
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
         "range_lo": 0.0, "range_hi": 1.0},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "bf16", "elems": "N_ROWS * N_EXPTS_ACT_HC"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
