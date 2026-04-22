"""BF16 load → FP32 upconvert → FP32 store probe.

Narrows the remaining MODE=1..4 miscompile (post d16_hi store fix).
Previous probes:
  * topk_forward_bisect_m_laneprobe   — match (store pipeline OK).
  * topk_forward_bisect_m_load_store  — match (basic tile load + bf16
                                        direct store OK).
  * topk_forward_bisect_m1_fp32       — match within ULP drift
                                        (reduction at num_warps=4 OK).

Remaining suspect: the bf16 INPUT PATH — `tl.load(*bf16)` followed by
an implicit fp32 upconversion inside tl.sum.  The source gfx1250
compiles this as `global_load_short_d16_hi` + `v_lshlrev_b32 v, 16, v`
(low-16 bf16 into upper-16 fp32 bit pattern), then reduces in fp32.

If salmon miscompiles the upconversion (e.g. a global_load_short_d16
/ d16_hi half-selector bug symmetric with the d16_hi store bug we
just fixed, or an `lshlrev 16` handling gap), this probe surfaces
it.

This kernel: loads column 0 of each row as bf16, upconverts to fp32,
stores as fp32 — no reduction, no cross-lane, just load + upconvert
+ store.  Any non-zero mismatch here is the bf16 upconvert bug.
"""

import triton
import triton.language as tl


@triton.jit
def _topk_forward_bisect_bf16_upconv(
    X, stride_xm,
    Yv, stride_ym,
    n_rows,
    BLOCK_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    # Load bf16 column 0, upconvert to fp32, broadcast.
    x_bf16 = tl.load(X + offs_m * stride_xm, mask=offs_m < n_rows, other=0.0)
    x_fp32 = x_bf16.to(tl.float32)
    y_values = tl.broadcast_to(x_fp32[:, None], [BLOCK_M, N_EXPTS_ACT])
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)


RECIPES = [{
    "name": "topk_forward_bisect_m_bf16_upconv",
    "kernel_fn":     _topk_forward_bisect_bf16_upconv,
    "kernel_symbol": "_topk_forward_bisect_bf16_upconv",
    "signature": {
        "X":         "*bf16",
        "stride_xm": "i32",
        "Yv":        "*fp32",
        "stride_ym": "i32",
        "n_rows":    "i32",
    },
    "constexprs": {"BLOCK_M": 32, "N_EXPTS_ACT": 4},
    "harness_constants": {"N_EXPTS_ACT_HC": 4, "N_COLS": 128},
    "scalar_args": {
        "stride_xm": "N_COLS",
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
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "fp32", "elems": "N_ROWS * N_EXPTS_ACT_HC"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
