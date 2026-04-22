"""`tl.topk` probe on fp32 at (BLOCK_M=32, BLOCK_N=32), k=4.

Companion to `canary_tl_topk_bf16`.  If bf16 WRONG + fp32 match,
the bug is in the `fpval_to_key` u16 bit-magic OR in how Triton
lowers `tl.topk` specifically when the key is 16-bit (packed
differently than 32-bit fp32 keys).  If both WRONG, the bug is
in the tl.topk cross-lane reduction skeleton independent of
the key width.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_topk_fp32(
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
    "name": "canary_tl_topk_fp32",
    "kernel_fn":     _canary_tl_topk_fp32,
    "kernel_symbol": "_canary_tl_topk_fp32",
    "signature": {
        "X":           "*fp32",
        "stride_xm":   "i32",
        "Yv":          "*fp32",
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
        {"name": "X", "dtype": "fp32", "elems": "N_ROWS * N_COLS",
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        {"name": "Yv", "dtype": "fp32", "elems": "N_ROWS * K_HC"},
    ],
    # fp32 drift from tl.topk cross-lane reordering would be
    # bounded by fp32 ULP at magnitude ~4 (~5e-7), so rel-rms 1e-5
    # has ~20x margin.  Anything bigger = structural bug.
    "comparator": {"kind": "rel-rms", "tol": 1e-5},
}]
