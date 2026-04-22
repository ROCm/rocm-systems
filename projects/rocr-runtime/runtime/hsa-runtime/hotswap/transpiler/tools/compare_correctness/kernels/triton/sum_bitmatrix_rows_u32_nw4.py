"""Triage probe: same as `sum_bitmatrix_rows_u32` but with
`num_warps=4` instead of 8.  Exists to isolate whether the salmon
crash on the `num_warps=8` variant is wave-count-projection-related
(num_warps=8 at wave32 source -> 8 wave32 groups ; at wave64 target
-> wg=256 = 4 wave64 groups, which collapses 2 source waves into 1
target wave) or something else.  If this variant matches while the
num_warps=8 variant EXIT=2s, the crash is specifically about the
source-warp-count / target-wave-count projection mismatch, and
source kernels with num_warps <= wave64_size / wave32_size = 2 are
a safe corner; source kernels with num_warps >= 4 are not.

Not a production recipe — candidate for removal once the triage
conclusion is documented.  See `sum_bitmatrix_rows_u32.py` for the
full rationale.
"""

import triton
import triton.language as tl
from triton_kernels.tensor_details.bitmatrix_details.sum_bitmatrix_rows \
    import _sum_bitmatrix_rows


_PARTIAL_BLOCK_M = 32
_TILE_SIZE       = 4
_BLOCK_MM        = _PARTIAL_BLOCK_M * _TILE_SIZE   # 128
_BLOCK_M         = _PARTIAL_BLOCK_M                # 32
_N_COLS_U32      = 1


RECIPES = [
    {
        "name": "sum_bitmatrix_rows_u32_nw4",
        "kernel_fn":     _sum_bitmatrix_rows,
        "kernel_symbol": "_sum_bitmatrix_rows",
        "signature": {
            "B":            "*u32",
            "shape_bm":     "i32",
            "Out":          "*i32",
            "OutPartials":  "*i32",
            "stride_pn":    "i32",
            "shape_pn":     "i32",
        },
        "constexprs": {
            "stride_bm":   _N_COLS_U32,
            "stride_bn":   1,
            "stride_pm":   1,
            "BLOCK_MM":    _BLOCK_MM,
            "BLOCK_M":     _BLOCK_M,
        },
        "harness_constants": {
            "N_COLS_U32": _N_COLS_U32,
            "TILE_SIZE":  _TILE_SIZE,
        },
        "scalar_args": {
            "shape_bm":  "N_ROWS",
            "stride_pn": "ceil_div(N_ROWS, BLOCK_MM) * TILE_SIZE",
            "shape_pn":  "N_COLS_U32 * 32",
        },
        "num_warps": 4,  # triage: differs only from sibling recipe in this field
        "shape_dim": "N_ROWS",
        "default_shapes": [128, 512, 2048, 8192],
        "grid": {
            "x": "ceil_div(N_ROWS, BLOCK_MM)",
            "y": "N_COLS_U32",
            "z": "1",
        },
        "inputs": [
            {"name": "B", "dtype": "u32",
             "elems": "N_ROWS * N_COLS_U32"},
        ],
        "outputs": [
            {"name": "Out", "dtype": "i32",
             "elems": "N_COLS_U32 * 32",
             "init": "zero"},
            {"name": "OutPartials", "dtype": "i32",
             "elems": "ceil_div(N_ROWS, BLOCK_MM) * TILE_SIZE * "
                      "N_COLS_U32 * 32"},
        ],
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
