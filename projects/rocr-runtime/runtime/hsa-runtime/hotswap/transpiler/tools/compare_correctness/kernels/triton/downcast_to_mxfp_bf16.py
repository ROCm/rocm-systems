"""Direct recipe for `_downcast_to_mxfp` at bf16 src — the non-fp32
branch of the MoE quantization path.

Why a second variant
====================

`downcast_to_mxfp_fp32` (sibling recipe) runs the fp32-src branch of
``_downcast_to_mxfp`` (which per ``NUM_WARPS = 4 if x.dtype ==
torch.float32 else 8`` in ``numerics_details.mxfp.downcast_to_mxfp``
uses ``num_warps=4``).  Today that configuration EXIT=2s on salmon
with a raise-time refusal on ``v_mad_nc_i64_i32 [VOP3]`` — the
handler doesn't cover that opcode yet.  This variant runs the bf16-
src branch (``num_warps=8``, wg=256 on gfx1250 / wg=512 on gfx942)
to discriminate whether:

* the refusal is specific to the fp32 src path's int64-MAD use (bf16
  src should take a different codegen path without 64-bit MADs), or
* the refusal is fundamental to the kernel at any src dtype.

Expected outcomes
=================

Either verdict is informative:

* ``salmon=match`` here while the fp32 sibling ``EXIT=2``s pins the
  handler gap to fp32-src-only and suggests that GPT-OSS's real
  deployment (which uses bf16 src per the captured blob
  ``_downcast_to_mxfp_124b6cd2a33b.hsaco``'s instruction profile —
  heavy ``v_and_b32_e32`` / ``v_bfe_u32`` / ``v_bitop3_b32``, zero
  ``v_fma_f32``) is actually on the green side today.
* ``salmon=EXIT=2`` again with the same ``v_mad_nc_i64_i32``
  diagnostic pins it as a shared handler gap across both src dtypes
  and argues for prioritising that opcode handler.
* ``salmon=WRONG`` would be the worst case — silent miscompile on
  bf16 src.  Triggers immediate investigation.
"""

import triton
import triton.language as tl
from triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp \
    import _downcast_to_mxfp


_BLOCK_OUT_DIM       = 32
_MICROBLOCK_SIZE     = 32
_BLOCK_QUANT_DIM     = _MICROBLOCK_SIZE * 4
_QUANT_DIM           = _BLOCK_QUANT_DIM
_DEQUANT_ROUNDING    = 0


RECIPES = [
    {
        "name": "downcast_to_mxfp_bf16",
        "kernel_fn":     _downcast_to_mxfp,
        "kernel_symbol": "_downcast_to_mxfp",
        "signature": {
            "mx_tensor_ptr":          "*u8",
            "stride_mxt_outer":       "i32",
            "mx_scale_ptr":           "*u8",
            "stride_mx_scale_outer":  "i32",
            "stride_mx_scale_quant":  "i32",
            "src_ptr":                "*bf16",
            "stride_src_outer":       "i32",
            "stride_src_quant":       "i32",
            "outer_dim":              "i32",
            "quant_dim":              "i32",
        },
        "constexprs": {
            "stride_mxt_quant":            1,
            "BLOCK_SIZE_OUT_DIM":          _BLOCK_OUT_DIM,
            "BLOCK_SIZE_QUANT_DIM":        _BLOCK_QUANT_DIM,
            "MICROBLOCK_SIZE":             _MICROBLOCK_SIZE,
            "DEQUANT_SCALE_ROUNDING_MODE": _DEQUANT_ROUNDING,
        },
        "harness_constants": {
            "QUANT_DIM":       _QUANT_DIM,
            "SCALE_PER_ROW":   _QUANT_DIM // _MICROBLOCK_SIZE,
            "FP4_PER_ROW":     _QUANT_DIM // 2,
        },
        "scalar_args": {
            "stride_mxt_outer":      "FP4_PER_ROW",
            "stride_mx_scale_outer": "SCALE_PER_ROW",
            "stride_mx_scale_quant": "1",
            "stride_src_outer":      "QUANT_DIM",
            "stride_src_quant":      "1",
            "outer_dim":             "N_ROWS",
            "quant_dim":             "QUANT_DIM",
        },
        # bf16 src -> num_warps=8 per the driver's NUM_WARPS line.
        "num_warps": 8,
        "shape_dim": "N_ROWS",
        "default_shapes": [32, 128, 512, 2048],
        "grid": {
            "x": "ceil_div(N_ROWS, BLOCK_SIZE_OUT_DIM)",
            "y": "ceil_div(QUANT_DIM, BLOCK_SIZE_QUANT_DIM)",
            "z": "1",
        },
        "inputs": [
            {"name": "src_ptr", "dtype": "bf16",
             "elems": "N_ROWS * QUANT_DIM",
             "range_lo": -4.0, "range_hi": 4.0},
        ],
        "outputs": [
            {"name": "mx_tensor_ptr", "dtype": "u8",
             "elems": "N_ROWS * FP4_PER_ROW"},
            {"name": "mx_scale_ptr", "dtype": "u8",
             "elems": "N_ROWS * SCALE_PER_ROW"},
        ],
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
