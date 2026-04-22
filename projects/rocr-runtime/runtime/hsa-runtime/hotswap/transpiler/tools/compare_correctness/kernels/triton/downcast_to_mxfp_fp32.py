"""Direct end-to-end recipe for `_downcast_to_mxfp` — the MoE router's
fp32 -> MXFP4 quantization kernel.

What it targets
===============

The `_downcast_to_mxfp` Triton kernel from
`triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py`.
Per `hotswap/docs/gpt-oss-derisking.md §5`, this kernel carries:

* C2-DPP: ``v_mov_b32_dpp ×8``

Why the fp32 src dtype
======================

The captured GPT-OSS blob (`_downcast_to_mxfp_124b6cd2a33b.hsaco`) was
compiled at `max_flat_workgroup_size=128`, which per the driver in
`numerics_details/mxfp.py` (``NUM_WARPS = 4 if x.dtype == torch.float32
else 8``) means the observed deployment runs with `num_warps=4` +
fp32 src.  For consistency with the captured blob and to put the
verdict directly on the same code path GPT-OSS exercises, this recipe
pins those same choices: `src_dtype=fp32`, `num_warps=4`.  On wave32
source that's wg=128 threads; re-targeted to wave64 on gfx942 that's
wg=128 threads = 2 waves — above the phantom-lane threshold, so the
classifier raises OK and the numerical result is directly comparable.

Why this complements the existing coverage
==========================================

The single-primitive `canary_dpp_reduce_fp32` + `canary_dpp_compound_add_u32`
canaries probe C2-DPP in isolation at `num_warps=1`.  This recipe
invokes the actual GPT-OSS MXFP4 downcast kernel at its production
shape so a composition-level bug (see the sibling
`sum_bitmatrix_rows_u32` recipe, which exposed a salmon-side runtime
crash at `num_warps>=4`) can be discriminated across kernels with
different obstruction mixes.

Shape choices
=============

* ``BLOCK_OUT_DIM = 32``, ``BLOCK_QUANT_DIM = 128``,
  ``MICROBLOCK_SIZE = 32`` — all match the driver's hardcoded
  defaults.  `BLOCK_QUANT_DIM = microblock_size * 4 = 128` per the
  same driver.
* ``quant_dim = BLOCK_QUANT_DIM = 128`` pinned.  Sweep `outer_dim =
  N_ROWS`.  Keeps grid_y = 1 so the only swept axis is grid_x
  (per-program obstruction class is stable across the sweep).
* ``DEQUANT_SCALE_ROUNDING_MODE = 0`` (ROUND_UP) matches the driver
  default.

Expected verdicts
=================

* ``native`` : ``gold``.
* ``legacy`` : likely ``SIG6`` (byte-level transpiler unrelated
  limitation).
* ``salmon`` : open empirical question.  The sibling
  `sum_bitmatrix_rows_u32` recipe crashes on salmon at runtime
  (HIP error 700); whether `downcast_to_mxfp` reproduces the same
  class of bug is exactly what this probe resolves.
"""

import triton
import triton.language as tl
from triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp \
    import _downcast_to_mxfp


# Shape constants pinned per the driver in
# `triton_kernels.numerics_details.mxfp.downcast_to_mxfp`.
_BLOCK_OUT_DIM       = 32
_MICROBLOCK_SIZE     = 32
_BLOCK_QUANT_DIM     = _MICROBLOCK_SIZE * 4    # 128 — driver constant
_QUANT_DIM           = _BLOCK_QUANT_DIM         # single column-tile sweep
_DEQUANT_ROUNDING    = 0                        # ROUND_UP (driver default)


RECIPES = [
    {
        "name": "downcast_to_mxfp_fp32",
        "kernel_fn":     _downcast_to_mxfp,
        "kernel_symbol": "_downcast_to_mxfp",
        # Signature (non-constexpr args in kernel def order):
        #   mx_tensor_ptr, stride_mxt_outer,
        #   mx_scale_ptr,  stride_mx_scale_outer, stride_mx_scale_quant,
        #   src_ptr,       stride_src_outer, stride_src_quant,
        #   outer_dim, quant_dim
        "signature": {
            "mx_tensor_ptr":          "*u8",
            "stride_mxt_outer":       "i32",
            # stride_mxt_quant: tl.constexpr (kernel asserts == 1) -> constexprs
            "mx_scale_ptr":           "*u8",
            "stride_mx_scale_outer":  "i32",
            "stride_mx_scale_quant":  "i32",
            "src_ptr":                "*fp32",
            "stride_src_outer":       "i32",
            "stride_src_quant":       "i32",
            "outer_dim":              "i32",
            "quant_dim":              "i32",
            # BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM, MICROBLOCK_SIZE,
            # DEQUANT_SCALE_ROUNDING_MODE all tl.constexpr -> constexprs
        },
        "constexprs": {
            "stride_mxt_quant":            1,           # kernel asserts == 1
            "BLOCK_SIZE_OUT_DIM":          _BLOCK_OUT_DIM,
            "BLOCK_SIZE_QUANT_DIM":        _BLOCK_QUANT_DIM,
            "MICROBLOCK_SIZE":             _MICROBLOCK_SIZE,
            "DEQUANT_SCALE_ROUNDING_MODE": _DEQUANT_ROUNDING,
        },
        "harness_constants": {
            "QUANT_DIM":       _QUANT_DIM,
            "SCALE_PER_ROW":   _QUANT_DIM // _MICROBLOCK_SIZE,    # 4
            "FP4_PER_ROW":     _QUANT_DIM // 2,                    # 64
        },
        "scalar_args": {
            "stride_mxt_outer":      "FP4_PER_ROW",     # 64
            "stride_mx_scale_outer": "SCALE_PER_ROW",   # 4
            "stride_mx_scale_quant": "1",
            "stride_src_outer":      "QUANT_DIM",       # 128
            "stride_src_quant":      "1",
            "outer_dim":             "N_ROWS",
            "quant_dim":             "QUANT_DIM",       # 128
        },
        "num_warps": 4,
        "shape_dim": "N_ROWS",
        "default_shapes": [32, 128, 512, 2048],
        "grid": {
            "x": "ceil_div(N_ROWS, BLOCK_SIZE_OUT_DIM)",
            "y": "ceil_div(QUANT_DIM, BLOCK_SIZE_QUANT_DIM)",
            "z": "1",
        },
        "inputs": [
            # fp32 src in a reasonable range — covers normals / small
            # subnormals; the quant path handles both branches.
            {"name": "src_ptr", "dtype": "fp32",
             "elems": "N_ROWS * QUANT_DIM",
             "range_lo": -4.0, "range_hi": 4.0},
        ],
        "outputs": [
            # FP4 packed tensor — 2 FP4 nibbles per u8.
            {"name": "mx_tensor_ptr", "dtype": "u8",
             "elems": "N_ROWS * FP4_PER_ROW"},
            # E8M0 scale bytes — one per MICROBLOCK_SIZE elements.
            {"name": "mx_scale_ptr", "dtype": "u8",
             "elems": "N_ROWS * SCALE_PER_ROW"},
        ],
        # Bit-exact u8 comparator — quantization output is an integer
        # bit pattern; any bit difference is a real divergence.
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
