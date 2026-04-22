"""Direct end-to-end recipe for `_sum_bitmatrix_rows` — heaviest-DPP
GPT-OSS kernel, invoked at its real `num_warps=8` shape.

What it targets
===============

The `_sum_bitmatrix_rows` Triton kernel from
`triton_kernels/tensor_details/bitmatrix_details/sum_bitmatrix_rows.py`.
Per `hotswap/docs/gpt-oss-derisking.md §5`, this kernel carries:

* C2-hard: ``ds_swizzle_b32 ×8``
* C2-DPP:  ``v_mov_b32_dpp ×96``   (largest C2-DPP load in the corpus)

The single-primitive canaries
(`canary_dpp_reduce_fp32`, `canary_dpp_compound_add_u32`,
`canary_ds_swizzle_swap1`, `canary_ds_swizzle_quad_perm`) each pin one
obstruction class in isolation, at `num_warps=1`.  This recipe
complements them by invoking the EXACT kernel the GPT-OSS MoE router
calls, at its real `num_warps=8` configuration and its real operand
shape (uint32 bitmatrix + per-column popcount).  If the single-
primitive canaries match but this recipe WRONGs, we have evidence of
a composition-specific silent miscompile that the single-class probes
can't reach; if this recipe also matches, the §9.2 silent-miscompile
prediction for the ``bitmatrix-metadata / sum_bitmatrix_rows`` class
is directly falsified on its real shape.

Why this and not a canary wrapper
=================================

`canary_bitmatrix_composite` is a Triton workalike for the same
obstruction class but at `num_warps=1`, which puts it in the
phantom-lane regime (``max_flat_workgroup_size=32 < targetWaveSize=64``)
and makes it graduate to principled ``EXIT=2`` refusal.  That verdict
is correct but it leaves the `num_warps>1` regime — where the actual
GPT-OSS kernel runs — empirically unverified on this obstruction
class.  This recipe closes that gap by running with ``num_warps=8``
(matching the driver in
``triton_kernels.tensor_details.bitmatrix.make_bitmatrix_metadata``),
which gives ``max_flat_workgroup_size = 8 * 32 = 256 >= 64`` — above
the phantom-lane threshold, so the classifier raises OK and the
numerical result is directly comparable native-vs-salmon.

Why `init: zero` on `Out`
=========================

The kernel's final line is
``tl.atomic_add(Out + offs_n, ..., sem='relaxed')``.  The driver
pre-zeroes ``Out`` (see `sum_bitmatrix_rows` torch wrapper:
``out = torch.zeros(..., dtype=torch.int32)[...]``).  This recipe
declares ``init: zero`` on ``Out`` so the harness's new per-output
init mode (plumbed in this commit) gives the kernel a zeroed initial
value.  Without ``init: zero`` the harness's default ``0xA5`` fill
would produce ``0xA5A5A5A5 + sum(columns)`` instead of
``sum(columns)`` — a silent off-by-`0xA5A5A5A5` that would falsely
report a salmon-vs-native miscompile.  ``OutPartials`` is written by
non-atomic ``tl.store`` and keeps the default sentinel so unwritten
lanes surface as ``0xA5A5A5A5`` in the diff.

Shape choices
=============

The kernel's signature parameterises over (n_rows, n_cols_u32) plus
block sizes.  Pinned here:

* ``N_COLS_U32 = 1`` — one uint32 column, 32 experts.  Keeps the grid
  axis-0 (``grid_m``) as the only swept dimension, so the obstruction
  profile per launch is stable across the sweep.  Exercising larger
  ``N_COLS_U32`` would add grid_n > 1 which doesn't change the
  per-program obstruction class (the DPP / ds_swizzle are per-program
  primitives).
* ``BLOCK_MM = 128``, ``BLOCK_M = 32``, ``TILE_SIZE = 4`` — matches
  the driver's hardcoded defaults (``PARTIAL_BLOCK_M = 32``;
  ``TILE_SIZE = max(1, 128 // PARTIAL_BLOCK_M) = 4``;
  ``BLOCK_MM = 128``).
* Swept shapes: multiples of 128 so every program lands on a full
  tile.  Partial-tile masking is out of scope for this probe (the
  kernel supports it but we probe the dense path).

Expected verdicts
=================

* ``native`` : ``gold``.  Triton's gfx942 build is the reference.
* ``legacy`` : ``SIG6`` or similar crash.  The byte-level transpiler
  is not expected to handle Triton's gfx1250 DPP / ds_swizzle
  primitives correctly; orthogonal limitation, not a salmon concern.
* ``salmon`` : expected ``match`` on every shape.  Falsifies §9.2's
  silent-miscompile prediction for the composition class.  A ``WRONG
  k/N`` verdict falsifies the optimistic reading of the canary grid
  (single-class canaries all match, but composition of classes
  diverges) and triggers investigation per the same playbook as
  ``canary_bitmatrix_composite``.

Regression contract
===================

* ``match`` -> ``WRONG`` on salmon: regression in one of the
  obstruction-class handlers (DPP compound-add, ds_swizzle, vpopc's
  per-tile reductions) at ``num_warps=8``.  First-mismatch index in
  Failures tells you which of the 32 output lanes diverged; pair with
  a raised-IR dump (``raise_cli ... --emit-ir``) to triage.
* ``match`` -> ``EXIT=2`` on salmon: a raise-time refusal fired that
  didn't exist before.  Likely candidates: the phantom-lane arm
  tightening to also refuse ``max_flat_workgroup_size ==
  targetWaveSize`` (today only the strict-less-than form refuses),
  or a new C5 pattern matched by the classifier.  Either is a
  deliberate narrowing; update expectations and document.
"""

import triton
import triton.language as tl
from triton_kernels.tensor_details.bitmatrix_details.sum_bitmatrix_rows \
    import _sum_bitmatrix_rows


# Shape parameters pinned per
# triton_kernels.tensor_details.bitmatrix.make_bitmatrix_metadata.
_PARTIAL_BLOCK_M = 32
_TILE_SIZE       = 4                                 # max(1, 128 // _PARTIAL_BLOCK_M)
_BLOCK_MM        = _PARTIAL_BLOCK_M * _TILE_SIZE     # 128
_BLOCK_M         = _PARTIAL_BLOCK_M                  # 32
_N_COLS_U32      = 1                                 # 32 experts (1 u32 per row)


RECIPES = [
    {
        "name": "sum_bitmatrix_rows_u32",
        "kernel_fn":     _sum_bitmatrix_rows,
        "kernel_symbol": "_sum_bitmatrix_rows",
        "signature": {
            "B":            "*u32",
            "shape_bm":     "i32",
            # stride_bm, stride_bn: tl.constexpr -> see constexprs
            "Out":          "*i32",
            "OutPartials":  "*i32",
            # stride_pm: tl.constexpr -> constexprs
            "stride_pn":    "i32",
            "shape_pn":     "i32",
            # BLOCK_MM, BLOCK_M: tl.constexpr -> constexprs
        },
        "constexprs": {
            # B is a row-major (N_ROWS, _N_COLS_U32) uint32 tensor.
            "stride_bm":   _N_COLS_U32,
            "stride_bn":   1,
            # OutPartials is a torch.transpose-produced view of a
            # (grid_n*32, grid_m*TILE_SIZE) row-major int32 tensor;
            # post-transpose row-stride is 1 element.
            "stride_pm":   1,
            "BLOCK_MM":    _BLOCK_MM,
            "BLOCK_M":     _BLOCK_M,
        },
        "harness_constants": {
            # Expression-scope entries referenced by elems / grid /
            # scalar_args below.  Must not overlap with constexprs or
            # shape_dim.
            "N_COLS_U32": _N_COLS_U32,
            "TILE_SIZE":  _TILE_SIZE,
        },
        "scalar_args": {
            # shape_bm = total row count the kernel masks its loads
            # against = the swept shape dim.
            "shape_bm":  "N_ROWS",
            # stride_pn = post-transpose OutPartials' column-stride in
            # elements = pre-transpose row-stride = grid_m * TILE_SIZE.
            "stride_pn": "ceil_div(N_ROWS, BLOCK_MM) * TILE_SIZE",
            # shape_pn = grid_n * 32 bits per column-tile = N_COLS_U32 * 32.
            "shape_pn":  "N_COLS_U32 * 32",
        },
        "num_warps": 8,
        "shape_dim": "N_ROWS",
        # Multiples of BLOCK_MM=128 so every tile is full (no partial-
        # tile masking path).  Power-of-two spacing covers both the
        # single-program case (128) and multi-program cases
        # (512/2048/8192) so a per-program miscompile has multiple
        # witnesses.
        "default_shapes": [128, 512, 2048, 8192],
        "grid": {
            "x": "ceil_div(N_ROWS, BLOCK_MM)",
            "y": "N_COLS_U32",
            "z": "1",
        },
        "inputs": [
            # Deterministic full-u32-range bitmatrix.  Integer dtypes
            # ignore range_{lo,hi} per the aot_compile.py contract —
            # the bytes are sampled across the full u32 bit range.
            {"name": "B", "dtype": "u32",
             "elems": "N_ROWS * N_COLS_U32"},
        ],
        "outputs": [
            # Out accumulates via `tl.atomic_add` — the kernel reads
            # the initial value, so the harness MUST pre-zero rather
            # than sentinel-fill.  See aot_compile.py's
            # `VALID_OUTPUT_INIT_MODES` for the contract.
            {"name": "Out", "dtype": "i32",
             "elems": "N_COLS_U32 * 32",
             "init": "zero"},
            # OutPartials is written deterministically by `tl.store`;
            # keep the default `sentinel` so any un-stored lane
            # surfaces as 0xA5A5A5A5 in the diff.
            {"name": "OutPartials", "dtype": "i32",
             "elems": "ceil_div(N_ROWS, BLOCK_MM) * TILE_SIZE * "
                      "N_COLS_U32 * 32"},
        ],
        # Bit-exact integer comparator.  The column sums are integer
        # popcounts over a fixed u32 input; there is no ULP to absorb.
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
