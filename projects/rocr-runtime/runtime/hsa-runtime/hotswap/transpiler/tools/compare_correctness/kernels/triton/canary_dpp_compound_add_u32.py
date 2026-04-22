"""Canary for compound integer DPP (``v_add_nc_u32_dpp``) — the exact
opcode GPT-OSS's ``bitmatrix_metadata_compute_stage1`` emits ×4
(§5 of ``hotswap/docs/gpt-oss-derisking.md``).

What it targets
===============

Companion to ``canary_dpp_compound_add_fp32``, tightening the integer
DPP-compound path the fp32 canary only approximates:

- ``canary_dpp_compound_add_fp32`` forces ``v_add_f32_dpp`` — compound
  DPP on a *float* add.  Empirically passes under salmon at every
  shape tried, which tells us the DPP-compound-add PATH is lifted
  correctly but does NOT rule out integer-specific handler bugs:
  ``v_add_nc_u32_dpp`` may route through a different
  ``handle_valu.cpp`` arm (integer VOP2 vs float VOP2), hit different
  modifier canonicalisation, or trigger different saturation /
  overflow semantics in the DPP-modifier lowering.

- ``canary_dpp_compound_add_u32`` (this one) forces
  ``v_add_nc_u32_dpp`` directly via an ``int32`` ``tl.sum``
  reduction.  This is the single highest-density DPP opcode in the
  GPT-OSS corpus (4 invocations in ``_bitmatrix_metadata_compute_stage1``
  alone), and it goes through the integer VOP2 arm — a handler code
  path the fp32 canary does not exercise.

``gpt-oss-derisking.md §7.3`` calls P5 (the DPP intrinsic lift) the
largest outstanding correctness risk.  §7.3's text assumes the raiser
canonicalises the DPP modifier away; empirical fp32 evidence narrows
that scope, and this canary narrows it further.

Observed verdicts at landing
=============================

* ``native`` : gold.  Triton recipes use the native-gfx942 run as
  the reference, so this is the reference itself.
* ``legacy`` : SIG6 CRASH on every shape — ``2 waitcnt, 0
  exec-widened, 1 unsupported``.  The legacy text-transpiler hits
  an ``Either SourceMgr should be available`` LLVM assertion
  trying to emit an unhandled DPP shape.  This is an orthogonal
  legacy limitation, not a salmon concern.
* ``salmon`` : match on every shape.  The raiser lifts all four
  ``v_add_nc_u32_dpp`` instances (row_shr:8, 4, 2, 1) and the
  trailing ``v_permlanex16_b32`` through their intrinsic
  equivalents; the gfx942 backend re-lowers to an equivalent
  reduction tree and the per-row sum matches bit-exactly.

The salmon ``match`` is the key signal: it tells us the raiser's
integer VOP2 + DPP-modifier lifting path is correct end-to-end on
a GPT-OSS-relevant opcode family.  A regression from ``match`` to
``WRONG`` surfaces as mismatches at every row position (each off
by the full row sum) or off-by-(sum - x[0]) if the DPP modifier
drops and the reduction collapses to a single-lane contribution.

Wave-dependence caveat
======================

Per the ``canary_dpp_compound_add_fp32`` docstring: gfx1250 (RDNA4)
emits *no* wave-size-dependent DPP modifiers — all ``row_shl:n`` /
``row_shr:n`` / ``row_xmask`` / ``quad_perm`` / ``dpp8`` the compiler
picks for the reduction tree operate within 16-lane rows or 4-lane
quads and are wave-invariant by ISA construction.  A dedicated
``canary_dpp_bcast15`` would have nothing to canary against; this
canary stays with the wave-invariant row_shl/shr/xmask envelope that
matches what GPT-OSS actually emits.

How it forces the instruction
=============================

One program per row, ``ROW = 32`` int32 columns, one wave per program.
Each row's sum is computed via ``tl.sum`` over int32 values and
subtracted from every element.  Triton's integer reduction lowering
on gfx1250 walks a DPP reduction tree over ``v_add_nc_u32_dpp`` for
the 4 stages (distances 1, 2, 4, 8), then one ``v_permlanex16_b32``
for the final distance-16 exchange.  The canary's job is to pin the
v_add_nc_u32_dpp stages; the permlanex16 overlap is already covered
by ``canary_permlanex16_rowmax_fp32``, so if this canary fails AND
the permlanex16 canary passes, the regression is on the integer-DPP
path specifically.

Harness schema notes
====================

Swept shape dim is ``N_ROWS``.  Input range is a narrow band of
non-negative integers (0..15) so the per-row sum stays well within
int32 (worst case 32 * 15 = 480, far below 2^31).  Output values
are signed — ``x[i] - sum`` can be as low as ``-480`` — which is
exactly representable in int32 and bit-exact under ``EQ``
comparison.

Bit-exact comparator: integer arithmetic has no ULP noise.  Any
mismatch between gold and candidate is a miscompile; a broken DPP
lift shows up as either the sum collapsing to a single-lane
contribution (subtraction of ~x[0] instead of the full sum) or as
the sum dropping entirely (output equalling input).  Both shapes
are loudly off.
"""
import triton
import triton.language as tl


@triton.jit
def canary_dpp_compound_add_u32_kernel(
    x_ptr, y_ptr,
    N_ROWS,
    ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, ROW)
    x = tl.load(x_ptr + pid * ROW + offs)
    s = tl.sum(x, axis=0)
    y = x - s
    tl.store(y_ptr + pid * ROW + offs, y)


RECIPES = [
    {
        "name": "canary_dpp_compound_add_u32",
        "kernel_fn": canary_dpp_compound_add_u32_kernel,
        "kernel_symbol": "canary_dpp_compound_add_u32_kernel",
        "signature": {
            "x_ptr":  "*i32",
            "y_ptr":  "*i32",
            "N_ROWS": "i32",
        },
        "constexprs": {
            "ROW": 32,
        },
        "num_warps": 1,
        "shape_dim": "N_ROWS",
        "default_shapes": [32, 128, 1024, 8192],
        "grid": {
            "x": "N_ROWS",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "x_ptr", "dtype": "i32", "elems": "N_ROWS * ROW",
             "range_lo": 0, "range_hi": 15},
        ],
        "outputs": [
            {"name": "y_ptr", "dtype": "i32", "elems": "N_ROWS * ROW"},
        ],
        # Integer arithmetic has no ULP noise — bit-exact compare is
        # what we want.  The harness's i32 path ignores `cmp.kind`
        # and does a straight `g[i] != a[i]` check regardless of the
        # recipe's comparator, but the Python validator still
        # requires a recognised kind — we declare `abs` with tol=0
        # to satisfy validation without changing runtime behaviour.
        # A DPP drop shows up as mismatches at every row position,
        # each off by the full row sum.  A DPP "collapse to single-
        # lane" regression shows up as off-by-(sum - x[0]).  Both
        # are loud under bit-exact compare.
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
