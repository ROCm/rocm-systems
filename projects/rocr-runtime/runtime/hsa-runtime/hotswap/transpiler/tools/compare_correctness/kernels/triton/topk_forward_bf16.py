"""Direct end-to-end recipe for `_topk_forward` — the MoE router's
top-k + bitmatrix-build kernel.

What it targets
===============

The `_topk_forward` Triton kernel from
`triton_kernels/topk_details/_topk_forward.py`.  Per
`hotswap/docs/gpt-oss-derisking.md §5`, this kernel carries:

* C2-DPP:    ``v_mov_b32_dpp ×10``
* C4:        ``s_and_saveexec_b32 ×1``

Direct-invocation caveat — tuple args
=====================================

`_topk_forward`'s real signature takes TUPLE-typed args
(``PeerYvs``, ``PeerYis``, ``PeerBits``) whose length is a
``tl.constexpr`` bound via ``N_PEERS: tl.constexpr = len(PeerYvs)``.
Triton's AST-source AOT-compile surface (used by
``kernels/triton/aot_compile.py``) maps a flat name->type dict into
the kernel's positional signature; there's no clean way to declare
a tuple arg in that surface.

Workaround: this file declares a thin ``@triton.jit`` wrapper
(``_topk_forward_single_peer``) that takes flat pointer args and
invokes ``_topk_forward`` with 1-element tuples.  The wrapper's IR
lowers 1:1 to ``_topk_forward``'s code after Triton's constexpr
static_range unrolls the 1-peer loop.  Non-all_gather GPT-OSS
deployments run with exactly 1 peer (per ``topk_forward`` driver in
``triton_kernels/topk.py``: ``make_empty`` returns
``(ret,)`` when ``all_gather=False``), so the 1-peer wrapper is
semantically identical to the production path.

Shape choices
=============

Fixed shape (single-value sweep):

* ``N_ROWS = 512`` — number of tokens.  Must be a multiple of
  ``BLOCK_M = 32``.
* ``N_COLS = 128`` — number of experts.  Power of 2 so
  ``N_EXPTS_PAD == N_COLS`` and the driver's ``n_cols_words = N_COLS
  // 32 = 4`` formula yields an integer.
* ``N_EXPTS_ACT = 4`` — top-k.  The captured GPT-OSS blob runs at
  k=4 (GPT-OSS MoE selects 4 of 128 experts per token).
* ``BLOCK_M = BLOCK_N = 32`` — matches the driver hardcode.
* ``num_warps = 4`` — matches the captured blob's
  ``max_flat_workgroup_size=128``.
* ``APPLY_SOFTMAX = True`` — the deployed MoE path applies softmax;
  false would skip several obstruction-carrying lines.
* ``USE_PROVIDED_INDX = False`` — means we go through the
  ``streaming_topk`` path rather than a pre-baked index gather.
  Covers the full DPP-reduction + permlane + C4 set.

The bitmatrix has a TRANSPOSED-THEN-SLICED layout
(``torch.transpose(bitmatrix_data, 0, 1)[:n_rows]`` in the driver).
Post-transpose strides are (1, cdiv(N_ROWS, 32) * 32) = (1, 512).
``stride_rm = 1`` and ``stride_rn = 512`` are both ``tl.constexpr``
at this fixed N_ROWS.

Expected outcomes
=================

Three possible verdicts, each informative:

* ``salmon=match`` — the kernel raises AND numerically matches.
  Pins another wave-size-class obstruction as correctly handled.
* ``salmon=EXIT=2`` at raise time on an unsupported opcode — like
  ``v_mad_nc_i64_i32`` in the ``downcast_to_mxfp`` probes — tells
  us this kernel shares the same handler gap.
* ``salmon=EXIT=2`` at runtime (HIP error 700) — like the
  ``sum_bitmatrix_rows_u32`` probe — tells us the salmon lowering
  crashes on composition-level obstruction at this shape.
* ``salmon=WRONG k/N`` — silent miscompile — immediate halt.
"""

import triton
import triton.language as tl
from triton_kernels.topk_details._topk_forward import _topk_forward


@triton.jit
def _topk_forward_single_peer(
    X, stride_xm,
    Yv, Yi, stride_ym,
    Bits,
    n_rows, n_expts_tot,
    dst_offs_m,
    USE_PROVIDED_INDX: tl.constexpr,
    APPLY_SOFTMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rn: tl.constexpr,
):
    """Thin 1-peer wrapper around `_topk_forward`.  Packs single
    pointer args into 1-element tuples so Triton's signature can
    express them as flat pointers (and so `aot_compile.py` can AOT-
    compile the kernel via its name->type signature map).

    The wrapper is semantically equivalent to the production
    ``topk_forward(..., all_gather=False)`` driver: the driver
    always passes 1-element tuples in the non-all_gather path
    (``make_empty`` returns ``(ret,)``), and Triton's static_range
    unrolls the 1-peer loop at compile time, so the wrapped
    kernel's generated code matches ``_topk_forward``'s code for
    N_PEERS=1 byte-for-byte.  Only the outer stack frame differs.
    """
    _topk_forward(
        X, stride_xm,
        (Yv,), (Yi,), stride_ym,
        USE_PROVIDED_INDX, (Bits,), stride_rm, stride_rn,
        n_rows, n_expts_tot,
        dst_offs_m, APPLY_SOFTMAX,
        BLOCK_M, N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N,
    )


# Fixed shape per the driver in `triton_kernels.topk.topk_forward`
# (non-all_gather, n_rows=512, n_expts_tot=128, k=4).
_N_ROWS          = 512
_N_COLS          = 128
_N_EXPTS_ACT     = 4
_N_EXPTS_PAD     = _N_COLS
_BLOCK_M         = 32
_BLOCK_N         = 32
_N_ROWS_PAD32    = ((_N_ROWS + 31) // 32) * 32  # = 512; strides derived below


RECIPES = [
    {
        "name": "topk_forward_bf16",
        "kernel_fn":     _topk_forward_single_peer,
        "kernel_symbol": "_topk_forward_single_peer",
        "signature": {
            "X":          "*bf16",
            "stride_xm":  "i32",
            "Yv":         "*bf16",
            "Yi":         "*i16",
            "stride_ym":  "i32",
            "Bits":       "*u32",
            "n_rows":     "i32",
            "n_expts_tot":"i32",
            "dst_offs_m": "i32",
        },
        "constexprs": {
            # USE_PROVIDED_INDX=False -> full streaming-topk path
            # (hits the richest obstruction profile).
            "USE_PROVIDED_INDX":  0,
            "APPLY_SOFTMAX":      1,
            "BLOCK_M":            _BLOCK_M,
            "N_EXPTS_PAD":        _N_EXPTS_PAD,
            "N_EXPTS_ACT":        _N_EXPTS_ACT,
            "BLOCK_N":            _BLOCK_N,
            # Post-transpose bitmatrix strides (see docstring above).
            "stride_rm":          1,
            "stride_rn":          _N_ROWS_PAD32,
        },
        "harness_constants": {
            # N_EXPTS_ACT is already a constexpr (baked into Triton).
            # N_COLS_WORDS / N_ROWS_PAD32 / N_COLS are pure harness-
            # scope entries referenced by elems / scalar_args / grid.
            "N_COLS":        _N_COLS,
            "N_COLS_WORDS":  _N_COLS // 32,
            "N_ROWS_PAD32":  _N_ROWS_PAD32,
        },
        "scalar_args": {
            "stride_xm":    "N_COLS",
            "stride_ym":    "N_EXPTS_ACT",
            "n_rows":       "N_ROWS",
            "n_expts_tot":  "N_COLS",
            "dst_offs_m":   "0",
        },
        "num_warps": 4,
        "shape_dim": "N_ROWS",
        # Fixed single-shape sweep — the bitmatrix strides depend on
        # ceil_div(N_ROWS, 32) * 32 and must be tl.constexpr; sweeping
        # would require per-shape recompilation which the harness
        # doesn't support.  See the docstring above for why a single
        # shape is still informative.
        "default_shapes": [_N_ROWS],
        "grid": {
            "x": "ceil_div(N_ROWS, BLOCK_M)",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            # bf16 router logits.  Range [-4, 4) covers a plausible
            # pre-softmax logit range.
            {"name": "X", "dtype": "bf16",
             "elems": "N_ROWS * N_COLS",
             "range_lo": -4.0, "range_hi": 4.0},
        ],
        "outputs": [
            # Top-k values (bf16).  Shape (N_ROWS, N_EXPTS_ACT).
            #
            # Per-output `rel-rms` comparator matches the MODE=4 entry
            # in `topk_forward_bisect.py::_recipe` — same pipeline
            # (streaming_topk + softmax + sort), same observed drift
            # class (rms(diff)/rms(gold) ~= 0.15 after closing the
            # V_FMA_MIX inline-constant and _D16_HI store bugs).  See
            # that file's mode-table comment for the full derivation.
            {"name": "Yv", "dtype": "bf16",
             "elems": "N_ROWS * N_EXPTS_ACT",
             "comparator": {"kind": "rel-rms", "tol": 0.25}},
            # Top-k indices (i16).  Same shape.
            #
            # NO per-output comparator override — falls through to
            # the top-level bit-exact comparator.  This deliberately
            # leaves the recipe's verdict as "WRONG" on Yi until the
            # divergence is characterised.
            #
            # What we've established so far (2026-04-22, after
            # closing FMA_MIX-op_sel and GLOBAL/FLAT_STORE_D16_HI):
            #
            # * Yv passes rel-rms(tol=0.25) — aggregate softmax
            #   output values match in RMS, so the PICKED VALUES'
            #   distribution is close between native and salmon.
            # * Yi (index output) has the SAME INDEX SET in only
            #   6.8% (35/512) of rows.  Top-1 pick alone differs in
            #   49% (249/512) of rows.  That's MUCH more divergence
            #   than a few bf16 ULPs of drift can explain under any
            #   reasonable tie-break-flip model.
            # * MODE=4 in topk_forward_bisect.py passes with
            #   rel-rms(tol=0.25) on Yv because that recipe does
            #   `tl.sort(y_values, dim=1, descending=True)` at the
            #   end — the final sort is by VALUE alone and is
            #   invariant to which set of 4 indices streaming_topk
            #   returned (as long as the values are similar).  The
            #   production `_topk_forward` kernel does NOT do this
            #   final sort; it writes streaming_topk's (value, index)
            #   output directly, so index-set divergence is visible
            #   in Yi.
            #
            # Hypotheses to investigate (not yet bisected):
            #
            # * `tl.topk` / `tl.bitonic_merge` under cross-widening
            #   produce different index-tie-breaks from native.
            # * The u32 (value_key << 16 | index_key) packing used
            #   by streaming_topk has a lane-local assumption that
            #   breaks under wave32->wave64 projection.
            # * `tl.load(X, mask, other=-inf)` with a cross-widened
            #   mask loads wrong lanes' X values in salmon (would
            #   explain why salmon's post-softmax Yv[0] is often
            #   LARGER than native's — reading a neighbouring
            #   lane's value that happens to be larger).
            #
            # A permissive comparator here (e.g. rel-rms(tol=0.5))
            # WOULD graduate this recipe to "match" on the RMS
            # metric (see analysis in commit message), but that
            # would mask the specific question of "does salmon pick
            # the same top-k SET as native" — which is exactly the
            # correctness invariant `_topk_forward` is supposed to
            # preserve.  Holding Yi bit-exact keeps the test honest
            # until the root cause is nailed down.
            {"name": "Yi", "dtype": "i16",
             "elems": "N_ROWS * N_EXPTS_ACT"},
            # Bitmatrix: transposed-then-sliced (N_ROWS_PAD32,
            # N_COLS_WORDS) u32.  We allocate (N_ROWS_PAD32 *
            # N_COLS_WORDS) elements; the kernel writes only the
            # [:N_ROWS, :] prefix in logical layout but the memory
            # backs the full pre-transpose rectangle.  Same
            # rationale as Yi: Bits is strictly derived from Yi, so
            # any Yi divergence causes corresponding Bits
            # divergence; bit-exact comparator keeps the signal
            # visible.
            {"name": "Bits", "dtype": "u32",
             "elems": "N_ROWS_PAD32 * N_COLS_WORDS"},
        ],
        # Fallback comparator for Yi / Bits (both use the fallback;
        # Yv has its own rel-rms override above).  Bit-exact keeps
        # the Yi divergence visible as WRONG until the underlying
        # streaming_topk / cross-widening semantic is understood —
        # see the Yi output comment for the full rationale.
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
