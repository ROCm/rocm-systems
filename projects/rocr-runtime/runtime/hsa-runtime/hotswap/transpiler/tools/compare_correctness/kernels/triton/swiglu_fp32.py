"""GPT-OSS primitive: SwiGLU activation (α = 1.702, limit = 7.0).

What it is
==========

The MoE expert activation in GPT-OSS.  From
``triton_kernels.swiglu_details._swiglu.compute_swiglu``
(@ triton @ HEAD, @ $HOME/rocm-systems/triton):

    gelu, linear = input[..., ::2], input[..., 1::2]
    gelu   = minimum(gelu, limit)            # one-sided clamp
    linear = clamp(linear, -limit, +limit)   # two-sided clamp
    s      = gelu / (1 + exp(-alpha * gelu)) # swish / silu
    out    = fma(s, linear, s)               # s * (linear + 1)

With ``alpha = 1.702`` and ``limit = 7.0``, both values taken straight
from GPT-OSS's ``PrecisionConfig``.  Input shape is ``(M, 2N)`` where
the last dim is interleaved gate / linear pairs; output is ``(M, N)``.

The scope-discovery capture flagged ``_swiglu`` as **clean on
wave-size** (outcome (a) in ``gpt-oss-derisking.md §5``): no
cross-lane primitives, no DPP, no EXEC writes — a straight
elementwise kernel.  Any salmon mismatch on this recipe therefore
attributes to elementwise-path issues, not to the wave-size rewrite
table.

Why this is in the `compare_correctness` corpus
==============================================

Two reasons:

1. **Coverage of GPT-OSS's activation path.**  SwiGLU is on the
   forward path of every MoE expert block in GPT-OSS.  Having a
   hand-authored version lets us verify salmon's elementwise
   lowering independently of the MoE-routing kernels (which also
   exercise the messier C2 obstruction classes).

2. **Second probe of the gfx11+ `v_*_num_*` bug.**  ``tl.minimum``
   and ``tl.clamp`` on gfx1250 lower to ``v_min_num_f32`` /
   ``v_max_num_f32`` (the gfx11+ IEEE-754-2019 quiet-NaN-suppressing
   variants).  ``rmsnorm_fp32`` already showed that the ``_num_f64``
   variant breaks salmon (via the ``v_rsq_f32`` lowering's internal
   use of ``v_min_num_f64``); if this recipe also fails and the
   disassembly shows ``v_*_num_f32`` opcodes, we have a
   pattern-level gap rather than a one-off sqrt-only bug, and the
   raiser fix is "add the whole ``_num_*`` opcode family to the
   decode map" rather than chasing one-off sites.

Status note (salmon verdict: bit-exact match on every shape as of
2026-04-21; see historical analysis below)
==========================================================

The failure described below landed as a bit-exact fix in the
``ModuloReplicationProjection::extractLaneBitFromWaveMask`` widen-
by-replicate patch.  Pre-fix, salmon miscompiled swiglu at every
shape; ``max|err| ≈ 56`` matched ``limit × (limit + 1) = 7 × 8``,
exactly half of the output elements were wrong, and the wrong
elements shared a constant value across shapes (``actual =
−2.87e-16``).  The historical analysis below is preserved as the
investigation anchor for a future similar-shape regression.

A previous revision of this docstring attributed the failure to the
gfx11+ ``v_*_num_*`` opcode family (this kernel emits
``v_dual_max_num_f32 × 8`` and ``v_med3_num_f32 × 8``).  That
attribution has been **refuted**:

- The gfx12 ``v_med3_num_f32`` is an LLVM assembly-mnemonic alias
  for the pre-existing ``V_MED3_F32`` pseudo (see
  ``VOP3Instructions.td:2100`` — ``VOP3_Realtriple_with_name_gfx12
  <0x231, "V_MED3_F32", "v_med3_num_f32">``).  The raiser's
  ``opcode_map.cpp`` already maps ``V_MED3_F32_e64`` to
  ``SemOp::V_MED3_F32`` and ``handle_valu.cpp`` produces the
  expected ``minnum(maxnum(s0, s1), s2)`` IR shape.  Verified by
  inspecting the raised IR: every ``tl.clamp(linear, −7, 7)`` call
  lifts cleanly to ``llvm.minnum.f32(7.0, llvm.maxnum.f32(−7.0, x))``.
- ``v_dual_max_num_f32`` routes through ``handle_vopd.cpp``'s
  dual-dispatch which internally reuses the same handler; the
  raised IR shows ``vopd_fmax = call float @llvm.maxnum.f32(...)``
  and ``vopd_fmin = call float @llvm.minnum.f32(7.0, ...)`` as
  expected.

A previous hypothesis blamed a miscompiled **stride-2 deinterleaved
load** (the kernel reads ``gelu = A[..., ::2]`` and ``linear =
A[..., 1::2]``; a raiser that dropped one of the two strides would
corrupt half the outputs).  That hypothesis has been **refuted** by
a Phase-0 probe on 2026-04-21:

1. ``HSA_SALMON_DUMP_DIFF=<dir> compare_correctness --recipe=swiglu_fp32``
   preserves the salmon and native output blobs; numpy-diffing them
   per index reveals the miscompile pattern is **wave-split**, not
   stride-2:
     * Row 0 (128 outputs): indices 0..31 OK, 32..63 wrong, 64..95
       OK, 96..127 wrong.
     * Same pattern in every row (M=16 rows, ~62 wrong per row,
       ~962 wrong total of 2048).
     * Wrong lanes always read ``-0.0`` (the OOB-sentinel-returns-
       zero signature of a buffer_load with a 0x80000000 offset).
2. Target gfx942 is wave64; cross-widening packs two source wave32s
   per target wave64.  The per-target-wave pattern "lanes 0..31
   correct, lanes 32..63 wrong" is **exactly** the upper-half-of-
   target-wave-miscompiles shape — not an even/odd stride split.
3. A minimum-reproducer kernel ``out[tid] = tid`` (128-thread
   workgroup, same wave32→wave64 modrep path) passes bit-exact, so
   the bug is NOT in basic ``workitem.id.x()`` / lane-id lifting.
4. A hipcc-authored stride-2 deinterleaved-load kernel
   (``out[tid] = A[2*tid]*10 + A[2*tid+1]``, identical addressing
   to swiglu's two loads) also passes bit-exact under salmon, so
   the bug is NOT a generic stride-2 load miscompile either.

So the failure is **specific to Triton's codegen shape**: the swiglu
kernel's source asm emits one stream via ``buffer_load_b32 v<dst>,
v<voff>, s[srd:srd+3], 0 offen`` (gelu, via an explicit SRD) and the
sibling stream via ``global_load_b32 v<dst>, v<voff>, s[base:base+1]
offset:4 scale_offset`` gated by a per-slot ``s_and_saveexec_b64 s,
vcc`` where ``vcc`` is derived from a narrow 32-bit SGPR mask
(``v_lshrrev_b32_e64 v, v22, s<narrow_mask>``).  The narrow mask is
the trunc'd ballot of the per-lane ``offs < N`` predicate.  For this
kernel the predicate is all-TRUE and narrow-vs-64-bit truncation
should be semantically equivalent; empirically it is NOT, and the
upper half of every target wave64 ends up with gelu loaded from the
OOB sentinel.

The root cause (confirmed by instrumenting the raised IR to emit a
per-lane probe of ``%vgpr11.1`` — the per-lane buffer_load offset
produced by the ``v_cndmask_b32_e64 v, 0x80000000, v<off>, s<mask>``
shape) was in
``ModuloReplicationProjection::extractLaneBitFromWaveMask``.  The
fallback path for a narrow (i32, source-wave-width) mask did a
plain ``zext i32 → i64`` when widening to the target wave-mask
width, then ``lshr wide, full_target_lane_id``.  For
target-wave lanes 32..63, ``lshr`` shifted into the zero-padded
upper half of the widened mask and always retrieved a zero — the
downstream ``select`` unconditionally picked its FALSE branch,
which in the Triton buffer_load shape is the 0x80000000 OOB
sentinel offset, and the per-wave-half stores all landed out-of-
bounds (silently dropped by the SRD bounds check).

The fix: replicate the narrow mask into the upper half before the
``lshr`` (``new_wide = zext | (zext << W_src)``), matching the
``WaveNativeProjection::extractLaneBitFromWaveMask`` widen-by-
replication path and MODREP's "target lane L reads bit ``L mod
W_src`` of the source mask" contract.  The fix lives in
``wave_projection.cpp`` under ``ModuloReplicationProjection::
extractLaneBitFromWaveMask`` and is pinned by the expanded lit
fixture ``lit_tests/v_cmp_cndmask_sgpr_scalar_clobber/``.

This failure is distinct from the ``corpus_layernorm_fp32`` C5
predicate-chain class (which structurally depends on a
``workitem.id.x()``-fed comparison); swiglu's predicate was
``offs < N`` with ``offs`` being either a literal constant or a
simple SGPR-derived uniform ``pid * 128`` for slot 1+, so no
``tid``-bearing ``icmp`` was involved.  The same fix does, as a
side-effect, graduate ``corpus_layernorm_fp32 N=1024`` from
``WRONG`` to ``match`` (the larger shape has a different narrow-
mask usage pattern that was also tripping the pre-fix zext-widen);
smaller layernorm shapes (N=128/256/512) remain the C5 predicate-
chain class and are out of scope for this fix.  Debugger plumbing
(``HSA_SALMON_DUMP_DIFF``) landed with this fix so a future
similar-shape regression can be root-caused in minutes rather than
hours.

Harness schema notes
====================

Swept shape dim is ``N`` (the number of *output* elements per row,
which is half the input width).  ``M`` (row count) is fixed via
``harness_constants`` — the harness only sweeps one named dim today.
``BLOCK_SIZE`` is 1024 so the default sweep (``N <= 1024``) hits one
program per row with no streaming.  ``ALPHA`` and ``LIMIT`` are
constexprs on the Triton side because GPT-OSS calls SwiGLU with a
fixed pair of values and the upstream kernel itself accepts
``limit`` as a constexpr.

Input range ``[-10, 10]`` is symmetric about zero and wider than
``limit = 7.0`` so we routinely exercise the clamp on both
branches; that's what makes the test sensitive to ``v_*_num_*``
miscompiles (a broken ``tl.minimum`` produces unclamped output
whose magnitude exceeds the canonical range, which shows up as
tolerance-exceeding error in the downstream ``fma``).
"""
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    A, Out,
    N,
    BLOCK_SIZE: tl.constexpr,
    ALPHA:      tl.constexpr,
    LIMIT:      tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    out_row = pid * N
    a_row = pid * (2 * N)
    mask = offs < N
    gelu   = tl.load(A + a_row + 2 * offs,     mask=mask, other=0.0)
    linear = tl.load(A + a_row + 2 * offs + 1, mask=mask, other=0.0)
    gelu = tl.minimum(gelu, LIMIT)
    linear = tl.clamp(linear, -LIMIT, LIMIT)
    s = gelu / (1.0 + tl.exp(-ALPHA * gelu))
    out = tl.fma(s, linear, s)
    tl.store(Out + out_row + offs, out, mask=mask)


RECIPES = [
    {
        "name": "swiglu_fp32",
        "kernel_fn": swiglu_kernel,
        "kernel_symbol": "swiglu_kernel",
        "signature": {
            "A":   "*fp32",
            "Out": "*fp32",
            "N":   "i32",
        },
        "constexprs": {
            "BLOCK_SIZE": 1024,
            "ALPHA":      1.702,
            "LIMIT":      7.0,
        },
        "harness_constants": {
            "M": 16,
        },
        "num_warps": 4,
        "shape_dim": "N",
        "default_shapes": [128, 256, 512, 1024],
        "grid": {
            "x": "M",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "A", "dtype": "fp32", "elems": "M * 2 * N",
             "range_lo": -10.0, "range_hi": 10.0},
        ],
        "outputs": [
            {"name": "Out", "dtype": "fp32", "elems": "M * N"},
        ],
        # Elementwise fp32 math with exp + fma.  Across any correct
        # lowering the two paths produce the same bits up to the exp
        # ULP budget (~3-4 ULPs worst-case in fp32) amplified by
        # the subsequent fma; rel tolerance 1e-5 comfortably covers
        # that.  A broken minimum / clamp lowering produces
        # out-of-range intermediates whose error shows up well
        # above tolerance.
        "comparator": {"kind": "rel", "tol": 1e-5},
    }
]
