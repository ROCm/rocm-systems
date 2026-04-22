"""Canary for the highest-risk composite obstruction pattern in
GPT-OSS: the one where multiple wave-size-sensitive primitives
compose within a single kernel (§5 of
``hotswap/docs/gpt-oss-derisking.md`` — "the MoE-router bitmatrix
pipeline carries every obstruction class GPT-OSS exhibits; highest-
risk pair, earliest correctness-validation target").

What it targets
===============

The single-primitive canaries in this directory each pin one
obstruction class:

- ``canary_dpp_reduce_fp32``         / ``canary_dpp_compound_add_fp32``
  / ``canary_dpp_compound_add_u32``  — C2-DPP
- ``canary_permlanex16_rowmax_fp32`` — C2-hard permlanex16
- ``canary_bpermute_scan_fp32``      — C2-bpermute
- ``canary_readlane_last_lane``      — C1 readlane (hand-crafted
  in `.hip` — inline asm is the only way to force a specific
  lane constant)
- ``canary_ds_swizzle_swap1``        — C2-hard ds_swizzle
  (hand-crafted in `.hip` for the same reason)

A composite like this one is valuable for a different reason: it
probes INTERACTIONS between obstruction classes that the single-
primitive canaries can't exercise in isolation.  Specifically:

- A writelane/readlane rewrite (``rewrite_cross_lane_divergent.cpp``)
  runs AFTER the DPP / permlane / bpermute lifts land in IR.  If
  the classifier's data-flow traversal misses a DPP-output ->
  readlane-input chain (because some intermediate compound DPP
  propagator was forgotten when ``isIntrinsicVGPRSafePropagator``
  was authored — see the writelane-rewrite graduation commit), the
  rewrite silently mis-fires on exactly the GPT-OSS-shaped
  reduction-plus-broadcast pattern.
- A ds_bpermute lift interacting with a DPP-emulation in the
  same basic block can trip latent ordering bugs in the raise-
  time EXEC-shadow: ds_bpermute is EXEC-sensitive (inactive lanes
  read undef) and DPP is wave-invariant; mixing them in one BB
  stresses any SPE lane-gating that's tagging the wrong sites as
  "propagator" vs "consumer".

Forcing shape
=============

One program per row, ``ROW = 32`` int32 columns, one wave per
program.  The kernel does TWO reductions-like operations in
sequence so every obstruction surfaces:

1. ``total = tl.sum(x, axis=0)``  — a full row sum.  Triton
   lowers this on gfx1250 as a DPP reduction tree
   (``v_add_nc_u32_dpp`` ×~4, row_shr:{8,4,2,1}) followed by a
   ``v_permlanex16_b32`` for the distance-16 exchange and a
   ``v_readlane_b32`` / broadcast for the final per-wave scalar
   result.  This fires **C1 (readlane)**, **C2-DPP (dpp add)**,
   and **C2-hard (permlanex16)** together.

2. ``cumulative = tl.cumsum(x, axis=0)``  — a prefix sum.
   Triton lowers this on gfx1250 via a ``ds_bpermute_b32``-based
   Kogge-Stone scan (see ``canary_bpermute_scan_fp32`` for the
   per-primitive story).  This fires **C2-bpermute** within the
   same BB as the sum above.

   Note: the current bpermute scan has a known Wave-size-sensitive
   miscompile on modulo-replication (``canary_bpermute_scan_fp32``
   is WRONG on every shape today; see that canary's docstring for
   the ``modrep-predicate-chain.md`` reference).  That miscompile
   on the scan dominates this canary's verdict — which is
   **the intended signal**: if we ever fix the scan, this canary
   will tell us whether the interaction with the sum's lift
   introduced any new miscompile that wasn't visible in isolation.

3. Output: ``y = total - cumulative``  — per-lane subtraction that
   keeps both reductions' results load-bearing for the output
   diff.  An unexpected "all-zero output" shape would indicate
   both reductions collapsed (neither fired) or the subtraction
   got folded away; an "all-sum output" shape means cumulative
   lifted to zero (scan dropped); an "all-cumulative output"
   shape means total lifted to zero (sum dropped).

Observed verdicts at landing
============================

The canary revealed a specific soundness gap that the single-
primitive canaries missed, and the fix for it landed in the same
commit as the canary — see the commit message for the full
timeline:

* ``native`` : gold.  Triton recipes use the native-gfx942 run as
  the reference.
* ``legacy`` : SIG6 CRASH on every shape (``7 waitcnt, 0
  exec-widened, 1 unsupported``).  Orthogonal legacy limitation,
  not a salmon concern.
* ``salmon`` : **EXIT=2** on every shape, principled refusal via
  ``c5_predicate_chain_classifier.cpp``'s phantom-lane arm.
  Stderr names the phantom-lane regime
  (``max_flat_workgroup_size=32 < target wavefront width 64``)
  plus the triggering ``icmp ugt`` against compile-time constant
  2 within (0, W_s-1=31], so triage tools can distinguish this
  refusal from the baseline MODREP refusal (which fires on any
  C5 site regardless of WG).

Canary evolution (chronological):

Phase 1 (pre-phantom-lane-rule) — salmon produced
**WRONG 960/1024** on every shape (93.75% silent miscompile).
Narrowing via diagnostic probe kernels (not landed) found:

* ``y = total - x`` (no cumsum) — salmon **match**.  Confirms
  the int32 DPP-compound-add + permlanex16 reduction path is
  clean (consistent with ``canary_dpp_compound_add_u32``
  passing in isolation).
* ``y = tl.cumsum(x, axis=0)`` (just the scan) — salmon
  **WRONG 960/1024**.  Isolates the miscompile to the
  ``tl.cumsum`` lift on the ``i32`` dtype specifically.

Phase 2 (root-cause analysis) — comparing salmon's raised IR for
``canary_bpermute_scan_fp32`` (fp32 cumsum, salmon was passing
numerically under the pre-phantom-lane WaveNative suppression)
vs the failing i32 cumsum revealed that both paths carry
``workitem.id.x()`` → ``icmp ugt/ult K, %tid`` predicate chains,
but the fp32 path's VOPD-specific lowering (fixed in commit
``bd04c268e7``) happens to materialise the predicate through
``ballot.i64`` + ``s_and_saveexec_b32`` which
``rewrite_cross_lane_divergent.cpp`` traces as wave-aware.  The
i32 path's non-VOPD ``v_cndmask_b32`` lowering bypasses that
chain and leaves the raw ``icmp`` downstream-visible.

Phase 3 (matched to the documented soundness gap) — the
per-source-lane EXEC model that ``WaveNativeProjection``
relies on to claim ``tid < K`` is wave-safe depends on target
lanes being a 1:1 image of source-kernel threads.  When the
HSACO's ``max_flat_workgroup_size`` is BELOW the target
wavefront width (32 for a ``num_warps=1`` Triton kernel vs 64
on gfx942 wave64), every launch activates spare "phantom"
target lanes via ``init_whole_wave``; their architectural
``tid`` is their hardware lane index (32..63), the source
kernel never modelled computation at those positions, and
convergent cross-lane ops (``ds_bpermute``, ``ds_swizzle``,
``permlane*``) can read the phantom lanes' unmodelled state.
This is precisely the soundness gap documented in the
``waveNative`` parameter contract of
``c5_predicate_chain_classifier.hpp`` — historically deferred
because no corpus kernel had empirically surfaced it.  THIS
canary is the evidence.

Phase 4 (fix in same commit) — the classifier's ``waveNative``
arm was narrowed to also refuse when the kernel's
``max_flat_workgroup_size`` is statically below
``targetWaveSize``.  Canary D graduates from silent WRONG to
principled EXIT=2.  ``canary_bpermute_scan_fp32`` (which had
the same phantom-lane regime but was numerically passing under
the pre-fix suppression, a coincidence — see that canary's
docstring) also graduates to principled refusal under the new
rule.

Regression contract:

* salmon ``EXIT=2`` -> ``match``  ==  somebody added a
  principled handler / rewrite for the phantom-lane case (e.g.
  a legitimate ``init_whole_wave`` alternative that models lane
  positions, or a mask-the-phantom-lanes-out lowering of
  ``ds_bpermute``).  Retire the EXIT=2 expectation; document
  the new rewrite.
* salmon ``EXIT=2`` -> ``WRONG``  ==  the phantom-lane arm of
  the classifier regressed (e.g. by losing the
  ``maxFlatWorkgroupSize`` thread-through in the raiser).
  Fixing it restores the EXIT=2 expectation.
* salmon ``EXIT=2`` -> different stderr  ==  the refusal
  diagnostic changed.  Verify the new diagnostic still names
  the phantom-lane regime or the C5 class so operators can
  bucket the refusal reason.

Harness schema notes
====================

Swept shape dim is ``N_ROWS``.  Input range is 0..15 to keep the
cumulative sum bounded at 32 * 15 = 480 (well within int32).
Comparator is bit-exact (int32 arithmetic has no ULP).
"""
import triton
import triton.language as tl


@triton.jit
def canary_bitmatrix_composite_kernel(
    x_ptr, y_ptr,
    N_ROWS,
    ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, ROW)
    x = tl.load(x_ptr + pid * ROW + offs)
    total = tl.sum(x, axis=0)
    cumulative = tl.cumsum(x, axis=0)
    y = total - cumulative
    tl.store(y_ptr + pid * ROW + offs, y)


RECIPES = [
    {
        "name": "canary_bitmatrix_composite",
        "kernel_fn": canary_bitmatrix_composite_kernel,
        "kernel_symbol": "canary_bitmatrix_composite_kernel",
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
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
