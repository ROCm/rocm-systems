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

The canary revealed a specific new bug that the single-primitive
canaries missed:

* ``native`` : gold.  Triton recipes use the native-gfx942 run as
  the reference.
* ``legacy`` : SIG6 CRASH on every shape (``7 waitcnt, 0
  exec-widened, 1 unsupported``).  Orthogonal legacy limitation,
  not a salmon concern.
* ``salmon`` : **WRONG 960/1024** on every shape (93.75%
  mismatch).

Narrowing the composite:

* ``probe_sum_only`` (same structure, ``y = total - x`` without
  the ``tl.cumsum``) — salmon **match**.  Confirms the int32
  DPP-compound-add + permlanex16 reduction path is clean
  (consistent with ``canary_dpp_compound_add_u32`` passing in
  isolation).
* ``probe_cumsum_i32_raw`` (just ``y = tl.cumsum(x, axis=0)``)
  — salmon **WRONG 960/1024**.  Isolates the miscompile to the
  ``tl.cumsum`` lift on the ``i32`` dtype specifically.

Root-cause mapping:

The ``canary_bpermute_scan_fp32`` companion uses ``tl.cumsum`` on
fp32 and passes, so the bug is not in the bpermute scan itself
at the IR level — it's in how salmon's rewrite passes handle
the Triton-emitted predicate chain for the i32 codegen lowering.

Comparing salmon's raised IR for both dtypes:

* fp32 scan: predicates materialise through ``ballot.i64`` +
  ``s_and_saveexec_b32`` (a VOPD-lowering byproduct) and the
  ``rewrite_cross_lane_divergent.cpp`` classifier correctly
  traces the ``ballot`` → ``saveexec`` chain as wave-aware.
  Salmon's C2 path uses that to emit per-source-wave ds_bpermute
  selectors.
* i32 scan: predicates come through as direct ``select i1
  %vcmp, ...`` where ``%vcmp = icmp ugt/ult i32 K, %tid`` — and
  ``%tid`` is ``call @llvm.amdgcn.workitem.id.x()``, a wave-size-
  sensitive value that does NOT go through a ballot/saveexec
  normalisation step.  This is exactly the C5 "Wave-size-
  sensitive predicate chain" class documented in
  ``hotswap/docs/modrep-predicate-chain.md``: under modulo-
  replication, upper-half target lanes (tid 32..63) fall outside
  the predicate's intended source-wave-relative range and read
  or skip ds_bpermute incorrectly.

Disposition — **not a fix to pursue in this canary's commit**:

The root cause lives in the C5 predicate-chain classifier /
rewrite in salmon's modulo-replication projection, which another
agent is already fixing end-to-end per
``modrep-predicate-chain.md``.  That fix will rewrite
``%tid``-derived predicate chains at their source (the ballot
materialisation that fp32 gets by accident will become
unconditional), and this canary will then graduate to ``match``.

Landing this canary as WRONG-on-every-shape turns the composite
failure into a regression gate that the C5 fix will close.  If
the C5 fix ever lands and this canary STILL shows WRONG, it
surfaces a residual interaction bug specific to sum+cumsum that
the single-primitive C5 repair didn't catch — exactly the
"interaction bug that single-primitive canaries missed"
contingency this composite was designed for.

Regression contract:

* salmon `WRONG 960/1024` -> `match`  ==  the C5 predicate-chain
  fix landed AND composition is clean; retire the WRONG
  expectation.
* salmon `WRONG 960/1024` -> different mismatch count  ==
  interaction with sum's lift path shifted; investigate.
* salmon `WRONG 960/1024` -> `EXIT=2`  ==  classifier upgraded
  from silent-miscompile to principled-refusal (e.g. the C5 fix
  emits an explicit refusal for the pattern rather than
  rewriting it); investigate whether the refusal is the intended
  graduation shape.

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
