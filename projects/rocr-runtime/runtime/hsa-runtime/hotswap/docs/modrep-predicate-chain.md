# MODREP predicate-chain class

**Status.** Option O1 (loud-refuse classifier) landed; O2 (mask
rewrite) deferred indefinitely on semantic grounds; O3 / O4 not
triggered. Under `WaveNativeProjection` (the default for wave32 →
wave64 cross-widening) the class is structurally suppressed.
Under `ModuloReplicationProjection` (opt-in via
`--disable-wave-native` / `enableWaveNative=false`) the class is
loud-refused by the narrow-O1 classifier in
`transpiler/c5_predicate_chain_classifier.{hpp,cpp}`.

**Scope.** Wave-size axis: kernels compiled for a source wave
width `W_s` that reach cross-widening (`W_t > W_s`) under
modulo-replication, pass the `wave-size-translation.md §6`
obstruction classes C1–C4, raise cleanly through `raise_cli`,
and yet compute mathematically wrong values on the target
because their kernel-emitted predicates reference
`workitem.id.x()` directly without wave-size normalisation.
Orthogonal translation axes (ABI, sync, matrix, TDM) are out of
scope.

**Audience.** Salmon raiser contributors and reviewers adding /
modifying cross-widening machinery; anyone wanting to know
whether a given source kernel shape is wave-size-safe under
MODREP.

---

## 1. Problem statement

The `wave-size-translation.md §6` obstruction classes (C1–C4)
enumerate the *structural* ways modulo-replication can be wrong:
absolute lane-ID leaks (C1), wave-baked cross-lane ops (C2),
inter-replica communication via shared state (C3), and
lane-position-dependent EXEC writes (C4). Every site G1 refuses
is one of these; everything else emits modulo-replication code
that the §5 SIMT Predicated Execution (SPE) projection assumes is
wave-size-oblivious by construction.

A recipe passes G1 iff every site in its disassembly is either
(a) wave-invariant, (b) rewritable per §5.3's landed table, or
(c) pending-rewrite-recognised. §7's soundness claim reads:

> Every kernel the tool emits code for is provably wave-size-
> oblivious; every kernel it refuses is one we cannot prove safe.
> There is no third category of "the tool ran and produced wrong
> code."

That claim holds for the C1–C4 axes. It does *not* hold for a
fourth concern G1 does not audit: **the kernel's downstream use
of `workitem.id.x()` inside predicate chains that gate side
effects**. When a source-wave kernel packs its per-lane work for
`W_s` threads and uses `tid` directly as a control variable, the
raiser lifts `workitem.id.x()` through the SPE prelude unchanged
— so on the target wave the predicate evaluates on the
target-wave `tid` range rather than the source-wave `[0, W_s)`.
For kernels that use this predicate as "which lanes participate
in stage `s` of a scan" or "which lanes write which slot of an
output tile", the result is a silent miscompile that is neither
classified by G1 nor rewritten by §5.3.

This document defines the class, argues it is orthogonal to
C1–C4, enumerates the fix options, and documents the landed
design.

---

## 2. Class signature

A kernel exhibits the predicate-chain class iff all three
ingredients hold at some site:

1. **Source**: `@llvm.amdgcn.workitem.id.x()` (or a one-hop
   derivation — `add`, `sub`, `lshr`, `and` with a constant
   ≠ `W_s − 1`, phi, select, cast) is read in the raised IR.
2. **Gate**: the derived value feeds either
   - an `icmp` whose result gates a side effect (store,
     SPE-wrapped compute, scan-stage update), or
   - an address expression that feeds a store.
3. **Wave-size sensitivity**: the icmp's other operand is a
   compile-time constant `K` with `0 < K ≤ W_s − 1` (the
   lane-position-gate shape), or the address expression
   admits target-wave lanes into a branch / slot the source
   wave never would have taken.

### Canonical example — Kogge-Stone scan

Triton's `tl.cumsum` lowers to a Kogge-Stone tree whose
stage-`s` guard is `if tid >= 2^s: x += bpermute(lane_id − 2^s)`.
For wave32 source with `num_warps = 1`, `tid ∈ [0, 32)` and the
guard partitions lanes correctly per stage (stages `s = 0..4`,
`2^s ∈ {1, 2, 4, 8, 16}` — all within `(0, W_s − 1]`).

The raised IR for this shape is:

```
%tid = call i32 @llvm.amdgcn.workitem.id.x()
...
%vcmp215 = icmp ult i32 1, %tid       ; stage-1: tid > 1
%vcmp359 = icmp ult i32 3, %tid       ; stage-2: tid > 3
%vcmp482 = icmp ult i32 7, %tid       ; stage-3: tid > 7
%vcmp592 = icmp ult i32 15, %tid      ; stage-4: tid > 15
```

Each of those four icmps matches ingredients (1) + (2) + (3):
the compile-time constants (1, 3, 7, 15) are all in
`(0, W_s − 1 = 31]`. Under modulo-replication to wave64, target
lanes in the upper half of each wavefront have `tid` values the
source compile did not plan for; the predicates evaluate
differently than they would have in the source wave.

### Passing baseline — Triton bounds check

Kernels like `vecadd_f16` emit the same structural IR shape but
the icmp operand is a *dynamic* kernarg rather than a
compile-time constant:

```
%tid = call i32 @llvm.amdgcn.workitem.id.x()
%vcmp = icmp ult i32 %tid, %arg_num_elements    ; dynamic bound
br i1 %vcmp, label %store_arm, label %skip_arm
```

This matches ingredients (1) + (2) but **not** (3) under the
narrow-O1 shape — the classifier is defined to treat
dynamic-operand icmps as bounds checks, not lane-position gates.
The decision to narrow ingredient (3) to compile-time K is what
keeps baselines like `vecadd_f16`, `rope_fp32`, `corpus_add_fp32`,
`corpus_asin_fp32` green. See §5 O1's rationale for why the
broader "unmasked tid reaches any icmp" rule was rejected.

---

## 3. Why existing obstruction classes don't catch this

Each of C1–C4 checked against the canonical example:

- **C1. Absolute lane-ID leak.** Predicated on
  `v_mbcnt_hi_u32_b32`, `v_readlane_b32` / `v_writelane_b32` with
  operand ≥ `W_s`, `llvm.amdgcn.ballot`,
  `llvm.amdgcn.wavefrontsize`. None of these. `%tid` via
  `llvm.amdgcn.workitem.id.x` is structurally NOT an absolute
  lane ID (its range is `[0, WG)`, not `[0, wave)`, and the two
  coincide only for single-wave workgroups). G1 correctly clears
  the scan-shaped kernel on C1.
- **C2. Wave-baked cross-lane op.** `ds_bpermute_b32`,
  `v_permlanex16`, `ds_swizzle_b32`, DPP with wave-wide
  semantics. All landed rewrites per §5.3. A Kogge-Stone scan
  emits many `ds_bpermute_b32`; the P1 lift produces
  correctness-equivalent IR per the handler's MODREP comment.
  The class's miscompile is downstream of the cross-lane op,
  not at the op itself.
- **C3. Non-commutative atomic.** Zero in all failing kernels —
  consistent with the wider corpus's zero hit rate.
- **C4. Lane-position-dependent EXEC write.** `v_cmpx_*`,
  `s_*_saveexec_b32` against a lane-id-derived mask. The SPE
  diamond's own `s_and_saveexec` is classified safe by the
  allow-list gate (G2).

So the classifier is unanimously green — modulo-replication is
emitted — and the kernel miscompiles anyway. The class gap
this document closes is:

> Predicates on `workitem.id.x()` (or derived SSA values) that
> gate side effects — stores, addr arithmetic, scan / reduction
> predicates — are NOT audited by any existing C-class and are
> NOT rewritten by §5.3, even though their semantics change
> under MODREP when `W_t > W_s`.

---

## 4. Root cause

### 4.1 The MODREP contract

`wave-size-translation.md §5.2` states the projection formally:

> The target wave runs as `R = W_t / W_s` replicas of the source
> wave. Correctness requires every replica observes the same
> state as the source — at every observable side effect, replica
> `r` with lane `l_target = r · W_s + l_source` must compute the
> same value the source's lane `l_source` would have.

The §5.4 SPE allow-list gate + §5.3 cross-lane rewrite table
jointly ensure this for every SemOp. The SPE prelude masks
`mbcnt`-derived `lane_id` by `W_s − 1` so the active-lane gate
evaluates on the source-wave-scoped `l_source`. Each landed
rewrite (P1–P6) preserves replica-independence at its
instruction-level boundary.

### 4.2 Why `workitem.id.x()` escapes the contract

`workitem.id.x()` is NOT a SemOp the rewrite table covers — it
is an `amdgcn` intrinsic the raiser passes through verbatim to
the LLVM backend, which re-lowers it to a VGPR read of the
architectural workitem-id register. That register returns
`l_target ∈ [0, W_t · num_warps · W_s)` on the target wave,
NOT `l_source`. Any downstream use of this value in a
kernel-emitted predicate, address computation, or control-flow
decision silently reads a target-wave-scoped value.

### 4.3 Sub-cases by launch shape

The class manifests differently depending on the dispatch
`blockDim.x` the runtime hands HIP, which is determined by the
source compile's `max_flat_workgroup_size` (the salmon path
honours the source WG size via the raised HSACO's kernel
descriptor; it does not upscale to target-wave alignment):

- **Sub-case 1: `blockDim.x = num_warps × W_s` and `num_warps > 1`
  forces `blockDim.x > W_t`.** The target dispatches multiple
  wave64 wavefronts, each with all 64 lanes active and `tid`
  values in `[0, W_t)` within each wavefront. Lanes in the upper
  half of each target wavefront (indices `W_s..W_t−1`) are
  MODREP replica-1; their `tid` value exceeds `W_s` and any
  kernel-emitted predicate `tid < K` with `K ≤ W_s − 1`
  evaluates *differently* from the source's corresponding
  replica-0 lanes.
- **Sub-case 2: `blockDim.x = W_s` (single-warp kernel).** The
  target dispatches one wave64 wavefront with only the low
  `W_s` lanes in EXEC at entry. Active lanes' `tid ∈ [0, W_s)`
  and kernel predicates evaluate correctly on the active subset.
  The class does not manifest in the predicate-chain per se, but
  convergent cross-lane primitives (`ds_bpermute`,
  `update.dpp`, etc.) participate over the full physical wave
  regardless of EXEC — their gathers can reach the inactive
  upper-half lanes whose VGPR state doesn't match what the
  source compile expected, and active-lane downstream computation
  reads those values. This secondary effect is NOT the
  predicate-chain class per (1)-(2)-(3) above; it is a related
  but distinct mode covered in part by `WaveNativeProjection`'s
  `init_whole_wave` default.

### 4.4 Class summary

Every site exhibiting the class shares the three ingredients of
§2 plus one of the two sub-cases in §4.3. The class is
orthogonal to C1–C4: C1 is about `mbcnt_hi` / `ballot` leaks
(already caught); this class is about `workitem.id.x()`.

The per-instruction cross-lane rewrites (P1–P6) are *necessary*
for wave-size obliviousness but not *sufficient* — a kernel that
passes the per-instruction audit can still miscompile if its
kernel-level predicate chain is wave-size-sensitive.

---

## 5. Fix options

Four options, each with a different coverage / invasiveness
profile.

### O1. G1 classifier extension + loud refusal

Add a post-raise IR-level classifier that identifies the class
signature of §2 and refuses the kernel with a new
`CrossWavePredicateChain` diagnostic. No rewrite attempted; the
classifier's correctness depends only on a forward use-chain
walk from each `@llvm.amdgcn.workitem.id.x()` call.

**Shape (narrow-O1).** Refuse iff the icmp has at least one
operand in the tid-unmasked-reachable set AND at least one
operand that is a compile-time constant K with
`0 < K ≤ W_s − 1`. Chains that pass through `and V, K`
with `K ≤ W_s − 1` mask the tid-divergence away and do not
contribute to the unmasked set.

**Coverage.** Catches the Kogge-Stone scan shape (stage guards
`tid > 2^s − 1` with `s < log2(W_s)`). Does not catch kernels
that use dynamic-operand bounds checks (`vecadd_f16` and
friends), which is the design intent.

**Invasiveness.** ~280 LoC classifier + ~150 LoC lit fixtures
+ ~330 LoC unit tests + ~20 LoC wiring into `raiser.cpp`.

**Risk.** Sound-but-incomplete by construction: false positives
(refusing a safe kernel whose predicate happens to match the
narrow-O1 shape) are benign; false negatives (missing a
wave-size-sensitive predicate whose operand is dynamic) leave a
residual silent-miscompile class for dynamic-operand shapes,
which §6 argues is covered empirically by the
WaveNative-default suppression + end-to-end corpus regression
testing.

### O2. Predicate-chain rewrite

A post-raise pass that rewrites every `@llvm.amdgcn.workitem.id.x()`
value in a predicate chain (icmp operand or GEP index feeding a
store) to `workitem.id.x() AND (W_s − 1)` — i.e., the
source-wave-scoped lane id. The rewrite would run before the
narrow-O1 classifier and would convert C5-refused kernels to
correctness-preserving lifts.

**Coverage.** Would fix sub-case 1 (multi-warp kernels where
replica-1 lanes' `tid` exceeds `W_s`, and the mask collapses
them back onto `[0, W_s)`). Does NOT fix sub-case 2: on
sub-case 2, active lanes already have `tid ∈ [0, W_s)` so the
mask is a no-op on the active set; the cross-lane-primitive
inactive-lane-leak is outside the rewrite's scope.

**Invasiveness.** ~500 LoC rewrite pass + use-chain classifier
+ ~300 LoC lit tests. Structurally analogous to §5.6.3's
`rewrite_cross_lane_divergent.cpp`.

**Risk.** False-positive rewrites that change the semantic of a
kernel that intentionally uses `[0, W_t)` scope. The mask also
imposes a hidden invariant: if `num_warps × W_s` crosses a
power-of-2 boundary the mask's arithmetic is wrong. More
fundamentally, the mask is *semantically incorrect* for
`num_warps > 1` multi-warp kernels where the source compile
expects source wave 1's lanes to see `tid ≥ W_s` (e.g. for
per-wave partitioning) — masking those lanes' `tid` by `W_s − 1`
forces their predicate to evaluate on the wrong range. This is
the primary reason O2 is deferred; see §6.

### O3. ThreadLoopProjection for scan-shaped patterns

The `ThreadLoopProjection` escape hatch: lower the kernel as a
target-wave-scoped serial loop over the `R = W_t / W_s` source
replicas. For each replica `r ∈ [0, R)`, run the kernel body
once with `workitem.id.x()` remapped to `(r · W_s) + lane_mod`.
Cost: `1/R` throughput (2× slowdown for wave32 → wave64).

**Coverage.** Strictly more than O2 — handles anything that can
be expressed as "run the source kernel R times, then reconcile",
including sub-case 2's inactive-lane-leak (each replica runs
serially with full EXEC, so inactive-lane VGPRs never participate
in a gather).

**Invasiveness.** The largest of the four.
`ThreadLoopProjection` is a skeleton in
`wave_projection.hpp`; every override `report_fatal_error`s
pending a real implementation. Multi-week piece of work.

**Risk.** High implementation cost for no additional corpus
coverage today. Reserved for a future recipe that demonstrably
needs it.

### O4. Harness-side constraint on `num_warps`

Add a check in `compare_correctness`'s AOT-compile flow
(`aot_compile.py`) that refuses to compile a recipe for
gfx1250 unless `num_warps × W_s ≥ W_t`. Ensures the target WG
is at least one full target wave and MODREP's replica factor
is 1.

**Coverage.** Narrow. Would force
`canary_bpermute_scan_fp32` to re-compile with `num_warps = 2`
to target wave64 — but does NOT fix kernels in
`scope_discovery/kernels/` captured from upstream corpora
where `num_warps` is set by the original author.

**Invasiveness.** Trivial — a single assertion. ~10 LoC.

**Risk.** Doesn't address the class; treating it as "the fix"
would leave a trap for recipes that happen to satisfy the
assertion today but break on a `num_warps` bump.

---

## 6. Landed design

### 6.1 Narrow-O1 classifier

Implementation:
`transpiler/c5_predicate_chain_classifier.{hpp,cpp}`. Wired into
`raiser.cpp` Phase 6.6 (post-mem2reg). Emits
`RaiseFailure::crossWavePredicateChain` on refusal, with
`ObstructionKind::WorkitemIdPredicateChain` +
`RaiseFailureReason::CrossWavePredicateChain`.

Two-pass walker:

- **Pass 1**: forward-walk from each `@llvm.amdgcn.workitem.id.x()`
  call, tagging every tid-reachable value as `unmaskedVisited`
  (walk did not cross an `and V, K ≤ W_s − 1` mask) or
  `maskedVisited` (walk did). A value can appear in both sets
  when it joins a masked arm and an unmasked arm via a phi.
  Propagators: casts, phi, select, gep, freeze, insert/extract
  element/value, shufflevector, integer binops (add/sub/mul/shl/
  lshr/ashr/or/xor/and — including a non-masking `and`), unary
  ops, and the numeric intrinsics
  `llvm.{smin,smax,umin,umax,abs,ctlz,cttz,ctpop,bitreverse,
  bswap,fshl,fshr,sadd_sat,ssub_sat,uadd_sat,usub_sat}`.
- **Pass 2**: scan every `icmp` in the function. Refuse iff the
  icmp has at least one operand in `unmaskedVisited` AND at
  least one operand that is a compile-time constant K with
  `0 < K ≤ W_s − 1`.

**Direction gate.** The classifier no-ops when
`W_t ≤ W_s` (same-wave / narrowing have no replica-1; the class
cannot manifest).

**Projection gate (`waveNative` parameter).** When invoked with
`waveNative = true`, refusal is suppressed but the walk still
runs and populates `observedSites` for debugging. Rationale:
`WaveNativeProjection`'s `init_whole_wave` + per-source-lane
modeled-EXEC model means each target lane's `tid` IS its own
source-wave tid (for num_warps > 1, target wavefront 0's lanes
0..31 run source wave 0 with tids 0..31, lanes 32..63 run
source wave 1 with tids 32..63); the MODREP "replica-1 shares
source wave 0's EXEC" assumption — the exact rationale the
refusal exists for — does not apply.

The suppression is a projection-model statement, not a universal
safety proof. If a launch configuration ever materialises with
"phantom" target lanes outside the source's WG index space, the
suppression is permissive — the classifier logs the observed
sites via `LLVM_DEBUG` under the `wave-projection` debug
category so regressions have an attribution trail.

**Regression guards.** See §7.

### 6.2 O2 deferred indefinitely

The proposed `tid AND (W_s − 1)` mask rewrite is semantically
incorrect for the multi-warp (`num_warps > 1`) shape and a
structural no-op for single-warp sub-case 2 (active lanes
already have `tid ∈ [0, W_s)`). No recipe in the current corpus
would be helped by it. Reopening criteria in §6.5.

### 6.3 `WaveNativeProjection` as cross-widening default

`WaveNativeProjection` is the default for wave32 → wave64
cross-widening (opt out via `--disable-wave-native` on
`raise_cli` or `enableWaveNative=false` on programmatic
callers). Under WaveNative:

- `@llvm.amdgcn.init_whole_wave` at kernel entry forces
  hardware `EXEC = -1` kernel-wide; the original per-lane
  active mask is saved into the (widened) EXEC alloca.
- Each target lane has its own modeled-EXEC bit at source
  width. For `num_warps > 1`, target wavefront 0's lanes
  0..31 model source wave 0 with their own EXEC, lanes 32..63
  model source wave 1 with its own EXEC. No replica sharing.
- Cross-lane primitives (`ds_bpermute`, WMMA collectives) run
  with hardware EXEC = -1 so every lane participates,
  eliminating sub-case 2's inactive-lane-leak class.

The predicate-chain class the narrow-O1 classifier detects is
MODREP-specific by construction — under WaveNative the replicas
don't exist and each target lane's `tid` is its own source-wave
`tid`. The classifier short-circuits the refusal under
WaveNative by design (§6.1's `waveNative` gate).

Rationale for the default choice:

- **WMMA / matrix-kernel correctness.** `wave-size-translation.md
  §5.6.1` documents the Wave64-collective correctness invariant:
  the WMMA → MFMA lowering and every convergent cross-lane
  primitive issues a single Wave64 intrinsic that reads all 64
  lanes regardless of EXEC; under partial-wave hardware EXEC
  those lanes' VGPR writes don't fire and downstream readers
  see undefined data. `init_whole_wave` + the kernel-wide
  HW EXEC = -1 + per-lane modeled EXEC is the documented
  solution.
- **`num_warps > 1` correctness.** MODREP's "replicas of source
  wave 0" model cannot express genuinely distinct source-wave
  EXEC registers. WaveNative's per-lane modeled EXEC does.

MODREP code is fully retained:

1. Same-wave and narrowing directions bypass WaveNative via
   its constructor's direction gate; MODREP is the only
   projection emitted there.
2. Explicit opt-out via `--disable-wave-native` /
   `enableWaveNative=false` is preserved for lit fixtures that
   pin MODREP-shape IR invariants
   (`c5_predicate_chain_tid`, `v_cmp_cndmask_sgpr_scalar_clobber`)
   and for operators debugging projection-specific behaviour.
3. The narrow-O1 classifier's refusal path runs under MODREP
   and short-circuits under WaveNative per §6.1.

### 6.4 Scope note: orthogonal fixes for canary / corpus_layernorm

`canary_bpermute_scan_fp32` and `corpus_layernorm_fp32` — the
two recipes originally expected to exhibit the predicate-chain
class as their end-to-end miscompile — have MATCH verdicts
today under the WaveNative default. Their mechanism, however,
turned out to be entirely orthogonal to this document's class:

A silent fallback in `handle_vopd.cpp::v_cndmask_b32` hardcoded
`loadVCC()` regardless of the instruction's explicit scalar
operand. gfx1250 VOPD's `v_dual_cndmask_b32` encoding can bind
the scalar operand to either `vcc_lo` OR an arbitrary `sN`; the
pre-fix handler routed both halves through VCC unconditionally,
collapsing distinct scalar conditions onto whatever last wrote
VCC. Triton's Kogge-Stone scan at distances 8 / 16 emits
paired `v_dual_cndmask_b32 ..., s0 :: v_dual_cndmask_b32 ...,
vcc_lo` where s0 is the stage's selector-advance guard and
vcc_lo is the value-update guard; the pre-fix handler mis-wired
the selector-advance to use the value-update guard's predicate.

The fix (`handle_vopd.cpp::v_cndmask_b32` SGPR-aware routing,
mirroring the `V_CNDMASK_B32_e64` handler in
`handle_valu_vop3p.cpp`) + a sibling fix extending the same
routing pattern to the six `V_{ADD,SUB,SUBREV}_CO_(CI_)U32`
carry-chain handlers in `handle_valu.cpp` closes the VOPD /
VOP3B SGPR-operand class independently of the predicate-chain
class this document scopes. See the lit fixtures in §7.2 for
the IR-level regression fences and the
`compare_correctness.cpp` Triton-corpus sweep for the
end-to-end verdicts.

The narrow-O1 classifier (§6.1) still serves as the MODREP-path
regression guard for the predicate-chain class — on a future
kernel that genuinely hits the class under MODREP, the
classifier loud-refuses rather than silently miscompiling.

### 6.5 Deferred options — reopening criteria

O2 / O3 / O4 reopen if:

- **A.** A new corpus recipe appears whose `workitem.id.x()`
  flows into an `icmp` against a compile-time constant in
  `(0, W_s − 1]` AND whose downstream miscompile mechanism is
  demonstrably the predicate-chain class (not a VOPD-cndmask /
  carry-chain SGPR-operand issue like §6.4, and not an
  inactive-lane-leak from sub-case 2). Triggers an O2 redesign
  with a real mechanism trace of the new recipe.
- **B.** A corpus recipe is shown to need the
  `ThreadLoopProjection` escape hatch (O3) — typically a kernel
  that legitimately requires `[0, W_t)` scope for some
  predicate and cannot be expressed with an AND-mask.
- **C.** A scope-discovery / GPT-OSS kernel surfaces a fix-up
  need beyond either O1's refusal or an eventual O2 mask — e.g.
  a mixed bitmatrix / scan / writelane pattern. Triggers an
  O3 scope paper or an entirely new class doc.

---

## 7. Test surface

### 7.1 Predicate-chain class (O1) regression guards

- **`lit_tests/c5_predicate_chain_tid/`** — paired RUN lines
  pin both projection paths:
  - `--disable-wave-native` MODREP path: expects loud refusal
    with `cross-wave-predicate-chain` diagnostic, the
    `compile-time constant 16` detail string, and
    `Class 5 / WorkitemIdPredicateChain` per-site trace.
  - WaveNative default path: expects clean raise + IR
    emission (the classifier's `waveNative` gate suppresses
    refusal).
- **`lit_tests/c5_predicate_chain_masked/`** — non-refusal
  sibling. Same K=16 icmp preceded by `and tid, 31` (= W_s−1);
  expects raise_cli success and the `and i32 ..., 31` anchor
  in the emitted IR.
- **`tests/c5_predicate_chain_test.cpp`** — `C5PredicateChain.*`
  gtest suite (13 cases). Audits the classifier on synthesised
  IR in isolation from the MC-level pipeline:
  `TidDirectSmallConstRefuses`, `TidMaskedBeforeCmpAccepts`,
  `TidSmallConstZeroAccepts`, `TidLargeConstAccepts`,
  `TidDynamicCmpAccepts`, `SameWaveDirectionGate`,
  `NarrowingDirectionGate`, `WaveNativeProjectionGate`,
  `NoCallsIsNoOp`, `PhiPropagatesTidDerivation`,
  `MaskedPhiThroughUnmaskedArmRefuses`,
  `CrossSubtreeMaskedVsUnmaskedAccepts` (pins that the
  `icmp unmasked-tid, masked-tid` cross-subtree shape is NOT
  refused — that widening was explored and falsified),
  `IntrinsicPropagatorRefuses` (pins the numeric-intrinsic
  propagator audit via `@llvm.umin`).
- **`lit_tests/cross_wave_warn/`** and **`lit_tests/v_cmpx_ballot/`**
  — updated to use `K ≥ W_s` constants (64 / 96) so their
  cross-wave EXEC-writer / ballot-routing coverage is not
  conflated with the C5 predicate-chain coverage. Pre-update
  those fixtures exercised K ≤ W_s − 1 predicates which the
  narrow-O1 classifier correctly refuses as latent-bug kernels.

### 7.2 Orthogonal VOPD / VOP3B SGPR-operand fixes

Separate class from §6.1, documented in §6.4 and cross-referenced
here because they close `canary_bpermute_scan_fp32` /
`corpus_layernorm_fp32` end-to-end:

- **`lit_tests/v_dual_cndmask_b32_sgpr_cond/`** — regression
  fence for the VOPD `v_dual_cndmask_b32` SGPR-condition fix.
  Inline asm emits a paired `v_dual_cndmask_b32 ..., s0 ::
  v_dual_cndmask_b32 ..., vcc_lo` with DIFFERENT scalar
  conditions on the two halves; FileCheck expects the two
  `vopd_cndmask` selects to consume different `i1` SSA values
  (the SGPR-path cmp and the VCC-path cmp). A regression that
  re-introduces hardcoded-VCC collapses the two conditions onto
  the same i1 and fails the CHECK pattern.
- **`lit_tests/v_add_co_u32_sgpr_carry/`** — regression fence
  for the carry-chain (`V_{ADD,SUB,SUBREV}_CO_(CI_)U32`) SGPR-
  operand fix. Inline asm emits `v_add_co_u32 ..., s0, ... ::
  v_add_co_ci_u32_e64 ..., s0, ..., s0` — the canonical 64-bit
  address-chain shape under VCC pressure. FileCheck expects the
  carry-out ballots through `amdgcn.ballot.i64` into the SGPR
  path, the second add's carry-in forwards the SAME `i1` SSA
  value the first add produced (fresh V_CMP shadow lookup — no
  `load i1, ptr %vcc` in between). Negative CHECK-NOT pin on
  `load i1, ptr %vcc` catches hardcoded-VCC regressions.

### 7.3 End-to-end regression surface

`compare_correctness`'s salmon path on the Triton corpus (run
against `gfx1250 → gfx942` cross-widening, WaveNative default):

- `canary_bpermute_scan_fp32` — MATCH 4/4 under WaveNative
  default; loud-refused 4/4 under `--disable-wave-native`
  (MODREP + narrow-O1).
- `swiglu_fp32`, `rmsnorm_fp32`, `corpus_layernorm_fp32` —
  MATCH 4/4.
- 7 baselines (`vecadd_f16`, `rope_fp32`, `corpus_add_fp32`,
  `corpus_asin_fp32`, `canary_dpp_compound_add_fp32`,
  `canary_dpp_reduce_fp32`, `canary_permlanex16_rowmax_fp32`)
  — MATCH 4/4.
- `corpus_softmax_fp32` — EXIT=2 via the writelane safety net
  (`wave-size-translation.md §5.6.3`); orthogonal class.

`ctest` + `llvm-lit`: no regressions introduced by the landed
design. `BatchRaise.AiterGfx950`: raise rate unchanged.
`Gfx1250Gpu.Matmul128x128_1tile`: unchanged under WaveNative.

### 7.4 Repro

Single-recipe salmon verdict:

```bash
cd $TRANSPILER/tools/compare_correctness
LD_PRELOAD=./libsalmon_intercept.so \
LD_LIBRARY_PATH=$ROCR_BUILD/rocr/lib \
  ./compare_correctness --recipe=canary_bpermute_scan_fp32
```

Raise a single kernel to IR under each projection:

```bash
# WaveNative default
$TRANSPILER/build/raise_cli <kernel>.gfx1250.co \
  --isa=gfx1250 --target-isa=gfx942 \
  --emit-ir=<kernel_name>

# MODREP (loud-refuses predicate-chain shapes)
$TRANSPILER/build/raise_cli <kernel>.gfx1250.co \
  --isa=gfx1250 --target-isa=gfx942 \
  --disable-wave-native \
  --emit-ir=<kernel_name>
```

---

## 8. References

- [`wave-size-translation.md`](wave-size-translation.md) — the
  canonical wave-size axis spec. Specifically §5.2
  (modulo-replication definition), §5.3 (cross-lane rewrite
  table), §5.6.1 (hardware vs modeled EXEC under
  cross-widening), §5.6.3 (writelane/readlane post-raise
  rewrite), §6 (obstruction classes C1–C4), §7 (decision
  procedure), §8 (principled fail-loudly gates).
- [`target-capability-dispatch.md`](target-capability-dispatch.md)
  §5 — `SemOpAttrs` extension pattern (relevant if a future
  revision hangs a `predicateChainLaneScoped` bit on tid-
  emitting SemOps).
- `transpiler/c5_predicate_chain_classifier.{hpp,cpp}` — the
  narrow-O1 classifier. Header carries the full design contract
  and the `waveNative` parameter docstring.
- `transpiler/handle_vopd.cpp` — the VOPD `v_cndmask_b32`
  SGPR-condition handler (§6.4).
- `transpiler/handle_valu.cpp` — the six carry-chain handlers
  with `readCarryInI1` / `writeCarryOutI1` helpers (§6.4).
- `transpiler/handle_valu_vop3p.cpp` — the non-VOPD
  `V_CNDMASK_B32` handler; the reference implementation whose
  SGPR-aware routing the VOPD + carry-chain fixes mirror.
- `transpiler/rewrite_cross_lane_divergent.cpp` — the
  writelane/readlane post-raise rewrite. Structural template
  for a future O2 landing.

---

## 9. Open questions

1. Is there evidence of a GPT-OSS / AITER kernel in the broader
   corpus (beyond compare_correctness's Triton set) that
   genuinely hits the predicate-chain class — i.e., matches the
   §2 signature AND miscompiles in a way not explained by the
   VOPD-cndmask / carry-chain class (§6.4) or the
   inactive-lane-leak sub-case 2 mode? If so, it's the
   reopening trigger A in §6.5 and motivates actually
   implementing O2.
2. Should `target-capability-dispatch.md`'s `SemOpAttrs`
   extension grow a `predicateChainLaneScoped` bit on
   `workitem.id.x`-emitting SemOps, so the classifier's C5
   pass can enumerate emission sites without hand-maintaining
   a list? This is the "hang new attrs off `SemOpAttrs`"
   pattern the README index section recommends.
3. Is the narrow-O1 classifier's constant-K range
   `(0, W_s − 1]` the right bound, or should it extend to
   `[1, W_t − 1]` to catch predicates that partition the
   target wave (e.g., `tid < W_t/2` half-wave broadcasts)?
   Current evidence is insufficient — no corpus kernel emits
   this shape — and the over-approximation discipline argues
   for the tighter bound. If a corpus kernel surfaces with
   `K ∈ (W_s − 1, W_t − 1]`, widen the bound to match.
