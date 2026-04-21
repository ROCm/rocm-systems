# SGPR Wave-Mask Translation — V_CMP → SGPR → V_CNDMASK under Cross-Widening

> **Status:** design accepted; **intra-BB shadow** implementation landing
> now (first PR). The **widened-SGPR-storage** alternative is documented
> in §4 for future reference; it is NOT implemented and not scheduled.
>
> **Scope:** wave32 source → wave64 target cross-widening (gfx1250 →
> gfx942 / gfx950). Same-wave and narrowing directions (wave64 → wave32
> — currently fatal-errored by `ModuloReplicationProjection::ballotI1ToWidth`)
> are not affected by this design.

---

## 1. Problem in one paragraph

Source-ISA `v_cmp_*_e64 sDST, a, b` writes a per-lane compare mask into
a single source-width SGPR. On wave32 source that SGPR is physically 32
bits. Under cross-widening to wave64 target hardware the ballot runs on
64 target lanes and produces 64 correct per-lane compare bits — but a
single SGPR still holds only 32 bits, so the writer truncates to 32
bits before the store. Target lanes 32..63's compare results are
destroyed at the write, before any consumer reads. A subsequent
`v_cndmask_b32_e64 vDST, a, b, sN` (the canonical libdevice-math branch
shape and the shape every Triton e64-across-BB epilogue lowers to) then
reads a mask whose upper half is either all-zero (modulo-replication's
`zext`) or a replication of the lower half (wave-native's `(v << 32) |
v`), and per-lane selection on those lanes silently misbehaves. On
`corpus_asin_fp32` the observable cost is ~44% of output elements wrong
with `|err|` up to ~π/2. No consumer-side cleverness can recover bits
that were never stored.

## 2. Where the information is lost

In `handle_valu_vcmp.cpp`'s V_CMP → SGPR branch:

```cpp
Type *sourceWidth = ctx.projection.sourceWaveMaskTy();      // i32 (wave32)
Value *mask = ctx.projection.ballotI1ToWidth(ctx.B, cmp,
                                              sourceWidth, "vcmp_ballot");
ctx.writeRegExecWidth(d, mask);
```

`ballotI1ToWidth(cmp, i32)` in both `ModuloReplicationProjection` and
`WaveNativeProjection` collapses to the same narrowing step when the
source width is smaller than the target hardware wave:

```cpp
if (wantedBits < waveBits)
  return B.CreateTrunc(waveMask, resultTy, name + "_trunc");  // <-- lossy
```

The comment there already acknowledges this as a residual:

> "…a documented residual lossy path that the obstruction classifier
> (`wave_size_obstruction.cpp`) still has to refuse downstream for
> kernels that consume the narrowed mask as a per-lane wave mask."

Either the classifier refuses such kernels (declined — loses corpus
coverage), or the data is preserved somewhere the consumer can still
reach it.

## 3. Design space

Three strategies were surveyed. The comparison is below; §3.1 picks one
and §4 documents the alternative that is deliberately NOT scheduled so
a future reader can pick it up without re-running this analysis.

| strategy | correctness domain | invasiveness | composes with future work |
|---|---|---|---|
| **3.1 Side-channel shadow** (chosen) | intra-BB V_CMP → V_CNDMASK without scalar interference — the dominant corpus pattern | localised: ~40 LOC touching 5 files, strictly additive | yes; becomes redundant once §4 or a future reaching-definitions pass lands |
| **3.2 Obstruction classifier refusal** (declined — conversation history) | all cases where §3.1 or §4 would fix it | ~50 LOC in `wave_size_obstruction.cpp` | parallel second-line defence regardless of which fix lands |
| **4 Widened SGPR storage** (not scheduled) | all V_CMP → SGPR → wave-mask-consumer patterns, cross-BB, scalar-interleaved, other consumers | systemic: reg-file model change, every SGPR handler audited | supersedes §3.1; orthogonal to §3.2 |

### 3.1 Chosen approach — intra-BB per-lane-i1 shadow

**Insight.** At the moment the V_CMP handler emits the narrow-width
ballot store, it still has the per-lane `i1` SSA value from its
`CreateFCmp`. That `i1` carries the full per-lane compare result for
every target lane. If a matching V_CNDMASK consumer could find it, no
ballot round-trip is needed.

**Mechanism.** Add one `DenseMap<int, Value*>` to `RaiseContext`:

```cpp
// For each SGPR baseIdx, the per-lane `i1` produced by the most recent
// V_CMP_*_e64 writer to that SGPR in the current basic block. Cleared
// on every BB entry. Invalidated whenever the SGPR is written by any
// scalar-semantics path (s_mov_b32, s_or_b32, …). A cache, never
// load-bearing for correctness: absence just routes the consumer
// through the existing `extractLaneBitFromWaveMask` fallback.
DenseMap<int, Value*> lastSgprWaveMaskI1;
```

**Writer side (handle_valu_vcmp.cpp).** After the existing narrow
ballot / `writeRegExecWidth` sequence for a V_CMP writing an SGPR
destination, *also* record the per-lane `i1`:

```cpp
if (sop == SemOp::V_CMP && d.kind == ParsedReg::SGPR)
  ctx.recordSgprWaveMaskI1(d.baseIdx, cmp);
```

The narrow store is preserved unchanged — scalar consumers of `sN`
(e.g. `s_mov_b32 sM, sN`) still see the source-semantic 32-bit value.
The shadow is strictly additive.

**Consumer side (handle_valu_vop3p.cpp, V_CNDMASK_B32 SGPR arm).**
Prefer the cached `i1`; fall back to the existing per-lane extract
otherwise:

```cpp
if (Value *freshCmp = ctx.lookupSgprWaveMaskI1(condReg.baseIdx)) {
  cond = freshCmp;
} else {
  Value *condVal = ctx.isa.isWave32()
                       ? ctx.regs.loadSGPR32(ctx.B, condReg.baseIdx)
                       : ctx.regs.loadSGPR64(ctx.B, condReg.baseIdx);
  cond = ctx.projection.extractLaneBitFromWaveMask(ctx.B, condVal);
}
```

**Invalidation.** Two sources:

1. Scalar write to the SGPR — wire a `onSgprScalarWritten(baseIdx)`
   callback from `AllocaRegFile::writeReg32 / writeReg64` on the SGPR
   path, analogous to the existing `onExecWritten` callback. The
   callback invalidates `map[baseIdx]`. Any new SGPR-writing handler
   that routes through the public `writeReg32/64` API picks up
   invalidation automatically.
2. BB transition — piggyback on the existing BB-boundary hook in
   `raiser.cpp` next to `ctx.vgprMSBs = 0;`, calling
   `ctx.clearSgprWaveMaskShadow()`.

Note `writeRegExecWidth` (V_CMP, V_CMPX, `s_*saveexec_*`) does *not*
invalidate — it is the path that *produces* fresh cache entries (V_CMP)
or operates on EXEC (V_CMPX / SAVEEXEC) rather than on an SGPR in its
scalar role.

**Correctness argument.** Three invariants:

- **I1 — Additive, not replacing.** Narrow store and extract reader
  are both preserved. Absence of the cache routes to the existing
  (known-semantics, lossy-under-cross-widening, correct-everywhere-else)
  path. Nothing the old code did is regressed.
- **I2 — Same-BB, SSA-monotonic.** Within a BB, the raiser processes
  instructions in program order. The SSA value in the cache *is* the
  last V_CMP's `i1` with no intervening write to the same SGPR.
- **I3 — Any interference defeats the cache.** Scalar writes invalidate
  via the reg-file callback; subsequent V_CMP writers overwrite
  `map[N]` (last-writer wins); BB transition clears everything.

No case where the consumer reads a stale or mismatched `i1`.

**Concrete impact on corpus_asin_fp32.** Each of 8 unroll blocks
contains two `V_CMP_*_e64 s6, |vN|, 0.5` → `V_CNDMASK_B32_e64 ..., s6`
pairs with no intervening write to `s6`. All 16 pairs in the kernel
are intra-BB and cache-covered. Expected post-fix outcome: asin
bit-exact against the CPU reference (or within fp32 polynomial rounding
noise, ≪ 1e-5 tolerance).

### 3.2 Rejected at conversation time — classifier refusal

Previously proposed in this project: detect `V_CMP → SGPR → consumed-as-
per-lane-mask` patterns in `wave_size_obstruction.cpp` and refuse the
kernel. Declined because it converts a correctness regression into a
coverage regression, which the Triton / AITER corpora cannot afford.
Can still land as a second-line defence behind §3.1, covering the
residual cases (cross-BB, scalar-interleaved, other consumers) that the
shadow does not reach.

## 4. Not scheduled — widened-SGPR-storage (full-correctness alternative)

Recorded here so a future investigator does not re-run the trade-off
analysis. This IS the correct answer if §3.1 proves insufficient.

### 4.1 Idea

Back every SGPR whose role includes "wave mask under cross-widening"
with a target-wave-width alloca (i64 on wave32 → wave64), not the
source-semantic i32. V_CMP's ballot stores the full 64 bits; V_CNDMASK
reads the full 64 bits; `extractLaneBitFromWaveMask` indexes into a
lossless mask.

### 4.2 Variants

**4.2.1 Widen-all.** Every SGPR becomes i64 under cross-widening.

- Simple implementation.
- Scalar reads of `sN` take the low 32 bits via explicit trunc.
- Scalar writers (`s_mov_b32 sM, sN`) must decide what to do with the
  "extra" 32 bits of sM: clear? Preserve from a prior wave-mask write?
  Source semantics do not define these bits, so any choice is an
  invention — and the choice is now part of every lift's output.
- Wastes alloca space (~2×) regardless of how the SGPR is actually used.

**4.2.2 Role-classified widening.** A pre-lift dataflow pass scans the
instruction stream and marks each SGPR number as SCALAR, WAVE_MASK, or
BOTH based on its writers and readers. Widen only WAVE_MASK and BOTH.

- Smaller alloca cost.
- Classification is non-trivial: SGPR reuse across unrelated roles is
  common (compilers spill scalar constants into the same SGPRs they
  used for masks a few instructions earlier — the asin kernel reuses
  `s6` as both a wave mask AND as a polynomial-coefficient scalar).
- A BOTH classification still has to solve the "what goes in the extra
  32 bits after a scalar write" problem of 4.2.1.

**4.2.3 Per-write-site role tagging.** Every SGPR write annotates its
destination with the write's semantic role. The alloca is a sum type:
it carries *either* a 32-bit scalar OR a 64-bit wave mask, never both;
a read of the other role triggers a lossy projection.

- Most precise.
- Requires a runtime/IR-time tag on every SGPR write, and type-dispatch
  on every read. Effectively reinvents a tiny type system inside the
  reg file.

### 4.3 Required machinery regardless of variant

- **Reg-file surgery.** `AllocaRegFile::sgpr[]` grows an alloca-width
  field per SGPR, or switches to a sum type. `storeSGPR32/64`,
  `loadSGPR32/64`, the pair helpers (`storeSGPR64` splitting across
  two `sgpr[i]` / `sgpr[i+1]`), and `writeRegExecWidth` / `readOpExec
  Width` all learn the new model.
- **SGPR-pair / subregister overlap.** `s[4:5]` currently maps to two
  i32 allocas; under widening, `s4` alone is i64, and `s[4:5]` is a
  2-element vector of i64 or an i128 depending on the pairing chosen.
  Every place that does `baseIdx + 1` on SGPR pairs (`writeReg64`,
  kernarg unpacking in `handle_smem.cpp`, `s_load_b128` decomposition,
  …) has to stay consistent with whatever the pair model becomes.
- **Scalar op semantics on widened SGPRs.** `s_mov_b32`, `s_and_b32`,
  `s_or_b32`, `s_xor_b32`, the entire SOP family — each must pick a
  rule for what happens to the high 32 bits of a widened destination.
  Source semantics are silent on those bits; we invent a rule, and the
  rule has to be the same across every handler to not miscompile.
  Candidates: "clear on every scalar write" (forces wave-mask role to
  go through a dedicated writer), "preserve unchanged" (lets scalar
  moves propagate wave-mask bits they never read), "broadcast the low
  32 into the high 32" (matches wave-native's `(v << 32) | v` read-side
  widening, but pretends the source semantics said so).
- **Kernarg / memory ABI.** Kernarg SGPRs start life as 32-bit values
  read out of a kernarg segment; they stay scalars. The widening must
  not change the on-memory representation or the kernel-descriptor
  layout; only the in-alloca representation changes.
- **Obstruction-classifier interaction.** Many of the "Class 4" lane-
  position-dependent refusals (`CmpxFromLaneId`, `SaveExecFromLaneId`)
  operate on the same SGPR dataflow. Widened storage potentially lets
  some of those refusals become emits — the classifier's decision
  procedure has to be re-audited with the new reg-file model.
- **Same-wave / narrowing no-ops.** On same-wave lifts (wave32 →
  wave32, wave64 → wave64) the widening is unnecessary; the design
  should degenerate to the current narrow storage to keep same-wave
  codegen quality unchanged. That means the reg-file constructor
  branches on `(srcIsa, tgtIsa)` and every handler has to handle both
  branches. (Or we widen unconditionally and rely on the backend to
  eliminate dead bits — which it mostly does but can miss edge cases.)
- **Test burden.** Every existing lit fixture that pins a specific IR
  shape involving an SGPR alloca has to be audited for the width
  change. Some pin the narrow form on purpose (regression guards for
  scalar-semantics invariants); those need negative assertions that
  the widened form is *not* used on same-wave lifts.

### 4.4 When to pull this in

Only when §3.1's limitations hurt a real corpus kernel. Specifically:

- A kernel's V_CMP producer and V_CNDMASK consumer are in different
  basic blocks, AND the intervening code path cannot be rewritten to
  keep them in the same BB (e.g. the compiler genuinely needed to
  spill the mask across a control-flow join).
- A kernel sandwiches scalar bitwise ops on a wave-mask SGPR between
  V_CMP and V_CNDMASK (e.g. `v_cmp s6, ...; s_andn2_b32 s6, vcc, s6;
  v_cndmask ..., s6` — the shadow invalidates at `s_andn2_b32`, and
  the fallback is the lossy extract). Widened storage does not need
  the shadow because the SGPR alloca itself carries full fidelity.
- The wave-mask consumers expand beyond V_CNDMASK_B32 into operations
  the shadow design cannot easily wire (dozens of new handlers each
  needing explicit record / lookup / invalidate).
- Cross-BB coverage becomes a hard requirement for a shipping kernel
  before a proper reaching-definitions pass on the raised IR is ready.

The existing runtime probe `tools/compare_correctness/kernels/v_cmp_
cndmask_sgpr.hip` naturally becomes the regression gate for whichever
design lands — with §3.1 the gate expands from `defaultBlocks={16,32}`
to the full sweep; with §4 it stays at the full sweep and additionally
pins a cross-BB / scalar-interleaved variant.

## 5. Scope boundaries of the chosen design (§3.1)

### In scope

- V_CMP_*_e64 writers of arbitrary SGPR destinations (single or pair).
- V_CNDMASK_B32_e64 consumers of SGPR masks.
- Intra-BB dataflow with no intervening scalar write to the mask SGPR.
- Invalidation on any scalar SGPR write routed through
  `AllocaRegFile::writeReg32 / writeReg64`.
- Invalidation on BB transition.

### Out of scope for this PR

- **Cross-BB V_CMP → V_CNDMASK.** Falls back to the extract. A future
  reaching-definitions pass on the raised IR (hinted at as the
  dataflow-upgrade TODO in `wave_size_obstruction.cpp`) is the natural
  landing site for this.
- **Scalar-interleaved pattern.** `V_CMP; s_mov_b32 sN, imm;
  V_CNDMASK` falls back correctly. Fixing this requires §4 or a scalar-
  write-that-preserves-wave-mask-role annotation scheme.
- **Other wave-mask consumers.** `S_AND_B32 sexec, sN, sM` and friends
  that read an SGPR as a wave mask on the EXEC path still go through
  `readOpExecWidth`, which has its own widening story (`widenToExec`'s
  `(v << W_src) | v` broadcast). That path is not changed; its
  correctness is bounded by the same narrow-write limitation at the
  V_CMP producer. Extending the shadow to drive those consumers is a
  natural follow-up.
- **V_CMPX.** V_CMPX writes EXEC, not an SGPR, and its consumer is the
  EXEC alloca read path (already wave-native-width-correct under
  `WaveNativeProjection`). Unchanged by this design.

## 6. Test plan

- **Lit fixture — fused path.** `lit_tests/v_cmp_cndmask_sgpr_fused/`.
  Same HIP kernel shape as the existing `v_cmp_cndmask_sgpr/` (inline-
  asm-forced `v_cmp_ge_f32_e64 s4, |x|, 0.5` + `v_cndmask_b32_e64
  r, -1.0, 1.0, s4`). The raised IR CHECK lines pin the direct-i1
  dataflow: the `fcmp oge` result flows into `select i1` without an
  intervening `ballot.i64` + `trunc` + `lshr` / `and` / `icmp ne 0`
  chain between them. A CHECK-NOT on `mask_lane_idx` for that
  specific pair proves the shadow path was taken.
- **Lit fixture — fallback companion.** `lit_tests/v_cmp_cndmask_
  sgpr_scalar_clobber/`. Adds an inline-asm `s_mov_b32 s4, 0x5A5A5A5A`
  between the V_CMP and the V_CNDMASK. The raised IR CHECK lines pin
  the extract chain is STILL emitted (`mask_lane_idx` / `lshr` / `and
  1` / `icmp ne 0`), proving invalidation fired. Negative assertion
  that no direct `select i1 %vcmpf, …` appears between the compare
  and the cndmask.
- **HIP recipe gate.** `tools/compare_correctness/kernels/v_cmp_
  cndmask_sgpr.hip` already exists. Under §3.1, widen its
  `defaultBlocks` from `{16, 32}` back to `{16, 32, 64, 128, 256}`:
  the larger blocks now fall entirely into the cache path and must
  match bit-exactly. The narrowed sweep becomes historical.
- **Corpus asin.** `corpus_asin_fp32` moves from 44 % wrong rows to
  match (or WRONG with `max|err|` ≪ recipe tolerance, purely from
  polynomial rounding). `status.py --run` table flips asin from
  `MISMATCH` to `ALL_MATCH`.
- **No regressions.** `batch_raise_test` raise rate is unchanged (the
  shadow is an emit-time optimisation, not a lift-blocker).
  `transpiler_tests` GPU tests pass as before.

## 7. Evolution path

- **Step 1 (this PR, §3.1).** Intra-BB V_CMP → V_CNDMASK cache.
  Closes the corpus_asin_fp32 miscompile.
- **Step 2 (follow-up, optional).** Obstruction-classifier refusal
  (§3.2) for the residual cases the cache does not cover. Converts
  remaining cross-BB / scalar-interleaved miscompiles into loud
  refusals rather than wrong answers.
- **Step 3 (conditional).** Reaching-definitions dataflow on the
  raised IR, leveraging LLVM's `UniformityAnalysis` on the amdgpu_
  kernel function. Upgrades the shadow from "intra-BB DenseMap" to
  "full-function per-SGPR last-wave-mask-writer". Wave-mask reads
  from any dominator anywhere in the function benefit.
- **Step 4 (conditional).** If any of the "when to pull this in"
  conditions in §4.4 materialise on a real corpus kernel, implement
  §4 (widened SGPR storage). The shadow from Steps 1 + 3 becomes
  redundant (widened storage is lossless; there is nothing for the
  shadow to repair) and can be deleted.
