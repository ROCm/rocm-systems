# MODREP predicate-chain class

**Status (2026-04-22, post-VOPD-fix).** The four originally-failing
Triton recipes this document was scoped to address have these
end-to-end verdicts today:

- `canary_bpermute_scan_fp32`: **MATCH 4/4** under WaveNative
  default (loud-refused 4/4 under `--disable-wave-native`).
- `swiglu_fp32`: **MATCH 4/4** under both projections.
- `corpus_layernorm_fp32`: **MATCH 4/4** under WaveNative
  default.
- `corpus_softmax_fp32`: **EXIT=2** unchanged (writelane
  safety-net class, orthogonal to this document per
  `wave-size-translation.md` §5.6.3).

The work that actually got here, in landing order:

1. `6648d27d5e` landed the narrow-O1 C5 classifier (this
   document's §5 O1) — MODREP-scoped loud refusal of
   `workitem.id.x() → icmp K, tid` kernels with compile-time
   `K ∈ (0, W_s-1]`.
2. `c3cc463112` graduated `WaveNativeProjection` to the
   cross-widening default. (Its empirical justification
   mis-attributed the swiglu / layernorm end-to-end flips to
   the projection switch; the real drivers are items 3 and
   `df6bf1d35a`. See §6's "Attribution caveat" and §9 (9).)
3. `22ade3c72f` principled-cleanup of the C5 classifier
   (two-pass refactor, intrinsic-propagator audit, WaveNative-
   default non-refusal contract lit fixture).
4. `bd04c268e7` — **the actually-driving fix**. A silent
   fallback in `handle_vopd.cpp::v_cndmask_b32` (hardcoded
   `loadVCC()` regardless of the instruction's explicit scalar
   operand) silently miscompiled any Triton VOPD cndmask pair
   with distinct scalar conditions on its two halves —
   precisely the shape Triton's Kogge-Stone scan (`canary`) and
   mean-of-squares LayerNorm reduction both emit. Fix routed
   the scalar operand through the same SGPR-aware machinery
   `V_CNDMASK_B32_e64` has always used. Entirely orthogonal to
   this document's predicate-chain class; fixes
   `canary_bpermute_scan_fp32` and `corpus_layernorm_fp32`
   end-to-end. §9.7 / §9.8 documents it.
5. `b8fd5f96d5` extended the same SGPR-operand routing fix to
   the six `V_{ADD,SUB,SUBREV}_CO_(CI_)U32` handlers —
   proactive class-closure with zero corpus exposure today.
   §9.10 documents it.

§5 O2 (`tid AND (W_s − 1)` mask rewrite) **remains deferred
indefinitely** per §9 (6): the mask's shape is semantically
incorrect for the norm-family recipes and a no-op on the
sub-case-2 scan-shaped recipes. No future recipe would be
helped by it as specified; reopening criteria in §6.C.

§5 O1 (the narrow-O1 classifier) **landed and is retained** as
a MODREP-scoped regression guard. Today it does not correspond
to any live miscompile — the canary's WRONG output was a VOPD-
cndmask bug, not a predicate-chain class bug — but a future
recipe genuinely hitting the predicate-chain class would
loud-refuse here under MODREP rather than silently WRONG.

MODREP remains fully intact as an opt-in via `--disable-wave-native`
(and is still the default for non-cross-widening directions —
same-wave and narrowing bypass WaveNative's direction gate).

See §6 for the full post-fix landing narrative with per-recipe
attribution; §6.A preserves the 2026-04-21 graduation framing
verbatim for traceability, with the attribution corrections
annotated inline.

**Scope.** One wave-size axis concern: kernels compiled for a source
wave width `W_s` that reach cross-widening (`W_t > W_s`) under modulo-
replication (MODREP), pass the §6 obstruction classes C1–C4, raise
cleanly through `raise_cli`, and yet compute mathematically wrong
values on the target. This doc covers the cross-recipe evidence for
the class, argues it is not covered by the current obstruction
taxonomy, and lays out a decision framework for which of four fix
options to pursue. Orthogonal translation axes (ABI, sync, matrix,
TDM) are not in scope.

**Audience.** Salmon raiser contributors and reviewers who need to
decide what shape the fix should take before writing code. Anyone
tracking what GPT-OSS recipes currently silently miscompile and why.

---

## 1. Problem statement

The `wave-size-translation.md §6` obstruction classes (C1–C4) are
the complete enumeration of *structural* ways modulo-replication can
be wrong: absolute lane-ID leaks (C1), wave-baked cross-lane ops (C2),
inter-replica communication via shared state (C3), and lane-position-
dependent EXEC writes (C4). Every site the G1 classifier refuses is
one of these; everything else emits modulo-replication code that the
§5 *SIMT Predicated Execution* (SPE) projection assumes is wave-size-
oblivious by construction.

A recipe passes G1 iff every site in its disassembly is either (a)
wave-invariant, (b) rewritable per §5.3's landed table, or (c)
pending-rewrite-recognised. The classifier's claim from §7 is:

> Every kernel the tool emits code for is provably wave-size-
> oblivious; every kernel it refuses is one we cannot prove safe.
> There is no third category of "the tool ran and produced wrong
> code."

That claim holds for the C1–C4 axes. It does *not* yet hold for a
fourth concern the classifier does not audit: **the kernel's
downstream use of `workitem.id.x()` inside predicate chains that
gate side effects**. When a source-wave kernel packs its per-lane
work for `W_s` threads and uses `tid` directly as a control variable,
the raiser lifts `workitem.id.x()` through the SPE prelude unchanged —
so on the target wave the predicate evaluates on `[0, W_t)` rather
than `[0, W_s)`. For kernels that use this predicate as "which
lanes participate in stage `s` of a scan" or "which lanes write
which slot of an output tile", the result is a SILENT MISCOMPILE
that is neither classified by G1 nor rewritten by §5.3.

Four compare_correctness recipes currently manifest the class
concretely. This doc collects the evidence, proves the failures are
not explained by C1–C4, narrows the root cause to the
`workitem.id.x()` lift, and lists the options for fixing it.

---

## 2. Observed failures

Raw data gathered `2026-04-21` by
`raise_cli --emit-ir` + `llvm-objdump --mcpu=gfx1250`
over the five recipes whose `compare_correctness` salmon column is
non-green. Repro commands in §7.

| recipe | src WG / wave | tgt WG / wave | raise_cli | salmon verdict | error signature |
|---|---|---|---|---|---|
| `canary_bpermute_scan_fp32` | 32 / 32 | 64 / 64 | OK 117/117 | WRONG 4/4 | `max\|err\|` 4.2 → 21.9, scales with `sqrt(N/BLOCK)` |
| `rmsnorm_fp32` | 32 / 32 | 64 / 64 | OK 389/389 | WRONG 4/4 | half wrong (1024/2048, 2048/4096, 4096/8192, 8192/16384); wrong elements all `actual = −2.87352e-16`; `max\|err\| = 55.9996` |
| `swiglu_fp32` | 128 / 32 | 256 / 64 | OK 325/325 | WRONG 4/4 | half wrong; wrong elements all `actual = −2.87352e-16`; `max\|err\| ≈ 56` |
| `corpus_layernorm_fp32` | 128 / 32 | 256 / 64 | OK 615/615 | WRONG 4/4 | ~99% wrong (2062/2080 at N=128 etc.); wrong elements all `actual = −2.87352e-16`; `max\|err\| ≈ 2.7` |
| `corpus_softmax_fp32` | 128 / 32 | 256 / 64 | **FAIL** — `writelane/readlane-post-raise-safety-net [cross-wave-lane-id-leak]` | CRASH (HIP 209) | loud refusal — included here for family attribution but out of scope for the fix (already covered by §5.6.3) |

The `−2.87352e-16` repeated constant is not a numerical artefact. As
`fp32` its bit pattern is **`0xA5A5A5A5`** — the raiser's VGPR
initialisation sentinel (see the `handle_ds.cpp` comment on
`DS_BPERMUTE_B32`: *"inactive lanes' src1 input is the ambient VGPR
value (possibly the 0xA5A5A5A5 sentinel)"*). When the sentinel leaks
into an output buffer's `actual` value, the corresponding output
slot **was never written by any target-wave lane**. That's the
first datum that frames the class: under MODREP, entire destination
slots are going unwritten.

### 2.1 Instruction mix on gfx1250 (source)

From the compiled `.gfx1250.co` for each recipe, the cross-lane and
numerics instructions that differentiate passing from failing
shapes:

| recipe | `ds_bpermute` | `v_permlanex16` | `v_pk_fma_f32` | `v_sqrt_f32` | `v_rcp_f32` | `v_exp_f32` | `v_dual_max_num_f32` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `canary_bpermute_scan_fp32` | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| `rmsnorm_fp32` | 0 | 1 | 8 | 1 | 2 | 0 | 0 |
| `swiglu_fp32` | 0 | 0 | 0 | 0 | 8 | 8 | 8 |
| `corpus_layernorm_fp32` | 0 | 2 | 8 | 1 | 3 | 0 | 0 |
| `canary_dpp_compound_add_fp32` (PASSES, baseline) | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

The passing baseline uses the same `v_permlanex16_b32` primitive as
the three failing norm-family recipes and also does a reduction +
broadcast. The delta between baseline and failing is **the multi-
element-per-lane arithmetic layer around the reduction**
(`v_pk_fma_f32`, `v_sqrt_f32`, `v_rcp_f32`, `v_dual_max_num_f32` /
`v_exp_f32` in swiglu) — not the reduction itself.

### 2.2 Predicate chain in raised IR

All five recipes emit the SPE prelude unchanged:

```
%cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
%cwd_lane_id    = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %cwd_lane_id_lo)
%tid            = call i32 @llvm.amdgcn.workitem.id.x()
...
%spe_lane_mod   = and i32 %lane_id, 31         ; SOURCE-WAVE-SIZED, correctly masked
%spe_lane_active = icmp ne i32 %spe_exec_bit, 0
```

Two lane-identity values are defined here: `%lane_id` (via `mbcnt`)
and `%tid` (via `workitem.id.x`). The SPE diamond's own predicate
correctly masks `%lane_id` by `W_s - 1 = 31` before comparing —
that's what makes the replica-aware execution gate sound for the
sites the §5 SPE projection wraps.

But subsequent **kernel-emitted** predicates in `canary_bpermute_scan_fp32` read `%tid` WITHOUT the mask:

```
%vsub  = sub i32 %tid, 1       ; scan distance-1 selector
%vcmp  = icmp eq i32 0, %tid   ; "am I lane 0?"
%vcmp215 = icmp ult i32 1, %tid    ; "is this lane in stage-2 set?"
%vcmp359 = icmp ult i32 3, %tid    ; stage-3 set
%vcmp482 = icmp ult i32 7, %tid    ; stage-4 set
%vcmp592 = icmp ult i32 15, %tid   ; stage-5 set
```

These are the Kogge-Stone scan stage guards. Triton compiles them
as `tid ≥ 2^s`, assuming `tid ∈ [0, W_s) = [0, 32)`. On the target
wave64, any hardware lane whose `tid ≥ W_s` — whether because the
dispatch blockSize was sized for the target wave (`tid` range
`[0, W_t)`) or because the hardware still sources an
architecturally-correct `workitem.id.x` for lanes beyond the active
EXEC (see §4 for the two sub-cases) — evaluates the predicate
differently than any source-wave lane would have. The SPE
projection's mask-by-`W_s` (on the `mbcnt`-derived `%lane_id`) is
not propagated into the kernel-emitted `icmp`s against `%tid`.

### 2.3 Why the sentinel leaks (rmsnorm / swiglu / corpus_layernorm)

The three norm-family recipes have a different manifestation: rather
than sum-tree drift, they leave output slots unwritten. Two
observations from the sidecar make the pattern concrete:

- The gfx1250 sidecar records the source compile's WG
  (`num_warps × W_s`): 32 for rmsnorm, 128 for swiglu /
  corpus_layernorm.
- The gfx942 sidecar records Triton's *native-for-gfx942* WG
  (`num_warps × W_t`) which is **double** the gfx1250 WG: 64 and
  256 respectively. That's the blockSize compare_correctness
  launches with on all three columns (native gold, legacy, salmon).
- The salmon-raised HSACO produced by `raise_cli --write-hsaco`
  preserves the source-side WG (32 / 128). Whether the runtime
  salmon hook reconciles this against the launched blockSize, or
  whether HIP launches at blockSize=64 / 256 regardless, is not
  conclusively established from the binaries alone — see open
  question §9 (2).

What IS conclusively established is that all three failing recipes
produce `actual = −2.87352e-16` (`= 0xA5A5A5A5` as fp32) in their
wrong output slots. That is the raiser's VGPR-init sentinel
(see the `handle_ds.cpp:608` comment block on `DS_BPERMUTE_B32`'s
inactive-lane fallback). The sentinel can only leak if the store
target slot is **never written** by any target-wave lane — which
implies either an address-computation mismatch (the lanes that
compute this slot's address do so with a different `tid` than the
source compile expected) or an aliased-overwrite race (two lanes
compute the same address but the "winning" lane's value was itself
computed from a wrong `tid`-derived intermediate).

The exact slot-distribution pattern varies per recipe
(half-wrong for rmsnorm / swiglu, ~99% wrong for corpus_layernorm),
consistent with per-recipe Triton layout choices. The unifying
invariant is the sentinel value appearing in wrong slots — every
such slot is provably the target of no active-lane store.

### 2.4 Runtime per-lane probe on `corpus_layernorm_fp32` (sub-case 1 ground truth)

**Added 2026-04-21** by a probe-based investigation
independent of §2.3's output-comparison evidence.
Methodology: emit the salmon-lifted IR via
`raise_cli --emit-ir`, textually insert a debug
`store i32 <ssa_value>, ptr addrspace(1) %arg1, i64 <probe_offset>`
just below each SSA site of interest, reassemble via the project's
own `llc` + `llvm-mc` + `ld.lld` toolchain (mirrors
`pipeline.cpp`'s invocation), launch the patched HSACO from a
one-off HIP harness that pre-fills the probe slots with a
`0xDEADBEEFu` sentinel. A post-launch sentinel at slot `i` means
target lane `i` never executed the probe store — a per-lane,
per-SSA-site ground-truth execution record at IR granularity.

Three findings feed back into §4.3's sub-case 1 story:

**(a) Var deviation scales exactly as `(BLOCK_SIZE − N) / N · K²`**
across a `{K, N}` sweep on `X = K` constant input, `num_warps = 4`,
wave64 target, `blockDim.x = 128`:

| N | predicted `(1024 − N) / N · K²` | observed `var + eps` (K=1 / K=2 / K=10) |
|---:|---:|---:|
| 128  | 7·K² | 7 / 28 / 700 |
| 256  | 3·K² | 3 / 12 / 300 |
| 512  | 1·K² | 1 / 4 / 100 |
| 1024 | 0   | `eps` (= 1e-5, saturated at rstd = 316.228) |

Bit-exact to fp32 precision in every cell. The `(BLOCK_SIZE − N)`
count is exactly the number of BLOCK slots past `N` that contribute
`K²` to `var_acc` before the `/N` normalisation — one-to-one with
the sub-case 1 mechanism.

**(b) Var-accumulator init is skipped for 96 of 128 target lanes.**
Probing `%vgpr8.14` (the `phi i32 [ 0, %spe_do2059 ], [ %vgpr8.4,
%bb_0x95C ]` SSA node that lifts the gfx1250 source's
`v_dual_mov_b32 v8, v3=0` init for the first var-accumulator VGPR)
per-target-lane:

| target lanes | `%vgpr8.14` after launch | interpretation |
|---|---|---|
| wave 0 lanes 0..31 (tid 0..31) | `0x00000000` | init fired (phi took `spe_do` arm) |
| wave 0 lanes 32..63 (tid 32..63) | sentinel | store never fired |
| wave 1 lanes 0..63 (tid 64..127) | sentinel | store never fired |

Each of the 96 sentinel-reading lanes enters the var pass with
whatever was already in `v8` (raiser init sentinel, mean-pass
residual, or whatever codegen had left there; the probe here
only establishes the init was *skipped*, not the specific surviving
value). A sibling probe on `%vgpr12.11` (the post-`tl.where`
value of `x − mean` for the same target lanes) returns
`0x00000000` for *every* target lane — cndmask zeroing is working
correctly under the SPE wrapper independently of the init skip.
So the downstream
`v_pk_fma_f32 v[8:9], v[12:13], v[12:13], v[8:9]` sequence reads
`v12 = 0` and `v8 = <stale>`: `v8 ← v12² + v8 = 0 + <stale> = <stale>`.
Whatever the 96 lanes' stale `v8` values are, they survive the
var-accumulator loop unchanged (because every squared-and-added
term is 0) and reach the cross-wave reduction as-is. The
`(BLOCK_SIZE − N) / N · K²` signature in (a) and the per-lane
init-skip count in (b) are two views of the same MODREP mismatch:
(a) counts the missing var-contribution at BLOCK-slot
granularity; (b) observes the init-skip at target-lane
granularity; both land on the same 96-lane delta.

**(c) The narrowing step is `trunc i64 to i32` on the cross-widened
ballot.** The SPE gate for the var-acc init's "do-init" branch is:

```llvm
%vcmp2033 = icmp sgt i32 %arg7, %vgpr7.15             ; tid-derived: %vgpr7.15 transitively depends on %tid
%vcmp_ballot2034 = call i64 @llvm.amdgcn.ballot.i64(i1 %vcmp2033)
%vcmp_ballot_trunc2035 = trunc i64 %vcmp_ballot2034 to i32
%new_exec2037 = and i32 %or2022, %vcmp_ballot_trunc2035
%exec_is_zero2039 = icmp eq i32 %new_exec2037, 0
br i1 %exec_is_zero2039, label %bb_0x95C, label %bb_0x950
```

`llvm.amdgcn.ballot.i64` on a wave64 target captures 64 per-target-
lane vcmp results; `trunc i64 to i32` keeps the lower 32 bits only.
For `K = 1, N = 128`:

- **Wave 0** (tid 0..63) — target lanes 0..15 have `cols 0..127 < N`
  → vcmp true; lanes 16..63 have `cols 128..511 ≥ N` → vcmp false.
  Ballot = `0x000000000000FFFF`. `trunc i64 to i32 = 0x0000FFFF`
  (non-zero) → branches to `bb_0x950` (the "do-init" arm); the
  subsequent per-lane SPE gate inside that arm narrows EXEC further
  and is where the "32 of 64 lanes fire" pattern (b) observed
  empirically crystallises. The specific lane partition is sensitive
  to which `and`-mask the `%or2022 & %vcmp_ballot_trunc2035` pair
  actually produces at runtime (the per-lane probe (b) captured the
  net effect — tid 0..31 active, tid 32..63 inactive — without
  reconstructing the exact step from ballot-trunc to post-
  `s_and_saveexec_b64` hardware EXEC; llc's lowering of the SPE
  diamond introduces one `s_and_saveexec_b64` per divergent br so
  the final active-lane set is a composition of all upstream
  narrowings).
- **Wave 1** (tid 64..127) — all threads have `cols 512..1023 ≥ N`
  → vcmp false on every target lane of the wave. Ballot =
  `0x0000000000000000`. `exec_is_zero` is true → entire wave
  branches to `bb_0x95C` (the "skip-init" arm) → no lane in wave 1
  executes the init store.

Both observed patterns — "half a target wave skipped" and "entire
target wave skipped" — derive from the same `trunc i64 to i32`
narrow write. The narrow write is a *symptom* of §4.3's root
cause: if the upstream `%vcmp2033` were computed with `tid AND
(W_s − 1)` rather than raw `tid`, replica-1's vcmp would match
replica-0's by construction, the 64-bit ballot would be two-way
symmetric (bits `[0, 32)` = bits `[32, 64)` by MODREP symmetry),
and `trunc i64 to i32` would be information-preserving. That
symmetry is exactly what O2 establishes.

**Alternative fix considered and rejected (2026-04-21).** The
`ballot.i64 → trunc i32` narrow write was briefly considered as
its own fix target — specifically, widen the raiser's EXEC shadow
from `i32` (source-wave-sized) to `i64` for wave64 targets under
MODREP, propagating the wider type through every `%exec.N` / `%or_N`
/ `%new_exec_N` SSA node in `raise_context.cpp`, every projection
read in `wave_projection.cpp::emitLaneActiveBit` / `ballotI1ToWidth`,
and every EXEC-write path in `handle_sop2.cpp` / `reg_file.cpp`.
Rejected in favour of O2 because (i) the widen-shadow approach
treats the symptom rather than the cause (the `trunc` is only
lossy because the upstream `%vcmp2033` is asymmetric, which is
exactly what O2's `tid` mask fixes at the root), (ii) the touched
surface overlaps with O2's post-raise pass in
`rewrite_cross_lane_divergent.cpp` and its classifier coupling
in `wave_size_obstruction.cpp`, creating a parallel-change
conflict if both land; and (iii) the empirical data in (a)(b)(c)
above is entirely explained by the upstream asymmetry — no
additional widen-shadow work is needed once O2 symmetrises the
ballot.

**Post-O2 verification hook.** The same probe methodology
(reassemble the salmon-lifted IR with a debug store at
`%vgpr8.14`, launch, read back) re-run *after* O2 lands and the
`workitem.id.x()` mask rewrite is active must report
`%vgpr8.14 = 0x00000000` for all 128 target lanes on this
recipe. That's the dispositive per-lane proof that O2's tid mask
is sufficient to recover the sub-case 1 recipes without
additional EXEC-shadow work.

---

## 3. Why existing §6 / §7 don't catch these

Each of the four C-classes, checked against the failing recipes:

**C1. Absolute lane-ID leak.** Predicated on
`v_mbcnt_hi_u32_b32`, `v_readlane_b32` / `v_writelane_b32` with
operand ≥ `W_s`, `llvm.amdgcn.ballot`, `llvm.amdgcn.wavefrontsize`.
None of the failing recipes emit these. `%tid` via
`llvm.amdgcn.workitem.id.x` is not in C1's enumeration — it is
structurally NOT an absolute lane ID (its range is `[0, WG)`, not
`[0, wave)`, and the two ranges coincide only for single-wave
workgroups). G1 correctly clears all four failing recipes on C1.

**C2. Wave-baked cross-lane op.** `v_permlane64_b32`,
`v_permlane32_swap_b32`, `v_permlane16_swap_b32`, `v_permlanex16`,
`ds_swizzle_b32`, DPP with wave-wide semantics, `ds_bpermute_b32`.
All landed rewrites per §5.3. `canary_bpermute_scan_fp32` has 20
`ds_bpermute_b32` calls all of which lift correctly via P1; the
other three recipes emit `v_permlanex16` which lifts via P2. The
IR's bpermute / permlane results go into the kernel's per-lane
computation, and the handler's MODREP comment asserts each
target-wave half reproduces the source scan / reduction
independently. That assertion IS correct at the single-site level —
the failing recipes' misbehaviour is downstream of the cross-lane
op.

**C3. Non-commutative atomic.** Zero in all four failing recipes —
consistent with `gpt-oss-derisking.md §7.5`'s corpus-wide zero.

**C4. Lane-position-dependent EXEC write.** `v_cmpx_*`,
`s_*_saveexec_b32` against a lane-id-derived mask. None of the
failing recipes use `v_cmpx` against an `mbcnt`-derived mask. The
SPE diamond's own `s_and_saveexec_b32` is classified as safe by
the allow-list gate (G2, §5.4).

So the classifier is unanimously green — modulo-replication is
emitted — and the kernel miscompiles anyway. The gap the failing
recipes expose is:

> **Predicates on `workitem.id.x()` (or derived SSA values) that
> gate side effects — stores, addr arithmetic, scan / reduction
> predicates — are NOT audited by any existing C-class and are NOT
> rewritten by §5.3, even though their semantics change under
> MODREP when `W_t > W_s`.**

That sentence is the entire class. The next section unpacks why it
happens and §5 lists ways to close it.

---

## 4. Root cause

### 4.1 The MODREP contract

§5.2 of `wave-size-translation.md` states the projection formally:

> The target wave runs as `R = W_t / W_s` replicas of the source
> wave. Correctness requires every replica observes the same state
> as the source — at every observable side effect, replica `r` with
> lane `l_target = r · W_s + l_source` must compute the same value
> the source's lane `l_source` would have.

The §5.4 SPE allow-list gate + §5.3 cross-lane rewrite table jointly
ensure this for every SemOp. The prelude masks `mbcnt`-derived
`lane_id` by `W_s - 1` so SPE's own active-lane gate evaluates on
the source-wave-scoped `l_source`. Each landed rewrite (P1–P6)
preserves the replica-independence at its instruction-level
boundary.

But `workitem.id.x()` is NOT a SemOp the rewrite table covers — it
is an `amdgcn` intrinsic that the raiser passes through verbatim to
the LLVM backend, which re-lowers it to a VGPR read of the
architectural workitem-id register. That register returns
`l_target ∈ [0, W_t · num_warps · W_s)` on the target wave, NOT
`l_source`. Any downstream use of this value in a kernel-emitted
predicate, address computation, or control-flow decision silently
reads a target-wave-scoped value.

### 4.2 Manifestation A — scan-tree drift (canary_bpermute_scan_fp32)

`tl.cumsum` on a `BLOCK_SIZE`-wide tile compiles to a Kogge-Stone
scan where stage `s` (distance `2^s`) is guarded by
`if tid ≥ 2^s: x += bpermute(lane_id − 2^s)`. On the source
wave32 with `num_warps = 1`, `tid ∈ [0, 32)` and the guard
partitions lanes correctly per stage.

When the same IR runs on target wave64, some hardware lanes have
`tid` outside `[0, W_s)`:

- If the dispatch blockSize is `W_t = 64` (whether by MODREP
  replica expansion or by the gfx942 sidecar's own launch
  blockSize), all 64 hardware lanes are active with `tid ∈ [0, 64)`.
  Lanes with `tid ≥ W_s = 32` take the "add" branch for every
  scan stage (`tid > 15` is trivially true for `tid ≥ 16`), which
  on the source wave would be a proper partition of the scan
  tree. Their stores then either alias the primary-half slots
  (cross-replica race) or land elsewhere.
- If the dispatch blockSize is `W_s = 32` and wave64 runs with
  only the low 32 lanes in EXEC, the active lanes' `tid ∈ [0, 32)`
  and the scan predicates evaluate correctly for the primary —
  yet the kernel's output still diverges from the gold by
  `max|err|` growing as `sqrt(N/BLOCK_SIZE)`. That drift is
  consistent with some component of the per-stage bpermute result
  being influenced by the inactive upper-half lanes' register
  state (bpermute is convergent; inactive-lane VGPR values
  participate in the gather even if their own outputs are
  masked), and the exact dataflow from inactive-lane reads to
  active-lane downstream computation needs a per-stage trace to
  pin down (see §9).

In both sub-cases the common ingredient is that the kernel's
scan-stage guard was compiled with `tid ∈ [0, W_s)` baked in; under
cross-widening, either `tid` is allowed to exceed `W_s` on
executing lanes (case 1) or the surrounding hardware context
(other lanes' register values, EXEC mask, etc.) no longer matches
the source kernel's launch assumption (case 2).

### 4.3 Manifestation B — sentinel-leak in output (rmsnorm / swiglu / corpus_layernorm)

The three norm-family recipes all produce `actual = 0xA5A5A5A5`
(as fp32, `−2.87352e-16`) in their wrong output slots. That
bit-pattern is the raiser's VGPR-init sentinel (documented in
`handle_ds.cpp:608`'s comment block on inactive-lane bpermute
sources). Its presence in a final output element is dispositive
evidence that the element's memory slot was **never written by
any active target-wave lane** — not "written with a wrong value",
but "not written at all, so the reader observes whatever
(sentinel) was in the output buffer pre-launch".

The store-coverage gap traces to the same root as §4.2: the
kernel-emitted store-address computation uses `tid` as an input
without masking by `W_s − 1`, and on the target the `tid` range
does not match the source's assumption. Two ways this manifests
depending on launch layout:

- **Target runs more active lanes than the source assumed**
  (dispatch blockSize ≥ `num_warps × W_t`): multiple lanes compute
  the same target slot (aliasing); the primary-half lane's value
  wins if it writes last, but whichever lane doesn't write ends
  up having "claimed" its computed slot (which is out of the
  intended tile), leaving another slot unclaimed and untouched.
- **Target runs the source's thread count but as a sub-wave**
  (dispatch blockSize = `num_warps × W_s`, `W_t > W_s`): the
  active lanes compute the intended per-lane slots, but the
  store is SPE-wrapped and the diamond's active-arm predicate
  was chained off a `tid`-derived comparison that may evaluate
  false for some slots (sending the store to the "inactive" arm
  which forwards the prior VGPR state via `phi i32 [ …, %spe_do
  ], [ undef, %spe_skip ]` — the `undef` leg of which becomes a
  sentinel in the final store).

Which sub-case applies per recipe depends on the launch blockSize
chosen by the harness's salmon path — open question §9 (2).
Importantly, BOTH sub-cases share the same root ingredient: a
`tid`-shaped predicate or address expression feeds a side effect
that the raiser wraps in an SPE diamond whose active-arm check
is not propagated into the address / predicate chain.

`corpus_softmax_fp32` has the same structural pattern but uses
`v_writelane_b32` in its row-base address computation, which trips
the §5.6.3 writelane/readlane safety net and refuses loudly —
that's why it CRASHES (loud) instead of WRONG (silent). The same
C5-proposed classifier pass that would catch manifestation A and
B would subsume softmax's current refusal with a narrower,
class-specific diagnostic.

### 4.4 Summary of the class

Every failing recipe is characterised by the same three ingredients:

1. `workitem.id.x()` (via `llvm.amdgcn.workitem.id.x`) is read in
   the raised IR.
2. Its value (or one-hop derivations — `add`, `sub`, `and` with a
   constant ≠ `W_s − 1`, `lshr`, etc.) feeds either
   - an `icmp` whose result gates a store / scan-stage update, or
   - an address computation that produces a store offset.
3. The target-wave execution context differs from the source's
   baked-in assumption in at least one of these ways:
   - target blockSize > source blockSize (the dispatch runs more
     lanes than the source kernel was compiled for);
   - target EXEC leaves inactive upper-half lanes whose VGPR state
     can reach active lanes through a convergent cross-lane op;
   - the predicate partitions `tid < 2^s` with `2^s < W_t` admit
     target-wave lanes into a branch the source wave never would
     have taken.

Sites that satisfy (1) + (2) + any of (3) miscompile silently under
the current classifier. Sites that satisfy only (1) are fine — a
kernel that reads `tid` only to store a VGPR into `Out[tid]` where
the WG naturally matches the target wave (the `rope_fp32` /
`vecadd_f16` baselines) passes because (2) or (3) doesn't apply.

The class is orthogonal to C1–C4: C1 is about `mbcnt_hi` / `ballot`
leaks (which the classifier already catches); this class is about
`workitem.id.x()` (which it does not). The relationship to existing
rewrites is: the per-instruction cross-lane rewrites (P1–P6) are
*necessary* for wave-size obliviousness but not *sufficient* — a
kernel that passes the per-instruction audit can still miscompile
if its kernel-level predicate chain is wave-size-sensitive.

---

## 5. Fix options

Four options, each with a different coverage / invasiveness profile.
The decision framework in §6 picks between them based on what §4
actually established.

### O1. G1 classifier extension + loud refusal

**Status.** Landed 2026-04-21 as `transpiler/c5_predicate_chain_classifier.{hpp,cpp}`,
wired into `raiser.cpp` Phase 6.6 (post-mem2reg). See `RaiseFailure::crossWavePredicateChain`,
`ObstructionKind::WorkitemIdPredicateChain`, and
`RaiseFailureReason::CrossWavePredicateChain`. Landed with the
Phase-2-narrowed rule below rather than the original sweeping
shape; §9.6 documents the narrowing evidence.

**Shape (narrow-O1, as landed).** Refuse iff all three hold:

> **C5. Wave-size-sensitive predicate chain.** Any
> `llvm.amdgcn.workitem.id.x()` SSA value whose forward use chain
> reaches an `icmp` whose other operand is a compile-time constant
> `K` with `0 < K <= W_s - 1`, and the chain from the intrinsic to
> the icmp is NOT AND-masked by `(W_s - 1)` first.

The compile-time-K narrowing is structural — it distinguishes
lane-position gates (scan-stage `tid >= 2^s`, half-wave broadcasts,
quad-level masks) from bounds-checks-against-kernargs
(`tid < num_elements`, Triton `mask = offs < N`). The original
O1 wording refused any `tid → icmp → side-effect` without a mask;
Phase-2 IR inspection (§9.6) showed that rule would also refuse
currently-passing baselines `vecadd_f16`, `rope_fp32`,
`corpus_add_fp32`, `corpus_asin_fp32`, `canary_dpp_*`, which have
structurally identical IR but compare against a dynamic kernarg.

In §7's decision procedure, a C5 site surfaces as
`CrossWavePredicateChain` — a new refusal diagnostic. G1 refuses;
no rewrite is registered (§5 O2 is deferred per §9.6). Classifier
marker: `WorkitemIdPredicateChain`; no `RewriteId` entry. Produced
only by the IR-level classifier, never by
`buildObstructionReport`'s MC walk (`workitem.id.x` is an IR-level
intrinsic, not a source-side SemOp).

**Coverage (as landed).** Refuses `canary_bpermute_scan_fp32`
(silent-WRONG → loud-refused): its Kogge-Stone scan-stage guards
emit `icmp ult i32 K, %tid` with K ∈ {1, 3, 7, 15}, all within
`(0, W_s - 1]`. Does NOT refuse the other three originally-failing
recipes (`rmsnorm_fp32` — separately fixed by another commit since
§9.6 was opened; `swiglu_fp32` / `corpus_layernorm_fp32` — bug
class is not predicate chain, see §9.6). Does NOT refuse any
baseline (`vecadd_f16`, `rope_fp32`, `corpus_add_fp32`,
`corpus_asin_fp32`, `canary_dpp_compound_add_fp32`,
`canary_dpp_reduce_fp32`, `canary_permlanex16_rowmax_fp32`) —
all confirmed end-to-end MATCH under the compare_correctness
sweep post-landing.

**Regression guards.** Two lit fixtures
(`lit_tests/c5_predicate_chain_{tid,masked}/`) pin both directions
of the classifier's contract (refuse on K ≤ W_s-1, OK when
AND-masked by 31 first). Ten unit tests
(`tests/c5_predicate_chain_test.cpp`, `C5PredicateChain.*` gtest
suite) audit the classifier in isolation: direction gate
(same-wave / narrowing short-circuit), constant-K boundary
(0 excluded, 32 excluded, 15 and 64 as the named cases), dynamic
vs constant operand distinction, phi propagation, and
mixed-masked/unmasked phi arm refusal.

**Invasiveness.** ~280 LoC classifier (.hpp + .cpp) + ~150 LoC
lit fixtures + ~330 LoC unit tests, plus ~20 LoC wiring into
raiser.cpp and the enum / RaiseFailure additions.

**Risk (unresolved).** The classifier is sound-but-incomplete by
construction: false positives (refusing a safe kernel whose
predicate happens to match the narrow-O1 shape) are benign; false
negatives (failing to refuse a kernel whose predicate chain IS
wave-size-sensitive but compares against a dynamic value) would
still let a silent miscompile through. Two currently-failing
recipes (`swiglu_fp32`, `corpus_layernorm_fp32`) are in the latter
category — their class is orthogonal to predicate chains (see
§9.6) and is out of scope for this classifier.

### O2. Predicate-chain rewrite

**Status.** Explicitly deferred as of the narrow-O1 landing
(2026-04-21). Phase-2 IR inspection (§9.6) established that
the proposed `tid AND (W_s − 1)` rewrite is:

- a structural no-op for `canary_bpermute_scan_fp32` and
  `rmsnorm_fp32` on their actual launch shape (sub-case 2,
  `blockDim.x = W_s`; active lanes already have
  `tid ∈ [0, W_s)`);
- semantically incorrect for `swiglu_fp32` and
  `corpus_layernorm_fp32` on their actual launch shape
  (`blockDim.x = 4·W_s`): target wavefront 0 spans WG `tid`
  values `0..63`, and the source compile with `num_warps = 4`
  expected source wave 1 to see WG `tid` values `32..63`.
  Masking those lanes' `tid` by `31` forces their predicate to
  evaluate on `0..31` where the source intended `32..63` — the
  wrong semantic collapse.

No currently-failing recipe is demonstrably improved by the
rewrite. A principled O2 would need (a) a single-element trace
of each failing recipe to establish what the actual miscompile
mechanism is, and (b) a rewrite shape derived from that mechanism
— not the `tid mod W_s` guess the doc originally proposed.

If O2 is revisited, the shape below is the starting point:

**Shape (originally proposed; semantically unsafe as written).**
A post-raise pass that rewrites every
`llvm.amdgcn.workitem.id.x()` value in a predicate chain
(icmp operand or GEP index feeding a store) to
`workitem.id.x() AND (W_s - 1)` — i.e., the source-wave-scoped
lane id. The rewrite must NOT touch `workitem.id.x()` uses that
correctly depend on target-wave scope (e.g., SPE-internal
`lane_mod` which is already masked, or uses that feed a
wavefront-wide `mbcnt` that scales with target wave size).

**Coverage under the Phase-2 finding.** No failing recipe is
cleanly covered. The original claim ("flips all four failing
recipes to MATCH") was falsified by §9 (2)'s Phase-0
blockSize trace + §9.6's Phase-2 IR inspection.

**Invasiveness.** Higher than O1: needs a use-chain forward
classifier like §5.6.3's writelane/readlane rewrite pass (the
pattern is already in the raiser — see
`rewrite_cross_lane_divergent.cpp`). Entry points: every
`workitem.id.x()` call; forward-walk to collect uses; classify
each use as predicate-context (rewrite), address-context
(rewrite), or SPE-internal (leave alone). Estimate: ~500 LoC
rewrite pass + classifier + ~300 LoC lit tests.

**Risk.** False-positive rewrites that change the semantic of
a kernel that intentionally uses `[0, W_t)` scope. Mitigated by
the same use-chain conservatism as the writelane rewrite: any
ambiguous use-class refuses the whole kernel (treats the rewrite
as not-applicable and falls through to O1). Also: the mask
imposes a hidden `W_s - 1` invariant — if a recipe's `num_warps`
times `W_s` pushes the source thread count past 256 or 512,
the mask is arithmetically the wrong expression and the rewrite
silently corrupts. Needs the raiser to know `W_s` from
`ISAProfile::fromSubtarget(source)` and apply the mask literally,
not relative to the architectural wave size.

### O3. ThreadLoopProjection for scan-shaped patterns

**Shape.** The §5.2 `ThreadLoopProjection` escape hatch that §8's
G6 gate reserves for "no other projection works". Lower the kernel
as a target-wave-scoped serial loop over the `R = W_t / W_s`
source-wave replicas: for each replica `r ∈ [0, R)`, run the
kernel body once with `workitem.id.x()` remapped to
`(r · W_s) + lane_mod`. Cost: `1/R` throughput on the target (2×
slowdown for wave32 → wave64).

**Coverage.** Strictly more than O2 — covers anything that can be
expressed as "run the source kernel R times, then reconcile". In
particular covers kernels where the predicate chain depends on
`[0, W_t)` scope in a way the simple mask wouldn't handle
(hypothetical; no current recipe needs this).

**Invasiveness.** The largest of the four. `ThreadLoopProjection`
is a "skeleton" per §9 line 881 — the machinery exists but there
is no implementation. Landing this option means designing and
implementing the whole scalarisation pass (per-replica serial
lowering, proper SPE wrapping, register-pressure management),
which is a multi-week piece of work.

**Risk.** High implementation cost for zero additional kernel
coverage over O2. Only worth pursuing if O2 proves insufficient
empirically or O1 refuses too many kernels to be acceptable.

### O4. Harness-side constraint on `num_warps`

**Shape.** Add a check in the `compare_correctness` harness (and
by extension any future AOT-compile recipe authoring flow) that
refuses to compile a recipe for gfx1250 unless
`num_warps × W_s ≥ W_t` — ensuring the target WG is at least one
full target wave and MODREP's replica factor is 1.

**Coverage.** Fixes the immediate failure in our own recipes
(`canary_bpermute_scan_fp32` would need `num_warps ≥ 2` to target
wave64 via MODREP without replication). Does NOT fix kernels in
`scope_discovery/kernels/` that we don't author — those are
captured from GPT-OSS verbatim and their `num_warps` choice is
upstream.

**Invasiveness.** Trivial — a single assertion in
`aot_compile.py`. Estimate: ~10 LoC.

**Risk.** Does not address the underlying class. Kernels that
pass the new assertion would still miscompile if they use
predicate chains that reference `[0, num_warps × W_s)` even when
that range exceeds a single target wave (e.g., a 4-warp source
kernel running as 2-wave64 target — still has
`tid ∈ [0, 128)` > `W_t = 64`, still triggers the class). The
assertion is necessary-but-not-sufficient; treating it as "the
fix" would leave a trap for recipes that happen to satisfy it
today and break with a future `num_warps` bump.

---

## 6. Decision framework

What evidence would pick each option, ordered by cost:

| If Phase A's evidence establishes | Pick |
|---|---|
| The predicate-chain rewrite (O2) is sound for every recipe in the failing set — every `workitem.id.x()` use is either predicate-context (safe to mask) or SPE-internal (already masked) — and the `num_warps × W_s` upper bound is stable across the corpus | **O2** |
| O2's classifier cannot prove safety for any recipe in the failing set (ambiguous use-class, unmaskable predicate, etc.) — i.e., O2 would refuse the recipe that O1 would also refuse | **O1** (strictly weaker but trivially sound) |
| A recipe exists that needs `[0, W_t)` scope legitimately and cannot be expressed with an AND-mask, but the kernel shape is otherwise wave-size-oblivious | **O3** (this is the `ThreadLoopProjection` trigger §5 reserves) |
| We need a short-term stopgap for the `compare_correctness` suite only, without committing to a raiser change | **O4** as a harness-side patch alongside O1 in the raiser |

The default recommendation is **O1 + O2 together**, in that order:

1. Land O1 first — it converts silent-WRONG to loud-refused at
   minimal raiser cost, eliminates the silent-miscompile class
   immediately, and keeps the corpus honest while O2 is being
   designed.
2. Follow with O2 — strictly additive over O1 (a site O2 rewrites
   would no longer be classified as C5; sites O2 cannot rewrite
   remain C5 refusals).

O3 and O4 are reserved for edge cases surfaced during O2's rollout.

**What would falsify this recommendation.** Either of these
observations during implementation would reopen the decision:

- **A.** A recipe whose predicate chain is provably safe but which
  O1's conservative classifier refuses, AND the classifier cannot
  be narrowed without losing soundness. Forces O2 to land first.
- **B.** A recipe in `scope_discovery/kernels/` (GPT-OSS captured)
  that needs fix-ups beyond the source-wave-size mask
  (e.g., scope-discovery's `_bitmatrix_metadata_compute_stage1`
  with its mixed bitmatrix / scan / writelane pattern). Forces
  O3 onto the roadmap.

**Picked — as landed 2026-04-22 (post-VOPD-fix rewrite; supersedes
the earlier WaveNative-graduation narrative now in §6.A).**

This section underwent two rewrites between 2026-04-21 and
2026-04-22. The git history tells the story:

   * `6648d27d5e` (2026-04-21) landed the narrow-O1 classifier —
     see §6.B for the original "narrow-O1 only" narrative.
   * `c3cc463112` (2026-04-21) graduated `WaveNativeProjection` as
     the default — the narrative for that landing is preserved in
     §6.A below, *with the attribution corrections noted inline*.
     The numbers in §6.A's table were measured correctly at
     graduation time but mis-attributed several recipes' fixes to
     WaveNative; §9.9 and §9.7 later falsified that attribution.
   * `bd04c268e7` (2026-04-22, §9.7) fixed the real root cause for
     `canary_bpermute_scan_fp32` and `corpus_layernorm_fp32`: a
     silent fallback in `handle_vopd.cpp::v_cndmask_b32`
     hardcoding `loadVCC()` regardless of the instruction's
     actual scalar-condition operand. This is the commit that
     drove the two largest end-to-end MATCH flips; it is entirely
     orthogonal to the predicate-chain class this document
     scopes.
   * `b8fd5f96d5` (2026-04-22, §9.10) extended the same
     SGPR-operand routing fix to the six
     `V_{ADD,SUB,SUBREV}_CO_(CI_)U32` handlers in `handle_valu.cpp`
     — proactive class-closure; no empirical regression today.

The net state of the work this document scoped is below.

### What landed for the predicate-chain class

1. **Narrow-O1 classifier** — landed at `6648d27d5e` per §5 O1.
   `transpiler/c5_predicate_chain_classifier.{hpp,cpp}` +
   raiser.cpp Phase 6.6. Refuses `workitem.id.x() → icmp K, tid`
   kernels where `K ∈ (0, W_s-1]` and the chain is not
   AND-masked by `(W_s - 1)` first. Two-pass walker (cleanup
   commit `22ade3c72f` refactored the original single-pass for
   soundness + clarity). Classifier is MODREP-scoped: runs
   under MODREP, short-circuits under WaveNative.

2. **§5 O2 (mask rewrite)** — **deliberately not landed.**
   §9 (6) falsified its shape during Phase 2: it would collapse
   `tid 32..63` onto `0..31` against the source compiler's
   intent for the norm-family recipes, and it is a no-op on
   sub-case 2 where the target's active lanes already have
   `tid ∈ [0, W_s)`. No recipe in the current corpus would be
   helped by it as specified.

3. **WaveNativeProjection graduated as default** — landed at
   `c3cc463112`. Preserved for (a) the WMMA / matrix-kernel
   correctness invariant §5.6.1 of
   `wave-size-translation.md` documented, (b) `num_warps > 1`
   EXEC-replication correctness that MODREP's replica model
   cannot express. *Attribution caveat in the dedicated
   subsection below* — the graduation's "swiglu / layernorm
   WRONG → match" claims were actually driven by other commits,
   not by the projection switch itself.

### What actually drove the end-to-end MATCH flips

Bit-for-bit attribution of each originally-failing recipe, as
the compare_correctness salmon sweep records it on
2026-04-22-post-VOPD-fix HEAD:

| recipe | pre-work verdict | today's verdict | driver commit(s) |
|---|---|---|---|
| `canary_bpermute_scan_fp32` | WRONG 4/4 | **match 4/4** | `bd04c268e7` (§9.7 VOPD-cndmask SGPR-condition fix). The predicate-chain class this doc scoped is NOT the bug; narrow-O1 classifier still serves as MODREP-path regression guard. |
| `swiglu_fp32` | WRONG 4/4 | match 4/4 | `df6bf1d35a` (pre-graduation MODREP widen-by-replicate half-wave fix). *Not* WaveNative graduation and *not* this document's predicate-chain class — the fix pre-dates both. §9.9 documents the attribution correction. |
| `corpus_layernorm_fp32` | WRONG 4/4 | **match 4/4** | `bd04c268e7` (§9.7 VOPD-cndmask fix). The "small-N residual" §9.8 previously documented was the same bug, not a separate reduction-ordering class. §9.8 is closed by the same commit. |
| `rmsnorm_fp32` | WRONG 4/4 at Phase-0 | match 4/4 | Orthogonal `v_div_scale_f32` / `v_rsq_f32` handler tightening (between Phase 0 and Phase 2 of this doc; never part of this document's scope). No C5 bearing. |
| `corpus_softmax_fp32` | CRASH then EXIT=2 | EXIT=2 | No change. Writelane safety-net class (§5.6.3 of `wave-size-translation.md`), orthogonal to this document. |

**Net end-to-end progress for the 4 recipes this document
scoped**: 3 of 4 MATCH (canary, swiglu, corpus_layernorm); the
4th is refused on an orthogonal class. `rmsnorm_fp32` is a
pre-existing MATCH via an unrelated commit (included above for
completeness — the doc's §2 listed it as WRONG at Phase-0
timing, which was true then).

### Attribution caveat for the WaveNative graduation

§6.A below (the 2026-04-21 graduation narrative) claimed
WaveNative flipped `swiglu_fp32` and `corpus_layernorm_fp32`
from WRONG to match. §9.9 (opened by the `22ade3c72f` cleanup
sweep) falsified that claim empirically:

- **`swiglu_fp32`**: post-`df6bf1d35a` MODREP also produces
  MATCH 4/4 on this recipe. The half-wave-wrong widen-by-
  replicate fix in `df6bf1d35a` was the actual mechanism; the
  WaveNative graduation was measured after that commit had
  already landed and credited the fix to the wrong layer.
- **`corpus_layernorm_fp32`**: the "partial-match under
  WaveNative with 20×–80× smaller error than MODREP" claim in
  §6.A's table was a red herring — both projections produced
  bit-identical numerics post-`df6bf1d35a`. The residual WRONG
  at small N was the VOPD-cndmask bug, fixed by `bd04c268e7`;
  the projection choice was incidental.
- **`canary_bpermute_scan_fp32`**: §6.A documented it as "WRONG
  4/4 under WaveNative" post-graduation. §9.7's mechanism trace
  fixed the bug; the projection choice was again incidental.

**What WaveNative graduation does actually buy**: (a) the
WMMA / matrix-kernel Wave64-collective correctness §5.6.1
requires (unchanged by the post-fix analysis — genuinely
WaveNative-dependent), (b) any future `num_warps > 1` kernel
whose source waves have genuinely distinct EXEC registers
(swiglu / corpus_layernorm turned out NOT to be such kernels;
they are `num_warps = 1` per their Triton declarations). So
WaveNative-as-default is not *wrong* — it's the more principled
projection for cross-widening — but the graduation commit's
empirical justification was mis-attributed. `--disable-wave-native`
/ `enableWaveNative=false` remain available; §9.9 spells out
the scoping options if you want to re-decide.

### MODREP retention

MODREP code is fully intact — no deletions, no structural changes.
Three surfaces retain MODREP as the active projection:

1. Same-wave and narrowing: WaveNativeProjection's constructor
   `report_fatal_error`s on `!(isa.isWave32() && !targetIsa.isWave32())`,
   so same-wave / narrowing directions always pick MODREP (via
   the `enableWaveNative && ...` guard in `raiser.cpp`'s projection
   selection).
2. Explicit `--disable-wave-native` on `raise_cli`, or
   `enableWaveNative=false` on `pipeline.hpp` / `raiser.hpp`
   callers. Preserved specifically for (a) lit fixtures that pin
   MODREP-shape IR invariants (`c5_predicate_chain_tid`,
   `v_cmp_cndmask_sgpr_scalar_clobber`), (b) surfacing the C5
   classifier's diagnostic for canary-class kernels (per 3 below),
   (c) operators debugging projection-specific behaviour.
3. The C5 classifier's refusal path runs under MODREP and
   short-circuits under WaveNative (see
   `c5_predicate_chain_classifier.hpp`'s `waveNative` parameter
   docstring). Today this is a regression guard, not a live fix —
   `canary_bpermute_scan_fp32` now MATCHes under the WaveNative
   default via `bd04c268e7` — but a future kernel that surfaces
   the same C5-shape icmp pattern and miscompiles under MODREP
   would loud-refuse here rather than silently WRONG.

### How to interpret this section going forward

- The predicate-chain class the doc scoped (§1–§4) is real; the
  narrow-O1 classifier correctly refuses it under MODREP. The
  class just turned out not to be the driver of any currently-
  observed end-to-end miscompile, once `bd04c268e7` landed the
  VOPD-cndmask fix.
- §5's O1 is **DONE** (landed as regression guard + loud-refusal
  on MODREP).
- §5's O2 is **DEFERRED INDEFINITELY** per §9 (6); reopening
  criteria in §6.C.
- §5's O3 / O4 are **not triggered** by any current recipe.
- Any future reopening should be driven by the reopening criteria
  in §6.C (a new corpus recipe surfacing the predicate-chain
  class as its actual miscompile mechanism, or a new scope-
  discovery kernel needing ThreadLoopProjection). Until then
  the class is closed.

---

## 6.A. Historical record: 2026-04-21 WaveNative graduation narrative (attribution superseded)

The "Picked: WaveNative" narrative below reflects the
`c3cc463112` graduation commit's original framing. The numbers
in the correctness table were measured correctly at the time but
the *causal attribution* to WaveNative was falsified by §9.9 and
§9.7 over the next 24 hours (see "Attribution caveat" in §6
above). Kept verbatim for traceability; do NOT cite the
attribution claims here as authoritative — the per-recipe
driver commits are in §6's "What actually drove the end-to-end
MATCH flips" table.

**Picked (as landed 2026-04-21, graduation commit
`c3cc463112`): WaveNative as default cross-widening projection;
MODREP retained as opt-in.**

### Why WaveNative graduated [2026-04-21 framing; see §6 attribution caveat]

Empirical evidence collected with `HSA_SALMON_WAVE_NATIVE=1` (the
temporary opt-in flag from Step A of the investigation, now
promoted to default):

**Correctness (compare_correctness salmon path, gfx1250 → gfx942):**

| recipe | MODREP (pre-graduation) | WaveNative (post-graduation) |
|---|---|---|
| `canary_bpermute_scan_fp32` | refused 4/4 (C5 classifier) | WRONG 4/4 (C5 short-circuits under WaveNative; underlying bug is projection-independent, see §9.7) |
| `swiglu_fp32` | WRONG 4/4 | **match 4/4** |
| `corpus_layernorm_fp32` | WRONG 4/4 | match @ N=1024; WRONG @ N=128/256/512 with 20×–80× smaller max\|err\| |
| `rmsnorm_fp32` | match 4/4 | match 4/4 |
| `corpus_softmax_fp32` | EXIT=2 (writelane safety net) | EXIT=2 (unchanged) |
| 7 baselines (`vecadd_f16`, `rope_fp32`, `corpus_add_fp32`, `corpus_asin_fp32`, `canary_dpp_compound_add_fp32`, `canary_dpp_reduce_fp32`, `canary_permlanex16_rowmax_fp32`) | match 4/4 | match 4/4 |

**Attribution correction per §9.9 (2026-04-22):** the "MODREP
pre-graduation" column above conflates two distinct MODREP
states — pre-`df6bf1d35a` (where swiglu and corpus_layernorm
were genuinely WRONG) and post-`df6bf1d35a` (where they were
already MATCH on MODREP, which is what a controlled three-way
sweep actually measures). `c3cc463112`'s evidence was captured
without isolating this, so the "swiglu flipped WRONG → match
under WaveNative" claim is false. The WaveNative-vs-MODREP
numbers on 2026-04-22 HEAD are identical on both projections for
the four non-canary recipes; the canary bug was projection-
independent and is now fixed by `bd04c268e7` rather than by the
graduation.

**Regression surface [still valid, unchanged by §9.9/§9.7]:**
`ctest` and `llvm-lit` produced identical pass/fail counts under
both projections (48 tests, 4 pre-existing failures; 89 lit
tests, 1 pre-existing `s_atomic_dec` failure).
`BatchRaise.AiterGfx950` raise rate unchanged (24/27, 3 expected
`v_permlane32_swap_b32` pendings). `Gfx1250Gpu.Matmul128x128_1tile`
unchanged (already exercised WaveNative explicitly pre-
graduation).

### Why WaveNative covers what MODREP fails on [one claim superseded]

- **`num_warps > 1` classes (swiglu / corpus_layernorm)** — **this
  bullet is SUPERSEDED.** Triton declares both recipes with
  `num_warps = 1` (see `swiglu_fp32.py` and
  `corpus_layernorm_fp32.py` in the Triton corpus). The
  `num_warps > 1` model is correct in principle but does not
  describe these recipes. Their pre-VOPD-fix WRONGness was the
  `handle_vopd.cpp::v_cndmask_b32` silent fallback documented
  in §9.7. This bullet is kept for the general `num_warps > 1`
  correctness argument — it just doesn't drive either of these
  two recipes.
- **WMMA / matrix kernels** — **still valid.** Already depend on
  WaveNative for the `init_whole_wave` + Wave64-collective
  correctness invariant (`wave-size-translation.md §5.6.1`).
  Graduation is a superset: everything that already worked
  keeps working; new recipes outside the matrix family pick up
  the same correctness guarantees.

### What doesn't fix under WaveNative [at 2026-04-21 graduation time; all closed by 2026-04-22]

- **`canary_bpermute_scan_fp32`** — **CLOSED** 2026-04-22 by
  `bd04c268e7` (§9.7). The mechanism was the VOPD-cndmask
  silent fallback, NOT the `handle_ds.cpp` DS_BPERMUTE_B32 shape
  this bullet originally speculated about. Now MATCH 4/4 under
  WaveNative default; remains C5-refused under MODREP (the
  refusal is correct-by-construction but no longer corresponds
  to a live miscompile, since the bug is fixed upstream of the
  classifier).
- **`corpus_layernorm_fp32` small-N residual** — **CLOSED**
  2026-04-22 by `bd04c268e7` (§9.7 / §9.8). The "reduction-
  ordering drift" hypothesis was false; it was the same
  VOPD-cndmask bug on the LayerNorm mean-of-squares reduction.
  Now MATCH 4/4 under WaveNative default.

---

## 6.B. Historical record: narrow-O1-only pick (2026-04-21 prior commit)

The prior "Picked" narrative below reflects the narrow-O1-only
landing that preceded the WaveNative graduation. Kept for
traceability; the "post-landing" verdicts below reflect the
2026-04-21 state only — by 2026-04-22 post-VOPD-fix the canary
and corpus_layernorm verdicts have changed (see §6's table for
the 2026-04-22 state).

**Picked (as landed 2026-04-21): narrow-O1 only.** Trigger: the
sequential falsification chain in §9:

1. **§9 (2) Phase-0 blockSize trace.** The salmon harness
   launches `hipModuleLaunchKernel` with the gfx1250-sidecar
   blockSize (matching the raised HSACO's kernel descriptor),
   not the gfx942-sidecar value the doc's §4 assumed. All four
   originally-failing recipes hit sub-case 2 of the doc's
   §4.2/§4.3 taxonomy (target's active lanes have
   `tid ∈ [0, num_warps × W_s)`), not a mix of sub-cases as the
   earlier `o1_o2_partial` option text framed it.
2. **§9 (5) sub-case-2 inactive-lane class.** The
   `canary_bpermute_scan_fp32` failure mechanism is
   convergent-cross-lane-op gathering sentinel VGPRs from
   inactive target lanes (`W_s..W_t-1`), not predicate-chain
   divergence. The predicate chain is mathematically correct on
   the active lanes; the bpermute / permlanex16 reach is the
   bug.
3. **§9 (6) Phase-2 IR inspection.** `swiglu_fp32` and
   `corpus_layernorm_fp32` have structurally identical tid-chains
   to `vecadd_f16` and `rope_fp32` (both pass) — the difference
   is orthogonal to predicate chains. The originally-proposed
   O2 mask is semantically WRONG for them (would collapse
   `tid 32..63` onto `0..31` against the source compile's
   intent). An O1 classifier broad enough to refuse them would
   also refuse the baselines.

This narrows the principled outcome to "refuse exactly the
recipes where the compile-time-K lane-position-gate signature
appears in IR". That is one recipe today:
`canary_bpermute_scan_fp32`.

**What landed:**

- Narrow-O1 classifier
  (`transpiler/c5_predicate_chain_classifier.{hpp,cpp}`,
  raiser.cpp Phase 6.6). Refuses iff the chain reaches
  `icmp` against compile-time `K ∈ (0, W_s-1]` without an
  AND-mask by `(W_s-1)` first.
- `RaiseFailureReason::CrossWavePredicateChain` +
  `RaiseFailure::crossWavePredicateChain` factory.
- `ObstructionKind::WorkitemIdPredicateChain` entry + docstring
  (no `RewriteId` pair — refuse-only).
- Two lit fixtures
  (`lit_tests/c5_predicate_chain_{tid,masked}/`) pinning
  refuse + non-refuse paths.
- `C5PredicateChain` gtest suite (10 cases) auditing the
  classifier in isolation on synthesised IR.
- Pre-existing `cross_wave_warn` and `v_cmpx_ballot` lit
  fixtures updated to use `K ≥ W_s` (64 / 96) so they exercise
  the cross-wave EXEC-writer / ballot machinery without
  tripping the new C5 classifier on what were latent
  wave-size-sensitive bounds checks.

**What didn't land:**

- §5 O2 (mask rewrite). Deferred — shape is unsafe for the
  failing recipes per §9 (6). A future design iteration can
  resurrect it once the norm-family miscompile mechanism has
  a single-element trace.
- §5 O3 (ThreadLoopProjection). Not triggered by any recipe.
- §5 O4 (harness-side `num_warps` constraint). Not triggered.
- Any fix for `swiglu_fp32`, `corpus_layernorm_fp32`, or the
  §9.5 sub-case-2 class. All out of scope for this document;
  tracked in §9.5 / §9.6 as separate design investigations.

End-to-end verdict change for each recipe (compare_correctness
sweep, 2026-04-21 post-landing):

| recipe | pre-landing | post-landing |
|---|---|---|
| `canary_bpermute_scan_fp32` | WRONG 4/4 | **refused 4/4** (EXIT=2, C5 diagnostic) |
| `swiglu_fp32` | WRONG 4/4 | WRONG 4/4 (out of scope, §9.6) |
| `corpus_layernorm_fp32` | WRONG 4/4 | WRONG 4/4 (out of scope, §9.6) |
| `rmsnorm_fp32` | WRONG 4/4 | match 4/4 (orthogonal recent fix, unrelated) |
| `corpus_softmax_fp32` | CRASH (writelane safety-net) | refused 4/4 (same writelane safety-net) |
| `vecadd_f16`, `rope_fp32`, `corpus_add_fp32`, `corpus_asin_fp32`, `canary_dpp_*`, `canary_permlanex16_*` | match 4/4 | match 4/4 (no classifier hit) |

Net: 1 silent-WRONG → loud-refused, 0 regressions. Strict
progress within the principled scope the Phase-0 + Phase-2
findings allow.

---

## 6.C. Deferred options — reopening criteria

O2 / O3 / O4 reopen the decision if:

- **A.** A new corpus recipe appears whose `workitem.id.x()`
  flows into an `icmp` against a compile-time constant in
  `(0, W_s-1]` AND whose downstream miscompile mechanism is
  demonstrably the predicate-chain class (not §9.5 /§9.6).
  Triggers an O2 redesign with a real mechanism trace.
- **B.** A corpus recipe is shown to need the
  `ThreadLoopProjection` escape hatch (§5 O3). Triggers an
  O3 scope paper.
- **C.** A scope_discovery / GPT-OSS kernel requires a
  fix-up beyond either O1's refusal or an eventual O2 mask
  — e.g. `_bitmatrix_metadata_compute_stage1` with its mixed
  bitmatrix / scan / writelane pattern. Triggers §5's O3
  pick or an entirely new class doc.

---

## 7. Test surface

### Existing canaries that pin this class

Verdicts below reflect 2026-04-22 HEAD (post-VOPD-fix). Per §6's
"Picked — as landed" section, O1 landed as classifier-only
regression guard; O2 deferred indefinitely per §9 (6); the actual
end-to-end MATCH flips for `canary_bpermute_scan_fp32` and
`corpus_layernorm_fp32` were driven by the orthogonal
`bd04c268e7` VOPD-cndmask fix (§9.7), not by the
predicate-chain machinery this document scopes. Each entry
below reflects that attribution.

- `canary_bpermute_scan_fp32` — pins the scan-predicate variant.
  Kernel: `tl.cumsum` over 128 elements per program with
  `num_warps = 1`. Kogge-Stone scan-stage guards `icmp ult K, tid`
  with K ∈ {1, 3, 7, 15}, all `≤ W_s - 1 = 31` — exactly the
  narrow-O1 signature. Salmon launch shape per §9 (2):
  `blockDim.x = 32 = W_s`. 2026-04-22 HEAD: **MATCH 4/4 under
  WaveNative default** (via `bd04c268e7` / §9.7's VOPD-cndmask
  fix — the actual miscompile mechanism was entirely orthogonal
  to the predicate-chain class, and the C5 shape of the scan
  guards is a coincidence). Under `--disable-wave-native`
  (MODREP path): **refused 4/4** via the narrow-O1 classifier
  — the refusal is still correct-by-construction but no longer
  corresponds to a live miscompile now that the upstream bug
  is fixed. The `lit_tests/c5_predicate_chain_tid/` fixture
  pins the refusal shape on MODREP structurally;
  `lit_tests/v_dual_cndmask_b32_sgpr_cond/` pins the actual
  root-cause fix at IR level; `compare_correctness` pins the
  end-to-end MATCH.
- `rmsnorm_fp32` — NOT a narrow-O1 match (its icmps compare
  against dynamic `%arg4`, not a compile-time K). Was WRONG 4/4
  at the start of this investigation; now MATCH 4/4 in the
  compare_correctness sweep — fixed by an orthogonal commit
  between Phase 0 and Phase 2 (likely `v_div_scale_f32` /
  `v_rsq_f32` handler tightening per swiglu_fp32's docstring
  commentary about the `_num_f*` lowering path). No C5 bearing.
- `swiglu_fp32` — NOT a narrow-O1 match (dynamic-kernarg
  icmps, structurally identical to `vecadd_f16`'s passing
  shape). 2026-04-22 HEAD: **MATCH 4/4 under both WaveNative
  default and `--disable-wave-native` MODREP.** The end-to-end
  fix came from `df6bf1d35a` (MODREP widen-by-replicate
  half-wave fix), pre-dating the WaveNative graduation and
  having nothing to do with the predicate-chain class. §9.9
  documents the attribution correction (the 2026-04-21
  graduation commit mis-credited this fix to WaveNative).
- `corpus_layernorm_fp32` — NOT a narrow-O1 match (same
  dynamic-kernarg icmp shape as swiglu). 2026-04-22 HEAD:
  **MATCH 4/4 under WaveNative default** (via `bd04c268e7` /
  §9.7's VOPD-cndmask fix — Triton's mean-of-squares reduction
  emits the same VOPD `v_dual_cndmask_b32 vX, ..., sN ::
  v_dual_cndmask_b32 vA, ..., vcc_lo` shape the scan canary
  does, for the same reason and with the same fix). §9.8 is
  closed by the same commit.
- `corpus_softmax_fp32` — already refused via the §5.6.3
  writelane safety net; listed here for completeness. Not
  additionally matched by the narrow-O1 classifier, but its
  existing refusal diagnostic already attributes the failure
  to the correct class (`writelane/readlane-post-raise-safety-net
  [cross-wave-lane-id-leak]`). 2026-04-22 HEAD: EXIT=2
  unchanged. Future work could graduate it to the C5
  diagnostic if evidence shows the same
  `workitem.id.x → icmp(K)` signature applies, but no such
  evidence exists today.
- `canary_dpp_compound_add_fp32` — the structural baseline that
  pins "reduction + broadcast + downstream use is safe when there
  is no `num_warps × W_s > W_t` gap and no VOP3P packed arithmetic".
  Pre- and post-landing: MATCH 4/4 (classifier must not
  refuse this recipe; does not, per the compare_correctness
  sweep).
- `vecadd_f16`, `rope_fp32`, `corpus_add_fp32`,
  `corpus_asin_fp32`, `canary_dpp_reduce_fp32`,
  `canary_permlanex16_rowmax_fp32` — additional baselines
  confirmed end-to-end MATCH 4/4 post-landing; none match the
  narrow-O1 signature.

### Landed regression guards (lit + gtest)

*Predicate-chain class (the narrow-O1 classifier — this document's
actual §5 O1 landing):*

- `lit_tests/c5_predicate_chain_tid/` — REFUSAL sibling. Inline
  asm emits `workitem.id.x() → icmp ult K=16, tid → cndmask →
  store`. First RUN line asserts refusal under
  `--disable-wave-native` (MODREP path); second RUN line
  asserts clean raise + IR emission under WaveNative default
  (added 2026-04-21 by `22ade3c72f` to pin the
  WaveNative-default non-refusal contract end-to-end). Lit
  expects `cross-wave-predicate-chain` diagnostic,
  `compile-time constant 16`, `Class 5 / WorkitemIdPredicateChain`
  per-site trace on the MODREP RUN, and the `workitem.id.x()
  call` + kernel label on the WaveNative RUN.
- `lit_tests/c5_predicate_chain_masked/` — NON-REFUSAL sibling.
  Same K=16 icmp, but preceded by `and tid, 31` (= W_s-1); lit
  expects raise_cli success and the `and i32 ..., 31` anchor
  in the emitted IR.
- `tests/c5_predicate_chain_test.cpp` — `C5PredicateChain.*`
  gtest suite (13 cases as of 2026-04-22, post-cleanup refactor
  in `22ade3c72f`). Audits the classifier on synthesised IR in
  isolation from the raiser's MC-level pipeline:
  `TidDirectSmallConstRefuses`, `TidMaskedBeforeCmpAccepts`,
  `TidSmallConstZeroAccepts`, `TidLargeConstAccepts`,
  `TidDynamicCmpAccepts`, `SameWaveDirectionGate`,
  `NarrowingDirectionGate`, `WaveNativeProjectionGate`,
  `NoCallsIsNoOp`, `PhiPropagatesTidDerivation`,
  `MaskedPhiThroughUnmaskedArmRefuses`,
  `CrossSubtreeMaskedVsUnmaskedAccepts` (pins the falsified
  cross-subtree widening the cleanup briefly explored),
  `IntrinsicPropagatorRefuses` (pins the numeric-intrinsic
  propagator audit — `llvm.umin` / `llvm.smax` / etc.).
- `lit_tests/cross_wave_warn/` and `lit_tests/v_cmpx_ballot/` —
  updated alongside the classifier landing to use `K ≥ W_s`
  constants (64 / 96) so their cross-wave EXEC-writer / ballot-
  routing coverage is not conflated with the new C5 predicate-
  chain coverage. Pre-update those fixtures exercised K ≤ W_s-1
  predicates, which the narrow-O1 classifier correctly refuses
  as latent-bug kernels.

*VOPD / VOP3B SGPR-condition class (the actually-driving fix for
canary + layernorm end-to-end — orthogonal to §5's predicate-
chain class, but tracked here because the landing sequence of
§9.7 / §9.8 closure depends on it):*

- `lit_tests/v_dual_cndmask_b32_sgpr_cond/` — REGRESSION FENCE
  for the `bd04c268e7` VOPD `v_dual_cndmask_b32` SGPR-condition
  fix (§9.7). Inline asm emits a paired
  `v_dual_cndmask_b32 ..., s0 :: v_dual_cndmask_b32 ..., vcc_lo`
  VOPD with DIFFERENT scalar conditions on the two halves. Lit
  expects two distinct `vopd_cndmask` selects whose conditions
  are different `i1` values (the SGPR-path cmp and the
  VCC-path cmp) — a regression that re-introduces hardcoded-
  VCC collapses the two conditions and fails the CHECK
  pattern.
- `lit_tests/v_add_co_u32_sgpr_carry/` — REGRESSION FENCE for
  the `b8fd5f96d5` carry-chain SGPR-operand fix (§9.10). Inline
  asm emits `v_add_co_u32 ..., s0, ... :: v_add_co_ci_u32_e64
  ..., s0, ..., s0` — the canonical 64-bit address-chain shape
  under VCC pressure. Lit expects the carry-out ballots through
  `amdgcn.ballot.i64` into the SGPR path, the second add's
  carry-in forwards the SAME `i1` SSA value the first add
  produced (fresh-shadow lookup — no `load i1, ptr %vcc` in
  between), and a CHECK-NOT pin on `load i1, ptr %vcc` catches
  any regression that re-introduces hardcoded-VCC.

### Future lit fixture (only if O2 revisited)

- `lit_tests/c5_predicate_chain_rewrite/` — reserved for a
  future O2 landing. Would pin the rewritten IR shape (tid →
  `and tid, 31` → unchanged icmp) for the same synthetic kernel
  as the refusal sibling. Not implemented today; §5 O2 is
  deferred per the Phase-2 findings in §9.6.

### Repro commands for this doc's evidence

Each table entry above was generated by (`$T` = transpiler root,
`$CC` = `$T/tools/compare_correctness`):

```bash
# Instruction mix on gfx1250
/opt/rocm-7.2.1/lib/llvm/bin/llvm-objdump \
    -d --mcpu=gfx1250 --triple=amdgcn-amd-amdhsa \
    $CC/kernels/build/<recipe>.gfx1250.co \
  | grep -cE 'ds_bpermute|v_permlanex16|v_pk_fma_f32|...'

# raise_cli verdict (source-to-target lift + G1 classification)
$T/build/raise_cli \
    $CC/kernels/build/<recipe>.gfx1250.co \
    --isa=gfx1250 --target-isa=gfx942

# Raised IR (post-mem2reg, pre-backend)
$T/build/raise_cli \
    $CC/kernels/build/<recipe>.gfx1250.co \
    --isa=gfx1250 --target-isa=gfx942 \
    --emit-ir=<kernel_symbol>

# Runtime verdict (native gold + salmon lift + numerical compare)
LD_PRELOAD=$CC/libsalmon_intercept.so \
    $CC/compare_correctness --recipe=<recipe>
```

The `<kernel_symbol>` comes from the recipe's sidecar
(`$CC/kernels/build/<recipe>.sidecar.json`, `kernel_symbol` field).

---

## 8. References

- [wave-size-translation.md](wave-size-translation.md) — the canonical
  wave-size axis spec. Specifically §5.2 (modulo-replication
  definition), §5.3 (cross-lane rewrite table, which includes the
  P1 `ds_bpermute_b32` landing), §5.6.3 (writelane/readlane
  post-raise rewrite, the closest existing analogue to O2), §6
  (obstruction classes C1–C4), §7 (decision procedure), §8
  (principled fail-loudly gates including G1).
- [gpt-oss-derisking.md](gpt-oss-derisking.md) §1 (four-class
  framework), §7 (per-class findings), §9.2 (engineering worklist).
- [tools/compare_correctness/kernels/triton/README.md](../transpiler/tools/compare_correctness/kernels/triton/README.md)
  — canary-set findings section, specifically findings #1 and #4
  (to be updated post-doc).
- `handle_ds.cpp:608` (`DS_BPERMUTE_B32` handler) — the P1 landing
  this class's `canary_bpermute_scan_fp32` exercise surfaced as
  incomplete-but-not-broken.
- `rewrite_cross_lane_divergent.cpp` (the writelane/readlane
  post-raise rewrite) — structural template for O2.
- `raise_context.cpp` (the `SPE` prelude that masks `mbcnt`-derived
  `lane_id` by `W_s - 1` correctly) — the pattern O2 extends to
  `workitem.id.x()` chains.
- Issue #13 — historical context; the original "silent bpermute
  miscompile" has graduated to loud-refuse, but the `canary_bpermute_scan_fp32`
  residual confirms the underlying class is unsolved.
- Commits: `d9bfd99626` (P1 intrinsic lift); `087f24d851`
  (writelane rewrite graduated-to-default-on, which is why
  `corpus_softmax_fp32` flipped from WRONG to CRASH and is listed
  in the evidence table as already refused).

---

## 9. Open questions

1. Is the `0xA5A5A5A5` sentinel leak's slot-distribution pattern
   (half-wrong vs almost-all-wrong per §2) fully explained by Triton
   layout choices, or is there a secondary raiser bug interleaved
   with the MODREP issue? Answering definitively needs a per-recipe
   manual trace of one element's write path — out of scope for this
   doc; would be follow-up investigation if O2's rewrite doesn't
   flip the recipe to MATCH.
2. **Launch blockSize negotiation under salmon.** Each recipe has
   two `max_flat_workgroup_size` values that disagree:
   - The gfx942 sidecar entry (from Triton's *native-for-gfx942*
     build) that the harness uses when calling
     `hipModuleLaunchKernel`: 64 (canary_bpermute_scan / rmsnorm),
     256 (swiglu / corpus_layernorm).
   - The salmon-raised HSACO's own kernel descriptor (produced by
     `raise_cli --write-hsaco`): 32 (canary_bpermute_scan /
     rmsnorm), 128 (swiglu / corpus_layernorm) — exactly the
     source-side WG.
   When HIP is handed the raised HSACO and asked to launch at the
   gfx942 sidecar's blockSize, does it (a) use the raised HSACO's
   smaller WG and mask out the upper lanes with EXEC, (b) silently
   upscale to the gfx942 sidecar's blockSize with all lanes active,
   or (c) reject the launch with a runtime error? The answer
   decides which sub-case of §4.2 / §4.3 each recipe hits at
   runtime and which sub-case the fix needs to prioritise. A
   targeted trace with `HSA_HOTSWAP_DEBUG` or equivalent + `strace`
   on `hipModuleLaunchKernel` will answer this cleanly — follow-up
   work before O2's implementation.

   **Answered 2026-04-21 (source + runtime evidence).** The
   question's premise — "the harness calls `hipModuleLaunchKernel`
   at the gfx942 sidecar's blockSize" — is **incorrect**. The
   salmon child of `compare_correctness` launches with the
   **gfx1250 sidecar's** `max_flat_workgroup_size`, which matches
   the salmon-raised HSACO's own kernel descriptor exactly (both
   carry the source-side WG).

   Evidence chain:
   - `compare_correctness.cpp:3753`: `wgSize = M.maxFlatWorkgroupSize`,
     where `M = archMeta.at(gCurrentChildIsa)`.
   - `compare_correctness.cpp:4261`: `gCurrentChildIsa = "gfx1250"`
     for `Mode::Salmon` and `Mode::Legacy`, `"gfx942"` only for
     `Mode::Native`. The salmon child therefore reads the gfx1250
     arch-meta entry, not the gfx942 one.
   - Runtime verification: a temporary trace added at the
     `hipModuleLaunchKernel` call site (`HSA_MODREP_TRACE=<file>`,
     reverted before commit) captured the actual
     `(grid, block, wgSize)` for one sweep of each failing recipe +
     the passing baselines. Summarised:

     | recipe | native (gfx942) | salmon (gfx1250) | source W_s | sub-case |
     |---|---:|---:|---:|---|
     | canary_bpermute_scan_fp32 | 64 | **32** | 32 | §4.2 sub-case 2 (`block = W_s`) |
     | rmsnorm_fp32 | 64 | **32** | 32 | §4.3 sub-case 2 (`block = W_s`) |
     | swiglu_fp32 | 256 | **128** | 32 | §4.3 sub-case 1 (`block = 4·W_s > W_t`) |
     | corpus_layernorm_fp32 | 256 | **128** | 32 | §4.3 sub-case 1 (`block = 4·W_s > W_t`) |
     | canary_dpp_compound_add_fp32 (baseline, MATCH) | 64 | **32** | 32 | §4.2 sub-case 2 |
     | rope_fp32 (baseline, MATCH) | 64 | **32** | 32 | §4.2 sub-case 2 |

     The salmon column is what HIP actually receives as `blockDim.x`
     at `hipModuleLaunchKernel`; no silent upscale (option (b) of
     the original question) and no runtime rejection (option (c))
     happens — the harness naturally matches the raised HSACO by
     reading from the gfx1250 sidecar (option (a), with no EXEC
     masking needed because the launch shape matches the KD).

   Implication for §4.2 / §4.3 sub-case split per recipe:
   - `canary_bpermute_scan_fp32` + `rmsnorm_fp32` hit **sub-case 2**
     (§4.2 / §4.3, "target runs the source's thread count but as a
     sub-wave"). On a gfx942 wave64 wavefront with `blockDim.x = W_s
     = 32`, hardware EXEC covers only the low 32 lanes at entry;
     architectural `workitem.id.x()` on active lanes is in
     `[0, W_s)` and every kernel predicate `tid < 2^s` evaluates as
     the source wave32 would have.
   - `swiglu_fp32` + `corpus_layernorm_fp32` hit **sub-case 1**
     (§4.3, "target runs more active lanes than the source
     assumed"). `blockDim.x = 128` dispatches two gfx942 wave64
     wavefronts, each with all 64 lanes active and `tid` values in
     `[0, 64)` within each wavefront. Lanes 32..63 of each
     wavefront are MODREP replica 1; their `tid` value exceeds
     `W_s = 32` and any kernel-emitted predicate `tid < K` with
     `K < W_t = 64` evaluates *differently* from the source's
     corresponding replica-0 lanes.
   - Baselines (`canary_dpp_compound_add_fp32`, `rope_fp32`) also
     hit sub-case 2 but do not miscompile — confirming that
     sub-case 2 alone is not a sufficient trigger for the failure
     class.

   Implication for O2's rewrite shape (the concrete §9 decision
   the original question teed up):
   - Sub-case 1 is *precisely* the predicate-chain bug §4 / §5
     describe: target replica-1 lanes see `tid ≥ W_s` and evaluate
     `icmp` predicates differently from replica-0. `tid AND
     (W_s − 1)` fixes it by collapsing replica-1's `tid` values
     back onto replica-0's `[0, W_s)` range. **O2 covers swiglu +
     corpus_layernorm.**
   - Sub-case 2 is NOT predicate-chain by construction: on active
     lanes `tid ∈ [0, W_s)` already, so `tid AND (W_s − 1)` is a
     no-op AND on those lanes. The O2 mask rewrite would emit
     bit-for-bit identical IR on `canary_bpermute_scan_fp32` and
     `rmsnorm_fp32`'s active-lane predicates. **O2 does not cover
     these two recipes as currently shaped.** A store-address
     recomputation of the same flavour would likewise be a no-op.

   The actual root cause for sub-case 2 is the one §4.2 identifies
   in its second bullet but does not yet name as a separate class:
   convergent cross-lane primitives (`ds_bpermute`,
   `amdgcn.update.dpp`, etc.) participate over the full physical
   wave regardless of hardware EXEC; inactive-at-entry lanes 32..63
   contribute their sentinel-initialised VGPR values (`0xA5A5A5A5`
   from `AllocaRegFile::init`) into active lanes' gather results.
   That is orthogonal to the §4 predicate-chain class — a bpermute
   reading inactive-lane data has nothing to do with how `tid` is
   used in surrounding `icmp`s — and is out of scope for O1 / O2 /
   O3 / O4 as the doc currently shapes them. See open question
   §9 (5) for the follow-on class this surfaces.

3. Is there a principled way to narrow O1's classifier to avoid
   refusing legitimate `tid`-bounds-check idioms (e.g., `if tid <
   num_elements: ...` where `num_elements > W_t`)? The §7
   over-approximation discipline says we err on the side of
   soundness, but any recipe the classifier refuses spuriously is a
   motivating case for O2.
4. Should the `target-capability-dispatch.md` `SemOpAttrs` extension
   (§5 of that doc) grow a new bit `predicateChainLaneScoped` on
   `AMDGPU::workitem.id.x`-emitting SemOps, so the classifier's
   C5 pass can enumerate emission sites without hand-maintaining
   a list? This is the "hang new attrs off `SemOpAttrs`" pattern
   the README index section recommends.
5. **Convergent-cross-lane-op inactive-lane leak under sub-case 2
   (opened 2026-04-21 by Phase-0 investigation of §9 (2); **STATUS:
   FALSIFIED** 2026-04-21 — superseded by §9.7's empirical finding
   that `canary_bpermute_scan_fp32`'s miscompile is projection-
   independent and not an inactive-lane-leak mechanism at all.
   Retained verbatim below as the historical attempted diagnosis;
   kept for the narrative thread, not as a live open question.)**

   ---

   **Correction (2026-04-21).** The claim below that
   `AllocaRegFile::init` "seeds VGPRs with a sentinel (the
   `0xA5A5A5A5` pattern)" is factually incorrect. `AllocaRegFile::init`
   (`transpiler/reg_file.cpp:61–128`) creates VGPR allocas without
   any explicit store — their contents are `undef` until first
   write, not any sentinel bit pattern. The `0xA5A5A5A5` bytes
   observed in §2's WRONG-output dumps come from the
   `hipMemset(dOut, 0xA5, bytes)` call in
   `compare_correctness.cpp:2212` — the TOOL's output-buffer
   sentinel for unwritten slots, used precisely so a kernel that
   fails to write a slot surfaces loudly in the comparator. What §2
   actually observes is "the raised kernel did not write some
   slots", not "the raised kernel stored a raiser-synthesised
   sentinel". The class framed below therefore rests on a wrong
   bridge; the actual cause of unwritten slots under sub-case 2 is
   a still-open question tracked in §9.7.

   ---

   **Historical text (kept for the narrative; do not rely on).**

   When the salmon path launches a kernel with `blockDim.x = W_s`
   (sub-case 2 of §4.2 / §4.3 — confirmed for
   `canary_bpermute_scan_fp32` and `rmsnorm_fp32` per §9 (2)'s
   trace), the hardware wave at entry has EXEC covering only the
   low `W_s` lanes; the upper `W_t − W_s` lanes are
   inactive-at-entry. The raised IR still emits convergent
   cross-lane primitives (`llvm.amdgcn.ds.bpermute`,
   `llvm.amdgcn.update.dpp`, `llvm.amdgcn.ds.swizzle`, …) that
   participate over the full physical wave regardless of EXEC —
   inactive lanes contribute their current VGPR contents into the
   gather. `AllocaRegFile::init` (`reg_file.cpp`) seeds VGPRs with
   a sentinel (the `0xA5A5A5A5` pattern documented in §2), so
   active lanes reading through `ds_bpermute` can pull sentinel
   values into their scan stages when the selector lands on an
   inactive lane (common for stage-0 of a Kogge-Stone scan where
   half the active lanes read a backward neighbour that in wave32
   would wrap within the source wave but in wave64 wraps into the
   inactive upper half).

   This class is orthogonal to the predicate-chain class this
   document addresses:
   - It is not structurally dependent on `workitem.id.x()` flowing
     through an `icmp` — it manifests through
     `ds_bpermute(selector, vgpr)` regardless of how `selector` or
     `vgpr` were computed.
   - It is not fixed by any `tid AND (W_s − 1)` rewrite. Active
     lanes already have `tid ∈ [0, W_s)` on sub-case 2; the
     arithmetic fed to the bpermute selector is correct; the
     corruption happens at the cross-lane gather's *source* side.
   - The §5 options as shaped do not cover it. O3
     (`ThreadLoopProjection`) would cover it in principle (each
     replica runs serially, so inactive-lane VGPRs never
     participate in a gather), but the same §5 cost analysis
     applies — multi-week implementation, a design topic of its
     own.

   Possible O2-adjacent mitigations (all structurally distinct
   from the §5 O2 mask rewrite; surfaced here for the human to
   scope, not for this document to pick):
   - Force hardware EXEC = -1 at kernel entry via
     `@llvm.amdgcn.init_whole_wave` (the §5.6.1 WaveNativeProjection
     trick), so every lane executes and inactive-lane VGPRs are
     defined by the source's `spe_do` path rather than the init
     sentinel. Currently MODREP opts out of this
     (`ModuloReplicationProjection::emitInitialExec` returns
     `-1 = all-ones-modeled-EXEC`) on the theory that the target
     runs with the source's EXEC — which is true for sub-case 1
     but not sub-case 2, where the source's EXEC covered more lanes
     than the target's hardware EXEC.
   - Emit a `select EXEC[L], val, neutral` wrapper on every
     convergent-cross-lane-op source VGPR so inactive-lane sources
     become a neutral value (e.g. `0` for a sum scan). Requires
     per-primitive knowledge of what "neutral" means.
   - Seed `AllocaRegFile::init` with an explicit `0` store for
     every VGPR under cross-widening. (Per the correction block
     above, the VGPR allocas are currently left `undef`, not
     `0xA5A5A5A5`-filled — an explicit-zero store would just
     replace an `undef` read with a `0` read on inactive-lane
     bpermute sources. If the true mechanism is undef-read
     propagating to the arithmetic, zeroing could plausibly help;
     if it's a handler-level arithmetic bug as §9.7 suggests,
     zeroing is a no-op. Kept only as a thought experiment now
     that §9.7 has falsified the underlying framing.)

   Evidence anchor: the Phase-0 trace in §9 (2) confirms
   `canary_bpermute_scan_fp32` launches at `blockDim.x = 32`
   (sub-case 2), and §2.2's observation that the baseline
   `canary_dpp_compound_add_fp32` passes at the same sub-case 2
   launch shape rules out "sub-case 2 is the problem" by itself —
   it's the sub-case-2-with-a-cross-lane-primitive-whose-selector-
   reaches-inactive-lanes shape. `canary_dpp_compound_add_fp32`
   uses a single `v_permlanex16` + broadcast that stays within each
   source wave's half (selector is XOR-16, never crosses the
   `W_s = 32` boundary), so no inactive-lane read. The scan-shaped
   recipes walk a strided gather across the whole source wave,
   which *does* cross the boundary.

   This is a scope expansion from the document's originally-named
   class. It does not invalidate §1–§8 — the predicate-chain class
   is real and affects sub-case 1 recipes — but it is a second,
   orthogonal class surfaced by the Phase-0 answer to §9 (2). The
   human decides whether to fold this into a revised §5 option
   list, split it into a separate design doc, or defer both the
   investigation and the sub-case 2 recipes until O2's sub-case 1
   coverage has been validated empirically.

   **Sibling class, same scope expansion (added 2026-04-21 by
   a Phase-0 investigation of `corpus_softmax_fp32`'s current
   loud refusal).** The
   `writelane/readlane-post-raise-safety-net` refusal on
   `corpus_softmax_fp32` was initially triaged as a candidate
   classifier-precision false positive ("the readlane's use chain
   reaches a `readfirstlane`, but that readfirstlane's input is
   provably uniform in the kernarg-derived SRD-construction path").
   A per-use-chain forward walk from every readlane result
   (instrumented on `rewrite_cross_lane_divergent.cpp`'s
   `classifyForwardUseChain` with an `HSA_HOTSWAP_CLASSIFY_DEBUG=1`
   env switch) refuted that hypothesis: the blocking `readfirstlane`
   call in the softmax refusal path is
   `@llvm.amdgcn.readfirstlane.i32(i32 %vgpr38.5)`, and the
   `%vgpr38.5` phi value's incoming-branch chain does trace back
   to `%rcp9380 = fdiv float 1.000000e+00, <reduction-sum>` — the
   softmax normaliser reciprocal — which itself is the forward
   closure of one of the 8 `v_readlane_b32 …, 31` reduction fan-in
   sites. So the refusal is **correct**: the source's
   `readfirstlane` assumes the normaliser is wave32-uniform (which
   is true on gfx1250 where the whole wave shares one reduction
   result); under MODREP cross-widening to wave64, each target
   wave64 contains two source-wave-32s with DIFFERENT normaliser
   values, and `readfirstlane` collapses to lane-0's value, wiping
   out source-wave-1's normaliser.

   This is a *C2-adjacent cross-widening-under-source-wave-
   uniformity* class, structurally different from the "convergent
   cross-lane inactive-lane leak" class above:
   - It is not about inactive lanes' VGPR contents — the lanes
     feeding the readfirstlane's input are all active and produce
     well-defined per-source-wave values.
   - It is about the source's use of `v_readfirstlane_b32` (or
     equivalent SGPR-forced sinks: `s_buffer_load` rsrc,
     `s_sendmsg` msg, etc.) as an explicit SCALARISATION of a
     "known-uniform-at-the-source" value, where the uniformity
     assumption is valid at `W_s` scope but breaks at `W_t > W_s`
     scope under MODREP.
   - It is not fixed by any of §5's O1–O4 as currently shaped:
     O2's `tid AND (W_s − 1)` mask rewrite is not applicable
     (there's no `tid` in the chain; the value is a legitimate
     reduction result), and O1's refusal is already what happens
     today.

   ThreadLoopProjection (O3) would cover it in principle — each
   MODREP replica runs serially, so the readfirstlane sees only
   its own replica's reduction result and the scalarisation is
   consistent with the per-replica source execution. The same §5
   O3 cost analysis applies.

   Evidence anchor: `HSA_HOTSWAP_CLASSIFY_DEBUG=1 raise_cli
   corpus_softmax_fp32.gfx1250.co --isa=gfx1250 --target-isa=gfx942
   --emit-ir` (debug switch is a temporary raiser-local
   instrumentation; not committed) prints the full 28k-line walk
   trace for softmax. The walk visits ~25k intermediate values on
   each of 8 readlane starts before hitting the blocking
   readfirstlane; the chain length rules out "classifier
   over-walks through an unrelated phi" — the path is genuinely
   contiguous through `bitcast`/`fdiv`/`phi`/`bitcast` with no
   cross-chain phi merges (verified by inspecting the walk tail
   leading into `%vgpr38.12 = phi [ %3784, … ], [ %vgpr38.11, … ]`,
   where `%3784 = bitcast float %rcp9380 to i32` and the rcp is
   directly on the chain from a readlane).

   Net: softmax and the canary/rmsnorm recipes from §9 (5) both
   manifest cross-widening issues under MODREP that §5's option
   list doesn't cover today, but the underlying classes differ
   (inactive-lane VGPR gather vs source-wave-uniformity-collapsed-
   by-SGPR-sink). Both likely graduate together under a future
   ThreadLoopProjection design — the value of enumerating them
   here is so the "what O1+O2 DON'T cover" list for the predicate-
   chain work stays accurate.
6. **The predicate-chain class as §§4-5 describe it does not have
   a clean IR-level syntactic fingerprint (opened 2026-04-21 by
   Phase-2 start-of-implementation IR inspection).** O1's
   classifier as specified — "refuse any `llvm.amdgcn.workitem.id.x()`
   whose transitive uses reach an `icmp` gating a side effect
   without being AND-masked by `W_s − 1` somewhere on the chain" —
   cannot be narrowed to distinguish the failing recipes from the
   currently-passing baselines at the IR level. Evidence collected
   by running `raise_cli --emit-ir` on each recipe's gfx1250 `.co`
   and inspecting the post-mem2reg IR:

   | recipe | verdict today | relevant tid-chain shape | K in the predicate icmp |
   |---|---|---|---|
   | `canary_bpermute_scan_fp32` | WRONG | `%tid` → scan-stage `icmp ult K, %tid` → br → store | **compile-time constants 1, 3, 7, 15** (all < W_t) |
   | `rmsnorm_fp32` | WRONG | `%tid` → `icmp sgt %arg4, %tid` → ballot → new_exec → SPE diamond → store | **dynamic kernarg** (`%arg4`) |
   | `swiglu_fp32` | WRONG | `%tid` → phi(unmasked + W_s-masked paths) → or → phi → `icmp sgt %arg2, %vgpr0.1` → ballot → new_exec → SPE diamond → store | **dynamic kernarg** (`%arg2`) |
   | `corpus_layernorm_fp32` | WRONG | same shape as rmsnorm | **dynamic kernarg** |
   | `vecadd_f16` (baseline, MATCH) | MATCH | `%tid` → `or 128, %vgpr0.16` / `or 256, %vgpr0.16` / … → phi → `icmp sgt %arg3, %vgpr0.16` → ballot → new_exec → SPE diamond → store | **dynamic kernarg** (`%arg3`) |
   | `rope_fp32` (baseline, MATCH) | MATCH | `%tid` → phi with unmasked path → `raw.buffer.store.i32(%vgpr0.2, …)` (store VALUE not pointer) | *no kernel-level predicate icmp* |
   | `canary_dpp_compound_add_fp32` (baseline, MATCH) | MATCH | `%tid` → scalar address arith → load/store (no tid-gated predicate visible) | *none* |

   Two narrowing rules fall out of this data, and both fail at
   least one recipe the user's `o1_o2_partial` choice expects to
   behave a particular way:

   - **"Refuse any `tid → icmp → side-effect` (no mask)".** The
     design doc's literal wording. Catches every failing recipe.
     Also catches `vecadd_f16` (same chain shape, also reaches a
     mask-gated store through the ballot/new_exec path). Violates
     the baseline-non-refusal contract in the user's instructions
     ("narrow the classifier, do NOT relax the no-silent-fallback
     rule").
   - **"Refuse only when the icmp's non-tid operand is a
     compile-time constant `K ≤ W_t`"**. Catches
     `canary_bpermute_scan_fp32` (K ∈ {1, 3, 7, 15}). Does NOT
     catch `rmsnorm_fp32`, `swiglu_fp32`, `corpus_layernorm_fp32`,
     `vecadd_f16`, or `rope_fp32` — their kernel-level icmps all
     compare against a dynamic kernarg, not a compile-time
     constant. Satisfies the baseline-non-refusal contract but
     only turns 1 of 4 silent-WRONG recipes into loud-refused
     (vs. the 4/4 the `o1_o2_partial` option text expects).

   There is no known intermediate narrowing that distinguishes
   failing `rmsnorm_fp32` / `swiglu_fp32` / `corpus_layernorm_fp32`
   from passing `vecadd_f16` by looking at the tid-chain structure
   alone. `vecadd_f16` has the exact same shape — a `tid`-derived
   value feeding `icmp sgt %arg<N>, %vgprX.Y` that ballots into
   `new_exec` and then gates a store through an SPE diamond — and
   it produces MATCH on every shape today. Any classifier that
   refuses the norm-family recipes refuses `vecadd_f16` too.

   Stronger implication (supersedes the earlier O2-coverage note
   in §9 (2)): **O2's `tid AND (W_s − 1)` mask is not correct for
   the norm-family recipes either**, even under the more permissive
   interpretation of sub-case 1. Consider `swiglu_fp32`'s predicate
   `icmp sgt %arg2, %vgpr0.1` at `blockDim.x = 128 = num_warps × W_s`:
   target wavefront 0 spans WG tid 0..63. Source wave 1 (the
   replica-1 lanes in target wavefront 0, WG tid 32..63) was
   compiled expecting its lanes to see WG tid 32..63 in
   `workitem.id.x()`. Masking their `%vgpr0.1` by 31 would force
   their predicate to evaluate `arg2 > (0..31)` instead of the
   source's intended `arg2 > (32..63)`. For most N values both
   evaluate to "true" (active-arm) across 0..63, so the mask
   probably happens to produce numerically-correct output on most
   shapes — but that is a coincidence of N being "large", not a
   principled preservation of source semantics. For
   `canary_bpermute_scan_fp32` the mask is a no-op (as
   already noted in §9 (2)).

   The implication list under §9 (2) and the "Picked: O1 + O2"
   paragraph appended to §6 are written on the assumption that
   sub-case 1 is a principled application domain for the mask
   rewrite. Phase-2 IR inspection makes this less clear for the
   norm-family recipes than the earlier Phase-0 `block > W_t`
   heuristic suggested.

   Given this, the current plan ("land O1 as narrow as possible,
   then O2 as a mask rewrite") may not be the right path for
   these recipes. The cleanest bail-out points for the human to
   consider:

   - **Land only the narrow O1** that refuses on compile-time
     `K ≤ W_t`. This flips `canary_bpermute_scan_fp32` from
     silent-WRONG to loud-refused (1 recipe). Baselines stay
     MATCH. `rmsnorm_fp32` / `swiglu_fp32` / `corpus_layernorm_fp32`
     stay silent-WRONG. The §9 (5) / §9 (6) classes are still
     open questions for a separate design iteration.
   - **Defer O1 entirely** pending a fresh design iteration that
     characterises the norm-family miscompile with a real
     single-element trace (what bit pattern leaks where?), rather
     than the §4.3 sub-case narrative the Phase-0 findings have
     eroded.
   - **Accept O1 as specified verbatim** (refuse every
     `tid → icmp → side-effect` without a mask). Baselines
     `vecadd_f16` / `rope_fp32` (if the classifier catches it) /
     some of the compare_correctness HIP recipes would also
     refuse. This trades "cover all 4 failing recipes" for
     "refuse some currently-passing kernels". Explicit violation
     of the user's baseline-non-refusal constraint.

   I stopped before writing any classifier code, per the
   project's "STOP writing code on falsifying findings" rule.
   Enum / diagnostic additions (`ObstructionKind::WorkitemIdPredicateChain`,
   `RaiseFailureReason::CrossWavePredicateChain`, the
   `c5_predicate_chain_classifier.hpp` draft, and the
   `selectFailureFromReport` case) that I had in flight have been
   reverted.
7. **§9.7 SPE-phi-undef hypothesis — FALSIFIED (opened + closed
   2026-04-21 by the Step-A/B empirical investigation of
   WaveNative graduation).**

   Hypothesis (per §9.5 / §9.6 of the pre-graduation design): the
   `canary_bpermute_scan_fp32` miscompile under MODREP (and,
   newly, under WaveNative) is caused by the SPE diamond's phi
   pattern `%vgpr_N = phi [%loaded, %spe_do], [undef, %spe_skip]`.
   Inactive source lanes take `spe_skip`; `PromoteMemToReg`
   materialises their phi as `undef`. A convergent cross-lane op
   (`ds_bpermute`, `permlanex16`, DPP) then reads the undef VGPR
   as its DATA input across all 64 hardware lanes. Under
   `init_whole_wave` + WaveNative (HW EXEC = -1), inactive-source
   lanes execute but their SPE-gated VGPR-alloca writes are still
   discarded by the AMDGPU backend's HW-EXEC-masked store lowering
   of the mem2reg-introduced phi.

   Experiment (`HSA_SALMON_VGPR_SPE_BYPASS=1`, env-var-gated patch
   in `raise_context.cpp` that replaces `emitUnderExec([&]{
   storeVGPR*(...) })` with the direct `storeVGPR*(...)` call for
   the VGPR / AGPR arms of `writeReg*` / `storeVGPR*` / `storeAGPR*`).
   Under the bypass, VGPR writes from load handlers run
   unconditionally; under WaveNative's `init_whole_wave` ambient
   HW EXEC = -1 all 64 lanes commit the computed value to their
   VGPR alloca. Predicted result: active lanes' bpermute gathers
   see defined values from every lane, `canary_bpermute_scan_fp32`
   flips from WRONG to match.

   Empirical result: **falsified**.

   - `canary_bpermute_scan_fp32` under WaveNative + SPE-bypass
     produces *identical* WRONG numerical output to WaveNative
     alone (and to MODREP). `max|err|` 6.22701, 15.4385, 18.4618,
     21.7873 on N=128/1024/8192/65536 respectively. Reproducible
     byte-for-byte across multiple runs.
   - `canary_permlanex16_rowmax_fp32` at N_ROWS=8192 regresses
     from match to EXIT=2 under the bypass — the SPE-bypass
     introduces a separate hazard at scale.

   Conclusion: the SPE-phi-undef pattern is NOT the mechanism of
   `canary_bpermute_scan_fp32`'s miscompile. The bug is
   projection-independent (identical output under MODREP,
   WaveNative, and WaveNative + SPE-bypass) and upstream of the
   projection layer entirely. Suspect surfaces for follow-up:

   - `handle_ds.cpp`'s `DS_BPERMUTE_B32` handler's IR shape
     (selector byte-to-lane scaling, data operand routing).
   - The scan's ambient-arithmetic handlers (`V_ADD_F32`,
     `V_CNDMASK_B32`, per-stage predicate emission) — Triton's
     gfx1250 cumsum might emit an instruction pattern that a
     handler lifts with subtle off-by-one / sign-extension bug.
   - The source-binary → SemOp decode for some specific
     instruction in the scan stages.

   Method: single-element trace. Pick one output slot where
   salmon produces WRONG output, instrument the scan stages on a
   side channel, compare against the native gfx942 build's
   arithmetic path element-by-element. The experiment code was
   reverted; no new code remained under investigation.

   This finding removes the §9.5 "convergent-cross-lane-op
   inactive-lane leak" class from the graduation's unresolved
   list — the original diagnosis was wrong, so the class doesn't
   exist as framed. `canary_bpermute_scan_fp32` stays under "open
   mechanism" until the single-element trace runs.

   ---

   **Mechanism IDENTIFIED 2026-04-22** (single-element trace at
   N=128 via `HSA_SALMON_DUMP_DIFF`):

   First divergence: index 8, salmon +3.4693 vs native +1.4988,
   delta +1.9706 ≈ 2 × (native[8] − native[0]) =
   2 × (1.4988 − (−0.2359)) = 2 × 1.7347 = 3.4694. Dead ringer
   for "stage-3 lane 8 fadd'ed with itself instead of lane 0".

   Root cause: **VOPD `v_dual_cndmask_b32` handler hardcoded VCC
   as the condition**, ignoring the explicit scalar condition
   operand gfx1250's VOPD encoding carries. Native scan body at
   `canary_bpermute_scan_kernel+0x194` (and mirrors at each
   distance stage) is:

   ```
   v_dual_cndmask_b32 v9, v9, v1, s0 :: v_dual_cndmask_b32 v2, v2, v7, vcc_lo
   ```

   — the first half is the stage-3 selector advance guarded by
   `s0 = (tid < 8)` (tells lanes < 8 to read themselves, lanes
   >= 8 to read `(tid - 8) * 4`), the second half is the stage-2
   value update guarded by `vcc = (tid > 3)` (lanes > 3 take the
   fadd result). `handle_vopd.cpp::v_cndmask_b32` routed BOTH
   halves' condition through `ctx.regs.loadVCC(ctx.B)`, so the
   selector advance inherited `(tid > 3)` from the paired
   instruction's vcc producer. Lanes 8..31 then got
   `(tid > 3) = true` → pick "self" (`vlshl`) instead of
   `(tid - 8) * 4`, making stage-3's bpermute read each lane's
   own value, and the subsequent fadd doubled the lane's stage-2
   partial sum. That's exactly the observed `2 × (lane_L - lane_0)`
   error pattern.

   Fix (2026-04-22, commit TBD): mirror the non-VOPD `V_CNDMASK_B32`
   handler in `handle_valu_vop3p.cpp`. Parse `operands[3]` of the
   VOPD half:

     * `vcc_lo` / `vcc` → `loadVCC` (old default, now explicit).
     * `sN` → prefer the per-BB V_CMP shadow `i1` from
       `lookupSgprWaveMaskI1(N)`; fall back to
       `projection.extractLaneBitFromWaveMask` on the raw SGPR
       alloca.
     * absent → `loadVCC` (safety fallback for VOPD encodings that
       don't surface a condition operand).

   Scope of the fix — `compare_correctness` salmon-path sweep
   post-fix vs pre-fix (WaveNative default, all 12 Triton
   recipes):

   | recipe | pre-fix | post-fix |
   |---|---|---|
   | `canary_bpermute_scan_fp32` | WRONG 4/4 | **match 4/4** |
   | `corpus_layernorm_fp32` | WRONG @ N=128/256/512 (max\|err\| 0.128/0.034/0.016) | **match 4/4** |
   | `swiglu_fp32`, `rmsnorm_fp32`, `rope_fp32`, `vecadd_f16`, `corpus_add_fp32`, `corpus_asin_fp32`, `canary_dpp_compound_add_fp32`, `canary_dpp_reduce_fp32`, `canary_permlanex16_rowmax_fp32` | match 4/4 | match 4/4 |
   | `corpus_softmax_fp32` | EXIT=2 (writelane safety net) | EXIT=2 (unchanged; orthogonal) |

   The `corpus_layernorm_fp32` small-N residual fix is a
   BONUS — Triton's LayerNorm kernel also emits VOPD
   `v_dual_cndmask_b32 vX, ..., sN` pairs in its mean-of-squares
   reduction path, and the same hardcoded-VCC bug silently
   corrupted 20 of 128 output rows at N=128 (max\|err\| 0.128
   pre-fix; the "20 of 128" matches the count of rows where the
   reduction tree's partial-sum cndmask fell on a non-self-
   cancelling lane pair). This collapses `§9.8` entirely — the
   small-N residual is the same bug as `canary_bpermute_scan_fp32`,
   the "1024 matches" was the error-rounding-into-tolerance
   artifact of larger reduction trees diluting the per-pair
   miscompile.

   Regression gate: `compare_correctness` salmon-path on the
   full Triton corpus + lit fixture
   `lit_tests/v_dual_cndmask_b32_sgpr_cond/` (added with the fix
   commit) pins the VOPD SGPR-condition shape so a future
   refactor that re-introduces the VCC-hardcoded path fails
   here rather than silently miscompiling the scan corpus.

   This supersedes §9.8's "small-N residual" open question:
   fixed by the same commit.

8. **Corpus_layernorm small-N residual (opened 2026-04-21 by
   WaveNative graduation; CLOSED 2026-04-22 — same VOPD
   `v_dual_cndmask_b32` SGPR-condition bug as §9.7; fixed by the
   same commit).**

   Under WaveNative, `corpus_layernorm_fp32` matches at N=1024
   but remains WRONG at N=128/256/512. The residual error
   magnitudes are tiny (max\|err\| 0.016 at N=512, 0.034 at N=256,
   0.128 at N=128) — 20×–80× smaller than the pre-WaveNative
   MODREP error (max\|err\| up to 2.7 at N=128). Error shrinks as
   N grows.

   Hypotheses worth investigating:

   - Reduction-ordering difference between the raised gfx942
     kernel and the Triton-native gfx942 gold, amplified by the
     smaller reduction tree at smaller N.
   - Rounding-mode or FP-precision drift in a handler (`v_rsq_f32`
     / `v_rcp_f32` Newton iteration paths — the same cluster that
     the prior `v_div_scale_f32` commit tightened orthogonally
     and incidentally fixed `rmsnorm_fp32`).
   - Sub-case-2 interaction specific to WaveNative on smaller
     blockDim.x ranges the pre-graduation sweep didn't observe as
     distinct.

   Method: same single-element trace on N=128 of `corpus_layernorm_fp32`.
   Compare the raised mean + var computation against the Triton
   native gold; find the first divergence.

9. **§6 graduation numbers falsified by controlled MODREP/WaveNative
   sweep (opened 2026-04-21 during classifier cleanup reflection,
   post-graduation commit `c3cc463112`; PARTIALLY SUPERSEDED
   2026-04-22 by §9.7's VOPD SGPR-condition fix — see
   "Post-fix status" below).**

   During the principled-cleanup pass on the C5 classifier, a
   controlled three-way sweep ran compare_correctness on the full
   Triton corpus under both projections by temporarily forcing
   `enableWaveNative=false` in `loader/executable.cpp`'s
   `runPipelineAllKernels` invocation (reverted before commit).
   Classifier was the post-cleanup version (narrowed back from the
   cross-subtree widening, equivalent in refusal scope to the
   original narrow-O1). Post-`df6bf1d35a` MODREP compared against
   post-graduation WaveNative:

   | recipe | MODREP (forced, post-df6bf1d35a) | WaveNative (default) |
   |---|---|---|
   | `swiglu_fp32` | **match 4/4** | match 4/4 |
   | `rmsnorm_fp32` | match 4/4 | match 4/4 |
   | `corpus_layernorm_fp32` | WRONG @ N=128/256/512 (max\|err\| 0.128/0.0342/0.0158), match @ N=1024 | **bit-for-bit identical** numerics to MODREP |
   | `canary_bpermute_scan_fp32` | **EXIT=2 (C5 refused 4/4, loud)** | WRONG 4/4 (silent) |
   | `corpus_softmax_fp32` | EXIT=2 | EXIT=2 |
   | 7 baselines (`vecadd_f16`, `rope_fp32`, `corpus_add_fp32`, `corpus_asin_fp32`, `canary_dpp_compound_add_fp32`, `canary_dpp_reduce_fp32`, `canary_permlanex16_rowmax_fp32`) | match 4/4 | match 4/4 |

   Spot-check on `Gfx1250Gpu.Matmul128x128_1tile` (all six row-pattern
   sub-tests): passes **under MODREP** when `enableWaveNative=false`
   is explicitly threaded into the gfx1250_gpu_test call — despite
   the test-site comment asserting the Wave64-collective
   correctness invariant requires WaveNative. Either the invariant
   is no longer needed (`df6bf1d35a` or an earlier commit
   incidentally fixed the upper-half issue) or the Uniform/RowId
   patterns happen to not surface the defect; either way the
   test-site comment is stale.

   **What this falsifies.** The §6 "Picked: WaveNative as default"
   graduation paragraph claims:

   - `swiglu_fp32` flips WRONG 4/4 → match 4/4 under WaveNative.
     **False.** MODREP post-`df6bf1d35a` also produces match 4/4.
     The pre-graduation "WRONG 4/4 under MODREP" was measured
     BEFORE `df6bf1d35a` landed, and the WRONG→match attribution
     belongs to `df6bf1d35a`, not to WaveNative.
   - `corpus_layernorm_fp32` partial-matches under WaveNative with
     20×–80× smaller error than MODREP. **False.** The
     max\|err\| numerics are bit-for-bit identical under both
     projections post-`df6bf1d35a`.
   - `canary_bpermute_scan_fp32` is unaffected by the graduation.
     **False in the worst direction.** MODREP refuses loudly (C5
     classifier); WaveNative silently miscompiles. The
     graduation converted a loud refusal into a silent
     miscompile for this recipe.

   **Net effect of the graduation on compare_correctness.** WaveNative
   default vs MODREP post-`df6bf1d35a`: **zero new matches, one
   regression** (`canary_bpermute_scan_fp32` loud-refused → silent-
   WRONG). On the Triton corpus alone the graduation was a strict
   regression.

   **What the graduation may still be right about.** Not tested by
   this sweep:

   - Non-matmul WMMA recipes on multi-tile / multi-wave shapes.
     `Gfx1250Gpu.Matmul128x128_1tile` is single-tile; the original
     §5.6.1 Wave64-collective argument was specifically about
     multi-tile fan-out where lanes 32..63 carry distinct source-
     wave-3 state.
   - AITER corpus / gpt-oss matmul kernels raised through the
     runtime hook (not via direct `runPipeline` calls). These go
     through the `loader/executable.cpp` salmon path and would
     pick up whatever default is set.
   - Pipeline interactions where MODREP's replica model breaks on
     `num_warps > 1` without the recipe also surfacing a numerical
     miscompile in the specific compare_correctness shape.

   **Possible responses (scoping note for the human):**

   - **(a) Revert the graduation commit `c3cc463112`**. Restore
     `enableWaveNative=false` as the default. Matmul128x128_1tile
     adds `enableWaveNative=true` at its call site (already there
     today, no change needed). AITER corpus and gpt-oss matmul
     surfaces would regain the pre-graduation MODREP default; spot
     test to confirm they still raise cleanly.
   - **(b) Keep the graduation but rewrite §6**. Acknowledge the
     sweep numbers as-measured-today, explicitly attribute
     swiglu_fp32 / corpus_layernorm_fp32 changes to `df6bf1d35a`
     rather than WaveNative, document canary_bpermute_scan_fp32
     as a REGRESSION the graduation accepted, and keep WaveNative
     as default purely on the §5.6.1 / wmma-safety argument for
     kernels that compare_correctness does not exercise (multi-
     tile WMMA, `num_warps > 1` edge cases).
   - **(c) Split the default by workload class**. Gate the
     `enableWaveNative=true` default on `hasWMMA(insts)` or
     equivalent at the raiser. Same-shape kernels keep the
     pre-graduation MODREP path with its loud-refusal on C5;
     WMMA-containing kernels flip to WaveNative.

   **What the cleanup does NOT decide.** The cleanup commit keeps
   the graduation in place (`enableWaveNative=true` default)
   because the graduation was already landed and the user's
   cleanup prompt is scoped to fixing the principled-reflection
   issues, not re-deciding the graduation. This §9 entry is the
   falsifying-evidence record that the principled prompt's
   "record in §9 and surface to the human" rule requires when
   design-doc numbers are falsified mid-implementation. The
   scoping options above are for the human to pick; the cleanup
   commit proceeds on the graduation-kept default.

   Evidence anchor: the sweep ran at `0e5a34cea7` + cleanup-in-
   progress, with the TEMP-SWEEP patch
   `runPipelineAllKernels(..., enableWaveNative=false)` in
   `loader/executable.cpp` (reverted pre-commit). Reproducer:
   apply the TEMP-SWEEP patch, rebuild `hsa-runtime64`, run
   `./compare_correctness --recipe=<any_triton>` with
   `LD_PRELOAD=./libsalmon_intercept.so` and
   `LD_LIBRARY_PATH=<build>/rocr/lib`.

   ---

   **Post-fix status (2026-04-22).** §9.7's VOPD SGPR-condition
   root cause and fix shift the evidence substantially but do
   not fully rehabilitate §6's graduation claims:

   | recipe | MODREP post-VOPD-fix | WaveNative post-VOPD-fix |
   |---|---|---|
   | `canary_bpermute_scan_fp32` | EXIT=2 (C5 refused, unchanged) | **match 4/4 (WRONG → match)** |
   | `corpus_layernorm_fp32` | expected: match (post-VOPD-fix — NOT yet re-measured under MODREP) | match 4/4 |
   | `swiglu_fp32`, `rmsnorm_fp32`, baselines | match 4/4 (unchanged) | match 4/4 (unchanged) |
   | `corpus_softmax_fp32` | EXIT=2 (writelane safety net, unchanged) | EXIT=2 (unchanged) |

   The **revised net effect** of the graduation on
   `compare_correctness` is now:

   - MODREP path: loses `canary_bpermute_scan_fp32` numerically
     (the scan was already loud-refused by C5 under MODREP, so
     the VOPD fix doesn't flip its MODREP verdict), keeps all
     baselines + `corpus_layernorm_fp32` now expected to match
     (per-VOPD-fix).
   - WaveNative path: **match 4/4 on all Triton recipes except
     `corpus_softmax_fp32`** (which is EXIT=2 via the
     writelane safety net, a separate class). Specifically,
     WaveNative now covers:
       * `canary_bpermute_scan_fp32` — the scan-corpus canary,
         now MATCH under WaveNative (via the VOPD fix); under
         MODREP it's still C5-refused (loud), so only WaveNative
         gets a successful-match for this recipe.
       * `corpus_layernorm_fp32` — all 4 shapes match; the small-
         N residual attributed to "reduction-ordering drift" in
         the pre-fix §9.8 was actually the same VOPD SGPR-
         condition bug.

   **What this means for the options in this section:**

   - Option (a) "revert graduation" is now strictly worse than
     before: canary_bpermute_scan_fp32 would regress from "match
     under WaveNative default" to "loud-refused under MODREP
     default" (a loss of numerical match, even if the refusal
     is the principled loud signal).
   - Option (b) "keep graduation, rewrite §6" is now the
     principled choice: WaveNative default + VOPD fix produces
     match on 11/12 Triton recipes (the 12th, corpus_softmax, is
     an orthogonal writelane-class EXIT=2 already documented in
     §6 / §9.8). The graduation's correctness domain is real
     post-VOPD-fix; the pre-VOPD-fix §6 claims were just
     mis-attributing a bug (VOPD cndmask) to a projection choice
     (MODREP-vs-WaveNative).
   - Option (c) "split default by workload class" is no longer
     necessary — the Triton corpus no longer distinguishes
     MODREP-safe from WaveNative-safe in a principled way
     post-fix.

   §6 should be rewritten once corpus_softmax_fp32's writelane
   EXIT=2 is triaged (separate class) to reflect the post-fix
   state: WaveNative is the default; MODREP is the opt-in for
   projection-specific debugging / lit-fixture pinning. The
   empirical evidence is now clean and monotonic.

10. **Carry-chain SGPR-operand silent fallback — class identified
    + proactively closed 2026-04-22 (sibling of §9.7).**

    Post-§9.7 audit of `loadVCC` / `storeVCC` call sites found six
    handlers in `handle_valu.cpp` with the same latent bug shape as
    the VOPD `v_dual_cndmask_b32` miscompile: the
    V_{ADD,SUB,SUBREV}_CO_(CI_)U32 family's e64 / VOP3B encoding
    carries an EXPLICIT scalar operand for carry-in (ci variants,
    MC src index 2) and carry-out (MC def index 1) that can bind
    to either `vcc_lo` / `vcc` OR an arbitrary `sN` — the compiler
    picks based on SGPR pressure. Pre-fix the six handlers routed
    both endpoints through `loadVCC` / `storeVCC` unconditionally,
    silently mis-wiring any kernel that bound the carry to a
    non-VCC SGPR.

    **Empirical corpus exposure pre-fix: zero recipes.** Sweep of
    every .co in the compare_correctness Triton corpus
    (gfx1250 + gfx942), the AITER TensileLite corpus (~290
    binaries), and the hotswap-testing reference kernels:

    - Triton gfx1250: uses `v_add_nc_u32` / `v_add_nc_u64`
      (gfx1250 added a native no-carry 64-bit add). No carry-chain
      instruction in the disassembly of any recipe.
    - Triton gfx942: uses `v_add_u32` (no-carry on gfx9) or the
      fused `v_lshl_add_u64` for address arithmetic. No carry
      chain.
    - AITER TensileLite: uses `v_add_co_u32` / `v_add_co_ci_u32_e64`
      but ALWAYS with `vcc_lo` as the scalar carry. The compiler's
      default path when VCC is live-available.

    So the latent bug was a pure silent fallback — correct in
    practice on every kernel the corpus exercises today, wrong
    if/when a compiler picks non-VCC SGPR carry (e.g. under VCC
    pressure, or a compiler-backend-specific choice, or a
    hand-written assembly kernel).

    **Fix.** `handle_valu.cpp` grows two file-local helpers —
    `readCarryInI1` and `writeCarryOutI1` — that mirror the
    SGPR-aware routing the non-VOPD V_CNDMASK_B32 handler has
    always done (`handle_valu_vop3p.cpp`'s e64 path):

      * SGPR carry-IN: `ctx.lookupSgprWaveMaskI1(N)` fresh-shadow
        lookup; fall back to
        `projection.extractLaneBitFromWaveMask` on the raw SGPR
        alloca (same lossy-under-cross-widening residual as the
        V_CNDMASK_B32 consumer).
      * SGPR carry-OUT: ballot i1 → source-wave-mask-width via
        `projection.ballotI1ToWidth`; `writeRegExecWidth` into the
        SGPR alloca; `recordSgprWaveMaskI1` to populate the fresh
        shadow for same-BB consumers.
      * VCC / NOREG / absent: preserve the pre-fix `loadVCC` /
        `storeVCC` path bit-exactly (zero regression risk on the
        all-VCC common case).

    Applied to all six handlers: V_ADD_CO_U32, V_SUB_CO_U32,
    V_SUBREV_CO_U32, V_ADD_CO_CI_U32, V_SUB_CO_CI_U32,
    V_SUBREV_CO_CI_U32.

    **Regression gate.** New lit fixture
    `lit_tests/v_add_co_u32_sgpr_carry/` pins the post-fix IR
    shape end-to-end: a kernel forces `v_add_co_u32 ..., s0, ...
    :: v_add_co_ci_u32_e64 ..., s0, ..., s0` via inline asm (the
    exact 64-bit address-chain shape a VCC-pressured compiler
    would emit), and FileCheck verifies (a) the carry-out
    ballots via `amdgcn.ballot.i64` into the SGPR path, (b) the
    second add's carry-in reads the SAME `i1` SSA value the
    first add produced (fresh-shadow forwarding — not a fresh
    `load i1, ptr %vcc`), (c) no `load i1, ptr %vcc` across the
    pair's span. A regression that re-introduces the
    hardcoded-VCC path breaks the SSA chain and fails the
    fixture loudly.

    **Doesn't fix anything empirically today** — no corpus
    kernel was silently miscompiling pre-fix. Closes the class
    proactively because it's the SAME pattern as the VOPD
    cndmask bug that DID miscompile end-to-end, and "never do
    silent fallbacks" is a non-negotiable project rule.
