# Learnings

Short, dated entries capturing what broke, why it broke, and the fix.
Written for the reader in a future where everything works — so they can
reconstruct the trap that was there, not the solution that's obvious now.

Append-only. Newest on top.

---

## 2026-04-22 — V_FMA_MIX inline-constant narrow-half: op_sel misapplied, silent bf16-reduction miscompile (closes topk_forward_bisect_m1_const_in)

**Context.** Every bf16-in + reduction + bf16-out Triton kernel was
silently miscompiled under cross-widening (gfx1250 → gfx942).  The
end-to-end signature: feed a 32-column `tl.sum(axis=1)` of bf16
`1.0`s — native writes `32.0`, salmon writes `1.0`.  Every reduction
step's multiplier silently evaporated and every bf16 accumulation
became a no-op.  Affected `topk_forward_bisect_m1..m4` and
`topk_forward_bf16`; a multi-day sequence of false leads (DPP
rewrite, EXEC-mask tracking, strict.wwm, d16_hi store) came and went
before the all-ones probe localised it.

**Root cause.** `V_FMA_MIX_F32` / `V_FMA_MIX_F32_BF16` encode narrow
sources via an `op_sel_hi[i]=1` flag plus an `op_sel[i]` VGPR-half
selector (0=low 16, 1=high 16).  `op_sel[i]` is a VGPR-half
selector — it assumes the 32-bit source holds two packed 16-bit
values so either half is a valid narrow datum.  For register
sources the assumption holds.  For INLINE CONSTANTS (and 32-bit
literals in narrow-operand slots) it does NOT:

  LLVM's AMDGPU disassembler pre-resolves narrow inline constants
  to the 16-bit value in the LOW 16 of the MCOperand Imm, upper 16
  zero-extended — `AMDGPUDisassembler.cpp::decodeMCOperand`, under
  `OPERAND_REG_INLINE_C_BF16` / `OPERAND_REG_IMM_BF16` /
  `OPERAND_REG_INLINE_C_FP16` / `OPERAND_REG_IMM_FP16` arms and
  their `getInlineImmValBF16` / `getInlineImmValF16` helpers.

The pre-fix handler applied op_sel unconditionally:

    if (opSel[i] == 0) bits = trunc(raw, i16);                // LO
    else               bits = trunc(lshr(raw, 16), i16);      // HI

For an inline bf16 `1.0` (= MC Imm `0x00003F80`) with
`op_sel[i]=1`, the handler ran `trunc(lshr(0x3F80, 16)) = 0x0000 =
bf16 0.0`.  The downstream shape `fma(bf16_val, 0.0, acc) = acc`
was an identity over all 32 reduction steps.

Triton's bf16 `tl.sum` / `tl.max` compiles to exactly this pattern
on gfx1250: `v_fma_mix_f32_bf16 v_acc, v_bf16, 1.0, v_acc
op_sel:[0,1,0] op_sel_hi:[1,1,0]`.  So every bf16 Triton kernel
fed a field of zeros through its reduction while the raised IR
looked structurally identical to native.

**Fix (commit 0d002aecf2).** In
`handle_valu_vop3p.cpp::readMixSrc`, when the operand slot is
non-register (`!op.isSrcReg(i)` — inline constant, 32-bit literal,
or expression), skip the op_sel-based half extraction and take the
LOW 16 unconditionally.  Register sources retain op_sel.  The
`op_sel` bit is VGPR-slot state; it has no meaning for a pre-
resolved immediate whose bit pattern IS the narrow value.

    bool isImmediateOperand = !op.isSrcReg(i);
    if (!isImmediateOperand && opSel[i] == 1)
      bits = trunc(lshr(raw, 16), i16);   // HI 16 (register half)
    else
      bits = trunc(raw, i16);             // LO 16 (immediate OR
                                          // register op_sel[i]=0)

**How it was pinned — the "constant-input probe" technique.**
Random-input probes (`topk_forward_bisect_m1`) showed noisy,
row-dependent wrong-to-right ratios (3.09x, 4.53x, 0.09x, ...) —
not a clean "subset of K elements summed" signature, because
partial sums of random values don't have stable ratios across rows.

The crucial move: replace the random input with a CONSTANT.  With
all X[r, c] = 1.0 and a 32-element `tl.sum(axis=1)`, the expected
output is exactly 32.0 for every row.  Any deviation becomes
VALUE-INDEPENDENT — the signature tells us HOW MANY elements
salmon actually summed, not what mix of them.

  * salmon writing `N × const` for some `N != 32` → reduction
    tree dropping `(32 - N)` contributions.  Examine which
    structural paths are skipped.
  * salmon writing `const` (N=1) → reduction is a full identity.
    The multiplier path is gone.  Inspect the CONSTANTS flowing
    into the reduction ops in the lifted IR.

Salmon wrote 1.0 (N=1).  Zero contributions.  Lifted-IR inspection
found every `@llvm.fma.f32` had `float 0.000000e+00` as its second
argument (should have been 1.0).  That constant literally doesn't
come from anywhere other than the FMA_MIX handler's readMixSrc —
root cause trivially localised from there.

**Wrong hypotheses enumerated and ruled out (two-day chase).**

- **H1 — DPP → ds_bpermute rewrite broke EXEC convergence.**  The
  ds_bpermute emitted inside an `emitUnderExec` diamond reads 0
  from EXEC-inactive source lanes (AMDGPU LDS bpermute semantics),
  whereas native DPP reads VGPR content regardless of EXEC.
  Plausible shape for the miscompile.  Ruled out by temporarily
  gating the rewrite off (one-line comment in
  `rewrite_cross_lane_divergent.cpp`'s site-collector): the
  const-input probe still produced 1.0 with faithful-lift
  `@llvm.amdgcn.update.dpp` in place.
- **H2 — `strict.wwm` would force wave-wide EXEC around the
  bperm.**  Followed from H1 as a candidate fix.  Wrapping the
  bperm result in `@llvm.amdgcn.strict.wwm` changed nothing.
  Reverted.
- **H3 — `permlanex16` emulation mis-reads source lane.**  The
  existing `canary_permlanex16_rowmax_fp32` matches, so the
  emulation is sound in isolation.  The m1 miscompile surfaces
  only in composition with FMA_MIX + packing, not in permlanex16
  alone.  Composition-level suspicion was plausible but empirically
  wrong — fixing FMA_MIX dropped m1's error by the full
  accumulator magnitude, and the residual is bf16 reduction-order
  drift.
- **H4 — `s_pack_ll_b32_b16` produces wrong halves.**  The scalar
  pack reads `low16 | (low16 << 16)` on two SGPR sources.  If one
  source had garbage in its low 16, the pack would carry it.
  Ruled out by tracing the pack's source SGPRs back to their
  `v_readlane_b32 s, v, 31` producers: those readlanes were on
  the downstream side of the reduction, not the source side.  A
  pack-side bug would have shown a different signature (wrong
  halves, not zero-accumulation).
- **H5 — `emitUnderExec` models inactive lanes with UNDEF phis;
  downstream cross-lane reads see UNDEF.**  Structurally possible
  but not the active bug here: the FMA_MIX fix resolved the
  symptom, proving the cross-lane read path was correct all
  along.  Remains a reasonable concern to audit in isolation if
  a future bug surfaces that FMA_MIX doesn't explain.

**Graduation.**

- `topk_forward_bisect_m1_const_in`: WRONG 2048/2048 (output 1.0)
  → `match` (output 32.0).
- `topk_forward_bisect_m1` (random): 2048/2048 WRONG `max|err|=39`
  → 1300/2048 WRONG `max|err|=0.25` (≤ 2 bf16 ULPs at every
  mismatched magnitude; 1404/2048 ≤ 1 ULP; residual is
  non-associative bf16 reduction-order drift between Triton's
  gfx1250 and gfx942 tree shapes — not a miscompile).  Relaxing
  the comparator from `tol: 0.0` to a few bf16 ULPs would
  graduate m1 to `match`; kept untouched here so the regression
  surface stays bit-exact.
- `topk_forward_bisect_m2..m4`, `topk_forward_bf16`: improved
  proportionally; remaining mismatches are the same bf16
  rounding drift composed with softmax / sort / argmax
  sensitivity.
- Canary grid (6 DPP / permlane / readlane / cvt recipes)
  continues to match bit-exactly.
- `lit_tests/v_fma_mix_f32_bf16/` extended with two
  inline-constant FMA sites (`op_sel:[0,0,0]` and
  `op_sel:[0,1,0]`) that MUST produce `float 1.000000e+00` as the
  fma's second arg.  Negative pin rejects `float 0.000000e+00`
  feeding any `fma.f32` call in this fixture — locks in the
  pre-fix miscompile shape as a regression guard.

**Generalised rule for the `readMixSrc`-like family.**  Any
handler that reads an MCOperand and applies MC-encoding-specific
post-processing (op_sel half extraction, sign extension, neg/abs
modifier bits, packed-vs-unpacked interpretation) MUST branch on
`op.isSrcReg(i)` vs the immediate forms.  MC's pre-resolution
path is dtype-aware (`OperandType` → which inline-imm helper
runs) and strips modifier state that would have applied to a
VGPR operand.  Extending the current handler's op_sel /
op_sel_hi logic into a new neg/abs-carrying variant without
auditing the isSrcReg branch is how this class of bug returns.

**The constant-input probe approach as a methodology.**  See the
"Diagnostic technique — value-independent constant-input probes"
entry below for mechanisation notes; this bug is the reference
case the entry is written against.

---

## 2026-04-22 — Diagnostic technique — value-independent constant-input probes

**Problem class.**  Cross-widening miscompiles whose symptom is
noisy under random inputs — different rows show different
wrong-to-right ratios with no clean structural pattern.  Examples
from the corpus today include the FMA_MIX inline-constant bug
(entry above), the `_D16_HI` store-upper-half bug (commits
2ebfadeb95 / b827c55899), and the `topk_forward` / reduction
miscompile class generally.  Random-input probes surface the
symptom but can't localise it — the noise floor swamps every
structural signal.

**Technique.**  For any recipe that computes a deterministic
function over its inputs, replace random inputs with a CONSTANT
and compare salmon output to the analytically predicted result.

  * For a reduction `tl.sum(axis=1)` over N elements of value v,
    expected output = `N * v`.
  * For a reduction `tl.max(axis=1)` over N elements of value v,
    expected output = `v`.
  * For an elementwise `y = f(x)`, expected output = `f(v)` per
    slot.

Salmon's deviation from the analytic prediction is now
value-INDEPENDENT.  The deviation ITSELF carries structural
information:

  * Output = `v` for a reduction over N ≠ 1 → the reduction
    tree is an identity over the multiplier; the mul path is
    broken.  Inspect the CONSTANTS appearing in the lifted IR's
    reduction ops.  Wrong-constant-at-a-specific-slot is the
    smoking gun.
  * Output = `K * v` for some `K < N` → the reduction drops
    `(N - K)` contributions.  Inspect which structural paths are
    skipped.  Does `K` equal a lane count, a warp count, a row
    block count?  That's the dimension that was collapsed.
  * Output = `v * scalar_not_in_N`s-divisor-set` → FP arithmetic
    is happening but with the wrong multiplier somewhere.  Inspect
    constants AND the modifier flags (neg, abs, scale_sel).
  * Output = `v + offset` → an additive bias is leaking in.
    Could be a prior register state not cleared, or an init-bias
    (e.g. bf16 RNE `+0x7FFF`) surfacing into the output.

**Mechanisation — tractable today.**  For every
`compare_correctness` recipe with a deterministic reduction shape,
auto-synthesise a `_const_in` sibling recipe:

```python
# Sibling generator: given a base recipe, emit a _const_in variant
# with the same kernel but inputs filled to a single value.
const_in_recipe = {
  **base_recipe,
  "name": base_recipe["name"] + "_const_in",
  "inputs": [
    {**inp, "range_lo": 1.0, "range_hi": 1.001}  # tight range
    for inp in base_recipe["inputs"]
  ],
}
```

Run both the base recipe AND its _const_in sibling.  The _const_in
verdict is binary (match / wrong) — and if WRONG, the pattern of
deviation mechanically maps to a suspect class via the table
above.  This can be a CI gate on every recipe that has a reduction
primitive in its kernel AST.

**Mechanisation — harder.**  Automated lifted-IR constant
inspection: for any `@llvm.fma.f32` / `@llvm.fmuladd.f32` / other
arithmetic intrinsic in the raised IR, flag any LITERAL constant
operand that equals a "silently-damaging" value (0.0 as a
multiplier, 1.0 as an addend, NaN anywhere, etc.) and print the
MCInst operand it came from.  This is what a human does during
triage — the Cursor / grep workflow is already mechanical-adjacent.
A proper lint pass would live under `tools/ir_audit/` or similar
and run as part of `raise_cli --audit`.

**What NOT to mechanise (yet).**  Static analysis of handler
source for "this code applies MC-encoding-dependent logic without
checking isSrcReg first" — the static surface is too noisy;
legitimate `op.srcF(i)` calls do not need the isSrcReg branch
unless they read MC-encoding-state AFTER the read (op_sel, neg,
abs, clamp, scale_sel).  Expressing that "after the read" condition
cleanly in a linter is harder than just writing one _const_in
probe per recipe.

**Corollary — probe recipe hygiene.**  Every new recipe added to
the direct-invocation corpus SHOULD include a `_const_in` sibling
unless the kernel has no reduction or no element-wise op with a
per-element closed form.  The cost is low (~20 lines of Python);
the return is catching exactly this class of bug BEFORE it needs
a multi-day hunt through EXEC / DPP / permlane / strict.wwm false
leads.  See `topk_forward_bisect_m1_const_in.py` (commit
b77e477908) for the reference template.

---

## 2026-04-21 — Matmul128x128 residual: fixed by V_CMP → V_CNDMASK per-lane-i1 shadow (closes the whole family)

**Context.** Follow-up (and close-out) to the `warp-3-specific, K-iter-0-
specific, upper-VGPR-bank A-load` entry directly below. The four
diagnostic probes had narrowed the defect to an exact shape — now we name
the ROOT CAUSE and retire all six `Gfx1250Gpu.Matmul*` XFAIL entries.

**Root cause.** V_CMP → SGPR → V_CNDMASK round trip across cross-widening.

The matmul's prologue computes the per-lane LDS address for the
upper-VGPR-bank A load through the pattern

```
v_cmp_*_e64  sN, ...       ; wave-mask (wave-width i1 per lane),
                           ; stored to SGPR narrow-width (i32 on
                           ; wave32 source)
... SGPR may be written / read scalarly ...
v_cndmask_b32  vdst, src0, src1, sN   ; per-lane select keyed on sN
```

Under `WaveNativeProjection` (wave32 source → wave64 target), the V_CMP
writer produces a **wave-width** (i64) per-lane i1. The SGPR destination
is **source-width** (i32), so `ballotI1ToWidth` truncates the i64 ballot
to i32 — destroying the upper 32 bits which carried the compare result
for target lanes 32..63 (the lanes holding source wave 3's share of the
workgroup). The V_CNDMASK consumer then reads the SGPR back and routed
it through `extractLaneBitFromWaveMask`, which **replicates** the low 32
bits into both halves — so target lanes 32..63 see source wave 2's (or
0's) compare result instead of their own. For warp 3's prologue A-load,
this mis-routes the `v_cndmask` that selects between two LDS-offset
candidates, so warp 3 reads from warp 0's LDS region for the K-iter 0
portion of its A fragment. The observed substitution (rows 124..127 ←
A[0], A[2], A[4], A[6]) is the lane-by-lane consequence.

**Fix (commit da404faf84).** Add a per-BB
`DenseMap<int, WaveMaskEntry>` shadow in `RaiseContext` that caches the
per-lane i1 produced by the most recent V_CMP_*_e64 writer to each SGPR.
The V_CNDMASK consumer consults the shadow FIRST and reads the i1
directly — bypassing the lossy narrow-ballot round trip entirely — and
falls back to `extractLaneBitFromWaveMask` only when the shadow is
empty (no fresh V_CMP writer in this BB, scalar SGPR write invalidated
the cache, or we crossed a BB boundary).

Three invariants ensure soundness (see `sgpr-wave-mask-translation.md`
section 3.1 for the full treatment):
- **I1 (Additive).** Narrow store + extract reader both preserved; the
  shadow is consulted in addition, not instead. If a kernel's V_CNDMASK
  path was correct before the fix it stays correct after.
- **I2 (SSA-monotonic within a BB).** The SSA value in the cache is the
  exact `cmp` produced by the last V_CMP writer to `sN`, with no
  intervening write. Linear handler dispatch guarantees this.
- **I3 (Any interference defeats the cache).** Scalar SGPR writes
  invalidate via `onSgprWritten` (fired by `storeSGPR32 / storeSGPR64`),
  and the shadow is cleared at BB boundaries.

**What the four diagnostic probes each contributed.**

- `RowIdA` (`A[i,k] = (i+1)·0.001`): identified the SUBSTITUTION arithmetic
  (`rows 124..127 ← A[0], A[2], A[4], A[6]`) rather than a miss/zero.
- `RowOnly124`: ruled out 2×2-grid-specific boundary handling
  (single-tile reproduces the same pattern).
- `EvenRows`: proved WAVE-3 SPECIFICITY — rows 12..15, 28..31, 44..47,
  60..63, 76..79, 92..95, 108..111 ("sub-tile-row-1 rows 12..15" bands
  for warps 0/1/2) were ALL CORRECT, only rows 125/127 wrong. Rules out
  any general pass-2 / row-12..15 / target-wave-upper-half defect.
- `KStripedRow124`: proved K-ITER 0 SPECIFICITY (got ≈ 44.8 = 48 - 3.2,
  missing the 0.1 strip, which lives at k in [0,32)). Rules out main-
  loop K-iter bugs, pins the defect to the prologue's upper-bank load.

All four pass bit-exact under the fix and remain in the test suite as
positive regression guards.

**Wrong hypotheses enumerated and ruled out (from the H1..H4 hand-off).**

- **H1 — Upper-VGPR-bank `ds_load_tr16_b128` destination write under
  `s_set_vgpr_msb`.** The ds_load is lifted correctly: dest VGPR
  baseIdx picks up the MSB adjust via `parseReg`'s
  `currentVGPRAdjust[]`, and the write is gated by `emitUnderExec`
  which reads the per-lane mask. The defect was upstream of the
  ds_load: the v_cndmask that computed the ds_load's per-lane ADDRESS
  was the corrupted site, not the ds_load itself.
- **H2 — Raiser ttmp8 seed wave-uniform.** Confirmed per-lane divergent
  at IR emission (`%ttmp8_val = shl i32 %wave_id_in_wg, 25` where
  `%wave_id_in_wg = lshr i32 %workitem.id.x, 5`). Not the defect.
- **H3 — LLVM AMDGPU backend regalloc miscompile for the upper-VGPR-
  bank path.** Bug reproduces byte-identically on both projections and
  with identical gfx942 regalloc output patterns; if regalloc were
  coalescing v300.. onto a warp-0 live range, changing projection
  would not preserve the error pattern.
- **H4 — Per-wave MSB drop at the `v_cndmask_b32_e32` dst.** The MSB
  adjust IS applied correctly (verified by runtime tracing: the
  v_cndmask's ParsedReg for dst has `baseIdx = 256` under
  `s_set_vgpr_msb 64`). The defect was the v_cndmask's COND operand
  (the SGPR wave mask) being truncated, not its DST.

**Why the obvious hypotheses all missed.** The substitution pattern
(warp 3 ← warp 0 A data) naturally suggests a cross-wave address or data
leak via LDS / bpermute / register-file aliasing. H1..H4 all point at
the DATA PATH that manifests the symptom. The actual defect was one
level upstream: the CONTROL-MASK (v_cmp result routed through an SGPR
to a v_cndmask) that steered the address arithmetic. Corrupting the
mask silently re-routes the data path in a way that looks identical to
a data-path bug.

**Graduation.**
- All six `Gfx1250Gpu.Matmul*` XFAIL entries removed from
  `tests/xfail.cmake` (commit da404faf84 landed the fix;
  this entry and the follow-up retire the `WILL_FAIL TRUE`
  annotations).
- The four diagnostic probes remain as positive regression guards —
  each covers a distinct axis of the defect (substitution arithmetic /
  wave specificity / K-iter specificity / upper-vs-lower bank) and
  would fire independently on any regression that re-opens the shape.

---

## 2026-04-21 — Matmul128x128 residual: narrowed to wave-3 + K-iter-0 (prologue) + A-data side

**Context.** Follow-up to the "data-substitution bug, NOT a collect-stage
defect" entry below. Two further diagnostic probes added to isolate the
defect axis.

**What was done.** Added `MatmulDataPattern::EvenRows` (A[i,·] = 1 iff i
is even, B = 1) and `MatmulDataPattern::KStripedRow124` (A[124,k] striped
per K-iter: 0.1 / 0.2 / 0.4 / 0.8 for k in [0,32)/[32,64)/[64,96)/[96,128)),
wired each through `doTestMatmul` and registered as XFAIL.

**What the evidence shows (in order of what each probe rules out).**

1. **`EvenRows` — wave-3 specificity.** Errors appear ONLY at rows 125
   and 127 of the output. Rows 12..15, 28..31, 44..47, 60..63, 76..79,
   92..95, 108..111 — every OTHER "sub-tile-row-1 rows 12..15" band
   across warps 0/1/2 — are correct. Under the kernel's `4-warp × 32-row`
   tiling (`maxFlatWorkgroupSize=128` → 4 source Wave32s; 128 output
   rows / 4 warps = 32 rows/warp; 32 rows = 2 sub-tile rows × 16), each
   warp has a "sub-tile row 1 rows 12..15" band that a general
   pass-2 / row-12..15 defect would affect. Only WARP 3's band
   (= rows 124..127) does. Source wave 3 = target wave 1 lanes 32..63 =
   pass-2 of `runGroupPass` on target wave 1; source wave 1 = target
   wave 0 lanes 32..63 is ALSO "pass-2" and is CORRECT, so the defect
   is NOT pass-2 in general.

2. **`KStripedRow124` — K-iter 0 specificity.** With A[124,·] striped
   so each K-iter contributes a unique amount (3.2 / 6.4 / 12.8 / 25.6,
   summing to 48.0), the observed `C[124, ·] ≈ 44.8 = 48.0 - 3.2`
   identifies the missing contribution as K-iter 0 (k in [0,32)).
   This is the prologue 16-WMMA block in the disassembly (lines
   1564..1649 of `matmul_f16_large_gfx1250.hsaco`), NOT the main
   K-loop body. The prologue uses `s_set_vgpr_msb` to address the
   upper VGPR bank (v[300:331] for A sources via `/*v[300:307]*/`
   aliasing); the main loop uses the lower bank (v[128:159]). The
   bug therefore implicates the upper-bank-loaded A path, not the
   lower-bank path the main loop uses.

**Refined defect statement.** "Warp 3's prologue WMMA output for rows
12..15 of its sub-tile row 1 (= global rows 124..127) receives
A-fragment contribution from warp 0's A-rows 0, 2, 4, 6 — a
cross-warp, wave-3-specific, K-iter-0-specific data substitution on
the A-fragment side of the WMMA chain, on the UPPER-VGPR-BANK path
(`s_set_vgpr_msb` active)."

**What NOT to waste time on.**

- The substituted source rows (0, 2, 4, 6 in GLOBAL A-matrix, not
  0, 2, 4, 6 intra-sub-tile) definitively rule out a local lane-map
  defect in `wmma_lowering.cpp`. See RowIdA arithmetic: `3.968 =
  4.000 - 32 * A[0] = 32 * (A[124] - A[0])` for row 124;
  intra-sub-tile row 0 would be A[112] = 113/1000, which produces
  `15.616`, not the observed `12.032`.
- The bug is invariant under WaveNative vs ModuloReplication
  projection (handoff confirmed, re-verified here: `EvenRows`
  produces identical errors under both). So `WaveNativeProjection::
  emitInitialExec` and the `init_whole_wave` hardware-EXEC path are
  not implicated.

**What to investigate next (sharpened from the earlier entry).**

1. **The upper-VGPR-bank A load chain for source wave 3.** In the
   prologue, `ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v72 /*v328*/`
   (and its 7 siblings at offsets 64/128/192/4608/4672/4736/4800)
   writes A data into v[300:331] under `s_set_vgpr_msb 0x41` (dst +
   src0 MSB=1). The LDS base address is in v72 (= v328 under MSB
   adjust), computed per-lane from `s8 + v229 + v230` (`v_add3_u32`,
   line 1534). Trace what warp 3 lanes see as v328 vs what warp 0
   lanes see — the `A[0,·]` substitution implies warp 3 is reading
   warp 0's LDS region for this specific load.

2. **Wave-ID plumbing under cross-widening.** The matmul lifts
   `s_bfe_u32 ttmp8, 0x50019` to `(workitem.id.x >> 5) & 0x1F`. For
   a 128-thread workgroup on target Wave64 lanes 0..127, this gives
   per-lane values `0,0,...,0 (×32) | 1,1,...,1 (×32)` on target
   wave 0 and `2,2,...,2 (×32) | 3,3,...,3 (×32)` on target wave 1.
   The source kernel MAY assume wave_id is WAVE-UNIFORM (one value
   per hardware wave). Under modulo-replication, target wave 1 has
   TWO different wave_id values (2 and 3) on its two halves. If
   ANY downstream use of this per-lane value goes through a scalar
   operation (readfirstlane, scalar spill/reload, or an SGPR-
   constrained consumer the writelane/readlane rewrite's forward-
   use-chain classifier didn't catch), the wave-3 lanes would see
   wave_id=2 and read warp-0's LDS region via a derivation like
   `s73 = s2 & 3` → LDS offset. But substituted = warp 0 (not warp 2),
   so it's NOT wave-id-collapse-to-peer (which would give warp 2).
   Check if there's a `readfirstlane` or similar on the wave_id-
   derived SGPR chain that could collapse warp 3's value to WARP
   0's (e.g. an SGPR broadcast from lane 0 under a narrow EXEC).

3. **The 8 `ds_load_tr16_b128` instruction offsets.** They split
   {0, 64, 128, 192} + {4608, 4672, 4736, 4800} into two groups of
   4. The group of 4 with larger offsets is at +4608 = +0x1200 = +4K
   into LDS — crosses into a second 4K LDS region. Check whether
   wave 3's LDS stride for the prologue's first load causes it to
   wrap around into wave 0's region.

4. **The `compare_correctness` `makeSSetVgprMsbRecipe()` probe.**
   It is already wired to exercise the upper-VGPR-bank path with a
   non-WMMA kernel. Run it under the same
   `--enable-writelane-rewrite + --enable-wave-native` config as
   the matmul and check whether warp-3-specific data substitution
   appears there too. Same result would confirm the bug is in
   MSB-path lifting; a clean run would localise to the
   LDS-side-of-WMMA interaction.

**Files touched:**
`tests/gfx1250_gpu_test.cpp` (`EvenRows`, `KStripedRow124` patterns
+ `TEST_F` registrations), `tests/xfail.cmake` (four `WILL_FAIL`
entries + commentary), this doc.

---

## 2026-04-21 — Matmul128x128 residual: data-substitution bug, NOT a collect-stage defect

**Context.** Follow-up investigation of the `Gfx1250Gpu.Matmul128x128*`
residual documented in the previous entry. Handed-off with the primary
hypothesis that the bug lives in the WMMA→MFMA "collect" stage of
`wmma_lowering.cpp::runGroupPass`, specifically the `ds_bpermute`-based
gather that maps a 4-VGPR MFMA output back into the 8-VGPR Wave32 C
fragment.

**What was done.** Added two diagnostic input patterns to `doTestMatmul`
in `tests/gfx1250_gpu_test.cpp`:

- `MatmulDataPattern::RowIdA`: `A[i,k] = (i+1) * 0.001` for all k, `B = 1`.
  Reference `C[i,j] = 128 * (i+1) * 0.001`, constant across columns.
  Every output row has a unique expected value — any per-row mis-routing
  is immediately visible numerically.
- `MatmulDataPattern::RowOnly124`: `A[i,k] = (i == 124 ? 1 : 0)`,
  `B = 1`. Reference `C[124, j] = 128` for all j, every other row = 0.
  If any of row 124's K-iters is lost, C[124,j] ≠ 128.

Also added a per-row / per-column error histogram to the error-summary
path.

**What the evidence shows.**

1. **RowOnly124 is the smoking gun.** Output row 124 comes out as
   `96.0` across all 128 columns — EXACTLY 32 units short of the
   reference 128.0. 32 is the K-dimension of a single WMMA call
   (`v_wmma_f32_16x16x32_f16`). So ONE WMMA instance's contribution
   to row 124 is being SILENTLY REPLACED with another row's data
   (which happens to be 0 under this pattern, since every row
   except 124 has A=0).

2. **RowIdA pins the substitution pattern.** For random-free data
   where `A[i] ∝ i`, output rows 124..127 come out with the missing
   K-iter's contribution equal to `32 * A[2*(row - 124)]`:

   | output row | expected    | got         | missing ctr | source row |
   |------------|-------------|-------------|-------------|------------|
   |      124   | 16.0        | 12.032      | 3.968       | A[0]       |
   |      125   | 16.125      | 12.1898     | 3.935       | A[2]       |
   |      126   | 16.25       | 12.3475     | 3.903       | A[4]       |
   |      127   | 16.3906     | 12.5170     | 3.874       | A[6]       |

   Equivalently, output row `(124 + r)` receives contribution from
   row `2 * r` for `r ∈ {0, 1, 2, 3}` on one specific WMMA call.

3. **Only output rows 124..127 are affected.** Rows 12..15, 28..31,
   44..47, …, 108..111 — i.e., rows 12..15 of every OTHER 16-row
   sub-tile row — are correct. If the defect were a general COLLECT-
   stage bug affecting Wave32 lanes 16..31 GPRs 4..7 uniformly across
   every WMMA, we would expect errors in ALL these rows. We don't.

4. **Collect-stage math is correct.** Spent significant time
   verifying the `srcLane = 32*(w32Lane>=16) + 16*(GPR_w>=4) +
   (w32Lane & 15)` formula both in `wmma_lowering.cpp` and in the
   `--emit-ir` dump (`/tmp/mm.ll`). For pass 2 at W64 lane 48
   (w32Lane=16, the first failing lane), the formula produces
   `srcLane = 48` for `gw ∈ {4..7}`, which reads MFMA `LG 3` at
   the expected `mfmaDwords[0..3]` indices — exactly what the
   file-header lane-layout equations prescribe. The emitted IR
   matches the formula literally.

5. **IR structure is correct.** Pass 1 and pass 2 of `runGroupPass`
   produce independent `result0[]` / `result1[]` arrays, packed via
   `select i1 is_group1` into the final `<8 x float>` result. No
   SSA sharing, no cross-pass aliasing.

**What this means.** The defect is NOT the hypothesised collect-stage
lane-map bug. The collect math is correct AND uniformly applied across
every WMMA. The rejection criterion: if it were a collect bug, it would
affect EVERY sub-tile's rows 12..15, not just rows 124..127.

**What the bug IS.** Under-specified. The data is: ONE specific
WMMA-call-worth of contribution is dropped for rows 124..127, and the
wrong data substituted in follows a very specific `2*(row - 124)`
pattern in A-row indexing. The substitution crosses what would be
sub-tile boundaries (rows 0..6 of the A matrix, not rows 112..118 of
the failing sub-tile), which rules out intra-sub-tile mis-routing and
suggests the defect is EITHER:

- In a specific WMMA's A-fragment load (the `ds_load_tr16_b128`
  emulation, or the raising of its VGPR-MSB-adjusted destination), OR
- In a post-WMMA shuffle / reduce step (an LDS rearrangement
  the Triton kernel does between WMMA accumulation and the final
  store) that crosses warp boundaries, OR
- In the interaction between `s_set_vgpr_msb` state and some handler
  whose operand-index map doesn't consult `ctx.currentVGPRAdjust[]`.

**What NOT to waste time on (next investigation).** Both `wmma_lowering.cpp`
collect-stage math and `redistributeInput` / `redistributeAcc` math are
verified correct against the documented lane-layout equations. The IR
dump shows these formulas emitted literally. Do not re-derive them from
first principles a third time.

**What to investigate next.**

1. Grep the disassembly for the specific WMMA whose dst is the
   accumulator covering rows 124..127. Trace its A-fragment load
   chain back through `ds_load_tr16_b128` and the s_set_vgpr_msb
   state surrounding it. If the dst / src0 MSB adjustment on that
   specific path is dropped somewhere, that would explain the
   "wrong A row" substitution (reading from a lower-bank VGPR
   instead of the upper-bank one).
2. Write a minimal HIP reproducer that has 4 chained WMMAs across
   2 source waves with `s_set_vgpr_msb` in the source kernel and
   non-uniform A/B data. If it reproduces, the bug is
   `s_set_vgpr_msb`-related; if it doesn't, suspect the Triton-
   kernel-specific LDS shuffle path.
3. The `compare_correctness` tool (tools/compare_correctness/)
   already has a `makeSSetVgprMsbRecipe()` probe. Run it end-to-end
   under the same `--enable-writelane-rewrite` + `--enable-wave-native`
   configuration as the matmul gtest. A mismatch there isolates the
   MSB path; a match rules it out.

**Value landed anyway.** The two diagnostic input patterns (RowIdA,
RowOnly124) and the per-row/per-column error histogram make the defect
shape trivially visible without any manual scripting. The xfail entries
for them are gated with the same `WILL_FAIL TRUE` pattern as the parent
matmul tests, so CTest still validates that the defect is reproducible
(a silent fix would flip them unexpected-pass and force a review).

**Files touched:**
`tests/gfx1250_gpu_test.cpp`, `tests/xfail.cmake`, this doc.

---

## 2026-04-21 — WaveNative projection alone does not fix the Matmul128x128 residual

**Context.** After the writelane/readlane symmetry fix (same day, below),
the Matmul128x128 random-input gtests still failed with ~3% numerical
errors localised to output rows 124–127 / 252–255. The
`wmma_lowering.cpp` file header and the `WaveNativeProjection`
docstrings both point at partial-EXEC during WMMA → MFMA as the
culprit, with `@llvm.amdgcn.init_whole_wave` as the documented fix.

**What was done.** Finished wiring `WaveNativeProjection` end-to-end:
added an `enableWaveNative` opt-in flag through `raiseToIR` /
`runPipeline*` / `raise_cli`, plumbed the projection-driven EXEC
alloca width + initial value through `reg_file.cpp`, widened the
source-width SGPR ⇄ EXEC-width bridging in
`raise_context.cpp::readOpExecWidth`, flipped the V_CMP → SGPR write
in `handle_valu_vcmp.cpp` to pick `sourceWaveMaskTy()` rather than
`execTy`, removed the now-obsolete `strict.wwm` wrappers in
`wmma_lowering.cpp` (they are subsumed by the kernel-entry
`init_whole_wave`), updated the 5 lit fixtures that expected the
wave-native IR shape, and flipped `doTestMatmul` to opt in. The
compiled HSACO for the matmul kernel grew from 83648 B to 133760 B
and now starts with `s_or_saveexec_b64 s[0:1], -1` — confirming the
projection reaches codegen exactly as intended.

**What the evidence showed.** The matmul random-input errors did not
change: same 490 count on the single-tile case, same 1985 on the 2×2
grid, same row pattern (124–127 / 252–255). Under both ModRep and
WaveNative the errors are byte-identical. That rules out partial-EXEC
at the MFMA collective as the cause — WaveNative provably fixes
partial EXEC, and the error pattern is insensitive to that fix.

**What we still don't know.** The residual must live deeper in the
WMMA → MFMA redistribution math — most likely in the "collect" stage
in `wmma_lowering.cpp` for the second half of source wave 3's
sub-tile, or in some bookkeeping the matmul exercises that the
smaller `wmma_*` lit fixtures don't. Investigation handed off via
`tests/xfail.cmake`'s rewritten commentary and the list of MFMA
lane-layout equations at the top of `wmma_lowering.cpp`.

**Value landed anyway.** WaveNative is a principled piece of
infrastructure: it unblocks the 5 previously-failing lit fixtures
(`v_cmpx_ballot`, the four `wmma_*`), provides the correct EXEC
shape for any future kernel whose WMMA path actually does suffer
from partial EXEC, and the `--enable-wave-native` flag structure
lets callers opt in per surface. The `strict.wwm` per-MFMA-output
strategy it replaces was documented to crash `SIPreAllocateWWMRegs`
on 128×128 matmul tiles, so the refactor is net-positive even though
it does not solve the specific matmul residual.

---

## 2026-04-21 — Writelane/readlane rewrite must be symmetric under cross-widening

**Symptom.** `Gfx1250Gpu.Matmul128x128*` faulted at dispatch with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (HIP 700) after
`--enable-writelane-rewrite` graduation. Raise succeeded
(`4261/4261`, HSACO 83648 B); the kernel launched and faulted mid-dispatch.

**Wrong hypothesis (the obvious one).** "The rewrite shifts the
`v_readfirstlane_b32` scalarisation point a few instructions later."
Falsified by the compiled HSACO: only 3 `v_readfirstlane_b32`
instructions exist, none consume a `ds_bpermute` result, and all three
are uniform-VGPR scalarisations pre-existing in every gfx942 matmul
build.

**Actual cause.** The rewrite was **asymmetric**:

- Uniform writelanes (e.g. `writelane(ka_lo, 0, undef)`) were preserved
  as the native `v_writelane_b32` — writes hardware lane 0 **only**.
- Divergent readlanes on the same VGPR were rewritten to
  `ds_bpermute(((lane_id & ~(W_s-1)) | lane_idx) << 2, src)` — reads
  **both** lane 0 **and** lane 32.

Target lanes 32..63 read `undef` from hardware lane 32, zext'd it into
the high half of an `i64` global-memory pointer, and the `global_load`
faulted on the bogus address. Uniform-diag test (`A=B=1.0`) passed
because its reference is position-invariant (`K = 128` everywhere) —
which was the diagnostic that isolated the defect to "addressing /
cross-replica data-placement," not to WMMA / accumulator state.

**Fix.** `rewrite_cross_lane_divergent.cpp` now rewrites **every**
writelane and **every** readlane site unconditionally under
cross-widening, regardless of operand divergence. The
`select`/`ds_bpermute` shapes are semantically equivalent to the
source opcodes for any operand-divergence combination, so unconditional
rewriting is correctness-preserving and makes the writelane/readlane
pair trivially self-consistent on any shared VGPR.

**Guarded by a forward use-chain classifier.** Unconditional
rewriting is sound only when the rewritten result never reaches an
SGPR-constrained consumer. If a readlane result flows into
`llvm.amdgcn.s.buffer.load`'s rsrc, an `llvm.amdgcn.s.sendmsg`
message, a `llvm.amdgcn.readfirstlane` sink, a load from
addrspace(4), inline asm with `"s"` constraint, or any unknown sink,
the backend inserts `v_readfirstlane` on the `ds_bpermute` output
and recreates the original source-wave collapse. The rewrite pass's
`classifyForwardUseChain` walks every writelane / readlane result's
transitive uses pre-rewrite; unknown users over-approximate to
SGPR-forced. If any site's chain is SGPR-forced, the pass refuses
the whole function via `CrossLaneDivergentRewriteReport::
sgprForcedDetail`, which the raiser surfaces as
`RaiseFailure::crossWaveRewriteOracleDisagreement`. The refusal is
all-or-nothing because a mix of rewritten and preserved sites on a
shared VGPR recreates the asymmetric-rewrite fault. Pinned by
`lit_tests/writelane_sgpr_forced_use` (fixture chains writelane ->
`v_readfirstlane_b32`; asserts refusal under the flag + clean raise
under flag-off).

**Why the old rule was wrong in principle.** "Preserve uniform sites to
keep codegen quality" sounds local-reasoning-safe but pairs a
hardware-single-lane write with a software-two-lane read across the
source-wave boundary. The rule is only sound if you can prove that
**no** rewritten readlane on the same VGPR ever reads lane `N + W_s`
— a non-local invariant the pass had no mechanism to enforce. The
symmetric rule replaces a non-local invariant with a local one
("every site uses per-source-wave semantics").

**Residual after fix.** `Matmul128x128_1tile_UniformDiag` now passes
bit-exact (0 errors). The two random-input `Matmul128x128*` variants
still fail with ~3% numerical errors, localised to source wave 3's
**last 4 output rows** (rows 124–127 single-tile; 252–255 in 2×2-grid).
Not caused by the writelane/readlane rewrite — belongs to the
WMMA→MFMA "collect" stage under `ModuloReplicationProjection`.
Re-added to `xfail.cmake` with that precise reason.

**Files touched:**
`rewrite_cross_lane_divergent.{hpp,cpp}`,
`lit_tests/writelane_uniform_noop/*`,
`wave-size-translation.md` §5.6.3,
`tests/xfail.cmake`,
`tests/gfx1250_gpu_test.cpp`,
`raise_cli.cpp` (added `--write-hsaco` for disassembly triage).
