# Learnings

Short, dated entries capturing what broke, why it broke, and the fix.
Written for the reader in a future where everything works — so they can
reconstruct the trap that was there, not the solution that's obvious now.

Append-only. Newest on top.

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
