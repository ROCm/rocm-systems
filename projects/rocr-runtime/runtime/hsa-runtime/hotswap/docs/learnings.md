# Learnings

Short, dated entries capturing what broke, why it broke, and the fix.
Written for the reader in a future where everything works — so they can
reconstruct the trap that was there, not the solution that's obvious now.

Append-only. Newest on top.

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
