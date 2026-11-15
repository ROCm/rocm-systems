# rj_isa_diff P0: findings and measured results

Base: `ROCm/rocm-systems` `origin/develop` @ `90ddc8aae5`, branch
`users/lialan/rj/incremental-isa-diff`. Built with g++-13, Release, on Ubuntu 22.04
(32 cores). Every number here comes from that build.

## Summary by evidence level

| label | what |
|---|---|
| **implemented** | Code-object and raw-word ingestion. Decode and CFG via `BasicBlock::build_reachable`. Precise SALU effect model. LocSet liveness (including SCC and VCC/EXEC halves). PC taint. LCS alignment with branch-target validation. Change classification. Conservative single-block region planner (merge, context, stop rules). Contracts with sources. Isolated old/new runners on `ComputeUnitCore::execute_instruction`. Mandatory boundary/NaN/corner-case sweep, run on every test (integer limits, f32/f16/bf16/f64 special encodings, EXEC lane patterns, operand-role values for shift amounts, S_BFE fields, sign-extension inputs and literal neighbours, carry/overflow/borrow pairs, constraint edges). Deterministic random policy drawing from the same catalogs, with a constraint checker. Observable/frame/effect-model comparison. Separate evidence-level results. Counterexample bundles, replay, synthetic injection. CLI `analyze/test/replay/inject`. Prepared-region key. Per-phase cost counters. |
| **tested** | 30 gtest validator cases (`rj_isa_diff_tests`, ctest prefix `isa_diff.`) covering every row of handoff section 13, plus effect, liveness, alignment, JSON and replay-tamper tests. They pass in Release and under ASan+UBSan+LSan with zero reports. The CLI end-to-end flow was run on clang/lld-built gfx950 code objects and on raw word streams. |
| **not run** | No real LLVM old/new pair produced by an optimisation switch: this is deferred to the separate LLVM plugin task, by user decision. No result-cache store: keys are computed and recorded, but nothing reuses results across runs. No kernel/context reproducer: reachability is not established in P0. No SMT. |
| **unsupported** | Regions containing VALU, memory, MFMA, waits, barriers, branches, hwreg access, indirect SGPR access, TTMP/FLAT_SCRATCH/XNACK operands, or PC-derived inputs. Multi-block regions. Code objects with several `.text` sections or with zero-size kernel symbols. |
| **inconclusive** | Regions whose two continuations read different locations (`INCONCLUSIVE_MAPPING`, e.g. register renaming across blocks). Mismatches under an `UNRESOLVED` contract. Zero-valid-case runs. |
| **synthetic mismatch** | Injected wrong opcode, immediate and operand: all detected, all replayed (`REPRODUCED`), all labelled `synthetic`. Sample: `samples/synthetic/`. |
| **real finding** | None. No LLVM pair was tested, so no LLVM bug is claimed. |
| **proven** | Nothing. `proof_status` is `NOT_RUN` everywhere. |

## Validator results (handoff section 13)

| row | test | observed |
|---|---|---|
| identical region, repeated | `IdenticalRegionIsStableAndLeakFree` | NO_MISMATCH_FOUND, 1500/1500 valid, repeated run identical |
| dead temporary changed | `DeadTemporaryDifferenceIsNotReported` | `s10` canary, not observable; NO_MISMATCH_FOUND |
| live SCC differs / dead SCC | `LiveSccDifferenceIsDetectedAndDeadSccIsNot` | `s_add_u32`→`s_add_i32` with `s_cselect` after: LOCAL_MISMATCH on `scc`. With `s_cmp` first: NO_MISMATCH_FOUND (2000 cases) |
| live-through clobber | `ClobberedLiveThroughRegisterIsDetected` | inserted `s_mov_b32 s20, 0`: LOCAL_MISMATCH on `s20` |
| 64-bit / half / overlap | `HalfRegistersPairsAndOverlapCompareCorrectly` | `vcc_lo`→`vcc` write: mismatch on `vcc_hi` only when `vcc_hi` is live; in-place `s[4:5]` shifts and `s6 = s6+s7` overlap: no mismatch |
| reset / interleave / same seed | `SameSeedIsDeterministicAcrossRunsAndRunners` | identical first mismatch and final state across runs; 200 interleaved reuses equal fresh runners |
| constraint-dependent rewrite | `ConstraintDependentRewriteSeparatesUnrestrictedFromConstrained` | unrestricted: LOCAL_MISMATCH. With `--context 1` (common `s_and` executed): NO_MISMATCH_FOUND, ESTABLISHED. With a hand constraint `s4 <= 0xff`: NO_MISMATCH_FOUND, ASSUMED |
| contradictory constraints | `ContradictoryConstraintsNeverPass` | 0 accepted, INCONCLUSIVE |
| dependent hunks with common code | `DependentHunksMergeInsteadOfPassingIndependently` | merged into one 3-instruction region: NO_MISMATCH_FOUND. With `--no-merge`: r0 LOCAL_MISMATCH (local), r1 ASSUMED, depends on r0 |
| RA rename without a mapping | `RenamingWithoutValueMappingIsInconclusive` | INCONCLUSIVE_MAPPING regions; overall INCONCLUSIVE, not a pass |
| unknown successor / descriptor / uncovered | `UnknownSuccessorDescriptorAndUncoveredChangesKeepScopePartial` | `s_setpc`: contract ASSUMED, coverage PARTIAL. `compute_pgm_rsrc1`: UNRESOLVED, PARTIAL. Entry-offset-only: layout, no effect. Unreachable bytes changed: INCONCLUSIVE, NOT_ESTABLISHED |
| EXEC=0, alias consistency | `ExecZeroIsExecutedNotFiltered` | samples with EXEC=0 are generated and executed (valid = accepted). EXECZ/VCCZ are derived on read, so they cannot be inconsistent |
| waitcnt / memory / PC-as-data / hwreg | `WaitcntMemoryPcAndHwregRegionsAreUnsupported` | UNSUPPORTED with typed reasons |
| decode failure / model-only / unsupported | `DecodeFailureAndModelOnlyAreClassifiedNotNops` | DECODE_FAILURE (0 executed). MISSING_EXECUTION_BACKEND at the failing index. Region UNSUPPORTED, 0 valid |
| liveness change invalidates the cache key | `LivenessChangeInvalidatesThePreparedKey` | identical words and PCs; different consumer gives a different `prepared_key` and a different verdict |
| injected opcode/immediate/operand | `InjectedErrorsAreDetectedReplayableAndSynthetic` | all three detected, replay REPRODUCED, `synthetic: true` |

Additional checks:
- The effect model was measured against RocJITsu's decoder. `InstDefUse` reports no SCC/VCC/M0 effects on gfx950, so the tool derives effects from operands.
- `ReplayRejectsAnEditedBundle`: editing one region word breaks the prepared key and replay refuses.
- Every case also checks that execution wrote nothing outside the declared write set; that would be a `HARNESS_ERROR`. It never fired.

## Mandatory boundary sweep

Every `test` runs the sweep in full before any random case (`--cases` counts random
cases only). It was added after review, because random sampling alone hits a specific
32-bit corner only by chance.

| test | observed |
|---|---|
| `EveryEntryRegisterSeesEveryCatalogValue` | every entry register takes every 32-bit catalog value, including f32/f16/bf16 NaN, Inf, -0 and denormals; 64-bit operands take f64 NaN/Inf and EXEC lane patterns; when SCC is an input, corner values run with SCC=0 and SCC=1; the sweep is deterministic |
| `RoleValuesAreAimedAtTheSensitiveOperands` | the shift amount gets 32/33/63/64/65 and high garbage; the S_BFE control gets offset/width edges; the sign-extension input gets 0x7f/0x80/garbage-above values; the register compared with literal 0x12345 gets c, c+-1, -c, ~c; the add sources get carry, overflow, borrow and NaN pairs; the 64-bit compare gets equal-high/differ-low |
| `OffByOneAgainstALiteralIsAlwaysCaught` | `s_cmp_gt_u32 x, 0x12345` -> `s_cmp_ge_u32 x, 0x12345` (wrong only at x == 0x12345) is caught in the boundary phase with `--cases 0` under seeds 1, 2, 3 and 99; the correct rewrite (`>= 0x12346`) passes |
| `ConstraintEdgesRunAndOutOfContractValuesAreRejectedNotChanged` | with `s4 <= 0xff`, the edges 0xff and 0xfe run, and 0x100 and larger catalog values are rejected and counted (`boundary constraint: above umax`), never clamped |
| `TruncatedSweepIsReportedAsPartialCoverage` | `--max-boundary-cases 10`: `truncated`, coverage PARTIAL, reason recorded |

No rocjitsu undefined-behaviour diagnostic fired for any sweep value in the suite.

## End-to-end runs

The code-object pair is `samples/codeobject/co_{old,new}.s`, built with clang/ld.lld for
gfx950. Both kernels contain `s_load_dwordx2` and `s_waitcnt` outside the region.
```
analyze: kernel k, old 8 insts / new 9 insts; 1 hunk; r0 READY old 1 / new 2, ESTABLISHED
         layout shift noted as non-semantic (no PC reads)
test --cases 5000: NO_MISMATCH_FOUND, 5000/5000 valid, executed old 5000 / new 10000 insts
```
With only `.amdhsa_next_free_sgpr` changed in the new descriptor:
```
unresolved [metadata] kernel descriptor field differs: compute_pgm_rsrc1
r0 NO_MISMATCH_FOUND, contract UNRESOLVED -> overall INCONCLUSIVE, coverage PARTIAL (exit 2)
```
Synthetic error injection (`samples/synthetic/`):
```
inject --inst 1 --xor 0x01000000: new inst 1 is now 's_lshr_b64 s[8:9], s[8:9], 1'
test: LOCAL_MISMATCH at case 0, s8 old 0x00000084 new 0x00000021 (synthetic)
replay: REPRODUCED: kind=observable locations=s8,s9
```

## Cost

`bench/bench.py --cases 20000` on 2048-instruction straight-line SALU kernels; `bench.json`
holds the raw numbers. The sweep does not change the per-case cost; it adds cases.
Measured with the sweep (boundary cases run in addition to the 20000 random ones):

| variant | boundary cases | per case (µs) | cases/s (wall) |
|---|---|---|---|
| r1 | 842 | 2.44 | 338,948 |
| r4 | 3,131 | 2.59 | 325,673 |
| r16 | 9,879 | 3.09 | 287,350 |
| r64 | 16,270 | 5.88 | 159,563 |
| r256 | 38,599 | 15.77 | 62,079 |
| whole-kernel control | 94,875 | 128.23 | 7,772 |

The sweep grows with the number of registers the region reads (about 150-400 cases per
register, plus pair cases), not with kernel size.

The detailed per-phase breakdown below was measured before the sweep existed; the
per-case figures still apply. Columns:
- **analyze**: one-time decode, CFG, liveness, alignment and planning, measured by wall
  clock around the process.
- **per case**: propose, state load, old execute, new execute and compare, measured by
  the tool's own counters. All accepted cases are included and none are excluded.
- **insts/case**: counted executed instructions (old + new), which shows that only the
  region ran.

| variant | region old/new | insts/case | analyze (ms) | runner setup (ms) | per case (µs) | old exec (µs) | new exec (µs) | load (µs) | propose (µs) | compare (µs) | cases/s (wall) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 1/1 | 2 | 8.2 | 0.55 | 2.02 | 0.065 | 0.051 | 0.58 | 0.93 | 0.40 | 404,607 |
| r4 | 4/4 | 8 | 7.7 | 0.52 | 2.24 | 0.155 | 0.129 | 0.60 | 0.96 | 0.41 | 373,119 |
| r16 | 16/16 | 32 | 7.7 | 0.50 | 2.86 | 0.488 | 0.449 | 0.56 | 0.98 | 0.39 | 304,312 |
| r64 | 64/64 | 128 | 7.9 | 0.48 | 5.45 | 1.823 | 1.721 | 0.58 | 0.93 | 0.40 | 169,559 |
| r256 | 256/256 | 512 | 9.2 | 0.26 | 16.03 | 7.155 | 6.881 | 0.57 | 1.02 | 0.40 | 60,437 |
| whole-kernel control (r16 + all prefix) | 1984/1984 | 3968 | 15.0 | 0.49 | 126.82 | 62.742 | 61.712 | 0.63 | 1.20 | 0.53 | 7,818 |

Reading:
- Execution cost is about 0.03 µs per SALU instruction per side.
- For small regions the fixed per-case cost dominates: state load of every location
  (~0.58 µs), input proposal (~0.95 µs, mostly logging boundary events), and comparison
  (~0.40 µs).
- At R=16 a case costs 2.86 µs against 126.8 µs for the whole-kernel control on the
  same 2048-instruction kernel, about 44× less.
- This ratio is not a general speed-up claim. It grows with kernel size and shrinks
  with region size; at R=256 it is about 8×.
- Analysis (~8 ms here) is paid once per pair. Runner creation (~0.5 ms) is paid once per region.
- The proposal cost was halved by replacing a per-case `std::mt19937_64` seed with a
  SplitMix64 stream. The remaining proposal cost is mostly string building for the
  event log, which could be made lazy if throughput matters.

## Unresolved and limitations

- **Mapping**: identity at common cutpoints only. There is no renaming inference. Asymmetric
  continuation liveness is reported `INCONCLUSIVE_MAPPING`.
- **Liveness of SCC/VCC/EXEC/M0** comes from this tool's own analysis, because RocJITsu's
  `LivenessAnalysis` tracks ordinary registers only. Outside P0, instructions are assumed to
  read EXEC/VCC/M0. The exceptions are a short list (`s_endpgm`, `s_waitcnt`, `s_barrier`,
  direct branches, `s_sleep`, `s_setprio`) whose condition registers are classified
  operands; the validator checks this for `s_cbranch_scc0/execz/vccnz`.
- **VCC aliasing**: RocJITsu's CDNA4 model keeps VCC wave-local, and that is the trusted
  semantics (see `implementation_map.md`).
- **Reachability and kernel context**: every mismatch is `LOCAL_ONLY`. The kernel/context
  reproducer is pending.
- **Metadata**: the AMDHSA note is compared by hash, so any note change makes scope partial.
  Descriptor fields are compared individually, and the entry offset alone is layout only.
- **Pending effects**: P0 regions contain no memory or wait instructions. The isolated state
  contract assumes nothing is pending at the cutpoint and records that assumption. It does
  not prove that the real kernel satisfies it.
- **Cache**: `prepared_key` binds the words, PCs, full contract and mode. `results.json`
  records the RocJITsu revision. No result store reuses them yet.
- **Build environment**: two develop targets do not build in this environment, unrelated to
  this work: `hsa_init_test` (old system HSA) and `interposer_dup_test` (gcc-13
  `-Wstringop-overflow` under `-Werror`). Every other target builds.
- **LLVM** (separate task): the `.s` frontend via LLVM MC as an `ArtifactLoader`, and real
  old/new pairs from a controlled LLVM switch with an end-to-end report.
