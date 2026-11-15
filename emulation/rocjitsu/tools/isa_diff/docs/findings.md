# rj_isa_diff: findings and final report (handoff section 15)

Base: `ROCm/rocm-systems` `origin/develop` @ `90ddc8aae5`, branch
`users/lialan/rj/incremental-isa-diff`. Built with g++-13, Release, on Ubuntu 22.04
(32 cores). LLVM: ROCm 7.2.3 `/opt/rocm/llvm/bin` (AMD LLVM 22.0.0git, `roc-7.2.3 26084
f58b06dce1f9`). Every number here comes from that build and toolchain.

## Summary by evidence level

| label | what |
|---|---|
| **implemented** | See the next section for the full list. |
| **tested** | 65 isa_diff tests pass in Release and under ASan+UBSan+LSan with zero reports. They are 56 validator cases (`rj_isa_diff_tests`) covering every row of handoff section 13, every feature below and VALU support; 8 LLVM plugin cases (`rj_isa_diff_llvm_tests`) on real ROCm-LLVM output; and the schema/CLI test. Three real LLVM old/new pairs were run end to end, and the 55 functions of LLVM PR 242731's changed tests were run with before/after `llc` (see "VALU support and LLVM PR 242731"). The full tree was built and its full ctest run. |
| **not run** | SMT/SemaAST proofs (`proof_status` is always `NOT_RUN`). Reachability witnesses: the entry-prefix diagnostic path runs a prefix with arbitrary entry values, but no dispatch-ABI model exists. Learned input policies. Program (instruction) reduction of counterexamples. Multi-block, branching and loop regions (P1, designed in `p1_p2_design.md`). |
| **unsupported** | Regions containing memory (SMEM/VMEM/LDS), MFMA, interpolation, indirect VGPR access, waits, barriers, branches, hwreg access, indirect SGPR access, TTMP/FLAT_SCRATCH/XNACK operands, PC-derived inputs, or relocated bytes. Multi-block regions. Code objects with several `.text` sections or zero-size kernel symbols. The real `-enable-misched=false` pairs, whose changes are reordered `s_load`s. |
| **inconclusive** | Regions whose continuations read different locations (`INCONCLUSIVE_MAPPING`). Local mismatches that changed code downstream may compensate (`compensable_downstream`): the v2/v4 kernels of PR 242731. Mismatches under an `UNRESOLVED` contract: the renamed SALU tail of the real `k` misched pair. Relocation-only differences. Zero-valid-case runs. |
| **synthetic mismatch** | Injected wrong opcode, immediate and operand: all detected, all replayed (`REPRODUCED`), all labelled `synthetic`. Sample: `samples/synthetic/`. |
| **real finding** | **None.** Three real LLVM pairs were tested: one passed under an `ESTABLISHED` contract with complete scope, one is unsupported, and one is inconclusive. In LLVM PR 242731, 6 of the 12 changed gfx950 kernels pass under `ESTABLISHED` contracts with complete scope and none mismatches as a verdict. No LLVM bug is claimed. |
| **proven** | **Nothing.** |

The implemented items:

- Ingestion of code objects, raw words, and `.s`/`.ll` sources through the LLVM plugin,
  with full provenance.
- Decode and CFG via `BasicBlock::build_reachable`.
- A precise SALU effect model. Implicit effects of non-P0 instructions are translated.
- LocSet liveness, including SCC and the VCC/EXEC halves.
- PC taint.
- `.text` relocations.
- Per-kernel AMDHSA metadata fields and descriptor fields.
- LCS alignment with branch-target validation, and change classification.
- A conservative single-block region planner: merge, context, and stop rules on length,
  expansion and analysis time.
- Contracts with sources:
  - known-bits constraints derived from the common prefix;
  - a pending-effects analysis;
  - a composition check;
  - manual regions with entry/exit register maps, never more than `ASSUMED`.
- Isolated old/new runners on `ComputeUnitCore::execute_instruction`.
- The mandatory boundary/NaN/corner sweep, followed by the deterministic random policy.
- An unrestricted re-run next to constrained verdicts.
- Automatic context-expansion variants.
- Counterexample reduction, self-contained bundles, replay, and synthetic injection.
- Learning hooks and the per-case JSONL log.
- Analysis and result caches with a build fingerprint.
- Explicit entry-prefix diagnostic regions.
- JSON Schema for every output, with a CLI-driving validator.
- CLI commands `analyze`, `llvm-pair`, `test`, `replay`, `inject`.

## What was built, and the design trade-offs

Files, all under `emulation/rocjitsu/`:

| area | files |
|---|---|
| state, effects, liveness | `tools/isa_diff/{location,effects,kernel_model,machine_state}.{h,cpp}` |
| artifacts, alignment | `tools/isa_diff/{artifact,diff}.{h,cpp}` |
| regions and contracts | `tools/isa_diff/{region,contract,region_planner}.{h,cpp}` |
| execution and inputs | `tools/isa_diff/{region_runner,boundary,roles,input_policy}.{h,cpp}` |
| results, bundles, cache | `tools/isa_diff/{differential,mutation,case_io,result_cache,json}.{h,cpp}` |
| CLI and build | `tools/isa_diff/rj_isa_diff_main.cpp`, `tools/isa_diff/CMakeLists.txt`, one line in `tools/CMakeLists.txt` |
| LLVM plugin | `tools/isa_diff/llvm/{llvm_frontend.h,llvm_frontend.cpp,CMakeLists.txt}` |
| tests | `tests/isa_diff/{isa_diff_test.cpp,llvm_plugin_test.cpp,fixtures/}` |
| docs, schema, samples, bench | `tools/isa_diff/{README.md,docs/,bench/bench.py}` |

No RocJITsu library code was changed.

Design choices:

- **Own effect model and liveness.** RocJITsu's `InstDefUse` reports no SCC/VCC/M0
  effects on gfx950, and `LivenessAnalysis` tracks ordinary registers only. SCC-precise
  liveness is required (live and dead SCC must be told apart), so the tool classifies
  operands itself. Hidden effects are handled as follows:
  - Of non-P0 instructions: translated into tracked locations where possible.
  - Of P0 instructions: they reject the region.
- **Identity mapping at common cutpoints by default.** Nothing is inferred from outputs.
  Hand-written entry/exit maps exist for experiments and stay at most `ASSUMED`.
- **Conservative by construction.** Unknown effects make everything live. An untested
  upstream change keeps a contract `ASSUMED` or `UNRESOLVED`. Uncovered hunks, metadata
  or relocations keep the scope `PARTIAL`.
- **LLVM is an external process, run by a separate library.** The core never links LLVM.
  Executed bytes are always the linked object decoded by RocJITsu, and the provenance
  says so.
- **Cache invalidation is deliberately coarse.** The build fingerprint hashes the whole
  executable and its mapped objects, which handoff 11 explicitly allows.

## Commands

```sh
# build and test (remote build host)
CC=gcc-13 CXX=g++-13 cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release ...   # see implementation_map.md
cmake --build build --target rj_isa_diff rj_isa_diff_tests rj_isa_diff_llvm_tests
ctest --test-dir build -R isa_diff          # 58 tests: 49 core, 8 LLVM, 1 schema/CLI
tools/isa_diff/docs/samples/regen.sh build/tools/isa_diff/rj_isa_diff

# the real LLVM pair (samples/llvm_pair)
rj_isa_diff llvm-pair --input k3.ll --common-args -O3 --old-args "" \
    --new-args -combiner-disabled --out case/
rj_isa_diff test --case case/ --cases 2000

# the replayable synthetic error (samples/synthetic)
rj_isa_diff inject --case co/ --region r0 --side new --inst 1 --xor 0x01000000 \
    --label wrong-opcode --out syn/
rj_isa_diff test --case syn/ --cases 1000                     # exit 1
rj_isa_diff replay --counterexample syn/counterexamples/r0-syn-wrong-opcode-case1.json
```

## Test results

| suite | result |
|---|---|
| `rj_isa_diff_tests` (Release) | 49/49 pass |
| `rj_isa_diff_llvm_tests` (Release, ROCm 7.2.3 LLVM) | 8/8 pass |
| `isa_diff.schema_and_cli` | pass: 10 CLI scenarios plus the committed samples. Every `case.json`, `results.json`, counterexample bundle and case-log line validates against `docs/schema/rj_isa_diff.v1.schema.json` |
| the same three under ASan+UBSan+LSan (`build-asan`) | 49/49, 8/8, CLI validator pass; zero sanitizer reports |
| full tree build | every target builds except the two pre-existing, environment-related ones (`hsa_init_test` link, `interposer_dup_test` `-Wstringop-overflow`) |
| full ctest (14,850 tests) | 97% pass. The 486 failed or not-run tests are all outside this tool. Most are interposer tests (`rocjitsu: execvp failed`), DRM topology generation in HIP runtime tests, HSA/DBT guest runtime tests, a ccache probe (`cmake.build_path_cache`), and an f16 host-rounding helper (`ValuFpModeTest.Shard4`). None of them builds or links isa_diff code. They were not re-run on the base revision. |

## Validator results (handoff section 13, P0 rows)

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

## Tests added for the rest of the handoff

| test | observed |
|---|---|
| `DerivedConstraintsComeOnlyFromCommonCode` | `s9 = 7` from a common `s_mov` becomes known bits on `s9`. `s4`, written by a changed region, gets no constraint. A kernel input gets none |
| `ConstraintDependentRewriteSeparatesUnrestrictedFromConstrained` (extended) | constrained: NO_MISMATCH_FOUND. `unrestricted`: LOCAL_MISMATCH, `constraint_dependent: true` |
| `IsaDiffManual.RegisterMapParsingAndPermutation` | `s4=s8,s8=s9` completes to the 3-cycle `s9 -> s4`. Duplicates, specials and malformed entries are rejected. JSON round trip |
| `IsaDiffManual.ManualRegionStaysAssumedAndLeavesOtherHunksUnresolved` | region `m0` is ASSUMED with composition NOT_CHECKED; 2 uncovered hunks stay unresolved; identity mapping gives LOCAL_MISMATCH; scope is not complete |
| `IsaDiffManual.RegisterMapReconcilesRenameButStaysAssumed` | with `s4=s8`: new `s8` receives old `s4`'s value; NO_MISMATCH_FOUND, ASSUMED. A wrong map (`s4=s9`) is caught |
| `IsaDiffManual.ManualRegionRejectsBadSpecsAndBlockCrossing` | 6 malformed specs are errors. A map without a manual region is an error. A range across a block is INCONCLUSIVE_MAPPING |
| `IsaDiffArtifact.MetadataFieldsAreComparedPerKernel` | `.sgpr_count 18 -> 20` is one named field. A note differing only outside this kernel is informational. An unparsed note falls back to the hash, semantic |
| `IsaDiffArtifact.AnalysisTimeLimitLeavesContractsUnresolved` | with a 1 µs limit: no variants; every contract UNRESOLVED with composition NOT_CHECKED; `skipped_for_time` recorded; the timing split is present |
| `IsaDiffCache.ResultCacheHitsOnlyForTheSamePreparedRegionAndBudget` | second run: hit, same verdict and counts; the cached counterexample replays. Another seed misses. The same bytes with one more observable location miss |
| `IsaDiffCache.AnalysisCacheIsKeyedByArtifactsAndPlanOptions` | hit on identical input; miss with another plan option or other artifact bytes |
| `IsaDiffDiagnostics.EntryPrefixRegionsRunOnlyWhenRequestedAndStayOutOfTheVerdict` | `r0.diag-entry` runs from instruction 0 and is reported under `diagnostics`; the verdict covers `r0` only. A region behind a branch: UNSUPPORTED, "not in the kernel's entry block" |
| `IsaDiffKernelModel.ImplicitStoreBaseReadsKeepLivenessPrecise` | `global_store_dword` reports its SGPR base as implicit uses, now translated into reads. The renamed dead temporary `s8`/`s10` is not observable: NO_MISMATCH_FOUND, ESTABLISHED |
| `IsaDiffLlvm.AssemblesKernelSourceAndBareListings` | `.s` with a descriptor goes through llvm-mc and ld.lld to a code object. A bare listing falls back to raw `.text`. Invalid assembly is a reported failure |
| `IsaDiffLlvm.RealPairCombinerSwitchIsVerifiedEndToEnd` | see the real pairs below |
| `IsaDiffLlvm.RealPairTouchingSmemIsReportedNotPassed` | see the real pairs below |
| `IsaDiffLlvm.RelocationsAreKeptComparedAndNeverExecuted` | `@g` vs `@h`: identical `.text` bytes. Two GOTPCREL relocations differ, so the change is a semantic layout change and never NO_SEMANTIC_DIFF. A manual region over the relocated literal: UNSUPPORTED |
| `IsaDiffLlvm.KernelMetadataIsComparedFieldByField` | one inserted kernel argument is exactly one metadata item: `.args differs (old 21 args, new 22 args); first difference at .args[2]` |
| `IsaDiffLlvm.RealRegisterRenamingNeedsAnEntryMapAndIsNeverEstablished` | see the real pairs below |
| `IsaDiffLlvm.SideArgumentsThatChangeTheTargetAreRejected` | `-mcpu`, `-mattr` etc. in side arguments are rejected |

## Real LLVM pairs (handoff 13)

All pairs compile one LLVM IR file twice with ROCm 7.2.3 `llc -O3`; only the new side gets
the switch. Each switch was tried with this `llc`, not assumed from memory. Upstream `llc`
(LLVM 24) gave no difference for the candidate kernels.

| pair | kernel old/new insts | hunks | result |
|---|---|---|---|
| uniform i16 `k3`, `-combiner-disabled` (`samples/llvm_pair/`) | 21 / 22 | 3, merged into one region of 7 / 8 insts | **NO_MISMATCH_FOUND**, contract ESTABLISHED (composition HOLDS, pending NONE), coverage COMPLETE_FOR_DECLARED_SCOPE. 3,169 valid pairs (1,169 boundary + 2,000 random); executed 7 / 8 instructions per case |
| `k2`, `-enable-misched=false` | 26 / 26 | 2 | **UNSUPPORTED**: both changes are reordered `s_load_dwordx4`. Coverage NOT_ESTABLISHED. The case was kept, as the handoff requires, instead of being hidden |
| `k`, `-enable-misched=false` | 31 / 31 | 5 | **INCONCLUSIVE** (detailed below) |

The `k`, `-enable-misched=false` pair in detail:

- Three regions are UNSUPPORTED: reordered `s_load`s.
- The SALU tail `r3` (6 / 6 insts) is READY, but its contract is UNRESOLVED:
  - composition FAILS, because the loaded i64 sits in `s[4:5]` on the old side and
    `s[8:9]` on the new side;
  - the descriptor `compute_pgm_rsrc1` and the metadata `.sgpr_count` (15 → 17) also
    differ.
- `r3` shows LOCAL_MISMATCH on `s4` under that UNRESOLVED relation, so the witness is
  LOCAL_ONLY and the overall result is INCONCLUSIVE with coverage PARTIAL. The tool does
  not call it a bug.
- The same tail as a manual region `old:12-18,new:12-18`:

| mapping | result |
|---|---|
| identity | LOCAL_MISMATCH on `s4` (5,588 valid pairs) |
| `--entry-map s4=s8,s5=s9` | **NO_MISMATCH_FOUND** (3,991 valid pairs) |
| `--map s4=s8,s5=s9` | `INCONCLUSIVE_MAPPING`, because the result is in `s4` on both sides |

  Even the reconciled run stays UNRESOLVED (metadata differs) and PARTIAL (uncovered
  hunks).

Region size against kernel size:

| pair | kernel insts | instructions executed per case |
|---|---|---|
| k3 | 21 / 22 | 7 / 8 |
| `samples/codeobject` | 8 / 9 | 1 / 2 |
| bench r16 | 2,048 | 16 / 16 |

## VALU support and LLVM PR 242731

LLVM PR 242731 ("[AMDGPU] Saturate i64 inputs before converting to half") changes the
gfx950 lowering of i64→f16 conversions. That code is VALU (`v_cvt_f32_i32`,
`v_ldexp_f32`, `v_cvt_f16_f32`, …), which the SALU-only P0 could only report as
unsupported. VALU support was built for this (README "VALU support"), and the experiment
was rerun.

**Setup.**

- Before: `llc` built at the PR's parent `7e87ece942`.
- After: `llc` built at the PR head `57ec09d1ec`.
- Both are upstream LLVM 24, Release with assertions, AMDGPU only, and identically
  configured.
- Every function of the four changed tests was compiled for `amdgpu9.50-amd-amdhsa`
  (gfx950).
- `slp-int-to-fp.ll` went through one `opt -passes=slp-vectorizer` run shared by both
  sides. Its `void` functions became kernels; the float-returning one was skipped.
- Each pair ran with the mandatory sweep plus 5,000 random cases.

| stage | kernels | result |
|---|---|---|
| unchanged on gfx950 | 42 | `NO_SEMANTIC_DIFF` |
| SALU-only tool (before VALU support) | 12 changed | none compared: 4 `INCONCLUSIVE` (the SALU part split from the VALU part, mismatched liveness), 8 `UNSUPPORTED` on VALU or memory opcodes |
| with VALU support | `s_sint_to_fp_i64_to_f16`, `s_uint_to_fp_i64_to_f16`, `s_sint_to_fp_32_sign_bits_to_f16`, `s_uint_to_fp_31_zero_bits_to_f16` | **NO_MISMATCH_FOUND**, `ESTABLISHED`, complete scope; 6,108–6,420 valid cases each |
| | `v_sint_to_fp_i64_to_f16`, `v_uint_to_fp_i64_to_f16` | **NO_MISMATCH_FOUND**, `ESTABLISHED`, complete scope; 5,603 and 5,302 valid cases, 64 lane inputs each |
| | `s_{s,u}int_to_fp_v2i64_to_v2f16`, `v_sint_to_fp_v4i64_to_v4f16` | `INCONCLUSIVE`, `LOCAL_ONLY`: a store-address or lane-setup rename whose consumer is changed code holding a `global_store` (unsupported); `compensable_downstream` |
| | `v_uint_to_fp_v4i64_to_v4f16`, `sitofp_i48_to_half`, `uitofp_i56_to_half` | `INCONCLUSIVE`: loads, `s_waitcnt` and register renames inside the changed code |

**What the passing kernels show.**

- For the scalar-input kernels, the base region (old 8–12 / new 3–6 instructions)
  mismatches on its intermediate f32. On a lane with input 2³², the old code yields
  `0x4f800000` (2³²) and the new code `0x4f000000` (2³¹, the saturated i32).
- The automatic expansion adds the next common instruction (`v_cvt_f16_f32`). That
  variant has no mismatch, so the result is `RESOLVED_BY_CONTEXT`.
- This is the PR's own argument: the f32 intermediate differs, but both values
  overflow to the same f16.
- The vector-input kernels pass directly: the region spans the whole conversion,
  14 / 9 and 10 / 6 instructions.

**Tool defects the experiment exposed, fixed before the final run.**

1. **Liveness made VCC observable before every global store.** Every instruction
   outside the executable subset was assumed to read EXEC/VCC/M0, so the new code's
   `v_cmp` into VCC showed up as a false mismatch. Implicit reads are now per class:
   - vector memory reads EXEC, plus M0 for its LDS forms;
   - SMEM reads none.
2. **Split regions were judged in isolation.** A local mismatch on a location that
   changed code downstream reads was counted as a verdict. It is now a witness only.
   The downstream code may compensate, and context expansion cannot reach past changed
   code. This is the `compensable_downstream` rule above.
3. **Register-count metadata was treated as semantic.** `.vgpr_count 4 → 3` alone made
   `v_uint_to_fp_i64_to_f16` UNRESOLVED. Register-count fields are now informational
   when:
   - the descriptors agree on every semantic field;
   - each kernel's VGPRs fit its allocation (`ACCUM_OFFSET`).

   A kernel naming a VGPR beyond its allocation is reported as semantic.

**VALU tests** (all in `rj_isa_diff_tests`):

| test | observed |
|---|---|
| `EquivalentVectorRewriteIsExecutedOnAllLanes` | `v_add`+`v_lshlrev 2` vs commuted add + two shifts by 1: ESTABLISHED via VGPR composition, `vgpr_count` 8, live-through `v4` observable, dead `v2` not; NO_MISMATCH_FOUND with `vector.lanes`/`vector.cross`/`vector.exec` phases; 2 / 3 instructions per case |
| `WrongShiftIsCaughtOnLanesAndReplays` | shift by 3: LOCAL_MISMATCH on `v3` with lane detail; bundle has `vgpr_locations` and per-lane differences, and replays |
| `InactiveLaneClobberIsDetected` | the new side writes `v3` under a forced full EXEC: mismatch only on lanes outside the original EXEC; with EXEC constrained to all lanes, NO_MISMATCH_FOUND |
| `DeadVectorTemporaryIsNotObservable` | `v5`/`v6` renamed temporaries: not observable, NO_MISMATCH_FOUND |
| `OutstandingLoadIntoAVectorInputLeavesTheContextUnresolved` | `global_load_dword v0` read before `s_waitcnt`: CONFLICT, UNRESOLVED |
| `ModeIsPerSideAndMattersForFloatConversion` | identical `v_cvt_f32_f16`; the new side flushes f16 denormals: LOCAL_MISMATCH on `v1` |
| `RegisterCountMetadataIsInformationalOnlyWithinTheAllocation` | `.vgpr_count` difference informational within `ACCUM_OFFSET`; naming `v4` with 4 allocated is semantic |
| `DependentHunksMergeInsteadOfPassingIndependently` (extended) | with `--no-merge`, r0's mismatch on `s4` is read by r1: `compensable`, overall INCONCLUSIVE, LOCAL_ONLY |

## Cost

**Analysis** (from `case.json` `analysis.timing`, both sides together):

| pair | compile (LLVM, 3 tool runs per side) | parse | decode + CFG | dataflow | align | plan |
|---|---|---|---|---|---|---|
| k3 llvm_pair | 51.0 ms | 0.24 ms | 0.19 ms | 0.005 ms | 0.04 ms | 0.47 ms |
| codeobject (`.s` via llvm-mc/ld.lld) | 17.7 ms | 0.08 ms | 0.13 ms | 0.004 ms | 0.01 ms | 0.25 ms |
| bench, 2,048-instruction kernels | n/a | 0.12 ms | 3.2 ms | 0.29 ms | — | — |

**Per case.** `bench/bench.py --cases 20000` on 2,048-instruction straight-line SALU
kernels (`bench.json`, final build). The per-case column is propose + state load + old +
new + compare, from the `--no-case-log` run. It includes every executed case; none are
excluded.

| variant | region | executed cases (boundary accepted + 20,000 random) | insts/case | runner setup (ms) | per case (µs) | old exec | new exec | load | propose | compare | cases/s wall, log on / off |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 1/1 | 20,263 | 2 | 0.45 | 2.93 | 0.08 | 0.06 | 0.65 | 1.75 | 0.40 | 84,026 / 117,534 |
| r4 | 4/4 | 21,151 | 8 | 0.47 | 3.03 | 0.16 | 0.13 | 0.63 | 1.72 | 0.39 | 81,046 / 112,575 |
| r16 | 16/16 | 24,767 | 32 | 0.40 | 3.85 | 0.50 | 0.45 | 0.63 | 1.87 | 0.40 | 68,230 / 91,239 |
| r64 | 64/64 | 25,376 | 128 | 0.22 | 6.36 | 1.77 | 1.67 | 0.65 | 1.87 | 0.40 | 45,373 / 55,165 |
| r256 | 256/256 | 37,282 | 512 | 0.17 | 16.35 | 6.85 | 6.54 | 0.64 | 1.91 | 0.42 | 20,758 / 22,253 |
| whole-kernel control (r16 + entire prefix) | 1984/1984 | 114,875 | 3,968 | 0.79 | 124.21 | 61.23 | 60.79 | 0.68 | 0.95 | 0.56 | 7,755 / 7,923 |

Reading:

- **Execution** costs about 0.027 µs per SALU instruction per side.
- **Small regions are dominated by fixed per-case cost**: input proposal (about 1.8 µs,
  mostly sweep-event strings), state load (0.65 µs) and comparison (0.4 µs).
- **Region against whole kernel.** At R=16 a case costs 3.85 µs, against 124 µs for the
  whole-kernel control on the same kernel: about 32× less. That ratio grows with kernel
  size and shrinks with region size (about 7.6× at R=256). It is not a general speed-up
  claim.
- **Why the boundary sweep is smaller than generated.** The bench kernels' common
  prefix gives 16-18 derived known-bits constraints per region. Out-of-contract boundary
  values are rejected and counted, never changed: for r256, 38,080 boundary cases, of
  which 17,282 were accepted.
- **Why wall cases/s is lower than the per-case numbers suggest.** Because those
  constraints are derived, `test` also runs every region unrestricted (handoff 3.3).
  The wall time includes that second run, while the cases/s columns count verdict cases
  only. The unrestricted runs found nothing: `constraint_dependent: false`.
- **The per-case log** costs about 1-4 µs per case in wall time (the log-on vs log-off
  columns).
- **The cache** adds about 0.2 s per process, for the build fingerprint, and only with
  `--cache-dir`. A hit then skips the region's whole test.

## Replayable synthetic error (`samples/synthetic/`)

Steps:

1. `inject --inst 1 --xor 0x01000000` on the code-object region turns new instruction 1
   into `s_lshr_b64`.
2. `test` reports LOCAL_MISMATCH at boundary case 1, with `s9` old `0x00000004` and new
   `0x00000001`. The result is labelled `synthetic` and the contract is ESTABLISHED.
3. Reduction took 6 runs and 2 accepted steps. The case is seed-independent, and the only
   non-zero input is `s5 = 1`.
4. `replay` re-executes the bundle from its recorded words: REPRODUCED.

## Unresolved context and mapping issues, and limitations

- **Mapping**:
  - Identity at common cutpoints only. No renaming is ever inferred.
  - Hand-written entry/exit maps are experiments that stay at most ASSUMED.
  - Asymmetric continuation liveness is `INCONCLUSIVE_MAPPING`.
  - The real misched pair is the open example: its tail is consistent only under a
    hand-written entry map, and its context stays UNRESOLVED.
- **Composition**:
  - The must-equality analysis runs over the old CFG.
  - It assumes untested upstream changes write only their decoded destinations, which
    keeps those contracts ASSUMED.
  - Any control-flow change reaching a region makes its contract UNRESOLVED.
- **Derived constraints** come from a forward must-analysis with a small transfer set:
  constants, masks, shifts, `s_bfe` and min/max. Anything else is unknown, so they are
  sound but incomplete.
- **Reduction** shrinks input values and seed dependence only. The program is not
  reduced.
- **Learning**: data and hooks only. No learned policy exists.
- **Liveness of SCC/VCC/EXEC/M0** is this tool's own analysis.
  - Outside P0, instructions never kill.
  - They are assumed to read EXEC/VCC/M0, except on a short list whose operands name
    everything they read.
  - Implicit uses are translated where possible; TTMP/FLAT_SCRATCH/PC stay unknown, so
    everything is live there.
- **VCC aliasing**: RocJITsu's CDNA4 model keeps VCC wave-local, and that is the trusted
  semantics (see `implementation_map.md`).
- **Reachability**: every mismatch is `LOCAL_ONLY`. The entry-prefix diagnostic needs a
  P0-only prefix; real kernels start with `s_load`, so it is UNSUPPORTED there.
- **Metadata**: compared field by field per kernel. Register-count fields such as
  `.sgpr_count` count as semantic, which is conservative.
- **Pending effects** follow the gfx9 counter rules (SMEM may return out of order). Only
  a counter field of 0 clears destinations.
- **Cache**: whole-build fingerprint. There is no finer dependency tracking.
- **Build environment**: `hsa_init_test` and `interposer_dup_test` do not build on this
  host, and the full-ctest failures listed above are environmental. All are unrelated to
  this tool.
- **Correction kept from the earlier report**: an earlier version said the contract
  recorded the "nothing pending" assumption before any analysis existed. That was wrong.
  It is fixed by the pending-effects analysis.

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

## Memory operations outstanding at the cutpoint (handoff 6.1)

`KernelModel::pending_before()` is a forward dataflow over the CFG. For each gfx9 counter
(vmcnt, lgkmcnt, expcnt) it tracks an upper bound of outstanding operations and the tracked
locations they will write:
- SMEM and LDS raise lgkmcnt, VMEM raises vmcnt, and anything else that may wait raises all three;
- `s_waitcnt` (fields decoded by rocjitsu's `decode_gfx9_waitcnt`) lowers a counter, and only
  a field of 0 clears its destinations;
- joins take the maximum count and the union of destinations.

At each region entry the contract records `pending_effects`:

| value | meaning | contract |
|---|---|---|
| `NONE` | nothing outstanding | unaffected |
| `DISJOINT` | outstanding, but no destination is read or written by the region | unaffected; recorded as a source |
| `CONFLICT` | an outstanding write targets a location the region reads or writes | UNRESOLVED; listed as an unresolved context |
| `UNKNOWN` | an outstanding operation has unknown effects | UNRESOLVED |

| test | observed |
|---|---|
| `OutstandingLoadIntoARegionInputLeavesTheContextUnresolved` | `s_load_dword s4` then the region reads `s4` before `s_waitcnt`: CONFLICT, UNRESOLVED. The commuted add has no local mismatch, but the overall result is INCONCLUSIVE with PARTIAL coverage, never a pass |
| `DisjointWaitedAndPartiallyWaitedLoads` | load into `s10` with the region touching `s4/s5`: DISJOINT, ESTABLISHED, NO_MISMATCH_FOUND. After `s_waitcnt lgkmcnt(0)`: NONE. Two SMEM loads with `lgkmcnt(1)`: still CONFLICT (out-of-order SMEM), and the counter bound is 1 with both destinations pending |

## Composition (handoff 5.2)

The planner now runs a forward must-equality analysis over the old CFG:
- **Kernel entry**: everything is equal at the entry, or nothing is if the kernel metadata
  differs.
- **Common code**: equality propagates through it. A write is equal only when all its inputs
  are, including the implicit EXEC/VCC/M0 reads of non-SALU instructions and memory/VGPR state
  for loads and VALU.
- **Branches**: a branch on an unequal condition makes nothing known on both successors.
- **Regions**: a READY region contributes its R_out. An unsupported region removes what it
  writes. A control-flow change removes everything.

Each region records `composition: HOLDS|FAILS`, and FAILS makes the contract UNRESOLVED.
This replaces the old coarse rule ("any non-READY upstream region makes the contract
UNRESOLVED") with a per-location check.

| test | observed |
|---|---|
| `UntestedUpstreamChangeMattersOnlyIfItReachesTheEntryRelation` | An unsupported upstream `s_getreg` change writes `s4`. A downstream region not reading `s4`: HOLDS, ASSUMED. One reading `s4`: FAILS, UNRESOLVED, naming `{s4}`. Register renaming across blocks: FAILS. Two dependent hunks tested separately: HOLDS through r0's R_out |

## Automatic context expansion (handoff 5.2)

`analyze` precomputes up to 12 larger variants of each READY region. They extend over
matched, common P0 instructions in the same block, backward and/or forward (1, 2, 4, ... up
to the maximum). After a base region shows a local mismatch, `test` tries them smallest
first. The base result keeps its local mismatch, and `expansion.resolution` records:
- `RESOLVED_BY_CONTEXT`: the named variant has no mismatch, and the overall verdict uses its
  result and contract;
- `PERSISTS_WITH_CONTEXT`;
- `NO_EXPANSION_AVAILABLE`.

| test | observed |
|---|---|
| `CompensatedLocalMismatchIsResolvedByContext` | `s2<<1` vs `s2<<2` masked by a common `s_and_b32 s4, s4, 1`: resolved by `r0.ctx-b0f1` (ESTABLISHED), overall NO_MISMATCH_FOUND, witness NONE |
| `RealDifferencePersistsAndPrerequisiteContextResolves` | the live-SCC change persists through all forward variants (overall LOCAL_MISMATCH). `s_lshr` vs `s_bfe`, equal only after the common `s_and 0xff`, is resolved by a one-instruction backward context |

## Counterexample reduction (handoff 10)

Each saved counterexample is shrunk before it is traced and written:
1. The seed-derived canaries are replaced with fixed distinct values (old 0, new all-ones).
   If the mismatch persists, the bundle is marked `seed_independent`.
2. Each shared entry value is tried at 0, at 1, at successive halvings, and with single bits
   cleared (highest first). A candidate is accepted only if it is simpler (fewer set bits),
   satisfies the contract's constraints, and reproduces the same mismatch kind.

The bundle keeps the original state under `reduction.original_initial_state`, and replay
runs the reduced state. Budget: `--reduction-budget` (default 4096 paired runs).
`--no-reduce` turns reduction off.

| test | observed |
|---|---|
| `CounterexamplesAreShrunkSeedIndependentAndStillReplay` | live SCC: every input except the two add sources becomes 0, the set-bit count does not grow, the case is seed-independent, and the bundle replays. Off-by-one: `s2=0x12345` stays and is the only non-zero input |
| `ReductionNeverLeavesTheContract` | with `s3` known-one bit 4, every reduced example keeps the bit and passes `ConstraintChecker` |

CLI, synthetic wrong opcode on the code-object sample: reduced in 6 runs to the single
non-zero input `s5=1`, seed-independent, REPRODUCED by replay.

## Learning hooks and per-case log (handoff 8)

- `InputPolicy::propose(features, contract, history, case)` now receives a bounded `History`:
  counts, categories, and the last 32 feedbacks.
- `observe(Feedback)` receives:
  - the phase and events;
  - the accepted/rejected status and rejection reason;
  - both exit statuses and a bounded category;
  - the differing locations;
  - the novelty bucket and whether it is new;
  - per-stage cost (propose, load, old, new, compare) and instruction counts.
- `ConstraintChecker::validate` is a separate class. A policy can still only propose inputs.
- **Novelty buckets** are bounded: the outcome category, plus, per written location, a coarse
  value class (zero/one/all-ones/sign/other) and whether old and new agree. Raw values never
  form a bucket.
- `test` writes `<case>/logs/<region>.jsonl` (turn off with `--no-case-log`). It has:
  - a header line: seed, policy, contract summary, per-instruction opcode/effect summary
    (mnemonic, reads, writes), roles, operand pairs and literal relations;
  - one line per case (boundary and random);
  - a summary line.

| test | observed |
|---|---|
| `PolicyGetsHistoryRichFeedbackAndEveryCaseIsLogged` | `observe()` sees every case of both phases. `propose()` sees the history grow from the sweep size and the window stays <= 32. Feedback carries exits, instruction counts, buckets and mismatching locations. Buckets: more than 1 and fewer than 64 for hundreds of cases. The log has a header, one line per case and a summary, all valid JSON |
