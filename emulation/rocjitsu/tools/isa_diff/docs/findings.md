# rj_isa_diff: findings and final report (handoff section 15)

Base: `ROCm/rocm-systems` `origin/develop` @ `90ddc8aae5`, branch
`users/lialan/rj/incremental-isa-diff`. Built with g++-13, Release, on Ubuntu 22.04
(32 cores). LLVM: ROCm 7.2.3 `/opt/rocm/llvm/bin` (AMD LLVM 22.0.0git, `roc-7.2.3 26084
f58b06dce1f9`). Every number here comes from that build and toolchain.

## Summary by evidence level

| label | what |
|---|---|
| **implemented** | See the next section for the full list. |
| **tested** | 85 isa_diff tests pass in Release. They are 67 validator cases (`rj_isa_diff_tests`) covering every row of handoff section 13, every feature below, VALU support, memory, callable functions, inline asm, NaN equivalence and poison kinds; 16 LLVM plugin cases (`rj_isa_diff_llvm_tests`) on real ROCm-LLVM output; the schema/CLI test; and the `rj_isa_diff_pr` end-to-end test. All pass under ASan+UBSan+LSan with zero reports. Three real LLVM old/new pairs were run end to end, and the 55 functions of LLVM PR 242731's changed tests were run with before/after `llc` (see "VALU support and LLVM PR 242731" and "Memory in regions: PR 242731 re-evaluated"). The full tree was built and its full ctest run. |
| **not run** | SMT/SemaAST proofs (`proof_status` is always `NOT_RUN`). Reachability witnesses: the entry-prefix diagnostic path runs a prefix with arbitrary entry values, but no dispatch-ABI model exists. Learned input policies. Program (instruction) reduction of counterexamples. Multi-block, branching and loop regions (P1, designed in `p1_p2_design.md`). |
| **unsupported** | Regions containing memory other than `global_load_*`/`global_store_*`/`s_load_dword*` (LDS, flat, buffer, scratch, atomics), MFMA, interpolation, indirect VGPR access, barriers, branches, hwreg access, indirect SGPR access, TTMP/FLAT_SCRATCH/XNACK operands, PC-derived inputs, or relocated bytes. Multi-block regions. Code objects with several `.text` sections or zero-size kernel symbols. |
| **inconclusive** | Regions whose continuations read different locations (`INCONCLUSIVE_MAPPING`). Local mismatches that changed code downstream may compensate (`compensable_downstream`). Mismatches under an `UNRESOLVED` contract. Relocation-only differences. Zero-valid-case runs. No real pair or PR 242731 kernel is inconclusive any more. |
| **synthetic mismatch** | Injected wrong opcode, immediate and operand: all detected, all replayed (`REPRODUCED`), all labelled `synthetic`. Sample: `samples/synthetic/`. |
| **real finding** | **None.** All three real LLVM pairs pass under `ESTABLISHED` contracts with complete scope. In LLVM PR 242731, all 12 changed gfx950 kernels pass under `ESTABLISHED` contracts with complete scope (6 of 12 before memory support). Through the one-command mode (`rj_isa_diff_pr`, CHECK lines), PR 237629's 84 changed functions and PR 242731's 6 pass, and 132 of 162 changed functions in the last 120 AMDGPU commits pass; none mismatches. No LLVM bug is claimed. |
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
  - a pending-effects analysis (in-order VMEM, out-of-order SMEM);
  - a composition check;
  - manual regions with entry/exit register maps, never more than `ASSUMED`.
- Isolated old/new runners on `ComputeUnitCore::execute_instruction`, and on
  `ComputeUnitCore::step()` for memory instructions against a shared, seeded memory
  image (stored bytes are observable).
- Register-allocation-only descriptor differences are informational when both kernels
  fit their allocation.
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
| unknown successor / descriptor / uncovered | `UnknownSuccessorDescriptorAndUncoveredChangesKeepScopePartial` | `s_setpc`: contract ASSUMED, coverage PARTIAL. `compute_pgm_rsrc1` granulated counts only, both kernels fit: informational, ESTABLISHED, NO_MISMATCH_FOUND. The same change on a kernel naming `s21`: semantic, INCONCLUSIVE. IEEE_MODE: UNRESOLVED, PARTIAL. Entry-offset-only: layout, no effect. Unreachable bytes changed: INCONCLUSIVE, NOT_ESTABLISHED |
| EXEC=0, alias consistency | `ExecZeroIsExecutedNotFiltered` | samples with EXEC=0 are generated and executed (valid = accepted). EXECZ/VCCZ are derived on read, so they cannot be inconsistent |
| waitcnt / memory / PC-as-data / hwreg | `PcAndHwregRegionsAreUnsupportedWhileMemoryRuns` | an extra `vmcnt` field: NO_MISMATCH_FOUND. `s_load_dword` from offset 0x0 vs 0x4: LOCAL_MISMATCH on `s4`, with a `memory_model` pointer constraint on `s[0:1]`. `s_getpc`/`s_getreg`: UNSUPPORTED with typed reasons |
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
| `IsaDiffKernelModel.ImplicitStoreBaseReadsKeepLivenessPrecise` | `global_store_dword` reports its SGPR base as implicit uses, now translated into reads; the store is region-executable and reads EXEC, never VCC. The renamed dead temporary `s8`/`s10` is not observable: NO_MISMATCH_FOUND, ESTABLISHED |
| `IsaDiffLlvm.AssemblesKernelSourceAndBareListings` | `.s` with a descriptor goes through llvm-mc and ld.lld to a code object. A bare listing falls back to raw `.text`. Invalid assembly is a reported failure |
| `IsaDiffLlvm.RealPairCombinerSwitchIsVerifiedEndToEnd` | see the real pairs below |
| `IsaDiffLlvm.RealPairReorderingScalarLoadsRunsTheLoads` | see the real pairs below |
| `IsaDiffLlvm.RelocationsAreKeptComparedAndNeverExecuted` | `@g` vs `@h`: identical `.text` bytes. Two GOTPCREL relocations differ, so the change is a semantic layout change and never NO_SEMANTIC_DIFF. A manual region over the relocated literal: UNSUPPORTED |
| `IsaDiffLlvm.KernelMetadataIsComparedFieldByField` | one inserted kernel argument is exactly one metadata item: `.args differs (old 21 args, new 22 args); first difference at .args[2]` |
| `IsaDiffLlvm.RealRegisterRenamingIsInternalToTheMergedRegion` | see the real pairs below |
| `IsaDiffLlvm.SideArgumentsThatChangeTheTargetAreRejected` | `-mcpu`, `-mattr` etc. in side arguments are rejected |

## Real LLVM pairs (handoff 13)

All pairs compile one LLVM IR file twice with ROCm 7.2.3 `llc -O3`; only the new side gets
the switch. Each switch was tried with this `llc`, not assumed from memory. Upstream `llc`
(LLVM 24) gave no difference for the candidate kernels.

| pair | kernel old/new insts | hunks | result |
|---|---|---|---|
| uniform i16 `k3`, `-combiner-disabled` (`samples/llvm_pair/`) | 21 / 22 | 3, merged into one region of 7 / 8 insts | **NO_MISMATCH_FOUND**, contract ESTABLISHED (composition HOLDS, pending NONE), coverage COMPLETE_FOR_DECLARED_SCOPE. 3,169 valid pairs (1,169 boundary + 2,000 random); executed 7 / 8 instructions per case |
| `k2`, `-enable-misched=false` | 26 / 26 | 2, one region of 2 / 2 insts | **NO_MISMATCH_FOUND**, ESTABLISHED, complete scope. The two reordered `s_load_dwordx4` run against the shared memory image: 2,112 valid pairs. The 526 rejected boundary values break the pointer constraint on the kernarg base `s[4:5]`. Before memory support: UNSUPPORTED |
| `k`, `-enable-misched=false` | 31 / 31 | 5, one region of 18 / 18 insts | **NO_MISMATCH_FOUND**, ESTABLISHED, complete scope; 2,120 valid pairs. Before memory support: INCONCLUSIVE (detailed below) |

The `k`, `-enable-misched=false` pair in detail:

- Before memory support:
  - three regions were UNSUPPORTED, because they held reordered `s_load`s;
  - the SALU tail `r3` (6 / 6 insts) had an UNRESOLVED contract. Composition failed,
    because the loaded i64 sits in `s[4:5]` on the old side and `s[8:9]` on the new side.
    `compute_pgm_rsrc1` and `.sgpr_count` (15 → 17) also differed;
  - `r3`'s LOCAL_MISMATCH on `s4` was therefore only a LOCAL_ONLY witness, and the
    overall result was INCONCLUSIVE.
- Now:
  - the loads run, so the five hunks merge into one 18 / 18 region from the first
    `s_load` to the tail;
  - the `s[4:5]`/`s[8:9]` rename is internal to the region;
  - the rsrc1 and `.sgpr_count` differences are register allocation only, and both
    kernels fit (informational).
- The same tail as a manual region `old:12-18,new:12-18` (contract ASSUMED, coverage
  PARTIAL because of the uncovered hunks):

| mapping | result |
|---|---|
| identity | LOCAL_MISMATCH on `s4` (5,588 valid pairs) |
| `--entry-map s4=s8,s5=s9` | **NO_MISMATCH_FOUND** (3,991 valid pairs) |
| `--map s4=s8,s5=s9` | `INCONCLUSIVE_MAPPING`, because the result is in `s4` on both sides |

Region size against kernel size:

| pair | kernel insts | instructions executed per case |
|---|---|---|
| k3 | 21 / 22 | 7 / 8 |
| k (misched) | 31 / 31 | 18 / 18 |
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

**The six that did not pass** (`s_{s,u}int_to_fp_v2i64_to_v2f16`, the two v4 kernels,
`sitofp_i48_to_half`, `uitofp_i56_to_half`) all had loads, waits or stores inside or
between their changes. They pass after memory support (next section).

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

## Memory in regions: PR 242731 re-evaluated

Three changes target the six PR 242731 kernels that did not pass (README "Memory" and
"Register allocation"):

1. **Memory instructions run inside regions.** `global_load_*`, `global_store_*`,
   `s_load_dword*` and `s_waitcnt` use the CU's issue path (`ComputeUnitCore::step()`).
   Memory is an input shared by both sides, and stored bytes are an output.
2. **Regions merge across memory and end at the stores.** Hunks separated by common loads
   and waits merge into one region, and the stores that consume the result belong to it.
   This removed every `INCONCLUSIVE_MAPPING` and `compensable_downstream` split in these
   kernels, so no separate fallback was needed.
3. **Register-allocation-only descriptor changes are informational** when each kernel's
   registers fit its allocation.

The same experiment was rerun (mandatory sweep plus 5,000 random cases per kernel):

| kernel | region old / new insts | result | valid cases |
|---|---|---|---|
| `s_sint_to_fp_i64_to_f16` | 12 / 6 | pass, `RESOLVED_BY_CONTEXT` (+1 after: `v_cvt_f16_f32`) | 6,414 |
| `s_uint_to_fp_i64_to_f16` | 8 / 3 | pass, `RESOLVED_BY_CONTEXT` (+1 after) | 6,108 |
| `s_uint_to_fp_31_zero_bits_to_f16` | 8 / 3 | pass, `RESOLVED_BY_CONTEXT` (+1 after) | 6,108 |
| `s_sint_to_fp_32_sign_bits_to_f16` | 12 / 5 | pass, `RESOLVED_BY_CONTEXT` (+1 before, +1 after) | 6,420 |
| `v_sint_to_fp_i64_to_f16` | 14 / 9 | pass | 5,603 |
| `v_uint_to_fp_i64_to_f16` | 10 / 6 | pass | 5,302 |
| `s_sint_to_fp_v2i64_to_v2f16` | 28 / 16 | **pass** (was INCONCLUSIVE) | 9,122 |
| `s_uint_to_fp_v2i64_to_v2f16` | 20 / 10 | **pass** (was INCONCLUSIVE) | 7,805 |
| `v_sint_to_fp_v4i64_to_v4f16` | 57 / 33 | **pass** (was INCONCLUSIVE) | 8,030 |
| `v_uint_to_fp_v4i64_to_v4f16` | 41 / 25 | **pass** (was INCONCLUSIVE) | 6,532 |
| `sitofp_i48_to_half` | 70 / 46 | **pass** (was INCONCLUSIVE) | 7,604 |
| `uitofp_i56_to_half` | 70 / 44 | **pass** (was INCONCLUSIVE), `RESOLVED_BY_CONTEXT` (+4 before) | 6,008 |

All 12 are `NO_MISMATCH_FOUND`, `ESTABLISHED`, with coverage
`COMPLETE_FOR_DECLARED_SCOPE`. The 42 unchanged functions stay `NO_SEMANTIC_DIFF`.

**What the context resolutions show.** Each base-region mismatch disappears once the code
that produces the inputs, or consumes the outputs, is included. None is a verdict.

- **Output masking** (the first three rows): the intermediate f32 differs (2³² against
  the saturated 2³¹), and the next `v_cvt_f16_f32` maps both to the same f16.
- **Input domain** (`s_sint_to_fp_32_sign_bits_to_f16`):
  - with arbitrary 64-bit inputs, even the f16 result differs on lane 32 (old inf
    `0x7c00`, new `0x7bf8`);
  - the variant that also runs the preceding `s_ashr_i64 s[2:3], s[2:3], 31` gives the
    input its known sign bits, and both sides agree;
  - the PR's lowering relies on that fact, and here it holds.
- **Input domain** (`uitofp_i56_to_half`):
  - the base region starts after the `global_load_ubyte` that loads the i56's top byte,
    so it treats that register as an arbitrary 32-bit value. One packed f16 then
    differs (old inf, new `0x53e0`), and the variants that add the store show it as 38
    differing stored bytes;
  - the 4-before variant starts at the kernarg `s_load_dwordx4`, so the top byte is a
    zero-extended byte, and the sides agree on all 6,008 valid cases.

**Tool defects found and fixed during this work.**

1. **The L2 had no backing memory.** Every global access was malformed and halted the
   wave with PC 0. The runner now sets `L2Cache::set_backing_memory`.
2. **Stale cache lines.** A scalar-cache line was filled before the words next to the
   loaded one were seeded, so the misched pair showed a false mismatch. Every memory step
   now starts with written-back and invalidated caches.
3. **The pending check ran only at the region entry.** A wait inside the region did not
   retire what it waited for, so the check now runs before every region instruction.
   VMEM is also modelled in order, so `vmcnt(N)` retires all but the newest N.
4. **A faulting case killed the runner.** The runner now re-dispatches the wave on the
   next load.
5. **The CU's wait checker warned on every case,** because each case re-issues the same
   PCs. It is off in the runner, since the planner models waits.

**Memory tests** (all in `rj_isa_diff_tests`):

| test | observed |
|---|---|
| `StoredBytesAreTheObservableOutput` | renamed temporary stored through the same address: NO_MISMATCH_FOUND. The same value stored at offset 4: LOCAL_MISMATCH with `memory: true`, bundle `memory_differs` and `memory_differences`, replay REPRODUCED |
| `LoadsReadTheSameInputOnBothSides` | renamed load destination and stored temporary: one region with `memory_reads` and `memory_writes`; NO_MISMATCH_FOUND, no harness errors |
| `VectorMemoryCompletesInOrder` | two loads and `vmcnt(1)`: reading the first load's destination is not a conflict. Reading the second load's: CONFLICT, UNRESOLVED |
| `PcAndHwregRegionsAreUnsupportedWhileMemoryRuns` | see the section 13 table |

## One command for an LLVM change: functions and inline asm

`rj_isa_diff_pr --repo LLVM --base B --head H --out DIR` (README "Testing an LLVM
change", "Callable functions", "Inline asm") reads each changed function's old and new
assembly from the autogenerated CHECK lines of the AMDGPU tests a change updates.

- LLVM is not built. ROCm's `llvm-mc` and `ld.lld` assemble the listings.
- Test functions are mostly callable functions fed by comment-only inline asm, so they
  are tested unmodified. The earlier PR runs had to rewrite them into kernels.

Every run below used `--cases 5000`, except the sweep (`--cases 1000`).

### PR 237629 (packed shuffles, `v_pk_mov_b32` pair swaps)

| | |
|---|---|
| command | `rj_isa_diff_pr --repo <llvm worktree> --base 805c6514dcfc --head 647de95be405` |
| tests | 7 changed, all autogenerated |
| CHECK-line targets | gfx900, gfx90a, gfx942. None of the tests has a gfx950 RUN line, so every result is a proxy result |
| functions compared | 84: 33 gfx942, 33 gfx90a, 18 gfx900 |
| result | 84 NO_MISMATCH_FOUND, ESTABLISHED, complete scope. No context expansion was needed, including the two `5_u_4` functions that needed it in the kernel rewrite |
| cases | 458,928 valid; at least 5,091 per function |
| wall time | 2.5 s on 32 cores |

The earlier manual run (each function rewritten as a kernel, compiled for gfx950 with
before/after `llc`) found the same 33 changed functions on gfx950, and they all passed.

**Sensitivity.** Each mutant was written into the regenerated CHECK lines and tested with
`--head WORKTREE`; every one was caught:

| mutant | gfx942 and gfx90a (shared CHECK body) |
|---|---|
| `v_pk_mov_b32` without `op_sel` | LOCAL_MISMATCH, 4 stored bytes differ |
| `op_sel:[1,1]` | LOCAL_MISMATCH |
| `global_store_short` instead of `_d16_hi` | LOCAL_MISMATCH, 2 stored bytes differ |
| the two inline-asm defs trade registers, but the code reading them does not follow | LOCAL_MISMATCH, 11 stored bytes differ |

The fourth mutant is the one that needs inline-asm pairing. Treated as nothing, the
blocks would make old and new read independent entry values, and every function would
mismatch, mutant or not.

### PR 242731 (saturate i64 inputs before converting to half)

- 4 tests changed. Only `int_to_fp_narrow_i64.ll` (gfx942) and `slp-int-to-fp.ll` (gfx900,
  gfx90a) have a tested RUN line, so 6 functions are compared.
- All 6 pass.
- Three needed context expansion, `ctx-b1f1`, `ctx-b0f1` and `ctx-b0f8`, as in the
  kernel runs.
- 36 changed functions sit on targets that are not executed: 30 with no `-mcpu`, and 2
  each on gfx1100, gfx1250 and gfx803.

The earlier `llc`-based run compiled every test for gfx950 and so covered 12 kernels. The
CHECK-line mode tests exactly what the test file states, for the targets it states.

### Sweep: the last 120 upstream commits that touch AMDGPU tests

Each commit was tested against its parent with `--cases 1000`:

| | |
|---|---|
| changed functions compared | 162 |
| result | 132 pass, 30 inconclusive, 0 mismatches, 0 harness errors |

The inconclusive ones are scope limits:
- 16 cross-block code motion (`continuation liveness differs`);
- 4 regions spanning blocks;
- LDS (`ds_write*`), AGPR moves and scratch spills inside a change;
- branch changes.

Skipped:

| reason | count |
|---|---|
| not autogenerated | 120 files |
| test added or removed | 26 |
| a RUN line for a tested target changed | 13 |
| CHECK body missing on one side (prefix conflicts) | 27 functions |
| `amdgpu_ps` calling convention | 18 functions |

Most changed functions are on targets the tool does not execute: 3,565 with no `-mcpu`
(SI encoding), 589 gfx1100, 472 gfx1250, 358 gfx1310.

The sweep found four tool problems, all fixed and covered by the runs above:
1. **TRAPSTS exception flags.** The sticky arithmetic exception flags (TRAPSTS.EXCP 0-6)
   were checked as frame state, so every float VALU region in a function (IEEE denormal
   mode) was a harness error. They are no longer frame (11 functions).
2. **Aperture registers.** `s_mov_b64 s[0:1], src_shared_base` had unknown effects, so
   composition failed after it. Aperture registers are now named untracked state (7
   flat-atomic functions).
3. **Context variants.** The variant list kept only the 12 smallest shapes. A renamed VGPR
   temporary that the old code overwrites 11 instructions later was a LOCAL_MISMATCH. The
   largest contexts are now always kept, and `ctx-b0f20` resolves it.
4. **Wrapping addresses.** A per-lane VGPR address near 2^64 (`v[2:3] = -2` after
   reduction) wrapped past zero. The memory model seeded no word for it, so the bytes read
   depended on load order: a false LOCAL_MISMATCH in `sitofp_i48_to_half`. Such an access
   is now a MEMORY_FAULT, as on hardware. When both sides fault, the input is excluded.

Before the fixes, the first sweep reported one more false mismatch: the byte type `b21`
was not sized as a return type, so all 32 return VGPRs were observed. `iN`/`bN` are now
ceil(N/32) VGPRs.

### What remains open in this mode

- Only GFX9-family listings run. GFX10+ and SI/CI/VI listings need their own decoders and
  semantics.
- gfx900-gfx90c listings are proxies (cache-policy names translated). A gfx900 listing
  that gfx950 cannot assemble (an odd-aligned VGPR pair, `v_mad_mix`) is UNSUPPORTED.
- 16-bit return values: fixed in the next phase (`--return-masks`).
- Inline-asm pairing assumes both compiles emit the blocks in the same order.
- Per-lane VGPR addresses are unconstrained. Wrapping lanes are faults, so cases that
  contain one are rejected; `sitofp_i48_to_half` keeps 3,336 of about 7,700.

## NaNs, undefined values and poison

Before this phase, a NaN result, an undefined value or a poison input made equivalent
compiles look different. Each fix was checked on a probe pair: a correct rewrite that must
pass, and a wrong one that must not (README "NaNs, undefined values and poison").

| probe (void function storing v0, unless noted) | before | now |
|---|---|---|
| `v_add_f32 v0, v0, v1` vs `v_add_f32 v0, v1, v0` | LOCAL_MISMATCH (NaN payload) | NO_MISMATCH_FOUND; 1,132 pairs equal only through NaN equivalence. `--strict-nan`: LOCAL_MISMATCH |
| the same for `v_max_f32` | LOCAL_MISMATCH | NO_MISMATCH_FOUND (896 tolerated); strict: LOCAL_MISMATCH |
| `v_xor_b32` sign flip vs `v_mul_f32 -1.0` (fneg must keep NaN bits) | LOCAL_MISMATCH | LOCAL_MISMATCH: only one side is a floating-point result |
| undefined third element stored as-is vs set to 0 (`--entry-vgprs 2`) | LOCAL_MISMATCH | NO_MISMATCH_FOUND (2,090 masked); without the argument count, still LOCAL_MISMATCH |
| an undefined value moved from v5 vs v7 (an `IMPLICIT_DEF` reallocated) | LOCAL_MISMATCH | NO_MISMATCH_FOUND |
| new stores garbage where old stores an argument | LOCAL_MISMATCH | LOCAL_MISMATCH (a real bug) |
| `half` return; new sets the high half (`--return-masks 0xffff`) | LOCAL_MISMATCH | NO_MISMATCH_FOUND; a wrong low half stays LOCAL_MISMATCH |
| `add nsw`: wrapping vs saturating add | LOCAL_MISMATCH | LOCAL_MISMATCH; `--poison-filter nsw`: NO_MISMATCH_FOUND, 2,031 overflowing inputs excluded |
| `shl` by a register: 32-bit vs 64-bit shift | LOCAL_MISMATCH | LOCAL_MISMATCH; `--poison-filter shift`: NO_MISMATCH_FOUND |

**The first undef mask was too small.** It was built from two random refills, and a bit
of an undefined value is only masked if the refills happen to differ in it. Run against
the probes, about a quarter of an undefined value's bits stayed "defined", so the probes
still mismatched. The refills are now all zeros, all ones, then random: every bit of a
copied undefined value differs between the first two.

**The driver.** `rj_isa_diff_pr` derives the ABI and the triage from the IR:
- `--entry-vgprs` and `--return-masks` come from the signature;
- the poison filters come from the IR constructs, and are tried on a mismatch.

`isa_diff_llvm.pr_driver` checks every verdict on a synthetic history:
- commuted `fadd`: pass, and mismatch with `--strict-nan`;
- an undefined shuffle element: pass;
- a `half` return: pass;
- `add nsw` that became a saturating add: `poison-only`;
- `add` that became `sub`: mismatch.

**Real changes.** The rules never make a comparison stricter, so earlier passes stay
passes:
- PR 237629: 84 of 84 pass;
- PR 242731: 6 of 6 pass;
- the last 120 AMDGPU commits: 132 pass, 30 inconclusive, as before. In 27 valid pairs
  the undef mask was what made them equal.

Those changes do not reorder floating-point operands or rely on poison inputs, so NaN
equivalence and poison triage did not trigger there.

**Limits.**
- NaN equivalence applies where both sides' values come straight from floating-point
  instructions, through moves and stores. NaN bits that flow through integer operations
  are handled in the next phase (NaN choice).
- Undefined values are VGPRs only here (scalars follow in the next phase), and the mask
  is sampled: three refills per case.
- The poison filter matches machine conditions, not IR values. An `nsw` filter excludes
  every overflowing 32-bit add in the region, which can remove far more inputs than the
  IR's poison does (2,031 of 2,224 in the probe). The driver therefore reports
  `poison-only` as its own verdict, never as a pass.

## NaN bits through integer code, undefined scalars

Two limits from the previous phase are closed (README "NaNs, undefined values and poison").

- **NaN bits used as integers.** A NaN's unspecified bits that pass through integer
  code (`bitcast` + `and`) before being observed were compared bit-exactly.
  - Fix: the NaN's bits are treated like an undefined value. When old produces a NaN in
    a case, it re-runs with every NaN a floating-point instruction produces replaced by
    the canonical quiet NaN, all ones, then a random quiet NaN.
  - Bits that change are don't care, and new re-runs with its own alternatives.
- **Undefined scalars.** Only VGPRs had an undefined state.
  - SGPRs above `--entry-sgprs` are now undefined until written, except a callable
    function's callee-saved ones. VCC, M0 and SCC are always undefined at the entry.
  - Kernels take the count from their descriptor.
  - The driver takes it from the IR: `inreg` arguments, plus 16 for special inputs,
    calls, stack use, or targets without architected flat scratch; 24 for kernels.

| probe (void function storing v0) | before | now |
|---|---|---|
| commuted `v_add_f32`, then `v_and_b32 v0, 0xffff, v0` | LOCAL_MISMATCH | NO_MISMATCH_FOUND (1,132 pairs equal only through NaN rules); `--strict-nan`: LOCAL_MISMATCH |
| the same with a wrong mask `0xfffe` on the new side | LOCAL_MISMATCH | LOCAL_MISMATCH (ordinary values differ) |
| `v_mov_b32 v0, s40` vs `v_mov_b32 v0, s41` (an undefined SGPR reallocated) | LOCAL_MISMATCH | NO_MISMATCH_FOUND (2,341 masked); `--entry-sgprs 102`: LOCAL_MISMATCH |
| new reads garbage s41 where old reads the argument s2 | LOCAL_MISMATCH | LOCAL_MISMATCH |
| callee-saved s34 vs s35 | LOCAL_MISMATCH | LOCAL_MISMATCH (callee-saved registers are defined) |

Re-run on the same changes, nothing changed:
- PR 237629: 84 of 84 pass;
- PR 242731: 6 of 6 pass;
- the last 120 AMDGPU commits: 132 pass, 30 inconclusive. 24 valid pairs there are equal
  only through the NaN rules, and 27 only through undefined-bit masking.

`isa_diff_llvm.pr_driver` adds two functions:
- `fadd_bits`: a commuted `fadd` whose bits are masked as an integer. Pass; mismatch under
  `--strict-nan`.
- `sgpr_undef`: `store i32 undef`, with the garbage SGPR reallocated. Pass.

**Limits.**
- The mask is sampled: three alternatives per case, joint for undefined registers and NaN
  choices. A bit that varies only under other choices is still compared (conservative).
- A bit that varies under some choice is accepted with any value, even one no choice
  produces (permissive on those bits only).
- When a region has both sources, a masked difference is credited to both counters. M0
  counts as an undefined input of every global store, because the decoder reports it
  as an implicit use.

## Budget and changed code only

`rj_isa_diff test` now runs 5000 random cases per region by default, after the mandatory
boundary sweep. `rj_isa_diff_pr` already used 5000.

Only changed code is tested:
- **Comment-only changes are skipped.** The driver compares CHECK bodies without
  comments before running anything. A function whose code did not change (renumbered
  block names such as `; %Flow7` → `; %Flow8`, spill notes) is counted as comment-only,
  not run. Inline-asm blocks count as code.
- **Only regions execute.** Within a changed function, only the regions around the
  hunks run. In PR 237629, old executes 2-8 instructions per case, the region length,
  of functions of 8-12. In PR 242731 the regions are 8-66 instructions of 15-72,
  because the change rewrites most of the conversion code.

| run (default budget) | result | valid cases | wall time |
|---|---|---|---|
| PR 237629 | 84 of 84 pass | 464,716 | 4.1 s |
| PR 242731 | 6 of 6 pass | 26,560 | 32.5 s |
| last 120 AMDGPU commits | 119 pass, 30 inconclusive, 0 mismatches; 13 comment-only changes not run | 891,501 | 80 s |

The 13 comment-only functions are the 13 that earlier sweeps ran and reported as
`NO_SEMANTIC_DIFF`.

## Every RocJITsu target (Phase J)

The tool runs on every GPU RocJITsu executes: CDNA1-4 (gfx908, gfx90a, gfx940-gfx942,
gfx950), RDNA1-4 (gfx1010-gfx1036, gfx1100-gfx1153, gfx1200/gfx1201) and gfx1250/gfx1251,
at wave64 and wave32. `rj_isa_diff_pr` runs every listing whose RUN line names one of them
natively, at the RUN line's wave size and features; Vega listings run as gfx908.

| run (default budget, 5000) | result | valid cases | wall time |
|---|---|---|---|
| PR 237629 | 84 of 84 pass (18 through the gfx908 proxy) | 445,512 | 4.1 s |
| PR 242731 | 10 of 10 pass (was 6: its GFX11 RUN lines now run) | 37,952 | 25.7 s |
| last 120 AMDGPU commits | 855 functions (was 149): 791 pass, 39 inconclusive, 25 mismatch | 5,351,374 | 145 s |

Functions of the sweep by target: gfx1100 wave32 492, gfx908 (Vega proxy) 110, gfx1200 99,
gfx1030 wave32 45, gfx1010 wave32 35, gfx942 24, gfx1250 21, gfx950 13, gfx1030 wave64 12,
gfx90a 2, gfx1010/gfx1100 wave64 1 each. Still not executed: RUN lines without `-mcpu`
(3642 functions), GFX13 (gfx1310, 358), SI/CI/VI (310).

The 25 mismatches:
- **20 true positives**: commit fc40c9454b6d, "[AMDGPU][GISel] Fix stack pointer shift
  direction in selectStackRestore" (gfx1030, wave32 and wave64). The old code shifted the
  stack pointer the wrong way (`s32` 0x07ffffff vs 0xffffffe0); the change is a fix, and
  the tool reports the behaviour change it makes.
- **4 poison-only**: `v_exp2_v2bf16_fast` (gfx1200) differs only on inputs the IR's
  `fast` flags make poison; the driver's triage reports `poison-only`.
- **1 open question**: `rsq_bf16_constant_4` (gfx1250), commit 66cb135fa288 "AMDGPU: Fold
  inline immediates in peephole-opt". Old `v_rsq_bf16_e64 v0.l, 4.0 op_sel:[1,0]` selects
  the high half of the f32 inline constant (0x4080, bf16 4.0); new
  `v_rsq_bf16_e64 v0.l, 4.0` assumes the constant is read as bf16 4.0. RocJITsu reads
  the low half of the f32 constant (0) and returns +inf. Either the fold or RocJITsu's
  bf16 inline-constant format for this encoding is wrong; it needs a hardware answer.

Classes of false mismatch the sweep exposed, all fixed:
- `s_delay_alu`'s immediate is not a constant operand to RocJITsu: hints now ignore
  their operands (530 inconclusive functions in one GFX11 commit became 523 passes).
- `<N x i8>` return values were unsized (all 32 return VGPRs observed): one VGPR per
  element, low byte compared.
- Inactive lanes: VGPR liveness never killed, so renamed temporaries compared on lanes
  nothing reads again. A write now kills when EXEC never changes afterwards, and such
  exits compare active lanes only.
- EXEC = 0 inputs: on wave32 the 64-bit EXEC boundary patterns gave EXEC 0, under which
  `v_readfirstlane` reads a canary. EXEC that is still the dispatch EXEC is never 0.
- Wave32 refills wrote lanes 32-63; gfx1250 returns are `s_set_pc_i64`; SALU float NaN
  payloads needed scalar provenance tags.

Seeded bugs: an `add` turned into a `sub` in a kernel and in a function, on 14
target/wave configurations, is caught in all 28; the PR 237629 mutants (4) are caught.

## Control flow in regions (Phase K)

Regions may contain direct branches and loops (README "Control flow").
- An equivalent rewrite of a branch condition (`s_cmp_eq_u32`/`s_cbranch_scc1` ↔
  `s_cmp_lg_u32`/`s_cbranch_scc0`) passes; the inverted condition is an
  `exit_divergence` mismatch.
- A counted loop whose body and back edge both change runs inside the region: 236 of 934
  cases complete within the step limit and are compared (the rest are boundary values
  outside the derived constraints, or counts past the limit), and a wrong body is caught.
- On the 120-commit sweep the verdicts did not change: its control-flow changes are
  entered from outside the changed range (2) or leave to targets without a partner (2).
  One function (a WWM test) briefly reported a mismatch: composition skipped an edge
  inside an untested region; only READY regions with control flow run their own edges.

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
- **Reachability**: every mismatch is `LOCAL_ONLY`. The entry-prefix diagnostic now runs
  on real kernels, because their kernarg `s_load`s run. On `k3` it executes 14 / 15
  instructions from the kernel entry, with no mismatch. Entry values still stand in for
  the dispatch ABI.
- **Metadata**: compared field by field per kernel. Register-allocation fields are
  informational only when both kernels fit their allocation; everything else is semantic.
- **Pending effects** follow the gfx9 counter rules: VMEM completes in order, and SMEM may
  return out of order, so only `lgkmcnt(0)` clears scalar-load destinations.
- **Memory model**: one flat, initially random image per case, with no aliasing
  analysis. Two pointers alias only if their values are equal. Pointers come from
  pointer-shaped words (64 KiB-aligned, high half 0x10..0x7f), so loads through
  unrelated bases rarely overlap. LDS, flat, buffer, scratch and atomics are not modelled.
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
- `s_waitcnt` (fields decoded by rocjitsu's `decode_gfx9_waitcnt`) lowers a counter. For
  lgkmcnt and expcnt only a field of 0 clears destinations. VMEM is in order: the vmcnt
  state keeps the outstanding operations oldest first, and `vmcnt(N)` keeps only the
  newest N;
- joins take the maximum count and the union of destinations.

Before each region instruction the planner checks what is still pending. The contract
records the worst result as `pending_effects`:

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
| `VectorMemoryCompletesInOrder` | two VMEM loads then `vmcnt(1)`: the first load's destination is ready, the second's is pending |

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
