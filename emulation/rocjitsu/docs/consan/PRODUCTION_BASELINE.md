# ConSan production-migration baseline

This document is the reviewed Slice 0 reference for the incremental migration
defined in [PRODUCTION_DESIGN.md](PRODUCTION_DESIGN.md). It freezes the
behavioral gates and the initial structural measurements. Later slices compare
against this reference; they do not treat the current implementation shape as
the product contract.

The production and test source revision is `e9cb062907dff325f7408d2e1d99a4204fabdea0`
(`rocjitsu: define ConSan production design`, 2026-08-25). The Slice 0 commit
that adds this document changes documentation only, so this remains the exact
instrumentation revision exercised by the recorded test run.

## Contract-test inventory

The host paths name the existing tests that must move with a responsibility.
The device column names behavioral gates rather than particular register,
patch, or placement choices. Every applicable device gate includes adjacent
correct and incorrect workloads.

| Proposed component | Host gate retained at Slice 0 | Paired-device gate retained at Slice 0 |
| --- | --- | --- |
| Request and configuration | ConSan cases in `tests/dbt/hsa_hooks_unit_test.cpp`, plus `ConSanTransformMemoryTest` and the generated capability-document check | `HookLoadDefaultsToRecordReplayLds` and `RecordReplay{Default,Sparse}SamplingReduction` on every simulator target and physical gfx950 |
| Target profile and classifier | `analysis_test.cpp`, `moi_engine_conformance_test.cpp`, target rows in `core_test.cpp` and `moi_common_test.cpp`, and `ConSan.CapabilityManifestMatchesDocumentation` | Common matrix on all five targets; CDNA MFMA/direct-to-LDS/transpose, RDNA WMMA/FP8, and gfx1250 cluster/TDM/high-bank/matrix/async-transfer extensions |
| Runtime capabilities | HSA memory-pool, automatic-report, binding, and failure-policy cases in `hsa_hooks_unit_test.cpp` | Record/Replay correct/incorrect rows on all five simulator backends and physical gfx950, including the fine-grained snapshot assertion in the default-activation pair |
| Program inventory | `analysis_test.cpp`, `composition_test.cpp`, the fault/ordinary/synchronization test files, and ownership/provenance cases in `moi_common_test.cpp` | Shared-helper, mixed-owner, dynamic-stack, group-FLAT, subword, indexed-alias, barrier, atomic, fence, and target-exclusive pairs |
| Engine policy | `supercollider_test.cpp`, `moi_common_test.cpp`, `moi_record_replay_test.cpp`, `moi_sampled_test.cpp`, `moi_inline_shadow_test.cpp`, and `moi_engine_conformance_test.cpp` | Baseline plus every applicable engine for the common matrix on all five simulator targets and physical gfx950 |
| Coverage ledger and trust | Site-disposition and lowering-outcome cases in `core_test.cpp`, the MOI engine tests, and static-coverage/diagnostic guards in `hsa_hooks_unit_test.cpp` | Every instrumented correct/incorrect pair; lifecycle pairs additionally require complete multi-object coverage |
| Evidence planner and decoder | `consan_moi_report_plan_test.cpp`, `moi_record_replay_model_test.cpp`, `moi_inline_atomic_test.cpp`, and report/snapshot cases in `hsa_hooks_unit_test.cpp` | Dispatch-identity, deterministic replay, module/graph lifecycle, sampling, bounded-pressure, and atomic/barrier/fence pairs |
| Resource planner | `resource_test.cpp`, `consan_moi_adversarial_test.cpp`, and resource/placement cases in all three MOI engine tests | Dynamic and mixed private state, large text/LDS, full-bank, AccVGPR, dense/sparse routing, many-owner, and target-specific pressure pairs |
| Lowering and placement | `supercollider_test.cpp`, all three MOI engine tests, `branch_only_relay_router_test.cpp`, `composition_test.cpp`, and exact mechanism tests in `core_test.cpp` | Common semantic pairs plus each real target-exclusive instruction family; device assertions remain output/diagnostic contracts rather than byte or register goldens |
| Runtime coordinator | HSA reader, replacement, executable, dispatch, report, concurrent-load, teardown, and rollback cases in `hsa_hooks_unit_test.cpp` | Repeated/cross-kernel dispatch identity, graph replay/update, module lifecycle, heterogeneous objects, and the final physical health row |
| Engine analyzers and verdict | `moi_record_replay_model_test.cpp`, `moi_inline_atomic_test.cpp`, Sampled/Inline model cases, and hook report-summary tests | Ordered members forbid diagnostics; incorrect MOI members require the selected engine's conflict signal across the applicable matrix |
| Mutation pipeline | `fault_injection_test.cpp`, `fault_barrier_test.cpp`, `fault_atomic_test.cpp`, `fault_ordinary_test.cpp`, `fault_composition_test.cpp`, and `composition_test.cpp` | Adjacent incorrect source members throughout the device matrix plus the focused atomic-order mutation pair; reviewed E2E fault manifests remain qualification evidence |

The complete host ownership list is the sources registered by
`rj_add_consan_unit_sources`, the ConSan cases in `hsa_hooks_unit_test`, and the
capability-manifest check. A slice that moves a responsibility must select the
focused rows above and then run the complete gates below. A filename in this
inventory does not make all of its patch-shape assertions permanent: section
8.3 of the production design governs correction of prototype-entrenching
tests.

## Behavioral-oracle audit

The configured device matrix contains 1,750 named contract families. Every
family has exactly one `.Correct` and one `.Incorrect` row, for 3,500 paired
rows. The only additional `consan-device` row is
`ConSanDeviceGfx950Physical.PostInstrumentationHealth`; it is intentionally an
unpaired health check rather than a sanitizer scenario.

- Each fixture's GoogleTest body checks its application result and control
  state. Baseline rows run those same correct and incorrect bodies without the
  hook, proving that the intentionally missing ordering edge is not itself an
  excuse for a failed application oracle.
- Correct MOI rows set `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1`; incorrect
  Record/Replay rows require a replay conflict and incorrect Sampled/Inline
  rows require diagnostics. Focused rows add stronger evidence or coverage
  regexes where the generic contract is insufficient.
- SuperCollider does not implement a happens-before race diagnosis. Its
  incorrect member is therefore explicitly labelled `mutation-only`: it must
  instrument successfully and preserve the independent application oracle,
  but it does not make a false diagnostic promise.
- `RJ_CONSAN_REQUIRE_PATCH=1`, strict policy, overflow guards, and the shared
  checked-output runner prevent an inert hook, incomplete transform, child
  failure, or missing expected diagnostic from passing as behavioral evidence.

The pair-name audit found no missing peer and the contract review found no
device scenario lacking its declared behavioral oracle. Any future exception
must be typed as not applicable or health-only; silently adding a one-sided
scenario is not allowed.

## Initial quantitative baseline

Measurements cover production files below
`lib/rocjitsu/src/rocjitsu/code/patch/consan` and
`lib/rocjitsu/src/rocjitsu/hooks/consan`. Counts are textual unless a row says
otherwise. Keeping the measurement rule beside the number prevents later
file moves from masquerading as architectural progress.

| Section 9.3 measure | Slice 0 value | Measurement rule and interpretation |
| --- | ---: | --- |
| ConSan patch lines | 81,509 | `wc -l` over every file in the patch directory |
| ConSan HSA-hook lines | 10,747 | `wc -l` over every file in the hook directory |
| Total patch and hook lines | 92,256 | Sum of the preceding rows; line count is context, not a deletion target |
| Public request fields | 0 | `ConSanRequest` does not exist yet |
| Internal compatibility-option fields | 135 | Direct Clang AST `FieldDecl` count for the current combined `ConSanOptions`; this becomes the initial compatibility shape |
| Program-inventory fields | 0 | No separated `ProgramInventory` exists yet |
| Observation-plan fields | 0 | No separated `ObservationPlan` exists yet |
| Result fields | 87 | Direct Clang AST `FieldDecl` count for the current combined `ConSanResult` |
| Target-ID references outside named profile/classifier/lowerer boundaries | 539 | Occurrences of `ROCJITSU_CODE_(ARCH\|TARGET)_*` in patch/hook production code. No such named boundaries exist at Slice 0, so the strict baseline charges every occurrence; Slice 1 must classify, move, or justify each one. |
| Engine-condition references in current site-admission files | 20 | `moi_engine` comparisons/switches in `consan_moi_candidates.inc` (4) and `consan_moi_pipeline.inc` (16). This proxy is paired with the named access/barrier/atomic/fence policy inventory in the design; it is not a claim that every comparison is duplicate code. |
| Direct hook patch-kind references | 87 | `ConSanPatchKind` occurrences in the hook directory |
| Direct hook patch-vector references | 12 | `result.patches`, `patch_result.patches`, or `inventory.patches` occurrences in the hook directory |
| Direct hook geometry-member references | 29 | Uses of anchor/trampoline sizes or offsets, scratch VGPR, or private/group-segment geometry on a `patch` in the hook directory; all patch coupling is confined to three hook files |
| `LegacyConSanLowering` consumers | 0 | The adapter and compatibility projection do not exist yet. Every consumer added later must be named and then removed by its owning slice. |

The hook coupling rows expose the current semantic problem more usefully than
one aggregate number: coverage, report attribution, dispatch-segment growth,
dynamic-private interception, require-patch policy, and detailed logging all
read concrete patches directly. Later comparisons must reduce decision-making
coupling; moving debug-only rendering behind a typed telemetry view is not the
same as deleting a behavioral consumer.

## Complete Slice 0 gate

The build is `/home/ossci/xx/rocjitsu-build`, configured from this source tree
with `RelWithDebInfo` and TheRock's ROCm distribution at
`/home/ossci/xx/TheRock/build/dist/rocm`. The reviewed commands are:

```sh
ninja -C /home/ossci/xx/rocjitsu-build -j64
ctest --test-dir /home/ossci/xx/rocjitsu-build \
  -L consan-device -L simulator --output-on-failure -j64
ctest --test-dir /home/ossci/xx/rocjitsu-build \
  -L consan-device -L physical --output-on-failure -j64
```

The registered device gate is 3,501 rows: 2,908 simulator rows and 593
physical-gfx950 rows. Simulator counts are gfx942 572, gfx950 592, gfx1100 538,
gfx1201 548, and gfx1250 658. The complete one-command run also exercises the
ConSan host tests and all non-ConSan RocJitsu tests:

```sh
ctest --test-dir /home/ossci/xx/rocjitsu-build \
  --output-on-failure -j64
```

Result on 2026-08-25: all 9,445 enabled CTest rows passed in 571.71 seconds
wall time (`user` 1,897.59 seconds, `sys` 1,949.35 seconds). CTest reported one
pre-existing disabled BinaryTranslator row and five expected skips; no ConSan
gate was disabled or skipped. The run included all 3,501 `consan-device` rows,
the complete ConSan host ownership above, and the rest of RocJitsu's registered
suite. The `consan-device` aggregate process duration was 5,042.53 seconds,
including 4,678.32 seconds for all simulator-labelled ConSan rows and 435.14
seconds for the serialized physical tier. Label durations overlap and therefore
must not be summed.

## E2E qualification record

Slice 0 changes no instrumentation behavior, and the external model/framework
campaigns do not fit the checked-in test timebox. They were not rerun. The
reviewed E2E evidence remains the 2026-08-24 snapshots in `STATUS_CDNA3.md`,
`STATUS_CDNA4.md`, `STATUS_RDNA3.md`, `STATUS_RDNA4.md`, and
`STATUS_GFX1250.md`, using the procedures and manifests in `VALIDATION.md`.
Any later slice that touches behavior absent from the checked-in pairs must
select the relevant E2E rows and record fresh evidence rather than inheriting
this deferral.
