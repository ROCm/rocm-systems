# ConSan Device-Test Plan

This document tracks the remaining work on the checked-in ConSan device-test
tier. [PLAN.md](PLAN.md) gives the broader project plan, while
[VALIDATION.md](VALIDATION.md) documents how to run the current device matrix.

## Overall goal

Build a checked-in, self-contained ConSan device-test suite that gives a
developer working on one GPU serious confidence that a change has not regressed
any supported architecture. Its coverage must be distilled from the concrete
workloads and low-level idioms in the current [VALIDATION.md](VALIDATION.md)
campaign and the Aorta suite, while remaining small, deterministic, free of
external model assets, and independent of the implementation being replaced.

Every semantic scenario must provide an adjacent correct/incorrect behavioral
pair: the correct workload proves exact results and the absence of a
diagnostic, while the incorrect workload preserves an independent oracle where
possible and requires the appropriate semantic diagnostic. Run these contracts
through the uninstrumented baselines and all four ConSan engines using RocJitsu
on `gfx942`, `gfx950`, `gfx1250`, `gfx1100`, and `gfx1201`, and run the same
CDNA4 coverage on the physical `gfx950`; do not use FFM.

The single whole-matrix CTest command must ultimately pass completely. Fix
every issue exposed while building the suite, including preexisting defects in
ConSan, RocJitsu, runtime support, or the tests themselves, and retain each fix
as a regression test rather than weakening or normalizing the failure. On the
current wide reference host at `-j64`, keep the complete suite within the
5--20-minute wall-clock heuristic: a faster suite prompts a coverage review,
while a slower suite prompts a review for redundant low-value cases.

## Directory layout and extension points

All paths below are relative to the RocJitsu source root unless identified as
build-tree paths.

| Location | Current role | How to extend it |
| --- | --- | --- |
| `tests/dbi/consan/device/` | The checked-in behavioral-conformance tier. Twenty-two all-engine `*_test.hip` pairs cover the portable workload abstractions. Shared focused pairs additionally cover group-FLAT and Sampled dual-address LDS budgeting on all five targets. Adjacent family/target sources cover CDNA MFMA/full-bank/AccVGPR/B96, clobbering LDS reduction loads, access-heavy two-tile Stream-K, dense routing, and large-text wave64 shapes; gfx950 large LDS; RDNA B96/WMMA/FP8; gfx11 VGLOBAL/lifecycle identity; and gfx1250 clustered/TDM plus high-bank/double-barrier LDS addressing. The gfx1250 cluster host fixture submits a real extended HSA dispatch packet rather than hiding cluster dimensions behind an ordinary HIP launch. | Add one descriptively named source per new semantic scenario. Prefer extending a scenario with another tightly related pair over creating broad clean/racy grab bags. |
| `tests/dbi/consan/device/consan_device_test_support.h` | Small shared fixture utilities used by the paired HIP sources. | Put only genuinely reusable fixture mechanics here; keep scenario semantics and expected results local to each test source. |
| `tests/dbi/consan/hip_consan_{lds,moi,moi_cdna,moi_rdna3,inline_shadow,spill_gfx950}_test.hip` and related support | Older target- or engine-specific device fixtures. They provide useful ISA and implementation test material but are not the common behavioral conformance tier. | Distill behaviorally justified idioms into the new paired suite. Keep implementation-specific assertions here when they remain useful, but do not count them as substitutes for portable correct/incorrect contracts. |
| `tests/consan/CMakeLists.txt` | Builds target-specific HIP executables and registers every baseline/engine/backend row, labels, diagnostic requirements, simulator configuration, target-scoped physical resource lock, and health dependency. The shared registration path rejects any physical device row without a lock. It is included by `tests/CMakeLists.txt`. | Register every new pair here across the common matrix. As the table grows, move the declarative pair inventory and registration helpers into a focused `tests/consan/device_tests.cmake` included from this file rather than duplicating per-architecture blocks. |
| `tests/consan/run_checked_test.cmake` and `tests/consan/check_capability_manifest.cmake` | Shared process/output and capability checks used by ConSan tests. | Add only generic fail-closed harness behavior here; scenario semantics and result oracles belong with the HIP workload. |
| `configs/gfx942_cdna3_kmd.json`, `configs/gfx950_mi355x_kmd.json`, `configs/gfx1250_mi455x.json`, `configs/gfx1100_w7900.json`, and `configs/gfx1201_r9700.json` | RocJitsu simulator descriptions selected by the CTest registrations. | Reuse these for the suite. Change them only for a genuine target-model correction, never to encode test-specific behavior. |

Create new common scenarios under `tests/dbi/consan/device/`. Prefer one
descriptively named HIP test source per semantic scenario, with its correct and
incorrect workloads, shared workload body, host oracle, and diagnostic
contract kept together. Put genuinely shared fixture utilities in a small
header in that directory. When source-level HIP cannot force an important ISA
idiom, add the narrowest family- or target-specific source beside the common
scenario and reuse the same behavioral oracle; do not fork the whole suite by
architecture.

The external Aorta checkout and the asset-heavy workloads referenced by
`VALIDATION.md` remain E2E inputs, not source locations for this checked-in
tier. Their reduced scenarios belong in the new RocJitsu device subtree above.
The current local build emits executables under
`/home/ossci/xx/rocjitsu-build/tests/` and isolated state under
`/home/ossci/xx/rocjitsu-build/tests/runtime/ConSanDevice*`; these are generated
artifacts and must never be edited or checked in.

## Behavioral contract

Every semantic scenario must have two adjacent variants built from the same
basic workload shape:

1. **Correct workload:** synchronization and memory ordering are valid, the
   program's exact result matches a host oracle, and ConSan emits no diagnostic.
2. **Incorrect workload:** one relevant synchronization or ordering property is
   deliberately broken, and ConSan emits the declared semantic diagnostic. The
   fixture should retain an independent output/control oracle wherever possible
   so that detection is distinguishable from a crash, hang, or corrupted
   dispatch.

The pair must differ only in the semantic property under test. Diagnostic
checks must describe the race or ordering violation, not implementation details
such as patch counts, generated instructions, helper layout, selected
registers, spill strategy, or code-cave placement. This contract is intended to
survive the production replacement of the prototype implementation.

Baseline execution remains useful for both variants: it proves that the code
object launches and that the independent control oracle works. An
uninstrumented incorrect workload is not expected to diagnose itself or to
produce a deterministic value from the deliberately racy data path.

## Current state

The portable tier has 22 adjacent correct/incorrect all-engine pairs. Each
common pair runs as baseline plus all four engines on five RocJitsu targets
and physical `gfx950`, for 60 rows per scenario. Shared and target-specific
pair names cover behavior only on the architectures and engines where the form
exists. E2E-derived focused regressions add further adjacent pairs where one
engine-specific resource/control path is the behavior under test. The current
registered `consan-device` matrix is 1,893 tests, including the 60-row
heterogeneous-object pair, the 60-row fence/barrier-publication pair, the
60-row ordinary release-store pair, the 24-row Sampled dual-address pair, the
36-row baseline/Inline/Sampled empty-EXEC scalar-spill pair, four physical
module-load lifecycle rows, four gfx1250 high-bank/double-barrier LDS rows, and the
physical post-instrumentation health row. The fresh one-command qualification
is fully green: all 1,540 simulator rows
and all 353 serialized physical rows pass, including the physical
post-instrumentation health check. The complete 1,893-row matrix takes 120.12
seconds of wall time and 2,389.66 seconds of summed CTest process time on the
current reference host.

| Scenario | Workload-derived contract | Status |
| --- | --- | --- |
| Ordered tile handoff | Cross-wave LDS publication and exact consumer values | Paired and green |
| Reduction | Tree-style LDS reduction with one missing publication edge | Paired and green |
| Shared helper owners | One noinline helper reached from two kernel owners | Paired and green |
| Independent workgroups | Three-dimensional dispatch identity without false LDS aliasing | Paired and green |
| Repeated dispatch identity | Reuse of the same code object across dispatches without stale sanitizer identity | Paired and green |
| Dynamic private stack | Private-stack state combined with ordered or unordered LDS traffic | Paired and green |
| Overlapping subwords | Adjacent non-overlap versus true byte-range overlap | Paired and green |
| Atomic arrival | Atomic counter publication followed by shared-data consumption | Paired and green |
| Atomic CAS publication | Release compare-exchange publication followed by a language-level acquire load; the incorrect member acquires a distinct ready state object | Paired and green; exact results and all 60 baseline/engine/backend rows pass |
| Atomic store publication | Language-level release store followed by an acquire compare-exchange; the incorrect member releases a distinct phase object while preserving the same store/CAS shape | Paired and green; exact results and all 60 baseline/engine/backend rows pass |
| Stream-K last arriver | Partial publication, fetch-add arrival, and last-arriver consumption | Paired and green |
| Tree atomic-OR | Bitmask publication, polling, and completion | Paired and green |
| Double-buffered pipeline | Wide LDS stages, storage reuse, lane exchange, and a missing stage edge | Paired and green |
| Histogram/scatter | LDS-bin atomics and collision-heavy global scatter updates | Paired and green |
| Segmented top-k | Value/index tuples, segmented phases, wide traffic, and non-multiple tails | Paired and green |
| Two-stage softmax | Max/sum-style local reduction plus a global intermediate | Paired and green |
| Backward/optimizer | FP16 gradient traffic, FP32 reduction/moment state, and AdamW-like update | Paired and green |
| MoE routing | Top-1 routing, empty experts, uneven tails, prefix offsets, and indexed gather/scatter | Paired and green |
| Continuous batching | Changing active rows, masked tails, repeated reset-state dispatch, and stable checksums | Paired and green |
| Heterogeneous object | Three executed attention/optimizer/MoE kernels, a shared helper, and an unexecuted high-pressure kernel in one object; only the attention publication edge differs | Paired and green; Sampled's cross-kernel attribution regression is host-guarded |
| VGLOBAL cache publication | Target-native release/acquire atomic publication and required cache sequencing | Paired and green |
| Sampled multi-range budget | Two dual-address LDS sites whose banked report ranges must not consume the site-count patch budget | Paired and green in baseline and Sampled |

The target-specific tranche adds CDNA MFMA/AccVGPR liveness, full-bank dynamic
Stream-K, an 80-access/five-barrier two-tile Stream-K relay shape, native
B128-to-AccVGPR, dense far-routed LDS access and barrier sites with live SCC,
a D128-derived B16 store between an SCC-producing compare and SCC-consuming
branch,
a gfx950 160-KiB/32-stage LDS pipeline, a
CDNA3/CDNA4 large generated-text wave64 shape, wide group-FLAT on all five
targets, CDNA and RDNA B96 aliasing
boundaries, instruction-encoded atomic scope, WMMA and FP8 staging, both gfx11
VGLOBAL address forms, repeated multi-stream image
identity, four real clustered gfx1250 dispatch contracts covering cluster
barriers, multi-cluster isolation, direct-to-LDS async load/wait, and
multicast, and a gfx1250 high-bank/double-barrier LDS contract distilled from
Tensile.

This expansion exposed and retained three more real CDNA defects. Inline Shadow
could place an entry owner/epoch backup through live AccVGPR-backed storage in
the segmented-top-k object. SuperCollider could not instrument a native-LDS
site while a dynamic private frame was active. The fixes now select legal
ordinary-register storage below the accumulator boundary and use an existing
bracket-local dynamic-stack spill frame, respectively; focused host tests
protect both resource contracts. PyTorch `torch.mode` then exposed that all 20
remaining unsupported CDNA4 accesses were the same rocPRIM `ds_read_b96`
idiom; the new CDNA B96 pair protects that capability on gfx942 and gfx950.

### E2E-to-device regression extractions

| E2E source | Failure distilled into a checked-in contract | Quick coverage |
| --- | --- | --- |
| Qwen, PyTorch selection/reduction, and Aorta heterogeneous-framework workloads | A compact object executes attention-like tile publication, a mixed-precision optimizer, and MoE-style LDS atomic routing through a shared epilogue while retaining an unexecuted register-rich multi-phase kernel. The adjacent members differ only by the attention publication edge and require exact results and control evidence from every independent dispatched stage. | The all-engine pair contributes 60 rows across all five RocJitsu targets and physical gfx950. Its first run exposed Sampled falsely comparing records from disjoint kernels in one code object because the reader ignored dispatch, cluster-workgroup, and static kernel-owner compatibility. A focused hook regression proves a real same-scope conflict remains while each mismatching identity dimension suppresses false attribution. All 167 hook tests, all 288 simulator Sampled rows, the 50 focused simulator rows, and the 10 physical rows pass. |
| Physical-gfx950 PyTorch `torch.sort`/`torch.topk`, Inline Shadow and Sampled | Full-pressure selection kernels can reach a displaced LDS operation with `EXEC=0`. The correct member forces the same scalar pressure, empties `EXEC`, and requires exact scalar preservation with no diagnostic. The adjacent incorrect member retains the pressure with nonempty waves and requires the exact LDS conflict diagnostic while preserving its scalar checksum. Both members contain eight access sites so that they exercise scalable appended-body routing rather than only a one-site compact case. | The shared target-native source runs on RocJitsu `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`, plus physical `gfx950`. Baseline, Inline Shadow, and Sampled registrations give 36 rows. The 24 baseline/Inline rows pass in 1.13 seconds; the ten new simulator Sampled rows pass in 1.15 seconds and the two physical rows in 0.55 seconds. The reduction first exposed fixed-stack gfx1201/gfx1250 Inline routing failures and now also protects Sampled's branch-only scalar-spill routing and leading empty-wave guard. Its cross-target run immediately exposed a second bug: an RDNA3 private-state prologue captured workitem identity only after its scalar-spill scratch had clobbered `v0`. A focused host regression pins capture-before-clobber ordering, and the corrected gfx1100 incorrect member again observes distinct wave owners and the required conflict. The repaired physical TopK Sampled E2E process passes both exact BF16/FP64 value/index oracles with zero diagnostics and complete dynamic evidence in 92.06 seconds, instrumenting 232,814/239,722 accesses and all 6,743 applicable barriers; its residual 6,908 no-first-SOPP-hop sites remain an E2E placement frontier rather than a missing behavioral contract in this pair. |
| Physical-gfx950 PyTorch `torch.sort`, Inline Shadow owner-local planning | After the empty-wave behavior was covered, the full generated object still corrupted 744 output indices because disconnected kernels with different SGPR tails were forced through an unsafe object-wide scalar ABI. This is not a new race contract: it is the resource-planning envelope around the existing eight-site correct/incorrect pair. | A focused host regression synthesizes disconnected 64- and 96-SGPR owners, requires both device access sites to survive lowering, and verifies that only the incompatible owner receives scalar-spill and persistent-dispatch overrides. All 722 `ConSanMoi.*` tests pass, while the final physical E2E run passes exact values and indices with complete 56,884-access/6,032-barrier coverage. Do not add a prototype-layout assertion to the device tier; expand the paired workload only if a future replacement exposes a distinct observable behavior not already covered by the empty/nonempty-wave contract. |
| Physical-gfx950 PyTorch norm/softmax, Sampled | The exact device oracle completed before the 30-second limit, but host teardown scanned all 46,080 allocated watchpoint slots even though the report contained zero claimed windows. The reader now stops after the committed `Ready` count and reconstructs deferred releases with one scan per relevant owner bank instead of one capacity scan per visible access. This is report-reader complexity, not a new device semantic: the existing adjacent two-stage-softmax pair owns the reduction/global-intermediate behavior and the existing Stream-K pairs own Sampled acquire-release publication. | Two focused host regressions pin zero examined slots for an empty allocated report and one capacity scan, rather than two, for two visible entries sharing a pending owner bank. All 186 hook tests pass. The 18 selected Sampled Stream-K correct/incorrect rows pass across all five RocJitsu targets plus physical gfx950 in 0.94 seconds. The final E2E run passes exact norm/softmax, complete 4,820-access/2,030-barrier coverage, and the ordinary 30-second contract in 28.98 seconds. |
| Physical-gfx950 PyTorch norm/softmax, Inline Shadow | Compiler-generated reduction loads use tied address/result operands (`ds_read_b32 vN, vN`) while ordinary and accumulator liveness leave only a compact spill window disjoint from `vN`. The paired reduction preserves the max/sum shape and exact live ordinary/AccVGPR values; its incorrect member removes only the LDS publication barrier. | The new source runs baseline plus all four engines on RocJitsu gfx942/gfx950 and physical gfx950, for 30 passing rows. Inline Shadow additionally requires selection of the 16-VGPR spill-backed recovery path. Three host regressions prove both overlapping and disjoint clobbered-address cases and forbid an out-of-window snapshot. This closes the compact-VGPR resource defect, but not the E2E cell: 41 full-pressure sites still need a scalar branch-only body whose nearest safe relay is roughly 1.5 MiB away. |
| Physical-gfx950 HIP Stream-K and rocBLAS, Record/Replay | Exact device oracles finished, but repeated field-wise scans and replay directly from 237--526 MB fine-grained HSA report allocations imposed a 45--120-second post-oracle latency floor. The reader now takes one sequential snapshot into cacheable host storage before any parser or replay pass. This is host report-reader complexity, not a new device semantic: the existing adjacent Stream-K last-arriver and 80-access/two-tile pairs own the publication and conflict behavior. | A focused hook regression seeds the real fine-grained allocation path and pins the full snapshot byte count. All 186 hook tests pass, and all 30 two-tile baseline/four-engine rows pass on gfx942/gfx950 simulation plus physical gfx950. The E2E simple, two-tile, and `rocblas_sgemm` exact workloads now complete with full static/dynamic coverage in 10.83, 28.44, and roughly 36 seconds respectively. |
| Physical-gfx950 PyTorch `torch.mode`, Record/Replay | Even after sparse compaction, the exact fully covered row spent roughly 13 seconds copying semantically visible fields from a 533-MB fine-grained HSA report during host teardown. Record/Replay auto reports now prefer coarse-grained storage when available and use the existing HSA bulk-snapshot path, while retaining fine fallback and leaving Sampled and Inline Shadow's fine preference unchanged; SuperCollider has a separate allocator. This is host allocation policy, not a new device race semantic: the existing reduction, B96 tuple-publication, segmented-top-k, and replay pairs own the relevant transformed-device behavior. | A focused host regression offers fine then coarse regions and requires Record/Replay's coarse preference and fine fallback plus the two other MOI engines' unchanged fine preference. All 166 hook tests, all 250 simulator Record/Replay device rows, all 58 physical Record/Replay rows, and the physical health check pass. The physical E2E row falls from 40.79 to 27.61 seconds while retaining the exact oracle, complete 25,523-access/3,920-barrier coverage, lossless replay of 13,017 accesses plus 49 barriers, and zero diagnostics. Do not add a prototype-specific device assertion for region selection; extend the behavioral pairs only if the production replacement exposes a distinct device-observable contract. |
| Physical-gfx950 PyTorch `torch.topk`, Record/Replay | The 20.8-MB generated object needs 403,542,016 bytes of alignment-inclusive growth, just above the former 384-MiB default, so strict validation rejected before either exact oracle. Qualified generated workloads now define a still-bounded 400-MiB default envelope without a workload-specific expert override. This is resource admission rather than a new device semantic: the adjacent segmented-top-k pair already owns value/index tuples, FP64/BF16-like pressure, tails, phase publication, and the missing-edge diagnostic. | A focused host policy regression pins admission of the observed envelope and all 15 growth-policy tests pass. The E2E row now exits cleanly and passes both exact value/index oracles with complete dynamic replay; it remains orange because the large object omits 6,908 accesses and 4,680 barriers and its FP64 device execution alone takes 93.19 seconds. A second focused host pair proves that the dead tail of an already-selected multiword access is an exact, validated relay donor for later selected accesses, while an intrinsically unreachable first hop is rejected before body reservation. No new device pair is claimed for that planner rule: `SegmentedTopK` owns the observable value/index and missing-edge behavior, `CdnaWave64LargeText` owns long-range execution, and the `ScalarSpillExec` rows own the cross-target scalar-pressure/empty-wave envelope. A combined no-cave/full-pressure/large-text pair would remain a useful future behavioral test, but it cannot be made green by adding fixture caves without deleting the condition under study; add it when the owner-local routing/layout design can instrument the real shape completely, and do not assert the prototype's selected-tail choice. |
| Physical-gfx950 PyTorch norm/softmax, Record/Replay | A barrier-only replay can have a zero-access compact input and zero-byte replay scratch while retaining the full report's nonzero diagnostic capacity. The validator conflated those producer contracts and rejected an otherwise exact, fully covered clean run. This is validation bookkeeping rather than a new device semantic: the adjacent two-stage-softmax pair already owns reduction/intermediate publication and the missing-barrier diagnostic. | Focused parser regressions pin both a barrier-only zero-input replay and a nonempty replay whose compact scratch capacity is clamped below the full report capacity; a third retains compatibility with historical summaries. All 212 validator tests pass. Re-parsing the retained E2E log accepts exact norm/softmax, full 4,820-access/2,096-barrier coverage, complete static/analysis/dynamic verdicts, and zero diagnostics. |
| Physical-gfx950 PyTorch `torch.sort`, Sampled | PyTorch radix-sort kernels contain dual-address LDS instructions: 56,884 static access sites expand to 70,420 logical report ranges. Sampled incorrectly compared those banked report slots with `max_patches`, a site-count budget, and silently omitted 3,980 sites. | A cross-target host regression uses two native dual-address sites, a two-site patch budget, and 32 banked report slots on CDNA3/4/5 and RDNA3/4. The adjacent device pair executes the same correct/missing-edge contract under baseline and Sampled on all five RocJitsu targets plus physical gfx950; all 24 rows pass. The repaired physical E2E row passes the exact value/index oracle with complete 56,884-access/6,032-barrier coverage. |
| Framework and RCCL-style producer/consumer state machines | A producer writes an LDS payload and publishes a phase transition with release compare-exchange; a consumer uses a language-level acquire load, which lowers to an ordinary load/wait/cache sequence on the relevant targets, before reading the payload. The incorrect member retains the same CAS and load shape but acquires an already-ready distinct phase object, so it does not import the producer's publication edge. | The adjacent all-engine pair contributes 60 green rows across all five RocJitsu targets and physical gfx950. Its first run exposed cross-architecture ordinary global-load decoding gaps, Record/Replay's split atomic/fence state, post-guest CAS event reservation, and missing Sampled/Inline handling for language-level ordinary atomic-acquire sequences. Focused host tests retain each transformation and model contract without asserting code-cave layout. |
| Framework and RCCL-style ordinary release stores | A producer writes an LDS payload and publishes readiness with a language-level agent-scope release store; a consumer performs an acquire compare-exchange before reading the payload. The incorrect member releases a distinct phase object while the consumed object is already ready, preserving the release-store/acquire-CAS instruction shape without importing the payload edge. | The adjacent all-engine pair contributes 60 green rows across all five RocJitsu targets and physical gfx950. Its first run exposed missing ordinary release-store association and engine semantics, incomplete compiler-release wait/cache recognition, and incorrect CDNA5 scaled-VGLOBAL effective-address reconstruction. Focused host tests retain exact association, per-engine publication state, and CDNA5 scale-bit/address materialization while rejecting malformed or cross-architecture scale use. |
| RocJitsu-gfx1250 Tensile `007_sk_mxf4gemm_tdm`, Record/Replay | Dense LDS accesses sit beside an addressed buffer acquire, device-scope global invalidate, and split-barrier signal/wait. A far fence with no ordinary island must reuse an access relay under owner-local scalar pressure, including space reserved by Record/Replay's barrier dispatcher. | A focused two-owner host regression pins 18 access patches, 4 barrier records, 2 far-fence patches, owner-local call-return routing, and final validation. The adjacent checked-in device pair composes an independent target-native workgroup barrier with cache-mediated LDS release/acquire publication. gfx1201/gfx1250 use native split signal/wait and gfx942/gfx950/gfx1100 use their monolithic semantic equivalent; all 50 simulator and 10 physical rows pass. Exact-size sharding now supersedes the monolithic 1,800-second timeout: the source-matched E2E bundle passes all 96/96 numerical rows with complete access/barrier/fence coverage and lossless replay, records a current 19.47x paired slowdown, and accepts a prospectively reviewed exact-one qualified miss with complete containment, health, and cleanup evidence. The E2E cell is green. |
| RocJitsu-gfx1250 Tensile `007_sk_mxf4gemm_tdm`, high-bank and adjacent-barrier LDS tail | Generated MXF4 kernels select a nonzero gfx1250 SRC0 bank and name an LDS address such as physical `v286` through encoded `v30`; high-pressure variants also require a spill-backed low-bank instrumentation window. The selected kernel places adjacent tensor-to-LDS split barriers around a descriptor wait. Record/Replay must capture the physical high-bank address before selecting low scratch without overwriting a live spill victim, then execute the guest instruction under its complete original bank mode. | The adjacent gfx1250-only device pair preserves the same encoded high-bank address in correct and missing-publication workloads. Its correct member now mirrors the E2E tail with split barriers separated by `s_wait_dscnt`; the incorrect member removes the publication edge. Baseline and Record/Replay contribute four green rows with exact results/no diagnostic for the correct member and the required conflict for the incorrect member. Focused host regressions separately pin appended placement despite inline padding, composite SRC0/DST mode normalization, save-before-capture ordering for a seven-VGPR spill window, and bounded inventory/fault sharding. The source-matched E2E bundle passes all 96/96 numerical rows, processes 5,281,152 accesses plus 108,992 barriers without loss, records 19.47x paired overhead, and accepts a prospectively reviewed redundant-barrier qualified miss with complete containment and health evidence. |

This extraction is intentionally phrased as an empty-wave spill behavioral
contract rather than a radix-sort or gfx950 implementation test. The scalar
pressure is architecture-neutral; only the target-native LDS mnemonic differs.
It also demonstrates why an E2E reduction should retain enough static sites to
cross meaningful planner thresholds: the eight-site pair covers both the
original CDNA4 failure and a distinct RDNA4-family routing defect without
increasing the test's wall time beyond the quick-suite scale.
Future E2E fixes should follow this pattern whenever a large workload exposes
a compact control, resource, instruction, or synchronization idiom.

## Workload-derived coverage model

The checked-in tier is a reduction of the E2E corpus, not a collection of
generic GPU examples. The current validation corpus contains:

- framework and model workloads: Qwen prefill; Sharktank TP1, TP2, and CLIP;
  PyTorch mode, top-k, sort, scatter-reduce, histogram, norm/softmax, compiled
  softmax, and split online-softmax; and llama.cpp quantized matvec and RMSNorm;
- generated and production kernels: Tensile Stream-K, sparse/TDM, and mixed
  FP4/FP8 GEMMs, plus RDNA4 production FP16/FP8 matmul; and
- focused workload reductions: hip-moi attention, pressure, Stream-K arrival,
  tree atomic-OR, and Jakub matmul.

The portable fixtures cover useful foundations: a B32 cross-wave LDS handoff,
an integer tree reduction, a shared helper with two kernel owners,
three-dimensional workgroup identity, dynamic private allocation, subword
overlap, and agent-scope acquire-release publication. The target-specific
behavioral tier now adds B96/B128, group-FLAT, forced spilling, atomic handoff,
repeated dispatch, matrix instructions, and clustered transfer. Older fixtures
that inspect the current report ABI, patch layout, register plan, or generated
code remain useful implementation tests, but do not substitute for these
behavior-first contracts that can survive the production replacement.

### Coverage gaps

| Area | Current checked-in evidence | Residual gap |
| --- | --- | --- |
| Synchronization faults | Every scenario is shape-matched and paired; reductions, helpers, dynamic stacks, pipelines, routing, batching, cache sequence, and instruction scope all change one real property. | A dedicated wrong-LDS-address pair remains a useful refinement. |
| Stream-K publication | Fetch-add last-arriver and tree atomic-OR pairs cover partial publication and completion. The CDNA two-tile pair adds 80 distinct static LDS sites and five phase barriers, with only the central publication edge removed by the incorrect member. | `ds_bpermute` broadcast and a closer relaxed-poll/final-RMW tree remain target-specific refinements. |
| Multi-stage pipelines | Double buffering, storage reuse, wide LDS, lane exchange, two-stage softmax, CDNA MFMA, RDNA WMMA, FP8 staging, repeated dispatch, and a gfx950-specific 160-KiB/32-stage/eight-wave publication pipeline are covered. | A separate FP16 production reduction remains pressure work. |
| Selection and reduction | Segmented value/index top-k, tails, two-stage softmax, and the MoE router cover multi-phase selection/reduction shapes. | Signed/unsigned D16-high variants and a sort-network-specific pair remain optional refinements. |
| Atomic diversity | Arrival add, tree OR, release CAS plus language-level acquire load, language-level release store plus acquire CAS, LDS histogram collisions, global scatter collisions, MoE bins, cache-qualified VGLOBAL publication, and gfx12 instruction-scoped atomics are paired. | BF16/FP32 atomic payload variants remain absent. |
| LDS and data movement | The suite exercises subwords, B32/B96/B128, group-FLAT wide traffic, B128-to-AccVGPR, gfx1250 high-bank SRC0 addresses beside adjacent split barriers and descriptor waits, target-native load/store spellings, and dual-address instructions whose logical ranges outnumber static sites. | Strided/transpose, D16-high, `ds_swizzle`, and `ds_bpermute` remain mostly in older implementation fixtures. |
| TDM and clusters | gfx1250 now uses real extended clustered-dispatch packets for two-CTA cluster barriers, two-cluster identity/isolation, `cluster_load_async_to_lds_b32` plus `s_wait_asynccnt`, and multicast. | Store-from-LDS, wider tensor fragments, scale-WMMA, more than two CTAs per cluster, and a distinct remote cluster-memory opcode remain unabstracted. |
| Resource and control pressure | Dynamic stack, shared helpers, MFMA/WMMA live state, CDNA AccVGPR destinations, full ordinary-VGPR-bank pressure, gfx1250 high-bank address capture with a spill-backed low-bank window and adjacent double-barrier tail, B96/B128 aliasing, a dense-routed B16 access with live SCC, native VGLOBAL forms, descriptor growth, repeated dispatch, and spill-backed scalar preservation under empty and nonempty `EXEC` are exercised. The tranche caught AccVGPR-boundary, dynamic-frame spill, dense-dispatch SCC, high-bank capture, and empty-wave scalar-restore defects. | Combined worst-case forms and production-sized placement/relay limits remain E2E or focused implementation-test responsibilities. |
| Object and dispatch shape | Shared helpers have multiple kernel owners; softmax uses multiple stages and a global intermediate; continuous batching repeats changing dispatches from reset state; the heterogeneous-object pair executes three independent kernels beside an unexecuted pressure owner and guards cross-kernel diagnostic attribution. | Multiple loaded objects, unload/reload, multi-stream module launch, and graph replay remain lifecycle envelopes rather than covered semantic categories. |
| Scale | The 1,893-row matrix crosses the portable and target-specific pairs plus focused E2E-derived contracts, five simulator targets, physical CDNA4, every engine, a heterogeneous framework-shaped object, large-LDS/deep-pipeline, large-generated-text, access-heavy relay, and module-load lifecycle cases while staying checked in and bounded. | Maximum-capacity and production-sized heterogeneous objects remain E2E responsibilities; add another medium object pair only if a distinct concrete failure mode justifies it. |

## Aorta workload expansion

The `ROCm/aorta` repository adds a user-facing workload catalog and a typed
RocJITsu sanitizer runner. It should serve two different purposes here:

1. Retain selected Aorta workloads as external E2E validation. These tests keep
   the real PyTorch, hipBLASLt, RCCL, stream, graph, and code-object lifecycle;
   they do not need to fit in the RocJITsu source tree.
2. Profile those workloads, inspect their important final code objects, and
   reduce recurring synchronization idioms into small checked-in device pairs.
   The reductions, rather than the large Aorta environments, run across the
   full architecture and engine matrix.

An Aorta workload is not automatically a useful ConSan test. It must have an
exact or otherwise strong correctness oracle, prove that the intended kernels
were instrumented, fail on incomplete coverage, and exercise behavior within
ConSan's stated scope. Performance success, finite output, or process exit zero
alone is insufficient.

### Retained E2E candidates

| Priority | Aorta area | Coverage it adds | Adoption work and limits |
| --- | --- | --- | --- |
| P0 | `llm_determinism` | A transformer forward, loss, backward, FSDP2, RCCL, dense or top-1 MoE, and bit-exact replay checks at every block boundary and for logits, loss, gradients, and parameters. This is a particularly strong clean-run oracle for detecting instrumentation-induced nondeterminism and localizing it. | Add a small dense and a small MoE ConSan E2E row on physical `gfx950` first. Preserve fresh-process isolation and fail on any replay difference. The existing singleton emulated recipe is useful for replayability but is not a multi-rank RCCL result. |
| P0 | `race` FSDP smoke | Real transformer forward/backward, all-gather, reduce-scatter, rank-filled communication oracles, and per-layer checksums under a multi-GPU stream schedule. | Retain as physical multi-GPU E2E coverage where hardware is available. ConSan may validate kernels loaded by the workload; it must not claim to diagnose host stream-ordering, transport, or inter-kernel races that are outside its model. Require `layers_verified > 0`, zero checksum mismatches, and complete sanitizer coverage. |
| P1 | `training` | DDP/FSDP training, dense and MoE transformers, cross-entropy backward, gradient reduction, and AdamW updates. This adds backward and optimizer-generated kernels that the current inference-heavy corpus largely misses. | Current recipes check only NaN/Inf. Before using them as clean ConSan gates, add a deterministic reference, replay checksum, or exact per-layer/gradient/parameter checksums. One-rank recipes are lifecycle smokes, not distributed coverage. |
| P1 | offline and continuous `inference` | Repeated prefill/decode, changing active batch size, ragged tails, dense or MoE blocks, and a stable logits checksum. | Retain small checksum-gated cases. The implemented KV cache is simulated rather than paged, so do not count it as paged-KV coverage. It complements rather than replaces the existing Qwen E2E row. |
| P1 | existing Aorta sanitizer recipes | An independent consumer of RocJITsu's public artifacts, report schema, exact code-object identity, fail-closed policy, and module-launch path. The clean/racy LDS recipes are an external guardrail; dispatched LDS reduction and a real hipBLASLt Tensile GEMM exercise more realistic object loading. | They are currently `gfx950`-only. Clean/racy largely duplicate the foundational device pair. LDS dispatch currently fails with zero recorded events, the GEMM times out during inventory/report allocation, and the no-LDS tiny object intentionally fails strict record requirements; preserve these as visible red/informational states rather than treating them as coverage. |
| P2 | selected hardware-queue workloads | MoE expert streams, comm/compute overlap, FSDP+TP, activation checkpoint recomputation, gradient accumulation, speculative decode, continuous batching, independent graph subgraphs, heterogeneous kernels, and compiled multi-region execution provide valuable concurrent-dispatch and lifecycle stress. | Nearly all currently inherit `BaseWorkload.validate_correctness`, which unconditionally succeeds; the two overrides mainly check finiteness and shape. Promote only individual workloads for which we add an exact reference/checksum and prove the intended path ran. In particular, require `torch.compile` success instead of swallowing fallback before counting compiled-kernel coverage. Performance-only transfer, queue, and tiny-dispatch stress does not by itself expand ConSan's semantic coverage. |
| P2 | HRX static, module, and HIP-graph launch probes | Exact output checks across static registration, `hipModuleLaunchKernel`, graph construction, graph parameter update, graph-exec parameter update, and replay. | Use as launch-path integration coverage after wrapping a meaningful ConSan pair. The current add-100 kernels test argument plumbing but contain no ConSan-relevant synchronization, so they are not standalone semantic coverage. |

Aorta's sanitizer frontend can already select stable kernel identities and run
Waitcheck or whole-code-object Record/Replay, but automatic profiling and
top-K ConSan execution are not complete. RocJITsu currently has no supported
kernel allowlist. Until scoped instrumentation exists, a requested top-K run
must remain `not_checked/worklist_scope_unsupported`; never silently instrument
an entire framework process. The intended E2E flow is:

1. Run the workload without instrumentation and collect both top-time and
   top-dispatch kernels through Magpie, TraceLens, or an equivalent dispatch
   trace.
2. Resolve exact code-object hashes, image indices, entry offsets, target, and
   kernel names. Add synchronization-bearing kernels that a time-only ranking
   would miss.
3. Inspect the final ISA and group kernels by semantic idiom, not by mangled
   name. Record which Aorta recipe, target, and identity motivated each device
   reduction.
4. Rerun the application with a future RocJITsu allowlist, preserving real
   arguments, streams, collectives, fused/JIT kernels, and framework state.
5. Require exact workload output, no clean-run diagnostic, complete selected
   coverage, and no report overflow or timeout.

The existing Aorta emulation recipes use RocJitsu through Mirage for an
MI350X-shaped target. If they are expanded, RocJitsu remains the emulator for
every target; do not add an FFM path. Large E2E rows need not run under every
emulated architecture, but every device reduction derived from them must run
under RocJitsu on `gfx942`, `gfx950`, `gfx1250`, `gfx1100`, and `gfx1201`, plus
the physical `gfx950` row.

### Device tests to distill from Aorta

Every item below is a correct/incorrect pair under the behavioral contract,
not a checked-in copy of a framework-generated object.

| Aorta idiom | Proposed checked-in pair | Required oracle and diagnostic |
| --- | --- | --- |
| Transformer backward and optimizer | **Implemented:** a small mixed-precision gradient reduction followed by an AdamW-like vector update. The correct variant publishes reduced gradients before consumption; the incorrect variant removes that edge while keeping the same backward/update instruction and resource shape. It includes FP32 accumulation and adjacent FP16 gradient traffic. | Exact gradients and updated parameters in the correct run, exact independent control data in both runs, no clean diagnostic, and the declared publication diagnostic in the incorrect run. |
| Top-1 MoE routing | **Implemented:** a bounded router with argmax, per-expert bin counts, prefix offsets, indexed gather/scatter, an empty expert, uneven token tails, and a small GLU-like expert transform. The incorrect variant breaks publication of staged tokens without changing routing decisions. | Exact token-to-expert assignments, offsets, scattered tokens, gathered outputs, and untouched canaries; the incorrect run requires the corresponding race/order diagnostic on every target. |
| Dynamic continuous batching | **Implemented:** a repeated prefill/decode-shaped kernel sequence whose active-row count changes each launch, including partial waves and masked inactive rows. The pair uses a uniform staged reduction with exactly one required edge removed. | Exact output for every active row, unchanged inactive rows/canaries, stable repeated-run checksum, and the expected diagnostic only for the broken variant. This covers tails and lifecycle without pretending Aorta's simulated cache is a paged KV cache. |
| FSDP/DDP and RCCL kernels | First inspect the synchronization-bearing RCCL and generated reduction kernels selected from Aorta. Reduce recurring intra-kernel idioms such as vectorized chunk copy/reduce, local staging, atomic flag/counter publication, polling, and acquire/release fences. The paired defect changes one order, scope, or local barrier edge. | Rank-filled or analytically generated chunk values, canaries around tails, and the matching semantic diagnostic. Do not manufacture a single-kernel "RCCL test" or claim that it covers network transport, host stream dependencies, or unsupported global-memory races. |
| Activation checkpoint and deterministic replay | Apply a two-run state-reset envelope to the backward and pipeline pairs: execute the same inputs twice after restoring parameters and scratch state. The incorrect member retains the same envelope but contains its one declared synchronization fault. | Bit-exact results and no stale diagnostic state across clean generations; the incorrect generation must diagnose every intended run while its independent control remains exact. This tests report/generation reset separately from numerical determinism. |
| Multi-stream, multi-object framework lifecycle | Load two or more small heterogeneous objects, dispatch clean members of a pair concurrently on several streams, unload/reload one object, and repeat. The adjacent incorrect case substitutes only the racy member of one object. | Exact per-stream outputs and object ownership, no cross-generation or cross-object attribution, no clean diagnostic, and a diagnostic attributed to the substituted kernel only. ConSan is not asked to infer missing host-side stream dependencies. |
| HIP module and graph launch | Launch the same meaningful pair through direct module launch and repeated HIP graph execution/parameter update. Keep graph dependencies correct; put the semantic fault inside the incorrect kernel. | Identical output and diagnostic contracts through ordinary, module, and graph paths. A deliberately missing graph edge is out of scope until ConSan explicitly models inter-kernel ordering. |
| Heterogeneous generated objects | **Implemented for one object:** three small attention/optimizer/MoE kernels execute beside a shared helper and an unexecuted high-pressure multi-phase kernel. The incorrect member changes only the attention publication edge. A future multi-object lifecycle extension remains separate. | Exact results and control evidence for every dispatched kernel, no false attribution to independent or unexecuted entries, no clean diagnostic, and the declared diagnostic only for the broken publication. |

The first Aorta-derived tranche--backward/optimizer, MoE routing, continuous
batching with repeated reset-state dispatch, and the single heterogeneous
object--is complete across the whole matrix. The next justified Aorta work is a
selected RCCL-kernel ISA survey, then multi-object and module/graph lifecycle
envelopes around existing pairs.
An RCCL-inspired pair must reflect an actual selected kernel and only behavior
ConSan can observe. Graph and high-stream-count cases remain integration
envelopes around semantic pairs, not new race categories in their own right.

Instruction-specific coverage may use small target-specific inline-assembly
kernels when source-level HIP cannot guarantee the required emitted form. Its
contract must still be behavioral: the correct variant checks an exact result
and no diagnostic, while the adjacent incorrect variant checks the semantic
diagnostic and an independent control result. Merely matching a mnemonic or
the prototype's lowering is not a sufficient oracle.

## Engine contract

Each pair must run through SuperCollider, Record/Replay, Sampled, and Inline
Shadow on every applicable architecture.

- Correct variants require exact output, successful instrumentation, complete
  applicable coverage, no overflow, and no diagnostic.
- Record/Replay incorrect variants require the expected replay conflict and
  its semantic attribution.
- Sampled and Inline Shadow incorrect variants require the expected diagnostic
  category and must reject incomplete or overflowed evidence.
- SuperCollider is explicitly classified as a mutation-only engine without a
  happens-before diagnostic model. Its incorrect rows require successful
  mutation and execution and carry a `mutation-only` label; they are not
  silently counted as diagnostic detections.

Avoid forcing every engine to use identical diagnostic text. The shared
contract is the diagnosed semantic violation; engine-specific evidence may
differ.

## Matrix and registration work

The registration is a declarative table of scenario pairs. Each row declares:

- correct target and test filter;
- incorrect target and test filter;
- applicable targets and engines;
- expected semantic diagnostic or engine-specific negative outcome;
- required access, barrier, atomic, and fence coverage classes;
- clean-output and independent-control oracles; and
- the end-to-end workload property that motivated the reduction.

CTest names make the pairing visible, for example `...Reduction.Correct` and
`...Reduction.Incorrect`, and labels allow a single scenario pair to be
selected across all applicable architectures. One all-engine pair/backend
application is ten rows: correct and incorrect under the baseline and four
ConSan flavors.
The 22 common pairs contribute 1,320 rows across the five RocJitsu targets plus
physical `gfx950`. The non-rectangular target and engine extensions contribute
568 rows, including 36 baseline/Inline/Sampled rows in the generalized
empty-EXEC scalar-spill pair and 24 baseline/Sampled rows in the dual-address
budget pair.
The physical module-load reduction contributes four rows and the physical
health check contributes one, for 1,893 total.

The per-configuration arithmetic is:

| Configuration | Pairs | Rows |
| --- | ---: | ---: |
| RocJitsu `gfx942` | 36 | 338; the dense-SCC pair omits unsupported Record/Replay rows and the production-threshold pair is SuperCollider-only |
| RocJitsu `gfx950` | 37 | 348; the dense-SCC pair omits unsupported Record/Replay rows and the production-threshold pair is SuperCollider-only |
| Physical `gfx950` | 38 | 353, including four module-load rows and health |
| RocJitsu `gfx1100` | 27 | 260 |
| RocJitsu `gfx1201` | 29 | 280 |
| RocJitsu `gfx1250` | 33 | 314 |

Run every pair using RocJitsu on CDNA3 (`gfx942`), CDNA4 (`gfx950`), CDNA5
(`gfx1250`), RDNA3 (`gfx1100`), and RDNA4 (`gfx1201`). Do not introduce an FFM
path. Run the identical CDNA4 contract on the physical `gfx950` as well.

### Suite latency budget

On the current wide reference host, the single CTest invocation covering the
entire device matrix at `-j64` should ideally take **5--20 minutes of wall-clock
time**. This is a coverage-and-cost heuristic, not an objective to make tests
artificially slower or to fill a time quota.

- Below five minutes, pause and ask whether the suite is missing enough
  meaningful workload-derived scenarios, architectural idioms, or lifecycle
  coverage to leave substantial regression risk. More tests are warranted only
  when they add real coverage.
- Above twenty minutes, review which tests can be dropped, consolidated, or
  sampled less often without materially reducing behavioral or architecture
  coverage.

The current one-command 1,893-test matrix passes in **120.12 seconds** on this
host. Its CTest rows sum to **2,389.66 seconds of process duration**, while the
command reports 729.81 seconds user and 887.16 seconds system CPU. This result
is below the five-minute lower review threshold, so the next coverage pass must
ask whether meaningful workload-derived, architecture-specific, or lifecycle
contracts are still missing; it is not a reason to add artificial work.
CTest's summed duration remains a concurrency-insensitive test-capacity
indicator, but it is not CPU time because individual processes can also wait
on the simulator, runtime, or physical GPU.

This measurement supersedes the earlier 67.91-second result, which allowed
all physical gfx950 processes to compete for one GPU and eventually reproduced
a GPU memory fault. All 293 physical rows now share the target-scoped
`consan_physical_gfx950` CTest resource lock. The five simulator targets still
run concurrently with each other and with the one active physical row; a
different physical target would use its own lock rather than creating a global
bottleneck. The honest wall time is inside the 5--20-minute target.

## Issues exposed and retained as regressions

The current matrix has no red cells. Building it exposed defects in the
prototype, runtime, and simulator, all fixed without weakening the behavioral
contracts:

- physical gfx950 CTest oversubscription from missing per-test resource locks;
- `gfx942` and `gfx950` Record/Replay multidimensional workgroup identity;
- `gfx942` and `gfx950` Sampled reduction snapshot formation;
- Sampled conflating a static patch-count limit with the banked report-slot
  capacity of dual-address LDS instructions;
- Sampled comparing overlapping records across different dispatches,
  cluster-workgroup identities, or disjoint kernel-owner scopes inside one
  heterogeneous code object;
- Sampled host-report teardown that scanned empty allocated capacity and
  reconstructed deferred releases in capacity-times-visible work;
- Record/Replay host-report teardown that made several sparse and full passes
  directly over large fine-grained HSA allocations instead of one cacheable
  process-local snapshot;
- Record/Replay auto-report allocation preferring fine-grained storage even
  when coarse storage enables a substantially faster HSA bulk snapshot;
- the default patched-image growth envelope rejecting a qualified PyTorch
  top-k Record/Replay transform by less than one MiB;
- Record/Replay validation conflating a report's full diagnostic capacity with
  the compact replay scratch capacity for barrier-only readers;
- gfx1250 Record/Replay reading an encoded high-bank LDS address only after
  selecting low scratch, then overwriting a live spill victim while correcting
  the capture order;
- `gfx1100` Inline Shadow atomic-arrival evidence;
- `gfx1250` Inline Shadow 3D-workgroup handling around the RocJitsu `HW_ID1`
  topology limitation;
- physical `gfx950` Record/Replay, Sampled, and Inline Shadow failures;
- non-returning DS-atomic emulation and Sampled atomic-watchpoint decoding;
- CDNA3/CDNA4 Inline Shadow recovery of a clobbered wide-load address outside
  a reduced spill window;
- CDNA3/CDNA4 Inline Shadow use of a disjoint compact spill window for an LDS
  load whose result clobbers its address;
- CDNA Inline Shadow owner/epoch backup selection at an ordinary-VGPR/AccVGPR
  boundary; and
- CDNA SuperCollider native-LDS instrumentation under an active dynamic
  private frame;
- CDNA4 Record/Replay barrier routing below the compact-count threshold when
  early sites cannot reach appended relays; and
- CDNA4 Record/Replay restoration of guest SCC between a dense dispatch-key
  comparison and a directly reached B16 access body.
- CDNA descriptor SGPR growth borrowing the six-register physical
  VCC/XNACK/FLAT_SCRATCH tail for a production-scale SuperCollider return-PC
  pair.

These implementation findings remain covered by device or focused unit
regressions. Any future red cell receives the same treatment: fix the owning
ConSan, shared runtime, target, or RocJitsu layer and leave the reproducer
enabled.

## Target-specific extensions

The rectangular common matrix proves portable behavior through five different
target backends; it does not prove target-exclusive features. The same HIP
source reaches useful target-native paths--for example CDNA3/CDNA4
`ds_read_b128` and singleton barriers versus RDNA4/CDNA5 `ds_load_b128`, split
barriers, and separated wait counters--and this has already found defects that
were family-specific. Nevertheless, a portable source that never requests a
cluster, MFMA/WMMA state, group-FLAT address, or target-specific order/scope
form cannot validate that capability merely by being compiled for the target.

Keep three evidence levels distinct:

1. **Common device contract:** exact correct/incorrect behavior runs through
   target-native decoding, planning, instrumentation, execution, and reporting.
2. **Host or E2E evidence:** focused transformation tests or an external
   workload prove additional forms, but do not provide the cheap adjacent pair.
3. **Target-specific device contract:** a checked-in correct/incorrect pair
   deliberately executes the special form and its behavioral oracle. This is
   now an implemented, non-rectangular tier whose residual gaps are tracked
   below.

Target-specific pairs are intentionally non-rectangular. A pair applicable to
one simulated target adds ten CTest rows: two outcomes times the baseline and
four engines. A `gfx950` pair that runs in both RocJitsu and on the physical GPU
adds twenty. Cross-generalize a pair wherever the form genuinely applies, but
do not create a nominal version for an architecture that lacks that form.

### Existing-test transport audit

The 2026-08-21 capability pass reviewed every family/target-local pair rather
than treating its original filename as its applicability boundary. It added 34
rows: AccVGPR B128, large-text relay geometry, and the dense SuperCollider
threshold moved from gfx950-only to both CDNA3/CDNA4 backends, while wide
group-FLAT moved from four targets to all five by adding gfx1250. All 1,306
simulator rows pass after the transport.

| Contract family | Applies and runs | Capability-based disposition elsewhere |
| --- | --- | --- |
| Twenty-two common pairs | CDNA3/4/5 and RDNA3/4, plus physical CDNA4 | None; these are the rectangular semantic baseline. |
| Wide group-FLAT | All five targets, plus physical CDNA4 | None. The gfx1250 omission was a registration gap, not an ISA limitation. |
| CDNA MFMA/full-bank/two-tile/B96/dense-route/B16/clobbered-load/AccVGPR/large-text pairs | CDNA3 and CDNA4, plus physical CDNA4 where available | RDNA uses distinct wave32 register and LDS forms; its WMMA/B96 pairs carry the transported pipeline and aliasing ideas. CDNA5 uses its gfx12-style LDS/register model and needs a native scaled-WMMA pair rather than a nominal CDNA encoding. |
| Dense SuperCollider threshold | CDNA3 and CDNA4, plus physical CDNA4 | The current 1,024-site cliff is specific to the CDNA dense dispatcher. Other targets retain smaller all-engine placement contracts; add a production-threshold pair only when their own router has a distinct evidenced cliff. |
| 160-KiB large-LDS pipeline | CDNA4 simulation and physical CDNA4 | Typed not-applicable where the selected target's modeled group-segment capacity is below 160 KiB; smaller common pipelines already transport the synchronization idea. |
| Native B96 aliasing | CDNA3/4 through the CDNA pair; RDNA4/CDNA5 through the gfx12-form pair | The claimed gfx1100 subset has no native B96 ordinary form. |
| Instruction-encoded atomic scope | RDNA4 and CDNA5 | CDNA3/4 and RDNA3 express the corresponding ordering through different cache-qualified sequences covered by `VglobalCachePublication`; duplicating the gfx12 encoding is not applicable. |
| WMMA/FP8 matrix staging | RDNA3/4 WMMA; RDNA4 FP8 | CDNA3/4 use the MFMA pair. The installed compiler correctly rejects the RDNA4 `wmma-128b-insts` builtin for gfx1250; a CDNA5-native scaled-WMMA device pair remains a real gap, not an expected-failure port. |
| gfx11 scalar-base VGLOBAL | RDNA3 | Other targets' vector/scalar address materialization is already exercised by their native common VGLOBAL pair and host matrix; this fixture's missing-GL0 negative is specifically the gfx11 cache sequence. |
| gfx1250 prefix, cluster, TDM, and multicast | CDNA5 | Typed not-applicable on targets without clustered dispatch and CDNA5 tensor-data movement. The portable top-k and publication contracts carry the cross-cutting ideas. |
| Physical module-load reduction | Physical CDNA4 | The semantic pair is portable, but module/graph and unload/reload envelopes are still a lifecycle coverage gap on simulated targets; this is not an ISA exclusion. |

### CDNA3 (`gfx942`)

**Implemented evidence:** 36 pair names produce 338 green RocJitsu rows. In addition
to the 22 common pairs, `CdnaMfmaPipeline` keeps MFMA/AccVGPR state live across
an LDS publication edge; `CdnaFullBankStreamk` combines all 256 ordinary VGPRs,
a dynamic private frame, shared helpers, cache-qualified VGLOBAL fetch-add,
and last-arriver consumption; `CdnaB96Boundary` pairs native 12-byte tuple
publication with an address/destination alias from PyTorch's rocPRIM kernels;
`CdnaStreamkTwoTile` executes 80 distinct LDS access sites and five barriers
across five publication phases; and `GroupFlatWide` executes native wide
group-FLAT traffic. `CdnaAccvgprB128` now proves a native B128 load directly
into AccVGPRs while independent accumulators and VCC remain live, and
`CdnaWave64LargeText` transports the generated-object relay geometry from
CDNA4. `CdnaDenseSccRoutes` covers far access/barrier routing under
extreme scalar pressure in the three engines that admit that shape, while
`CdnaB16SccRoute` covers all four engines with a D128-derived 16-bit store
between an SCC producer and consumer. `CdnaSuperColliderDenseThreshold`
crosses the same 1,024-stranded-site production threshold on CDNA3 as on
CDNA4. The common VGLOBAL pair supplies the CDNA cache/wait publication
contract, and the Sampled dual-address pair executes native two-range LDS forms
without charging their banked report slots against the static site limit.
These reductions came from the hip-moi attention, pressure, and Stream-K rows
in `VALIDATION.md`.

**Defects now guarded:** segmented top-k exposed illegal entry-state backup at
the ordinary-VGPR/AccVGPR boundary, and the dynamic-stack pair exposed missing
SuperCollider native-LDS spill support. Focused host regressions accompany the
green device rows.

**Residual gaps:** transpose/strided LDS forms remain host/ISA evidence; no
small pair yet combines MFMA, the full ordinary bank, and dynamic stack in one kernel
(those states compete for the unified register file). Floating atomic payload
variants and a production sort network are also absent. There is no
physical `gfx942` in this workspace, so physical qualification remains an
external availability gap.

### CDNA4 (`gfx950`)

**Implemented evidence:** 37 pair names contribute 348 green RocJitsu rows. Physical
gfx950 repeats the applicable contracts, adds four module-load rows and a
health check, and contributes 353 rows in total.
Alongside the CDNA3 MFMA,
full-bank Stream-K, B96, group-FLAT, and common VGLOBAL contracts, the
CDNA4-only `Gfx950LargeLdsPipeline`
adds a 160-KiB group segment, eight waves, 32 publication stages, wide DS
stores, scalar reads, and repeated dispatch. Exact outputs make clobbering
register state or LDS publication observable. The shared
`CdnaAccvgprB128` and `CdnaWave64LargeText` pairs add direct accumulator
destinations and a
large generated-text shape with an early LDS site beyond direct branch range,
plus a nearby paired publication contract. It executes the relocated wave64
guest instructions without asserting the current placement mechanism. This
abstracts Sharktank, HipKittens, PyTorch generated kernels, and the physical
hip-moi attention/pressure/Stream-K evidence in `VALIDATION.md`.

The shared CDNA `CdnaStreamkTwoTile` pair additionally preserves 80 static LDS
access sites and five static barriers while the incorrect member uniformly
skips only the central publication barrier. It directly guards the relay-space
competition reduced from the physical two-tile Tensile Stream-K E2E workload.
The two dense-SCC pairs add the corresponding pressure and B16 control-flow
shapes on both backends; the latter specifically guards the Record/Replay
dispatcher bug reduced from hip-moi D128-pressure.
`CdnaSuperColliderDenseThreshold` separately crosses SuperCollider's
production 1,024-stranded-site threshold with 1,040 static LDS writes and a
long scalar tail distilled from PyTorch radix sort. Its correct and incorrect
members run under baseline and SuperCollider on both CDNA simulators and
physical gfx950. They pin exact results, diagnostic behavior, and the CDNA descriptor
growth that keeps dense return-PC scratch below the physical special-register
tail without making the generalized all-engine fixture artificially large.
The separate `RepeatedDispatchIdentity` pair remains the portable behavioral
contract for Record/Replay identity across compact repeated launches. The
physical `torch.sort` no-evidence result was a validation-cadence issue rather
than new device semantics: a gfx950-only stride-1 policy now qualifies the
four-row E2E schedule, while unit tests prove other targets retain their
production defaults. No prototype-specific device fixture is added for that
harness policy.

**Defects now guarded:** the common and target pairs reproduce the reduced
wide-load spill-window corruption, the AccVGPR-boundary entry-state bug, and
dynamic-frame SuperCollider spill failure. The two-tile pair and its focused
host regression guard the Record/Replay compact-barrier allocator from
consuming seven-word access-pass relay reservations as eight-word generic
islands. The large-text pair and focused host
tests additionally guard wave64 relay scratch preservation and use of skipped
kernels as relay donors when non-text data makes an image large. They pass in
both backends rather than recording the earlier resource rejection as expected.

**Residual gaps:** production-sized Sharktank/CLIP placement and completeness
remain E2E obligations; transpose/strided LDS, floating atomic payloads, and one kernel combining
full-bank pressure with live MFMA are not behavioral pairs. The checked-in suite also does not replace multi-object,
module/graph, or unload/reload lifecycle validation.

### RDNA3 (`gfx1100`)

**Implemented evidence:** 27 pairs produce 260 green RocJitsu rows. Beyond the
22 common pairs, `GroupFlatWide` supplies the shared group-FLAT byte/numerical
oracle, `RdnaWmmaPipeline` cross-pollinates the green gfx12 attention idiom into
gfx11 WMMA plus LDS staging, `Gfx11ScalarVglobal` complements the vector-only
common VGLOBAL form with an SGPR-base acquire and a missing-GL0-invalidate
negative. The common repeated-dispatch pair provides the loaded-image/report
generation contract without claiming that ConSan models inter-kernel races.

**Residual gaps:** there is still no gfx1100 empirical workload survey
comparable to the gfx1201 study, so the relative production importance of the
covered idioms is inferred. A single pair does not yet combine WMMA, dynamic
private state, shared-helper ownership, and maximum placement pressure. D16-high,
swizzle/permute, floating atomics, multi-object
unload/reload, and graph or module launch remain untested. B96, clustered dispatch, and gfx12
instruction-encoded scope are not part of the claimed gfx11 subset and are not
gaps.

### RDNA4 (`gfx1201`)

**Implemented evidence:** 29 pairs produce 280 green RocJitsu rows. The target
extensions deliberately execute group-FLAT; native aliasing B96 loads/stores
with canaries; a returning VGLOBAL atomic whose incorrect member changes only
the instruction-encoded scope; WMMA interleaved with a split-barrier LDS
pipeline; and a production-shaped FP8 staged matmul. These are reductions of
the WMMA, production FP8, PyTorch selection, and generated-kernel evidence in
`VALIDATION.md`, not mnemonic-only checks.

**Residual gaps:** physical gfx1201 evidence remains in the external E2E
campaign rather than this machine's device matrix. A separate production-shaped
FP16 pair, large `torch.mode`/llama placement pressure, refreshed top-k fault
qualification and swizzle/permute forms remain absent. The common
heterogeneous-object pair now covers cross-kernel attribution and an unexecuted
pressure owner; production-size placement remains an E2E obligation.

### CDNA5 (`gfx1250`) summary

**Implemented evidence:** 33 pairs produce 314 green RocJitsu rows. In addition
to the 22 common pairs, gfx1250 now runs the wide group-FLAT contract, shares
native B96 and instruction-encoded atomic scope with gfx1201, and
`Gfx1250TopKPrefix` covers the target's prefix and
last-arriver idiom. `Gfx1250HighBankLdsAddress` covers SRC0-bank address
capture and its correct/missing-barrier behavioral contract; focused host
coverage adds composite-bank and spill-backed resource pressure. Four
exclusive contracts submit
`hsa_amd_ext_kernel_dispatch_packet_t` directly with real cluster dimensions:
two-CTA cluster synchronization, two-cluster identity/isolation,
`cluster_load_async_to_lds_b32` plus `s_wait_asynccnt`, and multicast to both
CTAs. Their correct members verify exact data and control; their incorrect
members remove only the relevant cluster publication edge. This replaces the
previous host-only cluster evidence with actual checked-in device execution
and distills the clustered/TDM E2E work described in `VALIDATION.md`.

**Residual gaps:** store-from-LDS, wider/32-byte tensor fragments, scale-WMMA,
clusters larger than two CTAs, and a distinct remote cluster-memory operation
remain uncovered. The multicast pair validates real clustered transfer and
communication but must not be described as proof of an opcode the compiler did
not emit. There is no physical gfx1250 in this workspace, so all such device
evidence is RocJitsu-emulated.

Generalize each extension to every architecture where its semantics apply.
Every correct member must retain an exact oracle and forbid diagnostics; every
incorrect member must change only the relevant synchronization, ordering, or
overlap property and require the applicable semantic result. Do not add a
device test whose only oracle is recognition of the current prototype's
generated code.

## Prioritized implementation sequence

The common workload sequence is complete through the first Aorta-derived
tranche. Retain the same pair across targets and specialize only the
architecture-dependent operation:

1. **Complete:** the seven foundational pairs and their semantic engine
   contracts.
2. **Complete:** Stream-K fetch-add publication and tree atomic-OR publication.
3. **Complete:** double-buffered LDS staging plus target-specific CDNA MFMA,
   RDNA WMMA, CDNA AccVGPR, B96, and FP8 staged pipelines.
4. **Complete:** histogram-style LDS atomics, collision-heavy global scatter
   atomics, and language-level release-store/acquire-CAS publication.
5. **Complete:** segmented value/index top-k with tails and two-stage softmax
   with a global intermediate.
6. **Complete:** backward/optimizer, MoE routing, and continuous-batching pairs
   distilled from Aorta.
7. **Complete:** target-specific MFMA/WMMA, AccVGPR, full-bank pressure,
   group-FLAT, VGLOBAL/cache sequence, B96, atomic scope, gfx11 identity, and
   gfx1250 cluster/TDM/multicast/high-bank/double-barrier pairs. Every implementation
   failure they exposed was fixed and retained as a regression.
8. **Next only when evidence justifies it:** a gfx1100 E2E/ISA survey, CDNA3
   LDS-to-AccVGPR, gfx1250 store-from-LDS/wider fragments/scale-WMMA, a separate
   RDNA4 FP16 reduction, and uncommon transpose/permute or atomic forms.
9. **Remaining lifecycle enhancement:** bounded multi-object, module/graph,
   unload/reload, and report-capacity envelopes around existing semantic pairs.

Do not consider an area complete because one architecture happens to lower a
generic HIP fixture to the desired instruction. Record the applicable semantic
and ISA classes for each pair and verify them on every target where they exist.

## Completion assessment

The current portable and target-specific tiers are fully qualified, but the
Part 1 coverage review remains open because the matrix now completes below the
five-minute lower heuristic. Twenty-two common all-engine pairs plus shared
engine-scoped and family/target pairs cover the
main synchronization, atomic, pipeline, selection/reduction, resource,
cluster/TDM, and repeated-dispatch idioms distilled from `VALIDATION.md` and
Aorta. Every pair runs through the baseline and each engine declared
semantically applicable in RocJitsu; CDNA4 repeats the same coverage on the
physical `gfx950`. The single `ctest -j64 -L consan-device` qualification passes
all 1,540 simulator and 353 physical rows without expected-failure exemptions
in 120.12 seconds wall time and 2,389.66 seconds of summed CTest process time.
The next pass must therefore revisit workload, architecture, and lifecycle
gaps before declaring the tier complete.

This materially shrinks, but cannot eliminate, regression risk. The remaining
gaps are narrower uncommon instruction forms, production-size placement,
missing physical hardware for four targets, and multi-object/module/graph
lifecycle envelopes. Keep those explicit and add them from E2E ISA evidence or
a concrete failure, not from prototype-specific patch expectations or a desire
to make the wall clock longer.

## Completion criteria

- Every common scenario has an adjacent correct and incorrect workload sharing
  the same essential compiler, resource, control-flow, and memory-access shape.
- Every correct workload checks exact results and forbids diagnostics.
- Every incorrect workload requires the declared semantic diagnostic or an
  explicitly justified engine-specific negative outcome, while retaining an
  independent control oracle where possible.
- All common pairs are registered on all five RocJitsu targets and physical
  `gfx950`; target-specific pairs run on every architecture where their form
  genuinely applies.
- Missing instrumentation, incomplete static or dynamic evidence, overflow,
  wrong diagnostics, crashes, hangs, and changed workload results all fail the
  test.
- Every P0 target-specific capability above has a checked-in behavioral pair
  on each applicable target, or a documented typed not-applicable disposition;
  lower-priority residual gaps remain explicit and are not counted as covered.
- The suite covers the workload-derived synchronization, atomic, LDS-form,
  pipeline, resource-pressure, object-shape, and bounded-capacity classes above,
  with a documented E2E owner for each abstraction.
- The suite remains checked in, bounded, free of external workload assets, and
  independent of the implementation being replaced.
- The single whole-matrix CTest invocation passes with no expected-failure
  exemptions, including for issues that predated this device-test work.
