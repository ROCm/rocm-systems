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
| `tests/dbi/consan/device/` | The checked-in behavioral-conformance tier. Forty-five all-engine pairs cover portable workload abstractions on all five targets plus physical gfx950, including RCCL-style reusable partial barriers, Stream-K and Top-K, runtime-indexed LDS aliasing, deterministic state replay, module and graph lifecycle, compiler-generated kernarg preloads, mixed private owners, large-text relay pressure, live-SCC subword traffic, clobbering LDS loads, full-low-bank Stream-K, and adjacent plus stride-64 dual-address B64 publication. Four additional common pairs deliberately scope a pressure contract to one engine plus baseline. Family/target sources cover CDNA MFMA/AccVGPR/direct-to-LDS/transpose, RDNA WMMA/FP8, and gfx1250 clusters/TDM/high-bank LDS, sparse/scaled matrix pipelines, native K=32 FP16 WMMA with 32-byte fragments, 160-KiB LDS, native tensor-descriptor staging, and async transfer in both global-to-LDS and LDS-to-global directions. The gfx1250 cluster host fixture submits a real extended HSA dispatch packet rather than hiding cluster dimensions behind an ordinary HIP launch. | Add one descriptively named source per new semantic scenario. Prefer extending a scenario with another tightly related pair over creating broad clean/racy grab bags. |
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

The portable tier has 45 adjacent correct/incorrect all-engine pairs. Each
common pair runs as baseline plus all four engines on five RocJitsu targets
and physical `gfx950`, for 60 rows per scenario. Four common pressure pairs are
deliberately baseline plus one applicable engine, while family and target
pairs run only where their semantic and ISA forms genuinely apply. The current
registered `consan-device` matrix is 3,285 tests: 2,728 simulator rows and 557
serialized physical-gfx950 rows, including the physical health check. The
latest exact timing is recorded under Suite latency budget below.

| Scenario | Workload-derived contract | Status |
| --- | --- | --- |
| Ordered tile handoff | Cross-wave LDS publication and exact consumer values | Paired and green |
| Reduction | Tree-style LDS reduction with one missing publication edge | Paired and green |
| Shared helper owners | One noinline helper reached from two kernel owners | Paired and green |
| Independent workgroups | Three-dimensional dispatch identity without false LDS aliasing | Paired and green |
| Repeated dispatch identity | Reuse of the same code object across dispatches without stale sanitizer identity | Paired and green |
| Deterministic state replay | The same device allocation is restored byte-for-byte and reused for two mixed-precision forward/backward/update launches; FP16 LDS checkpoints feed exact FP32 activation, gradient, and parameter results, while every input, parameter, output canary, control word, and guard is restored and independently checked. The incorrect member removes only the checkpoint-publication edge in both launches. | Paired and green in all 60 baseline/all-engine rows across five simulator targets and physical gfx950. Record/Replay observes two dispatch identities and diagnoses both replayed fault instances; InlineShadow now budgets code-object-lifetime diagnostic headroom across its dispatch banks, with a focused host regression guarding that capacity relationship. |
| Graph replay | Two captured HIP graphs launch on separate streams for three replay generations; exact per-stream/per-generation results and independent per-thread control remain distinct while the incorrect member removes only the in-kernel LDS publication edge | Paired and green in all 60 baseline/all-engine rows across five simulator targets and physical gfx950 |
| Graph parameter update | A graph node created with an argument array is redirected through the pre-packed `extra` path before instantiation, then the executable node is redirected again through the corresponding exec-update API; untouched sentinel state proves both updates took effect | Paired and green in all 60 baseline/all-engine rows across five simulator targets and physical gfx950 |
| Dynamic private stack | Private-stack state combined with ordered or unordered LDS traffic | Paired and green |
| Kernarg-preload dynamic private stack | The portable dynamic-stack contract compiled with real AMDHSA kernarg preloads; both firmware entries must initialize equivalent state | Paired and green on all five simulator targets and physical gfx950 |
| Mixed private owners | Disconnected fixed-stack and runtime-sized-stack owners retain distinct register/private resource shapes while only the dynamic owner's publication edge changes | Paired and green on all five simulator targets and physical gfx950 |
| Module lifecycle | Four independently compiled module objects remain live on separate streams for three generations while odd and then even subsets are unloaded/reloaded; exact per-object/per-generation control and every nonfault result are checked, and only one designated module selects the racy kernel | Paired and green in all 60 baseline/all-engine/backend rows; the three MOI engines require all eight initial/reloaded objects and all 32 access sites in their final complete verdict, while Record/Replay observes the designated fault in all three generations with four simultaneous report buffers and no overflow |
| Long-range live SCC | A publication access sits beyond 36,000 SCC-neutral scalar instructions while SCC controls the later barrier and exact two-wave tile result | Paired and green on all five simulator targets and physical gfx950 |
| CDNA mixed persistent state | Two disconnected owners combine live MFMA/AccVGPR and full scalar pressure with an empty-AccVGPR, full-VGPR dynamic stack; the pair changes only the latter owner's cross-wave publication edge | Paired and green on gfx942/gfx950 simulation and physical gfx950; Record/Replay proves complete 4/4 access and 2/2 barrier coverage, clean no-conflict replay, and the required incorrect conflict |
| CDNA kernarg-preload private state | The mixed fixed/dynamic-owner contract compiled with a real AMDHSA kernarg preload; ordinary and `+256` firmware entries must initialize the same Sampled state | Paired and green in eight gfx942/gfx950 simulator and four physical-gfx950 baseline/Sampled rows; focused host regressions cover both private-epoch and dynamic-stack owner/epoch entry paths |
| CDNA direct-to-LDS publication | MUBUF direct-global-to-LDS producers use implicit physical-lane/M0 destinations before peer DS readers; gfx950 executes the E2E-derived B32 and B128 forms into disjoint LDS regions, while gfx942 retains the applicable B32 transport. The adjacent members differ only by the publication barrier. | Paired and green in 20 gfx942/gfx950 simulator and 10 physical-gfx950 baseline/all-engine rows. The gfx950 correct member checks all five dwords per lane exactly; disassembly retains both `buffer_load_dword ... lds` and the HipKittens/Triton `buffer_load_dwordx4 ... lds` form followed by one VMEM wait. A host inventory regression accepts documented zero-VDATA B32/B128 forms and rejects B64 or nonzero-VDATA forms. |
| Overlapping subwords | Adjacent non-overlap versus true byte-range overlap | Paired and green |
| Adjacent subword writers | Two waves publish disjoint FP16 halves of one LDS dword and a third wave consumes both; only the consumer publication barrier changes | Paired and green in all 60 baseline/all-engine/backend rows; Sampled retains exact byte ranges so the two writers do not alias while the missing publication edge still diagnoses |
| Indexed LDS alias | Two producer waves use one runtime-computed indexed store site. A unit stride maps every logical gather/scatter slot to a disjoint LDS address; a zeroed stride collapses every slot onto one address without changing the instruction or synchronization shape. | Paired and green in all 60 baseline/all-engine/backend rows. The correct member checks every permuted result exactly; the incorrect member retains exact untouched-slot canaries and every active/inactive control word while Record/Replay, Sampled, and Inline Shadow require the write/write conflict. |
| Atomic arrival | Atomic counter publication followed by shared-data consumption | Paired and green |
| Atomic CAS publication | Release compare-exchange publication followed by a language-level acquire load; the incorrect member acquires a distinct ready state object | Paired and green; exact results and all 60 baseline/engine/backend rows pass |
| Atomic store publication | Language-level release store followed by an acquire compare-exchange; the incorrect member releases a distinct phase object while preserving the same store/CAS shape | Paired and green; exact results and all 60 baseline/engine/backend rows pass |
| Stream-K last arriver | Partial publication, fetch-add arrival, and last-arriver consumption | Paired and green |
| Tree atomic-OR | Bitmask publication, polling, and completion | Paired and green |
| Double-buffered pipeline | Wide LDS stages, storage reuse, lane exchange, a correct redundant adjacent-barrier publication pair, and an incorrect missing stage edge | Paired and green across all five RocJitsu targets and physical gfx950 |
| Multi-wave GEMM tile | A 256-thread workgroup cooperatively publishes a 16-KiB tile, reconverges at one publication edge, and makes every wave consume a peer wave's 16-word stripe; the incorrect member removes only that edge | Paired and green in 50 all-target RocJitsu rows and ten physical-gfx950 rows; distilled from the exact rocBLAS SGEMM Record/Replay campaign |
| Reused LDS GEMM pipeline | The same LDS tile is published and consumed, all readers retire before its storage is overwritten, the replacement stage is republished, and both stages plus their fused result are checked exactly; the incorrect member removes only the reader-retirement barrier | Paired and green in all 60 baseline/all-engine rows; distilled from Qwen's generated final-output matmul |
| Cooperative LDS initialization | Two waves cooperatively initialize LDS and consume peer-initialized slots; the incorrect member removes only the initialization-publication edge | Paired and green in all 60 baseline/engine/backend rows across CDNA3/4/5, RDNA3/4, and physical gfx950 |
| Histogram/scatter | Integer, FP32, and FP64 LDS-bin atomics plus exact integer, relaxed scalar-BF16, and target-native packed-BF16 collision-heavy global scatter updates | Paired and green in all 60 baseline/all-engine/backend rows; CDNA3/4/5 execute native `ds_add_f64` while RDNA3/4 execute the compiler's exact 64-bit LDS compare/exchange loop. The scalar BF16 path deliberately preserves the workload's arithmetic-only relaxed semantics and exercises every target's compiler-selected CAS lowering, while the packed path executes on CDNA3/4/5 and RDNA4 and is typed N/A on RDNA3 |
| Segmented top-k | Value/index tuples, segmented phases, wide traffic, and non-multiple tails | Paired and green |
| Two-stage softmax | Max/sum-style local reduction plus a global intermediate | Paired and green |
| Backward/optimizer | Three producer waves publish ragged FP16 partial gradients; a fourth wave performs an FP32 hierarchical reduction, first-moment update, and half-rounded AdamW-like parameter update | Paired and green in all 60 baseline/all-engine rows across five simulator targets and physical gfx950; exact active results, inactive canaries, and every thread control word are checked |
| MoE routing | Top-1 routing, empty experts, uneven tails, prefix offsets, and indexed gather/scatter | Paired and green |
| Continuous batching | Changing active rows, masked tails, repeated reset-state dispatch, and stable checksums | Paired and green |
| Heterogeneous object | Three executed attention/optimizer/MoE kernels, a shared helper, and an unexecuted high-pressure kernel in one object; only the attention publication edge differs | Paired and green; Sampled's cross-kernel attribution regression is host-guarded |
| VGLOBAL cache publication | Target-native release/acquire atomic publication and required cache sequencing | Paired and green |
| Sampled multi-range budget | Two dual-address LDS sites whose banked report ranges must not consume the site-count patch budget | Paired and green in baseline and Sampled |
| Dual-address B64 publication | Stream-K/Sharktank/Triton-derived adjacent and 4-KiB-separated 64-bit range pairs, using target-native CDNA3/4, gfx11, and gfx12 instruction/wait spellings | Paired and green in all 60 baseline/all-engine rows, including physical gfx950; one publication edge orders four exact 64-bit values per lane |

The target-specific tranche adds CDNA MFMA/AccVGPR liveness, full-bank dynamic
Stream-K, disconnected mixed private/register persistent-state owners,
direct-global-to-LDS publication with implicit physical-lane/M0 addressing, an
80-access/five-barrier two-tile Stream-K relay shape, native
B128-to-AccVGPR, dense far-routed LDS access and barrier sites with live SCC,
a D128-derived B16 store between an SCC-producing compare and SCC-consuming
branch,
a gfx950 160-KiB/32-stage LDS pipeline, a
CDNA3/CDNA4 large generated-text wave64 shape, wide group-FLAT on all five
targets, CDNA and RDNA B96 aliasing
boundaries, instruction-encoded atomic scope, WMMA and FP8 staging, both gfx11
VGLOBAL address forms, repeated multi-stream image
identity, five real clustered gfx1250 dispatch contracts covering two- and
four-CTA cluster barriers, multi-cluster isolation, direct-to-LDS async
load/wait, and multicast, and a gfx1250 high-bank/double-barrier LDS contract distilled from
Tensile. It now also includes a wave32 Top-K prefix contract on gfx1100,
gfx1201, and gfx1250; a native gfx950 `ds_read_b64_tr_b16` publication pair
with the exact 16-lane transpose oracle; a gfx1250 scaled-FP8 WMMA pipeline;
gfx1250's native K=32 FP16 WMMA consuming one 32-byte LDS fragment per lane;
and a PyTorch/Triton-derived tensor-descriptor pipeline with two tensor loads,
one tensor store, three tensor-count waits, and an exact add/canary oracle.
Both matrix results remain exact and live across LDS instrumentation.

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
| Physical-gfx950 HIP matmul, Record/Replay and Sampled | One code object contains fixed-stack kernels with live MFMA accumulator banks and a disconnected full-VGPR dynamic-stack kernel with no allocated accumulators. A code-object-wide persistent representation left the latter owner's sites uninstrumented. The adjacent device members retain both owner shapes and exact independent MFMA, scalar, VGPR, private-stack, and control results; only the dynamic owner's cross-wave publication edge differs. The contract deliberately avoids asserting whether the implementation selects private, VGPR, or another future persistent-state representation. | The CDNA pair contributes 20 green baseline/four-engine simulator rows on gfx942/gfx950 and 10 green physical-gfx950 rows. Record/Replay and Sampled both patch all 4/4 accesses and 2/2 barriers in the reduction: clean members have zero diagnostics, while incorrect members produce the required conflict. Focused host regressions cover access, barrier, ordered-atomic, fence, initially resource-failed-owner recovery, and the canonical native-executable manifest/argv/environment contract. The registered E2E row passes all three exact m128-n128-k128 numerical oracles with complete 739/739 access and 109/109 barrier coverage. Record/Replay measures 2.82× paired slowdown and emits 1,020 diagnostics for the reviewed initial tile-publication removal; Sampled measures 2.06× and emits 24 conflicts for that same exact-one removal, with 392 losslessly claimed windows. Both retain complete surviving coverage, bounded cleanup, exact containment, and healthy pre/post probes. The fault is the same observable publication behavior already owned by `CdnaMfmaPipeline`, `MultiWaveGemmTile`, and `CdnaMixedPersistentState`; their 24-row focused Sampled gate passes across all five RocJitsu targets and physical gfx950, so no duplicate implementation-shaped device fixture is warranted. |
| Physical-gfx950 D128 block attention, Sampled | The workload combines MFMA with a wave-0 K/V producer, cross-wave LDS consumers, exact and sampled Hip-MOI contexts, and disconnected resource owners. The source-matched clean row must cover every supported access and applicable barrier without inventing conflicts; a retained redundant adjacent-barrier mutation checks exact-one containment and qualified-miss bookkeeping. | The current E2E bundle passes the exact oracle with 128/128 access and 117/117 applicable-barrier coverage at 9.41x, and accepts a prospectively reviewed exact-one removal of the first of two consecutive unconditional FastContext barriers as `not_detected/pass`. Two stronger prospectively frozen K-publication detection trials, including a distinct stride-1 run, remain rejected rather than relabeled after physical scheduling masked both diagnosis and oracle manifestation. `OrderedTileHandoff` and `CdnaMfmaPipeline` already own the positive cross-wave publication diagnostic across applicable backends, while `SampledIndependentScalarProofs` retains the workload's two-owner placement shape across all five targets plus physical gfx950. No duplicate qualified-miss fixture is warranted. |
| Physical-gfx950 MFMA attention, Sampled | The exact-context kernel combines MFMA/live accumulator state with repeated score publication through adjacent physical barriers. A reviewed exact-one removal of the first score-publication barrier retains the following semantic edge and therefore checks containment and qualified-miss bookkeeping without changing the race contract. | The current E2E bundle passes its exact oracle with 58/58 access and 12/12 applicable-barrier coverage at 2.57x, then accepts the prospectively frozen `not_detected/pass` mutation with 58/58 access and 11/11 surviving-barrier coverage, bounded memory, cleanup, and healthy pre/post probes. `OrderedTileHandoff` owns the portable publication edge and `CdnaMfmaPipeline` transports it with native MFMA/AccVGPR pressure across CDNA3/CDNA4 and physical gfx950. Their focused 90-row gate passes, so adding a prototype-adjacent-barrier fixture would duplicate behavior rather than expand coverage. |
| Physical-gfx950 tree atomic-OR, Sampled | Three producer subgroups publish MFMA partials through release atomic-OR operations; the final subgroup polls the bitmask, performs a distinct acquire-release RMW, and consumes every producer's LDS partials. Weakening the producer-only release must be distinguished from the final acquire-release edge. | The source-matched clean row passes the exact MFMA-partials oracle with complete 48/48 access, 4/4 barrier, and 3/3 atomic coverage at 15.89x. A prospectively selected exact-one producer-release weakening at audited stride one preserves the numerical oracle and emits 12 Sampled conflicts with complete surviving coverage and bounded teardown. The checked-in `TreeAtomicOr` pair changes only the same producer-release edge and directly requires exact/no-diagnostic correct behavior versus the intended conflict for the incorrect member. All 60 baseline/all-engine rows pass across all five RocJitsu targets and physical gfx950. |
| Physical-gfx950 HipKittens BF16, FP8, and MXFP8, Record/Replay and Sampled | These kernels' DS readers consume fragments written by `buffer_load_dwordx4 ... lds`, whose architectural LDS destination is implicit in M0 and physical lane identity. A prospectively frozen reached BF16 barrier removal corrupted the exact result but initially emitted no diagnostic because those direct writers were absent from the inventory. The adjacent checked-in members retain both B32 and the exact B128 E2E producer form plus peer DS readers and differ only by that publication barrier. | The pair contributes 20 green baseline/all-engine simulator rows on gfx942/gfx950 and 10 green physical-gfx950 rows. gfx950 checks one B32 plus four B128 dwords per lane exactly; gfx942 executes the B32 semantic transport because its compiler/ISA subset rejects the B128 LDS modifier. A host regression pins the documented CDNA4 operand and destination contract while rejecting unsupported encodings. Record/Replay and Sampled E2E qualification preserve exact BF16, optimized/reference FP8, and 65,536-element MXFP8 results with complete 128/128 or 96/96 access and 32/32 or 5/5 barrier coverage. Record/Replay slowdowns are 4.52x, 1.65x, and 3.21x; Sampled slowdowns are 1.47x, 1.88x, and 1.17x. All six separately frozen reached barrier faults produce required detector-owned diagnoses with complete surviving coverage, containment, cleanup, and health evidence; the three Sampled trials emit 24 conflicts each and claim all 1,024 or 384 selected windows. The focused `CdnaDirectToLdsPublication` plus `CdnaMfmaPipeline` Sampled gate passes 12/12 rows. Datatype-specific copies would duplicate the same observable publication contract, so no new fixtures are warranted. |
| Qwen, PyTorch selection/reduction, and Aorta heterogeneous-framework workloads | A compact object executes attention-like tile publication, a mixed-precision optimizer, and MoE-style LDS atomic routing through a shared epilogue while retaining an unexecuted register-rich multi-phase kernel. A second portable pair distills Qwen's final-output GEMM reuse of one LDS tile: it separates reader retirement before overwrite from publication of the replacement stage. Adjacent members remove only the respective semantic edge and retain exact results and independent control evidence. | The two all-engine pairs contribute 120 rows across all five RocJitsu targets and physical gfx950. The heterogeneous-object pair exposed Sampled cross-kernel false attribution and is host-guarded. The reused-LDS pair's 60 rows pass and the source-matched physical Qwen Record/Replay bundle passes an exact clean oracle with 658/658 access and 46/46 barrier coverage at 2.08x, then accepts a prospectively reviewed exact-one reader-retirement removal that fails the oracle and emits 120,034 diagnostics. A current targeted physical Qwen Sampled campaign passes the exact output-matmul oracle with 76/76 accesses and 4/4 barriers, then applies the prospectively reviewed reader-retirement removal exactly once and emits 32 conflicts with 76/76 access and 3/3 surviving-barrier coverage. The compiler-kernarg-preload variant of `CdnaMixedPersistentState` adds 12 quick rows plus focused host coverage for both AMDHSA entry paths on CDNA3/CDNA4. A distinct current SuperCollider trial prospectively removes only the unconditional final-output convergence barrier after every possible store, preserves the exact oracle with no diagnosis, and passes exact-one containment and physical-device health; that qualified miss adds no new behavioral contract beyond the portable pair. The initial standard-cadence Record/Replay fault miss remains rejected rather than relabeled. Rebuilding the unchanged full gfx1250 model with the current O3 compiler removes obsolete runtime transpose initializers and restores the exact 151,936-logit baseline in 64.49 seconds. Record/Replay patches all 846 accesses and 80 barriers in 90 ms but remains execution-active after 28 dispatches at the bounded 180-second limit; Sampled likewise patches all 846 accesses and 74 applicable barriers in 61 ms and reaches 310 dispatches, including the final output kernel, without a verdict at the same bound. Preserve the portable reductions as the quick cross-target contracts; the residual gap is full-row emulator throughput/completion, not another device semantic to duplicate. |
| Physical-gfx950 PyTorch `torch.sort`/`torch.topk`, Inline Shadow and Sampled | Full-pressure selection kernels can reach a displaced LDS operation with `EXEC=0`. The correct member forces the same scalar pressure, empties `EXEC`, and requires exact scalar preservation with no diagnostic. The adjacent incorrect member retains the pressure with nonempty waves and requires the exact LDS conflict diagnostic while preserving its scalar checksum. Both members contain eight access sites so that they exercise scalable appended-body routing rather than only a one-site compact case. | The shared target-native source runs on RocJitsu `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`, plus physical `gfx950`. Baseline, Inline Shadow, and Sampled registrations give 36 rows. The 24 baseline/Inline rows pass in 1.13 seconds; the ten new simulator Sampled rows pass in 1.15 seconds and the two physical rows in 0.55 seconds. The reduction first exposed fixed-stack gfx1201/gfx1250 Inline routing failures and now also protects Sampled's branch-only scalar-spill routing and leading empty-wave guard. Its cross-target run immediately exposed a second bug: an RDNA3 private-state prologue captured workitem identity only after its scalar-spill scratch had clobbered `v0`. A focused host regression pins capture-before-clobber ordering, and the corrected gfx1100 incorrect member again observes distinct wave owners and the required conflict. The repaired physical TopK Sampled E2E process passes both exact BF16/FP64 value/index oracles with zero diagnostics and complete dynamic evidence in 92.06 seconds, instrumenting 232,814/239,722 accesses and all 6,743 applicable barriers; its residual 6,908 no-first-SOPP-hop sites remain an E2E placement frontier rather than a missing behavioral contract in this pair. |
| Physical-gfx950 PyTorch `torch.sort`, Inline Shadow owner-local planning | After the empty-wave behavior was covered, the full generated object still corrupted 744 output indices because disconnected kernels with different SGPR tails were forced through an unsafe object-wide scalar ABI. This is not a new race contract: it is the resource-planning envelope around the existing eight-site correct/incorrect pair. | A focused host regression synthesizes disconnected 64- and 96-SGPR owners, requires both device access sites to survive lowering, and verifies that only the incompatible owner receives scalar-spill and persistent-dispatch overrides. All 722 `ConSanMoi.*` tests pass, while the final physical E2E run passes exact values and indices with complete 56,884-access/6,032-barrier coverage. Do not add a prototype-layout assertion to the device tier; expand the paired workload only if a future replacement exposes a distinct observable behavior not already covered by the empty/nonempty-wave contract. |
| Physical-gfx950 PyTorch norm/softmax, Sampled | The exact device oracle completed before the 30-second limit, but host teardown scanned all 46,080 allocated watchpoint slots even though the report contained zero claimed windows. The reader now stops after the committed `Ready` count and reconstructs deferred releases with one scan per relevant owner bank instead of one capacity scan per visible access. This is report-reader complexity, not a new device semantic: the existing adjacent two-stage-softmax pair owns the reduction/global-intermediate behavior and the existing Stream-K pairs own Sampled acquire-release publication. | Two focused host regressions pin zero examined slots for an empty allocated report and one capacity scan, rather than two, for two visible entries sharing a pending owner bank. All 186 hook tests pass. The 18 selected Sampled Stream-K correct/incorrect rows pass across all five RocJitsu targets plus physical gfx950 in 0.94 seconds. The final E2E run passes exact norm/softmax, complete 4,820-access/2,030-barrier coverage, and the ordinary 30-second contract in 28.98 seconds. |
| Physical-gfx950 PyTorch norm/softmax, Inline Shadow | Compiler-generated reduction loads use tied address/result operands (`ds_read_b32 vN, vN`) while ordinary and accumulator liveness leave only a compact spill window disjoint from `vN`. The paired reduction preserves the max/sum shape and exact live ordinary/AccVGPR values; its incorrect member removes only the LDS publication barrier. | The new source runs baseline plus all four engines on RocJitsu gfx942/gfx950 and physical gfx950, for 30 passing rows. Inline Shadow additionally requires selection of the 16-VGPR spill-backed recovery path. Three host regressions prove both overlapping and disjoint clobbered-address cases and forbid an out-of-window snapshot. This closes the compact-VGPR resource defect, but not the E2E cell: 41 full-pressure sites still need a scalar branch-only body whose nearest safe relay is roughly 1.5 MiB away. |
| Physical-gfx950 HIP Stream-K and rocBLAS, Record/Replay | Exact device oracles finished, but repeated field-wise scans and replay directly from 237--526 MB fine-grained HSA report allocations imposed a 45--120-second post-oracle latency floor. The reader now takes one sequential snapshot into cacheable host storage before any parser or replay pass. The two-tile row then exposed eager event-count-squared causal metadata, unbounded per-fence logging, lane-scaled fence capacity being conflated with wave-scaled barrier capacity, and access/barrier placement consuming relays needed by later compiler fences. These are host model, report-policy, and target-native placement envelopes around device behavior already owned by the adjacent Stream-K pairs. | Focused host regressions pin the fine-grained snapshot, independent barrier/fence relays around 80 access sites, lane-scaled report headroom, demand-grown metadata on a 4,096-fence trace, and bounded detail logging. The exact simple, two-tile, and rocBLAS SGEMM commands are registered descriptors with host regressions pinning their complete contracts. All three source-matched Record/Replay rows are green. Simple preserves `Errors: 0`, covers 32/32 accesses, 3/3 barriers, and 2/2 fences at 7.01×, and detects its reviewed phase-publication deletion. Two-tile preserves `Errors: 0`, covers 80/80 accesses, 5/5 barriers, and 2/2 fences at 31.05×, losslessly replays 20,525 fences with only one retained causal component, and emits 281,397 diagnostics for its reviewed exact-one deletion. rocBLAS preserves the exact `Square_64x64` oracle, covers 49,435/49,435 accesses and 4,997/4,997 barriers at 59.60×, and emits 4,086 replay diagnostics for its prospectively selected exact-one tile-publication deletion while retaining the numerical result. All 752 `ConSanMoi.*` tests, 192 hook tests, and 314 simulator Record/Replay rows passed at the preceding implementation checkpoint. The new all-target `MultiWaveGemmTile` pair distills rocBLAS's 256-thread, 16-KiB cooperative tile and reconverged publication edge: its 50 simulator and ten physical baseline/all-engine rows pass. Existing CDNA MFMA and Stream-K pairs retain the target-native instruction and access-heavy aspects without prototype-layout assertions. |
| Physical-gfx950 HIP Stream-K simple and two-tile, Sampled | The simple row exercises a reached loop phase-publication barrier before a buffer-state switch and peer `ds_read2_b32` consumers. The two-tile row adds distinct producer waves that write adjacent FP16 halves of one LDS dword before a third-wave combined-value consumer; it originally exposed Sampled's false conflict from four-byte-cell range compression. | Current-tip paired rows preserve exact `Errors: 0` oracles with complete 32-access/3-barrier/2-atomic and 80-access/5-barrier/2-atomic coverage at 1.83x and 1.94x Sampled slowdown. Their prospectively reviewed exact-one publication removals produce one and six required Sampled conflicts with complete surviving coverage, 192/192 and 384/384 claimed windows, bounded teardown, and healthy physical-GPU probes. Expected sampling saturation is recorded without making a lossless-sampling claim. `StreamKLastArriver` transports the simple publication behavior to all five simulator targets plus physical gfx950; `AccessHeavyStreamK` transports the 80-access/five-phase shape to that same full matrix; and `AdjacentSubwordWriters` transports the exact-byte contract likewise. Focused host tests pin byte-range packing and overlap without exposing representation in the device contract. |
| Physical-gfx950 PyTorch `torch.mode`, Record/Replay | Even after sparse compaction, the exact fully covered row spent roughly 13 seconds copying semantically visible fields from a 533-MB fine-grained HSA report during host teardown. Record/Replay auto reports now prefer coarse-grained storage when available and use the existing HSA bulk-snapshot path, while retaining fine fallback and leaving Sampled and Inline Shadow's fine preference unchanged; SuperCollider has a separate allocator. This is host allocation policy, not a new device race semantic: the existing reduction, B96 tuple-publication, segmented-top-k, and replay pairs own the relevant transformed-device behavior. | A focused host regression offers fine then coarse regions and requires Record/Replay's coarse preference and fine fallback plus the two other MOI engines' unchanged fine preference. All 166 hook tests, all 250 simulator Record/Replay device rows, all 58 physical Record/Replay rows, and the physical health check pass. The physical E2E row falls from 40.79 to 27.61 seconds while retaining the exact oracle, complete 25,523-access/3,920-barrier coverage, lossless replay of 13,017 accesses plus 49 barriers, and zero diagnostics. Do not add a prototype-specific device assertion for region selection; extend the behavioral pairs only if the production replacement exposes a distinct device-observable contract. |
| Physical-gfx950 PyTorch `torch.mode`, Sampled | The dispatched int32 specialization performs an in-place LDS value/index compare/exchange stage. Its selected barrier follows the conditional writes and precedes the next LDS reads, but the E2E launch is one 64-thread gfx950 wave, so deleting that barrier cannot remove an inter-wave publication edge. The prospectively frozen contract is therefore a reached `not_detected/pass` qualified miss, not a positive race. | The paired E2E row passes the exact oracle with complete 25,523-access/3,920-barrier coverage at 64.74x device slowdown. Its exact-one occurrence-2 deletion at PC `0x6a904` is accepted under a bounded 60-second process envelope with one requested/planned/applied mutation, the exact oracle, 25,523 accesses, 3,919 surviving barriers, zero diagnostics or overflow, cleanup, and healthy pre/post probes. The new architecture-general `SortNetworkExchange` pair transports the algorithmic value/index exchange into two waves, making the stage-publication edge genuinely causal rather than copying the single-wave no-op. All 50 simulator and ten physical baseline/all-engine rows pass across the five RocJitsu targets plus physical gfx950. |
| Physical-gfx950 PyTorch `torch.topk`, Record/Replay | The 20.8-MB generated object needs 403,542,016 bytes of alignment-inclusive growth, just above the former 384-MiB default, so strict validation rejected before either exact oracle. Qualified generated workloads now define a still-bounded 400-MiB default envelope without a workload-specific expert override. This is resource admission rather than a new device semantic: the adjacent segmented-top-k pair already owns value/index tuples, FP64/BF16-like pressure, tails, phase publication, and the missing-edge diagnostic. The residual large-object gap then showed that a safe ordinary-instruction run whose appended copy is itself more than one SOPP hop away must be able to extend the route recursively through later materialized runs. | Focused host tests pin the growth envelope, recursive reservoir planning and transitive materialization, routed-reservoir validation, preservation of pristine-NOP capacity, and the new cross-cut capacity invariant. The paired host capacity fixture proves that surplus donors in earlier cuts cannot compensate for an under-capacity terminal cut. The adjacent `RecordReplayLongRangeFullPressure` device pair distills the observable TopK geometry into a 100,000-instruction no-cave block with broad scalar-bank and alternating full-VGPR pressure: its correct member requires exact warmup/publication/control values with no diagnostic, while the incorrect member removes only the cross-wave publication edge and requires the conflict. It deliberately asserts no prototype routing telemetry. The pair caught both unsafe reuse of a live spill-window VGPR and relocation across a reconvergence entry while developing a one-way owner-local entry. Focused host tests additionally require exact preplaced-entry routing, dead-backup liveness, single-entry CFG ownership, protected synchronization ranges, and final access/barrier validation. Baseline and Record/Replay contribute 24 green rows across all five RocJitsu targets and physical gfx950. The final 117.59-second physical TopK run still passes both exact oracles, emits zero diagnostics, and replays completely, but retains a static 232,814/239,722-access and 6,743/11,423-barrier frontier because none of its limiting sites satisfies every safe borrowed-entry proof. The 57,155 candidates across the object do not cure the limiting cut: its 400 adopted reservoirs provide 13,688 relay words against 23,176 required entry/return paths, a 9,488-word deficit. The E2E cell therefore remains orange pending a more general owner-local gateway/shared-dispatch or equivalent routing-ABI design. A device fixture scaled to that exact relay count would freeze prototype layout rather than add behavioral coverage. |
| Physical-gfx950 PyTorch norm/softmax, Record/Replay | A barrier-only replay can have a zero-access compact input and zero-byte replay scratch while retaining the full report's nonzero diagnostic capacity. The validator conflated those producer contracts and rejected an otherwise exact, fully covered clean run. The fully covered device interval then finished below 30 seconds while fixed Python/framework startup pushed the whole process just above the generic bound. Finally, a loaded FP32 norm specialization was not the one dispatched by the exact workload: mutation installation alone could not prove site reach. These are validation bookkeeping, target process-envelope, and fault-qualification issues rather than new device semantics. | Focused parser regressions pin zero-input replay, clamped compact capacity, historical summaries, and the target-resolved 60-second process envelope. A descriptor-to-dispatch mapping plus final-ISA control-flow review selects the actually executed specialization before freezing the fault policy. The prospective exact-one drop of its first adjacent barrier is accepted as a reached `not_detected/pass` qualified miss; the earlier loaded-but-undispatched trial remains rejected rather than being given retrospective reach. Exact norm/softmax retains full 4,820-access/2,096-barrier coverage and zero diagnostics. The architecture-general `TwoStageSoftmax` pair already owns reduction/intermediate publication, while `DoubleBufferedPipeline` owns redundant-adjacent versus missing-publication barriers across all five RocJitsu targets and physical gfx950; adding a prototype-shaped fixture would duplicate those observable contracts. |
| Physical-gfx950 PyTorch `torch.sort`, Record/Replay bounded replay and cooperative initialization | Dropping one reached float radix barrier corrupts lane-derived LDS offsets as well as the exact sorted values. The resulting trace contains ordinary conflict ranges and offsets near the 32-bit boundary; its dense scratch estimate is 1,073,741,824 entries. The old host reader skipped replay above one million entries, losing the very diagnostics needed for the incorrect workload. A later reached barrier separates cooperative LDS initialization by all waves from peer consumption. | Auto replay now keeps its one-million-entry memory bound but runs the sparse model fail-closed. A focused hook regression requires an overlimit record to produce a bounded replay, a metadata-full diagnostic, and an incomplete verdict rather than a skip. A second host regression admits the hook's `empty_accumulator_descriptor_growth` resource-plan vocabulary so complete rows do not become parser-incomplete. The original prospective mutation becomes diagnostic but retains its frozen rejected `oracle=pass` disposition. Four later frozen phase/handoff trials are also retained as rejected without relabeling. The prospectively frozen initialization drop is accepted: one mutation produces 23 replay diagnostics while the exact sort oracle passes, with complete static/dynamic/parser evidence and clean health. The new adjacent `CooperativeLdsInitialization` pair owns that initialization-publication behavior in all 60 rows across the five RocJitsu targets and physical gfx950. `SegmentedTopK` continues to own radix selection and `DoubleBufferedPipeline` owns cross-wave storage reuse plus wide publication. |
| Physical-gfx950 and RocJitsu-gfx1250 PyTorch `torch.histc` | The executed FP32 and FP64 shared-memory histograms have distinct zero-initialization-to-accumulation and accumulation-to-global-copyout barriers around collision-heavy bin updates. The FP64 specialization contributes a real `ds_add_f64` payload and a distinct final publication site. Both edges are semantically causal, while their numerical manifestation depends on scheduling. | The expanded checked-in `HistogramScatter` correct/incorrect pair owns the architecture-general collision and missing-publication contract across all five RocJitsu targets and physical gfx950, now including exact FP32 and FP64 LDS `atomicAdd` payloads plus exact integer/BF16 global collision sums. CDNA3/4/5 disassembly retains native `ds_add_f64`; RDNA3/4 deliberately transport the same collision contract through their compiler-selected 64-bit LDS compare/exchange loops. The scalar BF16 path explicitly requests relaxed device-scope ordering, matching scatter accumulation rather than manufacturing a synchronization edge; all five compilers select a CAS-loop lowering, which the 60 rows execute without a false diagnostic. The physical gfx950 E2E rows pass exact FP32/FP64 counts with complete 179-access/84-barrier coverage. Record/Replay diagnoses a prospectively frozen FP64 final-barrier drop and is green without relabeling two earlier rejected FP32 trials. Sampled records 209.80x maximum paired slowdown and accepts a distinct prospectively frozen exact-one FP32 copyout qualified miss at audited stride one; its stronger FP64-copyout and FP32-initialization detection hypotheses remain rejected. The quick pair deliberately keeps the stronger semantic missing-edge diagnostic rather than reproducing physical schedule masking. Cross-architecture refresh then found gfx1250 Inline Shadow growing a reused access dispatcher over a local barrier body emitted earlier in the same pass. The repair bounds reuse against both committed and current-pass inventories; a focused host regression pins the invariant. RocJitsu-gfx1250 now passes exact dual-precision oracles under all four engines with complete profile-appropriate coverage. |
| Physical-gfx950 PyTorch `torch.scatter_reduce` | BF16 and FP32 rows drive many colliding updates through real relaxed singleton global atomics. They require exact collision sums and no false diagnostic, but do not contain a release/acquire publication edge that an ordered-atomic mutation could truthfully remove. | The source-matched Sampled bundle passes both exact oracles with complete 27/27 ordinary-access coverage, zero diagnostics, complete verdicts, and 144.92x maximum paired slowdown. Current target-native inventory confirms the real BF16/FP32 atomic sites; a prospectively reviewed `atomic-weaken-order` policy is accepted as typed N/A because weakening a nonexistent synchronization edge is not a behavioral fault. `HistogramScatter` transports the exact relaxed scalar-BF16 collision payload across all five simulator targets and physical gfx950 and the native packed-BF16 form on CDNA3/4/5 and RDNA4, while its adjacent pair supplies a separate truthful LDS publication defect. No artificial `scatter_reduce` ordering fixture is added. |
| RocJitsu-gfx1250 Tensile `015_spmm_f8_ml`, Record/Replay | The generated sparse-FP8 kernels stage packed bytes and sparse metadata through `ds_store_b8`, `ds_store_b8_d16_hi`, `ds_load_u8`, `ds_load_tr8_b64`, and `ds_bpermute_b32` before executing `v_swmmac_f32_16x16x128_fp8_fp8`. Two fresh inventories bounded at 30 and 180 seconds remained in active 64-way Tensile code generation before loading their first object, so repeating the full-client cycle is tactically deferred rather than mistaken for a ConSan or RocJitsu hang. | The adjacent gfx1250-only `Gfx1250SparseFp8Pipeline` pair retains the target-native packed-byte, transposed-metadata, and sparse SWMMAC shape. Its correct member requires exact packed bytes `0x5aa5`, a 64.0 matrix result, and control/canary values with no diagnostic; its incorrect member removes only the producer/consumer publication barrier and requires the intended diagnostic. Baseline and all four engines contribute ten green rows. `Gfx1250ExecutionTest.DsStoreB8D16HiSelectsUpperHalf` separately pins exact D16-high plus offset semantics in the emulator. This is durable coverage extraction, not paired/fault E2E evidence: the clean full-client 172,468/172,468-access and 3,060/3,060-barrier result remains valid, but the Record/Replay cell stays yellow. |
| Physical-gfx950 PyTorch `torch.sort`, Sampled | PyTorch radix-sort kernels contain dual-address LDS instructions: 56,884 static access sites expand to 70,420 logical report ranges. Sampled incorrectly compared those banked report slots with `max_patches`, a site-count budget, and silently omitted 3,980 sites. | A cross-target host regression uses two native dual-address sites, a two-site patch budget, and 32 banked report slots on CDNA3/4/5 and RDNA3/4. The adjacent device pair executes the same correct/missing-edge contract under baseline and Sampled on all five RocJitsu targets plus physical gfx950; all 24 rows pass. The repaired physical E2E row passes the exact value/index oracle with complete 56,884-access/6,032-barrier coverage. |
| Framework and RCCL-style producer/consumer state machines | A producer writes an LDS payload and publishes a phase transition with release compare-exchange; a consumer uses a language-level acquire load, which lowers to an ordinary load/wait/cache sequence on the relevant targets, before reading the payload. The incorrect member retains the same CAS and load shape but acquires an already-ready distinct phase object, so it does not import the producer's publication edge. | The adjacent all-engine pair contributes 60 green rows across all five RocJitsu targets and physical gfx950. Its first run exposed cross-architecture ordinary global-load decoding gaps, Record/Replay's split atomic/fence state, post-guest CAS event reservation, and missing Sampled/Inline handling for language-level ordinary atomic-acquire sequences. Focused host tests retain each transformation and model contract without asserting code-cave layout. |
| Framework and RCCL-style ordinary release stores | A producer writes an LDS payload and publishes readiness with a language-level agent-scope release store; a consumer performs an acquire compare-exchange before reading the payload. The incorrect member releases a distinct phase object while the consumed object is already ready, preserving the release-store/acquire-CAS instruction shape without importing the payload edge. | The adjacent all-engine pair contributes 60 green rows across all five RocJitsu targets and physical gfx950. Its first run exposed missing ordinary release-store association and engine semantics, incomplete compiler-release wait/cache recognition, and incorrect CDNA5 scaled-VGLOBAL effective-address reconstruction. Focused host tests retain exact association, per-engine publication state, and CDNA5 scale-bit/address materialization while rejecting malformed or cross-architecture scale use. |
| RocJitsu-gfx1250 Tensile `007_sk_mxf4gemm_tdm`, Record/Replay | Dense LDS accesses sit beside an addressed buffer acquire, device-scope global invalidate, and split-barrier signal/wait. A far fence with no ordinary island must reuse an access relay under owner-local scalar pressure, including space reserved by Record/Replay's barrier dispatcher. | A focused two-owner host regression pins 18 access patches, 4 barrier records, 2 far-fence patches, owner-local call-return routing, and final validation. The adjacent checked-in device pair composes an independent target-native workgroup barrier with cache-mediated LDS release/acquire publication. gfx1201/gfx1250 use native split signal/wait and gfx942/gfx950/gfx1100 use their monolithic semantic equivalent; all 50 simulator and 10 physical rows pass. Exact-size sharding now supersedes the monolithic 1,800-second timeout: the source-matched E2E bundle passes all 96/96 numerical rows with complete access/barrier/fence coverage and lossless replay, records a current 19.47x paired slowdown, and accepts a prospectively reviewed exact-one qualified miss with complete containment, health, and cleanup evidence. The E2E cell is green. |
| RocJitsu-gfx1250 Tensile `007_sk_mxf4gemm_tdm`, high-bank and adjacent-barrier LDS tail | Generated MXF4 kernels select a nonzero gfx1250 SRC0 bank and name an LDS address such as physical `v286` through encoded `v30`; high-pressure variants also require a spill-backed low-bank instrumentation window. The selected kernel places adjacent tensor-to-LDS split barriers around a descriptor wait. Record/Replay must capture the physical high-bank address before selecting low scratch without overwriting a live spill victim, then execute the guest instruction under its complete original bank mode. | The adjacent gfx1250-only device pair preserves the same encoded high-bank address in correct and missing-publication workloads. Its correct member now mirrors the E2E tail with split barriers separated by `s_wait_dscnt`; the incorrect member removes the publication edge. Baseline and Record/Replay contribute four green rows with exact results/no diagnostic for the correct member and the required conflict for the incorrect member. Focused host regressions separately pin appended placement despite inline padding, composite SRC0/DST mode normalization, save-before-capture ordering for a seven-VGPR spill window, and bounded inventory/fault sharding. The source-matched E2E bundle passes all 96/96 numerical rows, processes 5,281,152 accesses plus 108,992 barriers without loss, records 19.47x paired overhead, and accepts a prospectively reviewed redundant-barrier qualified miss with complete containment and health evidence. |
| Physical-gfx950 D128-pressure attention, SuperCollider, Record/Replay, and Sampled | The full-KV FastContext loop contains consecutive workgroup barriers. Dropping exactly the first is a semantically redundant qualified miss because the adjacent second barrier preserves publication; all three engines must retain the pair in clean execution without inventing a conflict. A separate SuperCollider trial prospectively expected an ExactContext K/V-publication drop to fail, but physical scheduling masked it; that trial remains rejected without relabeling. | The architecture-general double-buffered-pipeline correct member retains two adjacent publication barriers, while its adjacent incorrect member removes the complete publication edge and requires the semantic conflict diagnostic. All 60 baseline/all-engine rows pass on RocJitsu gfx942, gfx950, gfx1100, gfx1201, and gfx1250 plus physical gfx950. The source-matched E2E bundles pass exact oracles with complete engine-applicable coverage, record 17.06x SuperCollider, 19.38x Record/Replay, and 17.32x Sampled paired overhead, and independently accept prospectively reviewed exact-one redundant-barrier misses with complete mutation, bounded or zero report memory, cleanup, health, and hook-hash evidence. No duplicate prototype-shaped device pair is warranted because the existing pair owns both the correct adjacent-barrier contract and the incorrect missing-publication contract. |
| Physical-gfx950 Jakub FP16 attention matmul, SuperCollider, Record/Replay, and Sampled | Pipelined and double-buffered MFMA kernels publish wide LDS fragments between eight producer/consumer waves. The fault campaigns distinguish an intentionally delayed producer from ordinary schedule-masked publication while retaining exact host-reference results. | The architecture-general double-buffered-pipeline correct/incorrect pair owns the observable publication contract across all five RocJitsu targets and physical gfx950; the CDNA MFMA pair adds target-native matrix/live-state pressure. Their combined 90-row gate passes. The source-matched E2E bundles pass all four exact oracles with 338/338 access coverage, record 2.96x SuperCollider, 4.37x Record/Replay, and 2.95x Sampled paired overhead, and independently accept prospectively reviewed exact-one schedule-masked qualified misses; Record/Replay and Sampled additionally retain complete 35/35 barrier coverage. The Sampled campaign retains its separately frozen rejected producer-skew trial after physical scheduling masked the expected diagnosis and oracle manifestation; no expectation was revised or relabeled. The schedule-masked outcome is containment evidence, not a valid incorrect-workload contract, so the existing generalized pairs remain the durable device tests. |
| Physical-gfx950 Stream-K/Sharktank/CLIP, retained Triton assembly, and gfx1250 Sharktank manifests, dual-address B64 LDS | Stream-K consumers and gfx1250 Sharktank fault identities use adjacent `ds_read2_b64` / `ds_load_2addr_b64`; generated attention and matmul kernels also use stride-64 `ds_read2st64_b64` / `ds_write2st64_b64 offset1:8` to move values between ranges 4 KiB apart. Host transformations covered decoding and placement, but no checked-in device pair previously executed both B64 layouts. | `StridedDualAddressPublication` uses one producer and one consumer wave, four exact independent 64-bit values per lane, and adjacent correct/missing-barrier members. It executes both adjacent and stride-64 native CDNA3/4 spellings plus their gfx11 DS and gfx12 VDS spellings/waits under baseline and all four engines on five RocJitsu targets plus physical gfx950. All 60 rows pass; the correct member also pins real native group-segment allocation through its exact result oracle. |
| Physical-gfx950 Sharktank TP2 family, Sampled | Six executed generated code objects combine prefill/decode attention with several matmul shapes. The prefill attention kernel has early DPP-phase synchronization that must be distinguished from its later semantic LDS publication edges; deleting the former exactly once is schedule-masked, while deleting an actual publication edge is the behavioral race the sanitizer must diagnose. | The current E2E bundle passes all three exact oracles with complete 1524/1524 access and 150/150 applicable-barrier coverage at a 1.69x maximum paired slowdown. Its prospectively frozen DPP-phase mutation is reached once and accepted as `not_detected/pass`, with 149/149 surviving barriers, 240 sampled windows, bounded teardown, and healthy pre/post probes. `DoubleBufferedPipeline` carries the actual correct/incorrect publication contract across all five simulator targets and physical gfx950, while `CdnaMfmaPipeline` adds native CDNA3/CDNA4 matrix and accumulator pressure; their 90-row all-engine gate passes. A device test whose incorrect member merely encodes this schedule-masked no-op would violate the behavioral-pair rule, so the qualified miss remains E2E containment evidence rather than a duplicate checked-in fixture. |

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
| Synchronization faults | Every scenario is shape-matched and paired; reductions, helpers, dynamic stacks, pipelines, routing, batching, cache sequence, and instruction scope all change one real property. `IndexedLdsAlias` additionally preserves every synchronization edge and changes only a runtime address stride, distinguishing effective-address races from missing-ordering faults. | Add another address transform only when an E2E case establishes behavior not carried by the zero-stride alias, subword overlap, B96 alias, or indexed routing contracts. |
| Stream-K and collective publication | Fetch-add last-arriver and tree atomic-OR pairs cover partial publication and completion. `TopKPrefix` transports lane-zero atomic reservation, ballot rank, lane broadcast, paired LDS loads, and exact prefix/payload oracles across wave64 CDNA3/4 and wave32 RDNA3/4/CDNA5. `RcclPartialBarrier` adds a reusable multi-generation partial-workgroup barrier: two worker waves release to a shared generation counter, poll it with acquire, reduce rank-filled vector chunks, and reuse the same scratch while non-worker waves remain independent on wave32 targets. | A closer relaxed-poll/final-RMW tree remains a refinement only if a selected E2E kernel proves behavior not already carried by these pairs. Inter-GPU transport and host stream ordering remain outside this single-kernel tier. |
| Multi-stage pipelines | Double buffering, storage reuse, wide LDS, lane exchange, two-stage softmax, a four-wave hierarchical FP16-partial/FP32-accumulation reduction with three distinct ragged producer tails, CDNA MFMA, RDNA WMMA, RDNA4 FP8, gfx1250 sparse SWMMAC and scaled WMMA, repeated dispatch, a portable 256-thread/16-KiB peer-wave GEMM tile, and a gfx950-specific 160-KiB/32-stage/eight-wave publication pipeline are covered. | Production-sized combined resource pressure remains an E2E obligation; the previously separate FP16 production-reduction gap is now represented by an exact portable pair. |
| Selection and reduction | Segmented value/index top-k, tails, two-stage softmax, the MoE router, cooperative initialization, and a two-wave value/index sort-network exchange cover multi-phase selection/reduction shapes. | Signed/unsigned D16-high variants remain optional refinements. |
| Atomic diversity | Arrival add, tree OR, reusable generation-counter release/acquire, release CAS plus language-level acquire load, language-level release store plus acquire CAS, integer/FP32/FP64 LDS histogram collisions, exact integer and relaxed scalar-BF16 global collision sums, native packed-BF16 global collision sums, MoE bins, cache-qualified VGLOBAL publication, and gfx12 instruction-scoped atomics are paired. | The FP64 histogram executes native `ds_add_f64` on CDNA3/4/5 and the compiler-selected LDS compare/exchange loop on RDNA3/4. The scalar BF16 payload uses the compiler-selected 32-bit CAS loop on every current target. The native packed-BF16 form runs on CDNA3/4/5 and RDNA4 and is typed N/A on RDNA3; a direct host regression independently covers packed FP16 and BF16 additions through both global and LDS collision paths. The atomics remain arithmetic-only exact oracles beside the pair's truthful LDS publication defect. |
| LDS and data movement | The suite exercises subwords, B32/B96/B128, group-FLAT wide traffic, B128-to-AccVGPR, CDNA direct-global-to-LDS writes with implicit physical-lane/M0 destinations, native gfx950 `ds_read_b64_tr_b16`, wave-native Top-K ballot/broadcast plus paired-address loads, and E2E-derived adjacent plus stride-64 dual-address B64 publication. The direct-to-LDS pair now executes both B32 and the production B128 form on gfx950 simulation and hardware, with an applicable B32 semantic transport on gfx942. The dual-address pair executes adjacent CDNA3/4 `ds_read2_b64`/`ds_write2_b64` and gfx11/12 `ds_load_2addr_b64`/`ds_store_2addr_b64` together with their stride-64 forms under one four-value oracle. gfx1250 high-bank SRC0 addresses, D16-high packed-byte stores plus transposed metadata loads, and target-native load/store spellings are also covered. CDNA5 additionally executes every supported B8/B32/B64/B128 width in both `global_load_async_to_lds` and `global_store_async_from_lds`, using four disjoint LDS regions and `s_wait_asynccnt`. | Further dual-address or transpose forms remain host/ISA evidence unless an E2E workload supplies behavior not represented by the adjacent/stride-64 pair. `ds_swizzle` has no LDS publication semantics and should not receive an artificial racy pair. |
| TDM, clusters, and gfx1250 matrix staging | gfx1250 uses real extended clustered-dispatch packets for two- and four-CTA cluster barriers, two-cluster identity/isolation, every B8/B32/B64/B128 `cluster_load_async_to_lds` width plus one `s_wait_asynccnt`, and multicast. The width family comes from Composable Kernel's production cluster-load abstractions; the ordinary and multicast correct members check every delivered byte/word exactly, while their incorrect members retain the same transfers and remove only the cluster publication edge. The four-CTA pair transports the two-CTA E2E-derived publication contract to a wider topology so a hidden two-participant assumption cannot pass. Separate exact-result pairs cover scaled FP8 WMMA, the Jakub/attention native K=32 FP16 WMMA with a 32-byte per-lane fragment, ordinary async global-to-LDS, the RCCL-derived LDS-to-global async-store widths, and the PyTorch/Triton tensor-descriptor path. The latter issues two `tensor_load_to_lds`, one `tensor_store_from_lds`, and three tensor-count waits in a three-wave exact add pipeline. | The tensor-descriptor pair proves native instruction preservation and a truthful adjacent LDS publication fault; it does not claim that the current prototype instruments descriptor-sized TDM ranges themselves. No further checked-in semantic gap is established by the synchronous `cluster_load_b*` family: those instructions read global address space and return a workgroup-broadcast value, rather than accessing LDS, so manufacturing a race around them would claim general global-memory race detection outside ConSan's model. Add another pair only when an E2E workload exposes a distinct in-scope LDS or ordering contract. |
| Resource and control pressure | Dynamic stack, shared helpers, MFMA/WMMA live state, CDNA AccVGPR destinations, full ordinary-VGPR-bank pressure, disconnected fixed-stack/private and dynamic-stack/register persistent-state owners, gfx1250 high-bank address capture with a spill-backed low-bank window and adjacent double-barrier tail, B96/B128 aliasing, a dense-routed B16 access with live SCC, native VGLOBAL forms, descriptor growth, repeated dispatch, and spill-backed scalar preservation under empty and nonempty `EXEC` are exercised. The tranche caught AccVGPR-boundary, mixed-owner recovery/composition, dynamic-frame spill, dense-dispatch SCC, high-bank capture, empty-wave scalar-restore, live borrowed-backup, and reconvergence-entry defects. Focused host fixtures prove that direct-reservoir demand must fit through every mandatory branch-range cut and that one-way borrowed entries are admitted only with a liveness-dead VGPR and a single-entry relocatable window. | Combined worst-case forms and production-sized placement/relay limits remain E2E obligations. In particular, physical TopK currently needs 23,176 paths across a cut with capacity for 13,688; its limiting sites do not satisfy the safe one-way-entry proof. A more general owner-local gateway/shared-dispatch design should gain a small device contract for its observable behavior, while the existing host pair continues to own the capacity invariant. |
| Object and dispatch shape | Shared helpers have multiple kernel owners; softmax uses multiple stages and a global intermediate; continuous batching repeats changing dispatches from reset state; deterministic replay restores every state byte into one retained device allocation before two identical mixed-precision launches; the heterogeneous-object pair executes three independent kernels beside an unexecuted pressure owner; portable module lifecycle keeps four independently instrumented objects and streams live while alternating odd/even unload/reload subsets across three generations; the graph-replay pair captures two independent graphs on two streams and replays both for three generations; graph and executable-node parameter updates each redirect packed arguments to independently checked state; mixed-state pairs combine incompatible owner-resource envelopes. | The four-object module envelope covers complete evidence for eight initial/reloaded object instances and bounded simultaneous report ownership. Larger/heavier process-capacity pressure remains a lifecycle refinement only when an E2E case establishes a distinct failure boundary. |
| Scale | The registered 3,285-row matrix crosses 45 portable all-engine pairs, four common engine-scoped pressure pairs, focused family/target contracts, five simulator targets, physical CDNA4, and every engine. It includes RCCL-style partial barriers, adjacent/strided dual-address B64 traffic, bidirectional CDNA5 async data movement, native tensor-descriptor staging, runtime-indexed LDS aliasing, deterministic in-place state replay, heterogeneous objects, module and graph lifecycle, a ragged mixed-precision backward reduction, large LDS/deep pipelines, matrix variants, long-range live state, compiler-generated kernarg preloads, and clustered dispatch while staying checked in and bounded. The complete matrix passes at `-j64`; its 225.92-second warm-run wall time remains just below the five-minute review threshold. | Maximum-capacity and production-sized heterogeneous objects remain E2E responsibilities; add another medium object pair only if a distinct concrete failure mode justifies it. |

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
| P2 | HRX static, module, and HIP-graph launch probes | Exact output checks across static registration, `hipModuleLaunchKernel`, graph construction, graph parameter update, graph-exec parameter update, and replay. | Retain the add-100 probes as external launch-path integration coverage. Their graph update paths are now distilled into a meaningful adjacent ConSan pair that uses the same pre-packed `extra` argument-buffer APIs and proves each update with untouched and independently updated sentinel state. |

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
| Transformer backward and optimizer | **Implemented:** three producer waves publish independently valued FP16 partial gradients with three different non-multiple tails; a fourth wave accumulates the available partials in FP32, updates FP32 first-moment state, and writes a half-rounded AdamW-like parameter. The correct variant publishes every partial before consumption; the incorrect variant removes only that collective edge while keeping the same producer, tail, reduction, and update shape. | Bit-exact reduced gradients, moments, and rounded parameters in the correct run; exact active-thread controls plus untouched inactive controls/output canaries in both runs; no clean diagnostic; and the declared publication diagnostic in the incorrect run. The pair is green in all 60 baseline/all-engine rows. |
| Top-1 MoE routing | **Implemented:** a bounded router with argmax, per-expert bin counts, prefix offsets, indexed gather/scatter, an empty expert, uneven token tails, and a small GLU-like expert transform. The incorrect variant breaks publication of staged tokens without changing routing decisions. | Exact token-to-expert assignments, offsets, scattered tokens, gathered outputs, and untouched canaries; the incorrect run requires the corresponding race/order diagnostic on every target. |
| Dynamic continuous batching | **Implemented:** a repeated prefill/decode-shaped kernel sequence whose active-row count changes each launch, including partial waves and masked inactive rows. The pair uses a uniform staged reduction with exactly one required edge removed. | Exact output for every active row, unchanged inactive rows/canaries, stable repeated-run checksum, and the expected diagnostic only for the broken variant. This covers tails and lifecycle without pretending Aorta's simulated cache is a paged KV cache. |
| FSDP/DDP and RCCL kernels | **First selected idioms implemented:** `RcclPartialBarrier` reduces RCCL's reusable partial-workgroup generation counter into two worker waves that publish two stages of rank-filled `uint4` chunks, release and acquire one shared counter, and reuse the same LDS scratch; on wave32 targets two further waves remain non-workers. Its incorrect member changes only the final consumer acquire to an already-ready unrelated counter. `Gfx1250GlobalAsyncFromLds` reduces AsyncDataCopier's CDNA5 path into B8/B32/B64/B128 LDS-to-global transfers plus `s_wait_asynccnt`, with only the producer/consumer workgroup barrier removed in the incorrect member. | Exact two-rank vector reductions for two generations, exact counter/control state, inactive-tail canaries, and the matching publication diagnostic on all five targets plus physical gfx950 for the partial barrier; exact values for all four transfer widths, exact controls, and the diagnostic on emulated gfx1250 for the async copier. These are intra-kernel contracts only: they do not claim network transport, host stream dependencies, or unsupported global-memory race coverage. |
| Activation checkpoint and deterministic replay | **Implemented:** restore inputs, parameters, outputs, scratch/control state, and guards into the same retained device allocation before each of two mixed-precision forward/backward/update launches. FP16 LDS checkpoint traffic feeds FP32 activation, gradient, and parameter updates; the adjacent incorrect member removes only the checkpoint-publication edge in both runs. | The correct member requires exact bit patterns on both launches and bit-identical replay, with all canaries, inputs, parameters, controls, and guards checked. The incorrect member retains the full restore/control oracle and requires the semantic diagnostic. All 60 rows pass; the pair also caught and host-guards InlineShadow report sizing that previously reserved diagnostics for only one launch of a code object. |
| Multi-stream, multi-object framework lifecycle | **Implemented at a medium bounded envelope:** four independently compiled code objects remain simultaneously loaded on four streams for three generations. Odd and then even subsets are unloaded/reloaded while the complementary objects stay live. The adjacent incorrect member selects the racy kernel only in one designated module; exact controls remain required for every object/generation and exact publication results remain required for all nine nonfault dispatches. | The three MOI engines require a complete final verdict for all eight initial/reloaded objects and 32/32 access sites. Record/Replay observes the designated fault in each of its three generations while four simultaneous report buffers peak at about 65.2 MB without overflow. ConSan is not asked to infer missing host-side stream dependencies. Extend beyond this only for an E2E-evidenced heavier object or process-capacity boundary. |
| HIP module and graph launch | **Implemented for direct module lifecycle, repeated graph replay, and both parameter-update paths:** one graph pair captures two independent semantic workloads on separate streams and replays both for three generations. A second creates its node through `kernelParams`, redirects it before instantiation through `hipGraphKernelNodeSetParams` with a packed `extra` buffer, then redirects the executable through `hipGraphExecKernelNodeSetParams`; sentinel state proves both updates. Graph dependencies stay correct and the semantic fault remains inside the incorrect kernel. | Identical output and diagnostic contracts through ordinary, module, replay, graph-update, and executable-update paths, with exact stream/generation or updated-state attribution. A deliberately missing graph edge is out of scope until ConSan explicitly models inter-kernel ordering. |
| Heterogeneous generated objects | **Implemented for one object:** three small attention/optimizer/MoE kernels execute beside a shared helper and an unexecuted high-pressure multi-phase kernel. The incorrect member changes only the attention publication edge. A future multi-object lifecycle extension remains separate. | Exact results and control evidence for every dispatched kernel, no false attribution to independent or unexecuted entries, no clean diagnostic, and the declared diagnostic only for the broken publication. |

The Aorta/workload-derived tranche--backward/optimizer, MoE routing, continuous
batching with repeated reset-state dispatch, deterministic in-place state
restore/replay, the single heterogeneous object, four-object mixed-reload
module lifecycle, repeated graph replay, the first selected RCCL partial-barrier
idiom, and CDNA5 AsyncDataCopier store-from-LDS--is complete across its
applicable matrix. Continue the RCCL ISA survey only when another selected
kernel exposes a synchronization or data-movement behavior not carried by
these pairs. A heavier process-capacity or lifecycle envelope likewise needs
concrete evidence distinguishing it from the implemented four-object case.
Any further RCCL-inspired pair must reflect an actual selected kernel and only
behavior ConSan can observe. Graph-update and high-stream-count cases remain
integration envelopes around semantic pairs, not new race categories in their
own right.

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
The 45 common all-engine pairs contribute 2,700 rows across the five RocJitsu
targets plus physical `gfx950`; four common baseline/one-engine pressure pairs
add 96. Non-rectangular family, target, and focused implementation regressions
add 488 more rows, and the physical health check adds one, for 3,285 total.
Notable extensions include Top-K prefix exchange, large-text relay, subword
live-SCC, clobbering-load reduction, full-low-bank Stream-K, and RCCL partial
barriers plus adjacent and stride-64 dual-address B64 publication on every
target/backend;
CDNA-specific MFMA/AccVGPR/direct-to-LDS;
gfx950 transpose; 160-KiB LDS on gfx950 and gfx1250; RDNA WMMA/FP8 forms; and
gfx1250 cluster/TDM/high-bank, sparse-SWMMAC, bidirectional async LDS/global
transfer, native tensor-descriptor load/add/store staging, scaled-WMMA, and
native K=32 FP16-WMMA pairs.

The per-configuration arithmetic is:

| Configuration | Pairs | Rows |
| --- | ---: | ---: |
| RocJitsu `gfx942` | 57 | 536; dense SCC omits unsupported Record/Replay, while long-range/full-pressure, dense-threshold, independent-scalar, and CDNA kernarg-preload contracts retain their declared engine scopes |
| RocJitsu `gfx950` | 59 | 556; the same engine restrictions apply, plus native transpose and 160-KiB LDS pairs |
| Physical `gfx950` | 59 | 557, including the health check and focused scalar-spill rows |
| RocJitsu `gfx1100` | 52 | 502 |
| RocJitsu `gfx1201` | 53 | 512 |
| RocJitsu `gfx1250` | 64 | 622 |

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

The most recent exact `ctest -j64 -L consan-device` qualification passes all
3,285 rows in **225.92 seconds** on this host. Shell process-tree accounting
reports **1,440.07 user + 1,632.66 system = 3,072.73 seconds of CPU time**. The
budget applies to this label-only device command. Its 225.92-second result is
below the five-minute lower review threshold, so the
next coverage pass must ask whether meaningful workload-derived,
architecture-specific, or lifecycle contracts are still missing; it is not a
reason to add artificial work.
CTest's summed duration remains a concurrency-insensitive test-capacity
indicator, but it is not CPU time because individual processes can also wait
on the simulator, runtime, or physical GPU.

This measurement supersedes the earlier 67.91-second result, which allowed
all physical gfx950 processes to compete for one GPU and eventually reproduced
a GPU memory fault. All 557 currently registered physical rows share the
target-scoped `consan_physical_gfx950` CTest resource lock. The five simulator targets still
run concurrently with each other and with the one active physical row; a
different physical target would use its own lock rather than creating a global
bottleneck. The honest wall time remains below the 5--20-minute heuristic and
therefore triggers a coverage review.

## Issues exposed and retained as regressions

The current matrix has no red cells. Building it exposed defects in the
prototype, runtime, and simulator, all fixed without weakening the behavioral
contracts:

- physical gfx950 CTest oversubscription from missing per-test resource locks;
- `gfx942` and `gfx950` Record/Replay multidimensional workgroup identity;
- `gfx942` and `gfx950` Sampled reduction snapshot formation;
- Sampled compressing exact LDS byte ranges into four-byte cells, which made
  disjoint FP16 halves written by different waves falsely overlap;
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
- InlineShadow auto-report sizing reserving diagnostic headroom for only one
  launch even though one code object owns the report across multiple dispatch
  banks. The deterministic restore/replay pair exposed the second-launch
  overflow on four emulated targets and physical gfx950; a focused host test
  now requires diagnostic capacity to scale with the selected dispatch-bank
  count;
- adaptive auto-report fitting consulting only the engine-wide safety ceiling
  before allocation enforced a smaller explicit caller cap. A one-cell Inline
  contract therefore requested 16,384 dispatch-banked diagnostic slots, fit
  under the 128-MiB engine ceiling, and was rejected later by the fixture's
  still-sufficient 1-MiB cap instead of reducing diagnostic headroom. The
  planner now carries the bounded caller ceiling through Record/Replay,
  Sampled, and Inline fitting. Focused host tests cover all three engines, and
  the eleven multidimensional/forced-spill simulator rows retain the complete
  runtime regression without raising their cap;
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
  comparison and a directly reached B16 access body;
- CDNA descriptor SGPR growth borrowing the six-register physical
  VCC/XNACK/FLAT_SCRATCH tail for a production-scale SuperCollider return-PC
  pair;
- validation provenance treating ASLR loader addresses as linkage identity,
  which falsely made clean and paired rows appear to use different artifacts;
  and
- interrupted validation being unable to resume an already complete matching
  fault row without rerunning the destructive physical-GPU mutation; and
- the CDNA mixed-persistent-state fixture consuming a compiler-generated
  dynamic-stack scratch-base load in synthetic scalar-pressure assembly before
  its required scalar-memory wait. The existing correct/incorrect pair now
  retains the wait, and final-ISA waitcheck plus repeated uninstrumented and
  SuperCollider physical executions guard the fixture itself;
- CDNA Sampled stride-one private state treating kernarg-preload as a
  dispatch-capture-only property, leaving the firmware `+256` entry without a
  prologue, and the dynamic-stack owner/epoch path rejecting that same valid
  paired-entry ABI. Focused host tests cover both state representations, while
  the compiler-generated kernarg-preload correct/incorrect device pair covers
  baseline and Sampled behavior on CDNA3/CDNA4 and physical gfx950; and
- SuperCollider, Record/Replay, and Sampled attempting to call
  `hsa_memory_free` for live auto-report allocations from inside ROCR's tool
  `OnUnload` shutdown path, which could deadlock after a test had otherwise
  passed. A deterministic host lifecycle regression requires all three engines
  to defer those live allocations to runtime reclamation, while explicit
  executable destruction continues to exercise and require the ordinary HSA
  free path. Repeated physical reproducers and the complete device matrix guard
  the end-to-end behavior.

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

The 2026-08-23 capability pass reviewed every family/target-local pair rather
than treating its original filename as its applicability boundary. The suite
now has 45 all-engine common pairs and four common engine-scoped pressure
pairs. In addition to the earlier lifecycle, graph, replay, D16-high,
kernarg-preload, mixed-owner, and live-SCC transports, this pass generalized
Top-K prefix, full-low-bank Stream-K, large-text relay, live-SCC subword
traffic, clobbering LDS-load reduction, and an E2E-derived adjacent/stride-64
dual-address B64 publication contract across wave64 and wave32 families.
It also generalized the SuperCollider dense threshold and the Record/Replay
and Sampled owner-pressure contracts, transported the 160-KiB LDS pipeline to
gfx1250 and native B96 to gfx1100, and added the all-target RCCL partial barrier
plus gfx1250 LDS-to-global async transfer, tensor-descriptor load/store staging,
four-CTA cluster topology, and native K=32 FP16 WMMA fragment staging. All
3,285 registered rows pass after
the audit.

| Contract family | Applies and runs | Capability-based disposition elsewhere |
| --- | --- | --- |
| Forty-five common all-engine pairs | CDNA3/4/5 and RDNA3/4, plus physical CDNA4 | None; these are the rectangular semantic baseline. The newest members transport RCCL partial-barrier generations, Top-K prefix, full-low-bank Stream-K, large-text relay, live-SCC subword stores, tied-address/destination LDS reduction, and adjacent plus 4-KiB-separated dual-address B64 publication across target-native wave sizes, waits, barriers, and LDS spellings. |
| Wide group-FLAT | All five targets, plus physical CDNA4 | None. The gfx1250 omission was a registration gap, not an ISA limitation. |
| CDNA MFMA/two-tile/dense-route/AccVGPR pairs | CDNA3 and CDNA4, plus physical CDNA4 where available | Matrix/AccVGPR and direct-global-to-LDS forms remain family-specific. Their cross-cutting publication and resource ideas are also carried by the common full-bank, pressure, and pipeline pairs; RDNA and CDNA5 retain their target-native matrix contracts. |
| Dense SuperCollider threshold | All five targets, plus physical CDNA4 | The same 1,040-site/long-text observable contract now crosses each target's production dense dispatcher using native LDS forms. It remains SuperCollider-scoped rather than becoming a redundant all-engine scale test. |
| 160-KiB large-LDS pipeline | CDNA4 simulation and physical CDNA4, plus CDNA5 simulation | gfx1250 uses wave32, gfx12 DS spellings/waits, and its wider LDS address. Typed not-applicable where modeled group-segment capacity is below 160 KiB; smaller common pipelines carry the synchronization idea. |
| Native B96 aliasing | CDNA3/4 through the CDNA pair; RDNA3/4 and CDNA5 through the wave32 pair | Every supported family with a native ordinary B96 form now has a pair. The paths deliberately retain their family-native spelling and wait. |
| Instruction-encoded atomic scope | RDNA4 and CDNA5 | CDNA3/4 and RDNA3 express the corresponding ordering through different cache-qualified sequences covered by `VglobalCachePublication`; duplicating the gfx12 encoding is not applicable. |
| Packed 16-bit floating-point atomic add | Native packed-BF16 global collision updates on CDNA3/4/5 and RDNA4, plus physical CDNA4; direct global/LDS packed-FP16 and packed-BF16 emulator contracts | RDNA3 has no applicable native packed-BF16 memory-atomic form in the supported compiler/ISA subset, so it retains the portable scalar-BF16 CAS-loop path as its typed alternative. The generator regression pins packed-format preservation independently of the current generated execute code. |
| Matrix and FP8 staging | CDNA3/4 MFMA; RDNA3/4 FP16 WMMA; RDNA4 FP8; CDNA5 native K=32 FP16 WMMA, sparse-FP8 SWMMAC, and scaled WMMA | The installed compiler correctly rejects the RDNA4 `wmma-128b-insts` builtin for gfx1250. gfx1250 instead has three native correct/incorrect pairs: a 32-byte-per-lane FP16 fragment feeding K=32 WMMA, packed-byte/transposed-metadata sparse SWMMAC, and scaled WMMA with exact FP8 matrix output. |
| gfx11 scalar-base VGLOBAL | RDNA3 | Other targets' vector/scalar address materialization is already exercised by their native common VGLOBAL pair and host matrix; this fixture's missing-GL0 negative is specifically the gfx11 cache sequence. |
| Top-K prefix | All five targets, plus physical CDNA4 | Wave32 targets use their native broadcast and paired LDS form; CDNA3/4 use wave64 ballot/rank and paired reads while retaining the same exact reservation and payload contract. |
| gfx1250 cluster, TDM, high-bank, and async transfer | CDNA5 | Typed not-applicable on targets without clustered dispatch, CDNA5 tensor-data movement, high-bank LDS operands, or async global/LDS transfer builtins. Every supported B8/B32/B64/B128 width runs in both global-to-LDS and LDS-to-global behavioral pairs; symmetric host inventory regressions pin all eight zero-offset semantic forms and reject a nonzero immediate in each direction. A separate PyTorch/Triton-derived pipeline executes native tensor-descriptor load and store instructions with exact arithmetic and canaries; its diagnostic remains grounded in explicit LDS publication rather than claiming unsupported descriptor-range instrumentation. Cross-cutting publication and matrix ideas are carried by common and family-native pairs. |
| Native b64-to-b16 transpose publication | CDNA4 simulation and physical CDNA4 | The compile probe confirms `ds_read_b64_tr_b16` is not a native gfx942 instruction; gfx942 evidence belongs to the separate CDNA4-to-CDNA3 translation lowering. Wave32 targets use different transpose layouts and require their own E2E-derived contracts. |
| Module and graph lifecycle | All five simulator targets plus physical CDNA4 | Four independently compiled objects remain live on four streams while odd/even subsets are reloaded across three generations; one designated object alone carries the incorrect kernel. All eight initial/reloaded object instances and 32 access sites require complete MOI evidence, and four concurrent Record/Replay reports remain bounded and overflow-free. Two-graph repeated replay and packed graph/executable parameter updates are also portable and green. A larger process-capacity envelope needs distinct E2E evidence rather than another nominal stream-count increase. |

### CDNA3 (`gfx942`)

**Implemented evidence:** 57 pair names produce 536 green RocJitsu rows. In addition
to the 49 common pair names (45 all-engine plus four baseline/one-engine pressure pairs),
`CdnaMfmaPipeline` keeps MFMA/AccVGPR state live across
an LDS publication edge; `FullBankStreamK` combines all 256 low-bank ordinary VGPRs,
a dynamic private frame, shared helpers, cache-qualified VGLOBAL fetch-add,
and last-arriver consumption; `CdnaB96Boundary` pairs native 12-byte tuple
publication with an address/destination alias from PyTorch's rocPRIM kernels;
`AccessHeavyStreamK` executes 80 distinct LDS access sites and five barriers
across five publication phases; and `GroupFlatWide` executes native wide
group-FLAT traffic. `CdnaAccvgprB128` now proves a native B128 load directly
into AccVGPRs while independent accumulators and VCC remain live, and
`LargeTextRelay` transports the generated-object relay geometry from
CDNA4. `CdnaDenseSccRoutes` covers far access/barrier routing under
extreme scalar pressure in the three engines that admit that shape, while
`SubwordSccRoute` covers all four engines with a D128-derived 16-bit store
between an SCC producer and consumer. `SuperColliderDenseThreshold`
crosses the same 1,024-stranded-site production threshold on CDNA3 as on
CDNA4. `CdnaMixedPersistentState` cross-pollinates HIP-matmul's disconnected
fixed-stack/live-AccVGPR and dynamic-stack/full-VGPR owner shape from CDNA4,
with exact correct/incorrect publication behavior rather than a placement
oracle. The common VGLOBAL pair supplies the CDNA cache/wait publication
contract, and the Sampled dual-address pair executes native two-range LDS forms
without charging their banked report slots against the static site limit.
The common `StridedDualAddressPublication` pair carries both the production
adjacent `ds_read2_b64`/`ds_write2_b64` and stride-64
`ds_read2st64_b64`/`ds_write2st64_b64 offset1:8` shapes under a four-value
exact 64-bit oracle instead of leaving them as decoding evidence. The
compiler-generated
kernarg-preload variant of `CdnaMixedPersistentState`
also requires both firmware entry addresses to initialize equivalent private
state for fixed- and dynamic-stack owners.
These reductions came from the hip-moi attention, pressure, and Stream-K rows
in `VALIDATION.md`.

**Defects now guarded:** segmented top-k exposed illegal entry-state backup at
the ordinary-VGPR/AccVGPR boundary, and the dynamic-stack pair exposed missing
SuperCollider native-LDS spill support. Focused host regressions accompany the
green device rows.

**Residual gaps:** `ds_read_b64_tr_b16` is not native gfx942 and therefore is
not a missing native port; further specialized forms require distinct E2E
behavior beyond the now-covered stride-64 dual-address transfer. No
small pair yet combines MFMA, the full ordinary bank, and dynamic stack in one kernel
(those states compete for the unified register file). The common histogram
pair now supplies native FP32 and FP64 LDS atomic payloads, an exact relaxed scalar-BF16
global collision payload, and exact native packed-BF16 global collision sums;
`SortNetworkExchange` transports PyTorch mode's in-place value/index exchange
into a portable two-wave publication contract. The packed-atomic work also
guards generator format preservation and independent global/LDS packed-FP16
and packed-BF16 emulator arithmetic. There is no
physical `gfx942` in this workspace, so physical qualification remains an
external availability gap.

### CDNA4 (`gfx950`)

**Implemented evidence:** 59 pair names contribute 556 green RocJitsu rows. Physical
gfx950 repeats the applicable contracts and contributes 557 rows including
the health check. The compiler-generated
kernarg-preload variant of `CdnaMixedPersistentState` requires equivalent
private-state initialization at both AMDHSA firmware entry addresses under
baseline and Sampled execution.
Alongside the CDNA3 MFMA,
full-bank Stream-K, B96, group-FLAT, and common VGLOBAL contracts, the
CDNA4-only `Gfx950LargeLdsPipeline`
adds a 160-KiB group segment, eight waves, 32 publication stages, wide DS
stores, scalar reads, and repeated dispatch. Exact outputs make clobbering
register state or LDS publication observable. The shared
`CdnaAccvgprB128` and `LargeTextRelay` pairs add direct accumulator
destinations and a
large generated-text shape with an early LDS site beyond direct branch range,
plus a nearby paired publication contract. It executes the relocated wave64
guest instructions without asserting the current placement mechanism. This
abstracts Sharktank, HipKittens, PyTorch generated kernels, and the physical
hip-moi attention/pressure/Stream-K evidence in `VALIDATION.md`.

The all-target `AccessHeavyStreamK` pair additionally preserves 80 static LDS
access sites and five static barriers while the incorrect member uniformly
skips only the central publication barrier. It directly guards the relay-space
competition reduced from the physical two-tile Tensile Stream-K E2E workload.
The two dense-SCC pairs add the corresponding pressure and B16 control-flow
shapes on both backends; the latter specifically guards the Record/Replay
dispatcher bug reduced from hip-moi D128-pressure.
`CdnaMixedPersistentState` retains the HIP-matmul Record/Replay mixed-owner
closure in a quick contract. It keeps live MFMA accumulator state in one
fixed-stack owner and a full-VGPR dynamic stack in a disconnected owner while
checking exact independent results; only the latter owner's publication edge
changes between the pair.
`SuperColliderDenseThreshold` separately crosses SuperCollider's
production 1,024-stranded-site threshold with 1,040 static LDS writes and a
long scalar tail distilled from PyTorch radix sort. Its correct and incorrect
members run under baseline and SuperCollider on all five simulators and
physical gfx950. They pin exact results, diagnostic behavior, and per-target
descriptor growth without making the generalized all-engine fixture
artificially large.
The separate `RepeatedDispatchIdentity` pair remains the portable behavioral
contract for Record/Replay identity across compact repeated launches. The
all-target `StridedDualAddressPublication` pair distills Stream-K's adjacent
`ds_read2_b64` consumer and the Sharktank/Triton stride-64 idiom into four
exact 64-bit values per lane with a real 5.5-KiB group segment; both simulated
and physical CDNA4 execute the two native forms. The gfx950-only
`CdnaTransposePublication` pair additionally distills Triton
attention/matmul's native `ds_read_b64_tr_b16` form and checks the ISA-defined
16-lane transpose exactly in both simulation and physical execution. The
physical `torch.sort` no-evidence result was a validation-cadence issue rather
than new device semantics: a gfx950-only stride-1 policy now qualifies the
four-row E2E schedule, while unit tests prove other targets retain their
production defaults. No prototype-specific device fixture is added for that
harness policy.

**Defects now guarded:** the common and target pairs reproduce the reduced
wide-load spill-window corruption, the AccVGPR-boundary entry-state bug, and
dynamic-frame SuperCollider spill failure. The histogram pair additionally
guards the packed-BF16 global atomic from being misclassified and executed as
a scalar FP32 add; focused generator and memory-pipeline tests cover all
generated packed forms and both packed element formats through global and LDS
collisions. The two-tile pair and its focused
host regression guard the Record/Replay compact-barrier allocator from
consuming seven-word access-pass relay reservations as eight-word generic
islands. The mixed-persistent-state pair and focused host regressions guard
owner discovery, private/register composition, and access/barrier/atomic/fence
lowering after an initially failed dynamic owner is recovered. The large-text
pair and focused host
tests additionally guard wave64 relay scratch preservation and use of skipped
kernels as relay donors when non-text data makes an image large. They pass in
both backends rather than recording the earlier resource rejection as expected.

**Residual gaps:** production-sized Sharktank/CLIP placement and completeness
remain E2E obligations; further specialized LDS forms and one kernel combining
full-bank pressure with live MFMA
are not behavioral pairs. Four-object/four-stream mixed module reload, graph
replay, and both packed parameter-update paths are now covered; only a
distinct E2E-evidenced heavier process-capacity boundary remains a lifecycle
refinement.

### RDNA3 (`gfx1100`)

**Implemented evidence:** 52 pairs produce 502 green RocJitsu rows. Beyond the
49 common pair names, `RdnaWmmaPipeline` cross-pollinates the gfx12 attention
idiom into gfx11 WMMA plus LDS staging, `Gfx11ScalarVglobal` complements the
vector-only common VGLOBAL form with an SGPR-base acquire and a missing-GL0
invalidate negative, and `RdnaB96Boundary` executes gfx11's native aliasing
12-byte load/store form. The common dual-address pair additionally transports
the E2E publication idea through gfx11's native adjacent and stride-64 B64
load/store spellings.
`TopKPrefix` uses wave32 ballot rank, lane broadcast,
atomic-return prefix reservation, paired LDS loads, and exact prefix/payload
results. The common repeated-dispatch pair provides the loaded-image/report
generation contract without claiming that ConSan models inter-kernel races.

**Residual gaps:** there is still no gfx1100 empirical workload survey
comparable to the gfx1201 study, so the relative production importance of the
covered idioms is inferred. `RdnaWmmaPipeline` already combines WMMA, dynamic
private state, shared-helper ownership, and full-bank pressure, while the
portable D16-high, module-lifecycle, and graph-replay pairs now run here. The
portable histogram pair adds an exact FP64 LDS compare/exchange-loop payload
alongside its relaxed scalar-BF16 CAS-loop payload;
the supported gfx1100 compiler/ISA subset has no native packed-BF16
memory-atomic form, so that path is typed N/A rather than untested. A
target-informed empirical workload survey remains absent. Clustered dispatch
and gfx12 instruction-encoded scope are not part of the claimed gfx11 subset
and are not gaps; native B96 is now covered.

### RDNA4 (`gfx1201`)

**Implemented evidence:** 53 pairs produce 512 green RocJitsu rows. The target
extensions deliberately execute group-FLAT; native aliasing B96 loads/stores
with canaries; a returning VGLOBAL atomic whose incorrect member changes only
the instruction-encoded scope; WMMA interleaved with a split-barrier LDS
pipeline; a production-shaped FP8 staged matmul; and the wave32 form of the common Top-K prefix
contract shared with gfx1100/gfx1250. The all-target dual-address pair executes
gfx12's adjacent and stride-64 native VDS B64 spellings plus its modern DS wait
under the same four-value publication oracle used on CDNA3/4 and gfx11. These
are reductions of
the WMMA, production FP8, PyTorch selection, and generated-kernel evidence in
`VALIDATION.md`, not mnemonic-only checks.
The common histogram pair also executes exact FP64 LDS collision sums and
native packed-BF16 global collision sums; the shared generator and
memory-pipeline regressions guard its
packed format and independent half-word arithmetic.

**Residual gaps:** physical gfx1201 evidence remains in the external E2E
campaign rather than this machine's device matrix. `RdnaWmmaPipeline` already
owns the production-shaped FP16 matrix/publication contract; large
`torch.mode`/llama placement pressure remains absent. The
checked-in Top-K pair owns the quick prefix/`ds_bpermute` behavior, while
production-sized fault qualification remains E2E. The common
heterogeneous-object pair now covers cross-kernel attribution and an unexecuted
pressure owner; production-size placement remains an E2E obligation.

### CDNA5 (`gfx1250`) summary

**Implemented evidence:** 64 pairs produce 622 green RocJitsu rows. In addition
to the 49 common pair names, gfx1250 runs the wide group-FLAT contract, shares
native B96 and instruction-encoded atomic scope with gfx1201, and
`TopKPrefix` covers the target's prefix and
last-arriver idiom together with gfx1100/gfx1201. The common dual-address pair
uses CDNA5's adjacent and stride-64 native VDS B64 forms plus `s_wait_dscnt`,
giving the Sharktank/Triton and retained gfx1250 fault-manifest ideas a device
behavioral contract rather than only host/ISA coverage.
`Gfx1250HighBankLdsAddress` covers SRC0-bank address
capture and its correct/missing-barrier behavioral contract; focused host
coverage adds composite-bank and spill-backed resource pressure. Five
exclusive contracts submit
`hsa_amd_ext_kernel_dispatch_packet_t` directly with real cluster dimensions:
two-CTA and four-CTA cluster synchronization, two-cluster identity/isolation,
all B8/B32/B64/B128 `cluster_load_async_to_lds` widths plus one
`s_wait_asynccnt`, and multicast to both CTAs. These widths mirror Composable
Kernel's cluster-load utility and the B32/B64/B128 CK Tile production wrapper;
the correct members compare all 256 delivered values per workgroup exactly.
The wider topology uses one four-CTA cluster rather than two two-CTA
clusters, exposing participant/identity assumptions that the prior topology
could not distinguish. Their correct members verify exact data and control;
their incorrect members remove only the relevant cluster publication edge.
This replaces the previous host-only cluster evidence with actual checked-in device execution
and distills the clustered/TDM E2E work described in `VALIDATION.md`.
`Gfx1250SparseFp8Pipeline` additionally distills Tensile sparse-FP8 staging:
packed low/D16-high byte stores, byte and transposed metadata loads, and a live
`v_swmmac_f32_16x16x128_fp8_fp8` result protected by the paired publication
edge. `Gfx1250ScaledWmmaPipeline` stages packed FP8 operands through LDS,
executes the native four-dword scaled-WMMA pair, and checks all eight exact
matrix result registers. `Gfx1250Fp16WmmaPipeline` transports the RDNA FP16
staging idea through gfx1250's distinct native K=32 instruction and a full
32-byte per-lane fragment. Its correct member checks all eight exact 64.0
accumulators after two WMMA steps; its incorrect member removes only the LDS
publication barrier. `Gfx1250GlobalAsyncToLds` and
`Gfx1250GlobalAsyncFromLds` cover ordinary non-cluster async transfer in both
directions with exact B8/B32/B64/B128 payloads in disjoint LDS regions and an
async-count wait.
`Gfx1250TensorDescriptorPipeline` distills the accepted PyTorch/Triton
tensor-descriptor add row into two native tensor loads, one native tensor
store, three tensor-count waits, and a three-wave exact add pipeline with
head/tail canaries. Its adjacent incorrect member removes only the first
publication barrier; an explicit LDS marker/read keeps the required diagnostic
truthful without claiming that the prototype instruments the TDM descriptor's
entire dynamic range. The transported 160-KiB
pipeline adds eight wave32 producers/consumers, 32 stages, gfx12 DS spellings,
and a wider LDS address. A direct host emulator unit pins the D16-high selector
and offset. The common histogram pair additionally executes native FP64 LDS
and packed-BF16 global collision sums, with shared generator and
global/LDS memory-pipeline regressions guarding format selection and
independent half-word arithmetic.

**Residual gaps:** the synchronous `cluster_load_b32/b64/b128` family reads
global address space and returns a workgroup-broadcast result; unlike the
already-covered asynchronous B8/B32/B64/B128 family, it does not access LDS.
General global-memory race detection is outside ConSan's stated model, so this
is a capability-based not-applicable disposition rather than a missing device
pair. There is no physical gfx1250 in this workspace, so all such device
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
4. **Complete:** histogram-style LDS atomics, collision-heavy integer,
   scalar-BF16, and applicable native packed-BF16 global scatter atomics, and
   language-level release-store/acquire-CAS publication. Packed-FP16/BF16
   generator selection and global/LDS collision arithmetic have direct host
   regressions.
5. **Complete:** segmented value/index top-k with tails and two-stage softmax
   with a global intermediate.
6. **Complete:** backward/optimizer, MoE routing, and continuous-batching pairs
   distilled from Aorta.
7. **Complete:** target-specific MFMA/WMMA, AccVGPR, full-bank pressure,
   group-FLAT, VGLOBAL/cache sequence, B96, atomic scope, gfx11 identity, and
   gfx1250 cluster/TDM/multicast/high-bank/double-barrier and sparse-FP8 SWMMAC
   pairs. The 2026-08-23 audit additionally completes portable Stream-K,
   D16-high, module-lifecycle, kernarg-preload, mixed-owner, and long-range SCC
   transport; four-object/four-stream mixed module reload, independent
   two-stream graph replay, and packed graph/executable parameter updates;
   Top-K prefix, full-bank, large-text, subword-SCC, clobbering-load, and
   engine-scoped pressure transport across all five targets; gfx950 transpose;
   native adjacent and stride-64 dual-address B64 publication on every target;
   a real
   four-CTA gfx1250 cluster topology and native 32-byte FP16-WMMA fragment;
   the 160-KiB pipeline on gfx1250; native B96 on gfx1100; the RCCL reusable
   partial barrier; and gfx1250 async transfer in both directions, native
   tensor-descriptor staging, plus scaled-WMMA staging. Every implementation
   failure they exposed was fixed
   and retained as a regression.
8. **Next only when evidence justifies it:** a gfx1100 E2E/ISA survey and
   further uncommon LDS or atomic forms with a truthful fault contract.
   CDNA3 LDS-to-AccVGPR is already covered by `CdnaAccvgprB128`. Synchronous
   `cluster_load_b*` is typed not-applicable because it reads global memory and
   returns a broadcast result rather than accessing LDS. Do not add a
   `ds_swizzle` race pair: it likewise has no LDS publication semantics.
9. **Complete at the current evidenced envelope:** four simultaneously live
   module objects and streams, alternating odd/even reload subsets, complete
   evidence for all eight object instances, a single designated racy module,
   and bounded overflow-free concurrent reports. Repeated two-graph replay and
   both packed parameter-update paths are also complete. Add a larger process
   capacity or lifecycle schedule only when an E2E workload establishes a
   distinct boundary.

Do not consider an area complete because one architecture happens to lower a
generic HIP fixture to the desired instruction. Record the applicable semantic
and ISA classes for each pair and verify them on every target where they exist.

## Completion assessment

The portable and target-specific tier is fully qualified in one combined
3,285-row invocation, including the cross-target pressure transports,
RCCL-style partial barrier, adjacent/stride-64 dual-address B64 publication, gfx1250
LDS-to-global transfer, native tensor-descriptor staging, runtime-indexed LDS
aliasing, deterministic state replay, adjacent-subword and Stream-K behavior,
four-object mixed-reload lifecycle, graph replay and parameter update,
kernarg-preload, mixed-owner, long-range SCC, matrix, transpose, cluster, and
async-transfer additions from the transport audit. Forty-five common
all-engine pairs plus four common engine-scoped pairs and family/target pairs
cover the main synchronization, atomic, pipeline, selection/reduction,
resource, cluster/TDM, and repeated-dispatch idioms distilled from
`VALIDATION.md`, Aorta, and selected RCCL device primitives. Every pair runs
through the baseline and each engine declared semantically applicable in
RocJitsu; CDNA4 repeats the same coverage on the physical `gfx950`. The latest
complete `ctest -j64 -L consan-device` qualification passes all 2,728 simulator
and 557 physical rows without expected-failure exemptions. Its 225.92-second
warm-run wall time remains below
the 5--20-minute review heuristic, so the evidence-driven residual-gap audit
remains open even though the matrix is green. Exact timing and process-duration
evidence is recorded above.

This materially shrinks, but cannot eliminate, regression risk. The remaining
gaps are narrower uncommon instruction forms, production-size placement,
missing physical hardware for four targets, and heavier process-level
lifecycle/report-capacity boundaries not represented by the current
four-object envelope. Keep those explicit and add them from
E2E ISA evidence or a concrete failure, not from prototype-specific patch
expectations or a desire to make the wall clock longer.

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
