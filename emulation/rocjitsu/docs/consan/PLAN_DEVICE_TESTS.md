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
external model assets, and independent of the prototype implementation that
Part 3 will replace.

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
| `tests/dbi/consan/device/` | The checked-in behavioral-conformance tier. Seventeen common `*_test.hip` pairs cover the portable workload abstractions. Adjacent family/target sources cover CDNA MFMA/full-bank/AccVGPR, group-FLAT, RDNA B96/WMMA/FP8, gfx11 VGLOBAL/lifecycle identity, and gfx1250 clustered/TDM behavior. The gfx1250 cluster host fixture submits a real extended HSA dispatch packet rather than hiding cluster dimensions behind an ordinary HIP launch. | Add one descriptively named source per new semantic scenario. Prefer extending a scenario with another tightly related pair over creating broad clean/racy grab bags. |
| `tests/dbi/consan/device/consan_device_test_support.h` | Small shared fixture utilities used by the paired HIP sources. | Put only genuinely reusable fixture mechanics here; keep scenario semantics and expected results local to each test source. |
| `tests/dbi/consan/hip_consan_{lds,moi,moi_cdna,moi_rdna3,inline_shadow,spill_gfx950}_test.hip` and related support | Older target- or engine-specific device fixtures. They provide useful ISA and implementation test material but are not the common behavioral conformance tier. | Distill behaviorally justified idioms into the new paired suite. Keep implementation-specific assertions here when they remain useful, but do not count them as substitutes for portable correct/incorrect contracts. |
| `tests/consan/CMakeLists.txt` | Builds target-specific HIP executables and registers every baseline/engine/backend row, labels, diagnostic requirements, simulator configuration, physical resource lock, and health dependency. It is included by `tests/CMakeLists.txt`. | Register every new pair here across the common matrix. As the table grows, move the declarative pair inventory and registration helpers into a focused `tests/consan/device_tests.cmake` included from this file rather than duplicating per-architecture blocks. |
| `tests/consan/run_checked_test.cmake` and `tests/consan/check_capability_manifest.cmake` | Shared process/output and capability checks used by ConSan tests. | Add only generic fail-closed harness behavior here; scenario semantics and result oracles belong with the HIP workload. |
| `configs/gfx942_cdna3_kmd.json`, `configs/gfx950_cdna4_kmd.json`, `configs/gfx1250.json`, `configs/gfx1100_w7900.json`, and `configs/gfx1201_r9700.json` | RocJitsu simulator descriptions selected by the CTest registrations. | Reuse these for the suite. Change them only for a genuine target-model correction, never to encode test-specific behavior. |

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
survive the Part 3 replacement of the prototype implementation.

Baseline execution remains useful for both variants: it proves that the code
object launches and that the independent control oracle works. An
uninstrumented incorrect workload is not expected to diagnose itself or to
produce a deterministic value from the deliberately racy data path.

## Current state

The portable tier now has 17 adjacent correct/incorrect pairs. Each common
pair runs as baseline plus all four engines on five RocJitsu targets and
physical `gfx950`, for 60 rows per scenario. Fourteen additional pair names
cover family- or target-specific behavior on only the architectures where the
form exists. The union is 31 behavioral pairs and the registered matrix is
1,281 tests, including the physical post-instrumentation health row.

| Scenario | Workload-derived contract | Status |
| --- | --- | --- |
| Ordered tile handoff | Cross-wave LDS publication and exact consumer values | Paired and green |
| Reduction | Tree-style LDS reduction with one missing publication edge | Paired and green |
| Shared helper owners | One noinline helper reached from two kernel owners | Paired and green |
| Independent workgroups | Three-dimensional dispatch identity without false LDS aliasing | Paired and green |
| Dynamic private stack | Private-stack state combined with ordered or unordered LDS traffic | Paired and green |
| Overlapping subwords | Adjacent non-overlap versus true byte-range overlap | Paired and green |
| Atomic arrival | Atomic counter publication followed by shared-data consumption | Paired and green |
| Stream-K last arriver | Partial publication, fetch-add arrival, and last-arriver consumption | Paired and green |
| Tree atomic-OR | Bitmask publication, polling, and completion | Paired and green |
| Double-buffered pipeline | Wide LDS stages, storage reuse, lane exchange, and a missing stage edge | Paired and green |
| Histogram/scatter | LDS-bin atomics and collision-heavy global scatter updates | Paired and green |
| Segmented top-k | Value/index tuples, segmented phases, wide traffic, and non-multiple tails | Paired and green |
| Two-stage softmax | Max/sum-style local reduction plus a global intermediate | Paired and green |
| Backward/optimizer | FP16 gradient traffic, FP32 reduction/moment state, and AdamW-like update | Paired and green |
| MoE routing | Top-1 routing, empty experts, uneven tails, prefix offsets, and indexed gather/scatter | Paired and green |
| Continuous batching | Changing active rows, masked tails, repeated reset-state dispatch, and stable checksums | Paired and green |
| VGLOBAL cache publication | Target-native release/acquire atomic publication and required cache sequencing | Paired and green |

The target-specific tranche adds CDNA MFMA/AccVGPR liveness, full-bank dynamic
Stream-K, native B128-to-AccVGPR, wide group-FLAT, B96 aliasing boundaries,
instruction-encoded atomic scope, WMMA and FP8 staging, both gfx11 VGLOBAL
address forms, repeated multi-stream image identity, and four real clustered
gfx1250 dispatch contracts covering cluster barriers, multi-cluster isolation,
direct-to-LDS async load/wait, and multicast.

This expansion exposed and retained two more real CDNA defects. Inline Shadow
could place an entry owner/epoch backup through live AccVGPR-backed storage in
the segmented-top-k object. SuperCollider could not instrument a native-LDS
site while a dynamic private frame was active. The fixes now select legal
ordinary-register storage below the accumulator boundary and use an existing
bracket-local dynamic-stack spill frame, respectively; focused host tests
protect both resource contracts.

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
behavior-first contracts that can survive the Part 3 replacement.

### Coverage gaps

| Area | Current checked-in evidence | Residual gap |
| --- | --- | --- |
| Synchronization faults | Every scenario is shape-matched and paired; reductions, helpers, dynamic stacks, pipelines, routing, batching, cache sequence, and instruction scope all change one real property. | A dedicated wrong-LDS-address pair remains a useful refinement. |
| Stream-K publication | Fetch-add last-arriver and tree atomic-OR pairs cover partial publication and completion. | `ds_bpermute` broadcast and a closer relaxed-poll/final-RMW tree remain target-specific refinements. |
| Multi-stage pipelines | Double buffering, storage reuse, wide LDS, lane exchange, two-stage softmax, CDNA MFMA, RDNA WMMA, FP8 staging, and repeated dispatch are covered. | Deeper production barrier schedules and a separate FP16 production reduction remain pressure work. |
| Selection and reduction | Segmented value/index top-k, tails, two-stage softmax, and the MoE router cover multi-phase selection/reduction shapes. | Signed/unsigned D16-high variants and a sort-network-specific pair remain optional refinements. |
| Atomic diversity | Arrival add, tree OR, LDS histogram collisions, global scatter collisions, MoE bins, cache-qualified VGLOBAL publication, and gfx12 instruction-scoped atomics are paired. | CAS, atomic loads, and BF16/FP32 payload variants remain absent. |
| LDS and data movement | The suite exercises subwords, B32/B96/B128, group-FLAT wide traffic, B128-to-AccVGPR, and target-native load/store spellings. | Two-address/stride, transpose, D16-high, `ds_swizzle`, and `ds_bpermute` remain mostly in older implementation fixtures. |
| TDM and clusters | gfx1250 now uses real extended clustered-dispatch packets for two-CTA cluster barriers, two-cluster identity/isolation, `cluster_load_async_to_lds_b32` plus `s_wait_asynccnt`, and multicast. | Store-from-LDS, wider tensor fragments, scale-WMMA, more than two CTAs per cluster, and a distinct remote cluster-memory opcode remain unabstracted. |
| Resource and control pressure | Dynamic stack, shared helpers, MFMA/WMMA live state, CDNA AccVGPR destinations, full ordinary-VGPR-bank pressure, B96/B128 aliasing, native VGLOBAL forms, descriptor growth, and repeated dispatch are exercised. The tranche caught both AccVGPR-boundary and dynamic-frame spill defects. | Combined worst-case forms and production-sized placement/relay limits remain E2E or focused implementation-test responsibilities. |
| Object and dispatch shape | Shared helpers have multiple kernel owners; softmax uses multiple stages and a global intermediate; continuous batching repeats changing dispatches from reset state. | Multiple loaded objects, unload/reload, multi-stream module launch, and graph replay remain lifecycle envelopes rather than covered semantic categories. |
| Scale | The 1,281-row matrix crosses 31 pair names, five simulator targets, physical CDNA4, and every engine while staying checked in and bounded. | Full large-object relay and capacity boundaries remain E2E responsibilities; add a medium heterogeneous-object pair only if a concrete failure mode justifies its cost. |

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
| Heterogeneous generated objects | Combine small attention/reduction, MoE routing, optimizer, and atomic-publication kernels with unexecuted high-pressure helpers in one or more objects. The incorrect variant changes one executed kernel only. | Exact results for every dispatched kernel, proof of which entries ran, no false attribution to unexecuted functions, and bounded complete reporting. This exercises the object shape seen in PyTorch without checking in a large model artifact. |

The first Aorta-derived tranche--backward/optimizer, MoE routing, and continuous
batching with repeated reset-state dispatch--is complete across the whole
matrix. The next justified Aorta work is a selected RCCL-kernel ISA survey,
then multi-object and module/graph lifecycle envelopes around existing pairs.
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
selected across all applicable architectures. One pair/backend application is
ten rows: correct and incorrect under the baseline and four ConSan flavors.
The 17 common pairs contribute 1,020 rows across the five RocJitsu targets plus
physical `gfx950`. The non-rectangular target extensions contribute 260 rows,
and the physical health check contributes one, for 1,281 total.

The per-configuration arithmetic is:

| Configuration | Pairs | Rows |
| --- | ---: | ---: |
| RocJitsu `gfx942` | 20 | 200 |
| RocJitsu `gfx950` | 21 | 210 |
| Physical `gfx950` | 21 | 211, including health |
| RocJitsu `gfx1100` | 21 | 210 |
| RocJitsu `gfx1201` | 22 | 220 |
| RocJitsu `gfx1250` | 23 | 230 |

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

The current 1,281-test matrix passed in **67.91 seconds** on this host at
`-j64`. Measured aggregate process CPU time was **3,955.54 seconds
(65m55.54s)**: 861.05s user and 3,094.49s system. CTest's summed test duration
was 4,314.16s. Use wall-clock latency for the budget and retain CPU time as a
separate capacity metric.

The wall time is below the five-minute review threshold. The review does not
justify artificial delays: the matrix already contains 1,281 independently
scheduled rows and roughly 66 CPU-minutes of work, so the unusually wide host
is the main cause. It does justify keeping the residual architecture and
lifecycle gaps below visible. Add tests for those gaps when grounded in E2E
ISA or a concrete failure, not merely to consume five minutes.

## Issues exposed and retained as regressions

The current matrix has no red cells. Building it exposed defects in the
prototype, runtime, and simulator, all fixed without weakening the behavioral
contracts:

- `gfx942` and `gfx950` Record/Replay multidimensional workgroup identity;
- `gfx942` and `gfx950` Sampled reduction snapshot formation;
- `gfx1100` Inline Shadow atomic-arrival evidence;
- `gfx1250` Inline Shadow 3D-workgroup handling around the RocJitsu `HW_ID1`
  topology limitation;
- physical `gfx950` Record/Replay, Sampled, and Inline Shadow failures;
- non-returning DS-atomic emulation and Sampled atomic-watchpoint decoding;
- CDNA3/CDNA4 Inline Shadow recovery of a clobbered wide-load address outside
  a reduced spill window;
- CDNA Inline Shadow owner/epoch backup selection at an ordinary-VGPR/AccVGPR
  boundary; and
- CDNA SuperCollider native-LDS instrumentation under an active dynamic
  private frame.

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

### CDNA3 (`gfx942`)

**Implemented evidence:** 20 pairs produce 200 green RocJitsu rows. In addition
to the 17 common pairs, `CdnaMfmaPipeline` keeps MFMA/AccVGPR state live across
an LDS publication edge; `CdnaFullBankStreamk` combines all 256 ordinary VGPRs,
a dynamic private frame, shared helpers, cache-qualified VGLOBAL fetch-add,
and last-arriver consumption; and `GroupFlatWide` executes native wide
group-FLAT traffic. The common VGLOBAL pair supplies the CDNA cache/wait
publication contract. These reductions came from the hip-moi attention,
pressure, and Stream-K rows in `VALIDATION.md`.

**Defects now guarded:** segmented top-k exposed illegal entry-state backup at
the ordinary-VGPR/AccVGPR boundary, and the dynamic-stack pair exposed missing
SuperCollider native-LDS spill support. Focused host regressions accompany the
green device rows.

**Residual gaps:** transpose and dual-range LDS forms remain host/ISA evidence;
there is no behavioral LDS-to-AccVGPR destination pair on CDNA3; and no small
pair yet combines MFMA, the full ordinary bank, and dynamic stack in one kernel
(those states compete for the unified register file). CAS, floating atomic
payload variants, and a production sort network are also absent. There is no
physical `gfx942` in this workspace, so physical qualification remains an
external availability gap.

### CDNA4 (`gfx950`)

**Implemented evidence:** 21 pairs run twice--210 RocJitsu rows and 210
physical rows--followed by a physical health check. Alongside the CDNA3 MFMA,
full-bank Stream-K, group-FLAT, and common VGLOBAL contracts, the CDNA4-only
`CdnaAccvgprB128` pair performs a native B128 LDS load directly into AccVGPRs
while separate accumulators and physical VCC remain live. Exact outputs make
clobbering either register class observable. This abstracts Sharktank and the
physical hip-moi attention/pressure/Stream-K evidence in `VALIDATION.md`.

**Defects now guarded:** the common and target pairs reproduce the reduced
wide-load spill-window corruption, the AccVGPR-boundary entry-state bug, and
dynamic-frame SuperCollider spill failure. They pass in both backends rather
than recording the earlier resource rejection as expected.

**Residual gaps:** production-sized Sharktank/CLIP placement and completeness
remain E2E obligations; transpose/dual-range LDS, CAS and floating atomic
payloads, and one kernel combining full-bank pressure with live MFMA are not
behavioral pairs. The checked-in suite also does not replace multi-object,
module/graph, or unload/reload lifecycle validation.

### RDNA3 (`gfx1100`)

**Implemented evidence:** 21 pairs produce 210 green RocJitsu rows. Beyond the
17 common pairs, `GroupFlatWide` supplies the shared group-FLAT byte/numerical
oracle, `RdnaWmmaPipeline` cross-pollinates the green gfx12 attention idiom into
gfx11 WMMA plus LDS staging, `Gfx11ScalarVglobal` complements the vector-only
common VGLOBAL form with an SGPR-base acquire and a missing-GL0-invalidate
negative, and `Gfx11DispatchIdentity` runs two rounds on four streams to test
loaded-image/report generation isolation. The latter is explicitly a lifecycle
envelope, not a claim that ConSan models inter-kernel races.

**Residual gaps:** there is still no gfx1100 empirical workload survey
comparable to the gfx1201 study, so the relative production importance of the
covered idioms is inferred. A single pair does not yet combine WMMA, dynamic
private state, shared-helper ownership, and maximum placement pressure. D16-high,
swizzle/permute, CAS/floating atomics, multi-object unload/reload, and graph or
module launch remain untested. B96, clustered dispatch, and gfx12
instruction-encoded scope are not part of the claimed gfx11 subset and are not
gaps.

### RDNA4 (`gfx1201`)

**Implemented evidence:** 22 pairs produce 220 green RocJitsu rows. The target
extensions deliberately execute group-FLAT; native aliasing B96 loads/stores
with canaries; a returning VGLOBAL atomic whose incorrect member changes only
the instruction-encoded scope; WMMA interleaved with a split-barrier LDS
pipeline; and a production-shaped FP8 staged matmul. These are reductions of
the WMMA, production FP8, PyTorch selection, and generated-kernel evidence in
`VALIDATION.md`, not mnemonic-only checks.

**Residual gaps:** physical gfx1201 evidence remains in the external E2E
campaign rather than this machine's device matrix. A separate production-shaped
FP16 pair, large `torch.mode`/llama placement pressure, refreshed top-k fault
qualification, swizzle/permute forms, and a medium heterogeneous object are not
checked in. Add a medium object only after reproducing a concrete large-object
failure cheaply; do not inflate the suite for size alone.

### CDNA5 (`gfx1250`) summary

**Implemented evidence:** 23 pairs produce 230 green RocJitsu rows. In addition
to the 17 common pairs, gfx1250 shares native B96 and instruction-encoded atomic
scope with gfx1201. Four exclusive contracts submit
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
4. **Complete:** histogram-style LDS atomics and collision-heavy global
   scatter atomics.
5. **Complete:** segmented value/index top-k with tails and two-stage softmax
   with a global intermediate.
6. **Complete:** backward/optimizer, MoE routing, and continuous-batching pairs
   distilled from Aorta.
7. **Complete:** target-specific MFMA/WMMA, AccVGPR, full-bank pressure,
   group-FLAT, VGLOBAL/cache sequence, B96, atomic scope, gfx11 identity, and
   gfx1250 cluster/TDM/multicast pairs. Every implementation failure they
   exposed was fixed and retained as a regression.
8. **Next only when evidence justifies it:** a gfx1100 E2E/ISA survey, CDNA3
   LDS-to-AccVGPR, gfx1250 store-from-LDS/wider fragments/scale-WMMA, a separate
   RDNA4 FP16 reduction, and uncommon transpose/permute or atomic forms.
9. **Remaining lifecycle enhancement:** bounded multi-object, module/graph,
   unload/reload, and report-capacity envelopes around existing semantic pairs.

Do not consider an area complete because one architecture happens to lower a
generic HIP fixture to the desired instruction. Record the applicable semantic
and ISA classes for each pair and verify them on every target where they exist.

## Completion assessment

The Part 1 device-test goal is now met at both portable and target-specific
levels. Seventeen common and fourteen family/target pair names cover the main
synchronization, atomic, pipeline, selection/reduction, resource, cluster/TDM,
and repeated-dispatch idioms distilled from `VALIDATION.md` and Aorta. Every
applicable pair runs through the baseline and all four engines in RocJitsu;
CDNA4 repeats the same coverage on the physical `gfx950`. All 1,281 rows pass
without expected-failure exemptions.

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
  independent of the prototype implementation that Part 3 will replace.
- The single whole-matrix CTest invocation passes with no expected-failure
  exemptions, including for issues that predated this device-test work.
