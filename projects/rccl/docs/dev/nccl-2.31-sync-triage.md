# NCCL 2.31 to RCCL sync triage

Gap analysis for syncing RCCL from NCCL 2.30.7 to NCCL 2.31.2.

This is an internal engineering document. It is deliberately not listed in
`docs/sphinx/_toc.yml.in`, so it does not publish to rccl.readthedocs.io.

| Field | Value |
| --- | --- |
| Baseline | RCCL 2.30.7 (`ROCm/rocm-systems` `develop`, `makefiles/version.mk`) |
| Target | NCCL `v2.31.2-1` (released 2026-08-11) |
| Upstream range | `v2.30.7-1..v2.31.2-1` |
| Size | 457 commits, 722 files, +97,411 / -25,391 |

## How to read this document

Every upstream item gets one verdict:

| Verdict | Meaning |
| --- | --- |
| **Landed** | Already present at the 2.30.7 baseline. Cited file proves it. |
| **Port** | Applies to AMD. Bring the upstream change over largely as-is. |
| **Adapt** | Applies conceptually, but the upstream implementation is CUDA-bound. Needs a HIP/ROCm equivalent. |
| **Skip** | No AMD hardware or software counterpart. The reason is recorded because it feeds the CHANGELOG and PR body. |
| **Defer** | Applies, but out of scope for this sync. Reason given. |

Upstream commits are cited by short SHA against `NVIDIA/nccl`. RCCL paths are
relative to `projects/rccl/`.

## Where the work actually is

Roughly half of the upstream source churn is unreachable on AMD hardware:

| Slice | Churn (lines changed) |
| --- | --- |
| `src/` total | 43,429 |
| of which GDAKI / DOCA / EFA / CFT / NVLS / CuTe paths | 23,190 |
| `contrib/` (samples and side projects, not the library) | 57,754 |
| `bindings/` (nccl4py, nccl4rust) | 11,612 |

Within `src/transport/net_ib/`, 14,791 of 16,649 changed lines are under
`gdaki/` and its vendored `doca-gpunetio` tree. The portable IB work is only
about 1,858 lines, concentrated in `wqe_lat_mon.cc`, `p2p_resiliency*.cc`,
`connect.cc` and `common.{cc,h}`.

The practical consequence: the headline features of the release (Compute Fabric
Transport, EFA GDA, PAT+NVLS) are almost entirely non-portable, while the items
that matter for RCCL are the cross-cutting API and infrastructure changes that
the release notes mention almost in passing.

## The structural refactor is the main risk

2.31 dissolves three monoliths that RCCL has customized heavily. This, not any
single feature, is what makes the sync expensive.

| Upstream 2.30.7 | Upstream 2.31.2 | RCCL exposure |
| --- | --- | --- |
| `src/graph/tuning.cc` (deleted) | `src/tuning/` (13 files, incl. `cost_model.cc`, `pat.cc`, `ring.cc`, `tree.cc`, `sym_model.cc`, `ce_model.cc`) | RCCL keeps hand-tuned constants and `gfx`-specific overrides here; commit `516ab80831` in the 2.28.9 sync exists solely to preserve them |
| `src/enqueue.cc` | `src/enqueue/` (`enqueue.cc`, `mgmt_task_enq.cc`, `task_prep/`, `task_sched/`) | RCCL's RMA validation, CE dispatch and `CTA_POLICY` gating all live in `src/enqueue.cc` |
| (none) | `src/config/` (`collconfig.cc`, `algorithm_parser.cc`, `algorithm_registry.cc`) | New tree, no RCCL conflict |
| (none) | `src/diagnostics/` (`p2p.cc`, `device/`) | New tree, no RCCL conflict |
| `src/transport/profiler.cc` (deleted) | folded into per-communicator profiler thread | RCCL added proxy-diagnostics and proxytrace hooks around this file |

The upstream "Enqueue Overhaul" (`a62c20e1`) and the tuning split are both pure
refactors from NCCL's perspective but land directly on RCCL's largest local
divergences. Treat these as their own work item, sequenced before the feature
ports, and expect this to dominate the conflict budget.

## Per-collective configuration and tuning

The single most valuable item in the release for RCCL, and the one upstream
explicitly designed for downstream forks.

| Item | Verdict | Notes |
| --- | --- | --- |
| `ncclCollConfig_t` and eight `nccl*Config()` collective entry points | **Port** | `75b572da`, `127ab70e`. `AllReduce`, `Broadcast`, `Reduce`, `AllGather`, `ReduceScatter`, `Alltoall`, `Gather`, `Scatter`. Absent in RCCL; only communicator-scoped `ncclConfig_t` exists in `src/nccl.h.in` |
| `ncclConfigExt_t` vendor extension list | **Port** | Upstream added a `vendorId`/`optionId` key-value list precisely so forks can carry private options without diverging the ABI. RCCL should claim a `vendorId` and migrate existing RCCL-only knobs onto it |
| Per-collective algorithm selection strings | **Port** | `src/config/algorithm_parser.cc`, `algorithm_registry.cc`. Needs RCCL addon algorithms (`RCCL_HIERARCHICAL_ALLGATHER`, `RCCL_HIERARCHICAL_REDUCESCATTER`) registered alongside upstream ones |
| Per-collective CTA/CGA and `CTAPolicy` override | **Adapt** | `cgaClusterSize` is a Hopper+ cluster-dimension launch attribute with no HIP equivalent; accept and ignore it rather than rejecting it, so portable applications still run |
| `userProfilerTag` plumbed to profiler | **Port** | `69e7c2c0`, `0dab2642`, `deb1f27a`. Depends on profiler v7 below |
| Unified cost model (`src/tuning/cost_model.{cc,h}`, `src/include/tuning.h`) | **Adapt** | Replaces `ncclTopoGetAlgoTime` plus the separate `sym_kernels.cc` `queryModel_*` path with one `ncclTuningInput_t`/`ncclTuningResult_t` interface covering legacy, symmetric and CE methods. RCCL's CSV tuner and `rccl_wrap.cc` overrides must be re-expressed against it |
| TMA kernels enabled by default and costed | **Skip** | RCCL pins `NCCL_PARAM(SymTmaEnable, "SYM_TMA_ENABLE", 0)` in `src/sym_kernels.cc`; the cost-model hooks come along with the refactor but stay inert |
| Algorithm selection on newer Intel CPUs | **Defer** | Bandwidth-table tuning only. Low value on AMD platforms; revisit if Intel-host MI systems matter |
| AMD EPYC topology modeling (`#2036`) | **Port** | Community contribution that improves AMD host modeling. RCCL currently classifies AMD CPUs only as `ZEN`/`ROME` in `src/graph/topo.cc`. Directly relevant despite arriving via upstream |

### Parallel Aggregated Tree

| Item | Verdict | Notes |
| --- | --- | --- |
| Hierarchical PAT using NVLS intra-node, PAT inter-node | **Skip** | `cfe95ab9`, `f39eddf5`. Depends on NVLS multicast; RCCL's `src/transport/nvls.cc` gates on `cuMulticastCreate`, which does not resolve on MI300/MI350 |
| PAT restricted to 1 GPU per node | **Landed** | `ncclPatEnable()` in `src/graph/tuning.cc:1106` already enforces `comm->nNodes != comm->nRanks -> 0` |

RCCL's hierarchical AllGather/ReduceScatter is a separate mechanism from
upstream's hierarchical PAT: it splits the communicator into intra- and
inter-node sub-communicators and runs a shuffle kernel, rather than layering
NVLS under PAT. The upstream design is not a drop-in replacement, and the two
should not be conflated when the tuning refactor lands.

## Diagnostics and profiling

The highest-value portable feature cluster in the release.

| Item | Verdict | Notes |
| --- | --- | --- |
| `NCCL_RUN_RAS_DIAGNOSTICS` framework | **Port** | `0cd65b68`, `eee30fa7`. New files `src/ras/diagnostics.cc`, `diagnostics_checks_common.cc`, `ras_param.cc`. RCCL's `src/ras/` has none of these |
| GPU inventory / model consistency check | **Adapt** | `0f1c44d4`. Upstream `diagnostics_gpu.cc` calls `ncclNvmlDeviceGetCount` / `GetHandleByIndex` / `GetName`. RCCL already has an AMDSMI wrapper layer (`src/misc/amdsmi_wrap.cc`, `alt_rsmi.cc`) to substitute |
| Driver version check | **Adapt** | `c438d81d`. Report HIP/ROCm versions, which RCCL already surfaces in `NCCL_DEBUG` output |
| Volatile ECC counter check | **Adapt** | `bf816d82`. Needs an AMDSMI ECC equivalent; `NCCL_DIAGNOSTICS_ECC_THRESHOLD` semantics carry over unchanged |
| Per-NVLink operational state check | **Adapt** | `acd1acd2`. Map to XGMI link state via AMDSMI. Keep the check, rename the user-facing wording away from "NVLink" |
| `NCCL_*` environment consistency check | **Port** | `f1431997`. Pure host logic, no vendor coupling. Genuinely useful for RCCL support cases |
| `NCCL_RUN_DIAGNOSTICS` P2P connectivity check | **Port** | `ff70ca13`, `0e87888b`, new `src/diagnostics/p2p.cc` (961 lines) plus `src/diagnostics.cc`. Active P2P probe with actionable remediation text |
| RAS CONTROL command to set profiler event mask job-wide | **Port** | `d26aae37` |
| Profiler v7 plugin API | **Port** | `716db255`, `996e0e49`. RCCL is on v6 (`src/include/plugin/nccl_profiler.h:98`). RCCL's proxy-diagnostics extension (`ncclProfilerProxyDiagEnabled`, `src/include/profiler.h:111`) must be rebased onto the v7 descriptor |
| Kernel-channel profiling on a dedicated per-communicator thread | **Adapt** | Upstream deletes `src/transport/profiler.cc` and moves polling off the shared proxy service thread, extending coverage to proxy-less and graph-captured collectives. RCCL still has that file, with proxytrace instrumentation wired through it |
| Symmetric-kernel phase events (`initial_sync`, `compute`, `final_sync`) | **Port** | Comes with v7; RCCL has symmetric kernels on gfx942/gfx950 that would benefit |
| Per-QP CPU WQE post-to-poll latency monitor | **Port** | `6795554a`, `8ace492b`, `cbc8911e`. New `src/transport/net_ib/wqe_lat_mon.{cc,h}` (447 lines). RCCL has a comparable RTT measurement but only inside the `ib-cast` scheduler, so this is consolidation |

## Transport and topology

| Item | Verdict | Notes |
| --- | --- | --- |
| Dynamic topology path link arrays | **Port** | `4be8abcb`. Changes `ncclTopoLinkList.list` from `ncclTopoLink*[NCCL_TOPO_MAX_HOPS]` to a grown-on-demand `ncclTopoLink**` with a `capacity` field. RCCL still carries the fixed 512-entry array (`src/graph/topo.h:88`), about 4.1 KiB per path entry. Architecture-neutral memory and init-time win |
| `IBV_EVENT_GID_CHANGE` handling and device-wide GID cache | **Port** | `35d4326d`, `c4b77f00`, `fa09ae42`, `5daa5b74`, `99152cd6`. RCCL currently only warns and returns (`src/transport/net_ib/common.cc:156`) |
| IB port recovery `ATTEMPTS_MAX` 5 to 20 | **Port** | `8b1d0d57`. One-line robustness fix on a path RCCL shares |
| `NCCL_IB_PKEY_VALUE` | **Port** | `a31f9987` |
| Event-based load balancing for `net_ib` | **Port** | `8a743dd0`. RCCL has RTT-weighted scheduling only in `net_ib_cast`; unifying reduces long-term divergence |
| Consistent start-NIC limited to Blackwell + ConnectX-8 | **Skip** | `696be2a5`. The gate itself is the fix; on AMD the condition is never true. Record so a future reader does not re-port it |
| `topo: add 90.2 to favor 8 channels for GB300 + PXN=0` | **Skip** | `7e3320c7`. GB300-specific constant |
| PXN `netId` vs `netDev` fix at `P2P_PXN_LEVEL=1` | **Port** | `4ef6b6a5`. Fixes init failures on multi-system topologies. RCCL supports `NCCL_P2P_PXN_LEVEL` and multi-system NPS nodes, so this is live |
| PXN teardown race with connection init | **Port** | Hang fix, portable |
| `transport/net.cc` `ssize_t` widening | **Port** | `6941d94b`. Integer overflow on large transfers |
| Transport selection for trimmed peers | **Port** | `869feafb` |
| GIN/RMA vendor+device PCI entries in topo | **Port** | `b13d895f` |
| Warn when GIN and NET device indices differ | **Port** | `fc914fd4` |
| CUDA 13.3 DMA-BUF mmap backend for GDRCopy | **Skip** | Internal CUDA backend. RCCL uses `hsa_amd_portable_export_dmabuf` via `src/misc/rocmwrap.cc` |
| `p2p: tune channels for registered NET buffers` | **Port** | `c3fa2e46`, portable channel tuning |

## GIN, RMA and symmetric memory

Most of the GIN work in 2.31 targets backends RCCL does not build. RCCL's GIN
backends at 2.30.7 are the host proxy, Anvil SDMA and rocSHMEM GDA
(`src/gin/`), with GDAKI compiled out on HIP by
`NCCL_GIN_GDAKI_ENABLE 0` (`src/include/nccl_device/gin/gin_device_common.h:34`).

| Item | Verdict | Notes |
| --- | --- | --- |
| EFA GDA backend (AWS, ~30 commits) | **Skip** | New `src/transport/net_efa_gda/` (1,228 lines) plus `nccl_device/efa_gda`. No AMD counterpart |
| GDAKI: FD reduction, event-based CQ errors, SPCX DDP, GRH, path-MTU | **Skip** | GDAKI is not built on HIP. Note that upstream's GDAKI GRH work (`33c4db56`) is unrelated to RCCL's existing RoCE GRH support in `src/transport/net_ib/connect.cc` |
| Per-DevComm GIN backend selection | **Port** | `b0b64b61` (pick backend supporting requested signal type), `77a0b423`. RCCL now has four GIN types (`PROXY`, `GPI`, `ROCSHMEM_GDA`, `ANVIL_SDMA`) and genuinely benefits from multi-backend selection |
| Backend fallback to proxy on failure | **Port** | `7def4deb`. Directly useful with RCCL's multiple device-initiated backends |
| Multi-backend GIN error codes | **Port** | `a9a38230` |
| Device-side timeouts on Flush / Wait / WaitSignal / WaitCounter | **Port** | RCCL's blocking GIN APIs poll on `abortFlag` only; only `ncclGinBarrierSession::sync` takes `timeoutCycles`. Timeouts turn silent hangs into diagnosable errors, which matters more on the host-proxy path RCCL relies on |
| `GIN_PROXY_NTHREADS` multi-threaded proxy progress | **Port** | `96be3aa2` (`#2279`). RCCL runs a single GIN progress thread (`src/gin/gin_host.cc`). Host-proxy throughput is the AMD default path, so this is high value |
| Multiple GIN ops per progress iteration | **Port** | `#2232`, plus doorbell coalescing `7763b13b`, `f6be6cfa` |
| RMA plugin interface v15 with `optFlags` | **Port** | `54f6e1ba` (`#2254`). RCCL is at v14 (`src/include/plugin/rma/rma_v14.h`); v15 adds the aggregation hint |
| Connecting GIN with custom strides | **Port** | `d5afac46` fixes device-API barriers for strided GIN |
| GIN barrier for backends without strong signals | **Port** | `71401c73`. Written for exactly the case where a backend lacks strong-signal support, which applies to RCCL's proxy path |
| `ncclGinFinalize` error with profiler plugin | **Port** | `c5e91f55` |
| Rounded-up GIN context count reported to app | **Port** | `a8d0780e` (`#2301`) |
| Multiple contexts and signals for one-sided RMA | **Port** | `8a074116`, `6b68f565` (`NCCL_NUM_RMA_INT_CTX`), `f0663f25`. RCCL plumbs `config.numRmaCtx` (`src/init.cc:2188`) but `src/enqueue.cc:3501` still rejects any `ctx != 0` or `sigIdx != 0`. Lifting this is what enables multi-NIC RMA from one rank |
| Hierarchical 0-SM AllGather / AllToAll over multiple contexts | **Adapt** | Builds on the above. RCCL has CE zero-SM collectives under `NCCL_CTA_POLICY_ZERO` in `src/ce_coll.cc`, so the hierarchical distribution is the new part |
| NVLink multicast for small-message AllGather | **Skip** | NVLS-dependent |
| Disable multi-context for hierarchical collectives under graph capture | **Port** | `d4bb4884`. Correctness fix accompanying the above |
| Symmetric window aliasing corruption (issue `#2198`) | **Port** | Data corruption when multiple windows are carved from one backing allocation. RCCL shares the upstream structure here (`symMemoryObtain` in `src/dev_runtime.cc:702`, `ncclDevrFindWindow` used from `src/enqueue.cc:3547`), so the same exposure is present. Confirm against AMD hardware before closing |
| First CE collective hang on 2-rank communicator (`#2241`) | **Port** | RCCL enables CE collectives |
| `hasSysmem` check missing in CE enqueue | **Port** | `8e110544` |
| Inconsistent NVLS enablement when ranks share a GPU (`#2257`) | **Skip** | NVLS-specific |
| `minCTAs`/`maxCTAs` rejected when only one is set (`#2256`) | **Port** | Config validation bug, fully portable |
| Device API comm creation: GIN signal/counter requirements (`#2208`), memory leak (`#2225`), resource leak (`#2206`), capture-mode restore (`#2229`) | **Port** | All portable leak and error-path fixes |
| `ipcHandleMultiSegmentRegistration` leak on error | **Port** | `08329597` |
| `ncclSymmetricTaskScheduler` BAD_SHIFT | **Port** | `ff65f53c` |
| Symmetric kernel init without strong or VA signal | **Port** | `61adedc0` |
| collnet init for child communicators | **Defer** | `a38201bc`. RCCL does not ship a CollNet plugin path in practice |

## Device API and Compute Fabric Transport

| Item | Verdict | Notes |
| --- | --- | --- |
| Compute Fabric Transport host and device API | **Skip** | `6095baad`, `88ad7e2f` and ~20 `cft:` commits, plus new `src/cft_dev_runtime.cc` and `src/include/nccl_device/cft.h`. Requires `cuLogicalEndpoint` driver entry points, Blackwell, and CUDA 13.3 |
| Backward compatibility for JIT-compiled device API code | **Adapt** | `12b90c6a`, `678911ee`. RCCL has no runtime JIT path, but the compile-time-version `DevCommRequirements` handling is the compatibility mechanism itself and should track upstream so device-API ABI versioning stays aligned |
| LTO IR support for device API | **Partially landed** | RCCL already emits `librccl_device.bc` behind the `EMIT_LLVM_IR` CMake option (`CMakeLists.txt:92`). Upstream's LTO IR path is NVIDIA toolchain specific; keep RCCL's bitcode mechanism and take only the API-shape changes |
| `nccl_device.h` compatible with C99 host translation units | **Port** | `6431a138`, `5161b8aa`. RCCL's device header is C++-only today. Upstream also adds a make target that compiles the headers under C99 into a temp dir, which is worth taking as a build-time guard against regressions |
| `loadConst` `__ldg` performance regression | **Skip** | `0d7243be`. `__ldg` is a CUDA intrinsic; RCCL's `loadConst` in `src/include/nccl_device/utility.h` does not use it |
| `barrier.sync` to `barrier.sync.aligned` | **Skip** | PTX-specific (`c405f021`) |
| Improve teardown of suspended communicators | **Port** | `187e9ea0` |
| `comm->opCount` consistency | **Port** | `ab7bf4df` |
| Legacy window cleanup, `ginMultiSegmentExtraWins` removal | **Port** | `d1b4d512`, `0aee4bc9`. Housekeeping that reduces future merge friction |
| Host-side Team API cleanup | **Port** | `cad2e5ed` |
| MNNVL: check `fabricHandleSupport` for all ranks | **Adapt** | `f828c346`. RCCL has AMD-adapted MNNVL in `src/mnnvl.cc` gated on AMDSMI fabric state; the all-ranks consistency check is the portable part |

## Contrib, bindings and build

| Item | Verdict | Notes |
| --- | --- | --- |
| PACE parallelism-aware collective engine | **Skip** | `a764de58` (`#2319`). Sample code in `contrib/`, CUDA kernels |
| `nccl4rust` host and device bindings | **Skip** | `c3b63105`. Experimental, LTO IR based |
| NIIN NVSHMEM-compatible headers | **Skip** | Built on NVSHMEM semantics and NCCL device API primitives; ROCm's equivalent is rocSHMEM |
| CuTeDSL device API bindings (~15 commits) | **Skip** | Requires `nvidia-cutlass-dsl` |
| Architecture learning guides (`#2081`), DeepEPv2 analysis | **Skip** | Documentation in `contrib/` |
| `nccl4py` 0.4.1: teams, rank translation, device resources, `ncclParam*` access | **Adapt** | RCCL's fork is at `0.3.0` (`bindings/nccl4py/nccl/_version.py`). The teams / rank-translation / params APIs are portable; the CuTe DSL and NCCL EP surfaces are not. Note upstream *removed* the NCCL EP bindings in this range (`f2e4b0c8`) |
| `nccl_ep` and `nccl_m2n` moved out of `contrib/` | **Skip** | Relocation notice only |
| Inspector OTLP export | **Port** | `4835df5f`. RCCL ships an inspector profiler plugin (`plugins/profiler/inspector/`), so OTLP export is directly applicable |
| Python wheel build improvements and CI smoke tests | **Defer** | `45ceaa16`, `3f0ab378`. Useful, but couple to NVIDIA CI |
| Parallel-make build races on shared test/os artifacts | **Port** | `6064603e`. Real build bug |
| `clang-format` / `.git-blame-ignore-revs` churn | **Port** | RCCL already adopted upstream clang-format style in `872bcc158d`; keeping the config aligned reduces future conflict noise |

## Appendix: 2.30.7 verification

RCCL 2.30.7 landed before this sync began. These checks confirm the release
items are present at the baseline, and correct two readings taken from a stale
2.30.4 worktree during preparation.

| 2.30.7 item | Status | Evidence |
| --- | --- | --- |
| GIN plugin API v14 | **Landed** | `ncclGinVersion[] = {14, 13}` in `src/plugin/gin.cc:31`; `src/include/plugin/gin/gin_v14.h` defines `getGinProperties` |
| RMA plugin API v14 and GIN/RMA split | **Landed** | `src/include/plugin/rma/rma_v14.h`; `src/rma/` carries `rma.cc`, `rma_ce.cc`, `rma_proxy*.cc` |
| GPI backend type | **Landed** | `NCCL_GIN_TYPE_GPI = 4` in `src/include/nccl_device/core.h:172` |
| Hierarchical zero-SM collectives / `NCCL_CTA_POLICY_ZERO` | **Landed** | `src/nccl.h.in:86` |
| Anvil SDMA and rocSHMEM GDA adapted to v14 | **Landed** | `src/gin/gin_plugin_anvil_sdma.cc`, `gin_plugin_rocshmem_gda.cc` |
| Tuner plugin v6 | **Landed** | `src/include/plugin/nccl_tuner.h:21,25` |
| Scalable AllGatherV (`NCCL_ALLGATHERV_ENABLE`) | **Landed** (RCCL-only) | Not an upstream item; RCCL addition in the 2.30.7 cycle |

Corrections to preparatory notes: GIN and RMA plugin versions are **v14**, not
v13, and `NCCL_GIN_TYPE_GPI` already exists. Any planning that assumed a v13
baseline understates how much 2.30.7 already absorbed.

## Recommended sequencing

The refactor has to come first, because every later port lands in files it
moves.

1. **Structural rebase.** Take the `src/tuning/`, `src/enqueue/` and profiler
   thread refactors, carrying RCCL's tuning constants, `rccl_wrap.cc` hooks, RMA
   validation and proxytrace instrumentation across. No behavior change
   intended. This is the conflict-heavy step and should be reviewed on its own.
2. **API surface.** `ncclCollConfig_t`, the `nccl*Config()` entry points,
   `ncclConfigExt_t` with an RCCL `vendorId`, and profiler v7 with
   `userProfilerTag`. ABI-visible, so it should land as one coherent change.
3. **Diagnostics.** RAS diagnostics with AMDSMI substitutes, plus
   `NCCL_RUN_DIAGNOSTICS` P2P checks. Self-contained new trees, low conflict
   risk, high support value.
4. **GIN and RMA.** Multi-backend selection and proxy fallback, proxy
   multi-threading, RMA v15, and lifting the `ctx != 0` / `sigIdx != 0`
   restriction.
5. **Transport, topology and portable bug fixes.** Dynamic path arrays, GID
   change handling, WQE latency monitor, and the assorted leak and overflow
   fixes, which can be cherry-picked independently at any point.

## Task list

Effort is relative sizing, not calendar time. Risk reflects the chance of
destabilizing existing RCCL behavior, which tracks how much AMD-specific code
the change displaces rather than how large the change is.

| # | Task | Area | Effort | Risk |
| --- | --- | --- | --- | --- |
| 1 | Rebase RCCL tuning constants and `rccl_wrap.cc` hooks onto `src/tuning/` | Tuning | L | High |
| 2 | Rebase RMA validation, CE dispatch and CTA policy gating onto `src/enqueue/` | Enqueue | L | High |
| 3 | Move proxytrace instrumentation off the deleted `src/transport/profiler.cc` | Profiling | M | High |
| 4 | Re-express the CSV tuner against the unified cost model interface | Tuning | M | Medium |
| 5 | Add `ncclCollConfig_t` and the eight `nccl*Config()` entry points | Public API | M | Medium |
| 6 | Claim an RCCL `vendorId` and expose RCCL knobs via `ncclConfigExt_t` | Public API | S | Low |
| 7 | Register RCCL addon algorithms in the algorithm registry | Public API | S | Medium |
| 8 | Upgrade profiler plugin API to v7 and plumb `userProfilerTag` | Profiling | M | Medium |
| 9 | Port RAS diagnostics framework and env-consistency check | Diagnostics | M | Low |
| 10 | Implement AMDSMI-backed GPU inventory, ECC and XGMI link-state checks | Diagnostics | M | Low |
| 11 | Port `NCCL_RUN_DIAGNOSTICS` P2P connectivity check | Diagnostics | M | Low |
| 12 | Port dynamic topology path link arrays | Topology | S | Medium |
| 13 | Port GID cache and `IBV_EVENT_GID_CHANGE` handling | Transport | M | Medium |
| 14 | Port the WQE post-to-poll latency monitor and reconcile with `ib-cast` | Transport | M | Medium |
| 15 | Port event-based load balancing into `net_ib` | Transport | M | Medium |
| 16 | Lift the `ctx != 0` / `sigIdx != 0` restriction on one-sided RMA | RMA | M | Medium |
| 17 | Port multi-context hierarchical CE collectives incl. graph-capture guard | RMA / CE | M | Medium |
| 18 | Update the RMA plugin interface to v15 | RMA | S | Low |
| 19 | Add per-DevComm GIN backend selection and proxy fallback | GIN | M | Medium |
| 20 | Add `GIN_PROXY_NTHREADS` and multi-op-per-iteration proxy progress | GIN | M | Medium |
| 21 | Add device-side timeouts to blocking GIN APIs | GIN | M | Low |
| 22 | Port the portable leak, overflow and validation fixes | Cross-cutting | S | Low |
| 23 | Make the device headers C99-clean and add the build-time guard | Device API | S | Low |
| 24 | Port AMD EPYC topology modeling | Topology | S | Low |
| 25 | Port inspector OTLP export | Profiling | S | Low |
| 26 | Advance the `nccl4py` fork from 0.3.0 to the portable subset of 0.4.1 | Bindings | M | Low |

Tasks 1 through 3 are the structural rebase and should land and be reviewed as
one unit before anything else starts, since every subsequent task touches files
they relocate. Tasks 12, 22, 24 and 25 are independent and can be cherry-picked
early to reduce the size of the final merge.

## CHANGELOG stub

Draft text for the eventual `CHANGELOG.md` section, capturing the Skip
rationales while they are fresh.

```markdown
## RCCL 2.31.2 for ROCm <version>

### Added
* Compatibility with NCCL 2.31.2.
* Per-collective configuration APIs (`ncclCollConfig_t` and the `nccl*Config()`
  entry points), including the `ncclConfigExt_t` vendor extension list.
* NCCL profiler plugin API v7, with per-call user profiler tags and
  symmetric-kernel phase events.
* RAS diagnostics (`NCCL_RUN_RAS_DIAGNOSTICS`) covering GPU inventory, ROCm
  runtime versions, ECC counters, XGMI link state and `NCCL_*` environment
  consistency, using AMDSMI in place of NVML.
* Communicator init diagnostics (`NCCL_RUN_DIAGNOSTICS`) with an active P2P
  connectivity check and remediation guidance.
* Per-QP CPU WQE post-to-poll latency monitoring for the IB transport.
* Multiple GIN proxy progress threads via `NCCL_GIN_PROXY_NTHREADS`.

### Breaking for RCCL users (2.30.7 → 2.31.2)
* **`NCCL_GIN_TYPE` remapped for AMD backends.** NCCL 2.31 inserted EFA GDA at 5.
  rocSHMEM GDA is now 6 (was 5); Anvil SDMA is now 7 (was 6). `NCCL_GIN_TYPE=6`
  now selects GDA, not SDMA. IB proxy remains 2. See `src/gin/README.md`.

### Changed
* Reduced communicator host memory by allocating topology path link arrays to
  their actual length instead of a fixed maximum hop count.
* One-sided RMA now supports multiple contexts and signals; the previous
  restriction to context 0 and signal index 0 has been lifted.
* Updated the RMA plugin interface to v15.

### Resolved issues
* Fixed data corruption when multiple symmetric windows are carved from the same
  backing memory allocation.
* Fixed a hang during the first Copy Engine collective on a two-rank
  communicator.
* Fixed a hang when PXN connection initialization races with communicator
  teardown.
* Fixed valid communicator configurations being rejected when only one of
  `minCTAs` or `maxCTAs` is set.
* Fixed communicator initialization failures on multi-system topologies when
  `NCCL_P2P_PXN_LEVEL=1`.

### Known issues
* The following NCCL 2.31 features are NVIDIA-specific and are not available in
  RCCL: Compute Fabric Transport (requires CUDA logical endpoints on Blackwell),
  the GDAKI and EFA GDA GIN backends, PAT combined with NVLS, NVLink-multicast
  AllGather, TMA-based symmetric kernels, and the CuTeDSL and `nccl4rust`
  bindings.
```
