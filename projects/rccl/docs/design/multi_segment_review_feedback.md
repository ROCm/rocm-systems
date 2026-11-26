# Multi-segment: review feedback and what changed

The review feedback in this doc was given on the old multi-segment pull requests, which the IB-MSEG fork stack supersedes: layer 05 ([PR 10884](https://github.com/ROCm/rocm-systems/pull/10884)) is superseded by forks 1 and 4, layer 06 ([PR 10886](https://github.com/ROCm/rocm-systems/pull/10886)) by fork 3, layer 08 ([PR 10889](https://github.com/ROCm/rocm-systems/pull/10889)) and the IB-CAST GIN/RMA "wire" layer ([PR 10888](https://github.com/ROCm/rocm-systems/pull/10888)) by planned fork 6, and layer 07 ([PR 10890](https://github.com/ROCm/rocm-systems/pull/10890)) and the coverage layer 09 by the fork tests. This doc records each review item and the fork that carries the response. The fork design itself is in the [fork plugin](multi_segment_fork_plugin.md) note.

Every item marked fixed was checked by reading the fork 4 code, which carries forks 1 to 4.

Status words:

- **fixed**: the fix is in fork code, under the name given in the row.
- **not verified in fork**: the item was recorded as done, but the fix could not be found in fork code.
- **no longer applies**: the code the item was about does not exist in the fork design.
- **fork 6 requirement**: an IB-CAST item that the planned fork 6 must satisfy; no fork code exists for it yet.
- **open**: no code change is planned, or the review thread has no reply yet.

## Summary

| Group | Items | Fixed | Not verified in fork |
|---|---|---|---|
| [Fork 1](#fork-1-core-gate) | 15 | 14 | 1 |
| [Fork 3](#fork-3-registration-and-p2p) | 22 | 19 | 3 |
| [Fork 4](#fork-4-rma-and-gin) | 11 | 8 | 3 |
| [Documentation and pull request text](#documentation-and-pull-request-text) | 1 | 1 | 0 |
| [No longer applies](#no-longer-applies) | 18 | | |
| [Fork 6 requirements](#fork-6-requirements) | 12 | | |
| [Open](#open-items) | 10 | | |
| Total | 89 | 42 | 7 |

Comment ids are GitHub review comment ids on the PR in the "Superseded PR" column. "internal" rows are findings from internal reviews of the superseded branches, and "class" rows are recurring defect classes compiled from about 350 automated (GitHub Argus) and Copilot review threads.

## Fork 1: core gate

| Comment | Superseded PR | Reviewer | Request | Fork code | Status |
|---|---|---|---|---|---|
| 4465923772 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Do not hand a packed offset to a plugin that does not understand it | `rcclNetRegMultiSeg` in `transport/net.cc` passes fd -1 only when `proxyState->rcclMultiSegNet` is set; fork 3 sets `comm->rcclMultiSegNet` in `init.cc` only for IB-MSEG | fixed |
| 4466886680 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | fd -1 is enough; do not pass the expected count | `rcclNetRegMultiSeg` passes fd -1 with `offset` carrying only the export flags; the plugin counts the segments itself (`rcclIbCountRangeSegments` in `net_ib_multiseg/reg.cc`) | fixed |
| 4259327158 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | nikitaxgusev | Generic `net.cc` should not build IB per-segment MRs | `net.cc` only calls the plugin's `regMrDmaBuf` with fd -1; the plugin walks and registers | fixed |
| 4350033709, 4350104900 | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | Bertan | The host owner fd fallback picks the owner from rewritten types | `dev_runtime.cc`: ranks agree on `hostOwnerFdStatus` (gathered as `ownerFdStatus`) before the collective registration; the fix commit borrows an fd only when exactly one rank is `hostExportable`, with no fallback to a non-exporter | fixed |
| 4258921622 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | amd-arozanov | RCCL-only code in upstream files needs `// RCCL:` markers | every fork commit marks its additions with `// RCCL:` | fixed |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | A rank whose owner-fd fetch fails returns alone after the LSA barrier | owner-fd status agreed on every rank before registration (`hostOwnerFdStatus`) | fixed |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | Owner-fd failure skips the collective registration on one rank | same agreement; `NCCLCHECKGOTO(mem->hostOwnerFdStatus, ...)` runs after the gather, on every rank | fixed |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | The fd -1 call goes to any loaded plugin | gated on `rcclMultiSegNet`, which is true only for IB-MSEG | fixed |
| internal | [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | internal review | RMA multi-segment registration lacks the `NCCL_DMABUF_ENABLE` reject | `rcclMultiSegOn` in `rccl_dev_runtime.cc` requires the switch and `NCCL_DMABUF_ENABLE`; `comm->rcclMultiSegRma` uses it, and only the IB-MSEG RMA backend sets it | fixed |
| class 1 | several | GitHub Argus, Copilot | Rank-local failure before a collective leaves peers waiting | the last case, the owner-fd fetch, is agreed collectively (`hostOwnerFdStatus`) | fixed |
| class 4 | several | GitHub Argus, Copilot | Every comm pays for multi-segment by default | `NCCL_MULTI_SEGMENT_REGISTER` defaults to 0 on HIP (`transport/generic.cc`); fork 3 allocates the CTS layouts only when `useMultiSeg`, and fork 4 deepens the send queue only on RMA comms | fixed |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | No test reaches the GIN owner-fd branch | `GinHostRegisterMicrotest` (`Register_OwnerFdPublished_RegistersEveryConnectionThroughRegMrSymDmaBuf` and the no-owner-fd case) in `test/host/gin/gin-host-test.cc` | fixed |
| class 7 | several | GitHub Argus, Copilot | Tests force probe macros | no test defines `RCCL_HIP_MEM_LOCATION_HOST_SUPPORTED`; `test/host/CMakeLists.txt` sets it from the build's `HIP_MEM_LOCATION_TYPE_HOST` | fixed |
| internal | [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | internal review | CHANGELOG placement | forks 1, 3 and 4 each add a `changelog.d` fragment (`mseg-core-dmabuf-multiseg.changed.md`, `netib-mseg-registration-p2p.added.md`, `netib-mseg-gin-rma.added.md`) | fixed |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | RMA enqueue publishes a request before the batch error | no counterpart of the old "ring the pending batch before publishing" change was found in `rma/` or `gin/gin_host_proxy.cc` | not verified in fork |

## Fork 3: registration and P2P

| Comment | Superseded PR | Reviewer | Request | Fork code | Status |
|---|---|---|---|---|---|
| 4466075968 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Failure cleanup should call the deregister helper | `rcclIbSegRegister` releases every slot in [0, index] through `deregOne` on failure; `rcclIbSegDeregister` is the one release path (`seg_reg_mseg.h`, used by `reg.cc` and `gin.cc`) | fixed |
| 4466103032 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | One allocation instead of four `calloc`s | `rcclIbRegMrDmaBufFromRange` allocates one block of `rcclIbSegWalkBytes(nSeg)` and `rcclIbSegWalkBind` points the four arrays into it | fixed |
| 4466846579 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | One splitter, one lookup and one table for RMA and P2P | `rcclIbSegTable`, `rcclIbSegCursor` (`seg_mseg.h`) and `rcclIbSegFillWr` (`seg_wr_mseg.h`); the P2P `rcclIbSendSegTable` embeds `rcclIbSegTable`, and `IbMsegSegmentForZeroLengthOffset` uses `rcclIbSegIndexOf`. P2P slices in `p2p_multiseg.cc` and RMA in `gin.cc` (fork 4, through `rcclIbSegAppendWrs`) both fill WRs with `rcclIbSegFillWr`. The IB-CAST half is a [fork 6 requirement](#fork-6-requirements) | fixed |
| 4466941047 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Do not copy `ncclIbMultiSend`; build the WRs from the develop function | the fork's `IbMsegMultiSend` (`p2p.cc`) calls `rcclIbFillSendWrs` for every chain, one WR when each side is one segment. The IB-CAST half is a [fork 6 requirement](#fork-6-requirements) | fixed |
| 4466000659 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Move a test-only header into the test tree | no separate header; `IbMsegRegMrDmaBufMultiSeg` is declared in `seg_reg_mseg.h`, because `IbMsegRegMrDmaBuf` calls it | fixed |
| 4466498776 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | The uniform-segment decline is artificial | `rcclIbRegMrDmaBufFromRange` declines only on the count (`rcclIbSegCountSupported`); there is no layout check | fixed |
| 4466713360 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Check cuMem and the switch once and cache the result | `IbMsegResolveUseMultiSeg` (`connect.cc`) computes `useMultiSeg` once per comm from `rcclMultiSegOn` (switch and DMA-BUF), the GIN role and resiliency (`IbMsegUseMultiSeg`); cuMem is not in the gate | fixed |
| 4466697539 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | One CTS that carries the layout, not a side table plus a CTS | the layout prefix sits right before the CTS row and `rcclIbPrepareCtsWrite` posts both as one RDMA write; `NCCL_IB_CTS_NREQS_LAYOUT` marks it (`seg_p2p_mseg.h`) | fixed |
| 4466483903 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | One flush read, not one per segment | `IbMsegIflush` posts one read per device of the first byte of the last receive, with the segment MR from `IbMsegMrForRange`, or the scratchpad read | fixed |
| AICOMRCCL-3124 | Jira task | Aleksandr | Register whatever segment count the walk returns | up to `RCCL_IB_REG_SEG_DEFAULT` segments per registration (`transport/rccl_mseg.h`), with `rcclIbSegCountSupported` as the one check; P2P posts one slice per progress call (`rcclIbSendCursor`), and fork 4 runs RMA in rounds (`seg_rma_round_mseg.h`); see [limits](multi_segment_current.md#limits) | fixed |
| Vadim | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Vadim | Move the multi-segment code into its own file | the classic hooks are in `p2p_multiseg.cc`, declared in `p2p_multiseg_mseg.h`, which `p2p.cc` includes; the shared layout code is in `seg_p2p.cc` | fixed |
| 4351936076 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Bertan | The classic sender QP overflows with three maximum chains | a segmented send posts one slice per QP per progress call (`rcclIbSendCursor`, `rcclIbSendProgress`), so P2P keeps develop's `2 * NET_IB_MAX_REQUESTS` (512) send WRs; fork 4 gives only RMA comms `RCCL_RMA_SEND_QP_WRS` (2080) | fixed |
| 4351967080 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Bertan | The classic segmented send ignores `bad_wr` | a slice write that does not post drops its held event (`rcclIbSendHoldSlice` in `p2p.cc`); covered by `NetIbSegmentedSendPrefix` | fixed |
| 4335727853 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | nikitaxgusev | Deduplicate flush reads | one flush read per device (`IbMsegIflush`) | fixed |
| class 11 | several | GitHub Argus, Copilot | Peer-written CTS and layout fields are used unbounded | `rcclIbLayoutRead` bounds the peer's `nSegments` and `ndevs` before sizing its read (`seg_p2p.cc`); the CTS row's `nreqs` is checked against 1 to `NCCL_NET_IB_MAX_RECVS` (`p2p_multiseg.cc`). The IB-CAST resume bound is a [fork 6 requirement](#fork-6-requirements) | fixed |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | MPI tests skip the production registration entry | `UBR_MultiSegment.NetProxyPartialFinalSegment` (`test/RegistrationMPITests.cpp`) drives `sendProxyRegBuffer` and `recvProxyRegBuffer` through `rcclIbRegMrDmaBufFromRange` and `IbMsegRegMrDmaBufMultiSeg` | fixed |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | No enabled test covers the flush read-post error | `NetIbMsegP2pMicrotest.Iflush_GpuScratchChainsWriteAndReadInOnePost` fails the second post and checks that the flush posts once | fixed |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | `NetIbMultiSegmentMPITest` jobs only in `net_ib_transport` | in `mi300x_mellanox_ib`, `mi300_mellanox_ib_func`, `mi355x_thor2_roce`, `mi355x_thor2_roce_func` and `net_ib_transport`; the AINIC configs get their entries with fork 5 | fixed |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | `env.rst` still says 16 segments | `NCCL_MULTI_SEGMENT_REGISTER` in `docs/userguide/source/env.rst` states up to `RCCL_IB_REG_SEG_DEFAULT` segments under `max_mr` | fixed |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | Four helpers used only by tests | the item does not name the helpers, so the fork could not be checked for them | not verified in fork |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | An fd -1 handle that walks to one segment leaks its arrays on deregistration | not re-checked in `reg.cc` | not verified in fork |
| class 6 | several | GitHub Argus, Copilot | CI filters that match nothing, missing switch | the fork's `*_mseg` jobs and filters were not re-audited | not verified in fork |

## Fork 4: RMA and GIN

| Comment | Superseded PR | Reviewer | Request | Fork code | Status |
|---|---|---|---|---|---|
| 4466041034, 4466043272 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | No magic numbers in the RMA WR budget | named macros in `seg_rma_mseg.h`: `RCCL_RMA_MAX_DATA_WRS` (64), `RCCL_RMA_MAX_SIGNAL_WRS` and `RCCL_RMA_SEND_QP_WRS`; none depends on the segment count | fixed |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | Every sender QP grows to 2080 and the in-flight cap silently lowers `NCCL_GIN_PROXY_QUEUE_SIZE` | `rcclIbSendQpWrs` (`connect.cc`) gives 2080 only to RMA comms (`base.isRma`); fork 1's RMA proxy (`rma_proxy.cc`) and GIN proxy (`gin_host_proxy.cc`) log an INFO line with the lowered request count (32) when the cap lowers the queue, so the lowering is no longer silent; the fork 4 changelog fragment says only that the GIN proxy caps in-flight requests per peer, and neither it nor `env.rst` states 32 or 2080 | fixed |
| internal | [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | internal review | In-flight cap and 2080-entry send queue even for single-segment windows | kept by design; the cap is visible in the proxies' INFO line (which prints 32) and named, without a number, in the fork 4 changelog fragment; the 2080-entry send queue is not logged or documented and shows only in code (`RCCL_RMA_SEND_QP_WRS` in `seg_rma_mseg.h`, `RCCL_RMA_MAX_INFLIGHT_REQUESTS` in `transport/rccl_mseg_rma.h`); `env.rst` states neither number | fixed |
| class 2 | several | GitHub Argus, Copilot | Send and flush queues sized for one maximum chain | P2P posts one slice per progress call (fork 3); RMA posts rounds of at most `RCCL_RMA_MAX_SIGNAL_WRS` (65) WRs with 32 requests in flight (`RCCL_RMA_MAX_INFLIGHT_REQUESTS`); there is no credit scheme | fixed |
| class 3 | several | GitHub Argus, Copilot | Events armed for WRs that never post | RMA through `rcclRmaCompletePostedRequest` (`gin.cc`), classic sends through `rcclIbSendHoldSlice` (fork 3); IB-CAST is a [fork 6 requirement](#fork-6-requirements) | fixed |
| class 8 | several | GitHub Argus, Copilot | Tests that pass with the fix reverted | tests read the segment count from the handle: `RegistrationMPITests.cpp` asserts `rcclRmaMrSegmentCount` on the RMA and GIN windows, and the multi-segment MR carries `rcclRmaSegMrTag` | fixed |
| class 15 | several | GitHub Argus, Copilot | Env docs and CHANGELOG | `env.rst` states the `RCCL_IB_REG_SEG_DEFAULT` segment limit, `max_mr` and the CTS memory; the fork 4 changelog fragment says ranks times segments is bounded, and the bound is `RCCL_RMA_SEG_TABLE_MAX_ENTRIES` in `transport/rccl_mseg_rma.h` | fixed |
| internal | [PR 10890](https://github.com/ROCm/rocm-systems/pull/10890) | internal review | RMA multi-node jobs without `RCCL_TEST_RMA_MSEG_REQUIRE=1` | the fork's RMA multi-segment jobs set it in `mi300x_mellanox_ib`, `mi300_mellanox_ib_func`, `mi355x_thor2_roce`, `mi355x_thor2_roce_func` and `net_ib_transport`; the IB-CAST job is a [fork 6 requirement](#fork-6-requirements) | fixed |
| class 12 | several | GitHub Argus, Copilot | Export and registration rounded differently | `rcclIbSegExportNext` (`seg_export_mseg.h`) exports the whole pages the MR spans; it is not confirmed that every registration path rounds the same way | not verified in fork |
| class 9 | several | GitHub Argus, Copilot | Fakes with stale signatures across stacked layers | not re-audited; fork 4's compile-twice host tests build the fork's fakes, but they were not checked against this class | not verified in fork |
| internal | [PR 10890](https://github.com/ROCm/rocm-systems/pull/10890) | internal review | Hybrid require knob and `hipMemLocationTypeHost` shim placement | not re-checked in the fork's hybrid tests (`test/common/HybridVmmHelpers.hpp`) | not verified in fork |

## Documentation and pull request text

| Source | Reviewer | Request | Change | Status |
|---|---|---|---|---|
| fork 2 draft pull request description | internal review | Describe the problem in neutral terms; do not justify the design by what it avoids editing or by who asked for it | the pull request descriptions open with the problem (a buffer of several physical segments needs one MR per segment), and the design docs state the separate plugin as a trade-off: `net_ib` and `net_ib_cast` stay at develop, at the cost of syncing the copy | fixed |

## No longer applies

These items were about code that the fork design does not have.

| Comment | Superseded PR | Reviewer | Request | Reason |
|---|---|---|---|---|
| 4466191539 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Put the capability in the metadata, not in the `devName` tail | no capability word and no `devName` trailer; each side sends its `useMultiSeg` in `IbMsegConnectionMetadata`, and connect fails with `ncclInvalidUsage` on a mismatch |
| 4466746471 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | The `connect.h` size assert is wrong | no `devName` trailer, so no size assert |
| 4466734379 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | No separate MR for the side table | no side table; the combined CTS slot array is one block |
| AICOMRCCL-3125 | Jira task | Aleksandr | Allocate the CTS side tables on first multi-segment use | no side table |
| Vadim | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Vadim | Split the segmented send into named helpers | there is no segmented copy of the send; `IbMsegMultiSend` builds every chain through `rcclIbFillSendWrs` |
| 4259441268, 4335727847 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | nikitaxgusev | 1.4 MiB side table per comm, paid eagerly | no side table; on a multi-segment connection each side's 128 KiB CTS FIFO becomes a 512 KiB slot array (about 384 KiB more per side) and the send side adds about 11 KiB, and GIN comms keep the legacy CTS |
| review body | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | nikitaxgusev | Enablement plan, and how PyTorch fits under 16 segments | no 16-segment cap; answered in an issue comment |
| 4161145011 | [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | GitHub Argus | Double refund of the RMA WR credit counter | no credit counters |
| class 5 | none; [PR 10887](https://github.com/ROCm/rocm-systems/pull/10887) is merged | GitHub Argus, Copilot | CE selector arms drift | fixed in layer 01, which is merged into develop; no fork code |
| class 10 | several | GitHub Argus, Copilot | Build consumers of moved paths | the fork adds `net_ib_multiseg/` and moves no existing path |
| class 13 | several | GitHub Argus, Copilot | 1.4 MiB side table and per-segment flush reads | no side table; the flush is one read per device (fork 3) |
| class 14 | several | GitHub Argus, Copilot | IB-CAST copies of the builder and registration | the fork is one renamed copy of `net_ib` by design, and fork 6 adds only IB-CAST-specific pieces on top of the fork's shared paths |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | No test fails if the WR credit reservation is reverted | no credit reservation |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | Every comm allocates and registers a 1.4 MiB side table, GIN included | no side table |
| internal | [PR 10890](https://github.com/ROCm/rocm-systems/pull/10890) | internal review | A runner job names a test that does not exist | that job is not in the fork's configs |
| internal | [PR 10890](https://github.com/ROCm/rocm-systems/pull/10890) | internal review | The old NET/IB multi-segment jobs are missing from the AINIC configs | those jobs are not in the fork; AINIC keeps IB-CAST under the switch, so the AINIC configs run only the host-only and plugin-selection fork jobs, and the data-path entries come with fork 5 |
| internal | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | internal review | The RMA-comm advertising test checks only the predicate | nothing is advertised; GIN comms keep the legacy CTS through `IbMsegUseMultiSeg` |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | GIN side-table skip covered by a boolean test only | no side table |

## Fork 6 requirements

These IB-CAST items are not fixed in fork code; planned fork 6 must satisfy them. See [fork plugin, fork 6](multi_segment_fork_plugin.md#fork-6-ib-cast-multi-segment-planned).

| Comment | Superseded PR | Reviewer | Requirement |
|---|---|---|---|
| internal | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | internal review | The multi-segment scope must exclude AINIC inline CTS (`IbCastAinicCtsInlineData`) |
| internal | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | internal review | The receiver must size its CTS slots from the peer's raw capability, not from `peerCaps` masked by its own scope |
| internal | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | internal review | The send-resume path must bound the peer-written `nreqs` to 1 to 8, as `IbCastIsend` does |
| 4466846579, 4466941047 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | IB-CAST P2P and RMA must build their WRs through the fork's shared builder and register through the shared helpers, not their own copies |
| 4467040965 | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | Aleksandr | IB-CAST multi-segment registration must live in the plugin's `reg.cc`, where classic registration lives |
| 4467649873 | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | Aleksandr | Post the receiver sizes write only when the request was split or `nreqs > 1`, and set the segmented BY_INDEX immediate flag only on a split send |
| 5767701446 | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | Aleksandr | Limit multi-segment to the validated scope (no CTS offload, BY_ORDER, QP sharing or resiliency) and decide it once at init |
| 4352193108 | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | Bertan | A QP that posted only a slice must drop its event so a failed request completes |
| internal | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889), [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | internal review | Enable every IB-CAST multi-segment job in every architecture config that can run it |
| internal | [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | internal review | A test must fail if the IB-CAST RMA post-path error handling is reverted |
| internal | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | internal review | A test must drain a partly posted IB-CAST send through a real completion |
| internal | [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | internal review | The IB-CAST RMA multi-node job must set `RCCL_TEST_RMA_MSEG_REQUIRE=1` |

## Open items

| Comment | PR | Reviewer | Item | State |
|---|---|---|---|---|
| 4467318519 | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | Aleksandr | Why the BY_INDEX completion protocol exists only on IB-CAST | no reply yet; the answer: BY_INDEX uses the immediate for the request index, classic puts the size there |
| 4466504452, 4466738828 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Where the 16-segment limit comes from | no reply yet; the answer: no 16 remains; one registration takes up to `RCCL_IB_REG_SEG_DEFAULT` segments and a larger buffer stages |
| 4465933531 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Does `numSegments < 1` mean anything | no reply yet; the answer: every branch keys on `numSegments > 1` (`transport/net.cc`) |
| 4466591874 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | Aleksandr | Is the `ncclIbMalloc` buffer still page aligned | no reply yet; the answer: `ncclIbMalloc` already uses `posix_memalign` on a page boundary (`include/alloc.h`) |
| 4126266662 | [PR 10887](https://github.com/ROCm/rocm-systems/pull/10887) (merged layer 01) | nusislam | World rank and LSA rank treated as one index in the CE slow path | no reply on the thread |
| 4136961982 | [PR 10887](https://github.com/ROCm/rocm-systems/pull/10887) (merged layer 01) | alexander-sannikov | Descriptive test comments | no reply on the thread |
| 4309321625 | [PR 10887](https://github.com/ROCm/rocm-systems/pull/10887) (merged layer 01) | garzaran | Document the new env variable | `env.rst` landed in develop; no reply points to it |
| internal | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | internal review | With IB-MSEG selected, every classic send, single-segment included, builds through `rcclIbFillSendWrs` | no switch-off latency or bandwidth comparison for single-segment sends yet (performance work, P2) |
| internal | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | internal review | The `reg.cc` unwind is tested only by develop's disabled AICOMRCCL-3112 test | stays disabled by decision |
| issue comment | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | nusislam | Check the RMA send depth against the HCA `max_qp_wr` | the in-flight cap is 32 requests per peer and context (`rcclRmaInflightRequestsCap`), so RMA comms get 2080 send WRs; that fits Thor2's `max_qp_wr` of 4351, Mellanox is not checked, and only a `static_assert` in `connect.cc` guards the size |
