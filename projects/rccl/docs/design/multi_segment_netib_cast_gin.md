# IB-CAST GIN/RMA: per-segment MRs and rounds

> [!NOTE]
> Old stack, wire layer ([PR 10888](https://github.com/ROCm/rocm-systems/pull/10888)), at the wire tip `ba3ab26f30a7`. The 11-04 version of this note described completion-driven send-queue credits with a bounded wait (`IbCastRmaReservePostComplete`, `rmaWrsOutstanding`, 575- and 271-entry queues) and a 16-segment consensus record; the credits and the cap are gone. In the planned design IB-CAST returns to develop and IB-CAST RMA multi-segment is rebuilt inside the IB-MSEG plugin in planned fork 6; see [the fork plugin](multi_segment_fork_plugin.md#fork-6-ib-cast-multi-segment-planned).

The IB-CAST RMA backend `IbCastRmaIbProxy` (`net_ib_cast/gin.cc`) uses the classic GIN/RMA design: one MR per segment, the shared consensus exchange, the shared segment cursor, and RMA rounds. This doc covers what IB-CAST adds: a versioned registration record with physical device numbers, rkey lookup by physical device, the per-WR cap, error-owner propagation, and the AINIC recovery hook. Read [GIN/RMA](multi_segment_gin_rma.md) first for the shared design.

## Scope

GIN/RMA never touches the CTS, so none of the [IB-CAST P2P](multi_segment_netib_cast.md) work applies here.

| Surface | IB-CAST P2P | IB-CAST GIN/RMA |
|---|---|---|
| Entry point | `netIbCast` send and receive | `IbCastRmaIbProxy` vtable |
| Remote layout | CTS layout prefix | symmetric table gathered at registration |
| Connect capability | `RCCL_IB_CAP_MULTISEG` | none |
| Switch accessor | `IbCastAdvertiseMultiSeg()` | `rcclRmaMultiSegEnabled()` |
| Split | the fill over the scheduler's chunk | rounds over `rcclIbSegCursor`, then `IbCastRmaCapWrs` |

Both accessors read the same parameter, so the value must match on every rank. IB-CAST `gin.cc` must not include `net_ib/gin.h`, because `gin_cast.h` uses the same include guard.

## Code layout

Develop's bodies stay in `net_ib_cast/gin.cc`. With the switch on, each entry point returns the matching `rcclIbCastRmaIbProxy*` function: `RegMrSymDmaBuf` (and `RegMrSym`, which calls it with `fd = -1`), `DeregMrSym`, `IPut`, `IGet`, `IPutSignal`, `IFlush` and `Test`. `IbCastRmaIbProxyConnect` ends in `rcclIbCastRmaIbProxyConnectFinish`, which allocates the registration records. Each context keeps per-peer round queues (`IbCastRmaIbProxyCtx::pending`). Outside `gin.cc`, the switch gates the RMA sender QP depth and CQ floor (`IbCastBuildDataQpCreateAttr`, `connect.cc`) and the AINIC abandon hook, only for `isRma` comms.

## Handle and indexing

`IbCastRmaProxySegMrHandle` starts with `tag` (`rcclRmaSegMrTag`) and `nSegments`, asserted at offsets 0 and 4, so tests read the segment count through `rcclRmaMrSegmentCount` on both backends. It then holds `segOff`, one `ncclIbMrHandle` per segment in `mrHandle`, the shared symmetric table `sym` (`rcclIbSegSymTable`: every rank's offsets, segment VAs and rkeys per device slot), `ibDevNs` as `[rank][slot]` (-1 unused), and `flushSeg`. All arrays are sized from the walked count.

IB-CAST registers through `cComm->recvComm` but posts on per-peer QPs whose device indices may name a different physical HCA. So `IbCastRmaDevSlot` maps a physical `ibDevN` to a registration slot with `rcclRmaDevSlotOf`: data WRs use the remote QP's device and flush reads use the local one. A missing slot is a WARN and `ncclInternalError`.

## Registration

The walk is the classic one, including whole-page export, the `segBase == 0` guard, the owner fd for imported host segments, and `rcclIbCastMrAdmit`. Registration goes through `rcclIbSegRegister` with `IbCastRegMrDmaBufInternal` per segment. The shared exchange (`rcclIbSegSymExchange`) gathers `{status, nSegments}`, fails every rank on any failed status or count mismatch, then gathers offsets, VAs and keys. After it, IB-CAST gathers `IbCastRmaProxyRegistration {magic 0x524d5347, version 1, ibDevNs[]}`, and a peer with a different magic or version fails every rank. Rollback deregisters only complete segments and frees the handle once.

`NCCL_DMABUF_ENABLE=1` is checked by the runtime gate `rcclDevrRmaRegistersMultiSegment` before any multi-segment window reaches the plugin; unlike classic `gin.cc`, the IB-CAST plugin has no own check.

## Data path

Operations go through the shared rounds of `rccl_mseg_rma_round.h`; IB-CAST supplies the hooks (`rcclIbCastRmaRoundBuild`, `rcclIbCastRmaRoundOutstanding`, `rcclIbCastRmaRoundPost`, `rcclIbCastRmaRoundRetire`, `rcclIbCastRmaRoundFailed`). The build hook (`IbCastRmaBuildSegmentedWrs`) emits slices with the shared `rcclIbSegAppendWrs` and looks up device slots and keys, then `IbCastRmaCapWrs` splits any slice above `IbCastMaxPostSendBytes`. A round that runs out of WRs ends early and covers fewer bytes; a single chain that cannot fit is refused with `ncclInternalError`. A chain with nothing queued ahead posts at once (`rcclIbCastRmaPostChain`); otherwise it queues behind the peer's earlier operations (`rcclIbCastRmaEnqueue`).

- **IPut, IGet.** Each round posts at most `NCCL_RMA_MAX_DATA_WRS` data WRs; only its last WR is signaled. Each round adds an event to the request, and a queued request holds one more until its last round posts, so the request completes only after every round.
- **IPutSignal.** The fetch-and-add on the segment holding `signalOff` rides the last round.
- **IFlush.** One loopback read of one byte of `flushSeg`, the first device segment, with the rkey of the slot holding the local device. A read cannot pass earlier posted writes, so one read flushes the window; a window that is all host memory posts nothing.

## Queues

The RMA sender data QP is `max(IbCastDataSendQueueDepth(IbCastMaxPostSendBytes), NCCL_RMA_SEND_QP_WRS)`, so it holds the 32 in-flight requests of 65 WRs (2080). Because a dead QP flushes every queued WR as a CQE, the RMA sender CQ is at least 2080 with the switch on, still clamped to the device `maxCqe`. The in-flight cap comes from `rcclRmaProxyMaxInflightRequests`, which recognises `IbCastRmaIbProxy` as well as `ncclRmaIbProxy`. With the switch off, IB-CAST RMA keeps develop's depths.

## Error completions

Any caller's CQ poll can retire any request on that CQ, so errors must reach the request that owns them:

- `IbCastRmaPollCq` logs one WARN per error completion with the GID of the peer's device 0 (the RMA proxy connects one QP to that device) and marks the owner `NCCL_NET_IB_REQ_FAILED`. After a QP error every flushed WQE completes with the owner's `wr_id`; only the first owns the event.
- A posted prefix keeps the request for `Test()` to drain, marked FAILED because the rejected signaled tail never produces a CQE; nothing posted frees the slot (`IbCastRmaCompletePostedRequest`).
- `rcclIbCastRmaIbProxyTest` on a FAILED request returns `done = 1` with an error, even when another request's poll drained its CQE.
- **AINIC port recovery.** `IbCastPortRecoveryQpsRestoreAinic` calls `IbCastRmaAbandonDeviceRequests` before destroying the QPs: for an RMA comm with the switch on, every in-use request with events on that device has its events zeroed and is marked FAILED. The rebuilt QPs go through the same create helpers as connect.

## Tests

`RmaMultiSegmentMPITest` and `RmaMultiSegmentPostMPITest` run under IB-CAST through `RmaMPITestBase`, which resolves `IbCastRmaIbProxy` when `NCCL_NET=IB-CAST`; the handle tag lets `RCCL_TEST_RMA_MSEG_REQUIRE=1` work on both backends. Host microtests in `test/host/transport/net_ib_cast/net-ib-cast-gin-test.cc` run against fake verbs:

| Suite | What it pins |
|---|---|
| `NetIbCastGinSegRegMicrotest` | per-segment registration, the record exchange and rollback |
| `NetIbCastGinRoundMicrotest` | rounds, per-peer order, the queued hold, failed rounds |
| `NetIbCastGinPostPathMicrotest` | putSignal, rkey lookup by `ibDevN`, flush rkey from the local device, prefix posts |
| `NetIbCastGinErrorLogMicrotest` | an error completion drained by another request is reported to its owner |
| `P2pResiliencyRecoveryAinicRmaMicrotest` (`net-ib-cast-p2p-resiliency-recovery-ainic-test.cc`) | AINIC restore fails the device's requests with the switch on and leaves them with it off |

Runner jobs are `cast_rma_multisegment` and `cast_rma_multisegment_hybrid`; see [testing](multi_segment_testing.md#runner-jobs).

## Review checklist

- `gin.cc` differs from develop only by one-line dispatches, the connect-finish call, the record free and the multi-segment block.
- The handle's first two fields are the tag and the count.
- Records are checked for status, count, magic and version on every rank before any table is used.
- rkeys and local MRs are found through the slot of the physical `ibDevN`.
- Rounds keep per-peer order; a FAILED request is reported even when another request drained its CQE.
- GDAKI `regMrSymDmaBuf` stays NULL; proxy RMA is the ROCm path.

## Symbol index

| Concern | Symbols (`net_ib_cast/gin.cc` unless noted) |
|---|---|
| Handle and lookup | `IbCastRmaProxySegMrHandle`, `IbCastRmaSegOf`, `IbCastRmaRangeOk`, `IbCastRmaDevSlot`, `IbCastRmaRemoteRkey`, `IbCastRmaRankOff` |
| Registration | `rcclIbCastRmaIbProxyRegMrSymDmaBuf`, `IbCastRmaProxyRegistration`, `IBCAST_RMA_REGISTRATION_MAGIC`, `IBCAST_RMA_REGISTRATION_VERSION`, `IbCastRmaRegAllGather`, `rcclIbCastRmaIbProxyConnectFinish`, `IbCastRmaSegMrHandleFree` |
| Rounds | `rcclIbCastRmaRoundBuild`, `rcclIbCastRmaRoundPost`, `rcclIbCastRmaRoundRetire`, `rcclIbCastRmaStartRequest`, `rcclIbCastRmaPostChain`, `rcclIbCastRmaEnqueue`, `rcclIbCastRmaSubmit`, `IbCastRmaCapWrs`, `IbCastRmaBuildSegmentedWrs` |
| Completions | `IbCastRmaPollCq`, `IbCastRmaAccountCompletion`, `IbCastRmaCompletePostedRequest`, `rcclIbCastRmaIbProxyTest` |
| Recovery | `IbCastRmaAbandonDeviceRequests` (`p2p_resiliency_recovery_ainic.cc`) |
| Queues | `IbCastBuildDataQpCreateAttr` and the RMA CQ floor (`connect.cc`); `NCCL_RMA_SEND_QP_WRS` (`transport/rccl_mseg_rma.h`) |
