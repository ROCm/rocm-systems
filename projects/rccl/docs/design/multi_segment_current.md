# Multi-segment: current contract

> [!NOTE]
> This note describes the IB-MSEG fork stack through fork 4. Multi-segment support lives in the IB-MSEG plugin (`src/transport/net_ib_multiseg/`) and in RCCL-marked core gates; classic `net_ib` and `net_ib_cast` stay at develop. On AINIC, the switch with `NCCL_NET=IB` selects IB-MSEG for net and RMA; with `NCCL_NET` unset, AINIC autodetect still selects IB-CAST. IB-CAST support waits on fork 5 (paused) and fork 6 (not started); see [the fork plugin](multi_segment_fork_plugin.md). File names below are relative to `src/`, and plugin files without a directory are in `transport/net_ib_multiseg/`.

This doc lists the rules the multi-segment code must keep, its limits, and what the switch does when it is off. It is the checklist for reviewing a change, not a design explanation; each item links to the doc that explains it. Terms are defined in the [overview](multi_segment_overview.md#terms). The review history behind these rules is in [review feedback](multi_segment_review_feedback.md).

## Invariants

### Registration

1. **One MR per physical segment, with a fixed cap.** `rcclIbSegCountSupported` (`transport/rccl_mseg.h`) is the one count check: a count from 1 to `RCCL_IB_REG_SEG_DEFAULT` registers. NET and RMA registration both check it before they export one fd per segment. There is no environment knob for the cap. See [limits](#limits).
2. **Single-segment buffers keep the develop path.** The core hands fd -1 only for a buffer of more than one segment, and a real fd stays one MR. In the P2P handle, `mrs[]` aliases segment 0 (`reg.cc`), so single-segment code is unchanged; per-segment MRs are freed only through `segMrs`, never through the alias.
3. **GIN/RMA exports whole pages.** `rcclIbSegExportNext` (`seg_export_mseg.h`) exports each segment from the page that holds the segment pointer, rounded up to whole pages. NET P2P exports the exact segment range (`rcclIbRegMrDmaBufFromRange`, `reg.cc`). See [GIN/RMA registration](multi_segment_gin_rma.md#registration).
4. **Host VMM length.** `cuMemGetAddressRange` can return `segBase == 0` for host VMM, so the bytes left in a segment are `segSize` when `segBase == 0`, else `segSize - (segPtr - segBase)` (`IbMsegBytesRemainingInSegment` in `seg_p2p_mseg.h`, and the same rule in `rcclIbSegExportNext`).
5. **GIN/RMA registration is collective and always reaches the first gather.** Every rank allgathers a `rcclIbSegSymRecord {status, nSegments}` (`seg_sym_mseg.h`) into `regRecords`, even after a local failure, so no peer is left waiting. `rcclRmaIbProxyConnectFinish` (`gin.cc`) allocates `regRecords` at connect, so registration cannot fail before its first collective. Any failed status fails every rank (`rcclIbSegSymAgree`). See [consensus](multi_segment_gin_rma.md#consensus).
6. **GIN ranks need equal segment counts, not equal sizes.** `rcclIbSegSymExchange` requires the same `nSegments` on every rank; each rank's own offsets are kept in the symmetric table, and each side of a transfer splits on its own rank's row. RMA also needs `nranks * nSegments` to fit `RCCL_RMA_SEG_TABLE_MAX_ENTRIES` (2^20, `rcclRmaSegTableFits` in `transport/rccl_mseg_rma.h`).
7. **NET registration walks the whole range.** `rcclIbCountRangeSegments` (`reg.cc`) counts segments by walking every byte of the range and fails on a zero-length step, and `rcclIbRegMrDmaBufFromRange` exports each segment's exact bytes, so no handle can advertise bytes beyond its last MR. Non-uniform layouts register.
8. **NET segment count is cached for the full registration.** `ncclReg::rcclNet.nSegments` holds the count of the whole registration, set by `rcclNetRegCommit` (`rccl_wrap.cc`, develop) only after every peer registered.
9. **`net.cc` hands fd -1 only to IB-MSEG.** `rcclNetRegMultiSeg` (`transport/net.cc`) calls `regMrDmaBuf` with fd -1 and the export flags in `offset` only when `proxyState->rcclMultiSegNet` is set, which copies `comm->rcclMultiSegNet`. Otherwise a multi-segment buffer stages. The whole-range HSA DMA-BUF export is skipped for multi-segment buffers. `IbMsegRegMrDmaBuf` (`reg.cc`) walks the range when it gets fd -1.
10. **Rollback deregisters only complete handles.** Registration goes through `rcclIbSegRegister` and `rcclIbSegDeregister` (`seg_reg_mseg.h`), which release exactly the MRs made so far, including a handle `regOne` stored before it failed.
11. **The MR count is admitted against the device.** `rcclIbMrAdmit` (defined in `reg.cc`, declared in `common_mseg.h`) declines a registration unless `nSeg` more MRs fit under `max_mr` on every rail. A NET buffer then stages; a GIN/RMA window fails on every rank through the consensus.
12. **Some comms decline multi-segment NET registration.** A resilient comm declines on both sides, because resiliency replays whole slots. A receive comm without the multi-segment CTS layout (a GIN comm) declines too, because the layout describes the receiver. These declines and the `max_mr` check run in `rcclIbMultiSegAdmit` (`reg.cc`) before any fd is exported.

### Data path

13. **Split at local and remote boundaries independently.** Local and remote request bases can start at different offsets in their registrations, so the splitters keep two `rcclIbSegTable`s in one `rcclIbSegCursor` (`seg_mseg.h`) and fill each slice with `rcclIbSegFillWr` (`seg_wr_mseg.h`). P2P (`p2p_multiseg.cc`) and RMA (`gin.cc`) use the same table type, cursor and fill; RMA chains its slices with fork 4's `rcclIbSegAppendWrs`. See [the offset bug](multi_segment_netib_wire.md#independent-offsets).
14. **Slices are at most `RCCL_IB_SGE_MAX_BYTES` (`UINT32_MAX`) bytes**, because `ibv_sge::length` is 32 bits.
15. **Only the final WR of a chain or round is signaled** (`rcclRmaWrIsSignaled`, `seg_rma_mseg.h`). `IPutSignal` signals only its trailing atomic, which must be 8-byte aligned and inside one segment (`rcclRmaSignalOffsetValid`).
16. **RMA posts in rounds.** An operation posts at most `RCCL_RMA_MAX_DATA_WRS` (64) data WRs per round, plus the atomic on the last round of `IPutSignal` (`RCCL_RMA_MAX_SIGNAL_WRS` = 65), and its next round once the plugin reports the previous one done. Operations queue per peer in request order (`seg_rma_round_mseg.h`), so a later operation never passes an earlier one. See [rounds](multi_segment_gin_rma.md#rounds).
17. **RMA in-flight requests are capped.** With `comm->rcclMultiSegRma` set, the RMA and GIN proxies keep at most `RCCL_RMA_MAX_INFLIGHT_REQUESTS` (32, `transport/rccl_mseg_rma.h`) requests per peer and context (`rcclRmaProxyMaxInflightRequests` in `rccl_dev_runtime.cc`, called from `rma/rma_proxy.cc` and `gin/gin_host_proxy.cc`), and log an INFO line when that lowers `NCCL_GIN_PROXY_QUEUE_SIZE`.
18. **A failed RMA post that accepted any WR fails its request.** `rcclRmaCompletePostedRequest` (`seg_rma_mseg.h`) frees the slot and returns the error when nothing posted. Any failed post that accepted at least one WR, including a batched doorbell failure after every WR was accepted, keeps the request (the accepted WRs reference its slot) and marks it FAILED; `Test()` then returns `ncclRemoteError` at once without draining, and the error is fatal for the context. A failed batch doorbell also marks FAILED every other request with a WR in that batch, including aggregated requests already returned (`rcclRmaIbProxyFailBatch`, `gin.cc`), because their signaled tail never posted. Segmented P2P sends drop the event of a QP whose slice did not post. The RMA proxy rings its pending batch before it publishes a queued request, so a doorbell failure leaves nothing queued.
19. **P2P sends post one slice per QP per progress call.** A segmented send keeps a `rcclIbSendCursor` (`common_mseg.h`); `rcclIbSendProgress` posts the next slice when the previous one completed (`rcclIbSendSliceDone`, `p2p_multiseg_mseg.h`), and slots post in FIFO order. So P2P send QPs keep develop's depth. See [send progress](multi_segment_netib_wire.md#send-progress).
20. **Flush is one read.** P2P direct-GDR flush (`IbMsegIflush`, `p2p.cc`) reads the first byte of the last non-empty receive with the MR of the segment that holds that byte, or posts develop's scratchpad read chained after the write. GIN flush reads one byte of `flushSeg`, the first device segment, and skips a window that is all host memory.
21. **Loopback flush QP device indices are set at create.** Both `devIndex` and `remDevIdx` name the receive device (`connect.cc`).

### NET/IB P2P wire format

22. **Both ends must make the same connect-time decision.** Each comm caches `useMultiSeg` from `IbMsegResolveUseMultiSeg` (`connect.cc`), which is `IbMsegUseMultiSeg(rcclMultiSegOn(true), forGin, resilient)` (`ib_multiseg_mseg.h`), and sends it in `IbMsegConnectionMetadata::useMultiSeg` (`connect_mseg.h`). The accepting side compares it with the peer's value (`IbMsegCheckPeerUseMultiSeg`) and, when they differ, WARNs (naming `NCCL_MULTI_SEGMENT_REGISTER`, `NCCL_DMABUF_ENABLE` and `NCCL_IB_RESILIENCY_PORT_FAILOVER`), closes the socket before replying and returns `ncclInvalidUsage`. The connecting side usually sees the socket close; its own check fires only if the peer replied. There is no capability negotiation and no fallback for a peer that decided otherwise.
23. **One CTS write.** A `useMultiSeg` comm uses `IbMsegCtsSlot` slots (`common_mseg.h`): `layout[8]` (`IbMsegSegLayout`, 192 bytes each) followed by `cts[8]`, 2048 bytes per slot and 256 slots (`NCCL_IB_CTS_SLOTS_BYTES`, 512 KiB). On each side the 128 KiB CTS FIFO becomes the 512 KiB slot array (384 KiB more per side): the receiver's legacy shadow FIFO (`IbMsegRemCtsFifo::elems`) is a pointer, allocated at accept only without `useMultiSeg`. With the slots the sender also allocates the send cursor and the referenced-layout cache (`sendCursor`, `layoutCache`, about 11 KiB); a legacy-CTS comm allocates only the cursor, on its first segmented send. The receiver posts the layout and the CTS as one RDMA write that ends at `idx`, so the sender never reads a layout before its CTS. A single-segment write starts at `cts`. One MR covers the slots.
24. **Inline or by reference.** A layout carries up to `RCCL_IB_CTS_INLINE_SEGS` (4) overlapping segments inline. A receive that overlaps more sends a `rcclIbSegLayoutRef` to the registration's published layout blob, and the sender reads it once into a 16-entry cache (`RCCL_IB_LAYOUT_CACHE_ENTRIES`, `seg_p2p_mseg.h`), keyed by address and generation.
25. **Bit 15 of `nreqs` marks a write with the layout prefix** (`NCCL_IB_CTS_NREQS_LAYOUT`). Only a `useMultiSeg` comm has the layout lines, and the connect check guarantees its peer has them too.
26. **The sender bounds the peer's `nreqs`.** `IbMsegIsend` rejects a peer-written count above `NCCL_NET_IB_MAX_RECVS`, the post path rejects a count outside 1 to 8, and the sender validates a layout (`IbMsegCtsRemoteLayoutValid`, `rcclIbLayoutBlobValid`) before use. A split failure, invalid peer layout, out-of-range device index or straddle WARNs and returns an error instead of posting a bad WR.
27. **Only P2P comms take the multi-segment CTS.** GIN and RMA comms and resilient comms keep develop's CTS FIFO.

### Windows and host memory

28. **Host VMM uses `hipMemLocationTypeHost` with id 0**, because the runtime rejects `HostNuma` (AIRUNTIME-2638). GIN stores host segments as `CU_MEM_LOCATION_TYPE_HOST_NUMA` (`dev_runtime.cc`) so the device put-fence treats them as sysmem.
29. **Mixed host and device windows fail closed.** With sysmem-handle reuse on, a host segment on one LSA rank and a device segment at the same index on another is rejected. On the non-symmetric path every LSA rank allgathers a fixed-size kind array before any IPC exchange (`rcclDevrProbeNonSymWindow`).
30. **Imported host segments register from an exporter's DMA-BUF fd**, because the importer cannot re-export them. A rank that exported its own host fd registers that fd. Another rank borrows an exporter's fd only when exactly one LSA rank reports `hostExportable` for that segment (`dev_runtime.cc`), and never falls back to a non-exporter. Ranks agree on the result (`hostOwnerFdStatus`, gathered as `ownerFdStatus`) before the collective registration, so a rank-local failure fails every rank. See [host owner DMA-BUF](multi_segment_gin_rma.md#host-owner-dma-buf).
31. **A window without an RMA registration never posts.** An RMA put on it returns `ncclInvalidUsage` (`ncclRmaProxyPutBuildOp`), and hierarchical CE falls back when either window has none (`ncclHierCeAvailable`).
32. **Develop's plugins are unchanged.** `net_ib` and `net_ib_cast` are develop code. Multi-segment code in IB-MSEG sits in its `seg_*` files and `p2p_multiseg.cc`, reached from `RCCL:`-marked hooks in the copied files; core changes are `RCCL:`-marked blocks.

## Limits

**Segment count.** The cap is `RCCL_IB_REG_SEG_DEFAULT` segments per registration, fixed at build time. A NET buffer above the cap or above `max_mr` stages; a GIN/RMA window above it fails registration on every rank. RMA also bounds `nranks * nSegments` at 2^20 table entries.

### QP depths

Depths at the fork 4 tip. With the switch off, every queue has develop's depth.

| Queue | IB-MSEG, switch on | Switch off |
|---|---|---|
| P2P sender data QP | 512 (`2 * NET_IB_MAX_REQUESTS`), develop's depth | same |
| P2P CTS QP | develop's depth | same |
| P2P loopback flush QP | 512 (`2 * NET_IB_MAX_REQUESTS`), develop's depth | same |
| GIN/RMA sender data QP | `RCCL_RMA_SEND_QP_WRS` = 32 × 65 = 2080 (`rcclIbSendQpWrs(isRma)`, `connect.cc`) | 512 |

`isRma` is `nQpsPerDev > 0`, so only GIN/RMA proxy comms take the deeper send queue: every per-peer context comm and also the GIN collective ring, which never posts multi-segment chains and pays memory only. One slice per progress call keeps P2P within develop's depth, and RMA rounds of at most 65 WRs with 32 requests in flight fit 2080. 2080 fits Thor2's `max_qp_wr` of 4351; Mellanox is not checked, and only a `static_assert` (`RCCL_RMA_SEND_QP_WRS >= 2 * NET_IB_MAX_REQUESTS`) guards the size.

## Switch-off behavior

With `NCCL_MULTI_SEGMENT_REGISTER=0` (the HIP default):

- IB-MSEG and `IbMsegRmaIbProxy` are not selected unless `NCCL_NET=IB-MSEG`. When they are, `rcclMultiSegOn` is false, so IB-MSEG behaves like develop's `net_ib`: no comm uses the multi-segment CTS and the multi-segment RMA operations are not installed.
- `comm->rcclMultiSegNet` and `comm->rcclMultiSegRma` are false. NET skips registration of a multi-segment buffer, including a one-segment slice of a multi-segment registration, and the collective stages.
- Develop's P2P, CTS, flush and RMA QP depths apply, and the RMA in-flight cap does not apply.
- `symMemoryRegisterGin` registers a device-only multi-segment window only when `rcclDevrGinRegistersMultiSegment` holds, which it cannot with the switch off. Otherwise a `NCCL_WIN_GIN_ONLY` window fails with `ncclInvalidUsage`, and any other window stays unregistered with GIN (INFO line) and remains usable for LSA. Windows that contain sysmem keep develop's per-segment path.
- `symMemoryRegisterRma` registers a multi-segment window only when `rcclDevrRmaRegistersMultiSegment` holds: `comm->rcclMultiSegRma` (the switch on, `NCCL_DMABUF_ENABLE=1` and the `IbMsegRmaIbProxy` backend), an equal count on every rank, and a table that fits; a window with host segments also needs `NCCL_SYM_REUSE_SYSMEM_HANDLES=1`. See [window gates](multi_segment_gin_rma.md#window-gates).
- Not gated by the switch, so they differ from develop even with it off: the host owner fd path, which runs whenever `NCCL_SYM_REUSE_SYSMEM_HANDLES=1` (export, fetch, the second allgather, publication, host marking, and `ncclGinRegister` calling `regMrSymDmaBuf` with the owner fd); the NULL-handle checks in `ncclRmaProxyPutBuildOp` and `ncclHierCeAvailable` (invariant 31); and the INFO line when RMA registration is skipped.

The switch selects the plugin, the handle type and the data path, so it is fixed for the life of the process and must be the same on every rank; a connection whose ends decide differently fails at connect (invariant 22), and the WARN asks for the same `NCCL_MULTI_SEGMENT_REGISTER`, `NCCL_DMABUF_ENABLE` and `NCCL_IB_RESILIENCY_PORT_FAILOVER` on every rank.

## Version gates and runtime tickets

- **Per-segment export** compiles when `CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE`. On HIP the gate needs the CMake probe `RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED` and a supported HIP version: native ROCm 7.12.60540 or later, or the 7.0.2.x backport (`ROCM_VER_7_0_2_2` up to `ROCM_VER_7_0_3_0`).
- **Host VMM** compiles under `NCCL_CUMEM_HOST_GATE`, which needs the CMake probe `RCCL_HIP_MEM_LOCATION_HOST_SUPPORTED` and `NCCL_CUMEM_HOST_VERSION_SUPPORTED`. Stock ROCm 7.0.2 has the backport's `HIP_VERSION` but no host location type. Tests no longer force-define the probe macro.
- **PCIe export flag.** `getHandleForAddressRangeFlags` must not set the PCIe DMA-BUF flag on the 7.0.2.2 backport; the multi-segment NET path passes the same helper's flags to IB-MSEG.
- **Runtime tickets.** ROCM-29812: CPU access to imported host VMM; hybrid tests skip collectively without it. ROCM-29810 (imported host VMM reported a device location) is fixed in the runtime, and fork 1 removes its TODO. AIRUNTIME-2638: `HostNuma` rejected. AIRUNTIME-2351: the original single-segment export limitation.

## Open follow-ups

- **Forks 5 and 6**: fork 5 is paused until forks 1 to 4 are stable on IB and AINIC, and fork 6 is not started. Their IB-CAST requirements from review are listed in [IB-CAST P2P open items](multi_segment_netib_cast.md#open-items).
- **Driver off-ramp**: no predicate yet lets a driver that exports whole ranges bypass per-segment registration; the workaround stays on whenever the switch is on.
- **Device-initiated GIN backends** (rocSHMEM GDA, Anvil SDMA): proposed in [rocSHMEM GDA and Anvil SDMA](multi_segment_gin_rocshmem_gda_and_anvil_sdma.md), not implemented; it waits on an owner decision.
- **Test and platform gaps** (Mellanox runs, the RMA send-queue depth on Mellanox, unported tests, switch-off performance) are listed in [known gaps](multi_segment_validation_plan.md#known-gaps).
- **Open review threads** are listed in [review feedback](multi_segment_review_feedback.md#open-items).

Develop already has layer 02 (`rcclDevrProbeNonSymWindow`, `rcclDevrGinSegmentRegType`, host location in `rocmwrap.cc` and `alloc.h`) and layer 03 (`rcclNetRegSegmentCount`, `rcclNetGraphRegSegmentCount`, `rcclNetRegCommit` in `rccl_wrap.cc`). The symbols each fork adds are in the [code map](multi_segment_code_map.md).
