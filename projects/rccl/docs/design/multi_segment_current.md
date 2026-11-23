# Multi-segment: current contract

> [!NOTE]
> This note describes the old stack (layers 05, 06, 08 and the IB-CAST GIN/RMA "wire" layer) at the wire tip `ba3ab26f30a7` on develop `c43b4a9e7307`. Layer 07 is folded into 05 and 06. The planned replacement keeps classic NET/IB and IB-CAST at develop and puts multi-segment into a separate IB-MSEG plugin; see [the fork plugin](multi_segment_fork_plugin.md). Where the fork differs from a rule below, the fork note says so.

This doc lists the rules the multi-segment code must keep, its limits, and what the switch does when it is off. It is the checklist for reviewing a change, not a design explanation; each item links to the doc that explains it. Terms are defined in the [overview](multi_segment_overview.md#terms). The review history behind these rules is in [review feedback](multi_segment_review_feedback.md).

## Invariants

### Registration

1. **One MR per physical segment, with no fixed 16-segment cap.** `rcclIbSegCountSupported` (`transport/rccl_mseg.h`) is the one count check: a count from 1 to `rcclIbRegSegCap` registers. The cap defaults to `RCCL_IB_REG_SEG_DEFAULT` (1024) and `RCCL_IB_MAX_REG_SEGMENTS` moves it within 1 to `RCCL_IB_SEG_PROTOCOL_MAX` (8192, the P2P IPC limit). Both plugins read the knob at init (`rcclIbRegSegCapFrom`), so it must match on every rank. See [limits](#limits).
2. **Single-segment buffers keep the develop path.** `nSegments == 1` reuses the caller's DMA-BUF fd and posts one WR per operation. In the P2P handle, `mrs[]` aliases segment 0, so single-segment code is unchanged.
3. **GIN/RMA exports whole pages.** Each segment is exported from the page that holds the segment pointer, rounded up to whole pages, and registered at the page-aligned distance from the segment base. NET P2P exports the exact segment range. See [GIN/RMA registration](multi_segment_gin_rma.md#registration).
4. **Host VMM length.** `cuMemGetAddressRange` can return `segBase == 0` for host VMM, so the bytes left in a segment are `segSize` when `segBase == 0`, else `segSize - (segPtr - segBase)` (`ncclIbBytesRemainingInSegment`).
5. **GIN/RMA registration is collective and always reaches the first gather.** Every rank allgathers a `rcclIbSegSymRecord {status, nSegments}` (`transport/rccl_mseg_sym.h`) into `regRecords`, even after a local failure, so no peer is left waiting. Classic allocates `regRecords` in `rcclRmaIbProxyConnectFinish`; IB-CAST adds `magic`, `version` and `ibDevNs[]` to its record. Any failed status fails every rank. See [consensus](multi_segment_gin_rma.md#consensus).
6. **GIN ranks need equal segment counts, not equal sizes.** `rcclIbSegSymAgree` requires the same `nSegments` on every rank; each rank's own offsets are kept in the symmetric table, and each side of a transfer splits on its own rank's row. RMA also needs `nranks * nSegments` to fit `RCCL_RMA_SEG_TABLE_MAX_ENTRIES` (2^20, `rcclRmaSegTableFits`).
7. **NET registration walks the whole range and rejects a partial walk.** `rcclIbRegMrDmaBufFromRange` (IB-CAST `IbCastRegMrDmaBufFromRange`) fails when the walk does not cover every byte, so no handle can advertise bytes beyond its last MR. Non-uniform layouts register; the earlier uniform-interior decline is gone.
8. **NET segment count is cached for the full registration.** `ncclReg::rcclNet.nSegments` holds the count of the whole registration, set by `rcclNetRegCommit` only after every peer registered.
9. **`net.cc` hands fd -1 only to a plugin that asks for it.** A plugin that sets `NCCL_PTR_DMABUF_MULTISEG` (0x8, `include/plugin/nccl_net.h`) gets `regMrDmaBuf` with fd -1 and walks the range itself; any other plugin sees develop's call (`rcclNetPluginMultiSeg`).
10. **Rollback deregisters only complete handles.** Registration goes through `rcclIbSegRegister` and `rcclIbSegDeregister` (`transport/rccl_mseg_reg.h`), which release exactly the MRs made so far. A failed registration leaves the caller's `*mhandle` untouched.
11. **The MR count is admitted against the device.** `rcclIbMrAdmit` (IB-CAST `rcclIbCastMrAdmit`) declines a registration that would push the device past `max_mr`, so a large segment count stages instead of failing in the verbs call.

### Data path

12. **Split at local and remote boundaries independently.** Local and remote request bases can start at different offsets in their registrations, so the splitters keep two tables in one `rcclIbSegCursor` (`rccl_mseg.h`) and emit slices with `rcclIbSegAppendWrs` (`rccl_mseg_wr.h`). Classic P2P, IB-CAST P2P and both RMA backends use the same cursor. See [the offset bug](multi_segment_netib_wire.md#independent-offsets).
13. **Slices are at most `RCCL_IB_SGE_MAX_BYTES` (`UINT32_MAX`) bytes**, because `ibv_sge::length` is 32 bits. IB-CAST also splits each WR at `IbCastMaxPostSendBytes`, the per-post cap that `RCCL_IB_MAX_MSG_SIZE` and the port set (`IbCastPostDataWithinCap` for P2P, `IbCastRmaCapWrs` for RMA).
14. **Only the final WR of a chain or round is signaled** (`rcclRmaWrIsSignaled`). `IPutSignal` signals only its trailing atomic, which must be 8-byte aligned and inside one segment (`rcclRmaSignalOffsetValid`).
15. **RMA posts in rounds.** An operation posts at most `NCCL_RMA_MAX_DATA_WRS` (64) data WRs per round, plus the atomic on the last round of `IPutSignal` (`NCCL_RMA_MAX_SIGNAL_WRS` = 65), and its next round once the plugin reports the previous one done. Operations queue per peer in request order (`rccl_mseg_rma_round.h`), so a later operation never passes an earlier one. See [rounds](multi_segment_gin_rma.md#rounds).
16. **RMA in-flight requests are capped.** With multi-segment chains possible, the proxies keep at most `NCCL_RMA_MAX_INFLIGHT_REQUESTS` (32) requests per peer and context (`rcclRmaProxyMaxInflightRequests` in `rccl_dev_runtime.cc`), and log an INFO line when that lowers `NCCL_GIN_PROXY_QUEUE_SIZE`.
17. **A prefix post keeps the request only if a completion will come.** `rcclRmaCompletePostedRequest` keeps a request whose signaled WR posted, marks it FAILED if the accepted prefix lost the signaled tail, and frees the slot when nothing posted. Classic and IB-CAST sends drop the event of a QP whose slice did not post. The RMA proxy rings its pending batch before it publishes a queued request, so a doorbell failure leaves nothing queued.
18. **P2P sends post one slice per QP per progress call.** A segmented send keeps a `rcclIbSendCursor`; `rcclIbSendProgress` posts the next slice when the previous one completed (`rcclIbSendSliceDone`), and slots post in FIFO order. So P2P send QPs keep develop's depth. See [send progress](multi_segment_netib_wire.md#send-progress).
19. **Flush is one read.** P2P direct-GDR flush reads the first byte of the last non-empty receive (`rcclIbIflushMultiSeg`), or posts develop's scratchpad read chained after the write. GIN flush reads one byte of `flushSeg`, the first device segment, and skips a window that is all host memory.
20. **Loopback flush QP device indices are set at create.** Both `devIndex` and `remDevIdx` name the receive device.

### NET/IB P2P wire format

21. **Capability in the metadata.** `ncclIbConnectionMetadata::caps` carries `RCCL_IB_CAP_MULTISEG` (bit 0). The `devName` trailer and its size assert are gone. Classic sets it from the cached `multiSegOk = ncclIbMultiSegOk(switch, forGin, resilient)`; IB-CAST from `IbCastMultiSegAdvertise(multiSegScopeOk, isRma)`.
22. **One CTS write.** An advertising comm uses `ncclIbCtsSlot` slots: `layout[8]` (`ncclIbSegLayout`, 192 bytes each) followed by `cts[8]`, 2048 bytes per slot and 256 slots (`NCCL_IB_CTS_SLOTS_BYTES`, 512 KiB), 384 KiB more per side than develop's FIFO. The receiver posts the layout and the CTS as one RDMA write that ends at `idx`, so the sender never reads a layout before its CTS. A single-segment write starts at `cts`. One MR covers the slots.
23. **Inline or by reference.** A layout carries up to `RCCL_IB_CTS_INLINE_SEGS` (4) overlapping segments inline. A receive that overlaps more sends a `rcclIbSegLayoutRef` to the registration's published layout blob, and the sender reads it once into a 16-entry cache (`RCCL_IB_LAYOUT_CACHE_ENTRIES`), keyed by address and generation.
24. **Bit 15 of `nreqs` marks a write with the layout prefix** (`NCCL_IB_CTS_NREQS_SIDE_TABLE`), sent only to a peer that advertised.
25. **A missing peer capability declines only the receiver** (`ncclIbDeclineMultiSegRegistration`). A sender splits its own buffer onto the receiver's single rkey and needs no capability.
26. **The sender bounds the peer's `nreqs`** to 1 to `NCCL_NET_IB_MAX_RECVS` and validates a layout (`ncclIbCtsRemoteLayoutValid`, `rcclIbLayoutBlobValid`) before use.
27. **Only P2P comms advertise.** GIN and RMA comms never advertise or allocate CTS slots.
28. **Classic picks the remote stride from the peer's raw capability.** The receiver allocates the combined slots for a capable peer. IB-CAST masks `peerCaps` with its own scope; see the open item in [IB-CAST P2P](multi_segment_netib_cast.md#open-items).

### IB-CAST specifics

29. **IB-CAST multi-segment scope.** `IbCastMultiSegScopeOk` excludes CTS offload, BY_ORDER matching, QP sharing and resiliency, and gates registration on both sides. AINIC defaults to BY_ORDER, so it needs `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0`. See [IB-CAST scope](multi_segment_netib_cast.md#scope).
30. **One segmented group at a time on IB-CAST** (`rcclIbCastHasOtherSegmentedSend`); a later slot defers.
31. **IB-CAST RMA looks up rkeys by physical `ibDevN`** (`rcclRmaDevSlotOf`), not by the QP's `remDevIdx`.
32. **The IB-CAST RMA sender CQ covers the send queue.** A dead QP flushes every queued WR as a CQE, so with the switch on the RMA sender CQ is at least `NCCL_RMA_SEND_QP_WRS`. AINIC recovery rebuilds RMA QPs through the same `IbCastBuildDataQpCreateAttr` as connect.

### Windows and host memory

33. **Host VMM uses `hipMemLocationTypeHost` with id 0**, because the runtime rejects `HostNuma` (AIRUNTIME-2638). GIN stores host segments as `CU_MEM_LOCATION_TYPE_HOST_NUMA` so the device put-fence treats them as sysmem.
34. **Mixed host and device windows fail closed.** With sysmem-handle reuse on, a host segment on one LSA rank and a device segment at the same index on another is rejected. On the non-symmetric path every LSA rank allgathers a fixed-size kind array before any IPC exchange (`rcclDevrProbeNonSymWindow`).
35. **Imported host segments register from an exporter's DMA-BUF fd**, because the importer cannot re-export them. The importer borrows the fd of the first LSA rank with `hostExportable` and never falls back to a non-exporter. Ranks agree on the result (`hostOwnerFdStatus`, gathered as `ownerFdStatus`) before the collective registration, so a rank-local failure fails every rank. See [host owner DMA-BUF](multi_segment_gin_rma.md#host-owner-dma-buf).
36. **Develop's functions carry only one-line hooks.** Multi-segment bodies are separate `RCCL:`-marked functions or files. See [sync and removal](multi_segment_sync_and_removal.md).

## Limits

**Segment count.** The protocol ceiling is 8192 segments (`RCCL_IB_SEG_PROTOCOL_MAX`), the default local cap 1024 (`RCCL_IB_MAX_REG_SEGMENTS`). A NET buffer above the cap or above `max_mr` stages; a GIN window above it fails registration on every rank. RMA also bounds `nranks * nSegments` at 2^20 table entries. The [past 16 segments](multi_segment_varseg.md) design is implemented in a different form; that note records the original plan.

### QP depths

Depths with the switch on, at the wire tip. With the switch off, every queue has develop's depth.

| Queue | Classic | IB-CAST | Switch off |
|---|---|---|---|
| P2P sender data QP | 512 (`2 * NET_IB_MAX_REQUESTS`) | develop's `IbCastDataSendQueueDepth` | same |
| P2P CTS QP | develop's depth | develop's depth | same |
| P2P loopback flush QP | 512 | 256 (`NET_IB_MAX_REQUESTS`) | same |
| RMA sender data QP | `NCCL_RMA_SEND_QP_WRS` = 32 × 65 = 2080 (`rcclIbSendQpWrs(isRma)`) | max of develop's depth and 2080 | develop's |
| RMA sender CQ | develop's | at least 2080 | develop's |

`isRma` is `nQpsPerDev > 0`, so only GIN/RMA proxy comms take the deeper send queue. One slice per progress call keeps P2P within develop's depth, and RMA rounds of at most 65 WRs with 32 requests in flight fit 2080. The WR credit scheme of the 11-04 stack was removed with this change. Whether every HCA accepts 2080 for `max_qp_wr` is not recorded.

## Switch-off behavior

With `NCCL_MULTI_SEGMENT_REGISTER=0` (the HIP default):

- Neither plugin advertises the capability or allocates CTS slots. IB-CAST also masks the peer's capability to 0.
- NET skips registration of a multi-segment buffer, including a one-segment slice of a multi-segment registration, and the collective stages.
- Develop's P2P, CTS, flush and RMA QP depths apply, and the RMA in-flight cap does not apply.
- Classic and IB-CAST `gin.cc` run develop's bodies.
- `symMemoryRegisterGin` fails a device-only multi-segment window with `ncclInvalidUsage` unless `rcclDevrGinRegistersMultiSegment` holds. Windows that contain sysmem keep develop's per-segment path.
- `symMemoryRegisterRma` registers a multi-segment window only when `rcclDevrRmaRegistersMultiSegment` holds: the switch on, `NCCL_DMABUF_ENABLE=1`, the `ncclRmaIbProxy` or `IbCastRmaIbProxy` backend, an equal count on every rank, and a table that fits. See [window gates](multi_segment_gin_rma.md#window-gates).

The switch selects both the handle type and the data path, so it is fixed for the life of the process and must match on every rank. Nothing on the old stack detects a mismatch.

## Version gates and runtime tickets

- **Per-segment export** compiles when `CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE`. On HIP the gate needs the CMake probe `RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED` and a supported HIP version: native ROCm 7.12.60540 or later, or the 7.0.2.x backport (`ROCM_VER_7_0_2_2` up to `ROCM_VER_7_0_3_0`).
- **Host VMM** compiles under `NCCL_CUMEM_HOST_GATE`, which needs the CMake probe `RCCL_HIP_MEM_LOCATION_HOST_SUPPORTED` and `NCCL_CUMEM_HOST_VERSION_SUPPORTED`. Stock ROCm 7.0.2 has the backport's `HIP_VERSION` but no host location type. Tests no longer force-define the probe macro.
- **PCIe export flag.** `getHandleForAddressRangeFlags` must not set the PCIe DMA-BUF flag on the 7.0.2.2 backport; the multi-segment path uses the same helper.
- **Runtime tickets.** ROCM-29812: CPU access to imported host VMM; hybrid tests skip collectively without it. ROCM-29810: imported host VMM reports a device location; the old stack keeps its TODO, fork 1 removes it. AIRUNTIME-2638: `HostNuma` rejected. AIRUNTIME-2351: the original single-segment export limitation.

## Code map

This table is the per-branch summary. The [code map](multi_segment_code_map.md) lists the symbols per file.

| Branch (`users/atulkulk/mseg/...`) | Multi-segment code |
|---|---|
| `02-amd-host-vmm-compat` (in develop) | `rccl_dev_runtime.cc`: `rcclDevrProbeNonSymWindow`, `rcclDevrGinSegmentRegType`; host location in `rocmwrap.cc`, `alloc.h` |
| `03-registration-range` (in develop) | `rccl_wrap.cc`: `rcclNetRegSegmentCount`, `rcclNetGraphRegSegmentCount`, `rcclNetRegCommit` |
| `05-gin-rma-multisegment` | shared `transport/rccl_mseg.h`, `rccl_mseg_rma.h`, `rccl_mseg_rma_round.h`, `rccl_mseg_sym.h`, `rccl_mseg_export.h`, `rccl_mseg_reg.h`, `rccl_mseg_wr.h`; `net_ib/ib_multiseg.h`; classic RMA bodies in `net_ib/gin.cc`; window gates, in-flight cap and owner fd in `rccl_dev_runtime.cc` and `dev_runtime.cc` |
| `06-classic-netib-multisegment` | shared `transport/rccl_p2p_multiseg.{h,cc}`; `net_ib/p2p_multiseg.{h,cc}` (send cursor, fill, flush, CTS hooks); hooks in `net_ib/p2p.cc`, `reg.cc`, `connect.cc`, `init.cc`; `transport/net.cc` (fd -1 behind the plugin bit) |
| `08-cast-netib-multisegment` | `net_ib_cast/ib_cast_multiseg.h`; IB-CAST P2P bodies in `net_ib_cast/p2p.cc`, registration in `net_ib_cast/reg.cc`, scope and CTS slots in `connect.cc` and `connect_cast.h` |
| `multi-segment-dmabuf-cast-gin` | IB-CAST RMA bodies in `net_ib_cast/gin.cc`, RMA depths and CQ floor in `net_ib_cast/connect.cc`, AINIC abandon hook |
| `09-tests-local` (local rebase of `users/bdoganca/mseg/09-mseg-tests-coverage`) | tests and runner configs only |

`net_ib/gin.h` and `net_ib_cast/gin_cast.h` share an include guard and must not be included together.

## Open follow-ups

- **Fork plugin**: the planned replacement of this stack; see [the fork plugin](multi_segment_fork_plugin.md).
- **IB-CAST P2P scope and stride**: two Highs from the 2026-11-14 local review are open on 08 and carried to fork 6; see [review feedback](multi_segment_review_feedback.md#summary).
- **Driver off-ramp**: the predicate that would let a driver fix bypass the workaround is not implemented; see [sync and removal](multi_segment_sync_and_removal.md#step-1-switch-it-off-without-deleting-code).
- **Device-initiated GIN backends** (rocSHMEM GDA, Anvil SDMA): proposed in [multi_segment_gin_rocshmem_gda_and_anvil_sdma.md](multi_segment_gin_rocshmem_gda_and_anvil_sdma.md).
- **IB-CAST RMA with subnet-aware routing**: under `NCCL_IB_SUBNET_AWARE_ROUTING=1` IB-CAST RMA fails with "no registration slot". The gap exists on develop too; see [the validation plan](multi_segment_validation_plan.md#known-gaps).
- **Superseded follow-ups**: [side table on first registration](multi_segment_sidetable_first_reg.md) (the side table is gone) and [flush read collapse](multi_segment_flush_read_collapse.md) (the flush is one read).
