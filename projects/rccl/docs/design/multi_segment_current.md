# Multi-segment: current contract

This doc lists the rules the multi-segment code must keep, its limits, and what the switch does when it is off. It is the checklist for reviewing a change, not a design explanation; each item links to the doc that explains it. Terms are defined in the [overview](multi_segment_overview.md#terms). The facts here match the wire branch `users/atulkulk/mseg/multi-segment-dmabuf-cast-gin`.

## Invariants

### Registration

1. **One MR per physical segment, at most 16.** `NCCL_IB_MAX_SEGMENTS` and `NCCL_RMA_MAX_SEGMENTS` are both 16 and are asserted equal. A buffer with more segments is rejected (GIN/RMA) or staged (NET). See [the cap](#limits).
2. **Single-segment buffers keep the develop path.** `nSegments == 1` reuses the caller's DMA-BUF fd and posts one WR per operation. In the P2P handle, `mrs[]` aliases segment 0, so single-segment code is unchanged.
3. **GIN/RMA exports whole pages.** Each segment is exported from the page that holds the segment pointer, rounded up to whole pages, and registered at DMA-BUF offset 0. NET P2P exports the exact segment range. See [GIN/RMA registration](multi_segment_gin_rma.md#registration).
4. **Host VMM length.** `cuMemGetAddressRange` can return `segBase == 0` for host VMM, so the bytes left in a segment are `segSize` when `segBase == 0`, else `segSize - (segPtr - segBase)` (`ncclIbBytesRemainingInSegment` on the NET path).
5. **GIN registration is collective and always reaches the first gather.** Every rank allgathers its consensus record into the per-communicator `regConsensus` buffer, even after a local failure, so no peer is left waiting. The buffer is allocated at connect, only with the switch on. Any failed status fails every rank. A second gather reconciles allocation failures before the address and rkey gathers. See [consensus](multi_segment_gin_rma.md#consensus).
6. **GIN ranks need equal segment counts, not equal sizes.** `ncclRmaSegmentCountsMatch` requires the same `nSegments` in 1..16 on every rank; each rank's own `segOff` row is kept in `rankSegOff`, and each side of a transfer splits on its own rank's row. `ncclRmaSegOffTableValid` rejects a peer table that does not start at 0 or that decreases.
7. **NET registration walks the whole range and rejects a partial walk.** `rcclIbRegMrDmaBufFromRange` (IB-CAST `IbCastRegMrDmaBufFromRange`) fails when the walk does not cover every byte or yields a different count, so no handle can advertise bytes beyond its last MR. Interior segments must equal the largest one; the first and last may be shorter (`ncclIbSegmentsUniform`), so every two-segment layout passes.
8. **NET segment count is cached for the full registration.** `ncclReg::rcclNet.nSegments` holds the count of the whole registration, set by `rcclNetRegCommit` only after every peer registered. The walk covers `endAddr - begAddr - pageSize + 1` bytes so it ends inside the last page of an unrounded `hipMalloc`.
9. **Rollback deregisters only complete handles.** Failure paths free exactly the MRs they made; P2P frees through `segMrs[][]`, never the `mrs[]` alias, so nothing is freed twice.

### Data path

10. **Split at local and remote boundaries independently.** Local and remote request bases can start at different offsets in their registrations, so the splitters keep two cursors (`ncclIbSplitTransferAtOffsets`, `ncclRmaBuildSegmentedWrs`). One shared offset picks the wrong key. See [the offset bug](multi_segment_netib_wire.md#independent-offsets).
11. **Slices are at most `UINT32_MAX` bytes**, because `ibv_sge::length` is 32 bits.
12. **Only the final WR of a chain is signaled** (`ncclRmaWrIsSignaled`). `IPutSignal` signals only its trailing atomic, which must be 8-byte aligned and inside one segment (`ncclRmaSignalOffsetValid`).
13. **GIN WR budget is 64 data WRs per operation** (`NCCL_RMA_MAX_DATA_WRS = 4 * 16`), plus one atomic for `IPutSignal` (65). A 16-segment 8 GiB window with `UINT32_MAX` slices needs 48. A wider chain is rejected before posting.
14. **Prefix post keeps the request and returns success.** If `ibv_post_send` accepts part of a chain, the request stays so `Test()` can drain the completion; if the accepted part lost the signaled tail, the request is marked FAILED. Only a post that accepted nothing frees the slot and returns the error (`ncclRmaCompletePostedRequest`). The P2P flush and the IB-CAST send follow the same rule.
15. **Flush fences every segment.** GIN flush posts one 1-byte read per segment of the window. P2P direct-GDR flush posts one read per segment each receive overlaps; the default scratchpad flush still posts a single read.
16. **Loopback flush QP device indices are set at create.** Both `devIndex` and `remDevIdx` name the receive device; otherwise flush reads uninitialized slots.
17. **Send queues are sized for the chains in flight.** Every unsignaled slice holds a send-queue slot until the chain's completion. GIN/RMA reserves credits before posting on both backends; P2P sizes the QP for the worst case instead. See [QP depths](#qp-depths).

### NET/IB P2P wire format

18. **The CTS entry stays 64 bytes** (32 bytes inline on IB-CAST AINIC). The receiver's segment table travels in the side table behind `NCCL_IB_CAP_MULTISEG`, carried in the `devName` capability trailer; the connect metadata size and offsets do not change.
19. **A missing peer capability declines only the receiver.** A receive buffer needs the peer to read its side table; a sender splits its own buffer onto the receiver's single rkey and needs no capability.
20. **Classic writes the side table only for slots with a multi-segment receive**, flags every CTS entry of that slot with bit 31 of `nreqs` (`NCCL_IB_CTS_NREQS_SIDE_TABLE`), and chains the unsignaled side WR before the CTS WR with the CTS `wr_id`. IB-CAST writes it for every slot to a capable peer.
21. **The sender bounds the peer's `nreqs`** to at most `NCCL_NET_IB_MAX_RECVS` (IB-CAST also at least 1) before waiting on the side table, and validates the side entry with `ncclIbCtsRemoteLayoutValid` before use.
22. **A segmented receive reports its full size.** Classic writes the sizes WR when `nreqs > 1` or when matching is not by index and the size is nonzero; IB-CAST always writes it for a segmented send.
23. **`remSegLayout` exists only for a capable peer.** The receiver's 1.44 MB staging table is allocated and registered only when the peer advertised the capability.

### IB-CAST specifics

24. **IB-CAST P2P declines on both sides** under CTS offload, BY_ORDER matching or QP sharing, and on the receive side for a peer without the capability. See [IB-CAST declines](multi_segment_netib_cast.md#when-multi-segment-is-declined).
25. **IB-CAST RMA looks up rkeys by physical `ibDevN`** (`ncclRmaDevSlotOf`), not by the QP's `remDevIdx`.
26. **IB-CAST RMA credit waits are bounded** by `4.096 µs × 2^NCCL_IB_TIMEOUT × (NCCL_IB_RETRY_CNT + 1)`, at least 2 s, and an error completion marks its owner FAILED even when another request drained it. See [IB-CAST credits](multi_segment_netib_cast_gin.md#credits).

### Windows and host memory

27. **Host VMM uses `hipMemLocationTypeHost` with id 0**, because the runtime rejects `HostNuma` (AIRUNTIME-2638). GIN stores host segments as `CU_MEM_LOCATION_TYPE_HOST_NUMA` so the device put-fence treats them as sysmem.
28. **Mixed host and device windows fail closed.** With sysmem-handle reuse on, a host segment on one LSA rank and a device segment at the same index on another is rejected. On the non-symmetric path every LSA rank allgathers a fixed-size kind array before any IPC exchange (`rcclDevrProbeNonSymWindow`), so a failing rank cannot strand its peers.
29. **Imported host segments register from the owner's DMA-BUF fd**, because the importer cannot re-export them. See [host owner DMA-BUF](multi_segment_gin_rma.md#host-owner-dma-buf).
30. **Develop's functions carry only one-line hooks.** Multi-segment bodies are separate `RCCL:`-marked functions in the file they hook. See [sync and removal](multi_segment_sync_and_removal.md).

## Limits

**Segment cap.** At most 16 segments per buffer, which bounds the smallest segment size for a given window: a 1 GiB window needs segments of at least 64 MiB. A NET buffer with more segments stages; a GIN window with more fails registration. [Past 16 segments](multi_segment_varseg.md) is the design to lift this.

**Uniform interior segments (P2P).** Registration declines a layout whose interior segments differ from the largest one, although the splitter handles it. Relaxing the guard is a possible follow-up.

### QP depths

Depths with the switch on, on the wire tip. With the switch off, every queue has develop's depth.

| Queue | Classic | IB-CAST | Switch off |
|---|---|---|---|
| P2P sender data QP | `NCCL_IB_MAX_SEND_WRS` = `2*256 + NCCL_STEPS*257` = 2568 | 769 (`2*256 + 257`), one segmented group at a time | 512 |
| P2P CTS QP | develop's depth plus `256 * (resiliency ? 1 : 2)` | develop's depth times 2 | develop's depth |
| P2P loopback flush QP | 1024 when the capability is advertised, else 512 | 1024 (`IbCastFlushQpSendDepth`) | classic 512, IB-CAST 256 |
| RMA send QP | shares the P2P sender QP: `NCCL_RMA_SEND_QP_WRS` = 2568 | `NCCL_RMA_MAX_SEND_QP_WRS` = 575 (`2*256 + 65 - 2`) | 512 |
| RMA flush QP | shares the P2P flush QP | `NCCL_RMA_MAX_FLUSH_QP_WRS` = 271 (`256 + 16 - 1`) | classic 512, IB-CAST 256 |

Layer 05 on its own (without 06) sizes the classic RMA sender to 577 (`2*256 + 65`) and the flush QP to 528. `static_assert`s in `net_ib/connect.cc` keep `NCCL_RMA_SEND_QP_WRS` equal to `NCCL_IB_MAX_SEND_WRS` and check that the P2P depths also fit the GIN signal and flush chains. Do not size a QP for `32 * 257`; providers reject that `max_qp_wr`.

Worst-case chain widths: a P2P send is 257 WRs (`8 * 2 * 16` slices plus the immediate), a P2P direct-GDR flush is 128 reads (`8 * 16`), a GIN `IPutSignal` is 65 and a GIN flush is 16. Classic P2P has no admission gate, so its sender holds one maximum chain per proxy step; IB-CAST admits one segmented group at a time instead. Classic and IB-CAST RMA reserve send-queue credits before posting: classic by sequence number with no deadline, IB-CAST by a counter with the bounded wait in item 26.

## Switch-off behavior

With `NCCL_MULTI_SEGMENT_REGISTER=0`:

- Neither plugin writes the capability trailer or sizes the CTS MR for the side table. IB-CAST also masks the peer's capability to 0.
- NET skips registration of a multi-segment buffer, including a one-segment slice of a multi-segment registration, and the collective stages. Called directly, the plugin declines a multi-segment registration on the receiver and accepts it on the sender.
- Develop's P2P, CTS, flush and RMA QP depths apply, `regConsensus` is not allocated, and the IB-CAST AINIC abandon hook does nothing.
- Classic and IB-CAST `gin.cc` run develop's bodies.
- `symMemoryRegisterGin` fails a device-only multi-segment window with `ncclInvalidUsage` ("GIN needs NCCL_MULTI_SEGMENT_REGISTER=1"); windows that contain sysmem keep develop's per-segment path.
- `symMemoryRegisterRma` registers a multi-segment window only when `rcclDevrRmaRegistersMultiSegment` holds; see [window gates](multi_segment_gin_rma.md#window-gates).

The switch selects both the handle type and the data path, so it is fixed for the life of the process and must match on every rank.

## Version gates and runtime tickets

- **Per-segment export** compiles when `CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE`. On HIP the gate needs the CMake probe `RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED` and a supported HIP version: native ROCm 7.12.60540 or later, or the 7.0.2.x backport (`ROCM_VER_7_0_2_2` up to `ROCM_VER_7_0_3_0`). Without it, GIN registration of more than one segment fails with "multi-segment registration requires HIP >= 7.12.60540".
- **Host VMM** compiles under `NCCL_CUMEM_HOST_GATE`, which needs the CMake probe `RCCL_HIP_MEM_LOCATION_HOST_SUPPORTED` and `NCCL_CUMEM_HOST_VERSION_SUPPORTED` (same versions). Stock ROCm 7.0.2 has the backport's `HIP_VERSION` but no host location type. On the backport `hip_compat.h` aliases `CU_MEM_LOCATION_TYPE_HOST_NUMA` until the hipify mapping ships (LCOMPILER-2603).
- **PCIe export flag.** `getHandleForAddressRangeFlags` must not set the PCIe DMA-BUF flag on the 7.0.2.2 backport; the multi-segment path uses the same helper, so it inherits the guard.
- **Runtime tickets.** ROCM-29812: CPU access to imported host VMM; hybrid tests skip collectively without it. ROCM-29810: imported host VMM reports a device location, which the owner-propagation code works around. AIRUNTIME-2638: `HostNuma` rejected. AIRUNTIME-2351: the original single-segment export limitation.

## Code map

This table is the per-branch summary. The [code map](multi_segment_code_map.md) lists every symbol per file, with line numbers and links to the design sections.

| Branch (`users/atulkulk/mseg/...`) | Multi-segment code |
|---|---|
| `02-amd-host-vmm-compat` | `rccl_dev_runtime.cc`: `rcclDevrProbeNonSymWindow`, `rcclDevrGinSegmentRegType`, `rcclDevrRmaRegistersMultiSegment`; host location in `rocmwrap.cc`, `alloc.h` |
| `03-registration-range` | `rccl_wrap.cc`: `rcclNetRegSegmentCount`, `rcclNetGraphRegSegmentCount`, `rcclNetRegCommit` |
| `05-gin-rma-multisegment` | `net_ib/rma_multiseg.h` (shared limits and helpers), `net_ib/gin.cc` (classic RMA bodies), host owner fd in `dev_runtime.cc` and `rccl_dev_runtime.cc` |
| `06-classic-netib-multisegment` | `net_ib/multiseg.h` (boundary math, trailer, CTS rules), `net_ib/p2p.cc`, `net_ib/reg.cc`, `net_ib/connect.cc`, `transport/net.cc` (`fd = -1` call), `transport/rccl_ib_multiseg.h` (private declarations) |
| `07-hybrid-stress-ci` | tests and runner configs only |
| `08-cast-netib-multisegment` | `net_ib_cast/p2p.cc`, `net_ib_cast/reg.cc`, `net_ib_cast/rccl_reg_multiseg_cast.cc`, `net_ib_cast/connect.cc` |
| `multi-segment-dmabuf-cast-gin` | `net_ib_cast/gin.cc` (IB-CAST RMA bodies) |

The shared headers are the only code both plugins include: `net_ib/multiseg.h` for P2P and `net_ib/rma_multiseg.h` for RMA. `net_ib/gin.h` and `net_ib_cast/gin_cast.h` share an include guard and must not be included together. `ncclIbRegMrDmaBufMultiSeg` and its IB-CAST twin are declared in `rccl_ib_multiseg.h`, not `net.h`. Per-path symbol indexes are in each transport doc.

## Open follow-ups

- **Past 16 segments**: design in [multi_segment_varseg.md](multi_segment_varseg.md); a follow-up pull request removes the cap.
- **Side table on first registration**: design in [multi_segment_sidetable_first_reg.md](multi_segment_sidetable_first_reg.md).
- **Flush read collapse**: design in [multi_segment_flush_read_collapse.md](multi_segment_flush_read_collapse.md).
- **Driver off-ramp**: the predicate that would let a driver fix bypass the workaround is not implemented; see [sync and removal](multi_segment_sync_and_removal.md#step-1-switch-it-off-without-deleting-code).
- **GIN RailA2A launch on ROCm 7.0.2.x**: `UBR_MultiSegment.Symmetric_LsaGin` stays in a disabled runner suite until the launch fix lands; see [known failures](multi_segment_testing.md#known-failures-and-dependencies).
- **Device-initiated GIN backends** (rocSHMEM GDA, Anvil SDMA): proposed in [multi_segment_gin_rocshmem_gda_and_anvil_sdma.md](multi_segment_gin_rocshmem_gda_and_anvil_sdma.md).
