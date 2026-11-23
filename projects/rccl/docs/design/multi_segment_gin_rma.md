# GIN/RMA: per-segment registration and RMA rounds

> [!NOTE]
> Old stack, layer 05 ([PR 10884](https://github.com/ROCm/rocm-systems/pull/10884)), at the wire tip `ba3ab26f30a7`. The 11-04 version of this note described a 16-segment cap, a fixed-size consensus record with `segOff[17]`, one flush read per segment and send-queue credits; all four are gone. In the planned design the core part of this layer (window gates, owner fd, in-flight cap) is fork 1 and the classic RMA backend moves into the IB-MSEG plugin in fork 4; see [the fork plugin](multi_segment_fork_plugin.md).

GIN on the host-proxy type posts every put, get, signal and flush through an RMA backend, so multi-segment support for GIN is a host-side change in that backend. This doc covers the classic backend `ncclRmaIbProxy` (`net_ib/gin.cc`) and the parts IB-CAST shares with it; IB-CAST's own differences are in [IB-CAST GIN/RMA](multi_segment_netib_cast_gin.md). It also covers the device-runtime pieces that DeepEP windows need: the window gates, the in-flight cap and the host owner DMA-BUF path.

## Why GIN is different from P2P

A GIN window is symmetric: one collective registers it on every rank, and each rank keeps a table of every peer's per-segment offsets, base addresses and rkeys. So the remote layout is known at registration, and no CTS is involved. Registration counts the segments, registers one MR per segment, checks through the [consensus exchange](#consensus) that every rank succeeded with the same segment count, and then allgathers the tables.

Each side of a transfer splits on its own rank's offsets, so ranks may have different segment sizes as long as the counts match:

```mermaid
%% caption: Two ranks with equal segment counts but different segment sizes
sequenceDiagram
 box rgb(229,241,255) GIN
 participant R0 as rank 0, 4 MiB GPU then 2 MiB CPU
 participant R1 as rank 1, 2 MiB GPU then 4 MiB CPU
 end
 R0->>R1: allgather status and nSegments 2
 Note over R0,R1: counts match, sizes differ
 R0->>R1: iput 6 MiB, split on rank 0's offsets locally and rank 1's remotely
```

## Thread hierarchy mapping

A GIN operation starts in a GPU thread group and ends as work requests on a QP. Only lane 0 of the cooperative group writes the 128-byte descriptor into the per-peer ring in host-pinned memory; the kernel chooses the context through the `ncclGin` context index. The host proxy progress thread polls every context and peer and calls the RMA backend, which posts on that context's QP to the peer. Multi-segment support changes only the last step. Same-node peers use LSA loads and stores and never reach the proxy.

```mermaid
%% caption: Thread hierarchy mapping from a GPU thread group to a QP on the NIC
flowchart TD
 subgraph gpu["GPU kernel on the user stream"]
  direction LR
  grid["grid"] --> block["block"] --> coop["cooperative group"]
 end
 gpu --> lane["lane 0 writes a 128-byte descriptor"]
 lane --> ring["per-peer ring in host-pinned memory"]
 ring --> poll["host proxy thread polls contexts and peers"]
 poll --> rma["RMA backend: iput, iputSignal, iget, iflush"]
 rma --> split["rounds of per-segment WRs on the context QP"]
 split --> done["NIC completion advances the counter"]
 gpu -.->|"same node"| lsa["LSA loads and stores, no proxy"]
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef rma fill:#f5ebfb,stroke:#af52de,color:#1d1d1f
 classDef hostmem fill:#f2f2f7,stroke:#8e8e93,color:#1d1d1f
 classDef devmem fill:#eeeefb,stroke:#5856d6,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 class grid,block,coop,lane gin
 class poll,rma rma
 class ring hostmem
 class split,done okpath
 class lsa devmem
```

## Code layout and dispatch

With the switch on, `ncclRmaIbProxyInit` installs the multi-segment entries (`rcclRmaIbProxyInstallMultiSeg`): every MR and data operation takes the `rcclRmaIbProxy*` body, because the multi-segment handle has its own layout. A one-segment handle runs develop's single-MR body through the handle's embedded `single` MR (`rcclRmaIbProxyUseSingle`).

| Develop function (`net_ib/gin.cc`) | Multi-segment body |
|---|---|
| `ncclRmaIbProxyConnect` | `rcclRmaIbProxyConnectFinish`, which allocates `regRecords` |
| `RegMrSymDmaBuf`, `RegMrSym`, `DeregMrSym` | `rcclRmaIbProxyRegMrSymDmaBuf`, `rcclRmaIbProxyRegMrSym`, `rcclRmaIbProxyDeregMrSym` |
| `IPut`, `IGet`, `IPutSignal`, `IFlush` | `rcclRmaIbProxyIPut`, `IGet`, `IPutSignal`, `IFlush` |
| `Test` | `rcclRmaIbProxyTestFailed`, for a request marked `NCCL_NET_IB_REQ_FAILED` |
| close | `free(cComm->regRecords)` |

The pieces both IB plugins share are in `src/transport/`:

| Header | Contents |
|---|---|
| `rccl_mseg.h` | segment limits (`RCCL_IB_SEG_PROTOCOL_MAX`, `RCCL_IB_REG_SEG_DEFAULT`, `rcclIbRegSegCap`, `rcclIbSegCountSupported`), `rcclIbSegTable`, `rcclIbSegCursor`, `rcclIbSegIndexOf`, `RCCL_IB_SGE_MAX_BYTES` |
| `rccl_mseg_rma.h` | RMA limits, `rcclRmaSegTableFits`, `rcclRmaInflightRequestsCap`, `rcclRmaCompletePostedRequest`, `rcclRmaDevSlotOf`, `rcclRmaSegMrTag`, `rcclRmaMrSegmentCount` |
| `rccl_mseg_rma_round.h` | rounds and per-peer queues |
| `rccl_mseg_sym.h` | `rcclIbSegSymRecord`, `rcclIbSegSymTable`, `rcclIbSegSymAgree`, `rcclIbSegSymExchange` |
| `rccl_mseg_export.h`, `rccl_mseg_reg.h`, `rccl_mseg_wr.h` | per-segment export, register and deregister, `rcclIbSegAppendWrs` |

Classic-only glue is in `net_ib/ib_multiseg.h` (`ncclIbMultiSegOk`, `NCCL_RMA_IB_WR_BATCHSIZE` 64, the `rcclRmaIbProxy*` declarations).

## Handle

```c
struct rcclRmaIbProxySegMrHandle {      // IB-CAST: IbCastRmaProxySegMrHandle
  uint32_t tag;                         // rcclRmaSegMrTag, asserted at offset 0
  int nSegments;
  size_t* segOff;                       // nSegments + 1, from the walked count
  void** mrHandle;                      // one ncclIbMrHandle per segment
  struct rcclIbSegSymTable sym;         // every rank's offsets, VAs and per-port rkeys
  int flushSeg;                         // first device segment, or -1 if all host
  struct ncclRmaIbProxyMrHandle single; // nSegments == 1: develop's layout over mrHandle[0]
};
```

Tests read the segment count through `rcclRmaMrSegmentCount`, which checks the tag, so the same check works on both backends. Classic indexes rkeys by the QP's `remDevIdx`; IB-CAST translates a physical `ibDevN` to a registration slot.

## Registration

```text
count segments: ncclCuMemGetAddressRange -> nSeg
nSeg outside rcclIbSegCountSupported, or table too big (rcclRmaSegTableFits), or past max_mr: local failure, still exchanged
nSeg == 1: reuse the caller's fd
for each segment s:
    cuMemGetAddressRange -> segBase, segSize
    inSeg   = (segBase == 0) ? segSize : segSize - (segPtr - segBase)
    thisLen = min(remaining, inSeg)
    fd = owner fd for this VA if one is published, else export whole pages
    register one MR per device through rcclIbSegRegister
bytes left after nSeg segments: local failure
consensus exchange, then the table gathers
```

The export covers whole pages and the MR offset is the page-aligned distance from the segment base. A local failure never returns before the first gather. Classic also rejects a multi-segment registration when `NCCL_DMABUF_ENABLE` is off.

## Consensus

Each rank allgathers a `rcclIbSegSymRecord {status, nSegments}` into `regRecords`, which `rcclRmaIbProxyConnectFinish` allocated for `nranks` records, so registration has nothing to allocate before its first collective. `rcclIbSegSymExchange` fails every rank on any failed status or unequal count, and then gathers each rank's offsets, segment VAs and keys into the symmetric table. IB-CAST gathers one more record with `magic`, `version` and `ibDevNs`.

```mermaid
%% caption: Consensus: every rank gathers the record, then succeeds or fails together
sequenceDiagram
 box rgb(245,235,251) RMA
 participant R as rank r
 participant C as regRecords gather
 end
 R->>R: count segments, register one MR per segment, record any local failure
 R->>C: allgather status and nSegments
 C-->>R: every rank's record
 alt any failed status or unequal counts
 rect rgb(255,236,235)
 R->>R: registration fails on every rank
 end
 else all records agree
 rect rgb(232,248,236)
 R->>C: allgather offsets, VAs and rkeys
 end
 end
```

## Segmented WR builder

`rcclRmaBuildSegmentedWrs` walks the transfer with `rcclIbSegCursor` over the local and remote rank's offsets and emits slices with the shared `rcclIbSegAppendWrs`, each cut at the nearer of the two boundaries and at `RCCL_IB_SGE_MAX_BYTES`. Out-of-range transfers are rejected before any WR is posted (`rcclRmaRangeOk`).

- **Put and get** use one lkey and one rkey per slice.
- **IPutSignal** adds one `IBV_WR_ATOMIC_FETCH_AND_ADD` on the segment holding `signalOff`, which must be 8-byte aligned and inside one segment (`rcclRmaSignalOffsetValid`); it rides the last round.
- **Flush** is one 1-byte read of `flushSeg`. A read cannot pass earlier posted writes, so one read flushes the window; a window that is all host memory posts nothing.
- **Signaling.** Each round signals only its last WR.

## Rounds

An operation wider than one round no longer fails. `rccl_mseg_rma_round.h` posts at most `NCCL_RMA_MAX_DATA_WRS` (64) data WRs per round, and the next round once the plugin reports the previous one done. Operations queue per peer in request order (`rcclRmaRoundQueue`), so a later operation never passes an earlier one; behind a queued operation every operation queues, including single-MR ones. The plugin supplies the hooks (`rcclRmaIbRoundBuild`, `rcclRmaIbRoundOutstanding`, `rcclRmaIbRoundPost`, `rcclRmaIbRoundRetire`, `rcclRmaIbRoundFailed`).

```mermaid
%% caption: RMA rounds: one op posted in rounds of at most 64 data WRs
flowchart TD
 op["operation"] --> q{"peer queue empty?"}
 q -->|no| enq["queue behind earlier ops"]
 q -->|yes| b["build up to 64 WRs"]
 enq --> b
 b --> p["post, signal the last WR"]
 p --> d{"bytes left?"}
 d -->|yes| w["wait for the round to complete"]
 w --> b
 d -->|no| done["request completes after its last round"]
 classDef rma fill:#f5ebfb,stroke:#af52de,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 class op,q,enq,b,p,d,w rma
 class done okpath
```

A chain with nothing queued ahead posts at once (`rcclRmaIbProxyPostChain`): data chains go through the classic WR batch (`NCCL_RMA_IB_WR_BATCHSIZE`), flushes post directly. The proxy rings its pending batch before it publishes a queued request, so a doorbell failure leaves nothing queued.

**Queues and in-flight cap.** With multi-segment chains possible (switch on, `NCCL_DMABUF_ENABLE=1`, IB or IB-CAST proxy backend), `rcclRmaProxyMaxInflightRequests` (`rccl_dev_runtime.cc`) caps each peer and context at `NCCL_RMA_MAX_INFLIGHT_REQUESTS` (32). The GIN and RMA proxies log an INFO line when this lowers `NCCL_GIN_PROXY_QUEUE_SIZE` or the per-peer request count. The RMA send QP is `NCCL_RMA_SEND_QP_WRS` = 32 × 65 = 2080, only on RMA comms (`rcclIbSendQpWrs(isRma)`). The 11-04 WR credit scheme (`rcclRmaWrCreditsReserve`) is gone.

## Prefix post and completion

If `ibv_post_send` accepts only a prefix of a round, the request is kept so `Test()` drains it. If the accepted prefix stops before the signaled tail, no completion will come, so the request is marked FAILED and `Test()` reports the error. Only a post that accepted nothing frees the slot and returns the error; classic also returns the sequence id with `rcclRmaReleaseSeqId`. The rule is `rcclRmaCompletePostedRequest`, applied by classic in `rcclRmaFinishPostedRequest`.

```mermaid
%% caption: Prefix post: which requests are kept, failed, or freed
flowchart TD
 p["ibv_post_send"] --> q{"any WR accepted?"}
 q -->|no| f["free slot, return the error"]
 q -->|yes| k["keep the request"]
 k --> tail{"signaled tail accepted?"}
 tail -->|yes| t["Test drains the completion"]
 tail -->|no| fl["mark FAILED, Test reports the error"]
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 classDef failpath fill:#ffeceb,stroke:#ff3b30,color:#1d1d1f
 class k,t okpath
 class f,fl failpath
```

## Deregistration and rollback

`rcclRmaIbProxyDeregMrSym` deregisters every segment (`rcclRmaDeregOneSegment` through `rcclIbSegDeregister`) and frees the table and handle (`rcclRmaIbProxySegMrHandleFree`). The registration failure path deregisters only completed segments and frees once.

## Window gates

`dev_runtime.cc` decides per window whether a multi-segment window reaches these backends. The inputs are communicator-wide, so every rank takes the same branch.

- `rcclDevrRmaRegistersMultiSegment` holds when the switch is on, `NCCL_DMABUF_ENABLE=1`, the RMA backend is `ncclRmaIbProxy` or `IbCastRmaIbProxy`, every rank has the same segment count, and `rcclRmaSegTableFits(nRanks, count)` holds. `symMemoryRegisterRma` registers a single-segment window always, a multi-segment window only under this gate, and otherwise skips RMA registration with an INFO line.
- `rcclDevrGinRegistersMultiSegment` also requires every active GIN backend to be the proxy type. `symMemoryRegisterGin` fails a device-only multi-segment window with `ncclInvalidUsage` when it does not hold. A window that contains host segments keeps develop's per-segment GIN path.

```mermaid
%% caption: Window gates: which multi-segment windows reach the RMA backend
flowchart TD
 w["multi-segment window"] --> sys{"contains host segments?"}
 sys -->|yes| dev["develop per-segment GIN path"]
 sys -->|no| g{"switch, DMA-BUF, IB proxy, equal counts, table fits?"}
 g -->|yes| mrs["per-segment MRs in the RMA backend"]
 g -->|no| inv["GIN fails, RMA skips"]
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 classDef stagepath fill:#fff9e0,stroke:#ffcc00,color:#1d1d1f
 classDef failpath fill:#ffeceb,stroke:#ff3b30,color:#1d1d1f
 class w,sys,g gin
 class mrs okpath
 class dev stagepath
 class inv failpath
```

## Host owner DMA-BUF

A hybrid DeepEP window maps one private GPU segment plus one host segment per local rank; every rank imports the other ranks' host segments. The importing process cannot get a DMA-BUF for an imported host segment, but the owner's POSIX fd is already a DMA-BUF that `ibv_reg_dmabuf_mr` accepts in any process, so RCCL registers imported host segments from it. The path runs only with `NCCL_SYM_REUSE_SYSMEM_HANDLES=1`.

| Step | Where | What |
|---|---|---|
| Export | `symMemoryMapLsaTeam` (`dev_runtime.cc`) | Each rank exports its own host segment before the intranode allgather; a successful export sets `hostExportable` in its message and keeps the fd. |
| Keep or fetch | `symMemoryGetHostOwnerFd` | An exporter keeps its own fd. Another rank asks the first LSA rank with `hostExportable`. With no exporter there is no fallback, because that proxy would repeat the failed export and never reply. |
| Agree | `hostOwnerFdStatus`, gathered as `ownerFdStatus` in `symMemoryObtain` | A rank-local fetch or allocation failure is gathered before the collective GIN/RMA registration, so every rank fails together. |
| Publish | `symMemoryPublishHostOwnerFds`, `rcclHostSegmentFdRegister` (`rccl_dev_runtime.cc`) | A mutex-protected process-wide table maps VA ranges to fds. |
| Register | `rcclRmaIbProxyRegMrSymDmaBuf` and the IB-CAST twin | `rcclHostSegmentFdLookup` runs before exporting. GIN stores the segment as `CU_MEM_LOCATION_TYPE_HOST_NUMA`, because the imported handle reads as device (ROCM-29810). |
| Lifetime | `symMemoryCloseHostOwnerFds` | Fds live as long as the backing memory; they are closed in `symMemoryDestroy` and on obtain failure. |

Fork 1 changes the fetch rule: an importer borrows an fd only when exactly one LSA rank can export, and removes the ROCM-29810 TODO. rocSHMEM GDA and Anvil SDMA accept `regMrSymDmaBuf` but ignore the fd ([proposal](multi_segment_gin_rocshmem_gda_and_anvil_sdma.md)). NET/IB P2P never registers a buffer with host segments. The hybrid contract and its tests are in [testing](multi_segment_testing.md#hybrid-deepep-windows).

## Symbol index

| Concern | Symbols |
|---|---|
| Dispatch | `rcclRmaIbProxyInstallMultiSeg`, `rcclRmaIbProxyUseSingle`, `rcclRmaIbProxyTestFailed` (`net_ib/gin.cc`); `rcclRmaMultiSegEnabled` (`rccl_mseg_rma.h`) |
| Handle and lookup | `rcclRmaIbProxySegMrHandle`, `rcclRmaSegOf`, `rcclRmaRankOff`, `rcclRmaRemoteRkey`, `rcclRmaRangeOk` (`net_ib/gin.cc`); `rcclRmaSegMrTag`, `rcclRmaMrSegmentCount` (`rccl_mseg_rma.h`) |
| Consensus | `rcclIbSegSymRecord`, `rcclIbSegSymExchange` (`rccl_mseg_sym.h`); `regRecords` (`net_ib/gin.h`); `rcclRmaIbProxyConnectFinish`, `rcclRmaRegAllGather` |
| Builder and rounds | `rcclRmaBuildSegmentedWrs`, `rcclRmaIbRound*`, `rcclRmaIbProxyStartRequest`, `rcclRmaIbProxyPostChain`, `rcclRmaIbProxyEnqueue`, `rcclRmaIbProxySubmit`, `rcclRmaIbProxyQueueWrs`, `rcclRmaFinishPostedRequest` (`net_ib/gin.cc`); `rcclRmaRoundOp`, `rcclRmaRoundQueue`, `rcclRmaRoundHooks` (`rccl_mseg_rma_round.h`) |
| Limits | `NCCL_RMA_MAX_DATA_WRS` (64), `NCCL_RMA_MAX_SIGNAL_WRS` (65), `NCCL_RMA_MAX_REQUESTS` (256), `NCCL_RMA_MAX_INFLIGHT_REQUESTS` (32), `NCCL_RMA_SEND_QP_WRS` (2080), `RCCL_RMA_SEG_TABLE_MAX_ENTRIES` (2^20) |
| Window gates | `rcclDevrRmaRegistersMultiSegment`, `rcclDevrGinRegistersMultiSegment`, `rcclRmaProxyMaxInflightRequests` (`rccl_dev_runtime.cc`) |
| Host owner fd | `symMemoryGetHostOwnerFd`, `symMemoryPublishHostOwnerFds`, `symMemoryCloseHostOwnerFds`, `rcclHostSegmentFdRegister`, `rcclHostSegmentFdLookup` |
| Tests | `test/RmaSegmentMathTests.cpp`, `test/transport/RmaMPI/RmaMultiSegmentMPITests.cpp`, host suites under `test/host/transport/net_ib/` and `test/host/gin/` |
