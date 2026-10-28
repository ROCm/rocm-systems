# Multi-segment DMA-BUF registration: overview

This is the entry point to the multi-segment design docs. It explains why RCCL registers one memory region per physical segment on ROCm, what the stack implements on each InfiniBand path, and how it serves DeepEP-style workloads. It also defines the terms the other docs use, lists the pull requests, and maps each doc to its readers.

## Executive summary

**The problem.** A virtual memory (VMM, `cuMem`/`hipMem`) buffer is one contiguous virtual address range backed by one or more physical allocations. To let the NIC reach it directly, RCCL exports a DMA-BUF file descriptor for the range and registers it with `ibv_reg_dmabuf_mr`. On ROCm, the DMA-BUF exported for a range that spans several physical allocations describes only the first one. Registering the whole range then fails with `EINVAL`, and an RDMA operation that crosses into a later allocation fails with `IBV_WC_REM_INV_REQ_ERR` or `IBV_WC_REM_ACCESS_ERR`, which can leave the queue pair (QP) in a fatal state and hang the collective.

**What was implemented.** Every InfiniBand path now registers one DMA-BUF memory region (MR) per physical segment, up to 16 segments, because a DMA-BUF exported for a single segment is complete. The data paths then split each transfer at segment boundaries on both the local and the remote side, so each work request uses the keys of the segment it touches. The same idea is applied four times: classic GIN/RMA, classic NET/IB point-to-point (P2P), IB-CAST P2P, and IB-CAST GIN/RMA. Single-segment buffers keep the existing code path, and the whole feature is behind the runtime switch `NCCL_MULTI_SEGMENT_REGISTER`, which is off by default on HIP.

**Why DeepEP needs it.** DeepEP-style workloads allocate elastic symmetric windows that mix GPU and host memory in one address range, for example `[GPU][CPU]` or `[GPU k][CPU 0]…[CPU N-1]` with one host segment per local rank. They use LSA for peers on the same node and GIN for peers on other nodes, through the RMA proxy, and they move bulk data with NET/IB P2P. Without per-segment registration those windows cannot be registered for GIN at all, and P2P buffers built from several allocations fall back to slow host staging. With it, a remote CPU segment can be read straight into a local GPU segment at independent offsets.

**Status.** Layers 01 to 03 are merged. Layers 05 to 08 and the IB-CAST GIN/RMA layer are open pull requests. The latest wire-branch run passed all 58 test-runner units on ROCm 10.3 and 57 of 58 on ROCm 7.0.2.2; the one failure is a known GIN kernel launch problem tracked in [Testing](multi_segment_testing.md#known-failures-and-dependencies).

## Design at a glance

A buffer goes through five stages. Allocation decides how many physical segments it has. Registration counts them, registers one MR per segment, and (for symmetric GIN windows) makes every rank agree on the layout. The data path splits each operation at segment boundaries. Deregistration frees every per-segment MR.

```mermaid
%% caption: The five stages of a multi-segment buffer
flowchart TD
 alloc["allocate VMM buffer of N physical segments"] --> sw{"switch on and N greater than 1?"}
 sw -->|no| legacy["existing single-MR path"]
 sw -->|yes| count["count segments, at most 16"]
 count --> mrs["export and register one MR per segment"]
 mrs --> kind{"which transport?"}
 kind -->|"GIN/RMA"| cons["allgather consensus record, then VAs and rkeys"]
 kind -->|"NET/IB P2P"| side["receiver publishes its layout in the side table"]
 cons --> split["split each operation at local and remote boundaries"]
 side --> split
 split --> dereg["deregister every per-segment MR"]
 classDef devmem fill:#eeeefb,stroke:#5856d6,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 classDef stagepath fill:#fff9e0,stroke:#ffcc00,color:#1d1d1f
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef netib fill:#e0f5f8,stroke:#008299,color:#1d1d1f
 class alloc devmem
 class mrs,split okpath
 class legacy stagepath
 class cons gin
 class side netib
```

The two transports differ in how the remote side learns the layout. GIN/RMA registers a symmetric window collectively, so every rank learns every peer's per-segment addresses and rkeys at registration time. NET/IB P2P registers each buffer locally, so the receiver sends its segment table with each clear-to-send (CTS) message.

```mermaid
%% caption: How each transport learns the remote segment layout
sequenceDiagram
 participant A as rank A
 participant B as rank B
 Note over A,B: GIN/RMA registration, collective
 A->>B: consensus record: status, nSegments, segOff
 B->>A: consensus record: status, nSegments, segOff
 Note over A,B: every rank checks counts and tables, then gathers VAs and rkeys
 Note over A,B: NET/IB P2P transfer, per buffer
 B->>A: side table write with per-segment addresses and rkeys
 B->>A: 64-byte CTS with segment 0 rkeys
 A->>B: one RDMA write per slice, each inside one local and one remote segment
```

The details are in [GIN/RMA](multi_segment_gin_rma.md), [classic NET/IB P2P](multi_segment_netib_wire.md), [IB-CAST P2P](multi_segment_netib_cast.md) and [IB-CAST GIN/RMA](multi_segment_netib_cast_gin.md). The rules every path must keep are in the [current contract](multi_segment_current.md).

## Memory spaces and scopes

Each rank's window is backed by up to 16 physical segments, each on the device or in host NUMA memory. Every segment gets its own DMA-BUF fd and one MR per NIC device. Who may touch the window sets the scope: the rank itself, the LSA team of same-node peers through loads and stores, every rank of the communicator through GIN/RMA, or one connection through NET/IB P2P.

```mermaid
%% caption: Memory spaces and access scopes of a multi-segment window
flowchart TD
 win["window on one rank"] --> segs["up to 16 physical segments, each device or host NUMA"]
 segs --> mr["one DMA-BUF fd per segment, one MR per segment per NIC"]
 win --> scope{"who accesses it?"}
 scope -->|"same node"| lsa["LSA team: loads and stores"]
 scope -->|"every rank"| gin["communicator: GIN/RMA"]
 scope -->|"one peer"| p2p["connection: NET/IB P2P"]
 gin -.->|"rkeys"| mr
 p2p -.->|"rkeys"| mr
 classDef devmem fill:#eeeefb,stroke:#5856d6,color:#1d1d1f
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef netib fill:#e0f5f8,stroke:#008299,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 class win,segs,lsa devmem
 class fd,mr okpath
 class gin gin
 class p2p netib
```

## Terms

These terms are used the same way in every multi-segment doc.

| Term | Meaning |
|---|---|
| Segment | One physical VMM allocation (`cuMemCreate` handle) mapped inside a contiguous virtual address range. A buffer of N segments has N-1 internal boundaries. |
| DMA-BUF | A Linux file descriptor that exports device or host memory to another driver, here the RDMA NIC. |
| MR | Memory region: a range registered with the NIC (`ibv_reg_dmabuf_mr`). A multi-segment buffer has one MR per segment per NIC device. |
| lkey, rkey | The local and remote keys of an MR. A work request must use the keys of the segment its bytes are in. |
| Slice | The part of a transfer that stays inside one local and one remote segment. Each slice becomes one work request (WR). |
| Consensus record | The fixed-size record `{status, nSegments, segOff[17]}` every rank allgathers during GIN/RMA registration, even after a local failure, so all ranks succeed or fail together. IB-CAST adds a magic number and a version. |
| `segOff` | A rank's table of segment start offsets within its window: `segOff[0] = 0` and `segOff[nSegments]` is the window size. |
| CTS | Clear-to-send: the 64-byte FIFO entry a NET/IB receiver RDMA-writes to the sender to describe one receive buffer. Its size does not change. |
| Side table | The per-CTS-slot array of `ncclIbSegLayout` entries (704 bytes each) that a NET/IB receiver writes next to the CTS FIFO to describe a multi-segment receive. |
| Capability trailer | A `{magic 0x4d534547, caps}` record in the reserved tail of the connect metadata's `devName` field. It carries `NCCL_IB_CAP_MULTISEG` without changing the metadata size. |
| Staging | The fallback in which a collective copies through RCCL's own registered buffers instead of registering the user buffer. |
| LSA | Load/store accessible: peers on the same node that map each other's window and access it with GPU loads and stores. |
| GIN | GPU-initiated networking: device-side put, get and signal operations on a symmetric window to peers on other nodes. The host-proxy type (`NCCL_GIN_TYPE=2`) runs them through an RMA backend. |
| RMA | The host-side remote memory access backend (`ncclRma_t`) that posts GIN operations as IB work requests: `ncclRmaIbProxy` (classic) or `IbCastRmaIbProxy` (IB-CAST). |
| Classic, IB-CAST | The two InfiniBand plugins: classic `net_ib` (`NCCL_NET=IB`) and RCCL's fork `net_ib_cast` (`NCCL_NET=IB-CAST`), which adds a QP scheduler, inline CTS on AINIC, and RCCL-specific recovery. |
| Credit | A reserved send-queue slot, taken before a WR chain is posted and returned when its completion arrives. |
| Final-WR signaling | Only the last WR of a chain asks for a completion, so one completion retires the whole chain. |
| Prefix post | `ibv_post_send` accepted the first part of a chain and rejected the rest. |
| Switch on, switch off | `NCCL_MULTI_SEGMENT_REGISTER=1` or `=0`. Off restores the develop code path everywhere. See [the switch](#the-switch). |

## The switch

`NCCL_MULTI_SEGMENT_REGISTER` (`ncclParamMultiSegmentRegister`, `transport/generic.cc`) gates every multi-segment path. It defaults to 0 on HIP and to 1 elsewhere, and it must have the same value on every rank. Classic NET/IB reads it through `ncclIbAdvertiseMultiSeg`, IB-CAST P2P through `IbCastAdvertiseMultiSeg`, and both RMA backends through `rcclRmaMultiSegEnabled`.

With the switch off, no capability is advertised, develop's QP depths and GIN/RMA bodies run, no consensus buffer is allocated, multi-segment NET buffers stage, and a device-only multi-segment GIN window fails registration. The full list is in [the current contract](multi_segment_current.md#switch-off-behavior).

## Stack and pull requests

The stack is linear on `develop`, one pull request per layer on [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems). Branch names share the prefix `users/atulkulk/mseg/`. There is no layer 04.

| Layer | Pull request | Branch | Scope | State |
|---|---|---|---|---|
| 01 | [PR 10887](https://github.com/ROCm/rocm-systems/pull/10887) | `01-ce-recv-window-offset` | CE AllReduce receive-window offset and bounds fixes | merged |
| 02 | [PR 10891](https://github.com/ROCm/rocm-systems/pull/10891) | `02-amd-host-vmm-compat` | host VMM location type, mixed host window classification, probe-kind allgather | merged |
| 03 | [PR 10885](https://github.com/ROCm/rocm-systems/pull/10885) | `03-registration-range` | NET segment-count cache, last-page walk fix, graph NET segments | merged |
| 05 | [PR 10884](https://github.com/ROCm/rocm-systems/pull/10884) | `05-gin-rma-multisegment` | classic GIN/RMA per-segment MRs, consensus, segmented WRs, credits, host owner DMA-BUF | open |
| 06 | [PR 10886](https://github.com/ROCm/rocm-systems/pull/10886) | `06-classic-netib-multisegment` | classic NET/IB P2P: side table, capability trailer, segmented send and flush | open |
| 07 | [PR 10890](https://github.com/ROCm/rocm-systems/pull/10890) | `07-hybrid-stress-ci` | hybrid DeepEP tests, stress tests, runner configs | open |
| 08 | [PR 10889](https://github.com/ROCm/rocm-systems/pull/10889) | `08-cast-netib-multisegment` | IB-CAST P2P on the same wire format, plus IB-CAST hardening | open |
| wire | [PR 10888](https://github.com/ROCm/rocm-systems/pull/10888) | `multi-segment-dmabuf-cast-gin` | IB-CAST GIN/RMA per-segment MRs and credits | open |

A later follow-up removes the 16-segment cap; its design is in [registering past 16 segments](multi_segment_varseg.md). The multi-segment code sits in the upstream-derived files it hooks, in `RCCL:`-marked blocks reached through one-line calls; [sync and removal](multi_segment_sync_and_removal.md) measures that footprint.

## Reading guide

| Doc | Covers | Read it if you |
|---|---|---|
| [Overview](multi_segment_overview.md) (this doc) | problem, design, terms, stack, color legend | are new to the feature |
| [Current contract](multi_segment_current.md) | invariants, switch-off behavior, limits, QP depths, version gates, code map, open items | review or change any multi-segment code |
| [GIN/RMA](multi_segment_gin_rma.md) | shared per-segment registration, consensus, segmented WR builder, credits, flush, host owner DMA-BUF, window gates | work on GIN, RMA or DeepEP windows |
| [Classic NET/IB P2P](multi_segment_netib_wire.md) | side table, capability trailer, segmented send, flush, rejected alternatives | work on `net_ib` P2P |
| [IB-CAST P2P](multi_segment_netib_cast.md) | IB-CAST differences: declines, 32-byte CTS, scheduler, failure handling | work on `net_ib_cast` P2P |
| [IB-CAST GIN/RMA](multi_segment_netib_cast_gin.md) | IB-CAST RMA handle, consensus with magic and version, completion-driven credits, error propagation | work on `net_ib_cast` GIN |
| [Code map](multi_segment_code_map.md) | every symbol the stack adds or changes, per file and layer, with line numbers and links to the design sections | look for where a behavior lives or which pull request owns it |
| [Testing](multi_segment_testing.md) | test strategy, coverage, hybrid DeepEP tests, runner jobs, how to run, known failures | run, add or debug tests |
| [rocSHMEM GDA and Anvil SDMA](multi_segment_gin_rocshmem_gda_and_anvil_sdma.md) | proposed support in the device-initiated GIN backends | plan GIN types 6 or 7 |
| [Sync and removal](multi_segment_sync_and_removal.md) | upstream-sync footprint and how to remove the workaround once the driver exports whole ranges | plan an upstream sync or a driver-fix rollout |
| [Past 16 segments](multi_segment_varseg.md) | design to drop the 16-segment cap | work on the cap follow-up |
| [Side table on first registration](multi_segment_sidetable_first_reg.md) | design to defer the 1.4 MiB side-table cost | work on connect memory cost |
| [Flush read collapse](multi_segment_flush_read_collapse.md) | design to drop duplicate flush reads | work on flush cost |

## Color legend

Every diagram and callout in the multi-segment docs uses this scheme. Component colors mark the transport or memory a node belongs to. Outcome colors mark how a decision ends, and they take precedence over component colors on outcome nodes. Sequence diagrams group participants in a box of the component color and shade outcome messages with a region of the outcome color. Uncolored nodes are steps shared by several components.

```mermaid
%% caption: Color legend for every diagram in this book
flowchart LR
 subgraph comp["Components"]
 direction TB
 lgin["GIN"]
 lrma["RMA"]
 lnet["classic NET/IB P2P"]
 lcast["IB-CAST"]
 lhost["host memory"]
 ldev["device memory"]
 end
 subgraph outc["Outcomes"]
 direction TB
 lok["success"]
 lstage["staging or fallback"]
 lfail["failure or error"]
 end
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef rma fill:#f5ebfb,stroke:#af52de,color:#1d1d1f
 classDef netib fill:#e0f5f8,stroke:#008299,color:#1d1d1f
 classDef ibcast fill:#fff1e0,stroke:#ff9500,color:#1d1d1f
 classDef hostmem fill:#f2f2f7,stroke:#8e8e93,color:#1d1d1f
 classDef devmem fill:#eeeefb,stroke:#5856d6,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 classDef stagepath fill:#fff9e0,stroke:#ffcc00,color:#1d1d1f
 classDef failpath fill:#ffeceb,stroke:#ff3b30,color:#1d1d1f
 classDef neutral fill:#f9f9fb,stroke:#d1d1d6,color:#1d1d1f
 class lgin gin
 class lrma rma
 class lnet netib
 class lcast ibcast
 class lhost hostmem
 class ldev devmem
 class lok okpath
 class lstage stagepath
 class lfail failpath
 class comp,outc neutral
```

| Class | Meaning | Stroke | Fill |
|---|---|---|---|
| `gin` | GIN | `#007aff` | `#e5f1ff` |
| `rma` | RMA | `#af52de` | `#f5ebfb` |
| `netib` | classic NET/IB P2P | `#008299` | `#e0f5f8` |
| `ibcast` | IB-CAST | `#ff9500` | `#fff1e0` |
| `hostmem` | host memory (sysmem, `HOST_NUMA`) | `#8e8e93` | `#f2f2f7` |
| `devmem` | device memory segment | `#5856d6` | `#eeeefb` |
| `okpath` | per-segment registration or success | `#34c759` | `#e8f8ec` |
| `stagepath` | staging, fallback or skip | `#ffcc00` | `#fff9e0` |
| `failpath` | failure: `ncclInvalidUsage`, WARN, error return | `#ff3b30` | `#ffeceb` |
| `neutral` | grouping subgraph | `#d1d1d6` | `#f9f9fb` |

Text callouts use GitHub alerts, which render as colored boxes:

| Callout | Color | Meaning |
|---|---|---|
| `> [!NOTE]` | blue | background and context |
| `> [!TIP]` | green | how to enable it, what to set |
| `> [!IMPORTANT]` | purple | a contract or invariant that must hold |
| `> [!WARNING]` | amber | fallback, staging or a limitation |
| `> [!CAUTION]` | red | failure modes, data corruption risk |
