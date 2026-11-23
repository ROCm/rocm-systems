# Multi-segment registration: code map

This doc lists the code the IB-MSEG fork stack adds or changes, one table per area, with the fork that owns each symbol. Use it to find where a behavior lives and which fork introduces it. The design docs stay concise and link here instead of naming every function. The forks and their scope are described in [the fork plugin](multi_segment_fork_plugin.md#forks).

## How to read

Line numbers are the line of each symbol's definition at fork 4 `9c67177e1f99` (branch `users/atulkulk/netib-mseg/04-rma-gin-multiseg`) on develop `c43b4a9e7307`, as of 2026-11-23. They drift as the code changes; search by symbol when they do. Paths are relative to `projects/rccl/src/` unless they start with `test/` or `tools/`.

The Fork column is the fork (1 to 4) whose commit introduces the symbol or, for a develop symbol, adds the multi-segment hook to it; "1, 4" means both forks change it. Kind is "new" when the stack adds the symbol and "changed" when it edits a develop symbol, usually through a short `// RCCL:` hook. Inside `transport/net_ib_multiseg/`, "changed" means changed against fork 2's copy of `net_ib`; symbols there carry the `IbMseg` prefix from `rename_map.txt`. `transport/net_ib/` and `transport/net_ib_cast/` are not touched by the stack. Develop already has the NET segment count from the merged registration-range work (`rcclNetRegSegmentCount`, `rccl_wrap.cc:3087`) and the switch `NCCL_MULTI_SEGMENT_REGISTER` (`transport/generic.cc:14`); they are not listed below. Abbreviations are defined in the [overview terms](multi_segment_overview.md#terms).

## How the forks connect

The registration entry points in the runtime decide whether a buffer takes the multi-segment path. Core code (fork 1) gates every path on a comm-level flag that is set only when the selected plugin registers one MR per segment; fork 2 makes IB-MSEG that plugin; forks 3 and 4 add the per-segment registration and data paths inside it.

```mermaid
%% caption: How registration reaches the IB-MSEG plugin, with the fork that owns each step
flowchart TD
 win["window or buffer registration"] --> sel["IB-MSEG selected (2)"]
 sel --> netreg["NET proxy: fd -1 to the plugin (1)"]
 sel --> devr["symmetric window gates (1)"]
 netreg --> mreg["one MR per segment (3)"]
 mreg --> p2p["P2P: one-write CTS, sliced send, one-read flush (3)"]
 devr --> ginreg["GIN register, owner fd (1)"]
 devr --> rmareg["RMA proxy register, in-flight cap (1)"]
 ginreg --> rma["IB-MSEG RMA: consensus, split WRs, rounds (4)"]
 rmareg --> rma
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef rma fill:#f5ebfb,stroke:#af52de,color:#1d1d1f
 classDef netib fill:#e0f5f8,stroke:#008299,color:#1d1d1f
 classDef neutral fill:#f9f9fb,stroke:#d1d1d6,color:#1d1d1f
 class win,sel,devr neutral
 class ginreg gin
 class rmareg,rma rma
 class netreg,mreg,p2p netib
```

| Fork | Branch (`users/atulkulk/netib-mseg/...`) | Tables |
|---|---|---|
| 1 | `01-core-backend-multiseg-gate` | [Core: device runtime](#core-device-runtime-and-symmetric-windows), [GIN and RMA proxies](#core-gin-and-rma-proxies), [shared limits](#core-shared-limits), most of [NET proxy](#core-comm-flags-and-the-net-proxy) |
| 2 | `02-net-ib-multiseg-copy` | [Core: plugin selection](#core-plugin-selection), [the copy](#ib-mseg-the-copy) |
| 3 | `03-registration-p2p-multiseg` | [Fork-local files](#ib-mseg-fork-local-files) (P2P and registration), [hooks](#ib-mseg-hooks-in-the-copied-files) |
| 4 | `04-rma-gin-multiseg` | [Fork-local files](#ib-mseg-fork-local-files) (RMA), [hooks](#ib-mseg-hooks-in-the-copied-files) in `gin.cc` and `connect.cc`, `include/alloc.h` |

Forks 5 and 6 (IB-CAST base and IB-CAST multi-segment) have no branch yet and no code in this map.

## Core: device runtime and symmetric windows

Design: [GIN and RMA](multi_segment_gin_rma.md).

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `dev_runtime.cc` | `symLsaMessage` | 483 | 1 | changed | adds `hostExportable`: 1 when this rank holds the creating POSIX fd for the segment |
| `dev_runtime.cc` | `symMemoryEnsureHostOwnerFds` | 583 | 1 | new | lazily makes the per-segment owner-fd table, every entry -1 |
| `dev_runtime.cc` | `symMemoryGetHostOwnerFd` | 591 | 1 | new | fetches the one exporter's fd, since an imported host handle cannot export a DMA-BUF |
| `dev_runtime.cc` | `symMemoryPublishHostOwnerFds` | 608 | 1 | new | publishes the owner fds by VA so per-segment registration can find them |
| `dev_runtime.cc` | `symMemoryCloseHostOwnerFds` | 618 | 1 | new | unpublishes each owner fd before closing it |
| `dev_runtime.cc` | `symMemoryMapLsa` | 635 | 1 | changed | exports the local host fd; borrows a peer's fd only when exactly one rank can export; the ROCM-29810 TODO is removed |
| `dev_runtime.cc` | `symMemoryRegisterGin` | 983 | 1 | changed | fails closed on a multi-segment device window unless `rcclDevrGinRegistersMultiSegment`; marks owner-fd segments as host |
| `dev_runtime.cc` | `symMemoryRegisterRma` | 1066 | 1 | changed | registers a multi-segment window when `rcclDevrRmaRegistersMultiSegment`; otherwise every rank skips together |
| `dev_runtime.cc` | `symMemoryUnregister` | 1082 | 1 | changed | deregisters the RMA window whatever its segment count |
| `dev_runtime.cc` | `symMemoryObtain` | 1166 | 1 | changed | gathers `minGlobalNumSegments`; all ranks agree on the owner-fd status; publishes owner fds |
| `dev_runtime.cc` | `symMemoryDestroy` | 1424 | 1 | changed | closes the owner fds published for the window |
| `dev_runtime_internal.h` | `ncclDevrMemory` | 63 | 1 | changed | adds `minGlobalNumSegments` (84), `hostOwnerFds` (94), `hostOwnerFdStatus` (95) |
| `rccl_dev_runtime.cc` | `rcclMultiSegOn` | 128 | 1 | new | comm flag: the plugin registers per segment, the switch is on and DMA-BUF is on |
| `rccl_dev_runtime.cc` | `rcclDevrRmaRegistersMultiSegment` | 132 | 1 | new | RMA flag set, same segment count on every rank, segment table fits |
| `rccl_dev_runtime.cc` | `rcclRmaProxyMaxInflightRequests` | 139 | 1 | new | per-peer in-flight cap for the RMA and GIN proxies |
| `rccl_dev_runtime.cc` | `rcclDevrGinRegistersMultiSegment` | 143 | 1 | new | same as RMA, and every active GIN backend is the proxy |
| `rccl_dev_runtime.cc` | `rcclHostSegmentFd` | 154 | 1 | new | VA-keyed owner-fd table entry |
| `rccl_dev_runtime.cc` | `rcclHostSegmentFdRegister`, `rcclHostSegmentFdDeregister`, `rcclHostSegmentFdLookup` | 163, 168, 179 | 1 | new | publish, unpublish and find an owner fd by VA range |
| `include/rccl_common.h` | declarations of the `rccl_dev_runtime.cc` functions above | 533 to 543 | 1 | new | |
| `include/alloc.h` | `rcclCuMemIsSysmemSegment` | 832 (stub 929) | 4 | new | whether the cuMem allocation at a pointer is host memory; used by `seg_export_mseg.h` |
| `include/alloc.h` | `ncclCuMemGetAddressRange` | 854 | 4 | changed | uses `rcclCuMemIsSysmemSegment` for its host-segment check |

## Core: GIN and RMA proxies

Design: [GIN and RMA](multi_segment_gin_rma.md).

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `gin/gin_host.cc` | `ginDevCommSetupWithBackend` | 318 | 4 | changed | passes `comm->rcclMultiSegRma` to the GIN proxy before it creates a context |
| `gin/gin_host.cc` | `ncclGinRegister` | 688 | 1 | changed | registers a host segment with its owner fd through `regMrSymDmaBuf` |
| `gin/gin_host_proxy.cc` | `rmaMultiSeg` | 94 | 1 | new | the comm's RMA flag, set by `rcclGinProxySetMultiSeg` |
| `gin/gin_host_proxy.cc` | `ncclGinProxyRegMrSymDmaBuf` | 524 | 1 | new | registers with a caller-supplied DMA-BUF fd; falls back to `regMrSym` |
| `gin/gin_host_proxy.cc` | `ncclGinProxyCreateContext` | 549 | 1 | changed | caps `NCCL_GIN_PROXY_QUEUE_SIZE` for multi-segment chains |
| `gin/gin_host_proxy.cc` | `ncclGinProxy` | 809 | 1 | changed | fills the `regMrSymDmaBuf` slot |
| `gin/gin_host_proxy.cc` | `rcclGinProxySetMultiSeg` | 832 | 1 | new | setter, declared in `include/gin/gin_host_proxy.h:23` |
| `rma/rma_proxy.cc` | `ncclRmaProxyCtxAlloc` | 169 | 1 | changed | caps requests in flight per peer for multi-segment chains |
| `rma/rma_proxy_launch.cc` | `ncclRmaProxyPutBuildOp` | 61 | 1 | changed | fails a put on a window whose RMA registration was skipped |
| `ce_coll.cc` | `ncclHierCeAvailable` | 1435 | 1 | changed | skips hierarchical CE when a window has no RMA registration |

## Core: comm flags and the NET proxy

Design: [NET/IB P2P](multi_segment_netib_wire.md).

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `include/comm.h` | `ncclComm::rcclMultiSegNet`, `ncclComm::rcclMultiSegRma` | 839, 840 | 1 | new | the comm's net and RMA plugin register one MR per segment |
| `include/proxy.h` | `ncclProxyState::rcclMultiSegNet` | 392 | 1 | new | proxy copy of the net flag |
| `proxy.cc` | `ncclProxyCreate` | 2252 | 1 | changed | copies the net flag into the proxy state |
| `init.cc` | `commAlloc` | 835 | 3, 4 | changed | sets `rcclMultiSegNet` for IB-MSEG (fork 3) and `rcclMultiSegRma` for its RMA backend, inherited by child comms (fork 4) |
| `init.cc` | `rcclSetIbGdrFlushFlag`, `fillInfo` | 1395, 1410 | 2 | changed | GDR flush flag through `rcclIbGdrFlushOpsFor`, so it also works on IB-MSEG |
| `transport/net.cc` | `rcclNetRegMultiSeg` | 466 | 1 | new | passes fd -1 only to a plugin that walks a multi-segment buffer itself; false means stage |
| `transport/net.cc` | `sendProxyRegBuffer`, `recvProxyRegBuffer` | 2654, 2743 | 1 | changed | multi-segment buffers go to `rcclNetRegMultiSeg` or stage; skip whole-range HSA DMA-BUF |
| `transport/net.cc` | `recvProxySetup`, `recvProxyConnect`, `recvProxyProgress` | 1178, 1541, 2138 | 2 | changed | GDR flush flag calls through `rcclIbGdrFlushOpsFor` |
| `transport/net.cc` | `sendProxyConnect` | 1260 | 2 | changed | applies `IbMsegNetP2pPolicy` on IB-MSEG |

## Core: plugin selection

Design: [the fork plugin, selection](multi_segment_fork_plugin.md#selection).

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `include/net.h` | `ncclNetIbMseg`, `IbMsegNetP2pPolicy` | 74, 75 | 2 | new | the IB-MSEG net vtable and its P2P policy |
| `include/net.h` | `IbMsegGdrFlushFlagLocalSupport` and the other `IbMseg*GdrFlushFlag*` entry points | 77 to 82 | 2 | new | the copy's GDR flush flag functions |
| `include/net.h` | `rcclIbGdrFlushOps`, `rcclIbGdrFlushOpsFor` | 133, 141 | 2 | new | picks `net_ib`'s or IB-MSEG's flush-flag functions by net vtable |
| `include/rma.h` | `IbMsegRmaIbProxy` | 20 | 2 | new | IB-MSEG's RMA backend |
| `plugin/net.cc` | `initPluginLibsOnceFunc` | 292 | 2 | changed | IB-MSEG takes the IB slot with `NCCL_MULTI_SEGMENT_REGISTER=1` or `NCCL_NET=IB-MSEG` |
| `plugin/net.cc` | `ncclNetInit` | 379 | 2 | changed | with the switch on, IB-MSEG serves a request for `IB` |
| `plugin/rma.cc` | `initPluginLibsOnceFunc` | 173 | 2 | changed | selects `IbMsegRmaIbProxy` under the same condition |
| `ras/diagnostics_checks_common.cc` | `rasDiagnosticsNetPluginSupported` | 26 | 2 | changed | lists `IB-MSEG` (entry at line 36) |
| `CMakeLists.txt` | `add_subdirectory(transport/net_ib_multiseg)`, `NET_IB_MULTISEG_SOURCES` | 15, 748 | 2 | new | builds the copy |
| `CMakeLists.txt` | `transport/rccl_mseg.h`, `transport/rccl_mseg_rma.h` | 620, 621 | 1 | new | shared limit headers |

## Core: shared limits

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `transport/rccl_mseg.h` | `RCCL_IB_REG_SEG_DEFAULT` | 17 | 1 | new | most segments one registration takes |
| `transport/rccl_mseg.h` | `rcclIbSegCountSupported` | 20 | 1 | new | the one segment-count check |
| `transport/rccl_mseg_rma.h` | `RCCL_RMA_SEG_TABLE_MAX_ENTRIES`, `rcclRmaSegTableFits` | 19, 20 | 1 | new | every rank holds every rank's segment table, so ranks times segments is bounded |
| `transport/rccl_mseg_rma.h` | `RCCL_RMA_MAX_INFLIGHT_REQUESTS`, `rcclRmaInflightRequestsCap` | 26, 29 | 1 | new | the in-flight cap (32 requests per peer) with multi-segment on |

## IB-MSEG: the copy

Design: [the copy and the rename map](multi_segment_fork_plugin.md#the-copy-and-the-rename-map).

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `transport/net_ib_multiseg/` (22 files) and `tools/scripts/net_ib_multiseg/rename_map.txt` | the renamed copy of `net_ib` | — | 2 | new | byte-identical to `net_ib` apart from the names in `rename_map.txt`; the copied symbols are not listed here |

The copy was produced by a rename tool kept outside the stack. `rename_map.txt` stays in fork 2 because the compile-twice host tests read it.

## IB-MSEG: fork-local files

These files exist only in the fork. Design: [NET/IB P2P](multi_segment_netib_wire.md) for fork 3 and [GIN and RMA](multi_segment_gin_rma.md) for fork 4.

| File (lines) | Fork | Holds |
|---|---|---|
| `seg_mseg.h` (92) | 3 | segment table, slice and cursor |
| `seg_reg_mseg.h` (84) | 3 | register-all-or-unwind helper and the walk block |
| `seg_wr_mseg.h` (73) | 3 | appends per-segment WRs and SGEs |
| `seg_p2p_mseg.h` (276), `seg_p2p.cc` (342) | 3 | CTS layout prefix, layout blob, per-segment MR set, send segment tables, layout cache |
| `p2p_multiseg_mseg.h` (73), `p2p_multiseg.cc` (368) | 3 | the P2P send and receive builders |
| `ib_multiseg_mseg.h` (49) | 3, 4 | connect-time decision (fork 3) and the RMA entry points (fork 4) |
| `seg_export_mseg.h` (93) | 4 | per-segment DMA-BUF export walk |
| `seg_rma_mseg.h` (101) | 4 | RMA WR limits, send-queue depth, signal and posted-prefix rules |
| `seg_rma_round_mseg.h` (241) | 4 | RMA rounds and per-peer queues |
| `seg_sym_mseg.h` (108) | 4 | per-rank segment consensus and key exchange |

### `seg_mseg.h`, `seg_reg_mseg.h`, `seg_wr_mseg.h` (fork 3)

| File | Symbol | Line | Role |
|---|---|---|---|
| `seg_mseg.h` | `RCCL_IB_SGE_MAX_BYTES` | 19 | largest SGE length |
| `seg_mseg.h` | `rcclIbSegTable`, `rcclIbSegSlice`, `rcclIbSegCursor` | 23, 29, 38 | segment table, one slice and the walk cursor |
| `seg_mseg.h` | `rcclIbSegIndexOf`, `rcclIbSegCursorInit`, `rcclIbSegCursorNext` | 47, 55, 67 | find a segment and step the cursor over local and remote tables |
| `seg_reg_mseg.h` | `rcclIbSegRegOps`, `rcclIbSegRegister`, `rcclIbSegDeregister` | 18, 39, 24 | register every segment or unwind the ones done |
| `seg_reg_mseg.h` | `rcclIbSegWalkBytes`, `rcclIbSegWalk`, `rcclIbSegWalkBind` | 53, 59, 67 | one allocation for the per-segment address, length and offset arrays |
| `seg_reg_mseg.h` | `IbMsegRegMrDmaBufMultiSeg` (declaration) | 81 | defined in `reg.cc` |
| `seg_wr_mseg.h` | `rcclIbSegWrIn`, `rcclIbSegAppendWrs` | 26, 39 | split one transfer into per-segment WRs |

### `seg_p2p_mseg.h` and `seg_p2p.cc` (fork 3)

| File | Symbol | Line | Role |
|---|---|---|---|
| `seg_p2p_mseg.h` | `RCCL_IB_MSEG_MAX_DEVS`, `RCCL_IB_MSEG_MAX_RECVS`, `RCCL_IB_CTS_INLINE_SEGS` | 27, 28, 33 | limits; up to 4 segments fit inline in the CTS slot |
| `seg_p2p_mseg.h` | `NCCL_IB_CTS_NREQS_LAYOUT`, `IbMsegCtsNreqs`, `IbMsegCtsHasLayout` | 37, 39, 41 | the layout bit in the CTS `nreqs` field |
| `seg_p2p_mseg.h` | `IbMsegSegmentIndexForRange`, `IbMsegCtsRemoteLayoutValid` | 45, 62 | find the segment for a range; validate a peer-written layout |
| `seg_p2p_mseg.h` | `rcclIbLayoutBlobBytes`, `rcclIbLayoutBlobOff`, `rcclIbLayoutBlobRkeys`, `rcclIbLayoutBlobValid` | 75, 78, 79, 84 | the out-of-line layout blob for more than 4 segments |
| `seg_p2p_mseg.h` | `IbMsegSegmentForZeroLengthOffset`, `IbMsegBytesRemainingInSegment` | 100, 108 | edge cases at segment ends |
| `seg_p2p_mseg.h` | `rcclIbSegLayoutRef`, `IbMsegSegLayout` | 113, 127 | the 192-byte layout slot the receiver writes with the CTS |
| `seg_p2p_mseg.h` | `rcclIbLayoutRkey`, `rcclIbLayoutSetRkey`, `rcclIbLayoutStamp`, `rcclIbLayoutArrived` | 159, 162, 165, 166 | read, write and stamp a layout slot; detect a torn write |
| `seg_p2p_mseg.h` | `rcclIbSegMr`, `rcclIbSegMrAt` | 173, 187 | per-segment, per-device MR set |
| `seg_p2p_mseg.h` | `rcclIbSendSegTable` | 210 | the sender's view of the remote segments |
| `seg_p2p_mseg.h` | `rcclIbLayoutState`, `rcclIbLayoutCacheEntry`, `rcclIbLayoutCache`, `rcclIbLayoutReadQp` | 235, 236, 248, 254 | cache of receiver layouts the sender read by RDMA |
| `seg_p2p.cc` | `rcclIbSegMrInit`, `rcclIbSegMrFreeArrays`, `rcclIbSegMrPublishLayout`, `rcclIbSegMrFreeLayout` | 17, 32, 43, 61 | build and free the MR set and its published layout |
| `seg_p2p.cc` | `rcclIbSegMrForRange`, `rcclIbSegLayoutFill` | 74, 79 | MR for a range; fill a layout slot for a receive |
| `seg_p2p.cc` | `rcclIbSegTableSingle`, `rcclIbSegTableLocal`, `rcclIbSegTableInline` | 130, 143, 161 | send segment table from one MR, the local MR set or an inline layout |
| `seg_p2p.cc` | `rcclIbLayoutCacheFind`, `rcclIbLayoutRead` | 194, 203 | look up or read a referenced layout |
| `seg_p2p.cc` | `rcclIbLayoutsReady`, `rcclIbLayoutReadDone`, `rcclIbLayoutCacheFree`, `rcclIbSegTableReferenced` | 257, 294, 309, 323 | wait for, complete and free layout reads; table from a referenced layout |

### `p2p_multiseg_mseg.h` and `p2p_multiseg.cc` (fork 3)

| File | Symbol | Line | Role |
|---|---|---|---|
| `p2p_multiseg_mseg.h` | `rcclIbSendCtsRow`, `rcclIbRecvCtsRow`, `rcclIbCtsLayout` | 39, 47, 52 | CTS row and layout slot addressing |
| `p2p_multiseg.cc` | `IbMsegMrForRange` | 18 | the MR covering a range on a device |
| `p2p_multiseg.cc` | `rcclIbCtsRemoteLayout`, `IbMsegCtsRemoteMultiSeg`, `rcclIbBuildRemoteSegmentLayout` | 25, 35, 41 | read the receiver's layout from the CTS |
| `p2p_multiseg.cc` | `rcclIbSegSendNextWr`, `rcclIbAppendSegmentedRequestWrs` | 52, 63 | build per-segment write WRs for one request |
| `p2p_multiseg.cc` | `rcclIbSegSendSizesWr`, `rcclIbAppendSegmentedTailWr` | 140, 152 | sizes write and the signaled tail |
| `p2p_multiseg.cc` | `rcclIbFillSendWrs`, `rcclIbSendSliceDone`, `rcclIbMultiSendNeedsSegmented` | 201, 235, 250 | sliced send: fill, completion and the decision to segment |
| `p2p_multiseg.cc` | `rcclIbPrepareCtsWrite` | 259 | the one-write CTS with the layout prefix |
| `p2p_multiseg.cc` | `rcclIbIrecvCheckHandle`, `rcclIbIrecvFillLayout` | 283, 297 | receiver: check the MR handle, fill the layout slot |
| `p2p_multiseg.cc` | `rcclIbLayoutReadStamp`, `rcclIbLayoutReadArg`, `rcclIbLayoutReadQp` | 309, 313, 319 | QP and WR for a layout read |
| `p2p_multiseg.cc` | `rcclIbSendLayoutsReady`, `rcclIbSendLayoutReadDone`, `rcclIbSendLayoutCacheFree` | 335, 344, 348 | send-side wrappers over the layout cache |
| `p2p_multiseg.cc` | `rcclIbCtsFlagLayout`, `rcclIbCtsLayoutsArrived` | 356, 363 | set the layout bit; wait for every layout of a row |

### `ib_multiseg_mseg.h` (forks 3 and 4)

| Symbol | Line | Fork | Role |
|---|---|---|---|
| `rcclRmaReleaseSeqId` | 17 | 4 | retire an RMA sequence id |
| `rcclRmaIbProxyConnectFinish`, `rcclRmaIbProxyRegMrSym`, `rcclRmaIbProxyRegMrSymDmaBuf`, `rcclRmaIbProxyDeregMrSym` | 22 to 27 | 4 | declarations of the multi-segment RMA registration entry points in `gin.cc` |
| `rcclRmaIbProxyIPut`, `rcclRmaIbProxyIGet`, `rcclRmaIbProxyIPutSignal`, `rcclRmaIbProxyIFlush`, `rcclRmaIbProxyTestFailed`, `rcclRmaIbProxyMrSegmentCount` | 28 to 40 | 4 | declarations of the multi-segment RMA data entry points |
| `IbMsegUseMultiSeg` | 45 | 3 | the connect-time decision: switch on, not GIN, not resilient |

### `seg_export_mseg.h`, `seg_sym_mseg.h` (fork 4)

| File | Symbol | Line | Role |
|---|---|---|---|
| `seg_export_mseg.h` | `rcclIbSegWalkState`, `rcclIbSegExport` | 25, 34 | walk state over a VA range and one exported segment |
| `seg_export_mseg.h` | `rcclIbSegExportNext`, `rcclIbSegExportClose`, `rcclIbSegWalkAdvance` | 50, 42, 85 | export the next segment's DMA-BUF (or use a published owner fd), close it, step |
| `seg_export_mseg.h` | `rcclIbSegExportIsHost` | 77 | host or device segment, through `rcclCuMemIsSysmemSegment` |
| `seg_sym_mseg.h` | `rcclIbSegSymTable`, `rcclIbSegSymRecord`, `rcclIbSegSymAllGatherFn` | 21, 28, 33 | per-rank segment table, one rank's record, the all-gather hook |
| `seg_sym_mseg.h` | `rcclIbSegSymVa`, `rcclIbSegSymKey`, `rcclIbSegSymTableFree` | 35, 39, 44 | look up a peer's segment VA and key |
| `seg_sym_mseg.h` | `rcclIbSegSymAgree`, `rcclIbSegSymExchange` | 54, 72 | consensus on status and counts, then exchange VAs and keys |

### `seg_rma_mseg.h`, `seg_rma_round_mseg.h` (fork 4)

| File | Symbol | Line | Role |
|---|---|---|---|
| `seg_rma_mseg.h` | `RCCL_RMA_MAX_DATA_WRS`, `RCCL_RMA_MAX_SIGNAL_WRS`, `RCCL_RMA_MAX_REQUESTS` | 23, 24, 26 | WR limits per round (64 data WRs, one more with the signal) and live request ids per comm |
| `seg_rma_mseg.h` | `RCCL_RMA_SEND_QP_WRS` | 28 | RMA send-queue depth, 32 times 65 = 2080 |
| `seg_rma_mseg.h` | `rcclRmaSegIndexOf`, `rcclRmaPeerSegOff` | 33, 52 | segment lookup in a rank's offset table |
| `seg_rma_mseg.h` | `rcclRmaWrIsSignaled`, `rcclRmaPrefixPostLostSignaledTail`, `rcclRmaSignalOffsetValid` | 37, 42, 46 | signaling rules for a split request |
| `seg_rma_mseg.h` | `rcclRmaPostedWrCount`, `rcclRmaCompletePostedRequest` | 59, 72 | keep-or-free after a partial post |
| `seg_rma_mseg.h` | `rcclRmaMultiSegEnabled`, `rcclRmaSegMrTag`, `rcclRmaMrSegmentCount` | 85, 88, 91 | process-wide switch; tag and count of a multi-segment MR handle |
| `seg_rma_round_mseg.h` | `rcclRmaRoundOp`, `rcclRmaRoundQueue`, `rcclRmaRoundQueues`, `rcclRmaRoundHooks` | 20, 39, 46, 51 | an op wider than one round, the per-peer queue and the backend hooks |
| `seg_rma_round_mseg.h` | `rcclRmaStampWrs`, `rcclRmaRoundClampWindow`, `rcclRmaRoundDataOp`, `rcclRmaRoundSetSignal` | 68, 76, 83, 101 | build an op and its WR window |
| `seg_rma_round_mseg.h` | `rcclRmaRoundQueueOf`, `rcclRmaRoundQueueBusy`, `rcclRmaRoundPush`, `rcclRmaRoundPop`, `rcclRmaRoundQueuesFree` | 117, 121, 124, 133, 140 | queue operations |
| `seg_rma_round_mseg.h` | `rcclRmaRoundFillNode`, `rcclRmaRoundAppendSignal`, `rcclRmaRoundBuild`, `rcclRmaRoundFirst` | 150, 158, 169, 188 | build the next round |
| `seg_rma_round_mseg.h` | `rcclRmaRoundPostNext`, `rcclRmaRoundPump`, `rcclRmaRoundTailDone` | 200, 221, 235 | post rounds as completions free the send queue |

## IB-MSEG: hooks in the copied files

These are edits to fork 2's copy of `net_ib` inside `transport/net_ib_multiseg/`.

### `connect.cc`, `connect_mseg.h`

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `connect.cc` | `IbMsegResolveUseMultiSeg` | 20 | 3 | new | this side's `useMultiSeg` |
| `connect.cc` | `IbMsegCheckPeerUseMultiSeg` | 29 | 3 | new | fails the connect with `ncclInvalidUsage` when the two ends differ |
| `connect.cc` | `IbMsegSendCommAllocCts`, `IbMsegSendCommFreeCts` | 38, 53 | 3 | new | the combined CTS slot block |
| `connect.cc` | `rcclIbSendQpWrs` | 66 | 4 | new | send-queue depth by role: `RCCL_RMA_SEND_QP_WRS` on a multi-segment RMA comm, the base depth otherwise |
| `connect.cc` | `IbMsegHandle` | 118 | 3 | changed | adds `isGin` |
| `connect.cc` | `rcclIbMarkGinListen` | 821 | 3 | new | marks GIN and RMA listens so their comms skip the layout |
| `connect.cc` | `IbMsegSenderQpsCreate` | 842 | 4 | changed | uses `rcclIbSendQpWrs` |
| `connect.cc` | `IbMsegSenderQpsToRts`, `IbMsegReceiverQpsCreateToRts` | 930, 1386 | 4 | changed | bound the peer's device index |
| `connect.cc` | `IbMsegConnectImpl` | 999 | 3, 4 | changed | exchanges and checks `useMultiSeg` (fork 3); sets `isRma` (fork 4) |
| `connect.cc` | `IbMsegCreateFlushQp` | 1535 | 4 | changed | sets the loopback flush QP's device indices |
| `connect.cc` | `IbMsegAcceptImpl`, `IbMsegCloseSend`, `IbMsegCloseRecv` | 1728, 2223, 2265 | 3 | changed | check `useMultiSeg`, allocate and free the CTS slots |
| `connect_mseg.h` | `IbMsegConnectionMetadata` | 51 | 3 | changed | adds `useMultiSeg` (line 74) |

### `common_mseg.h`, `gin_mseg.h`, `init.cc`

| File | Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|---|
| `common_mseg.h` | `IbMsegDev` | 97 | 3 | changed | adds `maxMr` (116) |
| `common_mseg.h` | `IbMsegRequest` | 225 | 3 | changed | adds `send.mh`, the MR handle the segmented send reads per-segment lkeys from |
| `common_mseg.h` | `IbMsegCtsSlot`, `NCCL_IB_CTS_FIFO_BYTES`, `NCCL_IB_CTS_SLOTS_BYTES` | 305, 314, 315 | 3 | new | combined CTS entry and layout slot |
| `common_mseg.h` | `rcclIbSendCursor` | 320 | 3 | new | the slot whose segmented send is still posting |
| `common_mseg.h` | `IbMsegMrHandle` | 421 (field) | 3 | changed | adds `seg`, the per-segment MR set |
| `common_mseg.h` | `IbMsegNetCommBase` | 429 | 4 | changed | adds `isRma` (456) |
| `common_mseg.h` | `rcclIbMrAdmit` (declaration) | 477 | 3 | new | defined in `reg.cc` |
| `common_mseg.h` | `IbMsegSendComm` | 578 | 3 | changed | adds `ctsSlots`, `useMultiSeg`, `sendCursor`, `layoutCache` (605 to 607) |
| `common_mseg.h` | `IbMsegRecvComm` | 687 | 3 | changed | adds `ctsSlots`, `ctsSlotStride`, `useMultiSeg` (699 to 701) |
| `common_mseg.h` | `IbMsegListenComm` | 745 | 3 | changed | adds `isGin` (751) |
| `gin_mseg.h` | `IbMsegGinIbCollComm` | 15 | 4 | changed | adds `regRecords` (40) |
| `init.cc` | `IbMsegInitDevices` | 550 | 3 | changed | records `devAttr.max_mr` (705) |

### `reg.cc`

| Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|
| `rcclIbMrAdmit` | 65 | 3 | new | admits a registration only if one MR per segment fits under `max_mr` |
| `IbMsegRegMrDmaBufInternal` | 78 | 3, 4 | changed | a real fd stays one MR (fork 3); unwinds MRs on earlier devices when a later device fails (fork 4) |
| `rcclIbCountRangeSegments`, `rcclIbRegMrDmaBufFromRange` | 111, 131 | 3 | new | fd -1: count the range's segments and register each |
| `IbMsegRegMrDmaBuf` | 198 | 3 | changed | routes fd -1 to the range path |
| `rcclIbMultiSegRegCtx`, `rcclIbMultiSegRegOne`, `rcclIbMultiSegDeregOne` | 232, 241, 254 | 3 | new | per-segment register and deregister callbacks |
| `rcclIbMrHandleFreeSegments`, `rcclIbMrHandlePublishLayout` | 262, 268 | 3 | new | free the MR set; publish its layout |
| `IbMsegRegMrDmaBufMultiSeg` | 278 | 3 | new | one MR per segment per device, all or nothing |
| `rcclIbDeregMrMultiSeg` | 332 | 3 | new | deregister a multi-segment handle |
| `IbMsegDeregMr` | 346 | 3 | changed | routes a multi-segment handle to `rcclIbDeregMrMultiSeg` |

### `p2p.cc`

| Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|
| `rcclIbSendHoldSlice` | 89 | 3 | new | holds one more event per request until a slice's signaled write completes |
| `IbMsegMultiSend` | 93 | 3 | changed | segmented send through `rcclIbFillSendWrs`; uses the validated `nreqs` |
| `rcclIbSendResume`, `rcclIbSendProgress` | 317, 348 | 3 | new | resume a partly posted segmented send |
| `IbMsegIsend` | 365 | 3 | changed | strips the layout bit; waits for layouts |
| `IbMsegPostFifo`, `IbMsegIrecv` | 531, 615 | 3 | changed | the one-write CTS with the layout prefix |
| `IbMsegIflush` | 763 | 3 | changed | one-read flush: the read uses the MR of the segment holding the last byte |
| `IbMsegCompletionEventProcess`, `IbMsegTest` | 1053, 1190 | 3 | changed | slice and layout-read completions |

### `gin.cc`

| Symbol | Line | Fork | Kind | Role |
|---|---|---|---|---|
| `IbMsegGinIbCloseColl` | 296 | 4 | changed | frees `regRecords` |
| `IbMsegGinIbGdakiListen` | 352 | 3 | changed | marks the listen as GIN |
| `rcclRmaIbProxyInstallMultiSeg` | 455 | 4 | new | swaps the RMA vtable's MR and data entries for the multi-segment ones |
| `IbMsegRmaIbProxyInit` | 465 | 4 | changed | installs them once when the switch is on |
| `IbMsegRmaIbProxyListen` | 491 | 3 | new | RMA listen that marks per-peer comms as GIN |
| `IbMsegRmaIbProxyConnect` | 497 | 4 | changed | calls `rcclRmaIbProxyConnectFinish` |
| `IbMsegRmaIbProxyCtx` | 520 | 4 | changed | adds `pending`, the per-peer round queues (527) |
| `IbMsegRmaIbProxyPostBatch`, `IbMsegRmaIbProxyDestroyContext` | 531, 673 | 4 | changed | pump and free the round queues |
| `IbMsegRmaIbProxyCreateContext` | 579 | 3, 4 | changed | marks the per-peer listen (fork 3); allocates the round queues (fork 4) |
| `rcclRmaIbProxySegMrHandle`, `rcclRmaIbProxyMrSegmentCount` | 772, 791 | 4 | new | the multi-segment MR handle and its count |
| `rcclRmaRemoteRkey`, `rcclRmaRankOff`, `rcclRmaSegOf`, `rcclRmaRangeOk` | 795, 799, 803, 807 | 4 | new | per-rank segment lookups and the range check |
| `rcclRmaBuildSegmentedWrs`, `rcclRmaPostWrs` | 814, 879 | 4 | new | split an op at segment boundaries and post it |
| `rcclRmaIbProxyQueueWrs`, `rcclRmaFinishPostedRequest`, `rcclRmaIbProxyStartRequest`, `rcclRmaIbProxyQueueRequest` | 896, 921, 937, 959 | 4 | new | request bookkeeping for a partial post |
| `rcclRmaIbRoundBuild`, `rcclRmaIbRoundOutstanding`, `rcclRmaIbRoundPost`, `rcclRmaIbRoundRetire`, `rcclRmaIbRoundFailed`, `rcclRmaIbRoundHooks` | 972, 981, 983, 989, 996, 1000 | 4 | new | the round hooks for `seg_rma_round_mseg.h` |
| `rcclRmaIbProxyQueueOf`, `rcclRmaIbProxyPeerBusy`, `rcclRmaIbProxyPostChain`, `rcclRmaIbProxyEnqueue`, `rcclRmaIbProxySubmit` | 1005, 1010, 1015, 1029, 1058 | 4 | new | per-peer ordering: post now or queue behind earlier rounds |
| `rcclRmaDeregOneSegment`, `rcclRmaIbProxySegMrHandleFree`, `rcclRmaSegRegCtx`, `rcclRmaRegOneSegment`, `rcclRmaRegAllGather` | 1075, 1080, 1093, 1102, 1127 | 4 | new | per-segment register, unwind and the consensus all-gather |
| `rcclRmaIbProxyConnectFinish` | 1132 | 4 | new | allocates the per-rank registration records |
| `rcclRmaIbProxyRegMrSymDmaBuf`, `rcclRmaIbProxyDeregMrSym`, `rcclRmaIbProxyRegMrSym` | 1146, 1289, 1318 | 4 | new | register a window one MR per physical segment after consensus |
| `rcclRmaIbProxyUseSingle` | 1307 | 4 | new | one-segment handles take the copied single-MR op bodies |
| `rcclRmaIbProxyIPut`, `rcclRmaIbProxyIGet`, `rcclRmaIbProxyIPutSignal`, `rcclRmaIbProxyIFlush`, `rcclRmaIbProxyTestFailed` | 1324, 1352, 1384, 1437, 1473 | 4 | new | multi-segment data ops |
| `IbMsegRmaIbProxy` | 1799 | 3 | changed | uses `IbMsegRmaIbProxyListen` |

## Tests

Design: [Testing](multi_segment_testing.md). Counts are test definitions at fork 4; "new" files are created by the stack.

| File | Suites | Fork | Covers |
|---|---|---|---|
| `test/host/dev-runtime-test.cc` | `SymMemoryMapLsaTest`, `SymMemoryObtainRegisterTest`, `SymMemoryRegisterGinTest`, `SymMemoryRegisterGinElasticTest`, `SymMemoryRegisterRmaTest`, `RmaProxyInflightCapTest` | 1 | owner fds, GIN and RMA window gates, the in-flight cap |
| `test/host/gin/gin-host-test.cc`, `gin-host-proxy-test.cc` | `GinHostRegisterMicrotest`, `GinHostProxyRegMrMicrotest`, `GinHostProxyContextMicrotest` (fork 1); `GinHostDevCommSetupMicrotest` (fork 4) | 1, 4 | owner-fd registration, the queue cap, the RMA flag handoff |
| `test/host/transport/net-test.cc` | `NetProxyRegBufferMicrotest`, `NetProxyRegBufferCuMemMicrotest` (fork 1); `NetProxySetupMicrotest`, `NetRecvProxyConnectMicrotest`, `NetRecvProgressMicrotest` (fork 2) | 1, 2 | fd -1 or stage; GDR flush flag on IB-MSEG |
| `test/host/ce-coll-test.cc`, `proxy-test.cc`, `rma/rma-proxy-test.cc`, `rma/rma-proxy-launch-test.cc` | `HierCeAvailableMicrotest`, `ProxyServiceFixture`, `RmaProxyAllocGraphTest`, `RmaProxyDescriptorTest` | 1 | skipped RMA registration and the copied net flag |
| `test/host/fakes/rccl_mseg_fakes.*` | fakes | 1 | the `rccl_dev_runtime.cc` gates for host tests |
| `test/host/init-test.cc`, `plugin/plugin-net-test.cc`, `plugin/rma-plugin-init-test.cc`, `ras/ras-diagnostics-common-test.cc` | `InitMicrotest`, `PluginNetMicrotest`, `RmaPluginListTest`, `RasDiagnosticsCommonMicrotest` | 2 (`InitMicrotest` also 3, 4) | plugin selection, comm flags, RAS list |
| `test/transport/NetIbMPI/GeneralTests.cpp` and the other `NetIbMPI` files | `NetPluginSelection` (6), `NetIbMPITest`; `NetIbMsegInspect.hpp` (new) | 2 | selection and the `*_mseg` reruns of the `net_ib` MPI suites |
| `test/NetIbMultiSegmentUnitTests.cpp` (new) | `NetIbMultiSeg` (8), `NetIbCtsLayout` (10), `NetIbLayoutBlob` (2) | 3 | connect decision, segment selection, CTS layout and blob checks |
| `test/transport/NetIb/NetIbSegmentedSendPrefixTests.cpp` (new) | `NetIbSegmentedSendPrefix` (30), `NetIbSegmentedFlushPrefix` (8), `NetIbSegmentedHooks` (19) | 3 | slice posts, resume, the one-read flush and the P2P hooks |
| `test/transport/NetIbMPI/NetIbMultiSegmentMPITests.cpp` (new) | `NetIbMultiSegmentMPITest` (18) | 3, 4 | P2P across segments: layouts, flush, GDR flush flag, switch off |
| `test/RegistrationMPITests.cpp` | `UBR_MultiSegment` (18: 13 from develop, 1 added in fork 3, 4 in fork 4) | 3, 4 | end-to-end registration, GIN and LSA windows, hybrid windows |
| `test/common/IbFakePostSend.hpp`, `MultiSegmentVmmHelpers.hpp`, `TestChecks.hpp` (new) | helpers | 3, 4 | fake post-send and multi-segment VMM allocation |
| `test/host/transport/net_ib_multiseg/connect-test.cc` | `NetIbMsegConnectMicrotest`, `NetIbMsegConnectMultiSegMicrotest`, `NetIbMsegConnectUseMultiSeg` | 3, 4 | connect mismatch check, CTS slots, QP depth by role |
| `test/host/transport/net_ib_multiseg/reg-test.cc` | `NetIbMsegRegMicrotest`, `NetIbMsegRegRangeMicrotest` | 3, 4 | per-segment registration, MR budget, unwinding |
| `test/host/transport/net_ib_multiseg/net-ib-p2p-test.cc` | `NetIbMsegP2pMicrotest` | 3 | P2P post and flush paths |
| `test/host/transport/net_ib_multiseg/gin-test.cc` | `NetIbMsegGinMicrotest`, `NetIbMsegGinRmaSeqId` | 3, 4 | RMA registration, consensus, rounds and the post path |
| `test/RmaSegmentMathTests.cpp` (new) | `RmaSegmentMathTest` (27: 3 in fork 1, 24 in fork 4), `RmaRoundTest` (6) | 1, 4 | table fit, in-flight cap, WR counting, keep-or-free rules and rounds |
| `test/transport/RmaMPI/RmaMultiSegmentMPITests.cpp` (new) | `RmaMultiSegmentMPITest` (32), `RmaMultiSegmentPostMPITest` (13) | 4 | put, get, signal and flush across segments; DeepEP and hybrid windows; WR batching, the cap, rejected ops |
| `test/common/HybridVmmHelpers.hpp`, `MultiSegmentTestSupport.hpp` (new) | helpers | 4 | hybrid host and device VMM windows, freed collectively |
| `test/host/alloc-test.cc` | `AllocMicrotest` | 4 | `ncclCuMemGetAddressRange` now releases the retained handle when the properties read fails (AICOMRCCL-3043) |
| `test/host/transport/derived_transport_tests.cmake`, `derived_transport_tests*.h`, `derived_transport_tests_check.cc` (new) | `NetIbMsegBaseTestExclusions` (3 self-checks) | 4 | compile-twice: build `net_ib`'s host tests against the fork through `rename_map.txt` |
| `test/host/transport/net_ib_multiseg/host_tests.cmake`, `shim/` (new) | build | 4 | the four IB-MSEG host-test binaries; the seam shims for `connect.cc`, `gin.cc`, `p2p.cc` |

The compile-twice commit replaced the fork's copied `net_ib` host tests and `transport_net_ib_mseg_*` fakes with the derived build, so the `net_ib_multiseg/*-test.cc` files now hold only the fork's own tests on top of the included `net_ib` tests. The design is in [compile-twice host tests](multi_segment_fork_plugin.md#compile-twice-host-tests).

## Runner configs

`tools/scripts/test_runner/configs/`: jobs added per fork. No fork removes a job. Fork 1 adds none.

| Config | Fork 2 | Fork 3 | Fork 4 | Note |
|---|---|---|---|---|
| `mi300x_mellanox_ib.json` | 7 | 8 | 5 | the two `ubr_multisegment_*` jobs were already on develop |
| `mi300_mellanox_ib_func.json` | 7 | 8 | 7 | |
| `mi355x_thor2_roce.json` | 7 | 8 | 7 | |
| `mi355x_thor2_roce_func.json` | 7 | 8 | 7 | |
| `net_ib_transport.json` | 10 | 5 | 5 | |
| `mi455_ainic_roce.json`, `mi455_ainic_roce_func.json` | 0 | 0 | 0 | AINIC entries come with fork 5 |

Fork 2 adds the `*_mseg` reruns of the `net_ib` suites, fork 3 the `net_ib_multisegment*` and `multisegment_*` P2P jobs, and fork 4 the `net_ib_rma_multisegment*`, `multisegment_rma_segment_math` and `ubr_multisegment_*` jobs.
