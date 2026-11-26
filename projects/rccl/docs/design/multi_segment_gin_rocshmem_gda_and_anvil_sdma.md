# Proposal: multi-segment windows on rocSHMEM GDA and Anvil SDMA

> [!NOTE]
> Status: proposal, not implemented. Rechecked against the current stack tip (forks 1 to 4 and the tests-only change on develop): neither backend plugin has changed, so nothing here is implemented yet. The next step is an owner decision on whether and in what order to take it up.

The host-proxy GIN type registers per physical segment through its RMA backend ([GIN/RMA](multi_segment_gin_rma.md)). The two AMD device-initiated GIN backends are rocSHMEM GDA (`NCCL_GIN_TYPE=6`) and Anvil SDMA (`NCCL_GIN_TYPE=7`); both plugins compile only with `ENABLE_ROCSHMEM_GIN`. This doc records what develop and the fork stack do with multi-segment windows on those two backends, what is missing, and the proposed changes. The fork stack does not change either backend plugin; it changes only the core registration code in front of them (`gin_host.cc`, `gin_host_proxy.cc`, `dev_runtime.cc`). Tickets: AIRUNTIME-2351, AICOMRCCL-1526 follow-on.

## Summary

- **Windows with host segments are split but not registered.** DeepEP's `[GPU][CPU]` and hybrid windows contain host segments, so develop builds one GIN segment per physical segment, and the device `ncclGin` put and get split at segment boundaries before they reach the backend. Registration, however, asks every active GIN backend for DMA-BUF support whenever a window has a host segment or more than one segment, and GDA and Anvil report only `NCCL_PTR_CUDA`, so `ncclGinRegister` fails these windows with `ncclInvalidArgument` on types 6 and 7. The same check is in develop and in fork 4.
- **Device-only windows with 2 to `RCCL_IB_REG_SEG_DEFAULT` segments are the other gap.** Because a whole-range DMA-BUF covers only segment 0 on ROCm, develop's HIP gate in `symMemoryRegisterGin` keeps these windows off GIN: a `NCCL_WIN_GIN_ONLY` window fails with `ncclInvalidUsage`, and any other stays unregistered with GIN and usable for LSA. Fork 1's `rcclDevrGinRegistersMultiSegment` lets them through only when every GIN backend is the proxy over IB-MSEG's RMA backend, so types 6 and 7 still hit the gate. Anvil exports nothing and addresses every segment through the LSA flat mapping, so only that gate (and the DMA-BUF check behind it) stops it.
- **Proposed:** let Anvil through both checks; give GDA one GIN window per segment so develop's device loop does the split; make GDA register host segments with the owner's fd and then report DMA-BUF support; stop Anvil spending an IPC-table row per segment.
- **Open:** whether SDMA reaches host segments, VA signal cells outside a window's first GIN segment, and GDA's missing device get.

## Develop changes reviewed

| Date | Change | Effect here |
|---|---|---|
| 10-27 | GDA VA signals: the signal is an RDMA atomic into a cell of a data window, so data MRs now enable remote atomics | a VA signal resolves through its window's keys, not only the signal array |
| 10-25 | Anvil flush drains dirty queues by bit scan | none; refactor |
| 10-25 | Anvil strong signals drain every SDMA queue to the peer first | a strong signal after a split stays ordered with no extra work |
| — | signals de-aliased per logical context on both backends (two commits) | none; signals are per context, not per window |
| 10-18 | Anvil small puts use IPC stores below `sdmaThreshold` | host-segment routing can reuse the IPC store path |
| — | GDA traffic class, ionic collapsed CQ, MI300X CPX gate, plugin finalize (four commits) | none |

Corrections to the first draft: the GIN types are 6 and 7 (type 5 is EFA GDA); Anvil performs no export of its own, so it never "maps only the GPU object"; GDA has no device get; and windows with host segments were never on a single-MR path.

## What develop does with multi-segment windows

`ncclDevrBuildGinSegmentInfos` (`dev_runtime_segments.cc`) sets `numGinSegments = 1` for a device-only window and one GIN segment per physical segment, with its memory type, otherwise. `symMemoryRegisterGin` then calls `ncclGinRegister` once per GIN segment, so a backend's `regMrSym` sees one physical segment at a time. Develop requires identical segment counts and sizes on every rank (`ncclDevrVerifySegmentLayouts`).

On AMD, before that, a HIP-only gate in `symMemoryRegisterGin` keeps a device-only window with more than one segment on some rank off GIN, because one HIP DMA-BUF covers only the first segment: a `NCCL_WIN_GIN_ONLY` window fails with `ncclInvalidUsage`, and any other gets `numGinSegments = 0` (INFO line) and stays usable for LSA. For the windows that pass, `rcclDevrGinSegmentRegType` (`rccl_dev_runtime.cc`) sets the pointer type to `NCCL_PTR_CUDA` for every segment, host or device, and asks for DMA-BUF when the window has more than one segment on any rank or any host segment; a host window without `NCCL_DMABUF_ENABLE=1` fails with `ncclInvalidUsage`. With DMA-BUF requested, `ncclGinRegister` (`gin/gin_host.cc`) fails with `ncclInvalidArgument` unless every connection of every active backend reports `NCCL_PTR_DMABUF`. `ginRocshmemGdaGetProperties` and `ginAnvilGetProperties` set `ptrSupport = NCCL_PTR_CUDA` only, so on types 6 and 7 no window with a host segment or with more than one segment registers today.

On the device, `ncclDevrAllocAndPopulateSegmentWindows` publishes `ginMultiSegmentWins[backend * numSegments + seg]` with each segment's size and type. With the `ncclGin_SegmentMixed` or `ncclGin_SegmentHostNuma` tag, `put` and `get` find the segment of each offset (`findSegmentFromWindow`), cut every slice at both boundaries (`getSegmentChunkSize`), and call the backend once per slice with that segment's GIN window. The signal and counter ride on the last slice, and a host source raises the first slice's release to system scope. The default tag, `ncclGin_SegmentDevice`, takes the one-window path; RCCL's own kernels use it, and only `test/device/GinBackendMaskDispatch_test.cpp` passes the mixed tag.

`ginWinsDefaultBackend` is copied from the window-level cache, which is filled only when there is one GIN segment. A one-window lookup on a multi-GIN-segment window, from the default tag or from a VA signal descriptor (`getGinWindow`), therefore gets no valid window on backend 0 and segment 0's window on other backends.

## What the fork stack adds

Fork 1 adds an exception to develop's HIP gate in `symMemoryRegisterGin` (`dev_runtime.cc`, AMD builds only): a device-only multi-segment window passes when `rcclDevrGinRegistersMultiSegment` (`rccl_dev_runtime.cc`, declared in `include/rccl_common.h`) holds, and otherwise keeps develop's outcome (`ncclInvalidUsage` for a GIN-only window, unregistered with GIN for any other). The gate runs before `ncclDevrVerifySegmentLayouts`. At the current stack tip the predicate is `rcclDevrRmaRegistersMultiSegment` plus a backend check, so it requires all of the following:

- `comm->rcclMultiSegRma` is set. `init.cc` sets it through `rcclMultiSegOn` only when `NCCL_MULTI_SEGMENT_REGISTER=1`, `NCCL_DMABUF_ENABLE=1` and the RMA backend is IB-MSEG's `IbMsegRmaIbProxy`; a child comm copies its parent's flag.
- Every rank has the same segment count, and `rcclRmaSegTableFits` holds: 1 to `RCCL_IB_REG_SEG_DEFAULT` (`transport/rccl_mseg.h`) segments, and ranks times segments at most `RCCL_RMA_SEG_TABLE_MAX_ENTRIES` (2^20, `transport/rccl_mseg_rma.h`).
- Every active, non-closed GIN backend is `NCCL_GIN_TYPE_PROXY`.

So a device-only multi-segment window is kept off GIN on type 6 and on type 7 whatever the other settings are, and on a proxy comm whose RMA backend is not IB-MSEG. A window that passes registers as one GIN window through `rcclGinRegisterMultiSeg` (`gin/gin_host.cc`), which always asks for DMA-BUF support and hands fd -1 only to the proxy so its RMA backend walks the segments; any other backend would get a plain `regMrSym` over the whole range. A window with host segments skips the gate and reaches the DMA-BUF check above, which rejects it on types 6 and 7.

Fork 1 also publishes host segments' owner DMA-BUF fds by VA (`rcclHostSegmentFdRegister`). A rank uses its own fd when it exported one, and an importer borrows the owner's fd only when exactly one LSA rank can export that segment. `ncclGinRegister` looks the fd up (`rcclHostSegmentFdLookup`) and, when it finds one, calls the backend's `regMrSymDmaBuf` with it. Both device backends implement `regMrSymDmaBuf` but ignore the offset and fd.

```mermaid
%% caption: How develop and the fork stack route a multi-segment window on GIN types 6 and 7
flowchart TD
 w["multi-segment window"] --> host{"any host segment?"}
 host -->|no| gate{"fork 1 exception?"}
 gate -->|"no: types 6 and 7"| rej["off GIN; GIN-only fails"]
 gate -->|"yes: proxy only"| proxy["proxy: per-segment MRs"]
 host -->|yes| per["one GIN window per segment"]
 per --> dma{"backend reports DMA-BUF?"}
 dma -->|"no: types 6 and 7"| inv["ncclInvalidArgument"]
 dma -->|yes| tag{"mixed segment tag?"}
 tag -->|yes| loop["device loop: one call per slice"]
 tag -->|no| one["one-window path: wrong past segment 0"]
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef rma fill:#f5ebfb,stroke:#af52de,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 classDef failpath fill:#ffeceb,stroke:#ff3b30,color:#1d1d1f
 class w,per gin
 class loop okpath
 class proxy rma
 class one,rej,inv failpath
```

## Current backend code

### rocSHMEM GDA

`ginRocshmemGdaRegMrSym` (`gin_plugin_rocshmem_gda.cc`) calls `rocshmem_gin_reg_mr_vmm(qpSet, data, size, atomic=1, ...)` once. That exports one DMA-BUF for the whole range with `hipMemGetHandleForAddressRange`, registers it with the VA as IOVA, and tries a fallback when the export fails. Remote atomics are on because a VA signal lands in the window. The plugin then allgathers one rkey and one base VA per rank. `ginRocshmemGdaRegMrSymDmaBuf` ignores `offset` and `fd` and calls `RegMrSym`.

The plugin finds its NICs in IB-CAST's device table (`IbCastInitDevices`, `IbCastDevs`, `IbCastGetPhysProperties`). Fork 5, which is paused, would move rocSHMEM GDA to the fork's device table; that would change where GDA gets its NIC properties, not the registration entry points this proposal changes.

The device handle is `ncclGinRocshmemGdaMemHandle {local_va, remote_vas[nRanks], lkey, rkeys[nRanks]}`. The put issues one `put_nbi(remote_vas[peer] + dstOff, rkeys[peer], local_va + srcOff, lkey, bytes)`, skips zero-byte RDMA but still signals, and sends the signal as an `atomic_add` on the same QP. A VA signal targets `remote_vas[peer] + signalOffset` with `rkeys[peer]` of the signal window. Put-value inlines the value into the WQE, and flush runs `quiet` on every peer QP.

Get, FlushAsync and Wait are `__builtin_trap()`, so GDA has no device get. `FlushesAllPutsOnAnySignal` is false. `ginRocshmemGdaQueryLastError` always reports no error, so device code has no way to report a failure. Signals and counters are per logical context, in separate allocations registered with remote atomics.

### Anvil SDMA

Anvil is intra-node and uses no IB MR and no DMA-BUF export. `ginAnvilRegMrSym` (`gin_plugin_anvil_sdma.cc`) resolves the LSA flat address with `ncclDevrGetLsaSelfAddr`, adds an IPC-table row on the first registration of an address (reference-counted), and publishes `baseAddr`, `vmmStride = devr->bigSize` and a `remote_vas[nRanks]` table computed from the same stride. `resolveRemotePeerVa` uses `base + (peer - rank) * stride + off` when the stride is non-zero, then `remote_vas[peer] + off`, then the IPC table.

The device runtime maps every segment of each LSA peer's window into that flat range (`symMemoryImportAndMapSegmentsForRank`), so stride addressing already reaches every segment of a device-only multi-segment window. `ginAnvilRegMrSymDmaBuf` ignores `fd`.

The put uses IPC flat stores up to `sdmaThreshold` bytes (default 128, `NCCL_GIN_ANVIL_SDMA_THRESHOLD`) and the peer's SDMA queue above it. SDMA copies are cut at `kGinPutSegBytes` = 128 MiB (`gin_anvil_sdma_put_policy.h`) because the linear-copy count is 30 bits and a fused copy and signal of 256 MiB or more hangs MI355X; an optional fused signal rides on the last piece. There is one in-order SDMA channel per peer. Weak signals quiet only this call's queue, strong signals drain every queue to the peer (`quietPeerQueues`), and a VA signal cell resolves like a put destination. Get is implemented over SDMA or IPC and traps when it cannot resolve an address.

> [!WARNING]
> The IPC table is a process-global array of 16 rows (`NCCL_GIN_ANVIL_IPC_MAX_BUFS`), each covering up to 16 LSA ranks (`NCCL_GIN_ANVIL_IPC_MAX_RANKS`), and registration fails when it is full. Develop's per-segment path takes one row per segment, so once host-segment windows register, one 8-rank hybrid window (1 GPU plus 8 CPU segments) uses 9 rows and a second one fails.

## Gaps

| Window | GDA (type 6) | Anvil (type 7) |
|---|---|---|
| One device segment | works | works |
| Device-only, 2 to `RCCL_IB_REG_SEG_DEFAULT` segments | one MR would cover segment 0 only; the HIP gate keeps it off GIN | addressing reaches every segment; the HIP gate keeps it off GIN |
| Host segments, local CPU segment | rejected at registration (no `NCCL_PTR_DMABUF`); past that, per-segment MRs and the mixed-tag split would work | rejected at registration; past that, per-segment rows and the split would work, and SDMA into the host segment is unverified |
| Host segments, imported CPU segment (hybrid) | rejected at registration; past that, re-exports instead of using the owner fd | rejected at registration; past that, works through the LSA mapping and IPC rows run out |
| VA signal cell in a multi-GIN-segment window | no valid window on backend 0 | no valid window on backend 0 |
| Device get | traps | works |

## Proposed changes

### Anvil: accept device-only windows

Change the backend check in `rcclDevrGinRegistersMultiSegment`: today it requires `rcclDevrRmaRegistersMultiSegment` and then rejects any non-proxy backend. Keep that RMA requirement for the proxy, and let Anvil pass with only `NCCL_MULTI_SEGMENT_REGISTER=1`, the same segment count on every rank, and `rcclIbSegCountSupported` (at most `RCCL_IB_REG_SEG_DEFAULT`). Anvil uses no RMA segment table, so the ranks-times-segments bound does not apply. A window that passes registers through `rcclGinRegisterMultiSeg`, which asks every backend for DMA-BUF support, so Anvil must report `NCCL_PTR_DMABUF` (it already accepts `regMrSymDmaBuf` and addresses by LSA VA), or the check must skip backends that do not register NIC memory. Anvil then gets a plain `regMrSym` over the whole range, so the window stays one GIN window over the LSA flat range, and no device code changes.

### Anvil: no IPC row per segment

When `vmmStride` is non-zero, every peer address resolves without the table, so skip `ncclGinAnvilIpcTableRegisterVmm` for data windows and keep the table for explicit rows such as signal spans. Raising `NCCL_GIN_ANVIL_IPC_MAX_BUFS` is the fallback if a path still needs the rows.

### Anvil: host segments

Host-segment windows reach Anvil once it reports `NCCL_PTR_DMABUF`. Check on MI300X and MI355X that SDMA reads and writes host segments mapped in the LSA range. If it cannot, record the segment kind on `ncclGinAnvilSdmaMemHandle` at registration and send those puts through IPC stores regardless of size. On AMD, `symMemoryRegisterGin` passes `NCCL_PTR_CUDA` for every segment, so the kind has to come from `ginSegmentInfos[segment].memType` (host segments are `CU_MEM_LOCATION_TYPE_HOST_NUMA` there) through a new argument or an RCCL-side lookup. Without GPU access to imported host VMM (ROCM-29812), skip collectively, as the hybrid MPI tests do; never hang in `waitSignal`.

### GDA: one GIN window per segment

When an active GIN backend is GDA, build one GIN segment per physical segment for device-only windows too, so each segment gets its own `rocshmem_gin_reg_mr_vmm` MR and develop's device loop splits every put. The fork 1 gate then needs a GDA branch like Anvil's (switch, same count, at most `RCCL_IB_REG_SEG_DEFAULT` segments), and `symMemoryRegisterGin` must send these per-segment GIN windows through `ncclGinRegister` at each segment's offset instead of `rcclGinRegisterMultiSeg`, which registers one window from the window base. GDA must also report `NCCL_PTR_DMABUF`, which it can do honestly only after the owner-fd change below. Default-tag callers would then hit the one-window path, so the device-only branch in `gin__funcs.h` must fall through to the loop when `numSegments > 1`. That is one compare in an upstream-shared file and needs a `// RCCL:` marker.

| Option | Idea | Verdict |
|---|---|---|
| A | per-segment GIN windows plus develop's device loop | recommended: no new backend device code; the loop already puts the signal on the last slice and raises the release for host sources |
| B | per-segment MRs inside the plugin plus a GDA device loop over `rkeys[rank * nSeg + s]` (the first draft) | duplicates the loop and needs a slice cap and an error channel that `QueryLastError` does not have |

```mermaid
%% caption: Proposed handling of a device-only multi-segment window per GIN backend
flowchart TD
 d["device-only window, 2 to RCCL_IB_REG_SEG_DEFAULT segments"] --> be{"GIN backend"}
 be -->|"proxy"| p["unchanged: per-segment MRs"]
 be -->|"Anvil"| a["one window over the LSA flat range"]
 be -->|"GDA"| g["one GIN window per segment"]
 g --> loop["device loop splits each put"]
 classDef gin fill:#e5f1ff,stroke:#007aff,color:#1d1d1f
 classDef rma fill:#f5ebfb,stroke:#af52de,color:#1d1d1f
 classDef okpath fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 class d gin
 class p rma
 class a,g,loop okpath
```

### GDA: owner fd for imported host segments

Implement `ginRocshmemGdaRegMrSymDmaBuf` with the fd fork 1 passes it: add a factory entry such as `rocshmem_gin_reg_mr_dmabuf(qpSet, fd, offset, addr, size, atomic, ...)` over the factory's `gin_ibv_reg_dmabuf_mr`, with the VA as IOVA like `reg_mr_vmm`. The caller keeps ownership of the fd. Fall back to `RegMrSym` when `fd < 0`. With that in place, GDA reports `NCCL_PTR_DMABUF` and host-segment windows pass `ncclGinRegister`.

### VA signals in multi-segment windows

Until the generic descriptor resolves the GIN segment that holds `signalOffset`, keep VA signal cells in windows with one GIN segment and say so in the API notes. The fix is a `findSegmentFromWindow` lookup in the VA signal descriptor builders, again in `gin__funcs.h`.

## DeepEP shapes

| Shape | Path today | GDA | Anvil |
|---|---|---|---|
| `[GPU][CPU]` non-hybrid | per-segment GIN windows, mixed tag; types 6 and 7 fail at registration | needs the owner-fd change and DMA-BUF support; then puts work, but no Engram IGet, since get traps | needs DMA-BUF support; then puts and gets work, and SDMA into the CPU segment is to verify |
| `[GPU k][CPU 0]…[CPU N-1]` hybrid ([hybrid DeepEP windows](multi_segment_testing.md#hybrid-deepep-windows)) | same | needs the owner-fd change | needs DMA-BUF support and the IPC-row change for a second window |
| GPU-only, N segments | kept off GIN by the HIP gate (GIN-only windows fail) | needs per-segment GIN windows | needs the gate change |

`NCCL_WIN_STRICT_ORDERING` stays a window flag. A split must not move the signal ahead of data slices: on GDA the atomic follows the last `put_nbi` on the same QP, and on Anvil the last slice carries the fused signal or the signal follows a quiet.

## Open questions

- Can SDMA read and write host segments mapped in the LSA range on MI300X and MI355X, or must host slices use IPC stores?
- Should the VA signal segment lookup land in the shared device layer, or should registration reject VA signal use on multi-GIN-segment windows?
- Does any GDA user need device get, for example DeepEP's Engram IGet? That is a GDA feature of its own, not a multi-segment change.

## Implementation sequence

1. Tests that pin today's behavior on types 6 and 7: the `ncclInvalidArgument` registration failure for windows with host segments, and the HIP gate for device-only multi-segment windows (`ncclInvalidUsage` for a GIN-only window, no GIN registration for any other).
2. Anvil DMA-BUF support, gate and IPC-row changes, then mixed-tag puts and gets on host-segment windows.
3. GDA owner-fd registration and DMA-BUF support, then mixed-tag puts on host-segment windows.
4. GDA per-segment GIN windows, its gate branch and the device-only fall-through.
5. Anvil host-segment SDMA check, then the VA signal rule or fix.
6. Do not wait on GDAKI (type 3) or GPI (type 4): RCCL does not build the internal GDAKI plugin on AMD, the optional IB-CAST GDAKI backend is separate work, and there is no AMD GPI plugin.

## Testing

Reuse the `RmaMultiSegmentMPITests.cpp` patterns in a GIN-device fixture that sets `NCCL_GIN_TYPE=6` or `7`.

| Case | GDA | Anvil |
|---|---|---|
| Single-segment regression | yes | yes |
| Device-only N segments, crossing put | after per-segment windows | after the gate change |
| Crossing get | not applicable; get traps | yes |
| `[GPU][CPU]` mixed-tag put | after the owner-fd change; today pins `ncclInvalidArgument` | after DMA-BUF support; checks SDMA into host; today pins `ncclInvalidArgument` |
| Hybrid 8-rank window, two windows | after the owner-fd change | after the IPC-row change |
| More than `RCCL_IB_REG_SEG_DEFAULT` segments | host reject | host reject |
| Layout mismatch | develop's identical-layout check, collective | same |
| VA signal in a multi-GIN-segment window | rejected or documented | same |
| Strong signal after a multi-slice put | atomic after the last slice | `quietPeerQueues` before the signal |
| No GPU access to imported host VMM (ROCM-29812) | skip | skip |

Capability skips must be collective.

## Other device GIN types

| GIN type | RCCL on AMD | Multi-segment status |
|---|---|---|
| 2 | proxy over an RMA backend | per-segment MRs over IB-MSEG's RMA backend (forks 1 to 4); IB-CAST's RMA backend would come with fork 6, which waits on the paused fork 5 |
| 3 | GDAKI: the internal plugin is not built on AMD; `IbCastGinIbGdaki` registers only with `RCCL_NET_IB_CAST_ENABLE_GDAKI` and `NCCL_NET=IB-CAST`, with `regMrSymDmaBuf = NULL` | out of scope |
| 4 | GPI: device templates only, loadable as an external plugin | not an RCCL deliverable |
| 5 | EFA GDA | not an AMD backend |
| 6 | rocSHMEM GDA | this proposal |
| 7 | Anvil SDMA | this proposal |

Out of scope: raising the `RCCL_IB_REG_SEG_DEFAULT` segment cap, multi-segment inside GDAKI or GPI, and making Anvil an inter-node path.

## Proposed file index

| Concern | File | Change |
|---|---|---|
| Window gate | `src/rccl_dev_runtime.cc`, `src/include/rccl_common.h` | `rcclDevrGinRegistersMultiSegment` accepts Anvil, and GDA once it has per-segment windows |
| DMA-BUF support | `src/gin/gin_plugin_anvil_sdma.cc`, `src/gin/gin_plugin_rocshmem_gda.cc` | report `NCCL_PTR_DMABUF` in the properties |
| GDA GIN segments | `src/dev_runtime_segments.cc`, `src/dev_runtime.cc` | one GIN segment per physical segment when GDA is active, registered through `ncclGinRegister` at each segment's offset |
| Device-only fall-through, VA signal lookup | `src/include/nccl_device/impl/gin__funcs.h` | loop when `numSegments > 1`; find the signal's segment |
| GDA DMA-BUF entry | `src/gin/gin_rocshmem_gda_factory.cc`, `src/gin/gin_plugin_rocshmem_gda.cc` | register with the given fd |
| Anvil IPC rows | `src/gin/gin_plugin_anvil_sdma.cc` | no row for stride-addressed windows |
| Anvil host segments | `gin_anvil_sdma_device_host_common.h`, `gin_anvil_sdma.h` | segment kind; IPC stores for host |
| Tests | GIN type 6 and 7 MPI suites; `test/device/GinBackendMaskDispatch_test.cpp` for the mixed tag | crossing puts and gets, hybrid windows, VA signals |
