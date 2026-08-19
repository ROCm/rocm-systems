# AllToAll DDA end-to-end trace (MI350 IPC vs MI450 fabric)

Snapshot: `origin/develop` @ `67d45e7ccfe` (2026-08-14). Companion to
`mi450-mi350-coexistence-develop.md`. Traces `ncclAllToAll` from the public API
down to the device kernel for both architectures.

MI450 = gfx1250 / CDNA5 (UALink/UALoE fabric). MI350 = gfx950 / CDNA4 (Infinity
Fabric / xGMI mesh). MI300 = gfx942 / CDNA3 (also uses the IPC path).

## 0. Shared entry -- `ncclAllToAll` (`projects/rccl/src/collectives.cc`)

All arches enter the same function and hit one DDA gate before falling back to
the generic ring/pivot path:

```cpp
// totalBytes = nRanks * count * typesize; per-arch 4 MiB caps (rccl_common.h)
if (rcclDdaEnabled(comm, comm->nRanks*count*ncclTypeSize(datatype),
                   kDdaAlltoAllGfx942ThresholdBytes,   // 4 MiB
                   kDdaAlltoAllGfx950ThresholdBytes,   // 4 MiB
                   kDdaAlltoAllGfx1250ThresholdBytes)) // 4 MiB
{
  if (IsArchMatch(comm->archName, "gfx1250")) { /* MI450 fabric tiers */ }
  else if (ncclAllToAllDdaIpcEligible(...))   { /* MI350/MI300 IPC   */ }
}
// else -> info = {ncclFuncAlltoAll,...}; ncclEnqueueCheck(&info)  (generic ring/pivot)
```

`rcclDdaEnabled` (`rccl_wrap.cc`) is the shared per-arch gate: gfx1250 uses its
own threshold; gfx942/gfx950 require `nRanks>=8`; any other arch returns false.
AllToAll skips the `symEligible` check (no symmetric AllToAll kernel). The arch
fork is `IsArchMatch(comm->archName, "gfx1250")`.

Note: an earlier `ENABLE_ROCSHMEM` branch may take the GDA AllToAll path
(`rcclUseAlltoAllGda` + `rocshmemThreshold`) before the DDA gate; that is the
GIN/rocSHMEM route, separate from the DDA IPC/fabric paths traced here.

## A. MI350 (gfx950) -- DDA IPC path

Files: `dda_alltoall_ipc.cu`, `algorithms/alltoall/alltoall_dda.h`.

1. Eligibility -- `ncclAllToAllDdaIpcEligible`: needs IPC state
   (`ddaIpcMemHandler`, `ddaScratch`, `ddaPeerPtrsDev`, `ddaIpcBarrierState`),
   `nNodes==1`, `nRanks==kDdaNranks` (8), dtype in {fp32,fp16,bf16}, fits
   scratch, `count*typesize % 16 == 0`.
2. Setup (once) -- `init.cc` -> `ncclDdaIpcCommInit` (because
   `ncclDdaUseFabricPath` is false on gfx950): allocate VMM scratch, exchange
   IPC mem handles over bootstrap, open peers (xGMI peer access), fill
   `ddaPeerPtrsDev` (8 peer scratch bases), build `IpcGpuBarrier`.
3. Host launch -- `ncclAllToAllDdaIpc` -> `...Typed<int8_t>(count*typesize)`:
   - small (`ddaAlltoAllSingleBlockGrid`): `kStagingCopyInKernel=true` (fuse
     send->scratch copy into the kernel).
   - larger: host `cudaMemcpyAsync(ddaScratch, sendbuff, ...)`, staging=false.
4. Device kernel -- `ddaAllToAllIpc<T, NRANKS=8, hasAcc=false, staging>`:

```cpp
if constexpr (kStagingCopyInKernel) { copy sendbuff -> ipcbuffs[selfRank]; barrier<true,true>(); }
else                                { barrier<false,true>(); }
for (idx ...) {
#pragma unroll NRANKS
  for (int r=0; r<NRANKS; ++r)
    recvbuff[idx + r*idxEnd] = ipcbuffs[r][idx + selfRank*idxEnd];  // uint4 (16B) gather
}
barrier<true,false>();  // hold until peers finish reading
```

## B. MI450 (gfx1250) -- DDA fabric path (3 tiers)

Files: `dda_alltoall_fabric.cu`, `dda_alltoall_fabric_ll.cu`,
`dda_alltoall_fabric_ll128.cu`, `algorithms/alltoall/alltoall_dda_fabric.h`.

0. Tier selection in `ncclAllToAll` (all under the 4 MiB DDA enable):
   - `a2aBytes <= DDA_LL_THRESHOLD` (32 KiB) & LL-eligible ->
     `ncclAllToAllDdaFabricLL` (no GPU barrier).
   - else `<= DDA_LL128_THRESHOLD` (32 MiB) & LL128-eligible ->
     `ncclAllToAllDdaFabricLL128`.
   - else -> `ncclAllToAllDdaFabric` (VMM + `FabricGpuBarrier`).
1. Eligibility -- `ncclAllToAllDdaFabricEligible`: needs
   `ddaFabricMemHandler` + `ddaFabricBarrierState` + scratch/peer table, dtype
   in {fp32,fp16,bf16}, fits scratch, `%16==0`, and `2 <= nRanks <=
   kDdaMaxNranks` -- explicitly allows cross-node within an MNNVL clique.
2. Setup (once) -- `init.cc` -> `ncclDdaFabricCommInit` (because
   `ncclDdaUseFabricPath = MNNVL==1 && gfx1250`): VMM scratch
   (`ddaScratchIsVmm=true`), `ncclFabricMemHandler` that exchanges + imports
   peer memory over the UALoE fabric (`exchangeMemPtrs` / `getPeerDeviceMemPtr`
   -> `ddaPeerPtrsDev`), `FabricGpuBarrier` + `DdaFabricBarrierState`.
3. Host launch (VMM tier) -- `ncclAllToAllDdaFabric` -> `...Typed<int8_t>`:
   always stages host-side (`cudaMemcpyAsync`), then dispatches a compile-time
   specialized kernel by clique size:

```cpp
switch (nRanks) {
  case 4:  ddaAllToAllFabric<T,4><<<grid,block>>>(...); break;  // fully unrolled
  case 8:  ddaAllToAllFabric<T,8><<<grid,block>>>(...); break;  // fully unrolled
  default: ddaAllToAllFabric<T,0><<<grid,block>>>(...); break;  // runtime nRanks, 8-wide unroll
}
```

4. Device kernel -- `ddaAllToAllFabric<T, NRANKS_CT>`:

```cpp
barrier.syncOnSameBlockIdx<false,true>();          // FabricGpuBarrier
for (idx ...) {
#pragma unroll kUnroll                              // NRANKS_CT if >0, else 8
  for (int r=0; r<nRanksEff; ++r)
    recvbuff[idx + r*idxEnd] = ipcbuffs[r][idx + selfRank*idxEnd];  // uint4 gather (identical math)
}
barrier.syncOnSameBlockIdx<true,false>();
```

## Side-by-side

| Stage | MI350 (gfx950, IPC) | MI450 (gfx1250, fabric) |
|---|---|---|
| Dispatch fork | `else if (ncclAllToAllDdaIpcEligible)` | `if (IsArchMatch(...,"gfx1250"))` |
| Tiers | single kernel (+/- in-kernel staging) | LL (<=32KiB) / LL128 (<=32MiB) / VMM |
| Eligibility | `nNodes==1`, `nRanks==8` | `2<=nRanks<=kDdaMaxNranks`, cross-node in MNNVL clique |
| Setup | `ncclDdaIpcCommInit` -- IPC handles, xGMI peer open | `ncclDdaFabricCommInit` -- fabric handle import (VMM) |
| Peer memory | `ipcbuffs[]` = xGMI IPC maps | `ipcbuffs[]` = fabric-imported VMM |
| Barrier | `IpcGpuBarrier` | `FabricGpuBarrier` |
| Staging | in-kernel (small) or `cudaMemcpyAsync` | always `cudaMemcpyAsync` |
| Kernel | `ddaAllToAllIpc<T,8,...>` | `ddaAllToAllFabric<T,{4,8,0}>` |
| Data movement | uint4 gather `recv[idx+r*E]=peer[r][idx+self*E]` | identical uint4 gather |
| Fallback if ineligible | generic `ncclFuncAlltoAll` (ring/pivot) | generic `ncclFuncAlltoAll` (ring/pivot) |

## Net

Both arches share the entry gate (`rcclDdaEnabled`, 4 MiB), the eligibility
shape (dtype / 16-byte / scratch fit), and the same gather kernel math. They
diverge only in how peer memory is shared (xGMI IPC vs UALoE fabric import) and
which barrier synchronizes ranks -- selected purely by
`IsArchMatch(comm->archName,"gfx1250")` + `MNNVL`. MI450 additionally has the
LL / LL128 low-latency tiers.

## Code references (`origin/develop`)

- `projects/rccl/src/collectives.cc` -- `ncclAllToAll` dispatch + tier selection.
- `projects/rccl/src/rccl_wrap.cc` -- `rcclDdaEnabled` per-arch gate.
- `projects/rccl/src/include/rccl_common.h` -- `kDdaAlltoAllGfx{942,950,1250}ThresholdBytes` (4 MiB).
- `projects/rccl/src/dda_alltoall_ipc.cu` -- IPC eligibility + host launch.
- `projects/rccl/src/dda_alltoall_fabric.cu` (+ `_ll.cu`, `_ll128.cu`) -- fabric tiers.
- `projects/rccl/src/include/algorithms/alltoall/alltoall_dda.h` -- `ddaAllToAllIpc` kernel.
- `projects/rccl/src/include/algorithms/alltoall/alltoall_dda_fabric.h` -- `ddaAllToAllFabric` kernel.
- `projects/rccl/src/fabric_init.cu` / `init.cc` -- `ncclDdaUseFabricPath`, fabric vs IPC comm init.
