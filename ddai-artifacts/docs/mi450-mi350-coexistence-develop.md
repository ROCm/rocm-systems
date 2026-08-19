# How MI450 and MI350 co-exist in `develop`

Snapshot: `origin/develop` @ `67d45e7ccfe` (synced from `upstream/develop`,
2026-08-14). MI450 = gfx1250 / CDNA5 (UALink/UALoE fabric). MI350 = gfx950 /
CDNA4 (Infinity Fabric / xGMI mesh). MI300 = gfx942 / CDNA3.

## Summary

Both architectures live in a single build. Co-existence is achieved on two
levels:

1. **Compile-time**: one multi-target fatbin with per-arch device kernels.
2. **Runtime**: host-side `IsArchMatch(comm->archName, ...)` branching that
   routes each arch to a different transport. There is no `#ifdef` fork on the
   host -- both paths are always compiled in and selected per communicator.

MI450 (gfx1250 + `MNNVL==1`) uses the **DDA fabric-heap** path (VMM scratch +
fabric handles + `FabricGpuBarrier`). MI350 (gfx950) uses the **DDA IPC** path
(xGMI peer access, single-node 8-GPU). They share the `rcclDdaEnabled`
scaffolding and thresholds but never execute each other's transport.

## 1. Compile-time: one fatbin, per-arch kernels

`projects/rccl/src/device/generate.py` emits per-arch kernel variants baked into
the fatbin, selected by the HIP loader for the running device:

```python
if "gfx950" == gfx_name:
    return (["1", "2"], ["0"])       # MI350: unroll {1,2}, pipelining disabled
elif "gfx1250" == gfx_name:
    return (["8", "16", "32"], all)  # MI450: larger unrolls, Unroll 8 for FP8, 32 default
```

Device headers (`device/all_reduce.h`, `all_gather.h`, ...) guard code with
`__gfx950__` / `__gfx1250__` macros; LL128 conditions include gfx1250.

## 2. Runtime: setup + teardown gated by arch

`projects/rccl/src/fabric_init.cu`:

```cpp
bool ncclDdaUseFabricPath(ncclComm* comm) {
  return comm->MNNVL == 1 && IsArchMatch(comm->archName, "gfx1250");
}
```

`projects/rccl/src/init.cc` (init ~2747 and fini ~528 both branch on it):

```cpp
if (ncclDdaUseFabricPath(comm)) ncclDdaFabricCommInit(comm);  // MI450: fabric heap
else                            ncclDdaIpcCommInit(comm);     // MI350/MI300: xGMI IPC
```

So the fabric heap is only initialized on gfx1250 with a detected UALoE fabric
(`MNNVL==1`). MI350 never satisfies the predicate and initializes the IPC path.

## 3. Runtime: per-collective dispatch

`rcclDdaEnabled()` (moved to `projects/rccl/src/rccl_wrap.cc`, declared in
`rccl_common.h`) gates the DDA fast path per arch, now with a dedicated gfx1250
threshold parameter:

```cpp
bool rcclDdaEnabled(comm, totalBytes, gfx942Default, gfx950Default, gfx1250Default) {
  if (gfx1250)          threshold = gfx1250Default ? gfx1250Default : RCCL_DDA_THRESHOLD;
  else if (gfx942||gfx950) { if (nRanks < 8) return false; ... per-arch default ... }
  else return false;    // all other arches: DDA off
  return threshold > 0 && totalBytes <= threshold;
}
```

Each collective then branches on `ddaFabricArch = IsArchMatch(comm->archName,
"gfx1250")`: MI450 -> `*DdaFabric{,LL,LL128}`, MI350/MI300 -> `*DdaIpc`. This is
wired for AllReduce, AllGather, ReduceScatter, and AllToAll.

## Split at a glance

| | MI350 (gfx950) | MI450 (gfx1250) |
|---|---|---|
| Setup | `ncclDdaIpcCommInit` | `ncclDdaFabricCommInit` (fabric heap) |
| Gate | single-node, 8 ranks | `MNNVL==1 && gfx1250` |
| Memory sharing | VMM + POSIX-fd IPC over xGMI | VMM scratch + fabric handles + `FabricGpuBarrier` |
| Collective kernels | `*DdaIpc*` | `*DdaFabric*` (LL / LL128 / Simple) |
| Device codegen | unroll {1,2}, no pipelining | unroll {8,16,32}, LL128 default-on |
| DDA threshold | per-arch default (nRanks >= 8) | `RCCL_DDA_THRESHOLD` / gfx1250Default + per-tier |

## Changes vs the pre-sync snapshot (`0813205a1b9`)

- `rcclDdaEnabled()` relocated to `rccl_wrap.cc` and extended with a per-arch
  `gfx1250Default` threshold.
- AllToAll now has the full gfx1250 fabric tier ladder (LL / LL128 / VMM),
  matching AllReduce / ReduceScatter.
- Teardown (`Fini`) now mirrors init: `ncclDdaFabricCommFini` vs
  `ncclDdaIpcCommFini` by the same `ncclDdaUseFabricPath` predicate.

## Code references (`origin/develop`)

- `projects/rccl/src/fabric_init.cu` -- `ncclDdaUseFabricPath`,
  `ncclDdaFabricCommInit` (VMM scratch + `ncclFabricMemHandler` + `FabricGpuBarrier`).
- `projects/rccl/src/init.cc` -- init/fini DDA path selection.
- `projects/rccl/src/collectives.cc` -- per-collective `ddaFabricArch` dispatch.
- `projects/rccl/src/rccl_wrap.cc` -- `rcclDdaEnabled` (per-arch thresholds).
- `projects/rccl/src/device/generate.py` -- per-arch unroll/pipeline codegen.
