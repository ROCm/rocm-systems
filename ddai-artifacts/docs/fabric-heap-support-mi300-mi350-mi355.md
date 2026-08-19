# Is the fabric heap supported on MI300, MI350, MI355?

Context: this branch's GIN-SDMA AllToAll design uses the VMM (cuMem) path with
POSIX-fd IPC handles, not the "fabric heap." This note records why the fabric
heap is not an option on these parts.

## Question

Is the fabric heap supported on MI300, MI350, and MI355?

("Fabric heap" here = fabric-handle-based VMM memory sharing: RCCL's DDA fabric
path via `ncclFabricMemHandler` + `FabricGpuBarrier`, and the
`CU_MEM_HANDLE_TYPE_FABRIC` / `hipMemFabricHandle_t` allocator handle type.)

## Answer

**No — the fabric heap is not supported on MI300, MI350, or MI355.** It is gated
by two independent requirements, and both fail on these parts.

### 1. Hardware — no UALink/UALoE memory fabric

The fabric-heap path is gated at runtime on a **UALoE/UALink scale-up fabric**,
detected via amd-smi (`fabricInfo.fabricSupported`, logged as "UALoE-enabled
(aka MNNVL)") and `/sys/class/drm/<card>/device/ualink/`
(`link_type: "UALoE" | "UALLink"`).

- **MI300 (gfx942 / CDNA3)** and **MI350X/MI355X (gfx950 / CDNA4)** use **AMD
  Infinity Fabric / xGMI** for scale-up — a direct 7-link full mesh across 8
  GPUs (153.6 GB/s bidirectional per link). They **do not implement UALink** for
  GPU-to-GPU scale-up.
- True UALoE (UALink-over-Ethernet) memory fabric is a **MI400 / MI455X "Helios"
  (CDNA5)** capability (single-hop 72-GPU UALoE domain).
- Marketing caveat: some AMD slides tag "MI350X ... UALoE72", but the platform
  briefs and independent analysis confirm MI350/MI355 scale-up is the 8-GPU xGMI
  mesh; the memory-fabric heap does not light up there.

Consequence in RCCL: `fabricInfo.fabricSupported` returns false, so
`ncclDdaFabricCommInit` is never selected, and the allocator's
`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` check fails and falls back to
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`.

### 2. Software — ROCm does not expose fabric handles yet

`HIP_FABRIC_API` (in `projects/rccl/src/CMakeLists.txt`) is only enabled if the
ROCm build provides both `hipMemFabricHandle_t` and
`hipMemImportFromShareableHandle`. On shipping ROCm those are absent: adding
`CU_MEM_HANDLE_TYPE_FABRIC` / `CUmemFabricHandle` / HIP fabric handles is an
**open feature request** (ROCm/rocm-systems #2170, Dec 2025). HIP VMM
(`hipMemCreate` / `hipMemMap`) exists, but shareable handles are POSIX-fd only.
RCCL keeps the hook alive but disabled: see the note in
`projects/rccl/src/include/p2p.h` ("Preserve AMD's HIP_FABRIC_API hook in case
it ever gets enabled").

### What runs instead on these GPUs

The equivalent scale-up sharing is **VMM + POSIX-fd IPC over xGMI** — exactly
what the GIN-SDMA A2A design in this branch uses (`cuMemCreate` / `cuMemMap`
with `NCCL_CUMEM_ENABLE=1`, shared intra-node via POSIX-fd handles). So the GIN-
SDMA design uses VMM rather than a fabric heap in part because the fabric heap
is simply not available on MI300 / MI350 / MI355.

| Part | Scale-up | Fabric heap? | Sharing mechanism used |
|---|---|---|---|
| MI300 (gfx942) | Infinity Fabric / xGMI mesh | No | VMM + POSIX-fd IPC |
| MI350X / MI355X (gfx950) | Infinity Fabric / xGMI mesh | No | VMM + POSIX-fd IPC |
| MI400 / MI455X (Helios, CDNA5) | UALoE fabric | Yes (target) | Fabric handles + UALoE |

## Code references (this branch)

- `projects/rccl/src/allocator.cc` — cuMem allocator; `FABRIC` handle falls back
  to `POSIX_FILE_DESCRIPTOR` when `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`
  is false.
- `projects/rccl/src/include/alloc.h` — `cuMemCreate` / `cuMemAddressReserve` /
  `cuMemMap` VMM allocation with an exportable `requestedHandleTypes`.
- `projects/rccl/src/init.cc` — UALoE/MNNVL fabric detection
  (`fabricInfo.fabricSupported`).
- `projects/rccl/src/misc/alt_rsmi.cc` — reads
  `/sys/class/drm/<card>/device/ualink/` (`UALoE` / `UALLink`).
- `projects/rccl/src/fabric_init.cu` — DDA fabric path
  (`ncclFabricMemHandler` + `FabricGpuBarrier`).
- `projects/rccl/src/CMakeLists.txt` — `HIP_FABRIC_API` compile-time detection.
- `projects/rccl/src/include/p2p.h` — preserved-but-disabled `HIP_FABRIC_API` hook.

## References

- AMD Instinct MI355X Platform brief (Infinity Fabric mesh, 7x 153.6 GB/s).
- AMD Instinct MI350 Series product page (CDNA4, Infinity Fabric scale-up).
- SemiAnalysis, "AMD Advancing AI: MI350X and MI400 UALoE72, MI500 UAL256."
- ROCm CDNA5 / Helios blog (UALoE scale-up fabric on MI455X).
- ROCm/rocm-systems issue #2170 — feature request for HIP fabric handles.
- ROCm HIP virtual memory management docs (`hipMemCreate` / `hipMemMap`).
