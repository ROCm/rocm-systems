# RCCL GIN Bitcode Feasibility

## Investigation status

The RCCL tree contains buildable GIN device implementations. The public device
API is in `src/include/nccl_device/gin.h`; its implementation bodies are
included by `src/include/nccl_device/impl/gin__funcs.h`. Backend implementations
are selected in `gin_device_common.h` and include the CPU-proxy backend in
`gin/proxy/gin_proxy.h`, as well as optional rocSHMEM GDA and Anvil SDMA
backends. This is not a declarations-only interface backed by a proprietary
object.

The device API has a C-shaped entry point already: `ncclGin_C_init`,
`ncclGinPut_v2`, `ncclGinSignal_v2`, `ncclGinWaitCounter`, and
`ncclGinWaitSignal`. The `rccl_gin_*` exports use these and create a
single-thread `ncclCoopAny` internally, so downstream JIT callers only pass
opaque pointers and scalar values. `ncclGinPut_v3` (strong signals, HOST_NUMA
segments) now also exists; the exports currently use `_v2`.

RCCL already builds device bitcode under `bindings/ir` (`EMIT_LLVM_IR`). The
`rccl_gin_*` thunks live there, and that build was extended to produce one
file per architecture, so no GIN-specific build exists.

## rocSHMEM relationship

RCCL has an optional `NCCL_GIN_ROCSHMEM_GDA_ENABLE` backend. Its implementation
uses rocSHMEM `QueuePair` operations when `ENABLE_ROCSHMEM_GIN` is enabled, but
the default RCCL GIN proxy backend is implemented directly in RCCL. RCCL's GIN
device bodies are therefore not built on rocSHMEM's bitcode.

The multi-architecture build follows `projects/rocshmem/cmake/DeviceBitcode.cmake`:
a loop over a list of architectures producing `librocshmem_device_<arch>.bc`
each, with feature suffixes kept for `--offload-arch` and stripped from file
names. RCCL's pipeline differs (`internalize,inline,globaldce` after a single
translation unit, rather than `opt -O3` over many sources).

RCCL's artifact is separate from rocSHMEM's because the consumer needs an RCCL
`ncclDevComm` and RCCL window ABI; a rocSHMEM bitcode library cannot safely be
substituted for it.

## Hardware status and architecture policy

Status as checked on 2026-10-26 (re-check before relying on it):

- ROCm issue [#11586](https://github.com/ROCm/rocm-systems/issues/11586) is
  **open**. It reports a reproducible `hipErrorIllegalAddress` for RCCL GIN on
  gfx942/ROCm 7.2.1 when the non-symmetric proxy window fallback is selected.
  This repository does not claim gfx942 is a validated target.
- PR [#11863](https://github.com/ROCm/rocm-systems/pull/11863) is **merged**
  (2026-09-24): window registration ordering before GIN registration.
- PR [#11895](https://github.com/ROCm/rocm-systems/pull/11895) ("build
  device-api with rocSHMEM GIN") was **closed without merging**.
- PR [#11260](https://github.com/ROCm/rocm-systems/pull/11260) (Anvil-SDMA
  AllToAll on MI455) was **closed without merging**.

Architecture selection is explicit through `BITCODE_GPU_ARCHS`. No
architecture has been validated on GIN hardware by this change; functional
validation remains the responsibility of the RCCL GIN/device-API CI
configuration.

The artifact is compiled with the same GIN backend configuration as the host
build. The default AMD build includes the proxy backend. With
`ENABLE_ROCSHMEM_GIN=ON`, `bindings/ir` adds the matching defines and rocSHMEM
include paths so the rocSHMEM GDA and Anvil SDMA device implementations are
included. Backend-specific symbol names are not needed: RCCL dispatches from the
live `ncclDevComm` backend mask/type at device runtime.
