# rccl_ep

Intranode expert-parallel dispatch and combine for RCCL on AMD GPUs: the
scale-up leg of an MoE all-to-all, where every peer is reachable by load and
store.

Dispatch is a push model: each rank writes tokens straight into its
peers' symmetric memory, so a receiver reconstructs the required receive
ordering -- by source rank, then by source token index -- by concatenating
per-source regions, with no sort and no host round-trip. Combine accumulates in
strict top-k order in fp32, which is required rather than incidental: float
addition is not associative, so any other order fails the bit-exact bar.

Beyond plain dispatch and combine the module offers an output layout already
grouped by expert, so an expert GEMM needs no permute; FP8; and cached replay,
which reuses a routing plan with reproducible row assignment.

Scale-up only. No scale-out path, no low-latency path. The path needs only
load/store access to peer memory -- no GIN backend and no rocSHMEM.

## Layout

    device/        header-only kernels: dispatch, combine, expand, wave primitives
    include/       window layout and mixed-architecture guard
    python/        C ABI translation unit and the ctypes package
    CMakeLists.txt standalone build

The Python surface is driven from HIP through a flat C ABI rather than a torch
extension, which keeps the binding independent of the torch version. No RCCL
library target builds these sources; the unit tests include the headers
directly, so the module is otherwise inert until built here.

## Build

Requires ROCm 7.13 or newer. The path uses RCCL symmetric memory, which needs
cuMem, which is gated on that version.

    cmake -S . -B build \
      -DRCCL_EP_RCCL_ROOT=/workspace/rccl-nogin \
      -DRCCL_EP_GPU_TARGETS=gfx950
    cmake --build build -j
    cmake --install build --prefix .

`RCCL_EP_RCCL_ROOT` must point at an RCCL built **without** `--rocshmem-gin`. A
GIN build leaves rocSHMEM device symbols for the consuming executable to resolve
at device-link time, which `python3` cannot do.

The install step drops `librccl_ep.so` beside `python/rccl_ep/__init__.py`,
which is where the package looks for it by default.

## Runtime contract

Four things are not discoverable and must be set by the caller:

| Variable | Why |
| --- | --- |
| `RCCL_EP_LIBRCCL` | Absolute path to the EP-capable `librccl.so.1`. Loaded `RTLD_GLOBAL` before the extension so that it and torch resolve to one library rather than two copies contending for one SONAME. Unset, the loader silently binds to whichever `librccl` arrived first -- usually torch's, which predates the device API this path needs. |
| `NCCL_CUMEM_ENABLE=1` | Short-circuits RCCL's auto-detect, which is a ROCm version gate that torch's bundled HIP fails. |
| `HSA_NO_SCRATCH_RECLAIM=1` | Required by the kernels' scratch usage. |
| `RCCL_EP_LIB` | Optional. Overrides the co-located `librccl_ep.so`; unnecessary after `cmake --install`. |

`NCCL_DMABUF_ENABLE` is deliberately **not** set: torch's older RCCL is live in
the same process and segfaults on it.

One more constraint applies to any process that also imports torch. torch ships
its own `libamdhip64`, `libhsa-runtime64` and `libamd_comgr`; symmetric memory
needs a cuMem-capable HIP that no published wheel provides, so all three must
resolve to the system ROCm, with `/opt/rocm/llvm/lib` on the search path.
Point all three at the system copies before importing torch and this package
together; the process must end up with exactly one HIP runtime loaded.
