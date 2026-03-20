# LDS Pointer Refactoring for Split Device Compilation

## Problem

RCCL's split-device-compile pipeline compiles device functions in separate
translation units and links them with `lld -shared`. Device functions receive
`ncclShmem` as a reference parameter (`ncclShmemData&`) and `ncclShmemPerWarp`
as `void*`. Because these are generic/flat pointers in the ABI, the compiler
emits `flat_load` / `flat_store` instructions for all shared memory accesses.
On CDNA architectures, `flat_` instructions to LDS carry ~40µs additional
latency compared to `ds_read` / `ds_write` instructions, measurable on
AllReduce benchmarks.

The production (non-split) build uses `-fgpu-rdc` with `extern __shared__`
globals, so the compiler knows the address space and emits `ds_` instructions.

## Solution

Explicitly annotate all shared memory pointers with AMDGCN address space 3
using a `LDSPtr<T>` type alias:

```cpp
template<typename T>
using LDSPtr = __attribute__((address_space(3))) T*;
```

This tells the compiler to emit `ds_` instructions regardless of how the
pointer was obtained or which TU the code lives in.

## What Changed

### New RCCL-specific files

- **`src/device/rccl_ptr.h`** — Defines `LDSPtr<T>` (conditionally, so it
  resolves to plain `T*` on the host side), `ncclShmemPerWarpPtr` (alias for
  `LDSPtr<uint8_t>`), and the `shmemCvtPtr` macro.

### Modified RCCL-specific files

- **`src/device/common.h`**:
  - Global `__device__ __shared__` declarations for `ncclShmem` and
    `ncclShmemPerWarp` (these get merged by `lld -shared` across TUs).
  - Templated `ncclScratchForWarp<T>()` returning `LDSPtr<T>`.
  - Three `copyToShmem16` overloads for global→LDS, LDS→LDS, and LDS→global.
  - `ncclKernelMain` declared `__forceinline__` — inlining is safe because
    all LDS accesses in the kernel entry points go through `LDSPtr`, avoiding
    `ds_`/`flat_` aliasing to the same memory.
  - `kernargPtr` field in `ncclShmemData` is non-volatile.

- **`src/device/common.cu`**:
  - Kernel entry points pass `LDSPtr<ncclShmemData>(&ncclShmem)` and
    `ncclShmemPerWarpPtr(ncclShmemPerWarp)` to `ncclKernelMain`.
  - `STORE_KERNARG_PTR` macro accesses `kernargPtr` through an `LDSPtr` cast
    (not through the global `__shared__` variable directly) to avoid emitting
    `flat_store` with `src_shared_base` aperture conversion.

- **`src/device/op128.h`** — `loadShmem128`, `storeShmem128`, and
  `loadShmemMisaligned128` accept `LDSPtr` parameters directly.

- **`src/device/generate.py`** — Generated function prototypes and
  `ncclDevFuncPtr_t` typedef use `LDSPtr<ncclShmemData>` and
  `ncclShmemPerWarpPtr`.

### Build-time sed transform (no source changes to NCCL-upstream files)

- **`cmake/scripts/lds_pointer_transform.sh`** — A post-hipify sed script
  that transforms NCCL-style code into LDS-pointer-style code at build time.
  Applied to all hipified device headers. Transformations include:
  - `ncclShmemData& ncclShmem` → `LDSPtr<ncclShmemData> ncclShmem`
  - `ncclShmem.field` → `ncclShmem->field`
  - `void* ncclShmemPerWarp` → `ncclShmemPerWarpPtr ncclShmemPerWarp`
  - Shared memory pointer declarations to `LDSPtr` types
  - `ncclScratchForWarp` call sites to templated `LDSPtr<T>` calls
  - Inlining guards for `runRing`/`runTree` on gfx942/gfx950 to match
    production's noinline wrapper pattern

  This keeps collective headers (`all_reduce.h`, `reduce.h`, `broadcast.h`,
  etc.), primitives files (`prims_simple.h`, `prims_ll.h`, `prims_ll128.h`),
  `sendrecv.h`, `unpack.h`, and `primitives.h` identical to NCCL upstream.

### Split device compilation pipeline

- **`cmake/SplitDeviceCompile.cmake`**:
  - Uses `-gline-tables-only` (matching production) instead of `-g`.
  - Assembly-level patching of kernel descriptors via `sed` to provision
    known callee maximums: 128 VGPRs, 64 AGPRs, 102 SGPRs, flat scratch,
    dynamic stack, indirect call, and recursion flags.

## Key Design Decisions

### Why `LDSPtr<T>` instead of `extern __shared__` globals

The split-compile pipeline cannot use `extern __shared__` in device function
TUs because the compiler requires `__shared__` declarations to be attached
to a kernel (without a compiler patch that is not permitted upstream).
Explicit address space annotation via `LDSPtr<T>` achieves the same codegen
without this requirement.

### Why `ncclKernelMain` must be `__forceinline__`

Production inlines `ncclKernelMain` to avoid the overhead of an extra
function call on every kernel launch. With LDS pointers, inlining is safe
as long as the kernel entry point does not access the global `__shared__`
variables via flat instructions. The `STORE_KERNARG_PTR` macro was the one
flat-to-LDS access point and was fixed by routing it through `LDSPtr`.

### Why `STORE_KERNARG_PTR` uses `LDSPtr` and not the global directly

When `ncclKernelMain` is inlined, the compiler can see the global
`__device__ __shared__` variable. If the kernel entry point accesses it
directly (e.g., `ncclShmem.kernargPtr = ...`), the compiler emits a
`flat_store` with `s_mov_b64 s[0:1], src_shared_base` to convert the LDS
offset to a flat address. This flat store aliases with `ds_` accesses to the
same LDS memory from the inlined code, causing hardware memory ordering
violations that manifest as `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
crashes. Routing through `LDSPtr<ncclShmemData>(&ncclShmem)->kernargPtr`
keeps everything on the `ds_` path.

### Why `runRing`/`runTree` are noinline on gfx942/gfx950

Production uses `USE_INDIRECT_FUNCTION_CALL` with noinline device function
wrappers that call noinline `runRing`/`runTree` implementations. This keeps
the `ncclDevFunc_*` entries as thin wrappers, avoiding code bloat and
register pressure from aggressive inlining. The sed script adds
`!defined(__gfx942__) && !defined(__gfx950__)` guards to match this pattern.

### Why build-time sed instead of source changes

The RCCL team maintains merge compatibility with NVIDIA's NCCL releases.
Modifying collective headers and primitives files in-place would create
extensive merge conflicts. The sed script approach (mirroring the existing
`add_unroll.sh` pattern) keeps NCCL-upstream files untouched while
applying the necessary transformations during the build.

## NCCL Merge Impact

- **Zero source changes** to: collective headers, primitives files,
  `sendrecv.h`, `unpack.h`, `primitives.h`
- **RCCL-only changes** to: `rccl_ptr.h` (new), `common.cu`, `common.h`,
  `op128.h`, `generate.py`
- **Build infrastructure** additions: `lds_pointer_transform.sh`,
  updates to `CMakeLists.txt` and `SplitDeviceCompile.cmake`
