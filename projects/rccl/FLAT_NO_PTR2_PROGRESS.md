# flat-no-ptr2: Progress Summary

## Branch / Project

Working branch: `lmeadows/flat-no-ptr2`
Source tree: `/work/lmeadows/fast-flat/projects/rccl` (also at `/work/lmeadows/flat-no-ptr2/projects/rccl`)

---

## Background

This project combines two earlier branches:

- **`fast-flat`**: passes `ncclShmem` and `ncclShmemPerWarp` as explicit arguments through the call chain rather than using `extern __shared__` globals.
- **`fast-lds`**: introduces `LDSPtr<T>` (`__attribute__((address_space(3))) T*`) to ensure the compiler emits LDS (local data share) instructions instead of flat loads/stores when accessing `__shared__` memory on AMD GCN GPUs.

The goal of `flat-no-ptr2` is to combine both: argument passing (from `fast-flat`) + `LDSPtr` address-space correctness (from `fast-lds`), with no `void*` shared-memory pointers.

---

## What Has Been Done

### 1. `rccl_ptr.h`

- Added `LDSPtr<T>` template alias: `__attribute__((address_space(3))) T*`
- Added `ncclShmemPerWarpPtr` typedef: `LDSPtr<uint8_t>`
- Added global-memory pointer typedefs (`u64_gptr`, `u32_gptr`, etc.) using `address_space(1)`

### 2. `common.h`

- Added file-scope `__device__ __shared__` declarations for `ncclShmem` (`ncclShmemData`) and `ncclShmemPerWarp` (`uint8_t[...]`), matching the `fast-lds` pattern.
- Changed `ncclScratchForWarp` to a template returning `LDSPtr<T>`:
  ```cpp
  template<typename T>
  __device__ __forceinline__
  LDSPtr<T> ncclScratchForWarp(ncclShmemPerWarpPtr ncclShmemPerWarp, int warp) {
    return LDSPtr<T>(ncclShmemPerWarp + warp * ncclShmemScratchWarpSize());
  }
  ```
- All function signatures that previously took `LDSPtr<ulong2> ncclShmemPerWarp` or `void*` updated to `ncclShmemPerWarpPtr`.

### 3. `common.cu`

- Removed per-kernel `__shared__` declarations (now at file scope in `common.h`).
- Added explicit `ncclShmemPerWarpPtr(ncclShmemPerWarp)` casts at kernel launch sites where the file-scope array decays to a plain pointer.

### 4. Collective headers (`all_reduce.h`, `all_gather.h`, `reduce_scatter.h`, `broadcast.h`, `reduce.h`, `sendrecv.h`, `alltoall_pivot.h`)

- All `LDSPtr<ulong2>` (and then `LDSPtr<uint8_t>`) parameters updated to `ncclShmemPerWarpPtr`.
- C-style casts removed from `ncclScratchForWarp` call sites; replaced with explicit template argument: e.g., `ncclScratchForWarp<ncclPatShmem>(ncclShmemPerWarp, 0)`.

### 5. `prims_simple.h`

- `ncclShmem` member: `__shared__ ncclShmemData& ncclShmem` → `LDSPtr<ncclShmemData> ncclShmem`
- `ncclShmemPerWarp` member: updated to `ncclShmemPerWarpPtr`.
- Constructor: initializes `ncclShmem(LDSPtr<ncclShmemData>(&shmem))`.
- All `ncclShmem.field` accesses changed to `ncclShmem->field`.
- `checkAbort(...)` call sites: `ncclShmem` → `*ncclShmem` (dereference to get reference).
- **Barriers**: `uint64_t* barriers` and `uint64_t* barriers_pat` members removed entirely. All
  `barrier_generic` call sites now materialize the pointer directly from `ncclShmem`:
  - `&ncclShmem->groups[group].barrier` (for `barriers`)
  - `&ncclShmem->barrier_pat` (for `barriers_pat`)
  This ensures the pointer is always AS3-typed (via `ncclShmem` which is `LDSPtr<ncclShmemData>`)
  with no class member involved, avoiding a ROCm 7.2 compiler bug (see below).

### 6. `prims_ll.h`

- `ncclShmem` member: → `LDSPtr<ncclShmemData>`.
- `ncclShmemPerWarp` member: → `ncclShmemPerWarpPtr`.
- Constructor updated accordingly; redundant delegating constructor removed.

### 7. `prims_ll128.h`

- `ncclShmem` member: → `LDSPtr<ncclShmemData>`.
- `ncclShmemPerWarp` member: → `ncclShmemPerWarpPtr`.
- **Barriers**: `uint64_t* barriers` member removed. `barrier_generic` call sites now use
  `&ncclShmem->groups[group].barrier` directly.

### 8. `network/unpack/unpack.h`

- `ncclNetDeviceUnpack` and `ncclNetDeviceUnpackInner`: `LDSPtr<ulong2>` parameter → `ncclShmemPerWarpPtr`.

### 9. `generate.py` + generated headers

- `generate.py`: updated template strings to use `ncclShmemPerWarpPtr` instead of `__attribute__((address_space(3))) ulong2*`; added `#include "rccl_ptr.h"` and `struct ncclShmemData;` forward declaration to `device_table_decl.h`.
- `build/release/hipify/gensrc/device_table_decl.h`: regenerated with correct types.
- `build/release/hipify/gensrc/device_table_impl.h`: regenerated with correct types.

### 10. `cmake/SplitDeviceCompile.cmake`

- No changes. The `amdgpu.max_num_named_barrier` workaround patch is not present; requires a
  ROCm build containing llvm/llvm-project PR #169851 (confirmed merged upstream).

---

## Compiler / Toolchain

- **Compiler**: AOMP_STANDALONE_23.0-1 (clang 23.0.0, commit `a4843e18e5`)
  — `AMD-Lightning-Internal/llvm-project`, at `/work/lmeadows/rocm/aomp`
- **ROCm runtime**: `/opt/rocm` (system install, used for HIP libraries and bundler probe)
- **Build time**: ~85 seconds (vs ~122 s with ROCm 7.0)
- **128-bit load/store intrinsics (DWORDX4)**: enabled ✓
- **Barrier bug**: fixed in this build (llvm/llvm-project PR #169851 present) ✓

---

## ROCm 7.2 Compiler Bug

**Symptom**: ROCm 7.2.1 assembler crashes with:
```
error: expression could not be evaluated
clang -cc1as: fatal error: error in backend: cannot evaluate equated symbol
  '_Z14ncclKernelMainI...RunWorkNop...PU3AS3h.num_named_barrier'
```

**Root cause**: The ROCm 7.2 compiler backend emits references to `amdgpu.max_num_named_barrier`
in `.set` expressions for `noinline` device functions that make indirect (function-pointer) calls.
However, the compiler forgot to emit the definition of `amdgpu.max_num_named_barrier` at the end
of the assembly file (alongside the existing `amdgpu.max_num_vgpr` etc.), so the assembler cannot
evaluate the expression.

**Upstream fix**: [llvm/llvm-project PR #169851](https://github.com/llvm/llvm-project/pull/169851)
— adds `amdgpu.max_num_named_barrier` to `EmitMCResourceMaximums`. Already merged upstream;
not yet in ROCm 7.2.1.

**Workaround**: The asm patch step in `SplitDeviceCompile.cmake` appends
`.set amdgpu.max_num_named_barrier, 0` to the generated `.s` file before assembly.

**Note**: ROCm 7.0 does not emit `amdgpu.max_num_named_barrier` at all and is unaffected.

---

## Performance Results

- **Baseline (flat loads)**: ~30µs higher latency across all message sizes vs production.
- **With LDS loads (this branch)**: latency at production levels (17–20µs range for small sizes).
  Incorporating LDS address space correctness eliminated the flat-memory penalty entirely.
- **Build time**: ~85s (vs ~122s with ROCm 7.0), due to 128-bit intrinsics and compiler improvements.
- **Outlook**: further tuning shows promise to reduce latency below previous production levels.

---

## Current Work: Eliminating All Flat Memory Operations

On AMD GCN GPUs, the compiler may emit **flat** load/store instructions when a pointer lacks an explicit address space qualifier. Flat instructions are slower than LDS-specific instructions because the hardware must resolve the address space at runtime.

### What "flat" means here

A flat operation on LDS memory arises when a pointer into `__shared__` memory is stored in a plain (`T*`) variable — i.e., without `address_space(3)`. The compiler loses the LDS qualification and falls back to flat instructions.

### Analysis of remaining flat-for-LDS operations

1. **`address_space(1)` usage** (`load128`, `store128` in `op128.h`): These explicitly use `u64_gptr` (global address space) for network buffer accesses. This is **intentional and correct** — not a concern.

2. **`barriers` / `barriers_pat`** — **FIXED**:
   - Were `uint64_t*` class members in `prims_simple.h` and `prims_ll128.h`.
   - Members removed; pointers now materialized directly from `ncclShmem` (AS3) at each
     `barrier_generic` call site, guaranteeing LDS instructions with no round-trip through
     generic address space.

3. **`storeShmem128`, `loadShmem128`, `loadShmemMisaligned128`** in `op128.h`:
   - These functions accept plain `uint64_t*` / `T*` parameters.
   - Call sites in `prims_ll128.h` pass `LDSPtr<uint64_t>` values (e.g. `shm8`), but the address-space qualifier is dropped on entry to the function.
   - This is a remaining source of flat instructions for LDS memory.
   - **Potential fix**: change the parameter types in these functions to `LDSPtr<uint64_t>` / `LDSPtr<T>`.

4. **`shmemCvtPtr`** in `op128.h`:
   - Strips `volatile` from `volatile uint64_t*`. By inspection, the pointers it operates on (`sendConnHeadPtr`, etc.) are for global memory (peer connection heads/tails). Not a flat-for-LDS issue.

### Summary of remaining work

- [ ] Change `storeShmem128`, `loadShmem128`, `loadShmemMisaligned128` in `op128.h` to take `LDSPtr<uint64_t>` / `LDSPtr<T>` parameters, and update all call sites in `prims_ll128.h` and `unpack.h`.
- [ ] Verify by inspection (or assembly review) that no other plain pointers into `__shared__` memory remain in the hot path.
- [x] ROCm 7.2 unit test failures (AlltoAll, Gather, Scatter, AlltoAllv, NonBlocking at 5/6/7 GPUs)
  — resolved: were artifacts of the old 7.2.1 container. With AOMP_STANDALONE_23.0-1 the only
  failure is the pre-existing `GroupCall.Different` on 7 GPUs (also present with ROCm 7.0).
