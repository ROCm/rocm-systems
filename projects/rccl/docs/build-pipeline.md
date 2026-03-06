# RCCL Build Pipeline

## Build Invocation

From the project root (`projects/rccl`):

```bash
rm -rf build; bash do-cmake; cd build/release; ninja
```

`do-cmake` invokes CMake with key flags:

- `SPLIT_SPECIALIZED=ON` -- enables the split compilation pipeline
- `ENABLE_IFC=ON` -- enables indirect function calls (device-side function pointer dispatch)
- `BUILD_LOCAL_GPU_TARGET_ONLY=ON` -- compiles only for the locally detected GPU arch

---

## Phase 1: CMake Configure

### 1a. Hipify

All source files are run through `hipify-perl` to convert CUDA idioms to HIP.
Output goes to `build/release/hipify/src/`. Device headers in `src/device/` are
further patched by:

- `cmake/scripts/add_unroll.sh` -- injects `COLL_UNROLL`, `USE_ACC`, `Pipeline`
  template parameters into collective headers
- `cmake/scripts/add_faults.sh` -- optionally injects fault-injection calls
  (when `FAULT_INJECTION=ON`)

### 1b. Code Generation (at configure time)

Three Python scripts run during `cmake` configure:

1. **`src/device/generate.py`** -- Always runs. Generates into `hipify/gensrc/`:
   - `device_table.h` -- Declares all `ncclDevFunc_*` device functions and
     defines `ncclDevFuncTable_1/2/4[]` (function pointer arrays indexed by
     funcId, one table per unroll factor). When IFC is off, also generates
     `Caller<N>` binary-search dispatch templates and `NCCL_CALL_FUNCTIONS_*`
     macros.
   - `host_table.cpp` -- `ncclDevFuncNameToId` map: packs (coll, algo, proto,
     redop, ty, acc, pipeline) into a 64-bit key, maps to an index into the
     function pointer table.
   - Per-collective `.cpp` files (e.g., `all_reduce_sum_f16.cpp`) -- Each file
     defines `ncclDevFunc_*` instances via `DEFINE_ncclDevFunc`. **When
     `SPLIT_SPECIALIZED=ON`, these `.cpp` files are unused** -- only
     `device_table.h` and `host_table.cpp` are compiled.

2. **`src/device/generate_specialized.py`** (when `SPLIT_SPECIALIZED=ON`) --
   Generates one `specialized_*.cpp` per (coll, algo, proto, redop, ty, acc,
   pipeline, unroll) combination, each containing a `DEFINE_ncclDevFunc` plus a
   wrapper `ncclDevKernel_*` (needed to satisfy the compiler's LDS check; see
   "Why Split-Specialized Exists" below). Also generates
   `specialized_kernel_selector.h`, `specialized_kernel_list.h`, and a cmake
   fragment listing the source files.

3. **`src/device/symmetric/generate.py`** (when `GENERATE_SYM_KERNELS=ON`) --
   Generates symmetric memory kernels.

---

## Why Split-Specialized Exists

The production build compiles all device functions together with `-fgpu-rdc` and
links them via the HIP device linker in a single whole-program link step. This
is unbearably slow.

The split-specialized pipeline exists to avoid that whole-program device link.
Each device function is compiled individually and linked with plain `lld`. To
make this work, two compiler limitations must be circumvented:

1. **Producing relocatable objects requires the bitcode path.** Compiling
   through the HIP driver end-to-end produces dynamic shared objects, not the
   relocatable ELF objects that `lld -shared` needs to link later. The
   workaround is a two-step compilation: first use `amdclang++` with
   `-fgpu-rdc --offload-device-only -emit-llvm` to produce bitcode, then feed
   that bitcode to bare `clang -x ir -target amdgcn-amd-amdhsa` to produce
   assembly, which is then assembled into relocatable objects. This bypasses the
   HIP driver's link machinery and avoids the slow whole-program `-fgpu-rdc`
   device link that the production build suffers from.

2. **The compiler rejects LDS references in device functions without a kernel.**
   RCCL device functions use LDS (shared memory), but the compiler considers LDS
   allocation to be the kernel's responsibility and errors on LDS references in
   a translation unit with no kernel. The ideal fix would be a compiler flag to
   disable this check; absent that, each specialized TU includes a fake wrapper
   kernel (`ncclDevKernel_*`) whose sole purpose is to make the compiler accept
   the LDS references and generate correct code. The wrapper kernel is then
   stripped from the assembly after compilation.

3. **Debug line numbers require care when stripping the kernel.** The compiled
   assembly contains `.debug_*` sections with address ranges and frame info
   covering both the kernel and device function code. Simply deleting the kernel
   code leaves dangling references in these sections. The solution:
   `strip_kernel.py` removes all `.debug_*` sections but preserves `.file`
   directives from the kernel block (needed by `.loc` directives in the kept
   device function code). The assembler regenerates correct `.debug_line` from
   the surviving inline `.loc` directives. Specialized TUs are compiled with
   `-gline-tables-only` (minimal debug info sufficient for line numbers) while
   the dispatcher uses full `-g`.

---

## Phase 2: Split-Specialized Device Compilation Pipeline

This is the core of the build for device code, implemented in
`cmake/SplitSpecializedCompile.cmake`. Everything below repeats **per GPU
architecture**.

### 2a. Specialized Kernels: Source to Bitcode to Assembly

Each `specialized_*.cpp` is compiled through two steps:

```
specialized_*.cpp  -->  .bc  -->  .s
```

- **Step 1 (source to bitcode):**
  `amdclang++ -x hip -fgpu-rdc --offload-device-only --offload-arch=<arch> -emit-llvm -c -O3 -DSPECIALIZED_KERNEL=1 -DUSE_INDIRECT_FUNCTION_CALL`
  This uses the HIP frontend to handle HIP language semantics and `-fgpu-rdc`
  linkage, stopping at LLVM IR.
- **Step 2 (bitcode to assembly):**
  `clang -x ir -target amdgcn-amd-amdhsa -mcpu=<arch> -O3 -S`
  This feeds the bitcode directly to the AMDGPU backend, bypassing the HIP
  driver and its whole-program link expectations. The result is clean AMDGPU
  assembly suitable for text-level manipulation.

### 2b. The Trick: Strip Kernel, Keep Device Function

Each specialized `.s` file contains two things:

- `ncclDevKernel_*` -- a fake GPU kernel entry point (wrapping the device
  function)
- `ncclDevFunc_*` -- the actual device function implementing the collective

The wrapper kernel exists to satisfy the compiler's requirement that a kernel be
present when LDS is referenced. It also causes the compiler to generate the
device function's code with correct register allocation and LDS layout, as if
called from a real kernel. The wrapper is discarded after compilation.

`tools/split_specialized/strip_kernel.py` processes each `.s` file:

1. **Strips** the `ncclDevKernel_*` function block, its `.amdhsa_kernel`
   descriptor, `.set` directives, and `.AMDGPU.csdata` block
2. **Extracts** a `.meta` sidecar file containing the kernel's hardware resource
   requirements from the `.amdhsa_kernel` block:
   - `kd.next_free_vgpr` (total VGPRs + AGPRs)
   - `kd.accum_offset` (number of VGPRs; AGPRs = next_free_vgpr - accum_offset)
   - `kd.next_free_sgpr`
   - `kd.private_segment_fixed_size`
3. **Removes** the `.amdgpu_metadata` YAML section (no kernels remain)
4. **Removes** `.debug_*` sections (dangling references to stripped kernel code)

Output: `*.stripped.s` (device function only) + `*.meta` (resource requirements)

### 2c. Assembly to Object

Each stripped `.s` is assembled:

```
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=<arch> -c
```

### 2d. Dispatcher Kernel: Compile and Patch

The dispatcher source is `common.cu` (hipified to `common.cu.cpp`), which
defines the actual kernel entry points: `ncclDevKernel_Generic_1`,
`ncclDevKernel_Generic_2`, `ncclDevKernel_Generic_4`.

It goes through the same bc-to-asm pipeline, but instead of stripping, its
assembly is **patched** by `cmake/scripts/patch_kernel_metadata.cmake`:

1. Reads all `.meta` sidecar files and computes **per-field maximums** across
   all callees (VGPR, AGPR, SGPR, private segment size)
2. Patches `.set amdgpu.max_num_vgpr`, `.set amdgpu.max_num_agpr`,
   `.set amdgpu.max_num_sgpr` in the assembly to the callee maximums
3. Resolves forward references to those symbols in `max()` expressions
4. Patches `.amdhsa_private_segment_fixed_size` for IFC kernels (those with
   `.amdhsa_uses_dynamic_stack 1`)
5. Patches YAML `.note` metadata entries (`.vgpr_count`, `.agpr_count`,
   `.sgpr_count`, `.private_segment_fixed_size`) for IFC kernel entries

This is necessary because the dispatcher calls device functions indirectly (via
function pointer), so the compiler cannot see callee register usage. Without
patching, the kernel descriptor would undercount resources and the hardware would
not allocate enough registers, causing silent corruption.

### 2e. Device Link

Per architecture, all device objects (stripped specialized + patched dispatcher)
are linked:

```
lld -shared -o combined.<arch>.so @response_file
```

### 2f. Fat Binary Bundle

All per-arch `combined.<arch>.so` files are bundled into a single
`combined.hipfb`:

```
clang-offload-bundler --type=bc \
  --targets=host-x86_64-...,hipv4-amdgcn-...-<arch1>,... \
  --input=/dev/null --input=combined.<arch1>.so ... \
  --output=combined.hipfb
```

---

## Phase 3: Host Compilation and Final Link

### 3a. Host Stubs

The dispatcher source (`common.cu.cpp`) is compiled again, this time for the
**host only**, with the fat binary embedded:

```
amdclang++ -x hip --offload-host-only \
  --offload-arch=<arch1> --offload-arch=<arch2> ... \
  -Xclang -fcuda-include-gpubinary -Xclang combined.hipfb \
  -c -O3 -fPIC -o common.host.o common.cu.cpp
```

This produces a host object containing HIP runtime registration code that embeds
the `combined.hipfb` blob.

### 3b. Host Library Sources

All non-device source files (transport, graph, scheduler, misc, etc.) are
compiled normally by `amdclang++` as a regular shared library. The
`SPLIT_SPECIALIZED` path deliberately **excludes** `-fgpu-rdc` and `--hip-link`
from the main library target.

The split-specialized fat object (`combined.fat.o`) is passed as a link option
to the `rccl` target:

```cmake
target_link_options(rccl PRIVATE ${SPLIT_SPECIALIZED_FAT_OBJ})
```

### 3c. Final Output

The linker produces `librccl.so` containing:

- All host-side RCCL code (init, enqueue, transport, graph, etc.)
- Embedded device fat binary (all GPU architectures) via the host stub

---

## The Dispatch Mechanism

### Host Side

1. `ncclDevFuncId(coll, algo, proto, redop, ty, acc, pipeline)` packs the
   parameters into a 64-bit key and looks up `ncclDevFuncNameToId` to get a
   `funcId` (index into the device function pointer table)
2. The kernel to launch is selected by unroll factor:
   `ncclDevKernel_Generic_1`, `_2`, or `_4`
3. The `funcId` is passed to the device via shared memory
   (`ncclShmem.funcId`)

### Device Side

The generic kernels (`ncclDevKernel_Generic_1/2/4` in `src/device/common.cu`)
call `ncclKernelMain<>` (`src/device/common.h`), which loops over work batches.
For each batch:

```cpp
// common.h lines 715-728
#if defined(USE_INDIRECT_FUNCTION_CALL)
  ncclDevFuncTable_{1,2,4}[ncclShmem.funcId]();   // IFC: indirect call via fn pointer
#else
  NCCL_CALL_FUNCTIONS_{1,2,4}(ncclShmem.funcId);  // non-IFC: binary-search dispatch
#endif
```

- **IFC mode** (used with `SPLIT_SPECIALIZED`): A true indirect function call
  through `ncclDevFuncTable_N[funcId]()`. The function pointer table is a
  `__device__` array of `ncclDevFuncPtr_t` (i.e., `void(*)()`) generated in
  `device_table.h`.
- **Non-IFC mode**: A compile-time binary search tree (`Caller<f,l>` templates)
  that narrows down to the correct `ncclDevFuncTable_N[constant]()` call,
  avoiding indirect calls but producing a large if/else chain.

### Why the Wrapper Kernel Trick Works

Each specialized source file compiles **both** `ncclDevKernel_*` (a `__global__`
kernel) and `ncclDevFunc_*` (a `__device__` function). The kernel exists
primarily to satisfy the compiler's refusal to compile device functions that
reference LDS without a kernel present in the translation unit. As a side
effect, the compiler also generates a kernel descriptor with accurate resource
requirements (VGPR/AGPR/SGPR counts, LDS size, private segment) that reflect
the device function's actual usage.

After compilation, `strip_kernel.py` discards the kernel entry point from the
assembly, keeping only the device function's machine code. The resource metadata
extracted from the discarded kernel descriptor is propagated to the dispatcher
kernel via `patch_kernel_metadata.cmake`, ensuring the dispatcher allocates
enough hardware resources for any callee it might invoke.

---

## Kernel Descriptor Audit (gfx950, custom build)

Both `ncclDevKernel_Generic_1` and `ncclDevKernel_Generic_2` are identical:

| Field | Value |
|-------|-------|
| `.group_segment_fixed_size` (LDS) | 70432 (= 4768 `ncclShmemData` + 65664 `ncclShmemPerWarp`) |
| `.private_segment_fixed_size` | 1112 |
| `.vgpr_count` | 200 (= 136 VGPRs + 64 AGPRs) |
| `.agpr_count` | 64 |
| `.sgpr_count` | 102 |
| `.max_flat_workgroup_size` | 512 |
| `.uses_dynamic_stack` | true |
| `.wavefront_size` | 64 |

Patched `.set` values (callee maximums):

| Symbol | Value |
|--------|-------|
| `num_vgpr` | max(91, 136) = 136 |
| `num_agpr` | max(0, 64) = 64 |
| `numbered_sgpr` | max(84, 102) = 102 |
| `has_dyn_sized_stack` | 1 |
| `has_indirect_call` | 1 |

`ncclDevKernel_Generic_4` is minimal (28 VGPRs, 0 AGPRs, 32 SGPRs,
`uses_dynamic_stack: false`) because no specialized functions with unroll=4
were generated for this gfx950 build.

LDS breakdown: `NCCL_MAX_NTHREADS=512`, `WARP_SIZE=64` (8 warps),
`ncclShmemScratchWarpSize()=8208`, so `ncclShmemPerWarp = 8208 * 8 = 65664`.

---

## Key Files Reference

| File | Role |
|------|------|
| `do-cmake` | CMake invocation script |
| `CMakeLists.txt` | Root project, GPU targets, IFC check |
| `src/CMakeLists.txt` | Library build: hipify, codegen, split-specialized wiring |
| `cmake/SplitSpecializedCompile.cmake` | Split-specialized pipeline implementation |
| `tools/split_specialized/strip_kernel.py` | Strip kernel from .s, extract .meta |
| `cmake/scripts/patch_kernel_metadata.cmake` | Patch dispatcher .s with callee resource maximums |
| `src/device/generate.py` | Generate device_table.h, host_table.cpp, collective .cpp |
| `src/device/generate_specialized.py` | Generate specialized kernel sources |
| `src/device/common.cu` | Dispatcher kernel definitions |
| `src/device/common.h` | `ncclKernelMain` dispatch loop |
