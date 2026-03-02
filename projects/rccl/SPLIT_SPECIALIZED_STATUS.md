# SPLIT_SPECIALIZED Pipeline — Current Status

**Date:** 2026-03-02
**Branch:** `lmeadows/device-linker`
**Build state:** Build directory was deleted, needs full reconfigure + rebuild.

## What This Pipeline Does

Replaces the custom `device_linker` tool with standard LLVM compilation:
1. Compile specialized kernels to bitcode (`.bc`) using `-fgpu-rdc`
2. Compile bitcode to AMDGPU assembly (`.s`)
3. Strip the kernel entry from the `.s`, save KD metadata to a sidecar (`.meta`)
4. Assemble stripped `.s` to device objects (`.o`)
5. Patch the dispatcher kernel's metadata with combined callee maximums
6. Link all device objects with `lld` to create a device `.so`
7. Bundle into `.hipfb` and embed into host objects

## Files Modified/Created

### New files (untracked):
- `cmake/SplitSpecializedCompile.cmake` — CMake module implementing the pipeline
- `cmake/scripts/patch_kernel_metadata.cmake` — Patches dispatcher KD + YAML with callee resource maximums
- `tools/split_specialized/strip_kernel.py` — Strips kernel from `.s`, extracts metadata sidecar

### Modified files (tracked):
- `CMakeLists.txt` — Integrates split pipeline, excludes `common.cu` from normal build, keeps `onerank.cu` in normal HIP build
- `src/device/common.h` — `ncclShmem` uses `__attribute__((weak)) __shared__`
- `src/device/common.cu` — Dispatcher kernel source

## Recent Changes (This Session)

### 1. Added `-DUSE_INDIRECT_FUNCTION_CALL` to specialized kernel compilation
**File:** `cmake/SplitSpecializedCompile.cmake`, line 135
**Why:** Without IFC, the compiler generates specialized device functions with varying ABIs:
- 19 LL128 kernels had `user_sgpr_count 8`, `dispatch_ptr 1`, `queue_ptr 1` etc. while the other 918 had `user_sgpr_count 2`
- LL128 kernels used dynamic LDS (`llvm.amdgcn.dynlds.offset.table`)
- Some kernels had `uses_dynamic_stack 1`, others `0`
With IFC, all device functions use the indirect function call ABI, matching what the dispatcher expects.

### 2. Fixed `.meta` sidecar extraction — reads actual KD values
**File:** `tools/split_specialized/strip_kernel.py`, function `extract_meta()`
**Was:** Reading `.set amdgpu.max_num_vgpr` etc. from bottom-of-file directives (wrong — these are indirect-call hints, not actual register usage)
**Now:** Reads `next_free_vgpr`, `accum_offset`, `next_free_sgpr`, `private_segment_fixed_size` from the `.amdhsa_kernel` block. Outputs `kd.*` prefixed fields.
**Why:** The old `.set` values reported max 128 VGPRs and 64 AGPRs. The actual KD values go up to 256 VGPRs and 96+ AGPRs. Under-allocating registers in the dispatcher = crash.

### 3. Fixed AGPR tracking — independent per-file max
**File:** `cmake/scripts/patch_kernel_metadata.cmake`
**Was:** Taking independent max of `next_free_vgpr` and `accum_offset` across all files, then computing `agpr = max_nfv - max_ao`. This gave AGPR=0 because the kernel with max nfv=256 also had ao=256.
**Now:** Computes per-file `agpr = next_free_vgpr - accum_offset`, tracks max VGPR and max AGPR independently. Combined `next_free_vgpr = max_vgpr + max_agpr`.
**Last verified values:** VGPR=256, AGPR=102, SGPR=100 → next_free_vgpr=358

### 4. Added `-gline-tables-only` to specialized kernel compilation
**File:** `cmake/SplitSpecializedCompile.cmake`, source→bc and bc→asm steps
**Why:** Produces `.loc`/`.file` directives in the `.s` (useful for reading assembly). However, the debug info can't survive into the final `.so` — `lld -shared` chokes on assembler-regenerated debug relocations ("unknown relocation (0)"). So `llvm-objcopy --strip-debug` is applied to specialized `.o` files before linking.

### 5. Added `strip_debug_sections()` to strip_kernel.py
**Why:** After kernel stripping, compiler-emitted `.debug_*` sections have dangling label references. Removing them lets the assembler regenerate `.debug_line` from inline `.loc` directives.

### 6. Dispatcher has `-g -O3` for full debug info
**File:** `cmake/SplitSpecializedCompile.cmake`, dispatcher bc and asm steps
**Why:** Allows `rocgdb` to resolve types and show line numbers for dispatcher code.

## Known Issues / What Needs Testing

### Smoke test still crashes
The smoke test (`test_smoke_allreduce`) was crashing with `HSA_STATUS_ERROR_EXCEPTION` before AND after these changes. The last run showed dispatch parameters `group_seg_size=73600, private_seg_size=1024` which don't match the assembly values (7936, 352) — this was likely due to stale build artifacts. **A full clean rebuild has not been tested yet.**

### Build command
The build directory was deleted. To rebuild:
```bash
cd /work/lmeadows/split-compile/projects/rccl
mkdir -p build/release && cd build/release
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx950 -DBUILD_TESTS=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-multiple-definition"
ninja -j$(nproc)
```
(Verify the cmake flags match your usual configuration.)

### After rebuilding, verify:
1. `grep "max_num_vgpr\|max_num_agpr\|max_num_sgpr" build/release/split_specialized/asm/common.patched.s`
   - Should show VGPR=256, AGPR=102, SGPR=100
2. `grep "group_segment_fixed_size" build/release/split_specialized/asm/common.patched.s | head -3`
   - Should show 7936
3. Run smoke test: `./build/release/test_smoke_allreduce`
4. Run unit tests: `./build/release/test/rccl-UnitTests`

## Architecture Summary

```
specialized_*.cpp  →  [hipcc -fgpu-rdc -DUSE_INDIRECT_FUNCTION_CALL]  →  .bc
                   →  [clang -gline-tables-only]  →  .s  (has kernel + devfunc)
                   →  [strip_kernel.py]  →  .stripped.s + .meta
                   →  [clang -c]  →  .o.tmp
                   →  [llvm-objcopy --strip-debug]  →  .o

common.cu          →  [hipcc -fgpu-rdc -DUSE_INDIRECT_FUNCTION_CALL -g]  →  .bc
                   →  [clang -g]  →  .s
                   →  [patch_kernel_metadata.cmake]  →  .patched.s  (KD + YAML patched)
                   →  [clang -c]  →  dispatcher.o

All .o files       →  [lld -r]  →  combined.o
                   →  [lld -shared]  →  combined.so
                   →  [clang-offload-bundler]  →  .hipfb
                   →  [hipcc --offload-host-only -fcuda-include-gpubinary]  →  host.o
                   →  linked into librccl.so

onerank.cu         →  normal HIP build path (not part of split pipeline)
```

## Pending Tasks
- [ ] Full clean rebuild + smoke test verification
- [ ] Clean up dead DEVICE_LINKER code paths if everything works
