# Device Linker Context

## What is the Device Linker?

The device linker is a build optimization for RCCL that achieves **~10x faster builds** by:
1. Compiling ~2500 specialized kernels in parallel (each as a separate .o file)
2. Merging all device code into a single ELF with function pointer dispatch

Instead of the production build's slow template instantiation, we pre-compile specialized kernels and dispatch via function pointer tables at runtime.

---

## Documentation

| Document | Contents |
|----------|----------|
| [Device_Linker_Architecture.md](Device_Linker_Architecture.md) | ELF structure, LDS layout, build pipeline, dispatch mechanism, HIP compilation |
| [Device_Linker_HowTo.md](Device_Linker_HowTo.md) | Build commands, testing, debugging, troubleshooting, common mistakes |

---

## Current Status

| Component | Status |
|-----------|--------|
| Build | **WORKING** |
| Kernel Load | **WORKING** |
| Kernel Execute | **RUNS** but output data not modified |
| DWARF Debug Info | **SUBTLE ERRORS** in merged image |

---

## Quick Build

```bash
cd /work2/lmeadows/rocm-systems/projects/rccl
rm -rf build
./install.sh -l --device-linker
```

To rebuild device_linker tool only:
```bash
cd tools/device_linker
./build.sh
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `tools/device_linker/device_linker.cpp` | **Main tool** - parses inputs, merges ELFs, builds function tables |
| `tools/device_linker/merged_minimal.hip` | Dispatcher kernel with `ncclDevFuncTable_{1,2,4}` |
| `tools/device_linker/build.sh` | Builds the device_linker binary |
| `src/device/common.cu` | Generic dispatcher kernels |
| `src/device/common.h` | Function table declarations, dispatch macros |
| `src/device/generate_specialized.py` | Generates specialized kernel sources |
| `src/enqueue.cc` | Host-side kernel launch (has `DEVICE_LINKER` paths) |
| `CMakeLists.txt` | Device linker pipeline integration (~line 1300+) |

---

## Critical Constraint: Compilation Flags

**Structure layout mismatches cause crashes/hangs.** All compilation units must have identical flags:

| Flag | Affects Layout |
|------|----------------|
| `ENABLE_FAULT_INJECTION` | **Yes** - adds `faults` field |
| `ENABLE_WARP_SPEED` | **Yes** - adds `warpComm`/`warpChannel` |
| `ENABLE_COLLTRACE` | **Yes** - adds trace fields |
| `ENABLE_PROFILING` | **Yes** - adds `prof` field |

CMake propagates these flags via `RCCL_DEFS` to specialized kernels and `DISPATCHER_DEFS` to the dispatcher. If you see hangs or crashes, verify flag consistency first.

---

## How Dispatch Works

1. Host determines unroll factor for GPU (1, 2, or 4)
2. Host launches `ncclDevKernel_Generic_{1,2,4}`
3. Dispatcher reads `funcId` from work batch in shared memory
4. Dispatcher calls `ncclDevFuncTable_{1,2,4}[funcId]()`
5. Specialized `ncclDevFunc_*` executes the collective

The device linker populates function tables with `R_AMDGPU_RELATIVE64` relocations that the HIP runtime resolves at load time.

---

## Common Pitfalls

1. **Forgetting to rebuild device_linker** after changing `device_linker.cpp` - run `./build.sh`
2. **Stale CMakeCache.txt** - delete it when changing build options
3. **Wrong CMake option name** - use `-DDEVICE_LINKER=ON` (not `DEVICE_LINKER_ENABLED`)
4. **Flag mismatch** - structure layout must match across host/dispatcher/kernels
5. **Testing with system RCCL** - run from `build/release/` or set `LD_LIBRARY_PATH`

---

## Debugging

```bash
# RCCL logging
NCCL_DEBUG=INFO ./test

# HIP/ROCm logging  
AMD_LOG_LEVEL=4 AMD_LOG_MASK=0xFFFFFFFF ./test

# Inspect merged ELF
llvm-readelf -S build/release/device_linker_output/merged_device.elf
llvm-readelf --symbols merged_device.elf | grep ncclDevFunc_

# Attach debugger to child process in unit test
UT_DEBUG_PAUSE=1 ./test/rccl-UnitTests --gtest_filter=AllGather.OutOfPlace
```

---

## Testing

```bash
# Quick test
./install.sh -l --device-linker -r

# Full unit tests
./install.sh -l --device-linker -t
./build/release/test/rccl-UnitTests

# Specific test with 2 GPUs
UT_MIN_GPUS=2 UT_MAX_GPUS=2 ./test/rccl-UnitTests --gtest_filter=AllGather.OutOfPlace

# Smoke tests
cd tools/device_linker/smoke_test
./run_tests.sh test_single_gpu
./run_tests.sh test_two_gpu_simple
```

---

## Architecture Summary

```
Specialized .cpp files (generated)
         │
         ▼ (parallel compile, -fno-gpu-rdc)
    831 .o files with device code
         │
         ▼
   device_linker tool
         │
         ├── Parses host_table.cpp for funcId → name mapping
         ├── Extracts device ELF from each .o
         ├── Merges .text sections
         ├── Builds ncclDevFuncTable_{1,2,4} with relocations
         │
         ▼
   merged_device.elf (single ELF with all kernels + dispatch tables)
         │
         ▼ (bundled with host registration code)
   librccl.so
```

---

## TODOs / Known Issues

### Issue 1: Output Data Not Modified (ACTIVE)

**Symptom:** Kernels load and execute without error, but collective operations (AllReduce, etc.) leave output buffers unchanged (contain zeros or original data).

**Status:** Under active investigation.

**What's known:**
- Kernel launches with correct funcId
- Kernel "completes" (hipStreamSynchronize returns)
- Function pointer dispatch appears to work
- No crashes or hangs

**Suspected areas:**
- Specialized kernel code may not be executing the actual computation
- Possible inlining/optimization issue with `noinline` attributes
- Connection pointer setup in multi-GPU case

### Issue 2: DWARF Debug Info Mismatch

**Symptom:** Subtle errors in DWARF information in the merged image compared to:
- Individual specialized kernel .o files
- Dispatcher DWARF information

**Impact:** May affect debugging with rocgdb (incorrect line numbers, function names, or source locations).

**Status:** Secondary priority - investigate after Issue 1 is resolved.

---

## Ground Rules (from .cursor/rules)

1. **No LLVM modifications** - work only in RCCL
2. **No sudo** without asking
3. **Rebuild completely** using `rm -rf build; ./install.sh -l --device-linker`
4. **No experimental branches** in main repo - copy to separate directory first
5. **Discussion before implementation** - get approval before substantial changes
6. **Debug loop limits** - stop after 2-3 failed fix attempts and discuss
