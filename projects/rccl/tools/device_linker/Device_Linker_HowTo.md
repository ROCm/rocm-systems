# Device Linker How-To Guide

## Quick Start

### Building with Device Linker

```bash
# Full rebuild from workspace root
cd /work2/lmeadows/rocm-systems/projects/rccl
rm -rf build
./install.sh -l --device-linker
```

### Options

| Option | Description |
|--------|-------------|
| `--device-linker` | Enable the device linker pipeline |
| `-l` | Build for local GPU only (faster) |
| `--amdgpu_targets=gfx942` | Specify target GPU architecture |
| `-t` | Build with unit tests |
| `-r` | Build and run quick tests (AllReduce.*) |

### CMake Direct Configuration

```bash
cmake .. \
  -DGPU_TARGETS=gfx942 \
  -DBUILD_LOCAL_GPU_TARGET_ONLY=OFF \
  -DDEVICE_LINKER=ON
```

---

## Current Status

| Test | Status |
|------|--------|
| Single-GPU | WORKING |
| Multi-GPU | WORKING |
| System RCCL Multi-GPU | WORKING |
| Smoke Test | PASSED |

---

## Critical Constraint: Compilation Flags

**All compilation units MUST have identical flags for consistent structure layouts.**

| Flag | Purpose | Affects Layout |
|------|---------|----------------|
| `DEVICE_LINKER` | Function table dispatch | No |
| `ENABLE_FAULT_INJECTION` | Adds `faults` field to ncclShmemData | **Yes** |
| `ENABLE_WARP_SPEED` | Adds `warpComm`/`warpChannel` fields | **Yes** |
| `ENABLE_LL128` | Protocol selection | Possibly |
| `ENABLE_COLLTRACE` | Adds `collTrace`/`collTraceTail` fields | **Yes** |
| `ENABLE_PROFILING` | Adds `prof` field to ncclShmemData | **Yes** |

If structure layouts mismatch between host, dispatcher, and specialized kernels, the kernel will access wrong memory offsets and crash or hang.

---

## Rebuilding Components

### Full Rebuild (Recommended)

```bash
rm -rf build; ./install.sh -l --device-linker
```

### Rebuild Device Linker Tool Only

```bash
cd tools/device_linker
./build.sh
```

---

## Testing

### Unit Tests

```bash
# Build with unit tests
./install.sh -l --device-linker -t

# Build and run quick tests
./install.sh -l --device-linker -r

# Run full test binary
./build/release/test/rccl-UnitTests

# Run specific test
UT_MIN_GPUS=2 UT_MAX_GPUS=2 ./build/release/test/rccl-UnitTests --gtest_filter=AllGather.OutOfPlace
```

### Smoke Tests

```bash
cd tools/device_linker/smoke_test

# Single GPU test
./run_tests.sh test_single_gpu

# Two GPU test
./run_tests.sh test_two_gpu_simple
```

### Test Environment Variables

| Variable | Purpose |
|----------|---------|
| `UT_MIN_GPUS=N` | Minimum GPUs for test |
| `UT_MAX_GPUS=N` | Maximum GPUs for test |
| `UT_VERBOSE=1` | Verbose test output |
| `UT_DEBUG_PAUSE=1` | Pause for debugger attachment |
| `NCCL_DEBUG=INFO` | RCCL debug logging |

---

## Debugging

### RCCL Logging

| Variable | Effect |
|----------|--------|
| `NCCL_DEBUG=INFO` | Info level logging |
| `NCCL_DEBUG=WARN` | Warning level logging |
| `NCCL_DEBUG_SUBSYS=ALL` | Enable all debug subsystems |
| `NCCL_DEBUG_FILE=path` | Write logs to file |

### HIP/ROCm Runtime Logging

| Variable | Effect |
|----------|--------|
| `AMD_LOG_LEVEL=4` | Detailed runtime logging (0-5) |
| `AMD_LOG_MASK=0xFFFFFFFF` | Log all subsystems |
| `GPU_DUMP_CODE_OBJECT=1` | Dump code objects |
| `AMD_COMGR_EMIT_VERBOSE_LOGS=1` | COMGR verbose logging |
| `HSA_ENABLE_DEBUG=1` | Extra HSA validation |

### Using rocgdb

```bash
# Attach to child process during test
UT_MIN_GPUS=2 UT_MAX_GPUS=2 UT_DEBUG_PAUSE=1 ./test/rccl-UnitTests --gtest_filter=AllGather.OutOfPlace
# In another terminal:
rocgdb -p <child_pid>
```

### Inspecting Merged ELF

```bash
# Check sections
llvm-readelf -S build/release/device_linker_output/merged_device.elf

# Check symbols
llvm-readelf --symbols merged_device.elf | grep ncclDevFunc_

# Check relocations
llvm-readelf -r merged_device.elf

# Disassemble kernel descriptors
llvm-objdump -Dr --section=.rodata merged_device.elf
```

---

## Common Build Mistakes

### CMake Configuration

| Mistake | Correct |
|---------|---------|
| `-DDEVICE_LINKER_ENABLED=ON` | `-DDEVICE_LINKER=ON` |
| Using `SPECIALIZED_KERNELS_ONLY` | OBSOLETE - use `DEVICE_LINKER` |
| `ONLY_FUNCS` cached from previous build | Delete `CMakeCache.txt` when changing config |

### Device Linker Pipeline

| Mistake | Correct |
|---------|---------|
| Forgetting to rebuild device_linker after code changes | Run `tools/device_linker/build.sh` |
| `--offload-compress` in dispatcher compile | Remove it - causes extraction failures |
| Wrong target arch | Auto-detected from rocminfo; override with `--target` |

---

## Files Modified by Device Linker

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Device linker pipeline integration |
| `src/device/generate_specialized.py` | Unroll factor in kernel names |
| `src/enqueue.cc` | `DEVICE_LINKER` path using merged kernels |

## Files Added by Device Linker

| File | Purpose |
|------|---------|
| `tools/device_linker/device_linker.cpp` | Main C++ tool |
| `tools/device_linker/merged_minimal.hip` | Dispatcher kernel |
| `tools/device_linker/extract_device_from_fatbin.py` | Fatbin extraction |
| `tools/device_linker/CMakeLists.txt` | Build for device_linker tool |
| `tools/device_linker/build.sh` | Build script for device_linker |

---

## Verification

After building, verify the library works:

```bash
# Check function mapping output during build
# Should show: Mapped 2493 functions (unroll1=831, unroll2=831, unroll4=831)

# Verify DWARF (if debug enabled)
llvm-dwarfdump --verify build/release/device_linker_output/merged_device.elf
```

---

## Limitations

- Requires ROCm 6.2+ for compressed fatbin support
- Device linker tool must be pre-built or built during configure

---

## Troubleshooting

### "cudaArch X ncclMaxSharedMem Y exceeds device/fn maxSharedMem Z"

LDS allocation exceeds device limit. The device linker caps LDS to device maximum, but verify your build matches the target GPU.

### Tests pass under rocprofv3 but fail standalone

rocprofv3 may load system RCCL instead of the build. Run from `build/release/` directory or set `LD_LIBRARY_PATH` to ensure the device-linker build is used.

### Kernel hangs during multi-GPU operations

Check that all compilation flags match between host, dispatcher, and specialized kernels. Structure layout mismatches cause memory access errors.

### "Mapped 0 kernel functions"

Wrong target arch or input directory. Ensure `--target-arch` matches the .o files and `--input-dir` path is correct.
