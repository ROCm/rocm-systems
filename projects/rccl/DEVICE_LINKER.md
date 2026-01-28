# Device Linker for Specialized Kernels

## Overview

The device linker is an optimization that dramatically reduces RCCL build times by compiling specialized kernels in parallel and merging them into a single dispatchable object. This approach achieves **~10x faster builds** compared to the production generic kernel compilation while maintaining identical runtime behavior.

## Build Time Comparison

| Build Type | Time | Speedup |
|------------|------|---------|
| Production (generic kernels) | ~15 min | baseline |
| Specialized (device linker) | ~1.5 min | **~10x faster** |

*Measured on gfx942 (MI300X) with all unroll factors (1, 2, 4)*

## How It Works

### Production Build (Generic Kernels)

In the production build, RCCL compiles generic kernels that handle all collective operations through template instantiation. The compiler must process massive template expansions, resulting in long compilation times.

### Specialized Build (Device Linker)

The specialized approach:

1. **Parallel Compilation**: Each collective operation (AllReduce, AllGather, etc.) is compiled as a separate specialized kernel file. These compile independently and in parallel.

2. **Device Function Extraction**: The device linker extracts `ncclDevFunc_*` device functions from each compiled object file.

3. **Merged Dispatch**: A small dispatcher kernel (`merged_minimal.hip`) contains function pointer tables that route to the extracted device functions based on `funcId`.

4. **ELF Merging**: The device linker merges all device code into a single ELF object with populated function tables.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Specialized Kernel Build                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ specialized_ │  │ specialized_ │  │ specialized_ │  ...      │
│  │ allreduce_   │  │ allgather_   │  │ reduce_      │           │
│  │ ring_simple_ │  │ pat_ll_      │  │ ring_ll128_  │           │
│  │ sum_f32_     │  │ sum_i8_      │  │ prod_f64_    │           │
│  │ unroll2.cpp  │  │ unroll2.cpp  │  │ unroll2.cpp  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    .o file   │  │    .o file   │  │    .o file   │  (parallel)│
│  │ (device code)│  │ (device code)│  │ (device code)│           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │  device_linker  │                             │
│                  │   (C++ tool)    │                             │
│                  └────────┬────────┘                             │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                    │
│         ▼                 ▼                 ▼                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ func_table_1│  │ func_table_2│  │ func_table_4│              │
│  │ (unroll=1)  │  │ (unroll=2)  │  │ (unroll=4)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                           │                                      │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │ merged_device_  │                             │
│                  │   final.o       │                             │
│                  │ (single ELF)    │                             │
│                  └─────────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### `tools/device_linker/device_linker.cpp`

The main C++ tool that:
- Parses `host_table.cpp` to extract funcId mappings
- Reads compiled `.o` files and extracts device code from compressed fatbins
- Demanges kernel names to match them with funcId entries
- Builds function pointer tables for each unroll factor (1, 2, 4)
- Merges all device code into a single ELF with populated tables

### `tools/device_linker/merged_minimal.hip`

The dispatcher kernel that:
- Declares three function pointer tables (`ncclDevFuncTable_1`, `_2`, `_4`)
- Provides `ncclDevKernel_Merged_1`, `_2`, `_4` entry points
- Dispatches to the appropriate `ncclDevFunc` based on `funcId`

### `tools/device_linker/extract_device_from_fatbin.py`

Python script that extracts device code from compressed `.hip_fatbin` sections (CCOB format).

### `src/device/generate_specialized.py`

Generates specialized kernel source files with:
- Unique kernel names including unroll factor (e.g., `ncclDevKernel_AllReduce_RING_SIMPLE_Sum_f32_unroll2_Specialized`)
- `ncclDevFunc_*` definitions that the device linker extracts
- Kernel selector headers for non-device-linker builds

## CMake Configuration

Enable the device linker with:

```cmake
cmake .. \
  -DGPU_TARGETS=gfx942 \
  -DBUILD_LOCAL_GPU_TARGET_ONLY=OFF \
  -DDEVICE_LINKER_ENABLED=ON \
  -DSPECIALIZED_KERNELS_ONLY=ON
```

### Options

| Option | Description |
|--------|-------------|
| `DEVICE_LINKER_ENABLED` | Enable the device linker pipeline |
| `SPECIALIZED_KERNELS_ONLY` | Build specialized kernels (auto-enabled by `DEVICE_LINKER_ENABLED`) |
| `BUILD_LOCAL_GPU_TARGET_ONLY` | When `ON`, only build for local GPU's optimal unroll factor |
| `GPU_TARGETS` | Target GPU architectures (e.g., `gfx942`) |

## Unroll Factor Selection

The unroll factor is determined by GPU architecture:

| GPU | Unroll Factor |
|-----|---------------|
| gfx908 | 2 |
| gfx942 (MI300X/MI300A) | 2 |
| gfx950 | 1, 2 |
| Others | 4 |

When `BUILD_LOCAL_GPU_TARGET_ONLY=OFF`, all unroll factors (1, 2, 4) are generated to match production behavior.

## Build Process

1. **CMake Configure**: Generates specialized kernel `.cpp` files
2. **Parallel Compile**: Ninja compiles all specialized kernels in parallel
3. **PRE_LINK Hook**: Before final link, the device linker pipeline runs:
   - Compile `merged_minimal.hip` (dispatcher)
   - Extract host-only object from dispatcher
   - Extract device code from dispatcher
   - Run `device_linker` to merge all device functions
   - Bundle merged device code with host registration code
4. **Final Link**: Link librccl.so with merged kernel object

## Runtime Behavior

At runtime:
1. Host code determines the appropriate unroll factor for the GPU
2. Selects `ncclDevKernel_Merged_1`, `_2`, or `_4` based on unroll
3. Dispatcher kernel looks up `funcId` in the appropriate table
4. Calls the specialized `ncclDevFunc_*` via function pointer

This matches production behavior exactly - the same funcId dispatch mechanism is used, just with pre-compiled specialized functions instead of template-instantiated generic ones.

## Files Modified

- `CMakeLists.txt`: Added device linker pipeline integration
- `src/device/generate_specialized.py`: Added unroll factor to kernel names, fixed multi-unroll generation
- `src/enqueue.cc`: Added `DEVICE_LINKER_ENABLED` path using merged kernels

## Files Added

- `tools/device_linker/device_linker.cpp`: Main C++ device linker tool
- `tools/device_linker/merged_minimal.hip`: Dispatcher kernel
- `tools/device_linker/extract_device_from_fatbin.py`: Fatbin extraction script
- `tools/device_linker/CMakeLists.txt`: Build for device_linker tool
- `tools/device_linker/hipcc_extract.sh`: Optional compiler wrapper
- `tools/device_linker/.gitignore`: Ignore build artifacts

## Building the Device Linker Tool

The device_linker tool must be built before the main RCCL build:

```bash
cd tools/device_linker
mkdir -p build && cd build
cmake .. -GNinja
ninja
cp device_linker ..
```

Or it will be built automatically if not present during the RCCL build.

## Verification

After building, verify the library works:

```bash
# Check that all functions were mapped
# Output should show: Mapped 2493 functions (unroll1=831, unroll2=831, unroll4=831)

# Run a simple test
hipcc test.cpp -I build/release/hipify/src/include -L build/release -lrccl -o test
./test
```

## Limitations

- Currently hardcoded for gfx942 in the CMake pipeline (can be extended)
- Requires ROCm 6.2+ for compressed fatbin support
- Device linker tool must be pre-built or built during configure
