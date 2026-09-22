# Canonical HIP Example

This is a canonical HIP example project that demonstrates three common HIP application patterns. It serves as a test case for the rocm-kpack kernel packing and runtime integration.

## Components

### 1. Standalone Executable (`standalone`)

A self-contained executable that directly runs a HIP kernel (vector addition). This demonstrates the simplest case of a HIP application with device code embedded in the executable.

**Source**: `standalone.hip`

**Usage**:

```bash
LD_LIBRARY_PATH=/path/to/rocm/lib ./standalone
```

**Expected Output**:

```
Standalone: Vector addition successful! (N=1024)
```

### 2. Shared Library with Host API (`libvector_lib.so`)

A shared library that exposes a C API for running HIP kernels. The library internally manages GPU memory, executes the kernel, and returns results to the caller.

**Source**: `vector_lib.hip`, `vector_lib.h`

**API**:

```c
// Performs vector addition on the GPU and returns the sum of all elements
// Returns 0 on success, -1 on failure
int vector_add_and_sum(int n, float* result_sum);
```

### 3. Client Executable (`client`)

A simple executable that uses the shared library API. This demonstrates the library/application separation pattern common in HIP applications.

**Source**: `client.cpp`

**Usage**:

```bash
LD_LIBRARY_PATH=/path/to/rocm/lib:. ./client
```

**Expected Output**:

```
Client: Calling vector_add_and_sum with N=1024...
Client: Success! Sum of results = 1571328.000000
Client: Expected sum = 1571328.000000
Client: Verification passed!
```

## Building

### Prerequisites

- ROCm SDK installed (tested with ROCm 6.3+)
- CMake 3.21+
- HIP-capable AMD GPU

### Build Commands

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake (adjust ROCM path as needed)
cmake \
  -DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/path/to/rocm/llvm/bin/clang++ \
  -DHIP_PLATFORM=amd \
  -DCMAKE_PREFIX_PATH=/path/to/rocm \
  ..

# Build all targets
make -j$(nproc)
```

### GPU Architecture Targets

By default, the project builds for `gfx1100` and `gfx1201` architectures. You can override this:

```bash
cmake -DGPU_TARGETS="gfx1100;gfx1201;gfx942" ...
```

## Fat Binary Verification

All HIP binaries (standalone executable and shared library) contain embedded device code for multiple GPU architectures in the `.hip_fatbin` ELF section.

You can verify this using:

```bash
# Check for .hip_fatbin section
readelf -S standalone | grep hip_fatbin
readelf -S libvector_lib.so | grep hip_fatbin

# Extract and list device code bundles (requires rocm-kpack tools)
PATH=/path/to/rocm/llvm/bin:$PATH python -m rocm_kpack.tools.bulk_unbundle standalone
PATH=/path/to/rocm/llvm/bin:$PATH python -m rocm_kpack.tools.bulk_unbundle libvector_lib.so
```

## Purpose

This example serves as:

1. A reference implementation for HIP application patterns
1. A test case for kernel packing (removing fat binaries and using external device code)
1. A basis for runtime modifications to support kpack'd binaries
1. A validation suite for ensuring packing doesn't break functionality

## Next Steps

After verifying the baseline functionality:

1. Pack the binaries using rocm-kpack tools
1. Modify the binaries to remove `.hip_fatbin` sections
1. Test runtime loading of external device code
1. Iterate on runtime integration until packed binaries function correctly
