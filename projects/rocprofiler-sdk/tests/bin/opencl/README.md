# OpenCL Vector Addition Test Application

## Overview

This is a simple ROCm OpenCL application that performs vector addition on the GPU. It is designed to be used with rocprofiler-sdk for profiling and testing OpenCL kernel execution.

## Features

- OpenCL platform and device initialization
- Vector addition kernel execution
- ROCTx annotations for profiling hotspots
- Result verification
- Error handling and device information display

## Application Workflow

1. **Platform/Device Setup**: Queries for OpenCL platform and GPU device
2. **Context Creation**: Creates OpenCL context and command queue
3. **Kernel Compilation**: Compiles the vector addition kernel at runtime
4. **Data Preparation**: Allocates and initializes host vectors
5. **Buffer Creation**: Creates GPU buffers for input/output
6. **Data Transfer**: Copies input data from host to device
7. **Kernel Execution**: Executes the vector addition kernel
8. **Result Retrieval**: Copies results back to host
9. **Verification**: Validates the computed results
10. **Cleanup**: Releases all OpenCL resources

## ROCTx Annotations

The application includes ROCTx annotations for key operations:
- Platform/device initialization
- Context and queue creation
- Program compilation
- Buffer creation and data transfers
- Kernel execution
- Result verification
- Resource cleanup

These annotations enable detailed profiling when used with rocprofiler-sdk.

## Building

The application is built as part of the rocprofiler-sdk test suite:

```bash
cd build
cmake ..
make opencl-vector-add
```

## Running

```bash
./opencl-vector-add
```

Expected output:
```
OpenCL Device Info:
  Device Name: <GPU Name>
  Vendor: Advanced Micro Devices, Inc.
  Max Work Group Size: <size>
  Compute Units: <count>

PASSED: Vector addition verified successfully!
```

## Requirements

- ROCm with OpenCL support
- rocprofiler-sdk-roctx library
- C++17 compatible compiler

## Kernel Details

The vector addition kernel is simple and efficient:
- Global work size: 1024 elements
- Local work group size: 256 threads
- Each thread computes one element: `c[i] = a[i] + b[i]`
