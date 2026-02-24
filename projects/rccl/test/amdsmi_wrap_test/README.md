# AMD SMI Wrapper Standalone Test

This directory contains a standalone test for the `amdsmi_wrap` module in RCCL.

## Overview

The `amdsmi_wrap` module provides a unified wrapper around AMD's system management interfaces:

- **AMD SMI Library** (`libamdsmi.so`) - Modern AMD System Management Interface
- **Internal ARSMI** - Lightweight fallback when AMD SMI is not available

## Prerequisites

- ROCm installation (tested with ROCm 6.0+)
- HIP compiler (`hipcc`)
- AMD GPUs (for runtime testing)

## Building

```bash
# Default build (uses /opt/rocm)
make

# Custom ROCm path
ROCM_PATH=/path/to/rocm make

# Show build configuration
make info
```

## Running

```bash
# Run with internal ARSMI fallback (default)
make run

# Run with AMD SMI library (requires libamdsmi.so)
make run-amdsmi
```

## Test Coverage

The test exercises these `amdsmi_wrap` functions:

- `amd_smi_init()` / `amd_smi_shutdown()`
- `amd_smi_getNumDevice()`
- `amd_smi_getDevicePciBusIdString()`
- `amd_smi_getDeviceIndexByPciBusId()`
- `amd_smi_getLinkInfo()`
- `amd_smi_getFirmwareVersion()`
- `amd_smi_ensureFabricInitialized()`
- `amd_smi_isFabricSupported()`
- `amd_smi_getFabricBandwidth()`
- `amd_smi_getFabricDeviceInfo()`
- `amd_smi_canUseScaleUpFabric()`

## Files

- `amdsmi_wrap_test.cpp` - Main test driver
- `test_stubs.h` - Stub definitions for RCCL dependencies
- `test_utils.cc` - Stub implementations
- `Makefile` - Build configuration

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RCCL_USE_AMD_SMI_LIB` | Set to `1` to use AMD SMI library |
| `ROCM_PATH` | Path to ROCm (default: `/opt/rocm`) |
