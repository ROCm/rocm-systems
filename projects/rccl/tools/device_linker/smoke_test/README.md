# Device Linker Smoke Tests

Simple tests to verify the device linker RCCL build works correctly.

## Running Tests

```bash
./run_tests.sh [BUILD_DIR]
```

Where `BUILD_DIR` is the RCCL build directory (default: `../../../build/release`).

## Test Status

| Test | Status | Notes |
|------|--------|-------|
| test_single_gpu | PASS | Basic AllReduce on single GPU |
| test_two_gpu | FAIL | Multi-GPU crashes with device linker build |
| test_multi_gpu_allreduce | FAIL | Multi-GPU crashes with device linker build |
| test_multi_gpu_broadcast | FAIL | Multi-GPU crashes with device linker build |
| test_multi_gpu_reducescatter | FAIL | Multi-GPU crashes with device linker build |

## Known Issues

**Multi-GPU tests crash with device linker build but pass with production RCCL.**

This indicates an issue in the device linker integration that only affects multi-GPU
paths. The single-GPU path works correctly, confirming the basic device linker
mechanism (function tables, specialized kernels) is functional.

Investigation needed:
- Check if multi-GPU initialization accesses uninitialized function table entries
- Verify all unroll variants (1, 2, 4) are properly handled
- Check for any multi-GPU specific code paths that reference device symbols differently

## Building Individual Tests

```bash
hipcc -O2 -I$BUILD_DIR/include test_single_gpu.cpp \
    -L$BUILD_DIR -lrccl -Wl,-rpath,$BUILD_DIR \
    -o test_single_gpu
```
