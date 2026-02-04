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
| test_two_gpu | HANG | Multi-GPU kernel hangs with device linker build |

## Known Issues

**Multi-GPU tests hang with device linker build but pass with production RCCL.**

The kernel launches successfully but never completes. Investigation suggests connection 
pointers in `ncclShmem.groups[].recvConns[]` and `sendConns[]` may be NULL or invalid.

Use `rocgdb` to debug - the device linker now produces correct DWARF5 debug info with
proper line numbers and function symbols.

## Building Individual Tests

```bash
hipcc -O2 -I$BUILD_DIR/include test_single_gpu.cpp \
    -L$BUILD_DIR -lrccl -Wl,-rpath,$BUILD_DIR \
    -o test_single_gpu
```
