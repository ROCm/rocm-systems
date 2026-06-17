# AMD GPU Performance Counter Test Suite

This directory contains a comprehensive test suite for validating AMD GPU performance counters in the perf-dkms project.

## Overview

The test suite consists of 14 HIP-based GPU applications that exercise specific hardware behaviors to validate that performance counters are working correctly. Each test targets different GPU hardware blocks (GL2C, SQ, TA, GRBM) and specific counter types.

## Directory Structure

```
test_apps/
├── common/                  # Shared test infrastructure
│   ├── test_framework.hpp   # Base test class with timing, error checking
│   ├── hip_utils.hpp        # HIP error checking macros and utilities
│   └── CMakeLists.txt       # Build common library
├── valu_heavy/              # Test #1: VALU compute intensive
├── salu_heavy/              # Test #2: SALU operations
├── lds_basic/               # Test #3: Basic LDS operations
├── lds_conflicts/           # Test #4: LDS bank conflicts
├── smem/                    # Test #5: Scalar memory operations
├── buffer_read/             # Test #6: Buffer read operations
├── buffer_write/            # Test #7: Buffer write operations
├── flat_memory/             # Test #8: FLAT addressing
├── mem_sizes/               # Test #9: Various memory transaction sizes
├── cache_hit/               # Test #10: L2 cache hits
├── cache_miss/              # Test #11: L2 cache misses
├── wave_modes/              # Test #12: Wave32/Wave64 tests
├── waits_stalls/            # Test #13: Wait states and stalls
├── basic_activity/          # Test #14: Basic GPU activity
├── CMakeLists.txt           # Root build configuration
├── run_all_tests.sh         # Script to run all tests
└── README.md                # This file
```

## Test Descriptions

### 1. VALU Heavy (`valu_heavy`)
**Target Counters:** `SQ_INSTS_VALU`
- Performs intensive floating-point FMA (Fused Multiply-Add) operations
- Creates high Vector ALU utilization
- 1M threads × 1000 iterations × 4 FMA ops

### 2. SALU Heavy (`salu_heavy`)
**Target Counters:** `SQ_INSTS_SALU`
- Performs scalar integer operations (shifts, masks, logical ops)
- Uses wave-uniform values for scalar execution
- 1M threads × 1000 iterations × 6 scalar ops

### 3. LDS Basic (`lds_basic`)
**Target Counters:** `SQ_LDS_*`
- Sequential Local Data Share (LDS) access without bank conflicts
- Simple shared memory read/write patterns
- 256K threads using 1KB LDS per block

### 4. LDS Conflicts (`lds_conflicts`)
**Target Counters:** `SQ_LDS_BANK_CONFLICT`
- Stride-32 access pattern designed to cause LDS bank conflicts
- Tests LDS banking on AMD GPUs (32 banks)
- 256K threads using 4KB LDS per block

### 5. SMEM (`smem`)
**Target Counters:** `SQ_INSTS_SMEM`
- Scalar memory operations using constant memory
- Multiple reads from constant cache
- 1M threads × 200 constant memory reads

### 6. Buffer Read (`buffer_read`)
**Target Counters:** `TA_*` (Texture Addresser)
- Coalesced global memory reads
- Tests memory fetch efficiency
- 16MB input buffer with 100 reads per thread

### 7. Buffer Write (`buffer_write`)
**Target Counters:** Memory write counters
- Coalesced global memory writes
- Tests memory store efficiency
- 400MB total writes (100 writes per thread)

### 8. FLAT Memory (`flat_memory`)
**Target Counters:** `SQ_INSTS_FLAT`
- Generic pointer operations using FLAT addressing
- Tests unified address space operations
- 1M threads × 200 FLAT ops (loads + stores)

### 9. Memory Sizes (`mem_sizes`)
**Target Counters:** `GL2C` transaction counters
- Tests 32-byte, 64-byte, and 128-byte transactions
- Uses int, int2, and int4 data types
- Validates transaction size detection

### 10. Cache Hit (`cache_hit`)
**Target Counters:** `GL2C` hit counters
- Small working set (1MB) to maximize L2 cache hits
- Repeatedly accesses same data
- 1M threads × 1000 accesses to small array

### 11. Cache Miss (`cache_miss`)
**Target Counters:** `GL2C` miss counters
- Large working set (256MB) for cache misses
- Streaming access pattern
- Tests memory subsystem under miss conditions

### 12. Wave Modes (`wave_modes`)
**Target Counters:** Wave occupancy counters
- Tests Wave32 and Wave64 execution modes
- Uses wave shuffle operations
- Validates wave-level instruction tracking

### 13. Waits/Stalls (`waits_stalls`)
**Target Counters:** `GRBM_*` wait/stall counters
- Creates memory dependencies causing pipeline stalls
- Tests wait state detection
- Intentional dependencies with synchronization

### 14. Basic Activity (`basic_activity`)
**Target Counters:** `GRBM_GUI_ACTIVE`
- Provides baseline GPU activity measurement
- Simple computation for reference
- Multiple iterations for stable measurements

## Building the Tests

### Prerequisites
- ROCm/HIP installation (tested with ROCm 5.x+)
- CMake 3.16 or later
- C++17 compatible compiler
- AMD GPU hardware

### Build Instructions

```bash
cd /path/to/perf-dkms/test_apps
mkdir build
cd build
cmake ..
make -j$(nproc)
```

The build system will:
1. Find HIP installation
2. Build common infrastructure library
3. Compile all 14 test applications
4. Display configuration summary

### Build Options

```bash
# Specify HIP path
cmake -DHIP_PATH=/opt/rocm ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (default)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Running the Tests

### Run All Tests
```bash
cd /path/to/perf-dkms/test_apps
./run_all_tests.sh
```

This script will:
- Run all 14 tests sequentially
- Report pass/fail status for each
- Print summary at the end
- Exit with code 0 if all pass, 1 if any fail

### Run Individual Test
```bash
cd /path/to/perf-dkms/test_apps/build
./valu_heavy/valu_heavy
./cache_hit/cache_hit
# ... etc
```

Each test will:
1. Print device information
2. Initialize test parameters
3. Execute the kernel
4. Report timing and status
5. Exit with 0 on success

## Test Output

Each test provides detailed output:

```
================================================================================
Test: VALU Heavy - FMA Operations
================================================================================
Using device 0: AMD Radeon RX 7900 XTX
  Compute capability: 11.0
  Total global memory: 24564 MB
  Multiprocessors: 96
  Warp size: 64

Initializing test...
  Array size: 1048576 elements
  Iterations per thread: 1000
  Total VALU operations: 4194304000
Running test...
Kernel: valu_heavy_kernel
  Grid size:  (4096, 1, 1)
  Block size: (256, 1, 1)
  Total threads: 1048576
Kernel execution completed successfully
Cleaning up...

--------------------------------------------------------------------------------
Result: PASSED
Execution time: 145.234 ms
================================================================================
```

## Integration with Perf-DKMS

These tests are designed to be used with perf-dkms performance monitoring:

```bash
# Load the perf-dkms module
sudo modprobe amd_gpu_perf

# Run a test while monitoring counters
perf stat -e amd_gpu/SQ_INSTS_VALU/ ./valu_heavy/valu_heavy

# Profile all tests
for test in build/*/test_*; do
    perf stat -e amd_gpu/... "$test"
done
```

## Validation Methodology

Each test is designed to:
1. **Exercise specific counters** - Targeted workloads for each counter type
2. **Produce measurable activity** - Sufficient operations for reliable counting
3. **Minimize interference** - Clean execution without extraneous operations
4. **Validate correctness** - Tests run successfully without errors

## Troubleshooting

### Build Issues

**HIP not found:**
```bash
export HIP_PATH=/opt/rocm
cmake -DHIP_PATH=/opt/rocm ..
```

**Compilation errors:**
- Ensure ROCm/HIP is properly installed
- Check that GPU compute capability is supported
- Verify C++17 support in compiler

### Runtime Issues

**No HIP devices found:**
- Verify AMD GPU is present: `rocminfo`
- Check GPU drivers are loaded: `lsmod | grep amdgpu`
- Ensure user has GPU access permissions

**Out of memory errors:**
- Reduce array sizes in test source files
- Test on GPU with more memory
- Run tests individually instead of in sequence

**Kernel launch failures:**
- Check dmesg for GPU errors
- Verify GPU is not hung: `rocm-smi`
- Reset GPU if needed: `rocm-smi --gpureset`

## Contributing

When adding new tests:
1. Create a new subdirectory under `test_apps/`
2. Implement test using `test_framework::TestBase` class
3. Add CMakeLists.txt for the test
4. Update root CMakeLists.txt to include new test
5. Add test to `run_all_tests.sh` script
6. Document the test purpose and target counters

## License

This test suite is part of the perf-dkms project and follows the same license terms.

## Contact

For issues or questions about the test suite, please refer to the main perf-dkms project documentation.
