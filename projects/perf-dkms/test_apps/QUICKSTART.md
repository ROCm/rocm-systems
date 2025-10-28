# Perf-DKMS Test Suite - Quick Start Guide

## Build Instructions

```bash
cd <path-to-rocm-systems>/projects/perf-dkms/test_apps
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Run All Tests

```bash
cd <path-to-rocm-systems>/projects/perf-dkms/test_apps
./run_all_tests.sh
```

## Run Individual Tests

```bash
cd <path-to-rocm-systems>/projects/perf-dkms/test_apps/build

# Test 1: VALU Heavy
./valu_heavy/valu_heavy

# Test 2: SALU Heavy
./salu_heavy/salu_heavy

# Test 3: LDS Basic
./lds_basic/lds_basic

# Test 4: LDS Conflicts
./lds_conflicts/lds_conflicts

# Test 5: SMEM
./smem/smem

# Test 6: Buffer Read
./buffer_read/buffer_read

# Test 7: Buffer Write
./buffer_write/buffer_write

# Test 8: FLAT Memory
./flat_memory/flat_memory

# Test 9: Memory Sizes
./mem_sizes/mem_sizes

# Test 10: Cache Hit
./cache_hit/cache_hit

# Test 11: Cache Miss
./cache_miss/cache_miss

# Test 12: Wave Modes
./wave_modes/wave_modes

# Test 13: Waits/Stalls
./waits_stalls/waits_stalls

# Test 14: Basic Activity
./basic_activity/basic_activity
```

## Test Suite Summary

| # | Test Name | Purpose | Target Counters |
|---|-----------|---------|-----------------|
| 1 | valu_heavy | Float-heavy FMA operations | SQ_INSTS_VALU |
| 2 | salu_heavy | Scalar integer operations | SQ_INSTS_SALU |
| 3 | lds_basic | Sequential LDS access | SQ_LDS_* |
| 4 | lds_conflicts | LDS bank conflicts | SQ_LDS_BANK_CONFLICT |
| 5 | smem | Constant memory reads | SQ_INSTS_SMEM |
| 6 | buffer_read | Coalesced memory reads | TA_* |
| 7 | buffer_write | Coalesced memory writes | Memory write counters |
| 8 | flat_memory | Generic pointer operations | SQ_INSTS_FLAT |
| 9 | mem_sizes | 32B/64B/128B transactions | GL2C transaction counters |
| 10 | cache_hit | L2 cache hits | GL2C hit counters |
| 11 | cache_miss | L2 cache misses | GL2C miss counters |
| 12 | wave_modes | Wave32/Wave64 operations | Wave occupancy counters |
| 13 | waits_stalls | Memory dependencies | GRBM_* wait/stall counters |
| 14 | basic_activity | Baseline GPU activity | GRBM_GUI_ACTIVE |

## Usage with Perf

Example monitoring a specific test:

```bash
# Monitor VALU instructions
perf stat -e amd_gpu/SQ_INSTS_VALU/ ./build/valu_heavy/valu_heavy

# Monitor multiple counters
perf stat -e amd_gpu/SQ_INSTS_VALU/,amd_gpu/SQ_INSTS_SALU/ ./build/valu_heavy/valu_heavy

# Profile with rocprof
rocprof --stats ./build/valu_heavy/valu_heavy
```

## File Structure

```
test_apps/
├── CMakeLists.txt          # Root build configuration
├── README.md               # Detailed documentation
├── QUICKSTART.md           # This file
├── run_all_tests.sh        # Script to run all tests
├── common/                 # Shared infrastructure
│   ├── hip_utils.hpp       # HIP error checking and utilities
│   ├── test_framework.hpp  # Base test class framework
│   └── CMakeLists.txt
├── valu_heavy/             # Test 1
│   ├── valu_heavy.cpp
│   └── CMakeLists.txt
├── salu_heavy/             # Test 2
│   ├── salu_heavy.cpp
│   └── CMakeLists.txt
... (and so on for all 14 tests)
└── build/                  # Build output directory
    ├── valu_heavy/valu_heavy
    ├── salu_heavy/salu_heavy
    ... (all test executables)
```

## Prerequisites

- ROCm/HIP installation (tested with ROCm 5.x+)
- CMake 3.16 or later
- C++17 compatible compiler
- AMD GPU hardware

## Troubleshooting

**Build fails with "HIP not found":**
```bash
export HIP_PATH=/opt/rocm
cmake -DHIP_PATH=/opt/rocm ..
```

**No HIP devices found at runtime:**
```bash
# Verify GPU is detected
rocminfo

# Check drivers are loaded
lsmod | grep amdgpu
```

**Tests crash or hang:**
- Check GPU status: `rocm-smi`
- Review dmesg for GPU errors: `dmesg | tail`
- Ensure sufficient GPU memory

## Support

For detailed information, see README.md in this directory.
For perf-dkms specific issues, refer to the main project documentation.
