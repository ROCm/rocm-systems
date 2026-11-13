# AMD GPU Mega Kernel Unit Test

## Supported GPU Architectures

### CDNA-CDNA4

| GPU Series | Architecture | GFXIP | Key Features |
|------------|--------------|-------|--------------|
| **MI250/MI250X** | CDNA2 | gfx90a | FP32/FP64/FP16/BF16, HW FP64 atomics, MFMA, DOT4/DOT8 |
| **MI300/MI300X/MI325X** | CDNA3 | gfx942 | All MI250 + FP8/BF8 (FNUZ-style cvt), TF32, Sparse MFMA |
| **MI350X/MI355X** | CDNA4 | gfx950 | All MI300 + FP6/FP4, OCP FP8/BF8 where applicable, Async LDS, Enhanced WMMA |

### RDNA4

| GPU Series | Architecture | GFXIP | Key Features |
|------------|--------------|-------|--------------|
| **RX 9070 XT/9070** | RDNA4 | gfx1201 | FP8/BF8 OCP, Async LDS, Wave32, WMMA (gfx12 path) |
| **RX 9060 series** | RDNA4 | gfx1200 | Same as gfx1201 with different CU count |

### RDNA 3.5(gfx115x-class APU)

| GPU Series | Architecture | GFXIP | Key Features |
|------------|--------------|-------|--------------|
| **Strix Point** | RDNA 3.5 | gfx1150 | 16 CUs, WMMA, Wave32, Async LDS |
| **Strix Halo** | RDNA 3.5 | gfx1151 | Up to 40 CUs, WMMA, Wave32, Async LDS |
| **Krackan** | RDNA 3.5 | gfx1152 | 32MB Infinity Cache, same as gfx1150 |

### gfx1250

| GPU Series | Architecture | GFXIP | Key Features |
|------------|--------------|-------|--------------|
| **gfx1250** | gfx1250 | gfx1250 | FP8/BF8 OCP, HW FP64 atomics, WMMA (gfx1250 builtins), TDM, cluster ops, cooperative atomics, Async LDS |

## Building

### Prerequisites
- ROCm 6.0+ with HIP
- hipcc compiler
- AMD GPU (e.g. Strix Halo / gfx1151, or other supported GFXIP)

### Using Make

```bash
# Build for gfx1151 (default; override with GPU_ARCH=gfx950, etc.)
make

# Build for specific architectures
make gfx950      # MI350/MI355X (CDNA4)
make gfx942      # MI300/MI300X/MI325X (CDNA3)
make gfx90a      # MI250/MI250X (CDNA2)
make gfx1150     # Strix Point (RDNA 3.5)
make gfx1151     # Strix Halo (RDNA 3.5)
make gfx1152     # Krackan (RDNA 3.5)
make gfx1201     # RX 9070 XT/9070 (RDNA4)
make gfx1200     # RX 9060 series (RDNA4)
make gfx1250     # gfx1250

# Build for all architectures (all ARCHES in Makefile)
make all-arch

# Build and run
make run

# Show help
make help

# Clean build artifacts
make clean
```

### Manual Build

```bash
# For MI350 (CDNA4)
hipcc -O2 -std=c++17 --offload-arch=gfx950 -o mega_kernel_test main.cpp mega_kernel.hip


# For Strix Point/Strix Halo/Krackan (RDNA 3.5)
hipcc -O2 -std=c++17 --offload-arch=gfx1150 -o mega_kernel_test main.cpp mega_kernel.hip  # Strix Point
hipcc -O2 -std=c++17 --offload-arch=gfx1151 -o mega_kernel_test main.cpp mega_kernel.hip  # Strix Halo
hipcc -O2 -std=c++17 --offload-arch=gfx1152 -o mega_kernel_test main.cpp mega_kernel.hip  # Krackan

# For gfx1250
hipcc -O2 -std=c++17 --offload-arch=gfx1250 -o mega_kernel_test_gfx1250 main.cpp mega_kernel.hip
```

## Running

```bash
# Run with default settings (batch_size=1024)
./mega_kernel_test

# Run with custom batch size
./mega_kernel_test -b 4096

# Run with larger batch size for stress testing
./mega_kernel_test --batch-size 65536

# Run with custom batch size and block size (64-256, power of 2)
./mega_kernel_test -b 65536 -t 256

# gfx1250 metric-health profile workload (default block size 256; do not use -t 32)
./mega_kernel_test_gfx1250 -b 655360

# Run kernel multiple times (for benchmarking/profiling)
./mega_kernel_test -n 100

# Run with larger batch and multiple iterations
./mega_kernel_test -b 65536 -n 50

# Run with only inline assembly MFMA tests
./mega_kernel_test --mfma-mode asm

# Run with only HIP builtin MFMA tests
./mega_kernel_test -m builtin

# Run both ASM and builtin MFMA tests (default)
./mega_kernel_test -m both

# Show help
./mega_kernel_test --help
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --batch-size <N>` | Set batch/problem size (64 to 16M) | 1024 |
| `-t, --block-size <N>` | Set thread block size (power of 2, 64-256) | 256 |
| `-n, --num-iterations <N>` | Number of kernel iterations (1 to 10000) | 5 |
| `-m, --mfma-mode <MODE>` | MFMA test mode: `both`, `asm`, or `builtin` (see note) | both |
| `-v, --verbose` | Enable verbose output | off |
| `-h, --help` | Show help message | - |

**Note on MFMA mode:** MFMA instructions require complex accumulator register allocation that is difficult to express in inline assembly constraints. As a result, the MFMA tests primarily use HIP builtins which properly handle register allocation. The `--mfma-mode` option controls which test paths are exercised:
- `builtin` (recommended): Uses HIP builtins (e.g., `__builtin_amdgcn_mfma_f32_32x32x2f32`)
- `asm`: Attempts inline assembly where supported (falls back to builtins for complex MFMA ops)
- `both` (default): Runs both code paths where applicable


```
================================================================================
AMD GPU MEGA KERNEL UNIT TEST
================================================================================
Device Information:
  Name:                  AMD Instinct MI350
  GCN Architecture:      gfx950
  Compute Units:         304
  ...
================================================================================
Detected Platform:        MI350 (CDNA4/gfx950)
================================================================================

Running GPU Mega Kernel for  MI350 (CDNA4/gfx950)...

TEST RESULTS
================================================================================

[Category: Warp/Wave Operations]
  Warp Shuffle (shfl/shfl_xor/shfl_up/down)      : PASS (4/1)
  Warp Ballot (__ballot/__any/__all)             : PASS (4/1)
  ...

[Category: Atomic Operations]
  Atomic Add FP64 (HW atomic on CDNA)            : PASS (4/1)
  ...


SUMMARY
================================================================================
Test Categories Passed:   44 / 44
Test Categories Failed:   0
Test Categories Bypassed: 0 (N/A for this architecture)
Kernel Execution Time:    0.523 ms

*** OVERALL: PASSED (44/44, 0 bypassed) ***
```

### Sample Output (gfx1151 - Strix Halo)

```
================================================================================
AMD GPU MEGA KERNEL UNIT TEST
================================================================================
Device Information:
  Name:                  AMD Radeon(TM) Graphics
  GCN Architecture:      gfx1151
  Compute Units:         40
  ...
================================================================================
Detected Platform:       Strix/Strix Halo/Krackan (RDNA3.5/gfx1150,gfx1151,gfx1152)
================================================================================

Running GPU Mega Kernel for Strix/Strix Halo/Krackan (RDNA3.5/gfx1150,gfx1151,gfx1152)...

TEST RESULTS
================================================================================

[Category: Warp/Wave Operations]
  Warp Shuffle (shfl/shfl_xor/shfl_up/down)      : PASS (4/1)
  ...

[Category: Atomic Operations]
  Atomic Add FP64 (HW atomic on CDNA)            : BYPASSED (N/A)
  ...

[Category: Dual-Issue VALU (MI350/gfx950 specific)]
  Dual-Issue Patterns (FP64+FP32 co-exec)        : BYPASSED (N/A)


SUMMARY
================================================================================
Test Categories Passed:   35 / 35
Test Categories Failed:   0
Test Categories Bypassed: 9 (N/A for this architecture)
Kernel Execution Time:    0.312 ms

*** OVERALL: PASSED (35/35, 9 bypassed) ***
```

## Architecture Comparison

| Feature | MI250 (gfx90a) | MI300 (gfx942) | MI350 (gfx950) | Strix/Krackan (gfx115x)  | RX 9070 XT (gfx1201) | gfx1250 |
|---------|:-------------:|:--------------:|:--------------:|:-------------------:|:--------------------:|:---------------:|
| Architecture | CDNA2 | CDNA3 | CDNA4 | RDNA 3.5 | RDNA4 | gfx1250 |
| FP8 Support | No | FNUZ | OCP | No | OCP | OCP |
| FP6/FP4 Support | No | No | Yes | No | No | No |
| WMMA Wave64 | Yes | Yes | No (Wave32) | No | No | No |
| VGPRs per Wave32 | 512 | 512 | 1024 | 512 | 512 | — |
| Packed FP16 Atomics | Yes | Yes | Yes | Yes | Yes | Yes |
| Packed BF16 Atomics | No | Yes | Yes | Yes | Yes | Yes |
| Dual VALU Issue | No | VOPD | Yes | VOPD | No | Yes |
| Async LDS (hardware) | No | Limited† | Yes | Yes | Yes | Yes |
| MFMA Matrix Ops | Yes | Yes | Yes | No | No | No |
| WMMA Matrix Ops | No | No | No | Yes | Yes | Yes (gfx1250) |

† This sample still treats gfx942 as without the async-LDS builtins path; see `mega_kernel_device_arch.h`.

### Memory/Atomic Instruction Syntax

| Instruction Type | CDNA (gfx9x) | RDNA4 (gfx12) |
|------------------|--------------|---------------|
| Load 32-bit | `*_load_dword` | `*_load_b32` |
| Load 64-bit | `*_load_dwordx2` | `*_load_b64` |
| Store 32-bit | `*_store_dword` | `*_store_b32` |
| Atomic add | `global_atomic_add` | `global_atomic_add_u32` |
| Atomic max (signed) | `global_atomic_smax` | `global_atomic_max_i32` |
| Return old value | `glc` flag | `th:TH_ATOMIC_RETURN` |
| Wait for VMEM | `s_waitcnt vmcnt(0)` | `s_waitcnt vmcnt(0)` |
| Wait for LDS | `s_waitcnt lgkmcnt(0)` | `s_waitcnt lgkmcnt(0)` |

## License

Copyright (c) 2026 - For testing purposes

## References

- [AMD Instinct MI350/ Architecture](https://rocm.docs.amd.com/)
- [AMD Radeon RX 9000 Series](https://www.amd.com/radeon/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
