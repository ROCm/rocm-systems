# rocjitsu Usage Examples

Comprehensive examples for debugging ROCm workloads using rocjitsu emulation.

## Overview

Each example is self-contained with:
- **README.md** - Debugging objectives and instructions
- **Source code** - Complete working examples
- **Makefile** - Build instructions
- **Expected output** - What to look for

## Examples

### Basic Debugging

1. **[vector-add-basic](01-vector-add-basic/)** - Debug a simple vector addition kernel
   - Kernel launch verification
   - Memory transfer debugging
   - Basic correctness checking

2. **[memory-bounds-error](02-memory-bounds-error/)** - Detect out-of-bounds memory access
   - Identifying buffer overruns
   - Memory access pattern analysis
   - Fixing indexing bugs

3. **[kernel-crash-debug](03-kernel-crash-debug/)** - Debug kernel crashes
   - Segmentation fault debugging
   - Invalid memory access
   - Kernel execution tracing

### Race Detection

4. **[data-race-simple](04-data-race-simple/)** - Find data races in shared memory
   - Race detector usage
   - Atomic operations
   - Synchronization primitives

5. **[global-memory-race](05-global-memory-race/)** - Detect global memory races
   - Multiple thread blocks
   - Unprotected writes
   - Race condition patterns

### Performance Debugging

6. **[memory-coalescing](06-memory-coalescing/)** - Analyze memory access patterns
   - Coalesced vs uncoalesced access
   - Performance profiling
   - Memory bandwidth optimization

7. **[occupancy-analysis](07-occupancy-analysis/)** - Optimize kernel occupancy
   - Register usage analysis
   - Shared memory usage
   - Thread block sizing

### Advanced Examples

8. **[gemm-debugging](08-gemm-debugging/)** - Debug matrix multiplication
   - Using rocBLAS
   - Numerical accuracy
   - Performance analysis

9. **[pytorch-model-debug](09-pytorch-model-debug/)** - Debug PyTorch models
   - Daemon mode setup
   - Training loop debugging
   - Gradient checking

10. **[multi-gpu-collective](10-multi-gpu-collective/)** - Debug RCCL collectives
    - Multi-GPU configuration
    - AllReduce debugging
    - Communication tracing

### Cross-Architecture Translation

11. **[dbt-cross-arch](11-dbt-cross-arch/)** - Dynamic Binary Translation
    - CDNA4 to CDNA3 translation
    - DBT verification
    - Cross-architecture testing

## Quick Start

```bash
# Navigate to an example
cd usage-examples/01-vector-add-basic

# Read the README
cat README.md

# Build the example
make

# Run with rocjitsu
make run

# Clean up
make clean
```

## Requirements

- CMake 3.22+
- C++20 compiler (GCC 13+ or Clang 16+)
- ROCm installation (for HIP examples)
- hipcc compiler (for GPU kernels)
- rocjitsu built and in PATH

## Directory Structure

Each example follows this structure:

```
example-name/
├── README.md          # Objectives, usage, expected output
├── Makefile           # Build and run commands
├── src/
│   ├── main.cpp       # Host code
│   └── kernel.hip     # GPU kernel (if applicable)
└── expected-output/   # Sample output files
    └── output.txt
```

## Getting Help

- [Main rocjitsu Documentation](../docs/)
- [Troubleshooting Guide](../docs/troubleshooting.md)
- [Configuration Reference](../docs/configuration.md)
- [GitHub Issues](https://github.com/ROCm/rocm-systems/issues)
