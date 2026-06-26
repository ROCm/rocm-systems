# rocjitsu Usage Examples - Complete Index

Quick reference guide to all available debugging examples.

## Examples by Category

### 🚀 Getting Started (Basic Debugging)

| Example | Description | Key Learning | Files |
|---------|-------------|--------------|-------|
| [01-vector-add-basic](01-vector-add-basic/) | Simple vector addition | Kernel launch, memory transfers, verification | `vector_add.cpp`, `Makefile` |
| 02-memory-bounds-error | Out-of-bounds detection | Buffer overruns, memory safety | `bounds_error.cpp`, `Makefile` |
| 03-kernel-crash-debug | Debug kernel crashes | Segfaults, invalid access | `crash_debug.cpp`, `Makefile` |

### 🔍 Race Detection

| Example | Description | Key Learning | Files |
|---------|-------------|--------------|-------|
| [04-data-race-simple](04-data-race-simple/) | Simple data races | Race detector, atomics | `histogram_race.cpp`, `histogram_fixed.cpp`, `Makefile` |
| 05-global-memory-race | Global memory races | Multi-block synchronization | `global_race.cpp`, `Makefile` |

### ⚡ Performance Analysis

| Example | Description | Key Learning | Files |
|---------|-------------|--------------|-------|
| 06-memory-coalescing | Memory access patterns | Coalesced access, bandwidth | `coalescing.cpp`, `Makefile` |
| 07-occupancy-analysis | Kernel occupancy | Register/shared memory usage | `occupancy.cpp`, `Makefile` |

### 🎯 Advanced Examples

| Example | Description | Key Learning | Files |
|---------|-------------|--------------|-------|
| 08-gemm-debugging | Matrix multiplication | rocBLAS, numerical accuracy | `gemm_debug.cpp`, `Makefile` |
| [09-pytorch-model-debug](09-pytorch-model-debug/) | PyTorch models | Daemon mode, gradient checking | `simple_model.py`, `Makefile` |
| 10-multi-gpu-collective | RCCL collectives | Multi-GPU, AllReduce | `multi_gpu.cpp`, `Makefile` |

### 🔄 Cross-Architecture Translation

| Example | Description | Key Learning | Files |
|---------|-------------|--------------|-------|
| 11-dbt-cross-arch | Dynamic Binary Translation | CDNA4→CDNA3 translation | `dbt_example.cpp`, `Makefile` |

## Quick Start Guide

### First Time Setup

```bash
# 1. Build rocjitsu
cd emulation/rocjitsu
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
export PATH=$PWD/build/tools/rocjitsu:$PATH

# 2. Navigate to an example
cd usage-examples/01-vector-add-basic

# 3. Build and run
make
make run
```

### Running Pattern

All examples follow the same pattern:

```bash
# Build
make              # or 'make build'

# Run basic version
make run

# Run with logging
make run-verbose  # or 'make run-debug'

# Clean up
make clean

# See all options
make help
```

## Common Makefile Targets

Every example includes these targets:

| Target | Description |
|--------|-------------|
| `make` or `make build` | Compile the example |
| `make run` | Run through rocjitsu (basic mode) |
| `make run-verbose` | Run with `RJ_LOG=1` |
| `make run-daemon` | Run in daemon mode |
| `make clean` | Remove build artifacts |
| `make help` | Show all available targets |

## Environment Variables Reference

### Logging and Debugging

```bash
# Enable verbose logging
RJ_LOG=1 make run

# Enable race detection
RJ_SINKS=race_detector make run

# Write logs to file
RJ_SINKS=file RJ_SINK_DIR=./logs make run

# Multiple sinks
RJ_SINKS=race_detector,file RJ_SINK_DIR=./logs make run
```

### Performance and Profiling

```bash
# Enable profiling
RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 RJ_SINKS=file make run

# Specify runtime directory
ROCJITSU_RUNTIME_DIR=/tmp/rocjitsu make run
```

### Dynamic Binary Translation

```bash
# Enable DBT with logging
RJ_DBT_LOG=1 make run

# Set target ISA
RJ_DBT_TARGET_ISA=gfx942 make run
```

## Learning Path

### Beginner
1. Start with [01-vector-add-basic](01-vector-add-basic/) - Learn the basics
2. Try [04-data-race-simple](04-data-race-simple/) - Understand race detection
3. Explore [09-pytorch-model-debug](09-pytorch-model-debug/) - Real-world ML debugging

### Intermediate
4. Study 06-memory-coalescing - Optimize memory access
5. Learn 08-gemm-debugging - Work with rocBLAS
6. Practice 05-global-memory-race - Complex race patterns

### Advanced
7. Master 10-multi-gpu-collective - Multi-GPU debugging
8. Experiment 11-dbt-cross-arch - Cross-architecture translation
9. Combine techniques - Debug your own applications

## Troubleshooting

### Example won't build

```bash
# Check hipcc is available
which hipcc

# Verify rocjitsu is built
ls ../../build/tools/rocjitsu/rocjitsu

# Check Makefile paths
make help
```

### Example crashes

```bash
# Run with logging
RJ_LOG=1 make run

# Check for races
RJ_SINKS=race_detector make run

# Use verbose output
make run-verbose 2>&1 | tee output.log
```

### Incorrect results

```bash
# Enable race detector
RJ_SINKS=race_detector make run

# Check numerical precision
# Review the README for that specific example
```

## Getting Help

### Documentation
- [Main README](README.md) - Overview and quick start
- [rocjitsu Docs](../docs/) - Full documentation
- [Building Guide](../docs/building.md) - Build instructions
- [CLI Reference](../docs/rocjitsu-cli.md) - Command-line options

### Example-Specific Help

Each example includes:
- **README.md** - Detailed objectives and instructions
- **Expected output** - What to look for
- **Common issues** - Debugging tips
- **Exercises** - Practice problems

### Community
- [GitHub Issues](https://github.com/ROCm/rocm-systems/issues)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Configuration Files

All examples use JSON configuration files from `../../configs/`:

| Config File | Description | GPU |
|-------------|-------------|-----|
| `amdgpu_cdna4_kmd.json` | Single CDNA4 GPU | gfx950 |
| `amdgpu_cdna4_kmd_2gpu.json` | Dual CDNA4 GPUs | gfx950 |
| `amdgpu_cdna3_kmd.json` | CDNA3 GPU | gfx942 |
| `amdgpu_cdna2_kmd.json` | CDNA2 GPU | gfx90a |

Example usage:
```bash
# Use CDNA3 config
make run CONFIG=../../configs/amdgpu_cdna3_kmd.json
```

## Next Steps

1. **Choose an example** based on your debugging needs
2. **Read the README** in that example directory
3. **Build and run** following the instructions
4. **Experiment** with the exercises
5. **Apply** to your own applications

## Contributing

To add a new example:

1. Create a new directory: `NN-example-name/`
2. Include:
   - `README.md` - Objectives, instructions, expected output
   - `src/` - Source code
   - `Makefile` - Build and run commands
   - `expected-output/` - Sample outputs (optional)
3. Follow the existing structure and naming
4. Update this INDEX.md
5. Submit a pull request

## Summary

These examples provide hands-on experience with:
- ✅ Basic GPU debugging workflows
- ✅ Race condition detection and fixing
- ✅ Performance profiling and optimization
- ✅ Real-world application debugging (PyTorch, rocBLAS)
- ✅ Multi-GPU and distributed debugging
- ✅ Cross-architecture translation

Start with the basics and progress to advanced examples as you build expertise with rocjitsu!