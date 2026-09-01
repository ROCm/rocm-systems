# Building

## Prerequisites

- CMake 3.22+
- C++20 compiler (GCC 13+, Clang 16+)
- Python 3.10+ (for ISA code generation only)
- ROCm toolchain (optional, for HIP test kernels and daemon tests)

Third-party dependencies (Google Test, FlatBuffers) are fetched
automatically via CMake `FetchContent`.

## Quick start

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## CMake options

| Option | Default | Description |
|---|---|---|
| `RJ_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer |
| `RJ_ENABLE_UBSAN` | `OFF` | Enable UndefinedBehaviorSanitizer |
| `RJ_ENABLE_TSAN` | `OFF` | Enable ThreadSanitizer |
| `RJ_ENABLE_MSAN` | `OFF` | Enable MemorySanitizer |
| `RJ_SANITIZER_RUNTIME` | `AUTO` | Select `AUTO`, `SHARED`, or `STATIC` sanitizer runtime linkage |
| `RJ_CLANG_TIDY` | `OFF` | Enable clang-tidy static analysis |
| `RJ_BUILD_BENCHMARKS` | `OFF` | Build the optional gfx950/gfx1250 benchmark workloads |
| `LTO` | `OFF` | Enable link-time optimization for Release/RelWithDebInfo |

`RJ_BUILD_BENCHMARKS` requires a ROCm SDK whose compiler supports both
`gfx950` and `gfx1250`. Install the pinned binary packages from
`benchmarks/requirements.txt` into a Python 3.12 environment and pass the
resulting SDK root as `ROCM_PATH`. The in-tree adapter links to the installed
HIP and hipBLASLt packages; CMake does not fetch or build benchmark
requirements from source. See
[benchmarking.md](benchmarking.md) for the complete workflow.

### Sanitizer builds

```bash
# AddressSanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_ENABLE_ASAN=1

# AddressSanitizer + UndefinedBehaviorSanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_ENABLE_ASAN=1 -DRJ_ENABLE_UBSAN=1

# ThreadSanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_ENABLE_TSAN=1

# MemorySanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_ENABLE_MSAN=1

# UndefinedBehaviorSanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_ENABLE_UBSAN=1
```

`AUTO` uses shared runtimes for ASan, UBSan, and TSan, and the only
supported static runtime for MSan. `SHARED` rejects MSan explicitly because
Clang does not provide a shared MSan runtime.

### Static analysis

```bash
cmake -B build -G Ninja -DRJ_CLANG_TIDY=ON
```

## Formatting

The repo uses pre-commit hooks for formatting (clang-format for C++,
black for Python, gersemi for CMake). The config is at the repo root
(`rocm-systems/.pre-commit-config.yaml`).

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Container setup for PyTorch

For running PyTorch workloads, use a persistent container with
ROCm and PyTorch pre-installed:

```bash
docker run -it --name rocjitsu-dev \
  -v $PWD:/workspace \
  rocm/pytorch:latest bash

# Inside the container, build rocjitsu and run:
cd /workspace
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
rocjitsu --daemon --config configs/gfx950_mi355x_kmd.json -- \
  python3 -c "import torch; print(torch.randn(4,4,device='cuda'))"
```
