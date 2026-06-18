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
| `RJ_SANITIZER` | *(empty)* | Enable a sanitizer: `asan`, `ubsan`, `tsan`, or `msan` |
| `RJ_CLANG_TIDY` | `OFF` | Enable clang-tidy static analysis |
| `LTO` | `OFF` | Enable link-time optimization for Release/RelWithDebInfo |

### Sanitizer builds

```bash
# AddressSanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_SANITIZER=asan

# ThreadSanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_SANITIZER=tsan

# UndefinedBehaviorSanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_SANITIZER=ubsan
```

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
rocjitsu --daemon --config configs/amdgpu_cdna4_kmd.json -- \
  python3 -c "import torch; print(torch.randn(4,4,device='cuda'))"
```
