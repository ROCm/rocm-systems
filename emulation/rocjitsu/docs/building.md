# Building

## Prerequisites

- CMake 3.22+ (3.28+ when `ROCJITSU_ENABLE_VFIO=ON`)
- C++20 compiler (GCC 13+, Clang 16+)
- Python 3.10+ (for ISA code generation and the VFIO guest launcher)
- `runtimes/api-headers` from the same source checkout (HSA and Linux UAPI headers)
- ROCm toolchain (optional, for HIP test kernels and daemon tests)

When VFIO is enabled, configuration fails immediately on CMake older than 3.28;
the non-VFIO build keeps the repository-wide 3.22 minimum.

Third-party dependencies (Google Test, FlatBuffers) are fetched
automatically via CMake `FetchContent`.

## Quick start

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

When building rocjitsu outside the super-repository, pass
`-DRJ_RUNTIME_API_HEADERS_SOURCE_DIR=/path/to/runtimes/api-headers` at
configuration. Keep that header tree at a matching revision.

## CMake options

| Option | Default | Description |
|---|---|---|
| `ROCJITSU_HOST_CPU_BASELINE` | `x86-64-v3` on Linux x86-64 with GCC/Clang; `default` otherwise | Host CPU baseline: `default`, `x86-64`, `x86-64-v3`, or `x86-64-v4` |
| `RJ_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer |
| `RJ_ENABLE_UBSAN` | `OFF` | Enable UndefinedBehaviorSanitizer |
| `RJ_UBSAN_VPTR` | `OFF` with GCC; `ON` otherwise | Enable UBSan's dynamic-type (vptr) check when `RJ_ENABLE_UBSAN` is on |
| `RJ_ENABLE_TSAN` | `OFF` | Enable ThreadSanitizer |
| `RJ_ENABLE_MSAN` | `OFF` | Enable MemorySanitizer |
| `RJ_SANITIZER_RUNTIME` | `AUTO` | Select `AUTO`, `SHARED`, or `STATIC` sanitizer runtime linkage |
| `RJ_CLANG_TIDY` | `OFF` | Enable clang-tidy static analysis |
| `RJ_RUNTIME_API_HEADERS_SOURCE_DIR` | `../../runtimes/api-headers` | Source directory of the shared runtime API headers |
| `LTO` | `OFF` | Enable link-time optimization for Release/RelWithDebInfo |
| `ROCJITSU_ENABLE_VFIO` | `OFF` | Build Linux VFIO-user support; requires CMake 3.28+ and Linux 6.1+ UAPI headers |

### Host CPU baseline

By default, Linux x86-64 binaries built with GCC or Clang require x86-64-v3
CPU and OS support. To build for older x86-64 hosts:

```bash
cmake -B build -G Ninja -DROCJITSU_HOST_CPU_BASELINE=x86-64
```

Use `ROCJITSU_HOST_CPU_BASELINE=default` to retain your compiler or toolchain's
CPU settings. Explicit x86-64 baselines are supported only for Linux x86-64
targets.

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

GCC UBSan builds disable dynamic-type (`vptr`) checks by default because
contention in the sanitizer's shared type cache can slow multithreaded emulation.
Add `-DRJ_UBSAN_VPTR=ON` to enable those checks. Clang UBSan builds default to
`ON` and explicitly enable `vptr`, which Clang excludes from
`-fsanitize=undefined`. Use `-DRJ_UBSAN_VPTR=OFF` to disable it. The option has
no effect when `RJ_ENABLE_UBSAN` is off, and CMake reports the effective `vptr`
setting when UBSan is enabled.

`AUTO` uses shared runtimes for ASan, UBSan, and TSan, and the only
supported static runtime for MSan. `SHARED` rejects MSan explicitly because
Clang does not provide a shared MSan runtime.

Standalone plugin modules omit `vptr` checks because they do not link the RTTI
of every class their code checks. The gfx1250 B0-to-A0 translator library omits
them only from its CDNA4-to-CDNA3 lowering, for the same reason. When
`RJ_UBSAN_VPTR=ON`, the build compiles separate objects for that code, so the
objects linked into the simulator and unit tests keep those checks.

With `vptr` checks on, the UBSan runtime briefly opens pipes on emulator threads,
so interposer tests that depend on specific file descriptor numbers or exhaust
the descriptor table are not registered. They run in unsanitized builds and in
UBSan builds with `RJ_UBSAN_VPTR=OFF`, including GCC's default.

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
