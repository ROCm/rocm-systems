---
myst:
    html_meta:
        "description": "Install and build rocJITsu, the AMD ROCm Just-in-Time Suite for GPU emulation, binary translation, and instrumentation."
        "keywords": "rocJITsu, ROCm, install, build, CMake, GPU emulation, binary translation"
---
# Install and build rocJITsu

## Prerequisites

rocJITsu requires:

-   **CMake** 3.22 or later
-   **C++20 compiler**: GCC 13 or later, or Clang 16 or later
-   **pthreads** (found automatically by CMake `find_package(Threads)`)

Optional prerequisites:

-   **Python 3.10 or later** --- required only for ISA code generation
    with the `amdisa` pipeline. See
    [Regenerate ISA and DBT source files](../how-to/regenerate-isa-codegen.md)
    for details.
-   **AMD ROCm toolchain** (`hipcc`, `libhsa-runtime64`) --- required
    only for HIP kernel tests and daemon-mode tests. When the ROCm
    toolchain is not found, those tests are disabled automatically.

Third-party dependencies (Google Test v1.15.2 and FlatBuffers v24.3.25)
are fetched automatically through CMake `FetchContent`. No manual
download is needed.

## Building from source

### Source download

rocJITsu lives in the `rocm-systems` super-repository. Use a sparse
checkout to download only the rocJITsu project:

``` shell
git clone --filter=blob:none --sparse \
    https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout set projects/rocjitsu
```

### Build commands

Configure and build with CMake and Ninja:

``` shell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the test suite:

``` shell
ctest --test-dir build
```

#### `ROCM_PATH` configuration

CMake resolves the ROCm installation path in this order:

1.  The `ROCM_PATH` CMake variable, if set explicitly.
2.  The `ROCM_PATH` environment variable, if set.
3.  `/opt/rocm`, if the directory exists.

When `ROCM_PATH` is set explicitly (by variable or environment) and the
directory does not exist, the configure step fails with an error.

#### Install prefix

When `CMAKE_INSTALL_PREFIX` is not set by the user and `ROCM_PATH` is
available, the install prefix defaults to the value of `ROCM_PATH`. This
matches the convention used by other AMD ROCm projects.

#### Install targets

After building, install with:

``` shell
cmake --install build
```

This installs:

-   Public headers to `<prefix>/include/rocjitsu/`
-   Shared libraries (`librocjitsu.so`, `librocjitsu_hooks.so`,
    `librocjitsu_kmd.so` on Linux) to `<prefix>/lib/`
-   The `rocjitsu` CLI binary to `<prefix>/bin/` (Linux only)
-   GPU topology JSON configuration files to
    `<prefix>/share/rocjitsu/configs/`
-   FlatBuffers schema files to `<prefix>/share/rocjitsu/schemas/`

#### Third-party dependencies

Google Test and FlatBuffers are fetched into a `third_party/` directory
inside the source tree by default. To relocate the download directory
(useful in container builds or CI), pass
`-DFETCHCONTENT_BASE_DIR=/path/to/deps`.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `RJ_SANITIZER` | *(empty)* | Enable a sanitizer build. Accepted values: `asan`, `ubsan`, `tsan`, `msan`. |
| `RJ_CLANG_TIDY` | `OFF` | Enable clang-tidy static analysis during the build. |
| `LTO` | `OFF` | Enable link-time optimization (IPO) for `Release` and `RelWithDebInfo` configurations. Incompatible with `RJ_SANITIZER`; setting both causes a fatal error. |
| `RJ_ENABLE_EXPENSIVE_CHECKS` | `OFF` | Enable expensive exhaustive test suites such as MFMA and WMMA SIMD bit-exactness checks. |
| `BUILD_TESTING` | `ON` | Standard CMake option. Set to `OFF` to skip building the test suite. |

### Sanitizer builds

rocJITsu supports four sanitizer modes. Each is activated by passing the
corresponding value to `RJ_SANITIZER`. Sanitizer builds are incompatible
with `LTO`.

**AddressSanitizer (ASan)**

``` shell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_SANITIZER=asan
cmake --build build
```

**UndefinedBehaviorSanitizer (UBSan)**

``` shell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_SANITIZER=ubsan
cmake --build build
```

**ThreadSanitizer (TSan)**

``` shell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_SANITIZER=tsan
cmake --build build
```

**MemorySanitizer (MSan)**

``` shell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRJ_SANITIZER=msan
cmake --build build
```

### Clang-tidy setup

Enable clang-tidy static analysis by setting `RJ_CLANG_TIDY=ON`. The
build system locates `clang-tidy` on `PATH` and, if found, sets
`CMAKE_CXX_CLANG_TIDY` so that every C++ compile invocation runs the
checker. If `clang-tidy` is not found, the option is silently ignored
with a warning.

``` shell
cmake -B build -G Ninja -DRJ_CLANG_TIDY=ON
cmake --build build
```

The build also exports `compile_commands.json`
(`CMAKE_EXPORT_COMPILE_COMMANDS` is always `ON`), which `clang-tidy` and
other tooling can consume for standalone analysis.

### Pre-commit formatting hooks

The repository uses pre-commit hooks for formatting: `clang-format` for
C++, `black` for Python, and `gersemi` for CMake files. The
configuration lives at the repository root
(`rocm-systems/.pre-commit-config.yaml`).

``` shell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Container setup for PyTorch workloads

To run PyTorch workloads under rocJITsu, use a container with AMD ROCm
and PyTorch pre-installed:

``` shell
docker run -it --name rocjitsu-dev \
  -v $PWD:/workspace \
  rocm/pytorch:latest bash
```

Inside the container, build rocJITsu and launch a workload in daemon
mode:

``` shell
cd /workspace
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
rocjitsu --daemon --config configs/amdgpu_cdna4_kmd.json -- \
  python3 -c "import torch; print(torch.randn(4,4,device='cuda'))"
```

For additional information about the `rocjitsu` CLI modes, see
[rocjitsu CLI reference](../reference/rocjitsu-cli.md). For JSON
topology configuration, see
[JSON topology configuration](../conceptual/json-configuration.md).
