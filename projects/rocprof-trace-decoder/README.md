# ROCprof Trace Decoder

> [!WARNING]
> This is an early preview of beta 0.2.0

- Please check the [CHANGELOG](CHANGELOG.md)
- Integration with TheRock is a work in progress.

## Description

The rocprof-trace-decoder library transforms wave (thread) trace binary data in .att files into a format that is consumable by tools. Wave (thread) trace is a profiling method that uses GPU hardware instrumentation to trace shader instructions that run on the GPU, capturing GPU occupancy, instruction run times, and other performance metrics.

### Supported devices

- AMD Radeon: 6000, 7000, 9000 series
- AMD Instinct: MI200 and MI300 series

## Building

### Quick Start

```bash
cmake -B build
cmake --build build -j$(nproc)
```

The library installs to `/opt/rocm` by default. Override with `-DCMAKE_INSTALL_PREFIX=/your/path`.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BUILD_TESTS` | `OFF` | Enable building tests (unit + integration) |
| `BUILD_UNIT_TESTS` | `ON` | Build unit tests (only effective when `BUILD_TESTS=ON`) |
| `BUILD_INTEGRATION_TESTS` | `ON` | Build integration tests (only effective when `BUILD_TESTS=ON`) |
| `USE_LLVM_DISASM` | `ON` | Use LLVM-C for AMDGPU disassembly. No `amd_comgr` / ROCm dependency. Falls back to `amd_comgr` if LLVM isn't found. |
| `LLVM_DIR` | (auto) | Path to a specific `LLVMConfig.cmake` (e.g. `/opt/rocm/llvm/lib/cmake/llvm` for the ROCm-bundled LLVM, useful for newest ASICs). Recorded in the package config so consumers automatically use the same LLVM. |
| `DISABLE_COMGR` | `OFF` | Skip the `amd_comgr` dependency. Combined with `USE_LLVM_DISASM=OFF` this builds with no disassembly backend (va2fo + symbol enumeration still work via inline ELF). Also disables the att-tool binary. |
| `CMAKE_BUILD_TYPE` | `Release` | Build type (`Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`) |

#### Disassembly backend

The library contains a header-inline disassembly path (`include/rocprof_trace_decoder/cxx/disassembly.hpp`). Three mutually-exclusive backends are selectable at configure time:

| Configuration | Backend | When to use |
|---|---|---|
| `USE_LLVM_DISASM=ON` (default) | LLVM-C | Default. Works everywhere LLVM with the AMDGPU target is available — system LLVM (`apt llvm-XX-dev`), Homebrew, ROCm-bundled (`/opt/rocm/llvm`), or a custom build. |
| `USE_LLVM_DISASM=OFF` (with ROCm) | `amd_comgr` | Legacy. Requires a ROCm install at `/opt/rocm`. |
| `USE_LLVM_DISASM=OFF DISABLE_COMGR=ON` | none | No disassembly at runtime; `va2fo` and symbol enumeration still work via inline ELF parsing. |

To build for the newest GPUs (e.g. gfx1250) before upstream LLVM ships disasm tables for them, point at the ROCm-bundled LLVM:

```bash
cmake -B build -DLLVM_DIR=/opt/rocm/llvm/lib/cmake/llvm
```

The chosen `LLVM_DIR` is baked into the generated `rocprof-trace-decoder-config.cmake`, so any consumer that does `find_package(rocprof-trace-decoder CONFIG)` automatically gets the same LLVM — no need to pass `LLVM_DIR` again on the consumer side.

### Build Targets

| Target | Description |
|---|---|
| `rocprof-trace-decoder` | Shared library (`.so`) |
| `rocprof-trace-decoder-static` | Static library (`.a`) — built but not installed; consumed via the build-tree CMake export |
| `unit_tests` | Unit test executable (requires `BUILD_TESTS=ON`) |
| `format` | Run clang-format and cmake-format on all sources |
| `docs` | Generate Doxygen API documentation |
| `coverage` | Run tests and generate code coverage report (see below) |

### Consuming from another CMake project

The build tree exports a CMake package, so downstream projects don't need to install the decoder:

```cmake
find_package(rocprof-trace-decoder REQUIRED CONFIG
             PATHS /path/to/rocprof-trace-decoder/build
             PATH_SUFFIXES source lib/cmake/rocprof-trace-decoder)

# Static link — recommended for self-contained binaries.
target_link_libraries(myapp PRIVATE rocprof-trace-decoder::rocprof-trace-decoder-static)

# Or shared:
# target_link_libraries(myapp PRIVATE rocprof-trace-decoder::rocprof-trace-decoder)
```

The namespaced target carries the disasm backend's include dirs, link libs, and compile defs as `PUBLIC` interface — consumers automatically inherit LLVM (or `amd_comgr`) without an additional `find_package` on their side. The `LLVM_DIR` used at decoder build time is recorded in the package config and reused on the consumer side.

## Testing

### Building and Running Tests

```bash
cmake -B build -DBUILD_TESTS=ON -DDISABLE_COMGR=ON
cmake --build build -j$(nproc)
cd build && ctest --test-dir test -j$(nproc)
```

Set `-DDISABLE_COMGR=ON` if `amd_comgr` is not installed. This skips the att-tool but all other tests still run.

### Running Only Unit Tests

```bash
ctest --test-dir build/test -R "regular/" -j$(nproc)
```

### Running Only Integration Tests

```bash
ctest --test-dir build/test -E "regular/|sanitize|ubsan|asan" -j$(nproc)
```

### Sanitizer Builds

```bash
ctest --test-dir build/test -R "asan/" -j$(nproc)   # AddressSanitizer
ctest --test-dir build/test -R "ubsan/" -j$(nproc)  # UBSan
```

## Code Coverage

```bash
# Requires gcov, lcov and genhtml

cmake -B build_coverage -DBUILD_TESTS=ON -DDISABLE_COMGR=ON \
    -DCMAKE_CXX_FLAGS="--coverage -fprofile-arcs -ftest-coverage" \
    -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
    -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
    -DCMAKE_BUILD_TYPE=Debug

# Build and Generate coverage report
cmake --build build_coverage -j$(nproc)
cd build_coverage && make coverage
```

## Usage with rocprofv3

```bash
rocprofv3 --att -- ./a.out
```

The decoder is linked at build time by rocprofiler-sdk. No library path argument is needed.

For information on how to generate thread trace data, see [using rocprofv3 to collect thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html)

### C API

```cpp
rocprof_trace_decoder_handle_t decoder{};
auto status = rocprof_trace_decoder_create_handle(&decoder);
```

For more information, see
* [The rocprofiler-sdk documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/api-reference/thread_trace.html)
* [The rocprofiler-sdk thread trace sample](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/samples/thread_trace/agent.cpp)

## API Versions

The library API version is controlled by `VERSION_MINOR` and reflected in the SONAME:

| VERSION_MINOR | SONAME | API |
|---|---|---|
| 1 | `librocprof-trace-decoder.so.0.1` | V1: stateless `rocprof_trace_decoder_parse_data(se_data_cb, trace_cb, isa_cb, userdata)`. Compatible with rocprofiler-sdk <= 7.13. |
| 2 (default) | `librocprof-trace-decoder.so.0.2` | V2: handle-based `rocprof_trace_decoder_parse(handle, data, size, trace_cb, userdata)` with `create_handle`, `destroy_handle`, `codeobj_load/unload`, and `set_isa_callback`. |

To build the V1 library for backwards compatibility with older rocprofiler-sdk releases:

```bash
cmake -B build -DVERSION_MINOR=1
cmake --build build -j$(nproc)
```
