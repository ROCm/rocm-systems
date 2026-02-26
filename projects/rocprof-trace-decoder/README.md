# ROCprof Trace Decoder

## This is an early preview of beta 0.1.7

- Please check the changelog for 0.1.7
- Integration with ROCk is a work in progress.

## Description

Thread trace is a profiling method that provides fine-grained insight into GPU kernel execution by collecting detailed traces of shader instructions executed by the GPU. This feature captures GPU occupancy, instruction execution times, fast performance counters, and other detailed performance data. Thread trace utilizes GPU hardware instrumentation to record events as they happen, resulting in precise timing information about wave (threads) execution behavior.

This library is responsible for transforming thread trace binary data (.att) into a tool consumable format.

## Usage as a rocprofiler-sdk plugin

### rocprofv3 tool

```bash
rocprofv3 --att -- ./a.out
```

By default, rocprofv3 searches ``LD_LIBRARY_PATH`` and the rocprofiler-sdk install location, normally ``/opt/rocm/lib``. For custom install locations, use

```bash
rocprofv3 --att --att-library-path /path/to/lib -- ./a.out
```

For information on how to generate thread trace data, see the documentation on [using rocprofv3 to collect thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html)

### Rocprofiler-sdk API

Rocprofiler-sdk requires the library path to be provided at resource creation:

```bash
rocprofiler_thread_trace_decoder_handle_t decoder{};
# Notes: Passing null string "" searches in LD_LIBRARY_PATH. Passing nullptr is not allowed.
auto status = rocprofiler_thread_trace_decoder_create(&decoder, "/opt/rocm/lib");
```

For more API information, see the [ROCm docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/api-reference/thread_trace.html), [SDK Samples](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/samples/thread_trace/agent.cpp) and [SDK API](https://github.com/ROCm/rocprofiler-sdk/tree/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace)

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
| `DISABLE_COMGR` | `OFF` | Skip the `amd_comgr` dependency (disables the att-tool binary) |
| `CMAKE_BUILD_TYPE` | `Release` | Build type (`Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`) |

### Build Targets

| Target | Description |
|---|---|
| `rocprof-trace-decoder` | Shared library (`.so`) |
| `rocprof-trace-decoder-static` | Static library (`.a`) |
| `unit_tests` | Unit test executable (requires `BUILD_TESTS=ON`) |
| `format` | Run clang-format and cmake-format on all sources |
| `docs` | Generate Doxygen API documentation |
| `coverage` | Run tests and generate code coverage report (see below) |

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
