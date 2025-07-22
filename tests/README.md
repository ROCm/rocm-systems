# ROCm Systems Profiler — Standalone Test Suite

This repository enables users to build and execute the test suite for the ROCm Systems Profiler independently using the packaged binaries installed with `ROCm`.

## Building and running the Test Suite

```sh
cmake -B rocprof-sys-test -S tests
cmake --build rocprof-sys-test --target all --parallel 8
```

The suite of tests can then be ran using the following command:

```sh
ctest --test-dir rocprof-sys-test
```

## Test Prerequisites

The standalone test suite requires the following for a minimal build:

- `ROCm` installed on your system
- `rocprofiler-systems` package
- `Python3` along with `pybind11-dev`

### Python Tests

The python suite of ctests requires the `rocprofsys` module. Should you need to disable all python dependencies, use the flag `-D ROCPROFSYS_USE_PYTHON=OFF`.

### Perfetto Validation Tests

The perfetto validation tests require the python `perfetto` module to be installed.

### rocDecode and rocJPEG Tests

These tests respectively require the `rocdecode-dev` and `rocjpeg-dev` packages.

### Overflow Tests

This family of tests require that the system's `perf_event_paranoid` be `<= 3`.
