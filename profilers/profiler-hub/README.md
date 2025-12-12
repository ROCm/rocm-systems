# profiler-hub

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build & Test](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-ci.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-ci.yml?query=branch%3Adevelop)
[![Sanitizers](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-sanitizers.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-sanitizers.yml?query=branch%3Adevelop)
[![Static Analysis](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-static-analysis.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-static-analysis.yml?query=branch%3Adevelop)
[![Coverage](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-coverage.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/profiler-hub-coverage.yml?query=branch%3Adevelop)

A C++ library for storing and retrieving ROCm profiling data using SQLite (rocpd database format).

## Overview

**profiler-hub** provides a high-performance storage layer for ROCm profiling tools. It offers a structured way to persist profiling data in the rocpd (SQLite) database format, enabling interoperability with ROCm profiling tools and analysis workflows.

This library is part of the [rocm-systems](https://github.com/ROCm/rocm-systems) monorepo and is used by rocprofiler-systems for trace data output.

## Requirements

- CMake 3.21+
- C++17 compatible compiler
- SQLite3 (bundled via CMake module)
- spdlog (for logging)
- Optional: `rocprofiler-sdk-rocpd` for schema compatibility

### System Package Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install libsqlite3-dev libspdlog-dev libfmt-dev
```

**RHEL/Rocky Linux:**
```bash
sudo dnf install sqlite-devel spdlog-devel fmt-devel
```

**openSUSE:**
```bash
sudo zypper install sqlite3-devel spdlog-devel fmt-devel
```

## Building

### Standalone Build

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `PROFILER_HUB_BUILD_TESTS` | ON | Build unit tests |
| `PROFILER_HUB_BUILD_BENCHMARKS` | ON | Build performance benchmarks |
| `PROFILER_HUB_ENABLE_LOGGING` | OFF | Enable debug logging |
| `PROFILER_HUB_ENABLE_COVERAGE` | OFF | Enable code coverage instrumentation (requires Debug build, gcov, and lcov or gcovr) |

## Installation

```bash
cmake --install build --prefix /opt/rocm
```

## Usage

### Linking with CMake

For projects using an installed profiler-hub:

```cmake
find_package(profiler-hub REQUIRED)
target_link_libraries(your_target PRIVATE profiler-hub::profiler-hub)
```

<!-- ## Benchmark -->

<!-- // TODO -->
<!-- | Benchmark           | Description                   | Time (ns) | -->
<!-- |---------------------|-------------------------------|-----------| -->

## License

MIT License — Copyright (c) 2025 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for details.
