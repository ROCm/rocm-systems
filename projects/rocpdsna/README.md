# rocpdsna

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build & Test](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-ci.yml/badge.svg?branch=sna-develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-ci.yml?query=branch%3Asna-develop)
[![Sanitizers](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-sanitizers.yml/badge.svg?branch=sna-develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-sanitizers.yml?query=branch%3Asna-develop)
[![Static Analysis](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-static-analysis.yml/badge.svg?branch=sna-develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-static-analysis.yml?query=branch%3Asna-develop)
[![Coverage](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-coverage.yml/badge.svg?branch=sna-develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocpdsna-coverage.yml?query=branch%3Asna-develop)

A C++ library for storing and retrieving ROCm profiling data using SQLite (rocpd database format).

## Overview

**rocpdsna** provides a high-performance storage layer for ROCm profiling tools. It offers a structured way to persist profiling data in the rocpd (SQLite) database format, enabling interoperability with ROCm profiling tools and analysis workflows.

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
| `ROCPDSNA_BUILD_TESTS` | ON | Build unit tests |
| `ROCPDSNA_BUILD_BENCHMARKS` | ON | Build performance benchmarks |
| `ROCPDSNA_ENABLE_LOGGING` | OFF | Enable debug logging |
| `ROCPDSNA_ENABLE_COVERAGE` | OFF | Enable code coverage instrumentation (requires Debug build, gcov, and lcov or gcovr) |

## Installation

```bash
cmake --install build --prefix /opt/rocm
```

## Usage

### Linking with CMake

For projects using an installed rocpdsna:

```cmake
find_package(rocpdsna REQUIRED)
target_link_libraries(your_target PRIVATE rocpdsna::rocpdsna)
```

<!-- ## Benchmark -->

<!-- // TODO -->
<!-- | Benchmark           | Description                   | Time (ns) | -->
<!-- |---------------------|-------------------------------|-----------| -->

## License

MIT License — Copyright (c) 2025 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for details.
