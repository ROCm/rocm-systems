# rocpdsna

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
