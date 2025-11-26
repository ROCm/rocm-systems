# rocstorage

A C++ library for storing and retrieving ROCm profiling data using SQLite.

## Overview

**rocstorage** provides a high-performance storage layer for ROCm profiling tools. It offers a structured way to persist profiling data.

## Requirements

- CMake 3.21+
- C++17 compatible compiler
- SQLite3 (bundled via CMake module)
- Optional: `rocprofiler-sdk-rocpd` for schema compatibility

## Building

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

## Installation

```bash
cmake --install build --prefix /opt/rocm
```

## Usage

### Linking with CMake

```cmake
find_package(rocstorage REQUIRED)
target_link_libraries(your_target PRIVATE rocstorage::rocstorage)
```

## License

MIT License — Copyright (c) 2025 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for details.
