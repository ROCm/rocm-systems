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

## Track-Based Reader API

profiler-hub exposes a type-aware reader API for retrieving profiling events by track. Eight track types are defined (`cpu_thread`, `gpu_queue`, `dma`, `counter`, `stream`, `memory`, `kernel_dispatch_pmc`, `memory_activity`), each fully described by `track_info_t` without side queries. Three methods cover the full event surface:

- `get_interval_track(track_id)` — interval events with correct per-track scoping and pre-computed nesting level/parent
- `get_scalar_track(track_id)` — scalar/PMC sample events with deterministic per-metric resolution
- `get_flows()` — causal flow edges between events (CPU→GPU correlations)

Both rocpd v3 and v4.0 schema backends are supported.

## License

MIT License — Copyright (c) 2025 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for details.
