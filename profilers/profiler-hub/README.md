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
- C++20 compatible compiler
- SQLite3 (bundled via CMake module)
- spdlog and fmt (system packages or FetchContent fallback)

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

At configure time, profiler-hub clones RocPD schema SQL from `rocm-systems` and
embeds it as generated C++ headers. No installed `rocprofiler-sdk-rocpd` package
is required.

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
| `PROFILER_HUB_FETCH_DEPENDENCIES` | OFF | Download missing dependencies at configure time. Required on Windows when SQLite3 is not already on `CMAKE_PREFIX_PATH` |
| `SQLITE3_AMALGAMATION_YEAR` | `2024` | sqlite.org release-year folder for the Windows amalgamation download when bumping `SQLITE3_GIT_TAG` |

## Installation

```bash
cmake --install build --prefix /opt/rocm
```

## Usage in Linux

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

## Windows (MSVC)

A native MSVC build is supported (CI runs on Linux only). fmt, spdlog, nlohmann_json, GoogleTest, and Google Benchmark are fetched when they are not already installed. SQLite3 is fetched only when `-DPROFILER_HUB_FETCH_DEPENDENCIES=ON` is set.

### Requirements (Windows)

| Tool | Notes |
|------|-------|
| **Visual Studio 2022 or 2026 Build Tools** | Install the **Desktop development with C++** workload (MSVC compiler + Windows SDK). Builds as **C++20**. |
| **CMake 3.25+** | Bundled with Visual Studio, or install standalone: `winget install -e --id Kitware.CMake` |
| **Git for Windows** | Used to clone the RocPD schema. SQLite on Windows is downloaded from sqlite.org (no git clone for SQLite). |

### Build (Windows)

1. **Install the Visual Studio C++ toolchain** (skip if already installed). You only need
   the Build Tools — the full Visual Studio IDE is not required.

   - Download from <https://visualstudio.microsoft.com/downloads/> : scroll to
     *Tools for Visual Studio*  **Build Tools for Visual Studio 2022** (or 2026), or the
     full **Community** edition.
   - Or install from the command line with winget:

   ```cmd
   winget install -e --id Microsoft.VisualStudio.2022.BuildTools
   ```

   - In the installer, select the **Desktop development with C++** workload (this pulls in
     the MSVC compiler, CMake, and the Windows SDK), then click **Install**.

2. **Open a Visual Studio developer shell** so `cmake` and `cl.exe` are on `PATH`.

   The simplest way is to open the Start menu and launch **x64 Native Tools Command
   Prompt for VS**  this sets up the environment for you.

   Alternatively, from any `cmd` window, run the `VsDevCmd.bat` that ships with your
   install. Its path depends on the Visual Studio *version number* (2022 = `2022`,
   2026 = `2026`) and the *edition* (`BuildTools`, `Community`, `Professional`, or
   `Enterprise`):

   ```cmd
   "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
   ```

3. **Configure and build** from the `profilers/profiler-hub` directory:

   ```cmd
   cd C:\path\to\rocm-systems\profilers\profiler-hub
   cmake -S . -B build -DPROFILER_HUB_FETCH_DEPENDENCIES=ON
   cmake --build build --config Release --parallel
   ```

4. **Run the tests** (optional):

   ```cmd
   ctest --test-dir build -C Release --output-on-failure
   ```

### Build artifacts

With the Visual Studio generator, artifacts live under a configuration subfolder
(`Release`/`Debug`), not directly under `build\lib\`:

| Artifact | Path (Release) | Description |
|----------|----------------|-------------|
| Shared library (DLL) | `build\bin\Release\profiler-hub.dll` | Loaded at run time |
| DLL import library | `build\lib\Release\profiler-hub.lib` | Link stub for the shared target `profiler-hub::profiler-hub` |
| Static archive | `build\lib\Release\profiler-hub-static.lib` | Full static lib for `profiler-hub::profiler-hub-static` |
| Fetched SQLite DLL | `build\bin\Release\profiler-hub-sqlite3.dll` | Separate SQLite library used by `profiler-hub.dll` |
| SQLite import library | `build\lib\Release\profiler-hub-sqlite3.lib` | Link stub for `profiler-hub-sqlite3.dll` |
| Public headers | `build\include\profiler-hub\` | Generated/copied API headers |
| CMake package | `build\profiler-hub-config.cmake` | Use from the build tree without installing |

On Windows a `.lib` can be either an import library or a static archive, so the two
profiler-hub targets use distinct base names: `profiler-hub.lib` (import lib, pairs with the DLL) vs
`profiler-hub-static.lib` (static archive).

### Windows-specific behavior

- **SQLite3:** with `PROFILER_HUB_FETCH_DEPENDENCIES=ON`, Windows downloads the official
  [amalgamation zip](https://www.sqlite.org/download.html) matching `SQLITE3_GIT_TAG`
  (default `version-3.45.3`, year folder `SQLITE3_AMALGAMATION_YEAR`). It is built as
  `profiler-hub-sqlite3.dll`, which exports the SQLite C API. Keep that DLL next to
  `profiler-hub.dll` (or on `PATH`) at run time.
- **Dependencies:** fmt, spdlog, nlohmann_json, GoogleTest, and Google Benchmark are
  fetched via CMake `FetchContent` when not found on the system.
- **Language standard:** library, tests, and benchmarks are built as **C++20**.

### Install (Windows)

```cmd
cmake --install build --config Release --prefix C:\opt\profiler-hub
```

Installed layout: `bin\profiler-hub.dll`, `lib\profiler-hub.lib` (import lib),
`lib\profiler-hub-static.lib` (static archive), `include\profiler-hub\*.hpp`, and the
`find_package` config under `lib\cmake\profiler-hub\`. When SQLite was fetched, also
ship `profiler-hub-sqlite3.dll` beside `profiler-hub.dll`.

### Usage (Windows)

For projects using an installed profiler-hub:

```cmake
find_package(profiler-hub REQUIRED)
target_link_libraries(your_target PRIVATE profiler-hub::profiler-hub)
```

`examples/installed-client` is a small program that uses that pattern to write one region and read it back. From that directory, after installing profiler-hub:

```cmd
cmake -S . -B build -DCMAKE_PREFIX_PATH=C:\opt\profiler-hub
cmake --build build --config Release
build\Release\writer-reader-example.exe
```

Point the consumer at the package with `-DCMAKE_PREFIX_PATH=C:\opt\profiler-hub` or
`-Dprofiler-hub_DIR=<build-tree>` to use the build tree directly. Ensure
`profiler-hub.dll` and, when SQLite was fetched, `profiler-hub-sqlite3.dll` are next
to the consumer executable or on `PATH` at run time.

## License

MIT License — Copyright (c) 2025 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for details.
