# schema_v3 Writer/Reader Examples

## Overview

These examples exercise the public profiler-hub writer/reader API end to end against the schema_v3 rocpd layout. Each program is a self-contained round trip: a `writer()` step registers the minimal dependency set and inserts one record into a local rocpd (SQLite) database, then flushes it to disk; a `reader()` step re-opens that database, reads the record back, and prints every reader-recoverable field as a `key=value` line on stdout.

## Source Files

- `memory_alloc_writer_reader.cpp` - Writes one memory allocation. Requires only `node` + `process` (the smallest dependency set), plus a correlated `event` so the row is readable via the reader's timeline query.
- `kernel_dispatch_writer_reader.cpp` - Writes one kernel dispatch. Requires the full dependency chain: `node`, `process`, `thread`, `agent`, `queue`, `stream`, and `kernel_symbol` (which in turn requires a `code_object`).
- `CMakeLists.txt` - Builds each `*.cpp` into an executable named `profiler-hub_<file>`, links the profiler-hub library, and emits the binaries to `<build>/bin/examples`.

## Prerequisites

- CMake 3.21+
- C++17 compiler
- profiler-hub library (the in-tree `profiler-hub` target, an installed `profiler-hub::profiler-hub` package, or an existing build tree)

## Building

**As part of the main project** (default; `PROFILER_HUB_BUILD_EXAMPLES` is `ON`):

```bash
cmake -S <project_root> -B build
cmake --build build --target profiler-hub_memory_alloc_writer_reader profiler-hub_kernel_dispatch_writer_reader
```

**Standalone build** (against an existing profiler-hub build tree or install):

```bash
# Against an existing build tree
cmake -S <project_root>/examples -B examples_build -DPROFILER_HUB_BUILD_DIR=<project_root>/build
cmake --build examples_build

# Against an installed package
cmake -S <project_root>/examples -B examples_build -DCMAKE_PREFIX_PATH=<install_prefix>
cmake --build examples_build
```

The executables are placed under `<build>/bin/examples`.

**Targets:**

| Target | Description |
| ------ | ----------- |
| `profiler-hub_memory_alloc_writer_reader` | Memory-allocation round trip (node + process) |
| `profiler-hub_kernel_dispatch_writer_reader` | Kernel-dispatch round trip (full dependency chain) |

## Running

Run a binary directly; it prints the recovered fields and exits `0` on success:

```bash
./build/bin/examples/profiler-hub_memory_alloc_writer_reader
./build/bin/examples/profiler-hub_kernel_dispatch_writer_reader
```

Example output (`key=value`, one field per line):

```
type=ALLOC
level=SCRATCH
size=8192
event.correlation_id=1
node_info.hash=123456789
process_info.node_info.node_id=1
```

The scratch database is created in the directory set by the `PHUB_INTEGRATION_TMP_DIR` compile definition (the target's build directory) and removed on exit.

## Validating with pytest

These binaries are the launchers for the pytest integration suite. The Python tests run a binary, parse its `key=value` output into a dict, and assert every field against Python-owned expected values:

```bash
# from the project root (after building the examples)
python -m pytest tests/integration/ -v
```

The harness locates a binary as `profiler-hub_<name>` under `<repo>/build/bin/examples` (override with `PHUB_EXAMPLE_BIN_DIR`). If a binary is missing the corresponding tests fail.

| Test file | Drives |
| --------- | ------ |
| `tests/integration/test_memory_alloc.py` | `profiler-hub_memory_alloc_writer_reader` |
| `tests/integration/test_kernel_dispatch.py` | `profiler-hub_kernel_dispatch_writer_reader` |

## Adding a new example

1. Add `my_record_writer_reader.cpp` here, following the `writer()` / `reader()` / `print()` structure of the existing files. Register only the dependencies the record's validator requires.
2. Append the base name (without `.cpp`) to `PROFILER_HUB_SCHEMA_V3_EXAMPLES` in `CMakeLists.txt`.
3. Add `tests/integration/test_my_record.py` with the expected `key=value` fields and have it call `run_launcher("my_record_writer_reader")`.
