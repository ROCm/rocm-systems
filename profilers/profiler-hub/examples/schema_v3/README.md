# schema_v3 writer examples

## Overview

These examples exercise the public profiler-hub writer API against the schema_v3 rocpd layout. Each program is a **writer launcher** for the pytest driven integration suite. This keeps the C++ side free of validation logic.

Each launcher takes exactly one argument, the database path:

```bash
profiler-hub_<name> <db_path>
```

It removes any pre-existing file at `<db_path>`, writes its record, flushes, and prints `db_path=<db_path>` on success (exit `0`). On a usage error it exits `2`; on a writer exception it exits `1`.

## Source Files

- `memory_alloc_writer.cpp` - Writes one memory allocation. Registers `node`, `process`, `thread`, `agent`, `queue`, and `stream`, plus a correlated `event`.
- `memory_copy_writer.cpp` - Writes one memory copy between two agents. Registers `node`, `process`, `thread`, a source and a destination `agent`, `queue`, and `stream`.
- `kernel_dispatch_writer.cpp` - Writes one kernel dispatch. Registers the full dependency chain: `node`, `process`, `thread`, `agent`, `queue`, `stream`, `code_object`, and `kernel_symbol`.
- `pmc_event_writer.cpp` - Writes one PMC event. Registers `node`, `process`, `thread`, `agent`, `track`, and `pmc_info`, plus a correlated `event` and a `sample`.
- `region_writer.cpp` - Writes one region with one argument. Registers `node`, `process`, `thread`, and `track`.
- `CMakeLists.txt` - Builds each example listed in `PROFILER_HUB_SCHEMA_V3_EXAMPLES` into an executable named `profiler-hub_<name>`, links the profiler-hub library, and emits the binaries to `<build>/bin/examples`. When tests are enabled it also generates CTest definitions from the pytest suite.

## Prerequisites

- CMake 3.21+
- C++17 compiler
- profiler-hub library (the in-tree `profiler-hub` target, an installed `profiler-hub::profiler-hub` package, or an existing build tree)

## Building

**As part of the main project** (built together with the tests; `PROFILER_HUB_BUILD_TESTS` is `ON` by default):

```bash
cmake -S <project_root> -B build
cmake --build build
```

To build a single example, target it by name, e.g.:

```bash
cmake --build build --target profiler-hub_pmc_event_writer
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

| Target | Record written | Registered dependencies |
| ------ | -------------- | ----------------------- |
| `profiler-hub_memory_alloc_writer` | Memory allocation | node, process, thread, agent, queue, stream |
| `profiler-hub_memory_copy_writer` | Memory copy | node, process, thread, src+dst agent, queue, stream |
| `profiler-hub_kernel_dispatch_writer` | Kernel dispatch | node, process, thread, agent, queue, stream, code_object, kernel_symbol |
| `profiler-hub_pmc_event_writer` | PMC event | node, process, thread, agent, track, pmc_info |
| `profiler-hub_region_writer` | Region | node, process, thread, track |

## Running

Run a binary directly, passing the database path to write. It creates the database, prints the path, and exits `0` on success:

```bash
./build/bin/examples/profiler-hub_memory_alloc_writer /tmp/memory_alloc.db
./build/bin/examples/profiler-hub_pmc_event_writer /tmp/pmc_event.db
```

Output is a single `key=value` line echoing the database path:

```
db_path=/tmp/memory_alloc.db
```

The launcher does not delete the database it writes; the pytest suite owns cleanup of the files it requests.

## Validating with pytest

These binaries are the launchers for the pytest integration suite. The Python tests run a launcher with a chosen SQLite path, then open that database directly and assert every persisted field against Python-owned expected values:

```bash
# from the project root (after building the examples)
python -m pytest tests/integration/ -v
```

The harness locates a binary as `profiler-hub_<name>` under `<repo>/build/bin/examples` (override with `PHUB_EXAMPLE_BIN_DIR`). If a binary is missing the corresponding tests fail rather than silently skipping. See `tests/integration/README.md` for full details.

| Test file | Drives |
| --------- | ------ |
| `tests/integration/test_memory_alloc.py` | `profiler-hub_memory_alloc_writer` |
| `tests/integration/test_memory_copy.py` | `profiler-hub_memory_copy_writer` |
| `tests/integration/test_kernel_dispatch.py` | `profiler-hub_kernel_dispatch_writer` |
| `tests/integration/test_pmc_event.py` | `profiler-hub_pmc_event_writer` |
| `tests/integration/test_region.py` | `profiler-hub_region_writer` |

## Adding a new example

1. Add `my_record_writer.cpp` here, following the `make_*()` / `writer()` / `main()` structure of the existing files. The launcher must accept one `db_path` argument, register only the dependencies the record's validator requires, insert the record, flush to disk, and print `db_path=<path>`.
2. Append the base name (without `.cpp`) to `PROFILER_HUB_SCHEMA_V3_EXAMPLES` in `CMakeLists.txt`.
3. Add or extend a `ProfilerHubDb.read_<record>_info()` helper in `tests/integration/profiler_hub_db.py` that reads the persisted rows into a flat `dict`.
4. Add `tests/integration/test_my_record.py` with an `EXPECTED` dict, have it call `run_launcher_db("my_record_writer", db_path)`, and assert the DB readback against `EXPECTED`.
