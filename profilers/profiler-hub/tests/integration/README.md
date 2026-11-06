# profiler-hub Integration Test Suite

## Overview

This is a pytest suite that validates profiler-hub writer output persisted to
SQLite. It drives the example binaries (`examples/schema_v3/*`) as black boxes:
each example writes one deterministic record to a Python-provided rocpd database
path and prints that path as `db_path=<path>`. The pytest files then read the
database directly with `profiler_hub_db.py` and assert the persisted fields
against Python-owned expected values.

## General Use

### Setup

A minimum Python version of 3.8 is required. Use of a virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Building the example binaries

The tests run the compiled examples, so build them first (they are built by the
main project when `PROFILER_HUB_BUILD_TESTS` is `ON`, which is the default):

```bash
cd <path to profiler-hub>
cmake -S . -B build
cmake --build build
```

The binaries are placed under `<build>/bin/examples`.

### Running Tests

The suite uses pytest as both the framework and the executor. From the project root:

```bash
python -m pytest tests/integration/ -v
```

You can also run a single file or filter by field:

```bash
python -m pytest tests/integration/test_kernel_dispatch.py -v
python -m pytest tests/integration/ -k "node_info.hash" -v
```

If an example binary cannot be located, the corresponding tests fail (they do not
silently skip), with a message pointing at the expected location.

Note: if pytest picks up the wrong Python, invoke it explicitly, e.g.
`/path/to/.venv/bin/python -m pytest tests/integration/`.

### Running via CTest

The build also registers this suite with CTest. When the project is configured
with tests enabled (`PROFILER_HUB_BUILD_TESTS`) and pytest is importable, the
`generate-profiler-hub-pytest-ctests` target runs pytest with
`--ctest-mode=generate` and writes one CTest entry per `test_*.py` file
(named `integration.<module>`) into the integration build directory. A
configure without pytest skips those CTest entries and still builds the example
binaries. After building, run them from the build directory:

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R integration -V
```

The generated CTest entries invoke pytest on each module with
`PHUB_EXAMPLE_BIN_DIR` pointed at the built example binaries, so no manual
environment setup is required.

### Environment Variables

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `PHUB_EXAMPLE_BIN_DIR` | Directory containing the `profiler-hub_<name>` example binaries | `<repo>/build/bin/examples` |

If `PHUB_EXAMPLE_BIN_DIR` is unset, the harness looks under `<repo>/build/bin/examples`.

### Per-test timeout

A test file can set its own CTest timeout with the `timeout` marker; otherwise it
defaults to 180 seconds.

```python
import pytest

# Whole module:
pytestmark = pytest.mark.timeout(300)

# Or a single test:
@pytest.mark.timeout(300)
def test_something():
    ...
```

The generated CTest entry for a module uses the largest `timeout` any of its
tests requests.

## How It Works

1. `conftest.py` exposes `run_launcher_db(name, db_path)` and
   `read_example_db(example, reader, filename)`. The launcher fixture locates
   `profiler-hub_<name>`, runs it with the requested SQLite path, checks the exit
   code, and verifies the launcher printed the same `db_path`. The database stays
   in the pytest temporary directory.
2. `profiler_hub_db.py` opens the generated database and reads the relevant
   schema_v3 tables into a flat `dict`. Node, process, thread, and schema
   metadata that every example shares live in `COMMON_METADATA_EXPECTED`.
3. Each `test_*.py` defines an `EXPECTED` dict of the record-specific values,
   merges the shared metadata, and asserts the database readback against it.
4. Building `generate-profiler-hub-pytest-ctests` runs pytest with
   `--ctest-mode=generate`, the same step rocprofiler-systems uses. Pytest
   writes `tests/integration/CTestTestfile.cmake` in the build tree. A stamp
   file is the build output, so that write does not replace the unit-test list.

## Test Files

| Test file | Drives example binary |
| --------- | --------------------- |
| `test_kernel_dispatch.py` | `profiler-hub_kernel_dispatch_writer` |
| `test_memory_alloc.py` | `profiler-hub_memory_alloc_writer` |
| `test_memory_copy.py` | `profiler-hub_memory_copy_writer` |
| `test_pmc_event.py` | `profiler-hub_pmc_event_writer` |
| `test_region.py` | `profiler-hub_region_writer` |

## Adding a Test

1. Add the corresponding writer-only example under `examples/schema_v3/` (see
   that directory's `README.md`). The example should accept one `db_path`
   argument, write deterministic data, flush it, and print `db_path=<path>`.
2. Add or extend a `ProfilerHubDb.read_<record>_info()` helper that reads the
   persisted schema_v3 rows into a flat `dict`.
3. Create `tests/integration/test_<record>.py` with an `EXPECTED` dict of the
   values read from the database.
4. Have the test call `run_launcher_db("<record>_writer", db_path)` and assert
   the DB readback against `EXPECTED`.
