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
with tests enabled (`PROFILER_HUB_BUILD_TESTS`), the
`generate-profiler-hub-pytest-ctests` target runs `generate_ctest.py`, which
collects the pytest modules and emits one CTest entry per `test_*.py` file
(named `integration.<module>`). After building, run them from the build
directory:

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

If `PHUB_EXAMPLE_BIN_DIR` is unset, the harness looks under `<repo>/build/bin/examples` and then falls back to a repo-wide search.

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

1. `conftest.py` exposes a `run_launcher_db(name, db_path)` fixture that locates
   `profiler-hub_<name>`, runs it with the requested SQLite path, checks the exit
   code, and verifies the launcher printed the same `db_path`.
2. `profiler_hub_db.py` opens the generated database and reads the relevant
   schema_v3 tables into a flat `dict`.
3. Each `test_*.py` defines an `EXPECTED` dict of persisted values and asserts
   the database readback against it.
4. `generate_ctest.py` (run by the CMake `generate-profiler-hub-pytest-ctests`
   target) collects the suite in `--ctest-mode=generate` and writes a
   `CTestTestfile.cmake` so each module is also runnable through `ctest`.

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
