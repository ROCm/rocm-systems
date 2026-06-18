# profiler-hub Integration Test Suite

## Overview

This is a pytest suite that validates the profiler-hub writer/reader round trip end to end. It does not contain any C++ of its own: it drives the  example binaries (`examples/*`) as black boxes. Each example writes one record to a local rocpd database, reads it back, and prints the recovered fields as `key=value` lines on stdout. The pytest files own the expected values and assert every field against the parsed output, so the C++ side stays free of test logic.

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
main project when `PROFILER_HUB_BUILD_EXAMPLES` is `ON`, which is the default):

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

### Environment Variables

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `PHUB_EXAMPLE_BIN_DIR` | Directory containing the `profiler-hub_<name>` example binaries | `<repo>/build/bin/examples` |

If `PHUB_EXAMPLE_BIN_DIR` is unset, the harness looks under `<repo>/build/bin/examples` and then falls back to a repo-wide search.

## How It Works

1. `conftest.py` exposes a `run_launcher(name)` fixture that locates `profiler-hub_<name>`, runs it, checks the exit code, and parses its `key=value` stdout into a flat `dict` (splitting each line on the first `=`).
2. Each `test_*.py` defines an `EXPECTED` dict of recovered values and asserts the parsed output against it, one field per test case.

## Test Files

| Test file | Drives example binary |
| --------- | --------------------- |
| `test_memory_alloc.py` | `profiler-hub_memory_alloc_writer_reader` |
| `test_kernel_dispatch.py` | `profiler-hub_kernel_dispatch_writer_reader` |

## Adding a Test

1. Add the corresponding example under `examples/` (see that directory's `README.md`).
2. Create `tests/integration/test_<record>.py` with an `EXPECTED` dict of the `key=value` fields the example prints.
3. Have the test call `run_launcher("<record>_writer_reader")` and assert against `EXPECTED`.
