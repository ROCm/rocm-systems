# rocprofiler-systems Pytest Suite

## General Use

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tests/pytest/requirements.txt
```

### Running Tests

Tests can run in two modes: **build** or **install**.

#### Build Mode (Default)

Runs tests using binaries from your build directory.

```bash
cd <path to rocprofiler-systems>
pytest tests/pytest/
```

Default output directory: `<build-dir>/rocprof-sys-pytest-output/`

If auto detection of the build directory fails, specify `ROCPROFSYS_BUILD_DIR=<path to build-dir>`

#### Install Mode

Runs tests using binaries from your install location.

```bash
ROCPROFSYS_INSTALL_DIR=<install prefix> pytest <install prefix>/share/rocprofiler-systems/tests/pytest/

# Using /opt/rocprofiler-systems
ROCPROFSYS_INSTALL_DIR=/opt/rocprofiler-systems pytest /opt/rocprofiler-systems/share/rocprofiler-systems/tests/pytest/
```

Default output directory: `/tmp/rocprof-sys-pytest-output/`

> **Note:** Install mode requires `ROCPROFSYS_INSTALL_EXAMPLES=ON` and `ROCPROFSYS_INSTALL_TESTING=ON` during build.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_BUILD_DIR` | Path to build directory | Auto-detected |
| `ROCPROFSYS_INSTALL_DIR` | Path to install prefix (enables install mode) | Not set |
| `ROCPROFSYS_SOURCE_DIR` | Path to source directory | Auto-detected |
| `ROCPROFSYS_KEEP_TEST_OUTPUT` | Keep test output on success (`ON`/`OFF`) | `OFF` |
| `ROCPROFSYS_USE_ROCPD` | Enable/disable ROCpd validation (`ON`/`OFF`) | `ON` if available |
| `ROCPROFSYS_VALIDATE_PERFETTO` | Enable/disable Perfetto tracing (`ON`/`OFF`) | `ON` if available|
| `ROCPROFSYS_TRACE_PROCESSOR_SHELL` | Path to trace_processor_shell binary | Auto-detected |
| `ROCM_PATH` | Path to ROCm installation | `/opt/rocm` |

### Common Commands

```bash
# Run all tests
pytest tests/pytest/

# Run a specific test file
pytest tests/pytest/test_transpose.py

# Run a specific test class
pytest tests/pytest/test_transpose.py::TestTranspose

# Run a specific test function
pytest tests/pytest/test_transpose.py::TestTranspose::test_sampling
```

### Custom Flags

| Flag | Description |
|------|-------------|
| `--show-output` | Show runner output when tests **pass** (requires -s flag) |
| `--show-output-on-subtest-fail` | Show runner output only when **subtests** fail |
| `--no-output` | Suppress all output (only show pass/fail) |

#### Output Display Logic

The `_result_output` fixture controls when runner output is printed:

| Scenario | Default | `--show-output-on-subtest-fail` | `--show-output` |
|----------|---------|--------------------------------|-----------------|
| Test passes | ❌ | ❌ | ✅ |
| Main test fails | pytest captures | pytest captures | pytest captures |
| Subtest fails | ❌ | ✅ | ✅ |

> **Perfetto GLIBC Issue:** If Perfetto validation fails due to GLIBC version mismatch, set `ROCPROFSYS_TRACE_PROCESSOR_SHELL` to a compatible binary.

---

## Writing a Test

### Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.gpu` | Requires a GPU |
| `@pytest.mark.mpi` | Requires MPI |
| `@pytest.mark.rocm` | Requires ROCm |
| `@pytest.mark.rocprofiler` | Uses ROCProfiler counters |
| `@pytest.mark.rocm_min_version("X.Y.Z")` | Requires minimum ROCm version |
| `@pytest.mark.gpu_category_exclude(["category"])` | Exclude specific GPU categories |
| `@pytest.mark.loops` | Tests loop instrumentation |
| `@pytest.mark.slow` | Marks test as slow |
| `@pytest.mark.rocpd("env_fixture")` | Injects `ROCPROFSYS_USE_ROCPD=ON` into the specified env fixture |

### File Structure

Tests generally follow the structure below.

```python
# 1. License header (required)
# MIT License...

# 2. Module docstring
"""
Tests for the <feature> example.
"""

# 3. Imports
import pytest
from pathlib import Path

# 4. Module-level fixtures (if needed)
# Note that env fixtures inherit the base environment
@pytest.fixture
def my_env() -> dict[str, str]:
    """Environment variables for this test module."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch",
    }

# 5. Test class
@pytest.mark.gpu
class TestMyFeature:
    """Tests for my feature."""

    # Class-level constants
    REWRITE_ARGS = ["-e", "-v", "2"]

    # Test methods...
```

### Template for Adding a Test Method

Testing occurs in two distinct phases:

1. **Run phase**: Execute the target binary using the `run_test` fixture
2. **Assert phase**: Verify output using assert fixtures

#### The `run_test` Fixture

The `run_test` fixture is a unified interface for all runner types. It handles:

- Runner creation and execution
- Result collection for output display
- Automatic failure/skip on errors

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runner_type` | `str` | Required | One of: `"baseline"`, `"sampling"`, `"binary_rewrite"`, `"runtime_instrument"`, `"sys_run"` |
| `target` | `str` | Required | Target executable name |
| `run_args` | `list[str]` | `None` | Arguments passed to the target |
| `env` | `dict[str, str]` | `None` | Environment variables |
| `timeout` | `int` | `300` | Timeout in seconds |
| `mpi_ranks` | `int` | `0` | Number of MPI ranks (0 = disabled) |
| `working_directory` | `Path` | `None` | Custom working directory |
| `skip_on_error` | `bool` | `False` | If True, skip instead of fail on error |
| `fail_on_not_found` | `bool` | `False` | If True, fail instead of skip when binary not found |
| `fail_message` | `str` | `None` | Custom failure message |
| `**kwargs` | | | Runner-specific args (see below) |

**Runner-specific kwargs:**

| Runner | Additional kwargs |
|--------|-------------------|
| `sampling` | `sample_args` |
| `binary_rewrite` | `rewrite_args`, `cleanup_on_success` |
| `runtime_instrument` | `instrument_args` |
| `baseline` | `command` |

#### Example Test

```python
def test_sampling(
    self,
    run_test,
    assert_regex,
    my_env: dict[str, str],
):
    result = run_test(
        "sampling",
        target="my_target",
        env=my_env,
        timeout=120,
    )

    assert_regex(result, pass_regex=[r"expected pattern"])
```

#### Example with Binary Rewrite

```python
def test_binary_rewrite(
    self,
    run_test,
    assert_regex,
    assert_perfetto,
    my_env: dict[str, str],
):
    result = run_test(
        "binary_rewrite",
        target="my_target",
        rewrite_args=["-e", "-v", "2", "--load", "pthread"],
        env=my_env,
        timeout=120,
    )

    assert_regex(result)
    assert_perfetto(result, categories=["host"])
```

### Available Assert Fixtures

Assert fixtures wrap validators and handle pytest logic (fail/skip, subtests, messages).

| Fixture | Description | Key Parameters |
|---------|-------------|----------------|
| `assert_regex` | Validate regex patterns in output | `pass_regex`, `fail_regex` |
| `assert_perfetto` | Validate Perfetto trace contents | `categories`, `labels`, `counts` |
| `assert_rocpd` | Validate ROCpd database | `rules_files` |
| `assert_timemory` | Validate timemory JSON output | `file_name`, `metric`, `labels` |
| `assert_file_exists` | Check file(s) exist | `path` (single or list) |
| `assert_causal_json` | Validate Causal JSON output | `file_name` |

See the respective definition for full details.

**Common parameters for all assert fixtures:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_on_fail` | `bool` | `False` | Skip instead of fail |
| `fail_message` | `str` | Varies | Custom failure message |
| `subtest_name` | `str` | Varies | Name shown in subtest output |

### Available Runners

| Runner | Description |
|--------|-------------|
| `baseline` | Run target without instrumentation. Auto-detects rocprof-sys binaries and uses appropriate environment |
| `sampling` | Run with `rocprof-sys-sample` |
| `binary_rewrite` | Rewrite binary with `rocprof-sys-instrument`, then run |
| `runtime_instrument` | Runtime instrumentation via `rocprof-sys-instrument` |
| `sys_run` | Run via `rocprof-sys-run` wrapper |

> **Note:** When using `baseline` with rocprof-sys binaries (`rocprof-sys-instrument`, `rocprof-sys-sample`, `rocprof-sys-run`, `rocprof-sys-avail`), the runner acts as if it were mimicking ROCPROFILER_SYSTEMS_ADD_BIN_TEST.

### Available Validators

| Validator | Description |
|-----------|-------------|
| `validate_regex()` | Validate regex patterns in test output |
| `validate_perfetto_trace()` | Validate Perfetto trace contents (categories, labels, counts) |
| `validate_rocpd_database()` | Validate ROCpd SQLite database against rules files |
| `validate_timemory_json()` | Validate timemory JSON output (labels, counts, depths) |
| `validate_causal_json()` | Validate causal profiling JSON output |
| `validate_file_exists()` | Check file exists and is non-empty |

Validators are wrapped in assert fixtures to avoid code bloat. Use the assert fixtures in tests.

---

## Directory Structure

```text
tests/pytest/
├── conftest.py          # Pytest configuration, fixtures, hooks
├── requirements.txt     # Python dependencies
├── CMakeLists.txt       # CMake integration
├── README.md            # This file
├── rocprofsys/          # Test utilities package
│   ├── __init__.py      # Package exports
│   ├── config.py        # RocprofsysConfig dataclass, discovery functions
│   ├── gpu.py           # GPU detection utilities
│   ├── runners.py       # Test runners (BaselineRunner, SamplingRunner, etc.)
│   └── validators.py    # Output validators (Perfetto, ROCpd, etc.)
└── test_*.py            # Test files
```
