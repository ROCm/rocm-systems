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
from rocprofsys import (
    RocprofsysConfig,
    SamplingRunner,
    validate_perfetto_trace,
    validate_rocpd_database,
)

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

### Template for adding a test method

```python
def test_sampling(
    self,
    rocprof_config: RocprofsysConfig,   # Always include
    test_output_dir: Path,              # Always include
    my_env: dict[str, str],             # Your environment fixture
    use_rocpd: bool,                    # Include if doing ROCpd validation
    use_perfetto: bool,                 # Include if doing Perfetto validation
    subtests,                           # **ONLY if using subtests**
    collect_result,                     # Always include
):
    # 1. Setup environment
    env = my_env.copy()
    if use_rocpd:
        env["ROCPROFSYS_USE_ROCPD"] = "ON"

    # 2. Create runner (wrap in try/except for skip)
    try:
        runner = SamplingRunner(
            config=rocprof_config,
            target="my-binary",
            output_dir=test_output_dir,
            env=env,
            timeout=120,
            # pass_regex: Patterns that MUST appear in output for success
            # fail_regex: Patterns that must NOT appear (defaults to abort patterns)
            pass_regex=["expected output pattern"],  # Optional
            fail_regex=["expected fail pattern"],  # Optional
        )
    except FileNotFoundError:
        pytest.skip("my-binary not found")

    # 3. Run and collect result
    result = runner.run()
    collect_result(result)

    # 4. Check runner success FIRST
    if not result.success:
        pytest.fail(f"Sampling failed: {result.failure_reason}")

    # 5. Subtests for validation (ALWAYS last)
    with subtests.test("Perfetto validation"):
        # Always have this if statement
        if not use_perfetto:
            pytest.skip("Perfetto is not enabled")
        if result.perfetto_file is None:
            pytest.fail("Perfetto trace not created")
        validation = validate_perfetto_trace(
            result.perfetto_file,
            rocprof_config.rocprofsys_tests_dir,
            categories=["hip_runtime_api"],
        )
        if not validation.is_valid:
            pytest.fail(f"Perfetto validation failed: {validation.message}")

    with subtests.test("ROCpd validation"):
        # Always have this if statement
        if not use_rocpd:
            pytest.skip("ROCpd is not enabled")
        # ... validation logic
```

### Critical Rules

| Rule | Reason |
|------|--------|
| Don't include `subtests` if unused | Affects `--show-output-on-subtest-fail` detection |
| Use `pytest.fail()` not `assert` | Better error messages with `result.failure_reason` |
| Always call `collect_result(result)` | Enables `--show-output` functionality |

### Available Runners

| Runner | Description |
|--------|-------------|
| `BaselineRunner` | Run target without instrumentation |
| `SamplingRunner` | Run with `rocprof-sys-run` sampling mode |
| `BinaryRewriteRunner` | Rewrite binary with `rocprof-sys-instrument`, then run |
| `RuntimeInstrumentRunner` | Runtime instrumentation via `rocprof-sys-instrument` |
| `SysRunRunner` | Run via `rocprof-sys-run` wrapper |
| `SysBinaryRunner` | Run rocprof-sys binaries directly (avail, instrument, etc.) |

### Available Validators

| Validator | Description |
|-----------|-------------|
| `validate_perfetto_trace()` | Validate Perfetto trace contents (categories, labels, counts) |
| `validate_rocpd_database()` | Validate ROCpd SQLite database against rules files |
| `validate_timemory_json()` | Validate timemory JSON output (labels, counts, depths) |
| `validate_causal_json()` | Validate causal profiling JSON output |
| `validate_file_exists()` | Check file exists and optionally match regex patterns |

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
│   ├── runners.py       # Test runners (SamplingRunner, etc.)
│   └── validators.py    # Output validators (Perfetto, ROCpd, etc.)
└── test_*.py            # Test files
```
