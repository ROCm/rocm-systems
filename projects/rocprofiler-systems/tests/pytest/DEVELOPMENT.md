# rocprofiler-systems Test Infrastructure

## rocprofsys Package

This package contains the code infrastructure required to run pytests. It contains 5 main components:

### capabilities.py

Contains functions that allow pytests to discover what the current system supports. Everything here is lazy-initialized for performance. This contains information on the system that is not required for every test. Should only be accessed through an instance of a `RocprofsysConfig` (see `config.py`).

### config.py

Contains configuration information used by every test. This includes the location of core binaries, an instance of the `SystemCapabilities` class, base environments, and some core helper functions.

### gpu.py

Contains information about the GPU on the system. Also finds `rocminfo` and `llvm-objdump`/`roc-obj-ls` binaries used to check if a test binary was compiled for the proper architecture.

### runners.py

Contains classes corresponding to each type of test. For example, a runtime instrumentation test would run through the `RuntimeInstrumentRunner` class. These runner classes handle test execution and return an instance of `TestResult`, which contains the information one would need to evaluate whether a test has passed or failed.

### validators.py

Contains wrappers for the validation scripts and other helper functions that can be used during a test. They return a `ValidationResult` which can be used to determine if a test has failed or passed. In practice, the `assert_` fixtures in `conftest.py` wrap these validators and should be used instead.

## conftest.py

`conftest.py` contains all the logic required for pytest to parse the `test_*.py` code and generate tests.

### Core Classes

- **`RocprofsysTest`**: Base class for all test classes. It auto-injects common fixtures (`run_test`, `assert_regex`, `test_output_dir`, etc.) onto `self` via its `_setup` fixture, so test methods can call `self.run_test(...)`, `self.assert_regex(...)`, etc. directly.

### Key Hooks

- **`pytest_configure`**: Registers all custom markers and sets up CLI options.
- **`pytest_collection_modifyitems`**: Handles test skipping based on marker conditions.
- **`_generate_ctest_definitions`**: Generates a `CTestTestfile.cmake` from collected pytest items.

### Subtests (Validation Fixtures)

These fixtures run as **subtests**, meaning multiple validations within a single test are independently reported (one can fail without blocking others). They are automatically injected into `RocprofsysTest` via its `_setup` fixture, so test methods access them as `self.assert_regex(...)`, `self.assert_perfetto(...)`, etc.

| Fixture | Description |
| --------- | ------------- |
| `assert_regex` | Validates test output against pass/fail regex patterns. Patterns can be per-mode (e.g., different patterns for `binary_rewrite` vs `sampling`). |
| `assert_file_regex` | Like `assert_regex` but validates against a file's contents. |
| `assert_perfetto` | Validates that a Perfetto trace was generated and optionally checks its contents. |
| `assert_rocpd` | Validates that a ROCpd database was created. Requires `@pytest.mark.rocpd("env_name")`. |
| `assert_timemory` | Validates timemory JSON output files. |
| `assert_file_exists` | Validates that a specific file exists in the output directory. |
| `assert_causal_json` | Validates causal profiling JSON output. |

See the docstrings in `conftest.py` for full argument details.

## Writing a Test

### General Rules

At the minimum, the following rules should always be followed. If you have a reason not to, a comment should be left explaining the reasoning behind your choice.

- Test methods should always be inside a test class.
- Every test class should inherit from `RocprofsysTest`.
- At minimum, a test should use both `run_test` and `assert_regex`. They have complementary responsibilities: `run_test` performs process-level validation (subprocess launch, exit code, signals, timeout), while `assert_regex` performs content-level validation (matching against pass/fail patterns in the captured output). A test that only calls `run_test` will detect a crash but not a silent regression where the binary runs to completion without emitting the expected instrumentation output.

### Runner Modes

When parametrizing a test to run against different runner modes (e.g., sampling, sys_run), use `"mode"` as the parameter name:

```python
class TestExample(RocprofsysTest):
    @pytest.mark.parametrize("mode", ["sampling", "sys_run"])
    def test(self, mode, ...):
        result = self.run_test(mode, ...)
        self.assert_regex(result)
```

If a test is hard-coded to a single mode (no `@pytest.mark.parametrize("mode", ...)` decorator), tag it with `@pytest.mark.<mode>` (e.g. `@pytest.mark.sampling`) so `_generate_ctest_definitions` records the correct CTest mode label. When `parametrize("mode", [...])` is used — even with a single-element list — this is handled automatically.

### Test Parametrization

Parametrizing a test affects the generated CTest name. The order of `@pytest.mark.parametrize` decorators determines parameter order in the name. The **runner mode should always come last** so that tests group naturally when sorted:

```python
# Good: parameters first, mode last
class TestTranspose(RocprofsysTest):
    @pytest.mark.parametrize("mode", ["sampling", "sys_run"])
    @pytest.mark.parametrize("iterations,tile_dim,block_rows", [(1, 16, 16), (2, 32, 32)])
    def test_parametrized(self, mode, iterations, tile_dim, block_rows):
        ...
```

This produces names like:

```text
Transpose_parametrized_1_16_16_sampling
Transpose_parametrized_1_16_16_sys_run
Transpose_parametrized_2_32_32_sampling
Transpose_parametrized_2_32_32_sys_run
```

Note: pytest applies decorators bottom-up, so the **bottom** `@pytest.mark.parametrize` varies first (outermost loop). Place `mode` on top so it varies last in the name.

### Standardized test names

During collection, `_standardize_test_name()` in `conftest.py` builds a short name used for `-k` filtering and CTest identity.

- The `test_` / `test-` prefix is stripped from the method name.
- By default, the **`Test`** prefix is stripped from the class name, then the class segment and method segment are joined.

**Shape:** Prefer names that read as **`<word>-<word>-...`**. Standardized names use **lowercase hyphenated** form: underscores become hyphens, repeated hyphens are collapsed, and the combined string is lowercased.

**Long CamelCase classes:** A class like `TestRocprofilerSystemsRun` would produce a long, hard-to-read segment. Put **`@pytest.mark.class_name("rocprofiler-systems-run")`** on the class to supply an explicit hyphenated prefix. Example:

```python
@pytest.mark.class_name("rocprofiler-systems-run")
class TestRocprofilerSystemsRun(RocprofsysTest):
    def test_help(self):
        ...
```

That yields a stable name such as `rocprofiler-systems-run-help`.

`@pytest.mark.depends_on(...)` refers to the **standardized** name; use the same naming rules when declaring dependencies.

### ROCpd Validation

ROCpd requires an environment fixture for injection. Mark the test with `@pytest.mark.rocpd("<env_fixture_name>")`. Do **not** set `ROCPROFSYS_USE_ROCPD=ON` explicitly. It is injected automatically when conditions are met.

```python
@pytest.fixture
def my_env() -> dict[str, str]:
    return {}  # Do NOT specify ROCPROFSYS_USE_ROCPD=ON

class TestExample(RocprofsysTest):
    @pytest.mark.rocpd("my_env")
    def test(self, my_env):
        result = self.run_test("sampling", "target", env=my_env)
        self.assert_rocpd(result)  # Skipped automatically if ROCpd is unavailable
```

### Custom Commands

To run an arbitrary command, use the `baseline` runner with the `command` option. You must provide absolute paths and handle missing targets yourself:

```python
class TestExample(RocprofsysTest):
    def test(self, rocprof_config):
        script_path = rocprof_config.rocprofsys_tests_dir / "example-script.sh"
        if not script_path.exists():
            pytest.skip("example-script.sh not found")

        try:
            target = rocprof_config.get_target_executable("example-binary")
        except FileNotFoundError:
            pytest.skip("example-binary not found")

        result = self.run_test(
            "baseline", target, command=[script_path, target]
        )
        self.assert_regex(result)
```

> If your custom command uses a rocprof-sys binary (e.g., `rocprof-sys-instrument`), add the corresponding marker (e.g., `@pytest.mark.runtime_instrument`).

### Timeouts

Tests have a default timeout of 300 seconds (including subtests). To override, use `@pytest.mark.timeout(<seconds>)`. This timeout is passed to CTest.

### Serialization

If a test requires significant resources, mark it with `@pytest.mark.serialize` to prevent it from running in parallel.

### Test Dependencies

If a test depends on the output of another test, use `depends_on` and `preserve` together.

Use the **standardized test name** of the producer in both `@pytest.mark.depends_on(...)` and any path under `test_output_base` (output directories are named after `request.node.name`, which is set during collection). Those names follow the rules in **Standardized test names** above—derive them from class + method (and parametrization) the same way, or copy the name from a generated `CTestTestfile.cmake` / `ctest -N` listing when in doubt.

- `@pytest.mark.preserve("file1", "file2", ...)` — Prevents files from being deleted after the test completes, even when `ROCPROFSYS_KEEP_TEST_OUTPUT=0`.
- `@pytest.mark.depends_on("producer-generate", ...)` — Declares a CTest dependency on one or more tests. Each argument must **exactly** match the producer’s standardized name (implementation: `_standardize_test_name()` in `conftest.py`).

```python
class TestProducer(RocprofsysTest):
    @pytest.mark.preserve("coverage.json")
    def test_generate(self, ...):
        ...

class TestConsumer(RocprofsysTest):
    @pytest.mark.depends_on("producer-generate")
    def test_consume(self, test_output_base):
        file_path = test_output_base / "producer-generate" / "coverage.json"
        ...
```

### Adding Markers

**Every marker must be registered in `pytest_configure()` before use.** Check the `non_functional_markers` and `generic_functional_markers` lists first; if the marker is not there, add it.

There are two types of markers:

- **Informational markers** — Used to categorize tests for CTest label filtering (e.g., `-L "avail"`). They don't affect test execution. Add these to the `non_functional_markers` list.

- **Functional markers** — Affect test behavior, typically used to skip tests based on system capabilities. Their logic is implemented in `pytest_collection_modifyitems`. Add these to the `generic_functional_markers` list (or register them individually with `config.addinivalue_line` if they need a custom description).

When adding a functional marker:

1. Register it in `pytest_configure()`.
2. Add skip logic in `pytest_collection_modifyitems`. The convention is to define a checker function named `<keyword>_unavailable_reason(...)` that returns either `None` (when the requirement is met) or a `str` explaining why the test must be skipped (e.g. `gpu_unavailable_reason`, `mpi_unavailable_reason`). Then plug that function into the `_skip(...)` call for your marker.
3. If the marker depends on a system capability not already tracked by `SystemCapabilities`, add it to `capabilities.py`.

**CTest label behavior:** By default, all markers are included as CTest labels (e.g., `ctest -L "rocm"` filters by the `@pytest.mark.rocm` marker). To change how a marker appears in the generated CTest definitions, add it to one of these sets in `_generate_ctest_definitions()`:

- `no_report_markers` — Marker is **not** added as a CTest label (e.g., `timeout`, `serialize`, `ci_disable`).
- `no_report_args_markers` — Marker name is added as a label, but its **arguments are hidden** (e.g., `rocpd`).
- `only_report_args_markers` — Only the marker's **arguments** are added as labels, not the marker name itself (e.g., `mpi_implementation`).

### Template

```python
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""<Description of what this test module covers>."""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [
    # Module level markers
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def <test>_env() -> dict[str, str]:
    # Environment for your test

# =============================================================================
# <Feature> tests
# =============================================================================

# Any class level markers here
class Test<NAME>(RocprofsysTest):

    # Any test level markers here
    def test_<NAME>(self, <test>_env):
        result = self.run_test(
            # mode,
            # target,
            # env =<test>_env,
            # ...
        )

        # Subtests
        self.assert_regex(result)
        # ...
```
