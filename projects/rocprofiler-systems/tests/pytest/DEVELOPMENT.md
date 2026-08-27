# rocprofiler-systems Test Infrastructure

## rocprofsys Package

This package contains the code infrastructure required to run pytests. It contains 6 main components:

### capabilities.py

Contains functions that allow pytests to discover what the current system supports. Everything here is lazy-initialized for performance. This contains information on the system that is not required for every test. Should only be accessed through an instance of a `RocprofsysConfig` (see `config.py`).

### config.py

Contains configuration information used by every test. This includes the location of core binaries, an instance of the `SystemCapabilities` class, the computed `LD_LIBRARY_PATH` and some core helper functions.

### environment.py

Contains `TestEnvironment`, which models the per-run environment as three layers ordered from lowest to highest precedence: `base` (framework defaults for the test type), `test` (settings supplied by the test, plus framework-injected values like `ROCPROFSYS_OUTPUT_PATH`), and `user` (the inherited shell environment). A runner builds these layers and merges them (highest layer wins) into the environment handed to the subprocess.

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
| `assert_unified_memory_output` | Validates unified-memory text and JSON outputs (`unified_memory*.txt` / `unified_memory*.json`) under the test output directory. |
| `assert_causal_json` | Validates causal profiling JSON output. |

> For precise, one-off checks, plain `assert` statements can also be used. They are reported as a single test failure (not a subtest), but the same runner output is included in the failure report.

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
@pytest.mark.class_name("transpose")
class TestTranspose(RocprofsysTest):
    @pytest.mark.parametrize("mode", ["sampling", "sys_run"])
    @pytest.mark.parametrize("iterations,tile_dim,block_rows", [(1, 16, 16), (2, 32, 32)])
    def test_parametrized(self, mode, iterations, tile_dim, block_rows):
        ...
```

This produces names like:

```text
transpose-parametrized-1-16-16-sampling
transpose-parametrized-1-16-16-sys_run
transpose-parametrized-2-32-32-sampling
transpose-parametrized-2-32-32-sys_run
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

**Every marker is declared once, with a `register_*` call** in the "Marker declarations" section of `conftest.py`. The declarative engine in `rocprofsys/markers.py` handles the rest: registration (`pytest --markers`), dependency injection, capability-based skipping, and CTest label export. There is no separate registration list, per-marker `if ... in item.keywords` block, or CTest export set to edit.

> **Strict markers are enforced.** `conftest.py` sets `--strict-markers`, so using a `@pytest.mark.<name>` that has no `register_*` declaration is a **hard collection error**, not a warning. Declare the marker below before using it.

There are two declaration functions plus one for dependencies:

- `register_marker(name, description=None, *, arg_hint=None, ctest=CTestExport.NAME)` — a **non-functional** marker (a label or behavior with no skip check), e.g. `slow`, `serialize`, `rocpd`.
- `register_functional_marker(name, description=None, *, arg_hint=None, skip_if=None, ctest=CTestExport.NAME)` — a **functional** marker whose `skip_if` is evaluated at collection.
- `add_marker_dependency_if(marker, *, when_present, when=None, on_unmet=OnUnmet.SKIP)` — declares that `marker` is added to a test whenever the `when_present` trigger marker is present (the declarative successor to the old `add_marker_if`).

The shared arguments:

- `name` — the marker name (`@pytest.mark.<name>`).
- `description` — text shown by `pytest --markers` (defaults to `"label test as <name>"`).
- `arg_hint` — argument signature for the registration line, e.g. `"version"` renders `name(version): ...`.
- `skip_if` (functional only) — a callable `ctx -> None | str`. Return `None` to run the test, or a skip-reason string to skip it. `ctx` is a `MarkerCtx` holding `ctx.config` (`RocprofsysConfig`), `ctx.gpu_info`, and `ctx.args` (the marker's arguments); use the `caps(ctx)` helper for system capabilities.
- `ctest` — a `CTestExport` policy controlling the emitted CTest label: `NONE` (not a label), `NAME` (name only, args hidden — e.g. `rocpd`), `ARGS` (args only, name hidden — e.g. `mpi_implementation`), or `ALL` (name plus `name[args]`). The default is `NAME`.

For dependencies, `add_marker_dependency_if` evaluates `when` (if given) with the trigger's `MarkerCtx`; a returned reason string means the requirement is unmet and `on_unmet` decides whether to skip the test (`OnUnmet.SKIP`) or leave it as-is (`OnUnmet.IGNORE`). Examples: `add_marker_dependency_if("gpu", when_present="multi_gpu")` and `add_marker_dependency_if("papi", when_present="annotate", when=_annotate_reason)`.

**Informational markers** (pure labels, e.g. `avail`) just need a name; add them to the `_INFORMATIONAL_LABELS` list (or call `register_marker(name, description=...)` for a custom description).

**Functional markers** call `register_functional_marker` with a `skip_if`. Example:

```python
register_functional_marker(
    "nic",
    description="requires PAPI network events",
    skip_if=requires(
        lambda ctx: caps(ctx).papi_nic_events and caps(ctx).perf_events_usable,
        "Requires PAPI network events and perf_event_paranoid <= 2 (or CAP_SYS_ADMIN)",
    ),
)
```

The `requires(predicate, reason)` helper builds a `skip_if` from a truthy predicate and a fixed message.

**Minimum-version markers** use the `min_version(...)` template, which takes a version accessor, produces the `skip_if`, and registers the marker for you:

```python
min_version(
    "rocm_min_version",
    lambda ctx: ctx.config.rocm_version,
    parts=3,
    label="ROCm",
    description="mark test as requiring minimum ROCm version",
)
```

If a marker depends on a system capability not already tracked by `SystemCapabilities`, add it to `capabilities.py` and reference it via `caps(ctx)` in the condition.

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
