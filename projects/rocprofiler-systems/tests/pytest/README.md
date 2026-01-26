# rocprofiler-systems Pytest Suite

## General Use

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests

Tests can run in two modes: **build** or **install**.

#### Build Mode (Default)

Runs tests using binaries from your build directory.

```bash
cd <path to rocprofiler-systems>
pytest <build-dir>/share/rocprofiler-systems/tests/pytest/
```

Default output directory: `<build-dir>/rocprof-sys-pytest-output/`

If auto detection of the build directory fails, specify `ROCPROFSYS_BUILD_DIR=<path to build-dir>`

#### Install Mode

Runs tests using binaries from your install location.

```bash
ROCPROFSYS_INSTALL_DIR=<install prefix> pytest <build-dir>/share/rocprofiler-systems/tests/pytest/

# Using /opt/rocprofiler-systems
ROCPROFSYS_INSTALL_DIR=/opt/rocprofiler-systems pytest <build-dir>/share/rocprofiler-systems/tests/pytest/
```

Default output directory: `/tmp/$USER/rocprof-sys-pytest-output/`

> **Note:** Install mode requires `ROCPROFSYS_INSTALL_TESTING=ON` during build.

#### Using the Standalone Package

A standalone `.pyz` package is included at `<install-dir>/share/rocprofiler-systems/tests/rocprofsys-tests.pyz`. This can be run directly with Python:

```bash
python3 <install-dir>/share/rocprofiler-systems/tests/rocprofsys-tests.pyz
```

All standard pytest flags work with the standalone package.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_BUILD_DIR` | Path to build directory | Auto-detected |
| `ROCPROFSYS_INSTALL_DIR` | Path to install prefix (enables install mode) | Not set |
| `ROCPROFSYS_SOURCE_DIR` | Path to source directory | Auto-detected |
| `ROCPROFSYS_KEEP_TEST_OUTPUT` | Keep test output on success (`ON`/`OFF`) | `ON` |
| `ROCPROFSYS_USE_ROCPD` | Enable/disable ROCpd validation (`ON`/`OFF`) | `ON` if available |
| `ROCPROFSYS_VALIDATE_PERFETTO` | Enable/disable Perfetto tracing (`ON`/`OFF`) | `ON` if available|
| `ROCPROFSYS_TRACE_PROCESSOR_SHELL` | Path to trace_processor_shell binary | Auto-detected |
| `ROCM_PATH` | Path to ROCm installation | `/opt/rocm` |

### Common Commands

**Running by marker** (`-m`): Use for running groups of tests with specific labels.

```bash
# See all available markers
pytest --markers

# Run tests with a specific marker
pytest <test-path> -m gpu
pytest <test-path> -m "slow and gpu"
pytest <test-path> -m "not slow"
```

**Running by keyword** (`-k`): Use for running specific test classes or methods.

```bash
# Run tests matching a keyword
pytest <test-path> -k transpose
pytest <test-path> -k "TestTranspose and sampling"
pytest <test-path> -k "not binary_rewrite"
```

**Quick Start Examples:**

| Mode | Command |
|------|---------|
| Run all tests | `pytest <test-path>` |
| Recommended | `pytest <test-path> -n auto -v --show-output-on-subtest-fail --show-config` |
| Standalone package | `python3 <pyz-path>` |

Where `<test-path>` is `<build-dir>/share/rocprofiler-systems/tests/pytest/`
and `<pyz-path>` is `<install-dir>/share/rocprofiler-systems/tests/rocprofsys-tests.pyz`.

### Parallel Execution (pytest-xdist)

Tests can be run in parallel using `pytest-xdist`:

```bash
pytest <build-dir>/share/rocprofiler-systems/tests/pytest/ -n auto  # Use all available cores
pytest <build-dir>/share/rocprofiler-systems/tests/pytest/ -n 4     # Use 4 workers
```

> **Warning:** Running tests in parallel can cause timeouts due to resource contention, especially for `runtime_instrument` tests. If you experience unexpected timeouts, try reducing the number of workers or running sequentially.

### Custom Flags

| Flag | Description |
|------|-------------|
| `--show-config` | Show test configuration in the pytest header |
| `--show-output` | Show runner output when tests **pass** |
| `--show-output-on-subtest-fail` | Show runner output only when **subtests** fail |
| `--output-dir=<path>` | Set the test output directory (default: `<build_dir>/pytest-output`) |
| `--output-log=<path>` | Write pytest output to the specified file (default: `<output_dir>/pytest-output.txt`) |
| `--monochrome` | Disable colored output and set `ROCPROFSYS_MONOCHROME=ON` for runners |
| `--allow-disabled` | Run tests with `@pytest.mark.disable` in CI mode (developer flag) |

**Tip:** Use `--tb=short` to hide source code in tracebacks, or `--tb=no` for no output.

#### Output Display Logic

The `_result_output` fixture controls when runner output is printed:

| Scenario | Default | `--show-output-on-subtest-fail` | `--show-output` |
|----------|---------|--------------------------------|-----------------|
| Test passes | ❌ | ❌ | ✅ |
| Subtest fails | ❌ | ✅ | ✅ |
| Main test fails | ✅ | ✅ | ✅ |

**Note:** With `--show-output`, runner output appears *before* the failure report. With `--show-output-on-subtest-fail`, it appears *after* (in the FAILURES section). This is due to how pytest processes report sections.

#### Perfetto GLIBC Issue

If Perfetto validation fails due to GLIBC version mismatch (this may occur on RHEL-8.x or SUSE-15.5), set `ROCPROFSYS_TRACE_PROCESSOR_PATH` to a compatible binary.

```bash
curl -L https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v47.0/linux-amd64/trace_processor_shell -o /tmp/$USER/trace_processor_shell
chmod +x /tmp/$USER/trace_processor_shell
export ROCPROFSYS_TRACE_PROCESSOR_PATH=/tmp/$USER/trace_processor_shell
```

Then run pytest with the environment variable set.
