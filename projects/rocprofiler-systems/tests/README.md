# rocprofiler-systems Testing Suite

## General Use

### Setup

A minimum Python version of 3.8 is required to run the test suite.
Use of a virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running Tests

The testing suite uses pytest as the testing framework and CTest as the test executor. It is not recommended that you run
pytest directly. Instead, the generated CTestTestfile.cmake should be prioritized.

If you have built rocprofiler-systems from source with testing enabled, the test suite can be run using:

```bash
cd <path to rocprofiler-systems>
ctest --test-dir <build-dir>
```

Default output directory: `<build-dir>/rocprof-sys-pytest-output/`

If you are using the installed CTestTestfile.cmake:

```bash
ctest --test-dir <install-prefix>/share/rocprofiler-systems/tests
```

Default output directory: `/tmp/$USER/rocprof-sys-pytest-output/`

Note: If the tests are picking up the wrong Python executable, set `ROCPROFSYS_TEST_EXECUTABLE`.
For example, in a venv, passing `ROCPROFSYS_TEST_EXECUTABLE=$(which python3)` should suffice.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_TEST_DIR` | Path to test package directory or .pyz file | Auto-detected |
| `ROCPROFSYS_TEST_EXECUTABLE` | Python (install mode) or pytest (build mode) executable to use | Auto-detected |
| `ROCPROFSYS_PYTHON_HINTS` | Additional search paths for versioned Python interpreters | Not set |
| `ROCPROFSYS_BUILD_DIR` | Path to build directory | Auto-detected |
| `ROCPROFSYS_INSTALL_DIR` | Path to install prefix (enables install mode) | Not set |
| `ROCPROFSYS_SOURCE_DIR` | Path to source directory | Auto-detected |
| `ROCPROFSYS_KEEP_TEST_OUTPUT` | Keep test output on success (`ON`/`OFF`) | `ON` |
| `ROCPROFSYS_USE_ROCPD` | Enable/disable ROCpd validation (`ON`/`OFF`) | `ON` if available |
| `ROCPROFSYS_VALIDATE_PERFETTO` | Enable/disable Perfetto validation (`ON`/`OFF`) | `ON` if available |
| `ROCPROFSYS_TRACE_PROC_SHELL` | Path to trace_processor_shell binary | Auto-detected |
| `ROCM_PATH` | Path to ROCm installation | `/opt/rocm` |

#### Perfetto GLIBC Issue

If Perfetto validation fails due to GLIBC version mismatch (this may occur on RHEL 8.x or SUSE 15.5), download a compatible `trace_processor_shell` binary and set `ROCPROFSYS_TRACE_PROC_SHELL`:

```bash
curl -L https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v47.0/linux-amd64/trace_processor_shell -o /tmp/$USER/trace_processor_shell
chmod +x /tmp/$USER/trace_processor_shell
export ROCPROFSYS_TRACE_PROC_SHELL=/tmp/$USER/trace_processor_shell
```
