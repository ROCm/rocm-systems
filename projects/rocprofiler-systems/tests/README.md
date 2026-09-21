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

Installed-tree test runs require configuring with `-DROCPROFSYS_INSTALL_TESTING=ON`
before `cmake --install`, otherwise the installed validator scripts under
`share/rocprofiler-systems/tests/` will be missing and pytest-driven CTest runs will
fail.

Note: If the tests are picking up the wrong Python executable, set `ROCPROFSYS_TEST_EXECUTABLE`.
For example, in a venv, passing `ROCPROFSYS_TEST_EXECUTABLE=$(which python3)` should suffice.

### Environment Variables

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `ROCPROFSYS_TEST_DIR` | Path to test package directory or .pyz file | Auto-detected |
| `ROCPROFSYS_TEST_EXECUTABLE` | Python (install mode) or pytest (build mode) executable to use | Auto-detected |
| `ROCPROFSYS_PYTHON_HINTS` | Additional search paths for versioned Python interpreters | Not set |
| `ROCPROFSYS_BUILD_DIR` | Path to build directory | Auto-detected |
| `ROCPROFSYS_INSTALL_DIR` | Path to install prefix (enables install mode and overrides build-tree autodetection) | Not set |
| `ROCPROFSYS_SOURCE_DIR` | Path to source directory | Auto-detected |
| `ROCPROFSYS_KEEP_TEST_OUTPUT` | Keep test output on success (`ON`/`OFF`) | `ON` |
| `ROCPROFSYS_USE_ROCPD` | Enable/disable ROCpd validation (`ON`/`OFF`) | `ON` if available |
| `ROCPROFSYS_VALIDATE_PERFETTO` | Enable/disable Perfetto validation (`ON`/`OFF`) | `ON` if available |
| `ROCPROFSYS_TRACE_PROC_SHELL` | Path to trace_processor_shell binary | Auto-detected |
| `ROCPROFSYS_DISABLE_TEST_CACHE` | Disables caching used by the tests | `OFF` |
| `ROCM_PATH` | Path to ROCm installation | `/opt/rocm` |

#### Trace Processor Shell

Perfetto validation needs a `trace_processor_shell` binary. The build downloads a pinned copy and
stages it in the build tree, where the validation script picks it up automatically. If none was
staged, or the staged one cannot run on the host, the Perfetto Python API downloads one on demand
instead, which needs network access at test time.

Set `ROCPROFSYS_TRACE_PROC_SHELL` to use a specific binary in preference to both:

```bash
curl -L https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v47.0/linux-amd64/trace_processor_shell -o /tmp/$USER/trace_processor_shell
chmod +x /tmp/$USER/trace_processor_shell
export ROCPROFSYS_TRACE_PROC_SHELL=/tmp/$USER/trace_processor_shell
```

Configure with `-DROCPROFSYS_TRACE_PROCESSOR_SHELL=<path>` to stage an existing binary,
`-DROCPROFSYS_DOWNLOAD_TRACE_PROCESSOR_SHELL=OFF` to skip the download, or
`-DROCPROFSYS_TRACE_PROCESSOR_SHELL_URL=<url>` and
`-DROCPROFSYS_TRACE_PROCESSOR_SHELL_SHA256=<sha256>` to fetch a different build.
