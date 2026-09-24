# hipFile Python binding tests

Tests for the high-level `hipfile` Python API, in two mutually exclusive modes.

**Unit (default).** Fully hermetic: the compiled Cython extension
`hipfile._hipfile` is replaced with a pure-Python fake injected into
`sys.modules` before `hipfile` is imported, so the suite runs on any machine —
no ROCm install, GPU, AIS-capable storage, or build step required. Filesystem
access in the `FileHandle` tests is mocked (`os.open` / `os.close`).

**System (`--system`).** No fake and no source-tree `sys.path` entry, so
`import hipfile` resolves to the *installed* package and its real extension.
Copies real bytes through GPU memory, so it needs a GPU and an AIS-capable
filesystem. These are the `test_system_*.py` modules.

The two cannot share a process — the fake lives in `sys.modules` for the whole
session — so `--system` deselects the unit tests and vice versa.

## Running

From the repository root within a virtual environment:

```
pip install pytest
cd projects/hipfile/python/tests
python3 -m pytest
```

For the system tests, install the bindings first, then:

```
python3 -m pytest . --system --ais-capable-dir /mnt/ais/ext4
```

Pass the test path explicitly and *first*: the options below are registered by
`conftest.py`, which pytest only loads once an argument points at this
directory.

| Option | Purpose |
|--------|---------|
| `--system` | Run the system tests against the real installed extension instead of the fake. |
| `--ais-capable-dir` | Directory on an AIS-capable filesystem to do the I/O in (default `/tmp`). Same name and default as the GTest suite's flag, which CMake feeds from `AIS_CAPABLE_DIR`. |
| `--allow-skip-fastpath` | Skip rather than fail when the environment cannot run the tests. Matches `HIPFILE_ALLOW_SKIP_FASTPATH_TESTS`. |

An unusable environment is a **failure** by default, not a skip — mirroring
`enforceFastpathGate()` in `test/system/amd/io.cpp`. A silent skip in CI is
indistinguishable from a passing run, which is the hole the system tests exist
to close.

## Layout

| File | Covers |
|------|--------|
| `conftest.py` | Selects the mode, installs the fake `hipfile._hipfile` in `sys.modules` (unit mode only), and exposes shared fixtures. Explains why the extension substrate is a *fake* (real ints + lambdas) rather than `Mock`/`MagicMock`. |
| `test_enums.py` | `OpError` / `FileHandleType` — values track the extension, members stay distinct, membership checks. |
| `test_error.py` | `HipFileException` — stored codes and `__str__`, including the `HIP_DRIVER_ERROR` branch. |
| `test_driver.py` | `Driver` — open/close success and error, `use_count` delegation, context-manager open-then-close ordering. |
| `test_buffer.py` | `Buffer` — null rejection, register/deregister success and error, no-op deregister, context manager. |
| `test_file.py` | `FileHandle` — `handle_type` setter guards, fd cleanup on registration failure, idempotent close, and the parametrized read/write return-code contract. |
| `test_properties.py` | `get_version` / `driver_get_properties` — success and error paths. |
| `test_system_roundtrip.py` | **System.** Round-trips a 2 MiB file through GPU memory and compares SHA-256, checks the runtime library version against the headers the extension was built from, and checks the driver refcount balances. |

## Notes

- Everything except `test_system_*.py` is a **unit** test — it verifies the
  Python wrapper's contract, not real GPU/driver/filesystem I/O.
- Per-test overrides use `unittest.mock.patch.object` on the *consuming* module
  (e.g. `hipfile.driver.hipFileDriverOpen`), since each module does
  `from hipfile._hipfile import ...` and holds its own reference.
