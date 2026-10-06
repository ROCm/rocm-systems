# hipFile Python binding tests

Unit tests for the high-level `hipfile` Python API.

The pytest suite is fully hermetic: the compiled Cython extension `hipfile._hipfile` is
replaced with a pure-Python fake injected into `sys.modules` before `hipfile` is
imported, so the suite runs on any machine — no ROCm install, GPU, AIS-capable
storage, or build step required. Filesystem access in the `FileHandle` tests is
mocked (`os.open` / `os.close`).

## Running

From the repository root within a virtual environment:

```
pip install pytest
cd projects/hipfile/python/tests
python3 -m pytest
```

`main.py` is *not* part of that suite — `python_files = test_*.py` keeps pytest
from collecting it. It is a standalone hardware check, run against an installed
`hipfile` from the repository root:

```
python3 python/tests/main.py INPUT OUTPUT
```

## Layout

| File | Covers |
|------|--------|
| `conftest.py` | Installs the fake `hipfile._hipfile` in `sys.modules` and exposes shared fixtures. Explains why the extension substrate is a *fake* (real ints + lambdas) rather than `Mock`/`MagicMock`. |
| `test_enums.py` | `OpError` / `FileHandleType` — values track the extension, members stay distinct, membership checks. |
| `test_error.py` | `HipFileException` — stored codes and `__str__`, including the `HIP_DRIVER_ERROR` branch. |
| `test_driver.py` | `Driver` — open/close success and error, `use_count` delegation, context-manager open-then-close ordering. |
| `test_buffer.py` | `Buffer` — null rejection, register/deregister success and error, no-op deregister, context manager. |
| `test_file.py` | `FileHandle` — `handle_type` setter guards, fd cleanup on registration failure, idempotent close, and the parametrized read/write return-code contract. |
| `test_properties.py` | `get_version` / `driver_get_properties` — success and error paths. |
| `main.py` | Standalone end-to-end hardware check (not collected by pytest): copies a file through GPU memory with the *real* extension and compares SHA-256 digests. Needs a GPU and AIS-capable storage. |

## Notes

- Everything except `main.py` is a **unit** suite — it verifies the Python
  wrapper's contract, not real GPU/driver/filesystem I/O.
- Per-test overrides use `unittest.mock.patch.object` on the *consuming* module
  (e.g. `hipfile.driver.hipFileDriverOpen`), since each module does
  `from hipfile._hipfile import ...` and holds its own reference.
