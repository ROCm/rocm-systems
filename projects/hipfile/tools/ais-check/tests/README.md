# ais-check tests

Unit tests for the `ais-check` tool. They are fully hermetic: every system
touchpoint (kernel config files, `/proc/kallsyms`, environment variables,
`glob` over `/opt/rocm`, and `ctypes.CDLL`) is faked with `monkeypatch`, so the
suite runs and produces identical results on any machine — no ROCm install,
GPU, or special privileges required.

## Running

From the repository root:

```
pip install pytest
cd projects/hipfile/tools/ais-check/tests
python3 -m pytest
```

## Layout

| File | Covers |
|------|--------|
| `conftest.py` | Loads the extensionless `ais-check` script as an importable `ais_check` module (via `SourceFileLoader`) and exposes it as a fixture. |
| `test_p2pdma.py` | `kernel_supports_p2pdma()` — each config source, missing/`=m`/commented options, and the "no configs found" warning. |
| `test_find_hip_runtimes.py` | `find_hip_runtimes()` — env-var search paths, symlink dedup, soname handling, and literal matching of glob metacharacters. |
| `test_hip_runtime_ais.py` | `hip_runtime_supports_ais()` — symbol probing via a fake HIP handle, old-runtime and load-failure paths, and the three-part symbol success check. |
| `test_amdgpu.py` | `amdgpu_supports_ais()` — symbol present/absent and the not-found/permission-denied branches. |
| `test_main.py` | `main()` — the exit-code contract and `-q`/`-v` output behavior. |

## Notes

- The `ais_check` fixture imports the script as a module, which means
  `if __name__ == "__main__"` does not run `main()` at import time.
- Tests patch attributes on the loaded module (e.g. `ais_check.os.uname`,
  `ais_check.ctypes.CDLL`), so the real environment never leaks in — capturing
  the original `glob.glob` before patching avoids recursion.
