# ais-check tests

Unit tests for the `ais-check` tool. They are fully hermetic: every system
touchpoint (kernel config files, `/proc/kallsyms`, `/proc/mounts`, the sysfs
block-device tree, environment variables, `glob` over `/opt/rocm`, and
`ctypes.CDLL`) is faked with `monkeypatch` or built under `tmp_path`, so the
suite runs and produces identical results on any machine — no ROCm install,
GPU, NVMe/LVM devices, or special privileges required.

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
| `conftest.py` | Loads the extensionless `ais-check` script as an importable `ais_check` module (via `SourceFileLoader`) and exposes it, plus the `make_sysfs_disk` and `make_fs_result` builder fixtures, as fixtures. |
| `test_p2pdma.py` | `kernel_supports_p2pdma()` — each config source, missing/`=m`/commented options, and the "no configs found" warning. |
| `test_find_hip_runtimes.py` | `find_hip_runtimes()` — env-var search paths, symlink dedup, soname handling, and literal matching of glob metacharacters. |
| `test_hip_runtime_ais.py` | `hip_runtime_supports_ais()` — symbol probing via a fake HIP handle, old-runtime and load-failure paths, and the three-part symbol success check. |
| `test_amdgpu.py` | `amdgpu_supports_ais()` — symbol present/absent and the not-found/permission-denied branches. |
| `test_collect_leaf_disks.py` | `_collect_leaf_disks()` and the `_whole_disk_sysfs`/`_read_sysfs` helpers — the sysfs device-stack walk over fake trees: partitions, LVM vs non-LVM dm layers, MD multi-leaf, and diamond dedup. |
| `test_inspect_filesystem.py` | `_inspect_filesystem()` — the per-filesystem verdict and each unsupported reason (fstype, non-NVMe, LVM), reason accumulation, and the stat/non-block/missing-sysfs branches. |
| `test_inspect_filesystems.py` | `_parse_proc_mounts()` and `inspect_filesystems()` — octal-escape decoding, read-error warning, and filtering of non-absolute/non-block sources plus source+mount dedup. |
| `test_print_filesystems.py` | `print_filesystems()` — default (supported-only) vs verbose (all entries with reasons) output. |
| `test_main.py` | `main()` — the exit-code contract, `-q`/`-v` output behavior, and that the filesystem report does not affect the exit code. |

## Notes

- The `ais_check` fixture imports the script as a module, which means
  `if __name__ == "__main__"` does not run `main()` at import time.
- Tests patch attributes on the loaded module (e.g. `ais_check.os.uname`,
  `ais_check.ctypes.CDLL`), so the real environment never leaks in — capturing
  the original `glob.glob` before patching avoids recursion.
- The filesystem tests build a fake sysfs block-device tree under `tmp_path`
  (directories, files, and symlinks via `make_sysfs_disk`) and point the
  script's `_SYS_DEV_BLOCK` base at it, so the real device walk runs unmocked
  against a controlled tree.
