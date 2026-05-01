# AMD SMI Python library

The AMD SMI Python interface offers an accessible way to interact
with AMD hardware through a user-friendly API. Find the documentation in the
`docs/` directory.

- [Install AMD SMI](../docs/install/install.md)
- [About the library and how to get started](../docs/how-to/amdsmi-py-lib.md)
- [Python API reference](../docs/reference/amdsmi-py-api.md)

## Online documentation

Explore the latest documentation on the [ROCm documentation
portal](https://rocm.docs.amd.com/projects/amdsmi/en/latest/index.html).

- [Install AMD
  SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)

- [Python library
  usage](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-py-lib.html).

- [Python API
  reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html).

## Install paths

The AMD SMI Python wrapper supports two coexisting install modes. Both
expose the same `import amdsmi` entry point.

| Mode | What ships | Loader resolves to |
|------|-----------|--------------------|
| System package (`amd-smi-lib` rpm/deb) | The wrapper to `/opt/rocm/share/amd_smi/amdsmi/` and an `amdsmi.pth` to the system Python's `site-packages` so plain `import amdsmi` works. The shared library lives at `/opt/rocm/lib(64)/libamd_smi.so`. | `libamd_smi.so` from the path-derived ROCm root, then `ROCM_HOME` / `ROCM_PATH`, then the dynamic linker. |
| `pip install amdsmi` (manylinux wheel) | The wrapper plus a SONAME-renamed `libamd_smi_python.so` directly inside `<site-packages>/amdsmi/`. | `libamd_smi_python.so` next to the wrapper. |

When both are installed, the pip wheel wins because `<site-packages>/amdsmi/`
precedes the `.pth` redirect on `sys.path`. The SONAME split (`libamd_smi.so`
vs `libamd_smi_python.so`) prevents a single process from double-loading the
same library.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ROCM_HOME` / `ROCM_PATH` | Fallback ROCm prefix used by the wrapper when neither the pip-bundled `libamd_smi_python.so` nor the path-derived ROCm root is usable. |
| `AMDSMI_LIB_OVERRIDE` | Absolute path to a `libamd_smi*.so` to load **instead of** the auto-detected one. Intended for local development against an in-tree build (e.g. `AMDSMI_LIB_OVERRIDE=$PWD/build/libamd_smi.so python3 -c "import amdsmi"`) and for ABI-compatibility tests that need to point the wrapper at a curated alternate library. When set, both pip and system context detection are bypassed. |
| `AMDSMI_DEBUG_LOAD` | Set to `1` to print the resolved `.so` path (or every path the loader tried) to stderr at import time. Use this first when diagnosing a load failure. |

## Diagnose a load failure

If `import amdsmi` succeeds but the first `amdsmi_*` call raises
`OSError`, the wrapper installed a `_MissingLibrary` sentinel because the
library could not be loaded. Re-run with `AMDSMI_DEBUG_LOAD=1`:

```
$ AMDSMI_DEBUG_LOAD=1 python3 -c 'import amdsmi'
[amdsmi] WARNING: <dlopen error from /opt/rocm/lib64/libamd_smi.so>
[amdsmi] WARNING: <dlopen error from /opt/rocm/lib/libamd_smi.so>
[amdsmi] Module imported in degraded mode; calling any amdsmi_* function will raise OSError.
```

The exact `WARNING` text is the underlying `OSError` string from
`ctypes.CDLL`, which is platform- and toolchain-dependent (e.g.
`cannot open shared object file: No such file or directory` on glibc).

The deferred `OSError` lists every searched path:

```
OSError: AMD SMI shared library could not be loaded.
Underlying error: ...
Searched the following paths (in order):
  - /opt/rocm/lib64/libamd_smi.so
  - /opt/rocm/lib/libamd_smi.so
Hint: install amd-smi-lib (rpm/deb) or set ROCM_PATH/ROCM_HOME to your ROCm
install root, or add the directory containing libamd_smi.so to LD_LIBRARY_PATH.
```
