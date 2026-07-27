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
| System package (`amd-smi-lib` rpm/deb) | The wrapper installed directly into the system Python's `site-packages` so plain `import amdsmi` works. The shared library lives at `/opt/rocm/lib/libamd_smi.so` and is registered with the dynamic linker via `ldconfig`. | `libamd_smi.so` resolved by the dynamic linker (SONAME). |
| `pip install amdsmi` (manylinux wheel) | The wrapper plus a SONAME-renamed `libamd_smi_python.so` directly inside `<site-packages>/amdsmi/`. | `libamd_smi_python.so` next to the wrapper. |

When both are installed, the pip wheel wins because the bundled
`libamd_smi_python.so` sits next to the wrapper and is loaded before the
system fallback is consulted. The SONAME split (`libamd_smi.so` vs
`libamd_smi_python.so`) prevents a single process from double-loading the
same library.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `AMDSMI_LIB_OVERRIDE` | Absolute path to a `libamd_smi*.so` to load **instead of** the auto-detected one. Intended for local development against an in-tree build (e.g. `AMDSMI_LIB_OVERRIDE=$PWD/build/libamd_smi.so python3 -c "import amdsmi"`) and for ABI-compatibility tests that need to point the wrapper at a curated alternate library. When set, it takes precedence over both the pip-bundled and system libraries. |

## Diagnose a load failure

If `import amdsmi` succeeds but the first `amdsmi_*` call raises
`OSError`, the wrapper installed a `_MissingLibrary` sentinel because the
shared library could not be loaded. The module still imports so that
doc/lint tooling works without a runtime ROCm install; any call into a
wrapped C symbol raises:

```
OSError: AMD SMI shared library could not be loaded.
Underlying error: <dlopen error from ctypes.CDLL>
Hint: install amd-smi-lib (rpm/deb) or pip-install the amdsmi wheel.
```

The `Underlying error` text is the platform-dependent `OSError` string
from `ctypes.CDLL` (e.g. `cannot open shared object file: No such file or
directory` on glibc). To load a specific library explicitly, set
`AMDSMI_LIB_OVERRIDE` to its absolute path.
