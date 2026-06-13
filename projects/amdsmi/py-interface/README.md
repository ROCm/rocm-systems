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

The AMD SMI Python wrapper supports two install modes. Both expose the
same `import amdsmi` entry point, and a user installs exactly one of them.

| Mode | What ships | Loader resolves to |
|------|-----------|--------------------|
| System package (`amd-smi-lib` rpm/deb) | The `amdsmi` module is installed directly into the system Python's `site-packages` (`dist-packages` on Debian/Ubuntu, `site-packages` on RHEL/SLES). The shared library ships at `/opt/rocm/lib` and is found through the `ld.so.conf.d` entry the package installs. | `libamd_smi.so.<MAJOR>` via the dynamic linker (SONAME lookup). |
| `pip install amdsmi` (manylinux wheel) | The wrapper plus a SONAME-renamed `libamd_smi_python.so` directly inside `<site-packages>/amdsmi/`. | `libamd_smi_python.so` next to the wrapper. |

The SONAME split (`libamd_smi.so` vs `libamd_smi_python.so`) means the
system library and the wheel-bundled library cannot accidentally
double-load in a single process. The loader does **not** walk up to a
ROCm root, consult `ROCM_HOME`/`ROCM_PATH`, or drop a `.pth` redirect;
resolution is the fixed three-step order below.

## Loader resolution order

`amdsmi_wrapper.py` resolves the shared library in this order:

1. `AMDSMI_LIB_OVERRIDE` — absolute path to a `libamd_smi*.so` to load
   **instead of** the auto-detected one. Intended for local development
   against an in-tree build and for ABI-compatibility tests, e.g.:

   ```
   AMDSMI_LIB_OVERRIDE=$PWD/build/src/libamd_smi.so.<MAJOR> python3 -c "import amdsmi"
   ```

2. The bundled `libamd_smi_python.so` sitting next to the wrapper (pip wheel).
3. The versioned SONAME `libamd_smi.so.<MAJOR>` via the dynamic linker
   (system rpm / deb).

## Diagnose a load failure

Import is **tolerant**: if the shared library cannot be loaded,
`import amdsmi` still succeeds and installs a `_MissingLibrary` sentinel,
so documentation, linting, and multi-stage container builds keep working
without a runtime ROCm install. The failure is deferred to the first
`amdsmi_*` call, which raises:

```
OSError: AMD SMI shared library could not be loaded.
Underlying error: <the underlying ctypes.CDLL error string>
Hint: install amd-smi-lib (rpm/deb) or pip-install the amdsmi wheel.
```

The `Underlying error` line is the platform-dependent message from
`ctypes.CDLL` (e.g. `libamd_smi.so.<MAJOR>: cannot open shared object
file: No such file or directory` on glibc). To force a specific library
while debugging, set `AMDSMI_LIB_OVERRIDE` to its absolute path.
