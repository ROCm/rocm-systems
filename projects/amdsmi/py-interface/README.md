# AMD SMI Python library

The AMD SMI Python interface offers an accessible way to interact
with AMD hardware through a user-friendly API.

## Install from PyPI

```shell
python3 -m pip install "amdsmi>=27.1.0"
```

The wheel contains the Python library and the native AMD SMI library it
needs. It does not include the `amd-smi` command-line tool or the C headers.

Requirements:

- Linux on x86_64 with glibc 2.27 or later.
- CPython 3.6 or later, or PyPy, with pip 20.3 or later. Older pip versions
  cannot install this wheel.
- The `amdgpu` driver and `libdrm_amdgpu` (see below).
- Access to the GPU device files: add your user to the `render` and `video`
  groups. Changing GPU settings requires root.

Call `amdsmi_init()` once per process and `amdsmi_shut_down()` when you are
done. Do not call them from several threads at the same time, and do not
request process lists for several GPUs from different threads at the same
time.

Do not install this wheel into the Python environment used by an `amd-smi`
command-line tool from ROCm 7.2 or earlier, or from TheRock 7.11 or earlier.
Those tools load whichever `amdsmi` package their Python finds first and do
not work with this one. Use a virtual environment instead.

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

The following install modes expose the same `import amdsmi` entry point:

| Mode | What ships | Loader resolves to |
|------|-----------|--------------------|
| System package (`amd-smi-lib` rpm/deb) | The wrapper installed directly into the system Python's `site-packages` so plain `import amdsmi` works. The shared library lives at `/opt/rocm/lib/libamd_smi.so` and is registered with the dynamic linker via `ldconfig`. | `libamd_smi.so` resolved by the dynamic linker (SONAME). |
| TheRock native package, tarball, or ROCm pip install | The wrapper under `<root>/share/amd_smi/amdsmi`; add `<root>/share/amd_smi` to `PYTHONPATH` for direct imports. | `<root>/lib/libamd_smi.so.<MAJOR>`, resolved relative to the wrapper. |
| Bundled-library wheel (`BUILD_PYTHON_WHEEL=ON`) | The wrapper plus a SONAME-renamed `libamd_smi_python.so` directly inside `<site-packages>/amdsmi/`. | `libamd_smi_python.so` next to the wrapper. |

AMD SMI 27.x wheels on PyPI include the native library and use the AMD SMI
library version, not the ROCm release number. See the
[installation guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)
for the available delivery channels.

The manylinux wheel needs the dynamically loaded `libdrm_amdgpu` for
DRM-backed GPU queries: install `libdrm-amdgpu1` on
Debian/Ubuntu, `libdrm` on RHEL/AlmaLinux/Rocky, or `libdrm_amdgpu1` on
SLES/openSUSE. Without it, some queries fail and the PCI address reported for
a GPU can be wrong, even when GPU discovery succeeds.

For raw source-built wheel dependencies, see
[Packaging and install paths](https://rocm.docs.amd.com/projects/amdsmi/en/develop/packaging.html).

When a bundled-library wheel and system package coexist, Python's import
path selects the wrapper. The wheel's wrapper loads its bundled
`libamd_smi_python.so` and disables system fallback. The distinct SONAME
and `-Bsymbolic-functions` isolate it from the system `libamd_smi.so`.
`PYTHONPATH` takes precedence over site-packages, so point it only at the
wrapper you intend to use.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `AMDSMI_LIB_OVERRIDE` | Absolute path to a `libamd_smi*.so` to load **instead of** the auto-detected one. Intended for local development against an in-tree build (e.g. `AMDSMI_LIB_OVERRIDE=$PWD/build/libamd_smi.so python3 -c "import amdsmi"`) and for ABI-compatibility tests that need to point the wrapper at a curated alternate library. When set, it takes precedence over both the pip-bundled and system libraries. |

The legacy PyPI wrappers do not implement `AMDSMI_LIB_OVERRIDE` or the
current missing-library behavior described below.

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
