# AMD SMI packaging and install paths

AMD SMI ships through several delivery channels. This page describes each one,
how the Python module locates the native library in each, which combinations
are supported, and how upgrades and downgrades behave. It is the reference for
the loader contract that `py-interface/amdsmi_wrapper.py` implements.

## Delivery paths

| Path | Native library (`.so`) | Python module | How the module finds the `.so` |
| ---- | ---------------------- | ------------- | ------------------------------ |
| System package (deb/rpm) | `/opt/rocm/lib/libamd_smi.so.<MAJOR>` (+ `ld.so.conf.d` entry) | Installed into the system interpreter's `site-packages`/`dist-packages` **and** `share/amd_smi` | SONAME via the dynamic linker |
| TheRock native package (`amdrocm-amdsmi`) | `/opt/rocm/core-X.Y/lib/libamd_smi.so.<MAJOR>` | `/opt/rocm/core-X.Y/share/amd_smi/amdsmi` only; importable after adding `share/amd_smi` to `PYTHONPATH` | Resolved relative to the wrapper (`../../../lib`) |
| TheRock tarball | `<root>/lib/libamd_smi.so.<MAJOR>` in the extracted tree | `<root>/share/amd_smi/amdsmi`; add `share/amd_smi` to `PYTHONPATH` for direct imports | Resolved relative to the wrapper (`../../../lib`) |
| ROCm via pip (TheRock `rocm-sdk-core`) | `<root>/lib/libamd_smi.so.<MAJOR>` | `<root>/share/amd_smi/amdsmi`; add `share/amd_smi` to `PYTHONPATH` for direct imports | Resolved relative to the wrapper (`../../../lib`) |
| ROCm via pip in a venv | Same as above, inside the venv | Same as above, inside the venv | Same as above |
| Community-built PyPI `amdsmi` releases through 7.0.2 | Not bundled; requires a matching ROCm `libamd_smi.so` | Interpreter `site-packages` | Legacy loader checks `ROCM_HOME` before `ROCM_PATH`, then the dynamic linker and `/opt/rocm/lib` |
| Bundled-library wheel (`BUILD_PYTHON_WHEEL=ON`) | Bundled `libamd_smi_python.so` next to the wrapper | Interpreter `site-packages` | The bundled `.so`; system fallback is disabled |

The community-built PyPI releases through 7.0.2 predate the bundled-library
wheel and use ROCm release numbers. They require that release's native library
and are not compatible with arbitrary newer AMD SMI libraries. Prefer the
wrapper shipped with your ROCm installation. Wheels built from this source
use the AMD SMI library version instead.

Bundling AMD SMI does not eliminate its native runtime prerequisites. A raw
`BUILD_PYTHON_WHEEL=ON` wheel requires `libnl-3`, `libnl-genl-3`, and `libmnl`
from the system; `tools/build_wheel.py --repair` vendors those libraries into
the repaired manylinux wheel. Both need `libdrm_amdgpu` for DRM-backed GPU
queries, which load it dynamically. The repaired wheel does not bundle it.
Install `libdrm-amdgpu1` on Debian/Ubuntu, `libdrm` on RHEL/AlmaLinux/Rocky,
or `libdrm_amdgpu1` on SLES/openSUSE. Without it, discovery can succeed while
some query results are unavailable or incorrect.

## The loader

The current `py-interface/amdsmi_wrapper.py` selects the library in this order
(the legacy PyPI wrappers do not implement this contract):

1. `AMDSMI_LIB_OVERRIDE` — explicit path, for ABI tests.
2. A bundled `libamd_smi_python.so` next to the wrapper — the bundled-library wheel.
3. The SONAME resolved relative to the wrapper (`parents[3]/lib`) — the TheRock
   `share/amd_smi` layout, where a venv has no `ld.so.conf.d` entry.
4. The bare SONAME via the dynamic linker — the system deb/rpm.

Steps 3 and 4 are skipped when `_AMDSMI_ALLOW_SYSTEM_FALLBACK` is `False`. The
committed wrapper and the system package keep it `True`; the wheel build flips
it to `False`, so the bundled-library wheel does not automatically load a
system `libamd_smi.so` (which could be a different version and would risk
symbol conflicts inside processes such as PyTorch or JAX that ship their own
copy). An explicit `AMDSMI_LIB_OVERRIDE` still takes precedence.

The wheel's `libamd_smi_python.so` has a distinct SONAME and is linked with
`-Bsymbolic-functions`, so the system and wheel libraries can be loaded in the
same process without the dynamic linker interposing one on the other.

## Two installed copies (system package)

The deb/rpm installs the module into **both** the interpreter's site-packages
and `share/amd_smi`. Both are required:

- site-packages makes a plain `import amdsmi` work.
- `share/amd_smi` is captured by the TheRock artifact flow (which packages only
  `/opt/rocm`, so it cannot reach the `/usr` site-packages tree) and is used by
  downstream tools that `sys.path.insert(ROCM_PATH + "/share/amd_smi")`.

A redirector or symlink is not viable because TheRock ships only `/opt/rocm`.
The build harness runs a guard (`tests/run_amdsmi_dual_copy_test.py`) asserting
the two copies stay byte-identical, so drift fails a build instead of shipping.

## Coexistence and precedence

A user installs one delivery path. When a bundled-library wheel is installed
alongside a system package, the wheel wins and the package uninstall does not
remove it, because they live in separate, file-manager-owned trees and
`sys.path` favors the wheel:

| Installed together | `import amdsmi` resolves to | Package uninstall removes the wheel? |
| ------------------ | --------------------------- | ------------------------------------ |
| deb + pip wheel (Debian) | wheel in `/usr/local/.../dist-packages` (precedes `/usr/lib`) | No — dpkg removes only its own files |
| rpm + pip `--user` wheel (RHEL) | wheel in `~/.local/.../site-packages` (precedes system) | No — rpm removes only its own files |
| deb/rpm + venv wheel | wheel in the venv (isolated) | No — separate tree |

The precedence above is the default `sys.path` ordering between install
locations. A global `PYTHONPATH` overrides it: Python searches `PYTHONPATH`
entries before any install location, so `import amdsmi` resolves there
regardless of where the wheel or package installed. Some ROCm container images
set `PYTHONPATH=/opt/rocm/share/amd_smi`, which makes the `share/amd_smi` copy
win over a pip wheel in that environment. Unset `PYTHONPATH` (or point it at the
copy you want) to restore the install-location precedence.

## Support matrix

Legend: ✅ supported and tested · 🟡 supported, pick one recommended · ⛔ unsupported.

### Single path

| Path | `amd-smi` CLI | `import amdsmi` |
| ---- | ------------- | --------------- |
| deb/rpm | ✅ | ✅ |
| TheRock tarball | ✅ | ✅ (with `PYTHONPATH` set) |
| ROCm pip | ✅ | ✅ (with `PYTHONPATH` set) |
| ROCm pip in venv | ✅ | ✅ (with `PYTHONPATH` set) |
| Community-built PyPI wrapper through 7.0.2 | ⛔ (Python bindings only) | Requires its matching older ROCm library; prefer the ROCm-shipped wrapper |
| Bundled-library wheel | ⛔ (Python bindings only) | ✅ (native runtime prerequisites above) |

### Two paths together

| Combination | Coexist? |
| ----------- | -------- |
| deb/rpm + bundled-library wheel | ✅ (wheel wins; package uninstall keeps the wheel) |
| deb/rpm + ROCm pip, same interpreter | 🟡 (discouraged; two library families) |
| Bundled-library wheel + ROCm pip | 🟡 (discouraged) |
| TheRock tarball + bundled-library wheel | ✅ (wheel uses its bundled `.so` unless `PYTHONPATH` selects the tarball wrapper) |
| ROCm pip + ROCm pip venv | ✅ (venv wins while active) |

### Unsupported

- Two system packages of different major SOVERSION on one prefix.
- Relying on a tarball's Python wrapper without adding `share/amd_smi` to `PYTHONPATH`.
- A bundled-library wheel automatically falling back to a system `/opt/rocm`
  library (blocked by design; `AMDSMI_LIB_OVERRIDE` is an explicit exception).
- A legacy PyPI wrapper with a native library from a different ROCm release.

## Upgrade and downgrade (deb/rpm)

| Transition | Behavior |
| ---------- | -------- |
| pre-7.14 (pip-era) → 7.14+ package | The old package's prerm still `pip uninstall`s the legacy module and removes its `.pth`; the new package owns the site-packages files. |
| 7.14+ → 7.14+ | Plain file replacement by the package manager. |
| 7.14+ → pre-7.14 (downgrade) | The old package re-adds the pip install; a user-installed bundled-library wheel in `/usr/local` or `~/.local` still wins and survives. |
| package removed, then bundled-library wheel | Package removal deletes only its own files; the wheel includes its own `.so` and disables fallback. |
| package removed, then legacy PyPI wrapper | The wrapper still requires a matching native library; installing it alone does not replace the removed library. |

## RPM interpreter dependency

On RPM distros the module installs into a version-specific site-packages
(e.g. `/usr/lib64/python3.9/site-packages`). The package therefore declares a
dependency on the matching interpreter so it installs only where that
interpreter (and thus the baked path) exists:

- RHEL/CentOS/Fedora/AlmaLinux/AzureLinux: `python(abi) = X.Y`.
- SLES/openSUSE: `pythonXY` (e.g. `python311`).

Debian's `dist-packages` is version-agnostic, so the deb keeps the loose
`python3 (>= 3.6.8)` dependency and the `#!/usr/bin/python3` CLI shebang serves
every python3 minor.
