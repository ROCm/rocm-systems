# AMD SMI packaging and install paths

AMD SMI ships through several delivery channels. This page describes each one,
how the Python module locates the native library in each, which combinations
are supported, and how upgrades and downgrades behave. It is the reference for
the loader contract that `py-interface/amdsmi_wrapper.py` implements.

## Delivery paths

| Path | Native library (`.so`) | Python module | How the module finds the `.so` |
| ---- | ---------------------- | ------------- | ------------------------------ |
| System package (deb/rpm) | `/opt/rocm/lib/libamd_smi.so.<MAJOR>` (+ `ld.so.conf.d` entry) | Installed into the system interpreter's `site-packages`/`dist-packages` **and** `share/amd_smi` | SONAME via the dynamic linker |
| Tarball | Present in the extracted tree | Not installed | n/a — CLI and `.so` work; `import amdsmi` is not provided |
| ROCm via pip (TheRock `rocm_sdk_core`) | `<root>/lib/libamd_smi.so.<MAJOR>` | `<root>/share/amd_smi/amdsmi` | Resolved relative to the wrapper (`../../../lib`) |
| ROCm via pip in a venv | Same as above, inside the venv | Same as above, inside the venv | Same as above |
| PyPI wheel | Bundled `libamd_smi_python.so` next to the wrapper | Interpreter `site-packages` | The bundled `.so`; system fallback is disabled |

## The loader

`py-interface/amdsmi_wrapper.py` selects the library in this order:

1. `AMDSMI_LIB_OVERRIDE` — explicit path, for ABI tests.
2. A bundled `libamd_smi_python.so` next to the wrapper — the PyPI wheel.
3. The SONAME resolved relative to the wrapper (`parents[3]/lib`) — the TheRock
   `share/amd_smi` layout, where a venv has no `ld.so.conf.d` entry.
4. The bare SONAME via the dynamic linker — the system deb/rpm.

Steps 3 and 4 are skipped when `_AMDSMI_ALLOW_SYSTEM_FALLBACK` is `False`. The
committed wrapper and the system package keep it `True`; the wheel build flips
it to `False`, so a wheel never loads a system `libamd_smi.so` (which could be a
different version and would risk symbol conflicts inside processes such as
PyTorch or JAX that ship their own copy).

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

A user installs one delivery path. When a PyPI wheel is installed alongside a
system package, the wheel wins and the package uninstall does not remove it,
because they live in separate, file-manager-owned trees and `sys.path` favors
the wheel:

| Installed together | `import amdsmi` resolves to | Package uninstall removes the wheel? |
| ------------------ | --------------------------- | ------------------------------------ |
| deb + pip wheel (Debian) | wheel in `/usr/local/.../dist-packages` (precedes `/usr/lib`) | No — dpkg removes only its own files |
| rpm + pip `--user` wheel (RHEL) | wheel in `~/.local/.../site-packages` (precedes system) | No — rpm removes only its own files |
| deb/rpm + venv wheel | wheel in the venv (isolated) | No — separate tree |

## Support matrix

Legend: ✅ supported and tested · 🟡 supported, pick one recommended · ⛔ unsupported.

### Single path

| Path | `amd-smi` CLI | `import amdsmi` |
| ---- | ------------- | --------------- |
| deb/rpm | ✅ | ✅ |
| tarball | ✅ | ⛔ (module not installed) |
| ROCm pip | ✅ | ✅ |
| ROCm pip in venv | ✅ | ✅ |
| PyPI wheel | ✅ | ✅ |

### Two paths together

| Combination | Coexist? |
| ----------- | -------- |
| deb/rpm + PyPI wheel | ✅ (wheel wins; package uninstall keeps the wheel) |
| deb/rpm + ROCm pip, same interpreter | 🟡 (discouraged; two library families) |
| PyPI wheel + ROCm pip | 🟡 (discouraged) |
| tarball + any pip/wheel | ✅ (module comes from pip; tarball `.so` unused by the module) |
| ROCm pip + ROCm pip venv | ✅ (venv wins while active) |

### Unsupported

- Two system packages of different major SOVERSION on one prefix.
- Relying on the tarball to provide `import amdsmi`.
- A wheel loading a system `/opt/rocm` library (blocked by design).

## Upgrade and downgrade (deb/rpm)

| Transition | Behavior |
| ---------- | -------- |
| pre-7.14 (pip-era) → 7.14+ package | The old package's prerm still `pip uninstall`s the legacy module and removes its `.pth`; the new package owns the site-packages files. |
| 7.14+ → 7.14+ | Plain file replacement by the package manager. |
| 7.14+ → pre-7.14 (downgrade) | The old package re-adds the pip install; a user-installed PyPI wheel in `/usr/local` or `~/.local` still wins and survives. |
| package removed, then newest PyPI wheel | Package removal deletes only its own files; the wheel is self-contained (bundled `.so`, fallback disabled). |

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
