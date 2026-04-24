---
description: "Use when: modifying CMakeLists.txt, packaging, RPM/DEB scripts, build configuration, cmake options."
---
# Build & Packaging

## CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `BUILD_TESTS` | OFF | C++ GTest suite |
| `BUILD_EXAMPLES` | OFF | Example programs |
| `BUILD_CLI` | ON | `amd-smi` CLI tool |
| `BUILD_WRAPPER` | OFF | Regenerate `amdsmi_wrapper.py` |
| `ENABLE_ESMI_LIB` | ON | Vendored E-SMI (CPU monitoring) |

## Packaging Paths

| Format | Path | Files |
|--------|------|-------|
| RPM | `RPM/` | `post.in`, `postun.in`, `preun.in` |
| DEB | `DEBIAN/` | `postinst.in`, `prerm.in`, `changelog.in`, `copyright.in` |
| pip (CLI) | `pyproject.toml` | Root-level |
| pip (bindings) | `py-interface/pyproject.toml.in` | Template, filled by CMake |

## Version

Defined in `include/amd_smi/amdsmi.h` (`AMDSMI_LIB_VERSION_MAJOR/MINOR/RELEASE`).
Extracted by `cmake_modules/utils.cmake` → `get_version_from_file()`.
