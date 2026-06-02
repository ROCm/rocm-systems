---
description: "Use when: reviewing API changes, adding/modifying amdsmi_* functions, checking cascade integrity, API renames, new C API functions, working with generator.py, update_wrapper.sh, amdsmi_wrapper.py, code generation, wrapper regeneration, navigating the codebase, finding source files, understanding project layout, onboarding, running tests, writing tests, checking test coverage, finding test files, modifying CMakeLists.txt, packaging, RPM/DEB scripts, build configuration, cmake options."
---
# Project Layout

| Component | Directory | Key Files |
|-----------|-----------|-----------|
| Core C++ library | `src/amd_smi/` | `amd_smi.cc` |
| NIC subsystem | `src/nic/` | AI-NIC, Broadcom NIC |
| Public C headers | `include/amd_smi/` | `amdsmi.h` (public API) |
| Python bindings | `py-interface/` | `amdsmi_interface.py`, `amdsmi_wrapper.py` (auto-generated), `amdsmi_exception.py` |
| Python CLI | `amdsmi_cli/` | `amdsmi_commands.py`, `amdsmi_parser.py`, `amdsmi_helpers.py` |
| Go shim | `goamdsmi_shim/` | |
| Rust bindings | `rust-interface/` | |
| Legacy compat | `rocm_smi/` | ROCm SMI compatibility layer |
| Vendored E-SMI | `esmi_ib_library/` | CPU monitoring (do not format/lint) |
| Build helpers | `cmake_modules/` | `utils.cmake`, `help_package.cmake` |
| Tools | `tools/` | See Tools below |

# API Cascade

Changes to the public C API must propagate through all layers in order:

1. `include/amd_smi/amdsmi.h` — C header declaration
2. `src/amd_smi/amd_smi.cc` — C++ implementation
3. `tools/generator.py` — wrapper generator (parses header)
4. `py-interface/amdsmi_wrapper.py` — **auto-generated, never edit manually**
5. `py-interface/amdsmi_interface.py` — Python API
6. `amdsmi_cli/amdsmi_commands.py` — CLI commands
7. `docs/` — documentation

Regenerate the wrapper with `tools/update_wrapper.sh`

## Quick Check

```bash
FUNC="amdsmi_get_gpu_new_feature"
grep -n "$FUNC" include/amd_smi/amdsmi.h src/amd_smi/*.cc py-interface/amdsmi_wrapper.py py-interface/amdsmi_interface.py amdsmi_cli/*.py
```

Missing results = cascade gap.

## Per-Layer Checklist

| Layer | Verify |
|-------|--------|
| `amdsmi.h` | Correct signature, `amdsmi_status_t` return, doxygen comment |
| `amd_smi.cc` | Implemented, params validated, no exceptions escaping |
| `generator.py` | Can parse the new signature |
| `amdsmi_wrapper.py` | ctypes binding matches C signature exactly |
| `amdsmi_interface.py` | Python function exists, raises `AmdSmiException` on error |
| `amdsmi_commands.py` | CLI exposes data if user-facing, JSON output includes field |
| `docs/` | API reference updated |

# Tools (`tools/`)

| Tool | Purpose |
|------|---------|
| `generator.py` | Parses `amdsmi.h`, emits ctypes wrapper code |
| `update_wrapper.sh` | Regenerates `py-interface/amdsmi_wrapper.py` (Docker + `generator.py`) |
| `update_rust_wrapper.sh` | Regenerates Rust bindings |
| `run-clang-tidy.sh` | Runs clang-tidy on C++ sources |

# Tests

| Suite | Path | Runner |
|-------|------|--------|
| C++ GTest | `tests/amd_smi_test/` | `build/tests/amd_smi_test/amdsmitst` |
| Python unit | `tests/python_unittest/unit_tests.py` | `python3` |
| Python integration | `tests/python_unittest/integration_test.py` | `python3` |
| Python CLI | `tests/python_unittest/cli_unit_test.py` | `python3` |
| Python perf | `tests/python_unittest/perf_tests.py` | `python3` |
| ABI checks | `tests/abi_check/` | CI workflow |
| API summary | `tests/api_summary.py` | `python3` |

All tests require GPU hardware. Python tests need `AMDSMI_PATH=/opt/rocm/share/amd_smi`.

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
