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

## What Triggers the Cascade

The full cascade applies when you **add or change a public function** (a new
`amdsmi_*` signature, or a changed one). That is what the name-grep below can
track across layers.

Non-function public-API edits do **not** all cascade the same way:

| Change | Cascade path |
|--------|--------------|
| New / changed `amdsmi_*` **function** | Full path below (name is grep-able in every name-bearing layer) |
| New **enum value / struct field / macro** in `amdsmi.h` | Header + impl + regenerate wrapper; interface/CLI only if user-facing. Not name-grep-able per layer |
| Bug fix or behavior change inside an existing function | Impl + tests; docs/changelog if user-visible. No new wrapper/CLI entry |
| Doxygen / comment-only edit | Header (or docs) only |

Changes to a **function** propagate through all layers in order:

1. `include/amd_smi/amdsmi.h` — C header declaration
2. `src/amd_smi/amd_smi.cc` — C++ implementation
3. `tools/generator.py` — wrapper generator (parses header; no per-function edit — verify by regenerating)
4. `py-interface/amdsmi_wrapper.py` — **auto-generated, never edit manually**
5. `py-interface/amdsmi_interface.py` — Python API
6. `amdsmi_cli/amdsmi_commands.py` — CLI commands (if user-facing)
7. `docs/` — documentation
8. `rust-interface/` and `goamdsmi_shim/` — bind `amdsmi_*` directly; update if the new function must reach Rust/Go consumers

Regenerate the wrapper with `tools/update_wrapper.sh`

## Quick Check

Greps only the five **name-bearing** layers. Layer 3 (`generator.py`) is a
generic parser with no per-function entry (verify it instead by regenerating the
wrapper), and layer 7 (`docs/`) is prose — check both separately, they are NOT
covered by this grep.

```bash
FUNC="amdsmi_get_gpu_new_feature"
grep -n "$FUNC" include/amd_smi/amdsmi.h src/amd_smi/*.cc \
  py-interface/amdsmi_wrapper.py py-interface/amdsmi_interface.py amdsmi_cli/*.py \
  rust-interface/src/*.rs goamdsmi_shim/smiwrapper/*
```

Missing results in a name-bearing layer = cascade gap. A clean grep does **not**
prove layers 3 and 7 are done.

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
| `rust-interface/` | `pub fn amdsmi_*` binding added if the function must reach Rust consumers |
| `goamdsmi_shim/` | `amdsmi_*` call added if the function must reach the Go shim |

# Tools (`tools/`)

| Tool | Purpose |
|------|---------|
| `generator.py` | Parses `amdsmi.h`, emits ctypes wrapper code |
| `update_wrapper.sh` | Regenerates `py-interface/amdsmi_wrapper.py` (Docker + `generator.py`) |
| `update_rust_wrapper.sh` | Regenerates Rust bindings |
| `run-clang-tidy.sh` | Runs clang-tidy on C++ sources |

# Tests

Full map: [tests/README.md](../../tests/README.md). Design rationale:
[docs/conceptual/test-design.md](../../docs/conceptual/test-design.md).
Python runner details: [tests/python/README.md](../../tests/python/README.md).

| Suite | Path | Runner | Hardware |
|-------|------|--------|----------|
| C++ GTest | `tests/amd_smi_test/{unit,functional}/` | `build/tests/amd_smi_test/amdsmitst` | `functional/` only |
| Python unit | `tests/python/unit/` | `tests/python/unit_tests.py` | No |
| Python functional | `tests/python/functional/` | `tests/python/integration_test.py` | Yes |
| Python CLI | `tests/python/cli/` | `tests/python/cli_unit_test.py` | Yes (installed `amd-smi`) |
| Packaging guards | `tests/python/test_*_guard.py`, `test_packaging_scriptlets.py`, `test_abi_compat.py` | `python3` (stdlib only) | No |
| Package-manager harnesses | `tests/run_amdsmi_*.py` | `sudo python3` | No |
| Build driver | `tests/amdsmi_build/` | `sudo python3 tests/amdsmi_build/run_amdsmi_build.py` | No |
| ABI checks | `tests/abi_check/` | `abi_check.py` (CI workflow) | No |
| DME integration | `tests/dme_integration/` | `PYTHONPATH=tests python3 -m dme_integration` | No |
| API summary | `tests/api_summary.py` | `python3` | No |

The three Python runners require root and resolve the `amdsmi` package via
`AMDSMI_PATH` → `ROCM_HOME` → `ROCM_PATH` → `/opt/rocm`. Each discovers only its
own subtree. Leaf `test_*.py` files have no `sys.path` bootstrap and are not
directly runnable, so always go through a runner with `-k`. Shared flags:
`-v -q -b -k PAT -x PAT -l`.

C++ suite names are the selection mechanism: `<Component><Type>[<Operation>]`,
e.g. `GpuUnit`, `GpuFunctionalReadOnly`, `GpuFunctionalReadWrite`. Per-ASIC
exclusions come from `amdsmitst.exclude` + `detect_asic_filter.sh`.

Install remaps `tests/python/` → `<share>/amd_smi/tests/python_unittest/`, so the
historical installed path still works.

Pre-commit gates: `tests/amd_smi_test/check_test_conventions.py` (layout/naming),
`tests/check_license_headers.py`.

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
