---
description: "Use when: navigating the codebase, finding source files, understanding project layout, onboarding."
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
| Tools | `tools/` | `generator.py`, `update_wrapper.sh`, `run-clang-tidy.sh` |
