---
description: "Use when: reviewing API changes, adding/modifying amdsmi_* functions, checking cascade integrity, API renames, new C API functions."
---
# API Cascade

Changes to the public C API must propagate through all layers in order:

1. `include/amd_smi/amdsmi.h` — C header declaration
2. `src/amd_smi/amd_smi.cc` — C++ implementation
3. `tools/generator.py` — wrapper generator (parses header)
4. `py-interface/amdsmi_wrapper.py` — auto-generated ctypes (DO NOT EDIT)
5. `py-interface/amdsmi_interface.py` — Python API
6. `amdsmi_cli/amdsmi_commands.py` — CLI commands
7. `docs/` — documentation

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

## Severity

| Gap | Impact | Severity |
|-----|--------|----------|
| Header but not implemented | Link error | BLOCKING |
| Wrapper out of sync | Runtime crash | BLOCKING |
| Interface missing function | Python users blocked | BLOCKING |
| CLI doesn't expose data | Feature incomplete | IMPORTANT |
| Docs not updated | Undiscoverable | IMPORTANT |
