---
name: api-cascade-checker
description: "Verify C API changes propagate through all layers. Use when: reviewing API changes, checking cascade integrity, new amdsmi_* functions, API renames."
---

# API Cascade Checker — amd-smi

Step-by-step verification that C API changes have propagated through all required layers.

## The Cascade Path

```
include/amd_smi/amdsmi.h          (1. C header — declaration)
        ↓
src/amd_smi/amd_smi.cc            (2. C implementation)
        ↓
tools/generator.py                 (3. Wrapper generator — parses header)
        ↓
py-interface/amdsmi_wrapper.py     (4. Generated Python wrapper — ctypes bindings)
        ↓
py-interface/amdsmi_interface.py   (5. Python interface — high-level API)
        ↓
amdsmi_cli/amdsmi_commands.py      (6. CLI commands — user-facing output)
        ↓
docs/                              (7. Documentation — API reference)
```

## Verification Steps

For each new or modified `amdsmi_*` function in the diff:

### Step 1: Header (`amdsmi.h`)
- [ ] Function declared with correct signature
- [ ] Returns `amdsmi_status_t`
- [ ] Parameters follow naming conventions
- [ ] Doxygen comment present
- [ ] Added to correct section (grouped by subsystem)

### Step 2: Implementation (`amd_smi.cc`)
- [ ] Function implemented
- [ ] All parameters validated (null checks, range checks)
- [ ] Error paths return appropriate `amdsmi_status_t`
- [ ] No exceptions escape to caller

### Step 3: Generator (`tools/generator.py`)
- [ ] Generator can parse the new function signature
- [ ] If function has special types, generator handles them
- [ ] Run `tools/update_wrapper.sh` to verify (or check wrapper manually)

### Step 4: Wrapper (`amdsmi_wrapper.py`)
- [ ] ctypes binding matches C signature exactly
- [ ] Parameter types correct (pointers, structs, enums)
- [ ] Return type is `amdsmi_status_t`

### Step 5: Interface (`amdsmi_interface.py`)
- [ ] High-level Python function exists
- [ ] Calls wrapper correctly
- [ ] Converts C types to Python types (handles → ints, structs → dicts/namedtuples)
- [ ] Raises `AmdSmiException` on error status
- [ ] Docstring present with parameter descriptions

### Step 6: CLI (`amdsmi_commands.py`)
- [ ] If user-facing: CLI command or flag exposes the new data
- [ ] Output format consistent with existing commands
- [ ] JSON output includes the new field
- [ ] Help text updated in `amdsmi_parser.py`

### Step 7: Documentation (`docs/`)
- [ ] API reference updated
- [ ] If breaking change: migration guide added
- [ ] CLI help text matches actual behavior

## Common Failures

| Failure | Impact | Severity |
|---------|--------|----------|
| Function in header but not implemented | Link error | ❌ BLOCKING |
| Wrapper out of sync with header | Runtime crash | ❌ BLOCKING |
| Interface missing new function | Python users can't access feature | ❌ BLOCKING |
| CLI doesn't expose new data | Feature incomplete but not broken | ⚠️ IMPORTANT |
| Docs not updated | Users can't discover feature | ⚠️ IMPORTANT |
| Doxygen comment missing | Generated docs incomplete | 💡 SUGGESTION |

## Quick Check Command

To find cascade gaps, search for the function name across all layers:

```bash
FUNC="amdsmi_get_gpu_new_feature"
echo "=== Header ===" && grep -n "$FUNC" include/amd_smi/amdsmi.h
echo "=== Impl ===" && grep -rn "$FUNC" src/amd_smi/
echo "=== Wrapper ===" && grep -n "$FUNC" py-interface/amdsmi_wrapper.py
echo "=== Interface ===" && grep -n "$FUNC" py-interface/amdsmi_interface.py
echo "=== CLI ===" && grep -rn "$FUNC" amdsmi_cli/
echo "=== Docs ===" && grep -rn "$FUNC" docs/
```

Any layer missing the function name indicates a cascade gap.
