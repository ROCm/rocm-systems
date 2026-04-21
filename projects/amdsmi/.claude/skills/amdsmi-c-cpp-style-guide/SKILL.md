---
name: amdsmi-c-cpp-style-guide
description: "C/C++ style guide for amd-smi. Use when: writing C/C++ code, reviewing C/C++ PRs, checking C/C++ style, naming conventions, error handling patterns, API design."
---

# C/C++ Style Guide — amd-smi

Enforced by **clang-format** (`.clang-format`: Google style, 100 col, left pointer alignment) via pre-commit.

## Core Principles

- Every public API function returns `amdsmi_status_t`
- Fail fast — return error codes, don't silently continue
- No memory leaks — RAII in C++, explicit cleanup in C

---

## Naming Conventions

### C Public API (`include/amd_smi/amdsmi.h`)

| Element | Pattern | Example |
|---------|---------|---------|
| Functions | `amdsmi_<verb>_<subsystem>_<noun>()` | `amdsmi_get_gpu_temperature()` |
| Return type | Always `amdsmi_status_t` | — |
| Enums | `amdsmi_<name>_t` | `amdsmi_temperature_type_t` |
| Structs | `amdsmi_<name>_t` | `amdsmi_gpu_metrics_t` |
| Handles | `amdsmi_<thing>_handle` | `amdsmi_processor_handle` |
| Constants | `AMDSMI_<CATEGORY>_<NAME>` | `AMDSMI_MAX_DEVICES` |

### C++ Internal (`src/amd_smi/`)

| Element | Pattern | Example |
|---------|---------|---------|
| Classes | PascalCase with `AMDSmi` prefix | `AMDSmiSystem`, `AMDSmiGPUDevice` |
| Methods | `PascalCase` | `GetTemperature()` |
| Helpers | `snake_case` | `parse_sysfs_value()` |
| Member vars | `snake_case_` (trailing underscore) | `device_count_` |
| Local vars | `snake_case` | `gpu_index` |

---

## Error Handling

```cpp
// Good — check every call, return on failure
amdsmi_status_t ret;
ret = amdsmi_get_processor_handles(nullptr, &count);
if (ret != AMDSMI_STATUS_SUCCESS) return ret;

// Bad — ignoring return value
amdsmi_get_processor_handles(nullptr, &count);
```

### Rules
- Always check return values from `amdsmi_*` calls
- Propagate errors up — don't swallow them
- Use `AMDSMI_STATUS_INVAL` for bad input, `AMDSMI_STATUS_NOT_SUPPORTED` for missing features
- No exceptions crossing the C API boundary — catch in C++ impl, return status code
- Resource cleanup on error paths (use RAII or goto-cleanup pattern)

---

## Memory & Resource Management

- C++ internal: RAII (smart pointers, scoped handles)
- C public API: caller-allocated buffers with size parameters
- No raw `new`/`delete` in C++ code — use `std::unique_ptr`, `std::vector`
- File handles: use `ScopedFd` (`src/amd_smi/scoped_fd.cc`) or RAII wrappers
- sysfs reads: validate file exists before opening, handle partial reads

---

## Header Guidelines

- Public headers (`include/amd_smi/`): C-compatible, no C++ constructs
- Use include guards: `#ifndef AMD_SMI_HEADER_NAME_H_`
- Forward-declare where possible to minimize includes
- Internal headers: OK to use C++ features

---

## Code Organization

- One class per file (implementation in `.cc`, declaration in `.h`)
- Keep functions < 50 lines, classes < 500 lines
- No magic numbers — use named constants or enums
- Group related functions in the same file
- New source files must be added to `src/CMakeLists.txt`

---

## Formatting

Enforced by clang-format. Key settings:
- **100 column limit**
- **Google style base**
- **Left pointer alignment** (`int* ptr`, not `int *ptr`)
- **4-space indent for continuations**

Run before committing: `pre-commit run clang-format --all-files`

---

## Review Checklist

When reviewing C/C++ code, verify:

- [ ] All public API functions return `amdsmi_status_t`
- [ ] Naming follows conventions (C public: `amdsmi_*`, C++ internal: PascalCase)
- [ ] Return values checked on all `amdsmi_*` calls
- [ ] No memory leaks — RAII or explicit cleanup on all paths
- [ ] No exceptions crossing C API boundary
- [ ] No raw `new`/`delete` — use smart pointers
- [ ] File handles properly closed (RAII or scoped)
- [ ] No magic numbers
- [ ] New files added to CMakeLists.txt
- [ ] clang-format clean
