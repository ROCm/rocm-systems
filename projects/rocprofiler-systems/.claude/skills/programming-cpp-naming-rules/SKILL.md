---
name: programming-cpp-naming-rules
description: Use when creating or renaming a C++ file, type, or namespace — folder matches namespace, no redundant prefixes
---

# C++ Naming Rules

Use this skill when creating new C++ files or refactoring file structure.
Identifier spelling (case, prefixes, minimum length) is enforced by
`.clang-tidy` and summarized in `.claude/rules/cpp-conventions.md`.

## Folder Structure = Namespace

Folder structure should correspond to namespaces. Don't repeat namespace prefixes in filenames.

```cpp
// ❌ BAD: amd_smi/amd_smi_driver.hpp
namespace amd_smi
{
class amd_smi_driver
{};  // Redundant prefix
}  // namespace amd_smi

// ✅ GOOD: amd_smi/driver.hpp
namespace amd_smi
{
class driver
{};  // Clean, folder provides context
}  // namespace amd_smi
```

## Rules

1. **No redundant prefixes**: If file is in `amd_smi/`, don't name it `amd_smi_driver.hpp` - use `driver.hpp`
2. **Namespace = folder path**: `foo/bar/baz.hpp` → `namespace foo::bar { class baz {} }`
3. **Class name = file name**: `driver.hpp` contains `class driver` or `struct driver`
4. **One class per file** (when practical): Easier to find and maintain

## Examples

| Folder Path | File Name | Namespace | Class/Struct |
| ------------- | ----------- | ----------- | -------------- |
| `amd_smi/` | `driver.hpp` | `amd_smi` | `driver` |
| `amd_smi/gpu/` | `metrics.hpp` | `amd_smi::gpu` | `metrics` |
| `core/utils/` | `string_helper.hpp` | `core::utils` | `string_helper` |
| `network/http/` | `client.hpp` | `network::http` | `client` |

## In this repository

- New code: the folder under `source/lib/` maps to the namespace below `rocprofsys`:
  `source/lib/backends/hipfile/backend.hpp` declares
  `rocprofsys::backends::hipfile::backend`. `core/`, `common/`, and `rocprof-sys/`
  predate this; do not rename their namespaces to match. Tests add a `testing`
  namespace.
- Use nested namespace syntax (`namespace a::b`) and `#pragma once`.
- Headers use `.hpp`.
