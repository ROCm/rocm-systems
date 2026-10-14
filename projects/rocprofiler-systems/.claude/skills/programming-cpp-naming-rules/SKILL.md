---
name: programming-cpp-naming-rules
description: C++ naming — folder matches namespace, private members m_, constants k_, abstract types end in _interface
---

# C++ Naming Rules

Use this skill when creating new C++ files or refactoring file structure.
Identifier spelling is the `.clang-tidy` configuration in `cpp-conventions`:
`lower_case` types and functions, private members `m_mem` (never `mem_`),
`constexpr` and other constants `k_`, globals `g_`, function-local statics
`s_`, abstract classes ending in `_interface`, template parameters
`CamelCase`, macros `UPPER_CASE`. Parameters, locals, bindings, and
exception names are at least 3 characters.

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

## Nested Namespaces

Use nested namespace syntax (C++17):

```cpp
// ✅ GOOD: nested namespace
namespace network::http
{
class client
{
    // ...
};
}  // namespace network::http

// ❌ AVOID: Old style (more verbose)
namespace network
{
namespace http
{
class client
{
    // ...
};
}  // namespace http
}  // namespace network
```

## Include Guards / #pragma once

Match the full path in include guards:

```cpp
// File: amd_smi/gpu/metrics.hpp

#pragma once  // Preferred

// OR traditional include guard matching full path:
#ifndef AMD_SMI_GPU_METRICS_HPP
#    define AMD_SMI_GPU_METRICS_HPP

// ...

#endif  // AMD_SMI_GPU_METRICS_HPP
```

## Summary

- **Folder = namespace context** - don't repeat it in filenames
- **Filename = class name** - easy to find
- **Path tells the full story**: `project/module/submodule/class.hpp`
