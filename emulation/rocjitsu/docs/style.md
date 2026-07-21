# rocjitsu Code Style Guide

Conventions for writing C++ in the `rocjitsu` codebase. Prefer clarity and
consistency over cleverness.

## Types (`struct` vs `class`)

- Use a `struct` **only** for plain-old-data (POD) types where every member is
  public.
- Use a `class` whenever you need constructors/destructors or any other non-POD
  behavior.
- Within a `class`, use a single `public:`, `protected:`, and `private:` section
  (only the ones you need), and always in that order.

```cpp
// POD: all members public, no behavior -> struct
struct Point {
  int x;
  int y;
};

// Has behavior (ctor/dtor/methods) -> class, sections in order
class Buffer {
public:
  Buffer();
  ~Buffer();
  void resize(std::size_t new_size);

private:
  std::size_t current_size;
};
```

## Naming

- Prefix global constants with `k` (e.g. `kMaxThreads`).
- Use CamelCase for data types and snake_case for methods and members.

```cpp
constexpr int kMaxThreads = 64;

class ThreadPool {         // CamelCase type
public:
  void submit_task();      // snake_case method

private:
  int active_thread_count; // snake_case member
};
```

## Documentation & Comments

- Write long, verbose descriptions for variables and functions.
- Prefer clarity over conciseness.

## Headers

- Use `#pragma once` for include guards.

```cpp
#pragma once

// ... header contents ...
```

## Namespaces

- Use one namespace per module/library, named after that library (matching its
  `lib/<name>/` directory). For example, `rocjitsu`, `simdojo`, and `util` each
  live in their own top-level namespace.
- Do not scatter code across ad-hoc namespaces (agents absolutely destroyed this
  rule). Nest a sub-namespace only when genuinely needed — for example, for
  arch/ISA specificity (`rocjitsu::amdgpu`).

```cpp
// Each library gets its own top-level namespace, named after its lib/<name>/ dir:
//   lib/rocjitsu/ -> namespace rocjitsu
//   lib/simdojo/  -> namespace simdojo
//   lib/util/     -> namespace util
namespace simdojo {

// simdojo module code lives here

} // namespace simdojo

// Nest a sub-namespace only when justified, e.g. arch/ISA specificity:
namespace rocjitsu::amdgpu {

// gfx-specific code lives here

} // namespace rocjitsu::amdgpu
```

## Language, STL & Preprocessor

- Always target C++20 and use the STL.
- If you need performance and the STL is a poor fit, build custom data structures
  in `util`.
- Never use `#define`, C conventions, or macros unless absolutely necessary.
- When you must use C standard library headers, prefer the `<c*>` variants.

```cpp
#include <cstdint> // not <stdint.h>
#include <cstdio>  // not <stdio.h>
```
