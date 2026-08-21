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

protected:
  // Available to derived buffer types.
  void reallocate(std::size_t new_capacity);

private:
  std::size_t current_size_;
};
```

## Naming

- Prefix global constants with `k` (e.g. `kMaxThreads`).
- Use CamelCase for data types and snake_case for methods and members.
- Suffix non-public data members (private and protected) with a trailing
  underscore (e.g. `current_size_`); public members (such as POD `struct` fields)
  take no suffix.
- Use clear, descriptive names. No single-letter variables.

```cpp
constexpr int kMaxThreads = 64;

class ThreadPool {          // CamelCase type
public:
  void submit_task();       // snake_case method

private:
  int active_thread_count_; // snake_case member, trailing _
};
```

## Documentation & Comments

- Write long, verbose descriptions for variables and functions.
- Prefer clarity over conciseness.
- When documenting public APIs, use Doxygen `@brief` / `@details`.
- Comment the *why*, not the *what*. Skip comments that just restate the code.
- Avoid decorative comment lines (e.g. `// ----`, `// ====`).

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
- Do not scatter code across ad-hoc namespaces. Nest a sub-namespace only when
  genuinely needed : for example, for arch/ISA specificity (`rocjitsu::amdgpu`).

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

- Always target C++20 and use the STL; prefer modern features such as concepts,
  ranges, and `std::format`.
- Use `typename` instead of `class` when declaring template type parameters. This
  makes occurrences of the `class` keyword easier to find when searching the
  codebase.
- If you need performance and the STL is a poor fit, build custom data structures
  in `util`.
- Never use `#define`, C conventions, or macros unless absolutely necessary.
- When you must use C standard library headers, prefer the `<c*>` variants.

```cpp
#include <cstdint> // not <stdint.h>
#include <cstdio>  // not <stdio.h>

template <typename ValueType> // not template <class ValueType>
class Container {};
```

- Use `auto` only when the type is obvious from the immediate context (e.g. the
  RHS spells out the type, such as a constructor call or a cast). Do not use
  `auto` when the type can't be determined by looking at the line itself.

```cpp
auto x = foo(); // BAD, type not obvious

auto x = std::make_unique<Foo>(); // OK, can tell from the RHS

Foo foo = static_cast<Foo>(bar); // OK but could use auto instead to avoid repetitiveness
```

## Logging

- Use `Logger` from `util/log.h`. `Logger::print<>()` and the per-group helpers
  are compiled out unless the group is enabled at configure time
  (`cmake -DRJ_LOG_GROUPS=...`; the default is `OFF`, see `cmake/rj_log.cmake`),
  so use them for tracing only.
- Use `Logger::warn()` for messages that must appear in a default build.
- Do not add `fprintf`, `printf`, or `std::cerr` to library code. Existing
  carve-outs are the CLI front end (`tools/`, user-facing `std::cerr` errors)
  and the ROCR hook layer (`lib/rocjitsu/src/rocjitsu/hooks/`, fatal `stderr`
  diagnostics that must survive a log-groups-OFF build, plus an
  async-signal-safe `::write` backtrace path).

## Error Handling

- Expected failures such as a rejected instruction encoding return `Result` or
  `FailureOr<T>`. Accept a `DiagnosticEmitter` when the caller needs the reason;
  its sink is non-owning and must not outlive its callable.
- Exceptions are only for unrecoverable initialization, configuration, or
  execution failures. Exception types live in `util/except.h`.
- Do not throw in simulation hot paths (event handlers, instruction execution,
  cache lookups).
- Do not add `try`/`catch` except at a boundary that must translate an error
  (e.g. the C API layer). Assume code is not exception-safe unless documented.

## Formatting

- The repo uses pre-commit hooks : clang-format (C++), black (Python), and
  gersemi (CMake). The config lives at the repo root
  (`rocm-systems/.pre-commit-config.yaml`). Install once, then run before every
  commit:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
