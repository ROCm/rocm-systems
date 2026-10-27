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

- Use the existing namespace for the owning module/library, named after that
  library (matching its `lib/<name>/` directory). For example, `rocjitsu`,
  `simdojo`, and `util` each live in their own top-level namespace.
- Do not introduce new namespaces. Reuse the existing namespace hierarchy,
  including established sub-namespaces such as `rocjitsu::amdgpu`.

```cpp
// Reuse the owning library's existing top-level namespace:
//   lib/rocjitsu/ -> namespace rocjitsu
//   lib/simdojo/  -> namespace simdojo
//   lib/util/     -> namespace util
namespace simdojo {

// simdojo module code lives here

} // namespace simdojo

// Reuse existing sub-namespaces for architecture-specific code:
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

## Operating System Calls

- Production code must not call the generic `syscall()` function or enter the
  kernel through inline assembly. This applies to the simulator and its tools.
- Use the named libc wrapper (`mmap`, `close`, `clone`, etc.). Libc provides typed
  interfaces, handles architecture and kernel ABI differences, and can select
  optimized implementations such as vDSO calls.
- Code that must bypass rocjitsu's interposed symbols must use
  `libc_passthrough()` from `kmd/linux/libc_passthrough.h`. Extend that table
  when another interposed operation needs a passthrough; do not bypass it with
  a direct system call. Preserve return values, `errno`, ownership, and lock
  ordering. Resolve interposed operations with `RTLD_NEXT`; non-interposed
  operations that must share sanitizer interception with direct calls (such as
  mutex acquisition/release) use `RTLD_DEFAULT`. Use non-`noexcept` function
  pointers where libc calls must allow forced unwinding. Mark optional entries
  explicitly and provide a tested fallback or unsupported-feature result.
- Libc `read`, `write`, and `close` remain cancellation points. Internal cleanup
  and publication must defer cancellation across the complete ownership
  transaction, including removal of closed descriptors from tracking containers.
  Use `ScopedCancellationDisable` for bounded cleanup and bounded internal
  lock-acquisition parking that must preserve `noexcept` acquisition and waiter
  cleanup. Keep blocking RPCs and cancellable application waits outside that
  guard. Application forwarding retains libc cancellation behavior.
  Asynchronous cancellation is not supported by these C++ paths.
- When a system call has no libc wrapper, use an equivalent library facility
  (for example, libc polling for interruptible waits). Resolve optional newer
  libc wrappers at runtime and handle their absence explicitly; never hard-code
  syscall numbers as a compatibility fallback.
- Keep loader initialization, fork-child, and signal-handler paths within their
  allocation, locking, and async-signal-safety constraints. Resolve passthroughs
  before entering such paths. Before table initialization, mapping hooks use the
  allocation-free libc ELF lookup in `bootstrap_mapping_symbol`; calling `dlsym`
  there can re-enter allocator initialization on older libc versions.
- Tests may use `syscall()` when deliberately exercising behavior outside the
  interposer or observing kernel state. Keep those calls confined to tests and
  explain why bypassing the named wrapper is necessary. Prefer libc wrappers
  for ordinary test setup and cleanup.

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
