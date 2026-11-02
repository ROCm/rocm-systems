---
name: cpp-conventions
description: C++ hard rules for rocprofiler-systems — standard, formatting, linting, legacy dependencies
---

# C++ Conventions — rocprofiler-systems

## Standard

- First-party project code targets C++20. Do not use C++23-only features.
- Respect an explicitly older standard in a self-contained example or vendored
  dependency; do not modernize `external/`.
- Prefer C++20 facilities over pre-C++20 workarounds: concepts over SFINAE,
  ranges/views, `std::span`, `std::format`, `consteval`/`constinit`,
  designated initializers, `std::erase`/`std::erase_if`.

## Formatting and linting

- Format C/C++ with clang-format 18 and the repository `.clang-format`
  (`ColumnLimit: 90`). Braces go on the next line, there is no space before
  `(`, and pointers and references bind left (`int*`, `const T&`). A top-level
  function's return type sits on its own line.
- Respect `.clang-tidy`. It enables an explicit list of checks, not entire
  check groups. Do not pass `--checks`, add a NOLINT suppression, or edit
  `.clang-tidy` to hide a finding. The one exception is the library-name
  mirroring case in the naming list below.
- Name identifiers the way `readability-identifier-naming` is configured.
  Do not copy a trailing underscore or a PascalCase type from an older file.
  - Types, functions, methods, parameters, variables, enumerators, and public
    members are `lower_case`.
  - Private and protected data members use an `m_` prefix: `m_mem`, never
    `mem_`.
  - Every `constexpr` variable uses a `k_` prefix, in every scope. Static and
    global constants do too.
  - A mutable namespace-scope variable, including a file-scope `static`, uses
    a `g_` prefix. A function-local `static` uses an `s_` prefix.
  - An abstract class is `lower_case` and ends in `_interface`.
  - Template parameters are `CamelCase`. Macros are `UPPER_CASE`.
  - Parameters, local variables, bindings, and exception names are at least
    3 characters. Allowed short names: loop counters `i`, `j`, `k`; a
    parameter `n`; an exception `e`.
  - Keep a library's own names: `SetUp`, `TearDown`, and `is_transparent`.
  - In `source/lib/backends/<library>/` and its `tests/mock_*.hpp`, a name
    that mirrors an external library name (a constant like
    `CALLBACK_PHASE_ENTER`, or a struct field that must match the library
    struct) keeps the library's spelling. Mark it with
    `// NOLINTNEXTLINE(readability-identifier-naming)`, or wrap a block in
    `NOLINTBEGIN`/`NOLINTEND` with a reason. This is the only allowed NOLINT
    for naming.
- Mark a function `[[nodiscard]]` when the caller must use the return value.
- A bare numeric literal other than `0`, `1`, `2`, `3`, or `4` is a named
  constant. The literal belongs in that constant's initializer.
- Pre-commit runs clang-format but not clang-tidy. After C++ changes, run
  `/clang-tidy`:

  ```bash
  ./scripts/clang-tidy-check.py --build-path build/debug --base develop -j $(nproc)
  ```

## Legacy timemory dependency

- **Never use timemory functions or types in new code.** `external/timemory`
  is legacy compatibility code. Prefer the standard library or a direct,
  purpose-built dependency.
- Treat changes to timemory integration as migration work; do not expand its
  API surface.

## Ownership and safety

- No raw `new`/`delete` in new code — `std::make_unique`/`std::make_shared`.
- Raw pointers mean "non-owning observer" only.
- No raw arrays — `std::array`, `std::vector`, or `std::span`.
- `override` on every overriding virtual function.
- Never call virtual functions from constructors/destructors.
- Use `noexcept` only when the function's contract guarantees it; move
  operations and `swap` should normally be `noexcept`.
- No C-style casts — `static_cast`/`dynamic_cast`/etc.
- A `reinterpret_cast` requires an explicit code comment explaining why it is
  safe; `.clang-tidy` does not enforce this rule.
