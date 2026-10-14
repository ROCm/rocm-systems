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
  (`ColumnLimit: 90`).
- Respect `.clang-tidy`. It enables an explicit list of checks, not entire
  check groups. Do not pass `--checks`, add a NOLINT suppression, or edit
  `.clang-tidy` to hide a finding.
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
