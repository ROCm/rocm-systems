---
name: cpp-conventions
description: C++ hard rules for rocprofiler-systems — standard, formatting, linting, legacy dependencies
---

# C++ Conventions — rocprofiler-systems

These rules are specific to this project's actual current configuration
(verified against `CMakeLists.txt`, `.clang-format`, `.clang-tidy`,
`.pre-commit-config.yaml`). They apply in addition to whatever generic
`programming-cpp*` skills are loaded.

## Standard

- Project targets **C++20** (`CMAKE_CXX_STANDARD 20`, `CMakeLists.txt`). Do
  not use C++23+-only features.
- Prefer C++20 facilities over pre-C++20 workarounds: concepts over SFINAE,
  ranges/views, `std::span`, `std::format`, `consteval`/`constinit`,
  designated initializers, `std::erase`/`std::erase_if`.

## Formatting & Linting — enforced by pre-commit, not optional

- `clang-format-18` is the formatter of record; settings are exactly
  `.clang-format` in this repo (`ColumnLimit: 90`, `Standard: Latest`). Do not
  hand-format against a different column width or style.
- `clang-tidy` runs with `.clang-tidy` in this repo. Enabled groups:
  `bugprone-*`, `cert-*`, `concurrency-*`, `cppcoreguidelines-*`, `google-*`,
  `misc-*`, `modernize-*`, `performance-*`, `readability-*`, with specific
  checks disabled (see `.clang-tidy` for the exact exclusion list — e.g.
  `cppcoreguidelines-pro-type-reinterpret-cast`,
  `cppcoreguidelines-owning-memory`, `modernize-use-trailing-return-type` are
  off). Don't assume a check is active without checking the file.
- `pre-commit install` is required per `CONTRIBUTING.md`; formatting/lint
  fixes must be re-run through pre-commit, not applied by hand and left
  unverified.

## Legacy dependency: timemory

- **Never use timemory functions or types in new code.** `external/timemory`
  is a vendored legacy dependency from the project's Omnitrace history, kept
  for backward compatibility during migration — not a library to build new
  functionality against. Use standard C++ or a direct, purpose-built
  alternative instead.
- If a change appears to require touching timemory integration code, treat it
  as a migration task and flag it explicitly rather than extending the
  timemory surface.

## Memory & Ownership

- No raw `new`/`delete` in new code — `std::make_unique`/`std::make_shared`.
- Raw pointers mean "non-owning observer" only.
- No raw arrays — `std::array`, `std::vector`, or `std::span`.

## Safety

- `override` on every overriding virtual function.
- Never call virtual functions from constructors/destructors.
- `noexcept` on move constructors, destructors, swap, `operator[]`.
- No C-style casts — `static_cast`/`dynamic_cast`/etc.
- No `reinterpret_cast` without an explicit, documented justification
  (clang-tidy's `cppcoreguidelines-pro-type-reinterpret-cast` is disabled
  in this repo, so this is a human-reviewed rule, not tool-enforced).

## Build

- **Never build a single CMake target in isolation** — no `--target <name>`
  on `cmake --build`, no bare `ninja <target>`/`make <target>`. Always build
  the whole project; see `.claude/rules/cmake-conventions.md`.
