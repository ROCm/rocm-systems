# C++ Coding Style Guidelines

Coding conventions for C++ in the ROCm Compute Profiler. The
[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
are the reference for anything these files do not cover.

The rules are split by topic so that each file is short enough to read in one
sitting and specific enough to link to from a review comment. This page is the
only index.

The rules apply to new code and to existing code. No file is exempt. Do not
rewrite untouched code to comply, and do not open a PR that only inventories
violations. A review flags what the change introduces or moves.

## Rules

| Topic | Rules |
|---|---|
| Language version, interfaces, functions, classes, RAII, errors, `const`, `constexpr`, `noexcept`, attributes | [`core.md`](.ai/rules/cpp/core.md) |
| Allocation, copies, moves, cache behaviour, the hot path | [`performance.md`](.ai/rules/cpp/performance.md) |
| Dependency injection, mockable seams, gtest and gmock conventions | [`testability.md`](.ai/rules/cpp/testability.md) |
| Threading, exceptions at a host boundary, process-lifetime state | [`callback-boundaries.md`](.ai/rules/cpp/callback-boundaries.md) |
| Type names | [`naming.md`](.ai/rules/cpp/naming.md) |
| Doxygen blocks and when an inline comment earns its place | [`comments-and-docs.md`](.ai/rules/cpp/comments-and-docs.md) |
| Algorithms instead of raw loops, picking a container | [`stl-algorithms.md`](.ai/rules/cpp/stl-algorithms.md) |
| Choosing a pattern, and the two this project requires | [`design-patterns.md`](.ai/rules/cpp/design-patterns.md) |
| Target-based CMake | [`cmake.md`](.ai/rules/cpp/cmake.md) |

## The short version

- C++17 for our code. `test-torch-trace-collector` compiles as C++20 only so it can include libtorch headers.
- RAII for every resource. No raw `new` or `delete`, except the never-destroyed process-lifetime object in [`callback-boundaries.md`](.ai/rules/cpp/callback-boundaries.md).
- Types are PascalCase.
- Everything must be unit testable without a GPU. Dependencies come in through a
  virtual interface.
- We run inside somebody else's callback: no I/O and only brief locks on the
  dispatch path, and no exception ever escapes into the host.
- Performance is part of correctness here. A slow collector changes the numbers
  it exists to measure.
