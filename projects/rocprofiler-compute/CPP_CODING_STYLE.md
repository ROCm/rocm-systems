# C++ Coding Style Guidelines

Coding conventions for C++ in the ROCm Compute Profiler. The
[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
are the reference for anything these files do not cover.

The rules are split by topic so that each file is short enough to read in one
sitting and specific enough to link to from a review comment. This page is the
only index.

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
| Target-based CMake | [`cmake.md`](.ai/rules/cpp/cmake.md) |

## Reference

Catalogues to consult when a problem calls for them. Not enforced in review.

- [`reference/design-patterns.md`](.ai/rules/cpp/reference/design-patterns.md)
- [`reference/constexpr.md`](.ai/rules/cpp/reference/constexpr.md)

## The short version

- C++17 everywhere, except `torch_trace_collector`, which is C++20 because
  libtorch requires it.
- RAII for every resource. No raw `new` or `delete`.
- Types are PascalCase.
- Everything must be unit testable without a GPU. Dependencies come in through a
  virtual interface.
- We run inside somebody else's callback: no allocation or blocking on the
  sampling path, and no exception ever escapes into the host.
- Performance is part of correctness here. A slow collector changes the numbers
  it exists to measure.
