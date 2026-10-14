# rocprofiler-systems Project Guide

ROCm Systems Profiler (formerly Omnitrace): profiling/tracing tool for
parallel applications (C, C++, Fortran, HIP, OpenCL, Python) on CPU and GPU.
C++20, CMake, part of the `rocm-systems` monorepo
(`projects/rocprofiler-systems/`).

## Project rules

Read the applicable files in `.claude/rules/` before making changes:

- `cpp-conventions.md` for C++ files.
- `cmake-conventions.md` for CMake and build work.
- `testing-conventions.md` for tests and coverage.

## Planning

Plan non-trivial work before implementation. Keep the plan proportional to the
scope and include verification.

## Skill loading

Load only skills relevant to the current task. Skills are in
`.claude/skills/` (`.cursor` points to the same directory).

| Task | Skill |
| --- | --- |
| General C++ implementation | `programming-cpp` |
| Create, move, or rename C++ files/types | `programming-cpp-naming-rules` |
| Containers, algorithms, ranges, or loops | `programming-cpp-stl-algorithms` |
| Object design or behavioral variation | `programming-cpp-design-patterns` |
| Compile-time dependency injection | `programming-cpp-policy-based-di` |
| Compile-time computation | `programming-cpp-constexpr` |
| Python work | `programming-python` |
| C++ tests | `testing-gtest-gmock` |
| Code review / smells | `code-quality`, `code-smells` |
| Refactoring | `refactoring-techniques` |

## Commands

- `/build` — configure + build with a chosen preset
- `/clang-tidy` — run `scripts/clang-tidy-check.py` against `.clang-tidy`
- `/coverage` — coverage-preset build, run unit tests, generate report
- `/test` — run the aggregated unit-test binary

## Core rules (non-negotiable)

- English output only, regardless of input language.
- C++20 only — no C++23+ features (see `cpp-conventions.md`).
- Never use timemory APIs in new code (see `cpp-conventions.md`).
- Whole-project builds only — never `--target` (see `cmake-conventions.md`).

## Human contributor process

For the non-AI contribution process (issue etiquette, PR guidelines, license,
pre-commit setup) see `CONTRIBUTING.md`.
