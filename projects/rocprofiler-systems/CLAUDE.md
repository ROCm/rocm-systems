# rocprofiler-systems — Claude Code Project Guide

ROCm Systems Profiler (formerly Omnitrace): profiling/tracing tool for
parallel applications (C, C++, Fortran, HIP, OpenCL, Python) on CPU and GPU.
C++20, CMake, part of the `rocm-systems` monorepo
(`projects/rocprofiler-systems/`).

## Rules (always active)

Four rule files in `.claude/rules/` are always active — internalize them:

- `cpp-conventions.md` — C++20, clang-format/clang-tidy config actually used
  in this repo, the "never use timemory in new code" migration rule, memory/
  safety rules.
- `cmake-conventions.md` — presets, the whole-project-build hard rule,
  gersemi formatting.
- `git-workflow.md` — this project's actual commit format (Conventional
  Commits + `(rocprofiler-systems)` scope, verified against merged history),
  branch naming, pre-commit requirements.
- `testing-conventions.md` — unit test layout, the aggregated
  `rocprof-sys-unit-tests` binary, GMock `StrictMock` rule, coverage workflow.

## Planning

Before any non-trivial task (more than 2 steps), invoke the matching planning
skill: `planning-feature`, `planning-bugfix`, `planning-refactor`,
`planning-docs`, or `planning-architecture`. See `planning-base` for the
shared phases all of them build on.

## Skill loading

If there's even a small chance a skill applies, load it — the cost of an
unused skill is near zero; the cost of missing one is inconsistent output.
See `.claude/skills/` for the full list; notable ones:

| Task | Skill |
| --- | --- |
| Any C++ work | `programming-cpp`, `programming-cpp-naming-rules`, `programming-cpp-stl-algorithms`, `programming-cpp-design-patterns`, `programming-cpp-policy-based-di`, `programming-cpp-constexpr` |
| Python work | `programming-python` |
| C++ tests | `testing-gtest-gmock` |
| Configure/build/run/coverage rocprofsys | `rocprofsys` |
| Code review / smells | `code-quality`, `code-smells` |
| Refactoring | `refactoring-techniques` |

## Commands

- `/build` — configure + build with a chosen preset
- `/coverage` — coverage-preset build, run unit tests, generate report
- `/test` — run ctest or the unit-test binary

## Core rules (non-negotiable)

- English output only, regardless of input language.
- C++20 only — no C++23+ features (see `cpp-conventions.md`).
- Never use timemory APIs in new code (see `cpp-conventions.md`).
- Whole-project builds only — never `--target` (see `cmake-conventions.md`).
- Plan before implementing anything non-trivial.

## Human contributor process

For the non-AI contribution process (issue etiquette, PR guidelines, license,
pre-commit setup) see `CONTRIBUTING.md`.
