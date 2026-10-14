---
name: cmake-conventions
description: Rules for writing and editing rocprofiler-systems CMake files — linkage, target scope, formatting
---

# CMake Conventions — rocprofiler-systems

Use this when writing or editing `CMakeLists.txt` and `*.cmake` files.
Configure and build instructions are in `cmake-build.md`.

## Linkage

- Link only libraries the target uses directly. Do not copy another target's
  link list or add a dependency that is not needed to compile or link it.
- Prefer `PRIVATE`. Use `PUBLIC` or `INTERFACE` only when dependents use that
  library's headers or symbols through this target.
- Do not link an aggregate, such as `rocprofiler-systems-object-library`, when
  a narrower library provides the required symbols.

## Style

- Prefer target-based commands with explicit `PUBLIC`, `PRIVATE`, or
  `INTERFACE` scope.
- Do not add directory-scope include, compile-option, or link commands to
  first-party code.
- CMake files are formatted with `gersemi` (`.gersemirc` in this repo,
  enforced by pre-commit).
- List compiled sources explicitly. Existing globs for generated or runtime
  assets are intentional; do not introduce new globs without justification.
