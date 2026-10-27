---
description: Configure and build rocprofiler-systems with a CMake preset
argument-hint: [preset=debug]
---

Preset: `${1:-debug}`. List the available presets with `cmake --list-presets`.

1. Confirm cwd is `projects/rocprofiler-systems/` (has `CMakeLists.txt` +
   `CMakePresets.json`), not the monorepo root.
2. Configure: `cmake --preset ${1:-debug}`
3. Build the whole project — **never** pass `--target`:
   `cmake --build build/${1:-debug} -j$(nproc)`
4. Report: configure success/failure, build success/failure, warnings, build
   directory path.

See the `cmake-build` rule for presets and the whole-project-build rule.
