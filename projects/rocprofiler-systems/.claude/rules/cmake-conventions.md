---
name: cmake-conventions
description: CMake hard rules for rocprofiler-systems — presets, whole-project builds, formatting
---

# CMake Conventions — rocprofiler-systems

## Presets

Configure via `CMakePresets.json`, not hand-rolled flag lists, whenever a
preset fits:

| Preset | Build type | Testing | Use for |
| --- | --- | --- | --- |
| `ci` | Release | ON | CI builds |
| `debug` | Debug | ON | Day-to-day development |
| `debug-optimized` | RelWithDebInfo | ON | Debugging with realistic performance |
| `release` | Release | OFF | Production builds |
| `coverage` | Debug | ON | gcov-instrumented dev loop (see `rocprofsys` skill) |

Binary dir is always `build/<CMAKE_PRESET>` (from each preset's `binaryDir`). Don't
invent a different build directory name for a preset build.

```bash
cmake --preset <CMAKE_PRESET>
cmake --preset release -DROCPROFSYS_BUILD_DYNINST=ON   # override on top of a preset
```

## HARD RULE: whole-project builds only

Never pass `--target <name>` to `cmake --build`, and never invoke a bare
`ninja <target>` or `make <target>`. This project's test/coverage binaries
link many `OBJECT` libraries scattered across `source/`; a partial build
silently produces stale or incomplete artifacts. Always:

```bash
cmake --build build/<CMAKE_PRESET> -j<N>
```

## Style

- Modern, target-based CMake: `target_include_directories`,
  `target_compile_options`, `target_link_libraries` with explicit
  `PUBLIC`/`PRIVATE`/`INTERFACE`. No directory-scope
  `include_directories()`/`add_compile_options()`/`link_libraries()`.
- CMake files are formatted with `gersemi` (`.gersemirc` in this repo,
  enforced via pre-commit) — do not hand-format against a different style.
- No `file(GLOB ...)` for source collection — list files explicitly, matching
  the existing `CMakeLists.txt` pattern in each module.

## Tests

Never run `ctest`. Run the aggregated unit-test binary directly instead
(see `testing-conventions` rule):

```bash
./build/<CMAKE_PRESET>/bin/rocprof-sys-unit-tests --gtest_filter=<pattern>
```
