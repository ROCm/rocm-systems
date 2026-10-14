---
name: cmake-conventions
description: CMake hard rules for rocprofiler-systems — presets, whole-project builds, formatting
---

# CMake Conventions — rocprofiler-systems

## Presets

Use `CMakePresets.json` whenever a preset fits:

| Preset | Build type | Testing | Use for |
| --- | --- | --- | --- |
| `ci` | Release | ON | CI builds |
| `debug` | Debug | ON | Day-to-day development |
| `debug-optimized` | RelWithDebInfo | ON | Debugging with realistic performance |
| `release` | Release | OFF | Production builds |
| `debug-mpi` | Debug | ON | Debug build with MPI |
| `release-mpi` | Release | OFF | Release build with MPI |
| `coverage` | Debug | ON | GCC coverage |

Preset binary directories are `build/<preset>`.

```bash
cmake --preset <preset>
cmake --preset release -DROCPROFSYS_BUILD_DYNINST=ON   # override on top of a preset
```

## Whole-project builds only

Never pass `--target` to `cmake --build` or invoke a target through Ninja or
Make directly. Unit tests aggregate object libraries from across `source/`, so
always build the complete configured tree:

```bash
cmake --build build/<preset> --parallel <jobs>
```

## Style

- Prefer target-based commands with explicit `PUBLIC`, `PRIVATE`, or
  `INTERFACE` scope.
- Do not add directory-scope include, compile-option, or link commands to
  first-party code.
- CMake files are formatted with `gersemi` (`.gersemirc` in this repo,
  enforced by pre-commit).
- List compiled sources explicitly. Existing globs for generated or runtime
  assets are intentional; do not introduce new globs without justification.
