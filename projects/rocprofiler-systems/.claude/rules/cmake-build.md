---
name: cmake-build
description: How to configure and build rocprofiler-systems — presets and whole-project builds
---

# CMake Build — rocprofiler-systems

Use this when configuring or building. Rules for editing CMake files are in
`cmake-conventions.md`.

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
