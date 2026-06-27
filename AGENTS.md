# Agent Notes

## Build and Test Parallelism

- Prefer the build system's default parallelism for builds, for example `cmake --build <build-dir> --target <target>`.
- Do not force single-threaded builds unless diagnosing a specific compiler error.
- `ctest` can be run in parallel, but cap it at `-j16` or lower to avoid OOMs.
