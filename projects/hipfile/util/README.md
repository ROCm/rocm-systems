# hipFile Development Utilities

## check-profiler-abi.sh

Compile-only guard that the hipFile dispatch-table ABI still satisfies the
`rocprofiler-sdk` profiler's static asserts. Compiles the profiler's
`abi.cpp` against hipFile's real headers so a forgotten ABI version bump fails
at hipFile PR time instead of breaking the profiler build post-merge. Requires
a ROCm/HIP toolchain; run inside a hipFile CI container. Used by CI.

## check-iwyu.sh

Checks the include-what-you-use runs that `iwyu-wrapper.sh` recorded. Fails
if IWYU didn't run or any run failed, and lists the files IWYU suggests
changing without failing on them. Used by CI.

## files-changed.sh

Determines if important files changed. Used by CI.

## format-source.sh

Formats applicable source in the repo using a specific version
of `clang-format`. Run it from the root.

## iwyu-wrapper.sh

Runs include-what-you-use for CMake and records each run's command, output,
and exit status for `check-iwyu.sh`, since CMake ignores IWYU's failures.
Used by CI.

## llvm-coverage.sh

Code coverage script used by CI.
