# hipFile Development Utilities

## check-profiler-abi.sh

Compile-only guard that the hipFile dispatch-table ABI still satisfies the
`rocprofiler-sdk` profiler's static asserts. Compiles the profiler's
`abi.cpp` against hipFile's real headers so a forgotten ABI version bump fails
at hipFile PR time instead of breaking the profiler build post-merge. Requires
a ROCm/HIP toolchain; run inside a hipFile CI container. Used by CI.

## ci-build-installed-examples.sh

Checks the installed examples: that every example directory was installed
with a `CMakeLists.txt`, and that each one configures and builds standalone
against the installed hipFile with warnings as errors. Runs the `get-version`
example, which needs no GPU, and checks that its header and library versions
match. Used by CI after it installs the hipFile packages.

## files-changed.sh

Determines if important files changed. Used by CI.

## format-source.sh

Formats applicable source in the repo using a specific version
of `clang-format`. Run it from the root.

## llvm-coverage.sh

Code coverage script used by CI.
