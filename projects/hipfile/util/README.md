# hipFile Development Utilities

## check-iwyu.sh

Checks the include-what-you-use runs that `iwyu-wrapper.sh` recorded. Fails
if IWYU didn't run or any run failed, and lists the files IWYU suggests
changing without failing on them. Used by CI.

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

## codeql-filter-sarif.py

Removes the CodeQL results that come from how GoogleTest works, such as the
unused static functions that `INSTANTIATE_TEST_SUITE_P` defines, from SARIF
files before CI uploads them. Each exclusion covers one query in one
directory, and the script says why each one is there. Used by CI.

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
