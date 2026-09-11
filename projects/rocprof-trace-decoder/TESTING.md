<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocprof-trace-decoder Testing

## Current status

Tests are enabled with `-DBUILD_TESTS=ON` and run through CTest. The suite
includes GoogleTest unit tests, Python and fixture-based integration tests,
and AddressSanitizer and UndefinedBehaviorSanitizer variants. The project CI
builds and runs the full suite.

## Building and running tests

```bash
cmake -B build -DBUILD_TESTS=ON -DDISABLE_COMGR=ON
cmake --build build -j$(nproc)
ctest --test-dir build/test -j$(nproc)
```

Set `-DDISABLE_COMGR=ON` if `amd_comgr` is not installed. This skips the
att-tool but allows the other tests to run.

Run an individual test group with:

```bash
# Unit tests
ctest --test-dir build/test -R "regular/" -j$(nproc)

# Integration tests
ctest --test-dir build/test -E "regular/|sanitize|ubsan|asan" -j$(nproc)

# Sanitizer tests
ctest --test-dir build/test -R "asan/" -j$(nproc)
ctest --test-dir build/test -R "ubsan/" -j$(nproc)
```

## Code coverage

Code coverage is generated locally with the CMake `coverage` target using
gcov, lcov, and genhtml. Sanitizer tests are excluded from coverage runs. We
aim for 90% line coverage across the project.

```bash
# Requires gcov, lcov, and genhtml
cmake -B build_coverage -DBUILD_TESTS=ON -DDISABLE_COMGR=ON \
    -DCMAKE_CXX_FLAGS="--coverage -fprofile-arcs -ftest-coverage" \
    -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
    -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
    -DCMAKE_BUILD_TYPE=Debug

cmake --build build_coverage -j$(nproc)
cmake --build build_coverage --target coverage
```

## Future work

- Run coverage regularly in CI and publish the report.
- Track and enforce the 90% project-wide coverage target.
- Add focused unit and integration tests as formats, architectures, and public
  APIs are added or changed.
