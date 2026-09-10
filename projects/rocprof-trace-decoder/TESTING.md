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

## Regenerating hidden-latency controls

`test/control/<dataset>/hidden_latency/compare_hidden_latency.csv` holds one row per program
counter with nonzero hidden latency, listing the instruction and its decoded latency next to
the hidden idle, stall, and issue cycles. `HiddenIdle` measures the gap before the instruction
was attempted, not part of its latency, so it routinely exceeds `Latency`. Regenerate one with
`--write`, pointing `--expected` at the file in the source tree rather than the build copy:

```bash
cd build/test
python3 ../../test/hidden_latency_csv_test.py \
    --lib ../lib/librocprof-trace-decoder.so --write \
    --expected ../../test/control/navi4_fifo/hidden_latency/compare_hidden_latency.csv \
    navi4_fifo/*.att --stats "control/navi4_fifo/*.csv"
```

Controls exist for one trace per architecture. The gfx950 traces have none because they hide
no cycles at all, which would leave a header-only file; `mi350_histo0_hidden_latency_empty`
and `mi350_scale_hidden_latency_empty` assert that instead.

`navi4_fifo_hidden_latency_selftest` is the negative control. The per-trace tests only prove
the comparison accepts a correct file; this one corrupts the control table five ways in memory
and requires every corruption to be caught, so a comparison that ignored a column or only
checked that the control is a subset of the output cannot keep passing.

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
