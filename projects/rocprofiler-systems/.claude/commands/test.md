---
description: Run rocprofiler-systems unit tests via the aggregated unit-test binary
argument-hint: [preset=debug] [--filter=<gtest_pattern>]
---

Preset: `${1:-debug}`.

Run the aggregated unit-test binary directly:
`./build/${1:-debug}/bin/rocprof-sys-unit-tests ${ARGUMENTS:+--gtest_filter=$ARGUMENTS}`

Never run `ctest`. Only the direct `rocprof-sys-unit-tests` executable is
used for testing (see `testing-conventions` rule).

Report pass/fail counts and, on failure, the failing test names and relevant
output.
