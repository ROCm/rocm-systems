---
description: Run rocprofiler-systems unit tests via the aggregated unit-test binary
argument-hint: [preset=debug] [--filter=<gtest_pattern>]
---

Preset: `${1:-debug}`.

Run the aggregated unit-test binary directly:
`./build/${1:-debug}/bin/rocprof-sys-unit-tests ${ARGUMENTS:+--gtest_filter=$ARGUMENTS}`

This command runs C++ unit tests. Integration and end-to-end tests use CTest;
see the `testing-conventions` rule.

Report pass/fail counts and, on failure, the failing test names and relevant
output.
