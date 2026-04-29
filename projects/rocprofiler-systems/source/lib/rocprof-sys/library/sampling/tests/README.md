# Sampling subsystem tests

## Structure

```
tests/
  unit/          — fast, in-process unit tests (no real signals, no tmp files)
  integration/   — full pipeline via test_sampling_policies (no real OS calls)
  bench/         — Google Benchmark throughput tests (separate binary)
  doubles/       — hand-written test doubles (no gmock)
```

## Test binaries

| Binary | How built | CTest label |
|---|---|---|
| `rocprof-sys-unit-tests` | merged into main unit binary | `sampling;unit` |
| `rocprof-sys-sampling-bench` | separate (Google Benchmark) | `sampling;bench` |

The unit + integration sources compile as the `rocprof-sys-sampling-tests-objs` OBJECT
library and are linked into the project-wide `rocprof-sys-unit-tests` binary, giving
one unified test runner for CI.

## Running unit + integration tests

```bash
# Quick run via ctest (uses debug build):
ctest --test-dir build/debug -R rocprof-sys-unit-tests --output-on-failure

# Direct binary run with filter:
build/debug/bin/rocprof-sys-unit-tests --gtest_filter="sampling_service*"
```

## No-skip policy

No test body may contain `GTEST_SKIP()`, `DISABLED_` prefix, or `getenv`-gated
assertions. Platform-specific tests are excluded at CMake time:

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _sampling_test_sources unit/linux_impl_smoke_test.cpp)
endif()
```

## Test doubles

All doubles live under `tests/doubles/` and are exposed via the
`rocprof-sys-sampling-test-doubles` INTERFACE library. Include with:

```cpp
#include "doubles/fake_clock.hpp"
#include "doubles/test_sampling_policies.hpp"
// etc.
```

Naming follows `.clang-tidy` rules: `lower_case` structs, `m_` prefix for all members.
