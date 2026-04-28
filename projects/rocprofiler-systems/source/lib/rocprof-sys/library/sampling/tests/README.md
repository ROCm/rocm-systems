# Sampling subsystem tests

## Structure

```
tests/
  unit/          — fast, in-process unit tests (no real signals, no tmp files)
  integration/   — full pipeline via test_sampling_policies (no real OS calls)
  stress/        — real signals, real threads, TSAN-enabled binary (Linux only)
  bench/         — Google Benchmark throughput tests (separate binary)
  doubles/       — hand-written test doubles (no gmock)
```

## Test binaries

| Binary | How built | CTest label |
|---|---|---|
| `rocprof-sys-unit-tests` | merged into main unit binary | `sampling;unit` |
| `rocprof-sys-sampling-stress-tests` | separate (`-fsanitize=thread`) | `sampling;stress;tsan` |
| `rocprof-sys-sampling-bench` | separate (Google Benchmark) | `sampling;bench` |

The unit + integration sources compile as the `rocprof-sys-sampling-tests-objs` OBJECT
library and are linked into the project-wide `rocprof-sys-unit-tests` binary. This keeps
one unified test runner for CI while allowing the stress test to carry incompatible TSAN
link flags in its own binary.

## Running unit + integration tests

```bash
# Quick run via ctest (uses debug build):
ctest --test-dir build/debug -R rocprof-sys-unit-tests --output-on-failure

# Direct binary run with filter:
build/debug/bin/rocprof-sys-unit-tests --gtest_filter="sampling_service*"
```

## Running the TSAN stress test (NFR-TS-3 hard gate)

Use the canonical wrapper — **do not run the stress binary directly** for gate purposes:

```bash
# Build the debug preset first:
cmake --preset debug
cmake --build build/debug --target rocprof-sys-sampling-stress-tests

# Run via wrapper (collects all races, parses TSAN output, exits non-zero on any warning):
scripts/run-sampling-tsan.sh build/debug/bin/rocprof-sys-sampling-stress-tests
```

The wrapper (`scripts/run-sampling-tsan.sh`) applies:
- `setarch $(uname -m) -R` — disables ASLR for deterministic stack addresses
- `TSAN_OPTIONS="halt_on_error=0:second_deadlock_stack=1:exitcode=0"`
  - `halt_on_error=0` — collect ALL races/deadlocks in one run
  - `second_deadlock_stack=1` — capture both lock stacks for lock-order bugs
  - `exitcode=0` — GTest exit code propagates; TSAN findings go to stderr/log
- Hard gate: any `WARNING: ThreadSanitizer:` line in output → exit 1

## Running with a custom binary or extra gtest args

```bash
# Run only the stress suite with a filter:
scripts/run-sampling-tsan.sh \
    build/debug/bin/rocprof-sys-sampling-stress-tests \
    --gtest_filter="SignalHandlerStress*"

# Override binary via env:
TSAN_BINARY=build/debug/bin/rocprof-sys-unit-tests \
    scripts/run-sampling-tsan.sh --gtest_filter="real_timer_trigger_smoke*"

# Override log path:
TSAN_LOG=/tmp/my-tsan.log scripts/run-sampling-tsan.sh
```

## No-skip policy

No test body may contain `GTEST_SKIP()`, `DISABLED_` prefix, or `getenv`-gated
assertions. Platform-specific tests are excluded at CMake time:

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _sampling_test_sources unit/linux_impl_smoke_test.cpp)
endif()
```

The stress binary is only built when `ROCPROFSYS_BUILD_SAMPLING_TSAN=ON` AND
`CMAKE_SYSTEM_NAME STREQUAL "Linux"`.

## Test doubles

All doubles live under `tests/doubles/` and are exposed via the
`rocprof-sys-sampling-test-doubles` INTERFACE library. Include with:

```cpp
#include "doubles/fake_clock.hpp"
#include "doubles/test_sampling_policies.hpp"
// etc.
```

Naming follows `.clang-tidy` rules: `lower_case` structs, `m_` prefix for all members.
