---
name: testing-conventions
description: Testing structure and conventions for rocprofiler-systems — unit tests and coverage
---

# Testing Conventions — rocprofiler-systems

## HARD RULE: never run `ctest` — direct unit-test executable only

Claude must never invoke `ctest` for this project, in any form (`ctest`,
`ctest --output-on-failure`, `ctest -R <name>`, etc.), including for
end-to-end/integration verification. Always run the aggregated GTest binary
directly instead:

```bash
./build/<CMAKE_PRESET>/bin/rocprof-sys-unit-tests --gtest_filter=<pattern>
```

## C++ unit tests

- Live next to the code, in a `tests/` subdirectory per module (e.g.
  `source/lib/common/tests/`, `source/lib/backends/amd_smi/tests/`,
  `source/bin/common/tests/`).
- Each module's tests build as an `OBJECT` library, aggregated in
  `source/tests/CMakeLists.txt` (`UNIT_TEST_OBJECTS`) into one self-contained
  GTest binary: `rocprof-sys-unit-tests`. New test files must be wired into
  their module's own `tests/CMakeLists.txt`, not registered standalone.
- **Never run `ctest`.** Always run the aggregated unit-test binary
  directly:

  ```bash
  ./build/<CMAKE_PRESET>/bin/rocprof-sys-unit-tests --gtest_filter=<pattern>
  ```

- When mocking with GMock, use `StrictMock<>` — never plain `Mock<>` — so
  unexpected calls fail loudly instead of being silently ignored.
- Mock types must be self-contained: never `#include` production headers in
  a mock header; write lightweight fakes instead.

## Coverage

Use the `coverage` CMake preset for day-to-day coverage visibility, run the
unit-test binary, then `scripts/generate-coverage.py` — full workflow and
common mistakes are in the `rocprofsys` skill.
