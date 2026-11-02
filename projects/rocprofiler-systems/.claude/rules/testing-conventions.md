---
name: testing-conventions
description: Testing conventions for rocprofiler-systems — unit, integration, mocks, and coverage
---

# Testing Conventions — rocprofiler-systems

## C++ unit tests

- Unit tests live beside their module in a `tests/` subdirectory.
- Each module builds an `OBJECT` library that links
  `rocprofiler-systems-googletest-library`. Add it to `UNIT_TEST_OBJECTS` in
  `source/tests/CMakeLists.txt`; do not register a standalone unit-test
  executable. Do not link a target that pulls in the external library
  (for example a backend target); the test must build without it installed.
- Run unit tests through the aggregated GTest binary:

```bash
./build/<preset>/bin/rocprof-sys-unit-tests --gtest_filter=<pattern>
```

## Integration and end-to-end tests

The suite under `tests/` uses pytest with CTest as its executor. Do not run
pytest directly. Run the generated CTest suite as documented in
`tests/README.md`:

```bash
ctest --test-dir build/<preset> --output-on-failure
```

Use the direct GTest binary for unit tests and CTest for integration tests; the
commands are not interchangeable.

## GMock

- Default to `StrictMock<>` so unexpected calls fail.
- Use `NiceMock<>` only when uninteresting calls are deliberate and explain
  that choice in the test.
- A unit test isolates one unit. Replace every collaborator that can be
  injected (backend wrapper, policy type, interface) with a mock or fake.
- A mock never includes an external library header (`<hipfile.h>`,
  `<amd_smi/amdsmi.h>`, `<rocprofiler-sdk/...>`). Redefine the library types
  it needs as plain test structs whose field names match the library's, as
  `backends/hipfile/tests/mock_wrapper.hpp` does. Unit tests must build on a
  machine where that library is not installed.
- A mock may include project-owned headers such as interfaces, `types.hpp`, or
  constants when the real contract needs them. It does not include the
  concrete implementation it replaces.

## Coverage

Configure and build the `coverage` preset, run the aggregated unit-test binary,
then generate the report with `scripts/generate-coverage.py`. See
`.claude/commands/coverage.md` for the exact command sequence.
