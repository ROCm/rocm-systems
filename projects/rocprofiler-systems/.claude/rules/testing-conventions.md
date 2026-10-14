---
name: testing-conventions
description: Testing conventions for rocprofiler-systems — unit, integration, mocks, and coverage
---

# Testing Conventions — rocprofiler-systems

## C++ unit tests

- Unit tests live beside their module in a `tests/` subdirectory.
- Each module builds an `OBJECT` library. Add it to `UNIT_TEST_OBJECTS` in
  `source/tests/CMakeLists.txt`; do not register a standalone unit-test
  executable.
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
- Mock headers may include production interfaces or shared types required to
  preserve the real contract. Avoid depending on concrete implementations
  when an interface or lightweight test type is sufficient.

## Coverage

Configure and build the `coverage` preset, run the aggregated unit-test binary,
then generate the report with `scripts/generate-coverage.py`. See
`.claude/commands/coverage.md` for the exact command sequence.
