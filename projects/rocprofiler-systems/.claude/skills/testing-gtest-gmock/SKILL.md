---
name: testing-gtest-gmock
description: Use when writing or changing C++ unit tests in rocprofiler-systems (GTest/GMock, mocks, fixtures, test CMake), or after finishing a feature, refactor, or bugfix that needs tests. Plans the tests, then writes one test at a time with user approval, with strict mock expectations.
---

# GTest/GMock Testing

Repository rules for test layout, the aggregated binary, and mock headers are in
`.claude/rules/testing-conventions.md`. Follow them. This skill adds the workflow and
the strictness contract.

## Workflow

1. After implementation, ask: "Which components and edge cases should the tests
   cover?" Do not start without an answer.
2. Analyze the unit: its public API, every dependency that must be replaced, the exact
   observable outputs (return values, state changes, call sequences), and the edge
   cases (empty, one, max, error paths).
3. Write the plan to `planning/tests-<component>.md`: scope and non-goals, the
   dependencies and how each is replaced, one line per test (`name — exact behavior
   and values`), the mocks and fixtures needed.
4. Write ONE test, show it, and stop: "Here is the test for <behavior>. Continue with
   the next?" Mark it done in the plan only after approval. Never write several at once.
5. Build the whole project and run the aggregated binary with
   `--gtest_filter=<pattern>`. Paste the output before claiming a test passes.

## Strict contract

A test must fail if the implementation is wrong. If the answer is "maybe not", the test
is too weak.

1. `StrictMock<>` always. `NiceMock<>` only when ignoring other calls is deliberate;
   say why in a comment (`testing-conventions.md`).
2. `EXPECT_CALL` carries an exact `Times(N)`. No `AnyNumber()` or `AtLeast(1)`.
3. Exact argument matchers (`Eq(value)`, a `MATCHER_P`). The wildcard `_` is forbidden.
4. Exact values: `EXPECT_EQ(count, 3)`, not `EXPECT_GT(count, 0)`; `ElementsAre(...)`,
   not `size() > 0`.
5. No `ON_CALL`. It sets a default and never checks the call happened.
6. Every call the code makes to a mock has its own `EXPECT_CALL`.

```cpp
// Wrong: passes even if save() is never called
ON_CALL(*m_db, save(_)).WillByDefault(Return(true));

// Right
EXPECT_CALL(*m_db, save(Eq(record{42}))).Times(1).WillOnce(Return(true));
```

## Gotchas

- **Replace what can be injected.** Policy-injected dependencies (`backend<Wrapper>`,
  policy bundles) are replaced with a fake or mock policy type, not a virtual
  interface. Virtual interfaces are mocked with `MOCK_METHOD(..., (override))`; the
  interface name ends in `_interface` (`database_interface`, mock `mock_database`).
- **Static wrapper policies cannot use gmock.** If the policy is a set of static
  functions, drive a test fake through `inline static` state and assert on a call
  counter. Call `reset()` in `SetUp()`; state is process-global.
  `backends/hipfile/tests/mock_wrapper.hpp` is the model.
- **Mocks stay free of external headers.** Redefine the library types as plain
  structs with the library's field names, and wrap the library-mirroring names in
  `NOLINTBEGIN(readability-identifier-naming)` with a reason. The test must build
  where the library is not installed.
- **Register, do not add an executable.** Create an `OBJECT` library next to the test,
  link `rocprofiler-systems-googletest-library`, and append its
  `$<TARGET_OBJECTS:...>` to `UNIT_TEST_OBJECTS` in `source/tests/CMakeLists.txt`.
  Link only what the test uses; do not link a target that pulls in the external library.
- **Public API only.** Never widen a class, add an accessor, or add a hook for a test.
  If a behavior cannot be observed, the dependency needs a seam.
- **Isolation.** A fresh SUT and fresh mocks in every `SetUp()`. No `static` or global
  state shared between tests; GTest may reorder them.
- **`ASSERT_*` before dereferencing** a pointer, optional, or container element that a
  check guards. `EXPECT_*` for the remaining field checks, so all failures show.
- **`InSequence` only when order is the contract** (open, write, close). When only the
  count matters, use separate `EXPECT_CALL` statements.
- **Naming.** Test and fixture names are `lower_case` per `cpp-conventions.md`, in the form
  `<unit>_<scenario>_<exact_result>`; members use `m_`; each named constant uses `k_`.
  Library-required names (`SetUp`, `TearDown`) keep their spelling.
- **No bare literals.** A numeric literal other than 0 to 4 is a named `constexpr`
  with a `k_` prefix.

## Techniques

Use these when the plain `EXPECT_CALL(...).WillOnce(Return(x))` does not fit.

| Need | Use |
| --- | --- |
| Mock fills an out-parameter | `DoAll(SaveArg<N>(&captured), Return(...))`, then `ASSERT_` and check `captured` |
| Short read, partial write, or other custom behavior | `Invoke` with a lambda |
| Domain type without `operator==`, or field-by-field matching hides intent | `MATCHER_P` in the mock header |
| Phase 2 expectations must not satisfy phase 1 | `Mock::VerifyAndClearExpectations(mock.get())` between phases |
| Loop or `TEST_P` over cases | `SCOPED_TRACE(...)` or `::testing::PrintToString(GetParam())` at the top |
| Message of a thrown error | catch inside `EXPECT_THROW`, check `e.what()` with `HasSubstr`, rethrow |
| Contract violation that aborts | `EXPECT_DEATH(stmt, "regex")` in a `_death_test` suite |

## Self-check before showing a test

- [ ] Every mock is `StrictMock` (or `NiceMock` with a stated reason); no `ON_CALL`.
- [ ] Every `EXPECT_CALL` has exact `Times(N)` and exact matchers (no `_`).
- [ ] Every mock call the unit makes has an expectation.
- [ ] Numeric and container assertions are exact.
- [ ] `ASSERT_*` guards each dereference or index; loops have `SCOPED_TRACE`.
- [ ] No shared state; no external-library header in a mock.
- [ ] If the implementation returned a wrong value, this test would fail.

If any box is unchecked, rewrite the test before showing it.
