---
name: testing-gtest-gmock
description: Use while writing or changing C++ GTest/GMock unit tests in rocprofiler-systems (mocks, fake wrapper policies, fixtures, test CMake), or when a test still passes after the code it covers is broken. Not for deciding which tests to write.
---

# GTest/GMock Testing

**Core principle:** a test must fail when the implementation is wrong. If deleting the
call or returning a wrong value would leave it green, the test is too weak.

Layout, the aggregated binary, and mock-header rules are in
`.claude/rules/testing-conventions.md`. Follow them; this skill does not repeat them.

## Strict contract

1. `StrictMock<>`. `NiceMock<>` only with a comment saying why other calls are ignored.
2. No `ON_CALL`: it sets a default and never checks the call happened.
3. Every `EXPECT_CALL` states `Times(N)` explicitly, even with `WillOnce`.
4. Exact matchers (`Eq`, `Field`, `NotNull`, a `MATCHER_P`). No wildcard `_`.
5. Exact values: `EXPECT_EQ(n, 3)`, not `EXPECT_GT(n, 0)`; `ElementsAre(...)`, not
   `size() > 0`.
6. `ASSERT_*` before dereferencing or indexing what it guards; `EXPECT_*` for the rest.
7. `InSequence` only when order is the contract (open, save, close).

```cpp
// Wrong: passes even if save() is never called
ON_CALL(*m_db, save(_)).WillByDefault(Return(true));

// Right
EXPECT_CALL(*m_db, save(Field(&record::id, Eq(k_id)))).Times(1).WillOnce(Return(true));
```

## Repo rules an agent cannot guess

- **Use the existing seam, never add one.** `backend<Wrapper>` and policy bundles take a
  fake policy type. Never add a virtual interface, constructor overload, accessor, or hook
  to production code for a test.
- **Static wrapper policies.** Either a state fake (`inline static` canned data plus a
  `call_count`, model: `backends/hipfile/tests/mock_wrapper.hpp`), or static functions
  forwarding to a `MockFunction<...>*` when exact arguments matter. Either way, reset all
  static state in `SetUp()` and null the pointer in `TearDown()`; state is process-global.
- **Virtual interfaces** end in `_interface`; mocks are `mock_<name>` with
  `MOCK_METHOD(..., (override))`.
- **Library-mirroring names** in mocks go inside `NOLINTBEGIN(readability-identifier-naming)`
  with a reason.
- **CMake:** an `OBJECT` library next to the test linking
  `rocprofiler-systems-googletest-library`, appended to `UNIT_TEST_OBJECTS` in
  `source/tests/CMakeLists.txt`. Link nothing that pulls in the external library.
- **Naming:** `lower_case` suites and tests, `<unit>_<scenario>_<exact_result>`; members
  `m_`; constants `k_`. Literals other than 0-4 are `constexpr k_` constants.
- **Proof:** build the whole project, run `rocprof-sys-unit-tests --gtest_filter=<pattern>`,
  paste the output before claiming a pass.

## Common rationalizations

| Excuse | Reality |
| --- | --- |
| "`WillOnce` already implies `Times(1)`" | Repo rule: state it. Adding a second `WillOnce` silently changes the count. |
| "I'll add a virtual interface so gmock works" | Use the template seam with a fake policy. |
| "The static pointer is set in each test" | A test that forgets dangles on the last test's mock. Reset in fixtures. |
| "`_` is fine, the argument is obvious" | Then match it exactly; it costs one word. |
| "In a hurry, `NiceMock` is quicker" | It hides the missing call you are testing for. |

## Techniques

| Need | Use |
| --- | --- |
| Mock fills an out-parameter | `DoAll(SaveArg<N>(&captured), Return(...))` |
| Short read, partial write, custom behavior | `Invoke` with a lambda |
| Type without `operator==` | `Field(...)` or a `MATCHER_P` in the mock header |
| Phase 2 must not satisfy phase 1 | `Mock::VerifyAndClearExpectations(mock.get())` |
| Loop or `TEST_P` over cases | `SCOPED_TRACE(...)` / `PrintToString(GetParam())` |
| Message of a thrown error | catch inside `EXPECT_THROW`, `HasSubstr` on `e.what()`, rethrow |
| Contract violation that aborts | `EXPECT_DEATH(stmt, "regex")` in a `_death_test` suite |

## Self-check before showing a test

- [ ] `StrictMock` (or reasoned `NiceMock`), no `ON_CALL`, no `_`, explicit `Times(N)`.
- [ ] Every mock call the unit makes has an expectation.
- [ ] Assertions are exact; `ASSERT_*` guards each dereference.
- [ ] Static fake state reset per test; no external-library header in a mock.
- [ ] No production code changed to make the test possible.
- [ ] Removing the call under test, or returning a wrong value, makes this test fail.
