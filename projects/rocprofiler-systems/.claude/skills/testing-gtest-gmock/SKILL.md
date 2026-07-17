---
name: testing-gtest-gmock
description: Use when writing C++ unit tests with GTest and GMock — creates a test plan and implements tests one at a time with user approval, enforcing strict mock expectations
---

# GTest/GMock Testing Skill

Use this skill when writing unit tests for C++ code using Google Test (GTest) and Google Mock (GMock).

<IMPORTANT>
**When to trigger this skill:**
After completing a feature, refactoring, or bugfix, ASK the user:

> "Implementation is complete. Would you like me to add unit tests?
>
> - Which components/functions should be tested?
> - Are there specific edge cases to cover?"

If user agrees, follow the planning process below.

**Test-by-test approval:**
Write ONE test at a time, then STOP and wait for user approval before writing the next test.
</IMPORTANT>

---

## The Strictness Mandate

Tests exist to **prove correctness**, not to satisfy a coverage metric. A test that always passes regardless of bugs is worse than no test — it creates false confidence.

**Every test must be a strict contract.** Ask: "If the implementation is wrong, will this test fail?" If the answer is "maybe not", the test is not strict enough.

### The Four Rules of Strict Testing

1. **StrictMock always** — catches every unexpected call as a test failure
2. **Exact call counts always** — `Times(N)` with a precise N, never open-ended
3. **Exact arguments always** — full matchers on every parameter, `_` is forbidden
4. **Exact values always** — `EXPECT_EQ(count, 3)` not `EXPECT_GT(count, 0)`

### Forbidden Patterns

| Pattern | Why it is wrong | Correct alternative |
| --------- | ----------------- | --------------------- |
| `NiceMock<mock_foo>` | Silently swallows unexpected calls; bugs hide | `StrictMock<mock_foo>` |
| `ON_CALL(m, f(...))` | Sets a default but never enforces the call happened | `EXPECT_CALL` with `Times(N)` |
| `AnyNumber()` / `AtLeast(1)` | Passes whether called 0 or 1000 times | `Times(3)` or `Times(1)` |
| `EXPECT_CALL(m, f(_))` | Accepts wrong arguments silently | `EXPECT_CALL(m, f(Eq(expected_val)))` |
| `EXPECT_GT(count, 0)` | Passes for any non-zero value | `EXPECT_EQ(count, 3)` |
| `EXPECT_TRUE(result.size() > 0)` | Does not verify what the result contains | `EXPECT_THAT(result, ElementsAre(...))` |
| Missing `EXPECT_CALL` for a mocked dep | Called without contract — silent success | Add `EXPECT_CALL` for every call the SUT makes |
| Direct external dependency | Can't control or observe behaviour | Inject via interface + `StrictMock` |

### ON_CALL Is Banned

`ON_CALL` sets a default return value but **never checks** whether the method was called. It is `NiceMock` behaviour hidden behind what looks like an expectation. Do not use it.

```cpp
// ❌ WRONG — ON_CALL does not enforce the call happened
ON_CALL(*m_mock_db, save(_)).WillByDefault(Return(true));
m_sut->process(record{42});
// Test passes even if save() was never called

// ✅ RIGHT — EXPECT_CALL enforces call count and arguments
EXPECT_CALL(*m_mock_db, save(Eq(record{42})))
    .Times(1)
    .WillOnce(Return(true));
m_sut->process(record{42});
```

### External Dependencies Must Be Mocked

If the code under test touches a file system, clock, network, GPU runtime, hardware counter, or any third-party API, that dependency **must** be hidden behind an interface and injected as a `StrictMock`. Direct calls to external APIs in tests make them non-deterministic and non-repeatable.

```cpp
// ❌ WRONG — test depends on real hardware counter
TEST(kernel_test, executes_kernel)
{
    kernel k;
    k.launch();
    EXPECT_GT(k.dispatch_count(), 0);  // flaky: hardware may not respond
}

// ✅ RIGHT — inject a mock, assert exactly
TEST_F(kernel_test, launch_dispatches_exactly_once_with_correct_grid)
{
    EXPECT_CALL(*m_mock_runtime, dispatch(Eq(grid_dim{256, 1, 1}), Eq(block_dim{64, 1, 1})))
        .Times(1)
        .WillOnce(Return(dispatch_result::ok));

    EXPECT_EQ(m_sut->launch(), launch_status::success);
}
```

### Test Isolation Rule

Each test must be **fully self-contained**. No `static` variables, no globals, no state shared between tests.

- Construct a fresh SUT and fresh mocks in every `SetUp()` or at the top of every `TEST`
- Never rely on execution order between tests — GTest may reorder them
- If two tests share setup code, use a fixture; never share instance state

```cpp
// ❌ WRONG — static state leaks between tests
static service* g_svc = nullptr;

TEST(service_test, first_test) { g_svc = new service(); ... }
TEST(service_test, second_test) { g_svc->do_thing(); ... }  // depends on first_test

// ✅ RIGHT — fixture creates fresh objects for every test
class service_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_mock_db = std::make_shared<StrictMock<mock_database>>();
        m_sut     = std::make_unique<service>(m_mock_db);
    }
    std::shared_ptr<StrictMock<mock_database>> m_mock_db;
    std::unique_ptr<service>                   m_sut;
};
```

---

## Phase 1: Analyze Testable Code

Before writing tests, identify:

1. **Public API** — Functions/methods that form the contract being tested
2. **External dependencies** — Everything that must be mocked (filesystem, clock, hardware, third-party libs)
3. **Exact observable outputs** — Return values, state changes, exact call sequences
4. **Edge cases** — Boundary conditions, error states, empty inputs, max values

For each dependency ask: "Can I inject this as a mock?" If not, redesign the interface first.

---

## Phase 2: Create Test Plan

Save to `planning/tests-<component>.md`:

```markdown
# Test Plan: <Component Name>

## Scope
<What is being tested and what is explicitly out of scope>

## External Dependencies (all must be mocked)
- `<dep_name>`: interface `<i_dep>`, injected via constructor/policy
- ...

## Test Cases

### <Function/Class 1>
- [ ] Test: <test name> — verifies <exact behaviour with exact values>
- [ ] Test: <test name> — verifies <exact behaviour with exact values>

#### Edge Cases
- [ ] <edge case> — expected exact result: <value>
- [ ] <edge case> — expected exact result: <value>

### <Function/Class 2>
...

## Mocks Required
- [ ] `mock_<name>` — mocks `i_<name>`, used in <which tests>

## Test Fixtures
- [ ] `<fixture_name>` — shared setup: constructs SUT with StrictMock deps
```

---

## Phase 3: Implement Tests (One by One)

<IMPORTANT>
**CRITICAL: Test-by-test workflow**

1. Write ONE test
2. Show it to the user
3. Say: "Here's the test for <description>. Is it OK? Should I continue with the next test?"
4. WAIT for user approval
5. Only after approval, proceed to next test
6. Mark completed test in plan file

DO NOT write multiple tests at once!
</IMPORTANT>

**Before writing each test, answer:**

- What exact return value / state change / call sequence does this test verify?
- What exact arguments will the SUT pass to each mock?
- Exactly how many times will each mocked method be called?
- If the implementation is wrong, will this test fail? (If "maybe not", redesign the test.)

---

## ASSERT_*vs EXPECT_* Discipline

`EXPECT_*` continues after failure — useful to collect all failures in one run.
`ASSERT_*` stops the test immediately — required when a failed check makes further checks meaningless or dangerous.

**Rule:** use `ASSERT_*` whenever a subsequent line would dereference a pointer, access a container element, or otherwise invoke undefined behaviour if the check failed.

```cpp
TEST_F(service_test, find_returns_record_with_correct_fields)
{
    EXPECT_CALL(*m_mock_db, find(Eq(42)))
        .Times(1)
        .WillOnce(Return(std::make_optional(record{.id = 42, .name = "alpha"})));

    auto result = m_sut->find(42);

    // ASSERT first — if nullopt, the next lines would crash
    ASSERT_TRUE(result.has_value());

    // EXPECT for remaining field checks — collect all failures
    EXPECT_EQ(result->id,   42);
    EXPECT_EQ(result->name, "alpha");
}
```

When in doubt: `ASSERT_*` before pointer dereference or container index; `EXPECT_*` everywhere else.

---

## GTest/GMock Patterns

### Basic Test Structure

```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(calculator_test, add_returns_sum_of_two_positive_numbers)
{
    calculator calc;
    EXPECT_EQ(calc.add(2, 3), 5);
}
```

### Test Fixture with StrictMock

```cpp
class service_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Always StrictMock — unexpected calls fail immediately
        m_mock_db    = std::make_shared<StrictMock<mock_database>>();
        m_mock_clock = std::make_shared<StrictMock<mock_clock>>();
        m_sut        = std::make_unique<service>(m_mock_db, m_mock_clock);
    }

    std::shared_ptr<StrictMock<mock_database>> m_mock_db;
    std::shared_ptr<StrictMock<mock_clock>>    m_mock_clock;
    std::unique_ptr<service>                   m_sut;
};
```

### Creating Mocks

All external dependencies go behind an interface:

```cpp
class i_database
{
public:
    virtual ~i_database() = default;
    virtual bool                    save(const record& r) = 0;
    virtual std::optional<record>   find(int id)          = 0;
};

class mock_database : public i_database
{
public:
    MOCK_METHOD(bool,                  save, (const record& r), (override));
    MOCK_METHOD(std::optional<record>, find, (int id),          (override));
};
```

### Strict Mock Usage — Exact Calls, Exact Args

```cpp
using ::testing::Eq;
using ::testing::Return;
using ::testing::StrictMock;

TEST_F(service_test, process_saves_record_with_correct_fields_exactly_once)
{
    const record expected_record{.id = 42, .name = "alpha", .value = 7};

    EXPECT_CALL(*m_mock_db, save(Eq(expected_record)))
        .Times(1)
        .WillOnce(Return(true));

    // StrictMock fails the test if any other method on m_mock_db is called.
    EXPECT_TRUE(m_sut->process(expected_record));
}
```

### InSequence — Order vs Count-Only

Use `InSequence` **only** when the order of calls is part of the contract. When only the count matters (not the order), use separate `EXPECT_CALL` statements without `InSequence`.

```cpp
// Use InSequence when ORDER is the contract (e.g., open before write before close)
TEST_F(pipeline_test, stages_execute_in_order_with_correct_data)
{
    ::testing::InSequence seq;

    EXPECT_CALL(*m_mock_stage_a, run(Eq(input_data{.value = 1})))
        .Times(1)
        .WillOnce(Return(stage_result{.value = 2}));

    EXPECT_CALL(*m_mock_stage_b, run(Eq(stage_result{.value = 2})))
        .Times(1)
        .WillOnce(Return(stage_result{.value = 3}));

    EXPECT_EQ(m_sut->execute(input_data{.value = 1}), final_result{.value = 3});
}

// No InSequence when only COUNT matters (order is an implementation detail)
TEST_F(batch_test, process_calls_save_exactly_three_times_for_batch_of_three)
{
    EXPECT_CALL(*m_mock_db, save(Eq(record{1}))).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*m_mock_db, save(Eq(record{2}))).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*m_mock_db, save(Eq(record{3}))).Times(1).WillOnce(Return(true));

    EXPECT_EQ(m_sut->process_batch({record{1}, record{2}, record{3}}), 3);
}
```

### Exact Container Contents

```cpp
// ❌ WRONG — "has something" is not a contract
EXPECT_GT(results.size(), 0);
EXPECT_FALSE(results.empty());

// ✅ RIGHT — exactly what the results must contain
EXPECT_THAT(results, ElementsAre(
    record{.id = 1, .value = 10},
    record{.id = 2, .value = 20}
));
```

### Capturing Out-Parameters with SaveArg

When the SUT passes a pointer or reference that the mock fills in, capture it and verify the written value after the call.

```cpp
using ::testing::SaveArg;
using ::testing::DoAll;

TEST_F(encoder_test, encode_writes_exactly_four_bytes_to_output_buffer)
{
    std::vector<uint8_t> captured_buf;

    EXPECT_CALL(*m_mock_sink, write(NotNull(), Eq(4u)))
        .Times(1)
        .WillOnce(DoAll(
            SaveArg<0>(&captured_buf),   // capture the pointer argument
            Return(write_result::ok)
        ));

    EXPECT_EQ(m_sut->encode(input_frame{0xDEAD}), encode_status::ok);

    ASSERT_EQ(captured_buf.size(), 4u);
    EXPECT_EQ(captured_buf[0], 0xDE);
    EXPECT_EQ(captured_buf[1], 0xAD);
}
```

### DoAll and Invoke for Complex Actions

`DoAll` chains multiple actions on one call. `Invoke` delegates to a lambda or function for behaviour that cannot be expressed with `Return`.

```cpp
using ::testing::Invoke;

// DoAll: set a side-effect AND return a value
EXPECT_CALL(*m_mock_db, find(Eq(99)))
    .Times(1)
    .WillOnce(DoAll(
        [](int) { /* log or signal */ },
        Return(std::make_optional(record{.id = 99}))
    ));

// Invoke: complex logic (e.g., simulate partial writes)
EXPECT_CALL(*m_mock_stream, read(NotNull(), Eq(8u)))
    .Times(1)
    .WillOnce(Invoke([](uint8_t* buf, std::size_t) -> std::size_t {
        buf[0] = 0xAB;
        buf[1] = 0xCD;
        return 2u;  // simulate a short read
    }));
```

### Mock::VerifyAndClearExpectations for Multi-Phase Tests

When testing a multi-phase operation in one fixture, use `VerifyAndClearExpectations` between phases to assert phase N is fully done before setting up phase N+1 expectations. This prevents expectations from one phase accidentally satisfying another.

```cpp
TEST_F(transaction_test, commit_follows_write_then_flush_in_two_phases)
{
    // Phase 1: write
    EXPECT_CALL(*m_mock_store, write(Eq(payload{42})))
        .Times(1)
        .WillOnce(Return(write_status::ok));

    m_sut->write(payload{42});

    // Assert phase 1 expectations are satisfied before moving on
    ::testing::Mock::VerifyAndClearExpectations(m_mock_store.get());

    // Phase 2: flush (separate expectation set)
    EXPECT_CALL(*m_mock_store, flush())
        .Times(1)
        .WillOnce(Return(flush_status::ok));

    EXPECT_EQ(m_sut->commit(), commit_status::ok);
}
```

### Custom Matchers for Domain Types

When a domain type does not support `operator==`, or field-by-field matching would obscure intent, write a `MATCHER_P` instead of reaching for `_`.

```cpp
// Define once in a mocks/ header
MATCHER_P(HasKernelId, expected_id, "kernel with id " + std::to_string(expected_id))
{
    return arg.id == expected_id && arg.is_valid();
}

// Use in tests — readable, strict, no wildcard
TEST_F(dispatcher_test, dispatch_sends_kernel_with_correct_id)
{
    EXPECT_CALL(*m_mock_runtime, submit(HasKernelId(7)))
        .Times(1)
        .WillOnce(Return(submit_result::ok));

    EXPECT_EQ(m_sut->dispatch(kernel_request{.id = 7}), dispatch_status::ok);
}
```

### SCOPED_TRACE for Parameterized and Loop Tests

When a test iterates or uses parameterized values, failure output shows only the line number — not which case failed. `SCOPED_TRACE` injects context into every failure message inside its scope.

```cpp
TEST_F(parser_test, parse_known_inputs_return_exact_token_counts)
{
    const std::vector<std::pair<std::string, int>> cases = {
        {"a + b",   3},
        {"a + b * c", 5},
        {"(a)",     3},
    };

    for (const auto& [input, expected_count] : cases)
    {
        SCOPED_TRACE("input: " + input);  // printed on failure: which case broke

        auto tokens = m_sut->tokenize(input);
        ASSERT_EQ(static_cast<int>(tokens.size()), expected_count);
    }
}
```

For `TEST_P`, add `SCOPED_TRACE(::testing::PrintToString(GetParam()))` at the top of the test body.

### Testing Exceptions

```cpp
TEST(parser_test, parse_invalid_throws_parse_error_with_line_number)
{
    parser p;
    EXPECT_THROW(
        {
            try {
                p.parse("bad input");
            } catch (const parse_error& e) {
                EXPECT_THAT(e.what(), HasSubstr("line 1"));
                EXPECT_EQ(e.line(), 1);
                throw;
            }
        },
        parse_error
    );
}
```

### Death Tests for Contract Violations

Use `EXPECT_DEATH` / `ASSERT_DEATH` when the code under test should `assert()`, `abort()`, or otherwise terminate on a contract violation. Place death tests in a separate `_death_test` suite — GTest runs them in a child process.

```cpp
TEST(buffer_death_test, access_out_of_bounds_aborts)
{
    ring_buffer<int, 4> buf;
    // operator[] on an empty buffer must abort, not return garbage
    EXPECT_DEATH(buf[0], "");  // second arg is a regex matched against stderr
}

TEST(contract_death_test, construct_with_zero_capacity_aborts)
{
    EXPECT_DEATH(ring_buffer<int, 0>{}, "capacity > 0");
}
```

`EXPECT_DEATH` accepts a regex for the stderr message — use `""` to match any output, or a specific pattern to assert the right assertion fired.

### Parameterized Tests

```cpp
class add_test : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(add_test, returns_exact_sum)
{
    SCOPED_TRACE(::testing::PrintToString(GetParam()));

    auto [a, b, expected] = GetParam();
    calculator calc;
    EXPECT_EQ(calc.add(a, b), expected);
}

INSTANTIATE_TEST_SUITE_P(
    calculator_tests,
    add_test,
    ::testing::Values(
        std::make_tuple(0,        0,       0),
        std::make_tuple(1,        1,       2),
        std::make_tuple(-1,       1,       0),
        std::make_tuple(INT_MAX,  0, INT_MAX)
    )
);
```

---

## Coverage Completeness Rule

For every mock in a test:

- List every method the SUT could call during that test
- Write an `EXPECT_CALL` for **each** expected call with exact args and `Times(N)`
- Any method that should NOT be called is verified absent automatically by `StrictMock`

If you cannot write an exact argument matcher because the value is computed internally, expose it through the interface or redesign the test to verify the output instead.

---

## Edge Cases Checklist

### Boundary Values

- [ ] Zero / empty — exact assertion on empty result
- [ ] One element — exact assertion on single-element result
- [ ] Maximum values (INT_MAX, SIZE_MAX) — exact expected return
- [ ] Minimum values (INT_MIN, 0 for unsigned) — exact expected return

### Error Conditions

- [ ] Each error path triggers exactly the right mock calls (no extras, none missing)
- [ ] Exception message contains exact expected text
- [ ] Return error code is exactly the right enum value, not just "non-zero"

### State Transitions

- [ ] Each state transition verified with exact mock call sequence and exact output

### Contract Violations

- [ ] Precondition violations covered by death tests where applicable

---

## Test Naming Convention

Pattern: `<unit>_<scenario>_<exact_expected_result>`

```cpp
TEST(parser_test, parse_empty_string_returns_nullopt)
TEST(parser_test, parse_valid_json_returns_document_with_three_keys)
TEST(parser_test, parse_invalid_json_throws_parse_error_mentioning_line_number)
TEST(calculator_test, divide_by_zero_throws_domain_error)
TEST_F(service_test, process_calls_save_exactly_once_with_correct_record)
TEST(buffer_death_test, access_out_of_bounds_aborts)
```

---

## Test File Organization

```text
tests/
├── CMakeLists.txt
├── unit/
│   ├── test_calculator.cpp
│   ├── test_parser.cpp
│   └── mocks/
│       ├── mock_database.hpp   # one mock per interface
│       └── mock_clock.hpp
└── integration/
    └── test_system.cpp
```

### CMakeLists.txt for Tests

```cmake
enable_testing()

find_package(GTest REQUIRED)

add_executable(unit_tests
    unit/test_calculator.cpp
    unit/test_parser.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        GTest::gtest_main
        GTest::gmock
        mylib
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
```

---

## Self-Check Before Submitting Any Test

Answer YES to all before writing the next test:

- [ ] Every mock is `StrictMock`?
- [ ] No `ON_CALL` anywhere?
- [ ] Every `EXPECT_CALL` has `Times(N)` with a concrete N?
- [ ] Every `EXPECT_CALL` uses exact argument matchers (no `_`)?
- [ ] Every numeric assertion uses `EXPECT_EQ`, not `EXPECT_GT` / `EXPECT_GE`?
- [ ] Every container assertion uses `ElementsAre` or equivalent, not `size() > 0`?
- [ ] Every call the SUT makes to a mock has a corresponding `EXPECT_CALL`?
- [ ] `ASSERT_*` used before any pointer dereference or container index that follows a check?
- [ ] Out-parameters captured and verified with `SaveArg` or `Invoke`?
- [ ] `InSequence` used only when order is the actual contract (not just an implementation detail)?
- [ ] `SCOPED_TRACE` added to any loop or parameterised test?
- [ ] Death tests added for each `assert()`/abort() contract?
- [ ] No `static` state or shared globals between tests?
- [ ] If the implementation passed wrong values, would this test catch it?

If any answer is NO, rewrite the test.

---

## References

- [GoogleTest Primer](https://google.github.io/googletest/primer.html)
- [GoogleTest Advanced](https://google.github.io/googletest/advanced.html)
- [GoogleMock for Dummies](https://google.github.io/googletest/gmock_for_dummies.html)
- [Matchers Reference](https://google.github.io/googletest/reference/matchers.html)
