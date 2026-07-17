---
name: programming-cpp-policy-based-di
description: Use when injecting dependencies in C++ at compile time — hiding system calls, third-party APIs, or multiple dependencies behind swappable policy types; use when writing testable C++ without virtual dispatch overhead
---

# C++ Policy-Based Dependency Injection

Compile-time DI via template policy parameters. Zero runtime overhead, full mockability, no virtual functions.

**Ownership rule:** The class always **owns** its dependencies as value members constructed from the policy type.
Never store dependency references or accept deps via constructor — that mixes two different DI styles and creates dangling-reference risk.

## Option 1 — Single Dependency (value member)

```cpp
struct real_dependency {
    return_type operation(args...) { return ::real_call(args...); }
};

template <typename Dependency = real_dependency>
struct my_class {
    result_type do_work(args...) {
        return dep_.operation(args...);
    }
private:
    Dependency dep_{};
};
```

## Option 2 — Policy Bundle (multiple dependencies)

Group all dependency type aliases into one policy struct. The outer class owns each dep as a value member:

```cpp
struct real_dep_a { return_type op_a(args...); };
struct real_dep_b { return_type op_b(args...); };

struct real_policy {
    using DepA = real_dep_a;
    using DepB = real_dep_b;
};

template <typename Policy = real_policy>
struct my_class {
    void method_a(args...) { dep_a_.op_a(args...); }
    void method_b(args...) { dep_b_.op_b(args...); }
private:
    typename Policy::DepA dep_a_{};
    typename Policy::DepB dep_b_{};
};
```

## Constraining Policies with Concepts (C++20)

Constrain the policy parameter with a concept so a mismatched type fails at the
declaration with a clear message instead of deep inside the template body:

```cpp
template <typename T>
concept Dependency = requires(T t, args_t a) {
    { t.operation(a) } -> std::same_as<return_type>;
};

template <Dependency Dep = real_dependency>
struct my_class {
    result_type do_work(args...) { return dep_.operation(args...); }
private:
    Dep dep_{};
};
```

A `MOCK_METHOD`-based wrapper still satisfies the concept through duck-typing —
no virtual base required.

## GMock Integration

GMock mock objects are non-copyable, so they cannot be owned by value inside the template class.
Use the **2-class + global** approach: a thin wrapper struct delegates to a global mock pointer.

```cpp
struct gmock_dependency {
    MOCK_METHOD(return_type, operation, (args...));
};

std::unique_ptr<gmock_dependency> g_mock;

struct mock_dependency {
    return_type operation(args...) { return g_mock->operation(args...); }
};

template <typename Dependency = real_dependency>
struct my_class {
    result_type do_work(args...) { return dep_.operation(args...); }
private:
    Dependency dep_{};
};

class my_class_test : public ::testing::Test {
protected:
    void SetUp() override    { g_mock = std::make_unique<gmock_dependency>(); }
    void TearDown() override { g_mock.reset(); }
};

using sut = my_class<mock_dependency>;

TEST_F(my_class_test, does_work) {
    EXPECT_CALL(*g_mock, operation(expected_args))
        .WillOnce(testing::Return(expected_value));
    sut obj;
    EXPECT_EQ(obj.do_work(args...), expected_value);
}
```

Policy bundle variant — one global per dep, one wrapper per dep:

```cpp
struct gmock_dep_a { MOCK_METHOD(return_type, op_a, (args...)); };
struct gmock_dep_b { MOCK_METHOD(return_type, op_b, (args...)); };

std::unique_ptr<gmock_dep_a> g_mock_a;
std::unique_ptr<gmock_dep_b> g_mock_b;

struct mock_dep_a { return_type op_a(args...) { return g_mock_a->op_a(args...); } };
struct mock_dep_b { return_type op_b(args...) { return g_mock_b->op_b(args...); } };

struct mock_policy {
    using DepA = mock_dep_a;
    using DepB = mock_dep_b;
};

class my_class_test : public ::testing::Test {
protected:
    void SetUp() override {
        g_mock_a = std::make_unique<gmock_dep_a>();
        g_mock_b = std::make_unique<gmock_dep_b>();
    }
    void TearDown() override { g_mock_a.reset(); g_mock_b.reset(); }
};

using sut = my_class<mock_policy>;

TEST_F(my_class_test, method_a_calls_op_a) {
    EXPECT_CALL(*g_mock_a, op_a(expected_args))
        .WillOnce(testing::Return(expected_value));
    sut obj;
    obj.method_a(args...);
}
```

## When to Use Each

| Scenario | Pattern |
| --- | --- |
| One external call to hide | Option 1 — single template param |
| 2+ related dependencies | Option 2 — policy bundle |
| GMock for any policy dep | 2-class + global (mocks are non-copyable) |
| Runtime polymorphism needed | Virtual interface instead |
| Hot path, zero overhead required | Either option (no vtable) |

## MANDATORY: Always Use GMock for Tests

**Never write hand-rolled spy structs.** Every test for policy-based DI code must use `MOCK_METHOD`.

`MOCK_METHOD` satisfies any policy concept through duck-typing — no virtual base class required.

**Why GMock, not hand-rolled spies:**

- `EXPECT_CALL` verifies exact arguments — spies require manual `EXPECT_EQ` per field
- `WillOnce(Return(...))` controls return values — spies hardcode them
- Argument matchers (`HasSubstr`, `_`, `Eq`, `AllOf`) handle complex inputs
- Failure messages name the expectation — spy failures say "expected X got Y" with no context
- `InSequence` verifies call order — spies cannot

**Red flag:** If you find yourself writing `struct spy_X { std::vector<...> captured; ... }` — stop. Use `MOCK_METHOD` instead.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Storing dep as `Dep& dep_` member | Dangling reference risk — store as value `Dep dep_{}` |
| Accepting dep in constructor (`my_class(Dep& d)`) | Mixes constructor injection with template DI — pick one style |
| Passing dep via `shared_ptr`/`unique_ptr` ctor arg | Same mixing problem — let the class construct its own dep |
| Hand-rolled spy structs in tests | Always use `MOCK_METHOD` |
| Wrapper calls itself recursively | Qualify the real call explicitly (e.g. `::free_function()`) |
| Unconstrained `auto` parameter (abbreviated template) | Constrain it with a concept (`void op(std::integral auto x)`) or use explicit `template<typename T>` |
| Missing `typename` on nested alias | `typename Policy::DepA`, not `Policy::DepA` |
| Calling `g_mock.op()` on `unique_ptr` | Dereference first: `g_mock->op()` |
| Forgetting `override` on `SetUp`/`TearDown` | Always add `override`; base declares them virtual |
