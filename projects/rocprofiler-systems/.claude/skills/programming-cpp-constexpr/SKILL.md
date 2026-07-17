---
name: programming-cpp-constexpr
description: Use when moving computation from runtime to compile time — precomputed lookup tables, compile-time constants, type-based template dispatch, or zero-cost conditional branching with if constexpr; use when a function or value could be evaluated before the binary runs
---

# C++ constexpr — Compile-Time Computation (C++20)

Move work to compile time: zero runtime cost, no dependencies, trivially testable.

## Core Forms

### constexpr variable

```cpp
constexpr int cache_line_bytes = 64;
constexpr double pi = 3.14159265358979;
```

### constexpr function

Evaluated at compile time when all arguments are compile-time constants; falls back to runtime otherwise:

```cpp
constexpr int square(int x) { return x * x; }

constexpr int a = square(8);  // compile time
int b = square(n);            // runtime — n is not constexpr
```

### consteval function (immediate function)

`consteval` REQUIRES compile-time evaluation — every call must be a constant expression, or it fails to compile. Use it when running at runtime would be a bug (e.g. validating a literal):

```cpp
consteval int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

constexpr int f5 = factorial(5);  // OK — compile time
int n = read();
int bad = factorial(n);           // ERROR — n is not a constant expression
```

Prefer `consteval` over the old "assign to a `constexpr` variable to force it" trick when the function should *never* run at runtime.

### constinit variable

Guarantees a static / thread-local is initialized at compile time — eliminates the static-initialization-order fiasco. Unlike `constexpr`, the variable stays mutable:

```cpp
constinit int g_counter = compute_seed();  // constant-initialized, still mutable
```

### if constexpr

Compile-time branch — the non-taken branch is not compiled, not just skipped:

```cpp
template <typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>)            return val * 2;
    else if constexpr (std::is_floating_point_v<T>) return val * 2.0;
    else                                            static_assert(false, "unsupported type");
}
```

Prefer `if constexpr` over tag dispatch or SFINAE — same result, far more readable.

### Compile-time lookup table

```cpp
constexpr auto build_lut() {
    std::array<int, 256> t{};
    for (int i = 0; i < 256; ++i) t[i] = i * i;
    return t;
}
constexpr auto lut = build_lut();  // built entirely at compile time

int val = lut[x];  // pure array access at runtime — zero computation
```

In C++20 most standard algorithms are `constexpr`, so you can use them inside a `constexpr` function instead of a hand-written loop:

```cpp
constexpr auto build_lut() {
    std::array<int, 256> t{};
    std::ranges::generate(t, [i = 0]() mutable { return i * i++; });
    return t;
}
```

## When to Use

| Situation | Use |
| --- | --- |
| Constant known at compile time | `constexpr` variable |
| Pure function on compile-time inputs | `constexpr` function |
| Template branch on type property | `if constexpr` |
| Precomputed mapping / LUT | `constexpr` function returning `std::array` |
| Enforce compile-time evaluation | `consteval` function (or assign result to a `constexpr` variable) |
| Function must NEVER run at runtime | `consteval` |
| Static/global needing constant init (avoid SIOF) | `constinit` |

## Testing

`constexpr` functions are pure — no side effects, no dependencies — so no mocking is ever needed.

```cpp
// Compile-time: catches bugs before the binary exists
static_assert(square(8) == 64);
static_assert(lut[4] == 16);

// Runtime: works with GTest unchanged
TEST(Square, PositiveInput) { EXPECT_EQ(square(8), 64); }
```

## Forcing Compile-Time Evaluation

Two ways, strongest first:

```cpp
// 1. consteval — the function itself can only ever run at compile time
consteval int checked(int x) { return x; }

// 2. Assign to a constexpr variable — forces evaluation at this call site
constexpr int result = expensive_fn(args);  // fails to compile if args are not constexpr
```

Use `consteval` when *no* call should ever fall back to runtime; use the `constexpr`-variable trick when the same function must also be usable at runtime elsewhere.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Calling non-`constexpr` function inside `constexpr` function | Mark the callee `constexpr` or extract the computation |
| `if constexpr` outside a template | Only valid in a template context |
| Assuming `constexpr` function always runs at compile time | Use `consteval`, or assign to a `constexpr` variable to enforce it |
| Using `consteval` for a function that also needs a runtime path | `consteval` forbids runtime calls — use plain `constexpr` instead |
| Expecting `constinit` to make a variable immutable | It only guarantees constant *initialization*; the variable stays mutable — use `constexpr` for immutability |
| Mutating state inside `constexpr` function | Allowed but the function must remain pure (no I/O, no globals) |
