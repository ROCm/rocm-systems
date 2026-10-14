---
name: programming-cpp-constexpr
description: Use when moving computation from runtime to compile time — precomputed lookup tables, compile-time constants, type-based template dispatch, or zero-cost conditional branching with if constexpr; use when a function or value could be evaluated before the binary runs
---

# C++ constexpr — Compile-Time Computation (C++20)

Move work to compile time: no runtime cost, no dependencies, testable with `static_assert`.
Every `constexpr` variable takes the `k_` prefix (`cpp-conventions.md`).

## Choose the form

| Situation | Use |
| --- | --- |
| Constant known at compile time | `constexpr` variable |
| Pure function that may run at compile time or runtime | `constexpr` function |
| Function that must never run at runtime (validating a literal) | `consteval` function |
| Static or global that needs constant initialization, but stays mutable | `constinit` |
| Branch on a type property inside a template | `if constexpr` |
| Precomputed mapping | `constexpr` function returning `std::array` |

## Forms

```cpp
constexpr int k_cache_line_bytes = 64;

[[nodiscard]] constexpr int
square(int value)
{
    return value * value;
}

constexpr int k_demo_input = 8;
constexpr int k_compiled   = square(k_demo_input);  // compile time
const int     runtime      = square(read_input());  // runtime: same function

[[nodiscard]] consteval int
factorial(int number)
{
    return number <= 1 ? 1 : number * factorial(number - 1);
}
```

`consteval` fails to compile for a non-constant argument. Prefer it to assigning the
result to a `constexpr` variable when no call may ever run at runtime. Use the variable
form when the same function also needs a runtime path.

`constinit int g_counter = compute_seed();` removes static-initialization-order
problems. It does not make the variable immutable; use `constexpr` for that.

## if constexpr

The untaken branch is not instantiated. Prefer it over tag dispatch or SFINAE.

```cpp
template <typename T>
inline constexpr bool k_always_false = false;

template <typename T>
[[nodiscard]] auto
process(T value)
{
    if constexpr(std::is_integral_v<T>) return value * 2;
    else if constexpr(std::is_floating_point_v<T>) return value * 2.0;
    else static_assert(k_always_false<T>, "unsupported type");
}
```

## Lookup table

```cpp
constexpr int k_lut_size = 256;

[[nodiscard]] constexpr auto
build_lut()
{
    std::array<int, k_lut_size> table{};
    for(int i = 0; i < k_lut_size; ++i)
        table[i] = i * i;
    return table;
}
constexpr auto k_lut = build_lut();  // runtime cost is one array read
```

Most standard algorithms are `constexpr` in C++20; use `std::ranges::generate` or
`std::ranges::fill` instead of a hand-written loop when it reads better.

## Test

A `constexpr` function is pure, so it needs no mock. Check it at compile time and, when a
runtime path exists, once more with GTest:

```cpp
static_assert(square(k_demo_input) == k_demo_input * k_demo_input);
```

## Gotchas

- `static_assert(false, ...)` in a discarded `if constexpr` branch is ill-formed before
  C++23 on older compilers. Use a dependent value such as `k_always_false<T>`.
- A `constexpr` function runs at compile time only when every argument is constant and
  the result is used in a constant expression. Otherwise it runs at runtime.
- A `constexpr` function may only call `constexpr` functions. Mark the callee or move
  the computation out.
- `if constexpr` outside a template does not discard the branch.
- A `constexpr` function stays pure: no I/O, no globals.
