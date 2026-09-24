# Reference: constexpr

Reference material, not a rule. The rule is the one line in
[`../core.md`](../core.md): if it can be computed at compile time, it should be.
This file is the detail behind it.

Work moved to compile time costs nothing at runtime and is trivially testable.

## Forms

```cpp
// Constant
constexpr int kCacheLineBytes = 64;

// Function. Runs at compile time when the arguments are constants,
// at runtime otherwise.
constexpr int square(int x) { return x * x; }

constexpr int a = square(8);   // compile time
int b = square(n);             // runtime, n is not constexpr
```

`if constexpr` branches at compile time. The branch not taken is not compiled at
all, which is why it can contain code that would not even be valid for the other
type:

```cpp
template <typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>)            return val * 2;
    else if constexpr (std::is_floating_point_v<T>) return val * 2.0;
    else static_assert(always_false<T>, "unsupported type");
}
```

Prefer it to tag dispatch or SFINAE. Same outcome, far easier to read. It is
only valid inside a template.

## Lookup tables

```cpp
constexpr auto build_lut() {
    std::array<int, 256> t{};
    for (int i = 0; i < 256; ++i) t[i] = i * i;
    return t;
}
constexpr auto kLut = build_lut();   // built entirely at compile time

int val = kLut[x];                   // plain array read at runtime
```

## Forcing compile-time evaluation

A `constexpr` function is allowed to fall back to runtime. Assigning the result
to a `constexpr` variable makes the compiler prove it can be done at compile
time:

```cpp
constexpr int result = expensive_fn(args);  // fails to compile otherwise
```

That is the C++17 way to get what `consteval` gives in C++20. `consteval` and
`constinit` are not available. See [`../core.md`](../core.md).

## Testing

`constexpr` functions are pure, so there is nothing to mock.

```cpp
static_assert(square(8) == 64);      // caught before the binary exists
static_assert(kLut[4] == 16);

TEST(Square, PositiveInput_IsSquared) { EXPECT_EQ(square(8), 64); }
```

## Mistakes

| Mistake | Fix |
|---|---|
| Calling a non-`constexpr` function from a `constexpr` one | Mark the callee `constexpr`, or pull the computation out |
| `if constexpr` outside a template | It is only valid in a template |
| Assuming a `constexpr` function always runs at compile time | Assign the result to a `constexpr` variable |
| Reaching for `consteval` or `constinit` | C++20. Use a `constexpr` variable |
| Touching global state or doing I/O inside a `constexpr` function | It has to stay pure |
