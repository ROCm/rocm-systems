# C++ Performance

This code runs inside somebody else's profiling loop. A slow collector changes
the numbers we are there to measure, so performance is part of correctness here,
not a later optimization pass.

The [Core Guidelines performance
section](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-performance)
is the reference. The short version:

- Do not optimize without a reason, and do not optimize what is not hot.
- Complicated code is not automatically faster than simple code. Neither is
  low-level code.
- Do not claim a speedup without a measurement.
- Design so that optimization is possible later. Lean on the static type system.
- Move work from run time to compile time.
- Access memory predictably.

## The hot path

The sampling and dispatch callbacks are the hot path. On that path:

- No allocation.
- No blocking.
- No I/O.

Everything else follows from those three. What counts as the hot path and what
you are allowed to do on it is in
[`callback-boundaries.md`](callback-boundaries.md).

## Allocation

Allocate outside the loop, not inside it.

```cpp
// Bad: one allocation per iteration
void process_items(const std::vector<Item>& items) {
    for (const auto& item : items) {
        auto result = std::make_unique<Result>();
        // ...
    }
}

// Good: one object, reused
void process_items(const std::vector<Item>& items) {
    Result result;
    for (const auto& item : items) {
        result.reset();
        // ...
    }
}
```

Call `reserve()` whenever the final size is known.

## Copies

```cpp
// Bad: copies the whole vector
void process(std::vector<int> data);

// Good: read-only access
void process(const std::vector<int>& data);

// Good: by value when you intend to own it, so the caller can move in
void take_ownership(std::vector<int> data) {
    m_data = std::move(data);
}
```

Return by value and let RVO handle it. Never return a reference to a temporary.

In a range-for, bind with `const auto&` unless you need a copy.

## Move semantics

Use `std::move` when transferring ownership. For a class that owns something
heavy, write the move constructor and move assignment, and mark both `noexcept`.

## Cache behaviour

Contiguous storage beats pointer chasing. `std::vector` is the default
container.

Walk arrays in the order they are laid out:

```cpp
// Bad: a cache miss on every access
for (int col = 0; col < cols; ++col)
    for (int row = 0; row < rows; ++row)
        matrix[row][col] = 0;

// Good: sequential
for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
        matrix[row][col] = 0;
```

## Pitfalls

| Pitfall | Instead |
|---|---|
| `std::endl` in a loop | `'\n'`, which does not flush |
| `std::map` for a handful of entries | `std::vector` plus a linear scan |
| Building a string by concatenation in a loop | Reserve, or use a stream |
| `shared_ptr` where nothing is shared | `unique_ptr`, no atomic refcount |
| A virtual call in a hot loop | A template or CRTP |
| `std::function` on the hot path | A template or a function pointer |
| Exceptions as control flow | A return value or `std::optional` |
| Copying in a range-for | `const auto&` |

## Checklist

- [ ] No allocation, blocking, or I/O on the hot path
- [ ] Large objects passed by `const&`, small ones by value
- [ ] `std::move` on ownership transfer
- [ ] `reserve()` where the size is known
- [ ] Sequential memory access
- [ ] `constexpr` for anything computable at compile time
- [ ] No unnecessary virtual calls in hot code
- [ ] Measured before micro-optimizing
