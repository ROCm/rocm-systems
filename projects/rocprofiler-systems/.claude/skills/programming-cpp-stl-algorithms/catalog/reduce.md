# Reduce

Catalog id: `reduce`

A loop folds elements into one value. `<numeric>` has no `std::ranges::` overloads in C++20; use iterator pairs here.

## Implement this

```cpp
[[nodiscard]] double
mean_latency(const std::vector<double>& latencies)
{
    if(latencies.empty())
    {
        return 0.0;
    }
    return std::reduce(latencies.begin(), latencies.end(), 0.0) /
           static_cast<double>(latencies.size());
}

[[nodiscard]] double
total_bytes(const std::vector<entry>& entries)
{
    return std::transform_reduce(entries.begin(), entries.end(), 0.0, std::plus<>{},
                                 [](const entry& item) { return item.bytes; });
}
```

## Switch only if the order of floating-point additions must be exact

`std::reduce` may reorder operations, so the last bits can change between builds. For reproducible output, use `std::accumulate(latencies.begin(), latencies.end(), 0.0)`.

String join has no C++20 algorithm (`std::views::join_with` is C++23). Write a `+=` loop:

```cpp
std::string joined;
for(const auto& part : parts)
{
    if(!joined.empty())
    {
        joined += separator;
    }
    joined += part;
}
```

## Pitfalls

- The init value sets the result type. `std::reduce(values.begin(), values.end(), 0)` on doubles sums into `int`. Write `0.0`, or `std::uint64_t{0}` for byte counts.
- An `int` sum of `uint32_t` counters overflows. Choose the init type for the total, not for the element.
- `std::ranges::fold_left` is C++23.
- `<numeric>` algorithms call their functions directly, not through `std::invoke`. A member pointer like `&entry::bytes` does not compile there; write a lambda.
