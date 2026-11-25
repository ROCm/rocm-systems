# Top-k

Catalog id: `top-k`

Only the first k elements of the sorted order are needed. A full sort does O(n log n) work for k results.

## Implement this

```cpp
[[nodiscard]] std::vector<std::uint64_t>
top_k(const std::vector<sample>& samples, std::size_t count)
{
    std::vector<std::uint64_t> top(std::min(count, samples.size()));
    std::ranges::partial_sort_copy(samples | std::views::transform(&sample::ts), top,
                                   std::ranges::greater{});
    return top;
}
```

## Switch only if the k elements need no order among themselves

```cpp
std::ranges::nth_element(values, values.begin() + count,
                         std::ranges::greater{});  // values[0..count) are the largest
values.resize(count);
```

## Pitfalls

- Size the output with `std::min(count, total)`. `values.begin() + count` past `end()` is undefined behavior; check `count < values.size()` before `nth_element`.
- `partial_sort_copy` leaves the input untouched. `partial_sort` and `nth_element` reorder it.
