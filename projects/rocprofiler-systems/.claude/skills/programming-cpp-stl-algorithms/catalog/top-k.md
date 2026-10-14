# Top-k

Catalog id: `top-k`

Only the first k elements of the sorted order are needed. A full sort does O(n log n) work for k results.

## Implement this

```cpp
std::vector<std::uint64_t> top_k(const std::vector<sample>& samples, std::size_t k) {
    std::vector<std::uint64_t> top(std::min(k, samples.size()));
    std::ranges::partial_sort_copy(samples | std::views::transform(&sample::ts), top,
                                   std::ranges::greater{});
    return top;
}
```

## Switch only if the k elements need no order among themselves

```cpp
std::ranges::nth_element(v, v.begin() + k, std::ranges::greater{});  // v[0..k) are the k largest
v.resize(k);
```

## Pitfalls

- Size the output with `std::min(k, n)`. `v.begin() + k` past `end()` is undefined behavior; check `k < v.size()` before `nth_element`.
- `partial_sort_copy` leaves the input untouched. `partial_sort` and `nth_element` reorder it.
