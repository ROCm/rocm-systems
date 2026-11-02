# Sort

Catalog id: `sort`

Elements are ordered by a key. A projection names the key; a comparator lambda is only for compound keys.

## Implement this

```cpp
std::ranges::sort(records, {}, &record::name);                      // ascending by name
std::ranges::sort(records, std::ranges::greater{}, &record::time);  // descending by time
std::ranges::sort(records, [](const record& left, const record& right) {  // compound key
    return std::tie(left.device, left.time) < std::tie(right.device, right.time);
});
```

## Switch only if equal keys must keep their input order

```cpp
std::ranges::stable_sort(records, {}, &record::device);
```

## Pitfalls

- A comparator must be a strict weak ordering. `<=` instead of `<` is undefined behavior and can crash `sort`.
- `NaN` keys break the strict weak ordering. Remove them with `std::erase_if` before sorting.
- Sorting to read only the first k elements is `top-k`.
