# Set operations

Catalog id: `set-ops`

Two collections are combined as sets. The `set_*` algorithms need sorted input and produce sorted output in one linear pass.

## Implement this

```cpp
std::vector<int> merge_unique(std::vector<int> a, std::vector<int> b) {
    a.insert(a.end(), b.begin(), b.end());
    std::ranges::sort(a);
    auto [first, last] = std::ranges::unique(a);
    a.erase(first, last);
    return a;
}

std::vector<int> only_in_a;
std::ranges::set_difference(sorted_a, sorted_b, std::back_inserter(only_in_a));
```

## Switch only if one side is small and unsorted

```cpp
std::unordered_set<int> excluded(small.begin(), small.end());
std::erase_if(large, [&excluded](int x) { return excluded.contains(x); });
```

## Pitfalls

- Unsorted input gives a silently wrong result.
- Duplicates in the input survive with multiset semantics. Dedup each input first if the result must be unique.
- `std::set` inputs are already sorted. Pass them to `set_*` directly; do not copy into vectors first.
