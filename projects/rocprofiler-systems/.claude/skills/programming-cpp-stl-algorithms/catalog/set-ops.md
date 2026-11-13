# Set operations

Catalog id: `set-ops`

Two collections are combined as sets. The `set_*` algorithms need sorted input and produce sorted output in one linear pass.

## Implement this

```cpp
[[nodiscard]] std::vector<int>
merge_unique(std::vector<int> left, std::vector<int> right)
{
    left.insert(left.end(), right.begin(), right.end());
    std::ranges::sort(left);
    auto [first, last] = std::ranges::unique(left);
    left.erase(first, last);
    return left;
}

std::vector<int> only_in_left;
std::ranges::set_difference(sorted_left, sorted_right, std::back_inserter(only_in_left));
```

## Switch only if one side is small and unsorted

```cpp
std::unordered_set<int> excluded(small.begin(), small.end());
std::erase_if(large, [&excluded](int value) { return excluded.contains(value); });
```

## Pitfalls

- Unsorted input gives a silently wrong result.
- Duplicates in the input survive with multiset semantics. Dedup each input first if the result must be unique.
- `std::set` inputs are already sorted. Pass them to `set_*` directly; do not copy into vectors first.
