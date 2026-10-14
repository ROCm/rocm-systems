# Dedup

Catalog id: `dedup`

Duplicates are removed or detected.

## Implement this

```cpp
std::ranges::sort(names);
auto [first, last] = std::ranges::unique(names);
names.erase(first, last);

// by value: sorts a copy
[[nodiscard]] bool
has_duplicates(std::vector<std::uint32_t> ids)
{
    std::ranges::sort(ids);
    return std::ranges::adjacent_find(ids) != ids.end();
}
```

## Switch only if the input order must be kept

```cpp
std::unordered_set<std::string> seen;
std::erase_if(names,
              [&seen](const std::string& name) { return !seen.insert(name).second; });
```

## Pitfalls

- `unique` removes only adjacent duplicates. Sort first, by the same key you deduplicate on.
- `unique` erases nothing. It returns the tail to erase.
- Use `std::set` only when elements arrive one at a time and must stay sorted and unique throughout. For a batch, sort + unique on a vector is faster.
