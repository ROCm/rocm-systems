# Filter

Catalog id: `filter`

Elements matching a condition are copied into another container.

## Implement this

```cpp
std::ranges::copy_if(in, std::back_inserter(out), &sample::valid);
```

## Switch only if the filtered elements are also transformed

```cpp
auto recent = entries | std::views::filter([cutoff](const entry& e) { return e.timestamp >= cutoff; })
                      | std::views::transform(&entry::name);
std::vector<std::string> names(recent.begin(), recent.end());
```

## Pitfalls

- Filtering in place is `erase`, not `copy_if` into a temporary and swap.
- A `filter_view` caches `begin()`. Iterate it through a non-const reference, and do not reuse it after the source changes.
- `out.reserve(in.size())` is an upper bound. Use it only when most elements pass.
