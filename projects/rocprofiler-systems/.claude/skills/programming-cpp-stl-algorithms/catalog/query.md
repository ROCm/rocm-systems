# Query

Catalog id: `query`

A loop returns on the first match or counts matches. The algorithm name states the question.

## Implement this

```cpp
auto it = std::ranges::find(entries, name, &entry::name);
if (it != entries.end()) { use(*it); }

bool any_stale = std::ranges::any_of(entries, [cutoff](const entry& e) { return e.timestamp < cutoff; });
auto n_valid   = std::ranges::count_if(samples, &sample::valid);

bool known = symbols.contains(name);  // std::map / std::set / std::unordered_map member, C++20
```

## Switch only if the same data is searched repeatedly

A `find` inside a loop over another container is O(n*m). Build a `lookup-table` once, or sort once and use `sorted-search`.

## Pitfalls

- `std::ranges::contains` and `std::string::contains` are C++23. Use `find(...) != end()` and `str.find(x) != std::string::npos`.
- `std::find` on a `std::map` / `std::set` is linear. Use the member `find` / `contains`.
- `count(...) > 0` to test existence scans everything. Use `any_of` / `find`, which stop at the first match.
