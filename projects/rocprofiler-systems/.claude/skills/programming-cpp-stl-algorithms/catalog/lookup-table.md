# Lookup table

Catalog id: `lookup-table`

Values are looked up by key.

## Implement this

```cpp
struct string_hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};

class symbol_cache {
public:
    void add(std::string name, std::uint64_t addr) { addresses_.try_emplace(std::move(name), addr); }  // keeps the first

    std::optional<std::uint64_t> find(std::string_view name) const {
        auto it = addresses_.find(name);  // no std::string temporary
        if (it == addresses_.end()) { return std::nullopt; }
        return it->second;
    }

private:
    std::unordered_map<std::string, std::uint64_t, string_hash, std::equal_to<>> addresses_;
};
```

Overwrite instead of keep: `insert_or_assign(key, value)`.

## Switch only if

| Condition | Container |
| --- | --- |
| Iteration in key order, or output must be deterministic | `std::map<std::string, V, std::less<>>` |
| Built once, then only read | sorted `std::vector<std::pair<K, V>>` + `sorted-search` |
| Keys are dense integers | `dense-index` |

## Pitfalls

- `m[key]` to read inserts a default value on a miss and does not compile on a `const` map. Read with `find`.
- `insert` and `emplace` may build the value even when the key exists, and neither overwrites it. `try_emplace` builds nothing on a hit.
- `std::unordered_map` iteration order changes between runs and library versions. Never write output files or test expectations in that order.
- Without a transparent hash, `find(std::string_view)` does not compile, and `find(std::string(sv))` allocates on every lookup.
