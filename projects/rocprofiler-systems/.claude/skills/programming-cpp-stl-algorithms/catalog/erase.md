# Erase

Catalog id: `erase`

Elements are removed from a container in place. C++20 `std::erase` / `std::erase_if` work for every standard container, including `std::map` and `std::unordered_map`.

## Implement this

```cpp
std::erase_if(entries, [cutoff](const entry& e) { return e.timestamp < cutoff; });
std::erase_if(counts, [](const auto& kv) { return kv.second == 0; });  // std::map
std::erase(path, '\\');                                                 // std::string
```

## Switch only if each removed element needs a side effect

```cpp
for (auto it = handles.begin(); it != handles.end();) {
    if (it->second.closed()) {
        release(it->second);
        it = handles.erase(it);
    } else {
        ++it;
    }
}
```

## Pitfalls

- `std::remove_if` alone erases nothing. It moves kept elements forward and leaves a moved-from tail.
- `c.erase(it)` followed by `++it` uses an invalidated iterator. Always assign `it = c.erase(it)`.
- The predicate must not modify the container or depend on how many elements were already removed.
