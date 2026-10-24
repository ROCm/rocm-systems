# Erase

Catalog id: `erase`

Elements are removed from a container in place. C++20 `std::erase` / `std::erase_if` work for every standard container, including `std::map` and `std::unordered_map`.

## Implement this

```cpp
std::erase_if(entries, [cutoff](const entry& item) { return item.timestamp < cutoff; });
std::erase_if(counts, [](const auto& entry) { return entry.second == 0; });  // std::map
std::erase(path, '\\');  // std::string
```

## Switch only if each removed element needs a side effect

```cpp
for(auto iter = handles.begin(); iter != handles.end();)
{
    if(iter->second.closed())
    {
        release(iter->second);
        iter = handles.erase(iter);
    }
    else
    {
        ++iter;
    }
}
```

## Pitfalls

- `std::remove_if` alone erases nothing. It moves kept elements forward and leaves a moved-from tail.
- `container.erase(iter)` followed by `++iter` uses an invalidated iterator. Always assign `iter = container.erase(iter)`.
- The predicate must not modify the container or depend on how many elements were already removed.
