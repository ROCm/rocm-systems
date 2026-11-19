# Sorted search

Catalog id: `sorted-search`

A value is looked up in a range already sorted by that key. O(log n), no allocation.

## Implement this

```cpp
const auto iter = std::ranges::lower_bound(symbols, address, {},
                                           &symbol::start);  // first start >= address
if(iter != symbols.end() && iter->start == address)
{
    use(*iter);
}

for(const record& item : std::ranges::equal_range(records, device, {}, &record::device))
{
    use(item);
}
```

## Switch only if the answer is yes or no

```cpp
const bool present = std::ranges::binary_search(sorted_ids, key);
```

## Pitfalls

- The range must be sorted with the same comparator and projection as the search. Otherwise results are silently wrong.
- `std::ranges::lower_bound` on a `std::map` / `std::set` walks iterators linearly. Use the member `lower_bound`.
- To find the interval containing `address`, use `upper_bound` and step back one, after checking it is not `begin()`.
