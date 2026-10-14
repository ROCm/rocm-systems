# Sorted search

Catalog id: `sorted-search`

A value is looked up in a range already sorted by that key. O(log n), no allocation.

## Implement this

```cpp
auto it = std::ranges::lower_bound(symbols, addr, {}, &symbol::start);  // first start >= addr
if (it != symbols.end() && it->start == addr) { use(*it); }

for (const record& r : std::ranges::equal_range(records, device, {}, &record::device)) { use(r); }
```

## Switch only if the answer is yes or no

```cpp
bool present = std::ranges::binary_search(sorted_ids, id);
```

## Pitfalls

- The range must be sorted with the same comparator and projection as the search. Otherwise results are silently wrong.
- `std::ranges::lower_bound` on a `std::map` / `std::set` walks iterators linearly. Use the member `lower_bound`.
- To find the interval containing `addr`, use `upper_bound` and step back one, after checking it is not `begin()`.
