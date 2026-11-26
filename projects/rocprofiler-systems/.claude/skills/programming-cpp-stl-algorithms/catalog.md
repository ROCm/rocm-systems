# C++ STL Catalog

Search this file, then open **only** the matching example and write its **Implement this** code.

```bash
rg -n -i "operation or container" catalog.md
```

Ids: `query` `transform` `filter` `erase` `reduce` `sort` `top-k` `dedup` `min-max` `sorted-search` `set-ops` `modify-while-iterating` `sequence-container` `dense-index` `lookup-table` `priority-queue`

| id | search | symptom | write this | switch only if | example |
| --- | --- | --- | --- | --- | --- |
| query | find, find_if, contains, any, all, none, count, exists | A loop returns or counts on the first match | `std::ranges::find` / `any_of` / `count_if` with a projection. Map/set: member `contains` / `find` | The same data is searched repeatedly: build a `lookup-table`, or sort once for `sorted-search` | [example](catalog/query.md) |
| transform | map, convert, project, names of, back_inserter, view | A new container is built from each element of another | `std::views::transform` copied into a container, or `std::ranges::transform` into a reserved vector | The result feeds one more loop: iterate the view directly | [example](catalog/transform.md) |
| filter | copy_if, select, keep, valid, matching | Elements matching a condition are copied out | `std::ranges::copy_if(in, std::back_inserter(out), pred)` | The filtered result is also transformed: `std::views::filter` piped into `transform` | [example](catalog/filter.md) |
| erase | remove, erase, drop, prune, remove_if, delete from | Elements are removed from a container in place | `std::erase(c, value)` / `std::erase_if(c, pred)` | Each removed element needs a side effect: iterator loop with `it = c.erase(it)` | [example](catalog/erase.md) |
| reduce | sum, total, mean, average, accumulate, product, join | A loop folds elements into one value | `std::reduce` or `std::transform_reduce` with a typed init (`0.0`, `std::uint64_t{0}`) | Order of floating-point adds must be exact: `std::accumulate`. Strings: a `+=` loop | [example](catalog/reduce.md) |
| sort | sort, order by, descending, comparator, stable | Elements are ordered by a key | `std::ranges::sort(v, {}, &T::key)` | Equal keys must keep input order: `std::ranges::stable_sort` | [example](catalog/sort.md) |
| top-k | top, largest, smallest, k best, nth, partial | Only the first k of the sorted order are needed | `std::ranges::partial_sort_copy` into a vector of `min(k, n)` | The k elements need no order among themselves: `std::ranges::nth_element` | [example](catalog/top-k.md) |
| dedup | unique, duplicates, distinct, has_duplicates | Duplicates are removed or detected | Sort, then `std::ranges::unique` + `erase`. Detect: sort a copy, then `std::ranges::adjacent_find` | Input order must be kept: `std::unordered_set` with `erase_if` | [example](catalog/dedup.md) |
| min-max | min, max, smallest, largest, clamp, range of | The extreme element or a bounded value is needed | `std::ranges::max_element(v, {}, &T::key)` checked against `end()`; `std::clamp` | Both ends are needed: `std::ranges::minmax_element` | [example](catalog/min-max.md) |
| sorted-search | binary search, lower_bound, upper_bound, equal_range, sorted vector | A value is looked up in a sorted range | `std::ranges::lower_bound(v, key, {}, &T::key)` | Only a yes/no answer: `std::ranges::binary_search` | [example](catalog/sorted-search.md) |
| set-ops | union, intersection, difference, merge, both, either | Two collections are combined as sets | Sort and dedup both, then `std::ranges::set_union` / `set_intersection` / `set_difference` | One side is small and unsorted: `std::unordered_set` of it and `erase_if` on the other | [example](catalog/set-ops.md) |
| modify-while-iterating | invalidation, push_back in loop, erase in loop, append while iterating, dangling | A container grows or shrinks while it is iterated | Index loop that re-reads `size()` for appends; `std::erase_if` for removals | The loop must stay a range-for: collect new elements in a second vector, append after the loop | [example](catalog/modify-while-iterating.md) |
| sequence-container | vector, array, deque, list, span, vector<bool>, reserve | A container type is chosen for a sequence | `std::vector` with `reserve`; `std::array` for a compile-time size; `std::span` for parameters | Push/pop at both ends: `std::deque`. Splice or stable iterators under middle insert: `std::list` | [example](catalog/sequence-container.md) |
| dense-index | per thread, per device, per id, tid, index, small integer key | Keys are dense integers 0..N-1 | `std::vector<T>` indexed by id, sized once before use | Ids are sparse or unbounded: `std::unordered_map` | [example](catalog/dense-index.md) |
| lookup-table | map, unordered_map, cache, dictionary, key to value, operator[], try_emplace, string_view | Values are looked up by key | `std::unordered_map` with `try_emplace` / `insert_or_assign` and `find` for reads. String keys: transparent hash | Ordered iteration or deterministic output: `std::map<K, V, std::less<>>`. Built once, read many: sorted vector | [example](catalog/lookup-table.md) |
| priority-queue | priority, deadline, smallest first, scheduler, heap, top, pop | The smallest or largest element is always taken next | `std::priority_queue` with a named comparator; copy `top()`, then `pop()` | Priorities change after insert, or arbitrary removal: `std::set` used as an ordered queue | [example](catalog/priority-queue.md) |
