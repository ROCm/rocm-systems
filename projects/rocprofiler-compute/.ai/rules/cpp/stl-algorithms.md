# C++ STL Algorithms and Containers

Prefer a named STL algorithm to a hand-written loop. The algorithm says what you
meant, it is harder to get wrong at the edges, and the optimizer already knows
it.

Reference: [cppreference, algorithms](https://en.cppreference.com/w/cpp/algorithm).

C++20 ranges are not available outside `torch_trace_collector`. Use iterator
pairs. See [`core.md`](core.md) for the language version rule.

## Replace a loop with

| A loop that... | Use |
|---|---|
| Finds an element | `std::find`, `std::find_if` |
| Checks whether any or all match | `std::any_of`, `std::all_of`, `std::none_of` |
| Counts | `std::count`, `std::count_if` |
| Copies | `std::copy`, `std::copy_if` |
| Transforms | `std::transform` |
| Sums or folds | `std::accumulate`, `std::reduce` |
| Removes | `std::remove_if` plus `erase` |
| Sorts | `std::sort`, `std::stable_sort`, `std::partial_sort` |
| Finds the smallest or largest | `std::min_element`, `std::max_element`, `std::minmax_element` |
| Reverses | `std::reverse` |
| Fills | `std::fill`, `std::generate`, `std::iota` |
| Checks order | `std::is_sorted` |
| Nests to intersect or union two ranges | `std::set_intersection`, `std::set_union`, `std::set_difference` |

## Picking a container

`std::vector` is the default. Reach for something else only with a reason.

| Container | Use when | Cost |
|---|---|---|
| `std::vector` | Default. Random access, growth at the end | O(1) access, amortized O(1) `push_back` |
| `std::array` | Size fixed at compile time | O(1) access, no heap |
| `std::deque` | Growth at both ends | O(1) access, O(1) push at either end |
| `std::list` | Frequent insert or erase in the middle, no random access | O(1) splice, O(n) access |
| `std::set`, `std::map` | Sorted, ordered iteration matters | O(log n) |
| `std::unordered_set`, `std::unordered_map` | Lookup only, order does not matter | O(1) average |
| `std::stack`, `std::queue`, `std::priority_queue` | LIFO, FIFO, always want the extreme | Follows the underlying container |

For fewer than about twenty elements, a `std::vector` with a linear scan usually
beats a `std::map`. See [`performance.md`](performance.md).

## Common forms

```cpp
// Search
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 5; });
bool ok = std::all_of(v.begin(), v.end(), pred);

// Erase-remove
v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());

// Transform into a new container
std::vector<std::string> names;
names.reserve(people.size());
std::transform(people.begin(), people.end(), std::back_inserter(names),
               [](const Person& p) { return p.name; });

// Filter into a new container
std::vector<int> evens;
std::copy_if(v.begin(), v.end(), std::back_inserter(evens),
             [](int x) { return x % 2 == 0; });

// Fold
int sum = std::accumulate(v.begin(), v.end(), 0);

// Binary search on a sorted range
auto lo = std::lower_bound(v.begin(), v.end(), value);   // first >= value
auto hi = std::upper_bound(v.begin(), v.end(), value);   // first > value

// Clamp
int c = std::clamp(value, low, high);
```

`reserve()` before a `back_inserter` loop, or you pay for the regrowth you were
trying to avoid.

## Numeric operations

From `<numeric>`: `std::accumulate`, `std::reduce`, `std::inner_product`,
`std::partial_sum`, `std::adjacent_difference`, `std::transform_reduce`,
`std::iota`.

## Set operations

These need both ranges sorted:

```cpp
std::vector<int> result;
std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                      std::back_inserter(result));
```

Also `std::set_union`, `std::set_difference`,
`std::set_symmetric_difference`, `std::includes`, and `std::merge`.

## When a raw loop is fine

- The body does several unrelated things and splitting it would be worse.
- The loop breaks early on a condition no predicate captures cleanly.
- The algorithm version needs a lambda so convoluted that the loop reads better.
