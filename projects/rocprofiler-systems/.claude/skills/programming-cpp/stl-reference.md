# C++ STL Algorithms & Containers Reference

Reference: [CppReference - Algorithms](https://en.cppreference.com/w/cpp/algorithm)

## Container Selection

### Sequence Containers

| Container | Use When | Complexity |
| ----------- | ---------- | ------------ |
| `std::vector` | **Default choice**, random access, add at end | O(1) access, O(1) amortized push_back |
| `std::array` | Fixed size known at compile time | O(1) access, no heap |
| `std::deque` | Add/remove at both ends | O(1) access, O(1) push_front/back |
| `std::list` | Frequent insert/remove in middle | O(1) insert/remove, O(n) access |
| `std::forward_list` | Minimal memory, forward-only | O(1) insert after |

### Associative Containers

| Container | Use When | Complexity |
| ----------- | ---------- | ------------ |
| `std::map` / `std::set` | Sorted, ordered iteration | O(log n) |
| `std::unordered_map` / `std::unordered_set` | Fastest lookup, order doesn't matter | O(1) avg |
| `std::multimap` / `std::multiset` | Duplicate keys allowed | O(log n) |

### Adaptors

| Container | Use When |
| ----------- | ---------- |
| `std::stack` | LIFO |
| `std::queue` | FIFO |
| `std::priority_queue` | Always access largest/smallest |

## Non-modifying Operations

```cpp
auto it = std::find(v.begin(), v.end(), value);
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 5; });

bool any = std::any_of(v.begin(), v.end(), pred);
bool all = std::all_of(v.begin(), v.end(), pred);
bool none = std::none_of(v.begin(), v.end(), pred);

int count = std::count(v.begin(), v.end(), value);
int count = std::count_if(v.begin(), v.end(), pred);

auto it = std::search(v.begin(), v.end(), sub.begin(), sub.end());
auto it = std::adjacent_find(v.begin(), v.end());
```

## Modifying Operations

```cpp
std::copy(src.begin(), src.end(), dest.begin());
std::copy_if(src.begin(), src.end(), std::back_inserter(dest), pred);

std::transform(v.begin(), v.end(), v.begin(), [](int x) { return x * 2; });
std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>{});

std::fill(v.begin(), v.end(), value);
std::generate(v.begin(), v.end(), generator);
std::iota(v.begin(), v.end(), 0);  // 0, 1, 2, 3...

// Erase-remove idiom
v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());

std::replace(v.begin(), v.end(), old_val, new_val);
std::reverse(v.begin(), v.end());
std::rotate(v.begin(), v.begin() + n, v.end());
v.erase(std::unique(v.begin(), v.end()), v.end());
```

## Sorting & Searching

```cpp
std::sort(v.begin(), v.end());
std::sort(v.begin(), v.end(), std::greater<>{});
std::stable_sort(v.begin(), v.end());
std::partial_sort(v.begin(), v.begin() + n, v.end());
std::nth_element(v.begin(), v.begin() + n, v.end());

bool sorted = std::is_sorted(v.begin(), v.end());

// Binary search (sorted range required)
bool found = std::binary_search(v.begin(), v.end(), value);
auto it = std::lower_bound(v.begin(), v.end(), value);
auto it = std::upper_bound(v.begin(), v.end(), value);
auto [lo, hi] = std::equal_range(v.begin(), v.end(), value);
```

## Min/Max

```cpp
int m = std::min(a, b);
int m = std::max({a, b, c, d});
auto [lo, hi] = std::minmax(a, b);

auto it = std::min_element(v.begin(), v.end());
auto it = std::max_element(v.begin(), v.end());
auto [min_it, max_it] = std::minmax_element(v.begin(), v.end());

int clamped = std::clamp(value, low, high);
```

## Numeric (`<numeric>`)

```cpp
int sum = std::accumulate(v.begin(), v.end(), 0);
int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<>{});
int sum = std::reduce(v.begin(), v.end());  // parallelizable

int dot = std::inner_product(a.begin(), a.end(), b.begin(), 0);
std::partial_sum(v.begin(), v.end(), result.begin());
std::adjacent_difference(v.begin(), v.end(), result.begin());
auto result = std::transform_reduce(v.begin(), v.end(), init, binary_op, unary_op);
```

## Set Operations (sorted ranges)

```cpp
std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
bool includes = std::includes(a.begin(), a.end(), b.begin(), b.end());
std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
```

## Heap Operations

```cpp
std::make_heap(v.begin(), v.end());
v.push_back(value); std::push_heap(v.begin(), v.end());
std::pop_heap(v.begin(), v.end()); v.pop_back();
std::sort_heap(v.begin(), v.end());
bool is_heap = std::is_heap(v.begin(), v.end());
```

## Common Patterns

### Transform + Back Inserter

```cpp
std::vector<std::string> names;
std::transform(people.begin(), people.end(),
    std::back_inserter(names), [](const Person& p) { return p.name; });
```

### Copy If + Back Inserter

```cpp
std::vector<int> evens;
std::copy_if(v.begin(), v.end(), std::back_inserter(evens),
    [](int x) { return x % 2 == 0; });
```

### Accumulate with Custom Op

```cpp
std::string result = std::accumulate(strings.begin(), strings.end(),
    std::string{}, [](const std::string& a, const std::string& b) {
        return a.empty() ? b : a + ", " + b;
    });
```

## C++20 Ranges & Views

Prefer the range overloads — they take the container directly and compose:

```cpp
std::ranges::sort(v);
std::ranges::find(v, value);
auto it = std::ranges::find_if(v, [](int x) { return x > 5; });
bool all = std::ranges::all_of(v, pred);

// Lazy, composable pipelines (no intermediate containers)
auto evens_doubled = v
    | std::views::filter([](int x) { return x % 2 == 0; })
    | std::views::transform([](int x) { return x * 2; });
for (int x : evens_doubled) { /* ... */ }

// Other useful views: take, drop, reverse, iota, join, split, keys, values
for (int i : std::views::iota(0, 10)) { /* 0..9 */ }
```

Projections let an algorithm key off a member without a lambda:

```cpp
std::ranges::sort(people, std::less<>{}, &Person::age);  // sort by .age
auto it = std::ranges::find(people, "Ada", &Person::name);
```

Use `std::erase` / `std::erase_if` instead of the erase-remove idiom:

```cpp
std::erase_if(v, [](int x) { return x % 2 == 0; });  // replaces remove_if + erase
std::erase(v, value);
```
