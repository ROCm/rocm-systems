---
name: programming-cpp-stl-algorithms
description: C++ STL algorithms and data structures - suggests appropriate containers and algorithms when they solve the problem at hand
---

# C++ STL Algorithms & Data Structures Skill

Use this skill when implementing C++ code that could benefit from STL containers or algorithms.

<IMPORTANT>
When you recognize a problem that an STL algorithm or container solves, **suggest it to the user** instead of writing manual loops or custom implementations.

**Prefer STL algorithms over raw loops** - they are:

- More readable (express intent)
- Less error-prone
- Often more efficient
- Easier to parallelize

This project is **C++20**. Prefer `std::ranges` overloads and `std::views` over iterator-pair algorithms when the input is a whole container. Prefer `std::erase` / `std::erase_if` over the erase-remove idiom. `std::ranges::to` and `std::ranges::fold_left` are C++23 — do not use them.
</IMPORTANT>

Reference: [CppReference - Algorithms](https://en.cppreference.com/w/cpp/algorithm)

## When to Suggest

### Instead of Manual Loops

| When you see... | Suggest... |
| ----------------- | ------------ |
| Loop to find an element | `std::ranges::find`, `std::ranges::find_if` |
| Loop to check if any/all elements match | `std::ranges::any_of`, `std::ranges::all_of`, `std::ranges::none_of` |
| Loop to count elements | `std::ranges::count`, `std::ranges::count_if` |
| Loop to copy elements | `std::ranges::copy`, `std::ranges::copy_if` |
| Loop to transform elements | `std::ranges::transform` |
| Loop to accumulate/sum values | `std::accumulate`, `std::reduce` (`std::ranges::fold_left` is C++23) |
| Loop to remove elements | `std::erase`, `std::erase_if` |
| Loop to sort | `std::ranges::sort`, `std::ranges::stable_sort`, `std::ranges::partial_sort` |
| Loop to find min/max | `std::ranges::min_element`, `std::ranges::max_element`, `std::ranges::minmax_element` |
| Loop to reverse | `std::ranges::reverse` |
| Loop to fill with values | `std::ranges::fill`, `std::ranges::generate` |
| Loop to check if sorted | `std::ranges::is_sorted` |
| Nested loops for set operations | `std::ranges::set_union`, `std::ranges::set_intersection`, `std::ranges::set_difference` |

## Data Structures - When to Use What

### Sequence Containers

| Container | Use When... | Complexity |
| ----------- | ------------- | ------------ |
| `std::vector` | Default choice, random access, add at end | O(1) access, O(1) amortized push_back |
| `std::array` | Fixed size known at compile time | O(1) access, no heap allocation |
| `std::deque` | Need to add/remove at both ends | O(1) access, O(1) push_front/back |
| `std::list` | Frequent insert/remove in middle, no random access needed | O(1) insert/remove, O(n) access |
| `std::forward_list` | Singly-linked list, minimal memory | O(1) insert after, forward iteration only |

### Associative Containers

| Container | Use When... | Complexity |
| ----------- | ------------- | ------------ |
| `std::set` | Unique sorted elements, fast lookup | O(log n) insert/find |
| `std::map` | Key-value pairs, sorted by key | O(log n) insert/find |
| `std::multiset` | Sorted elements with duplicates | O(log n) insert/find |
| `std::multimap` | Key-value with duplicate keys | O(log n) insert/find |

### Unordered Containers (Hash-based)

| Container | Use When... | Complexity |
| ----------- | ------------- | ------------ |
| `std::unordered_set` | Unique elements, fastest lookup, order doesn't matter | O(1) avg insert/find |
| `std::unordered_map` | Key-value, fastest lookup, order doesn't matter | O(1) avg insert/find |

### Container Adaptors

| Container | Use When... |
| ----------- | ------------- |
| `std::stack` | LIFO (Last In First Out) |
| `std::queue` | FIFO (First In First Out) |
| `std::priority_queue` | Always access largest/smallest element |

## Algorithm Categories

### Non-modifying Sequence Operations

```cpp
// Finding elements
auto it = std::find(v.begin(), v.end(), value);
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 5; });

// Checking conditions
bool any = std::any_of(v.begin(), v.end(), pred);
bool all = std::all_of(v.begin(), v.end(), pred);
bool none = std::none_of(v.begin(), v.end(), pred);

// Counting
int count = std::count(v.begin(), v.end(), value);
int count = std::count_if(v.begin(), v.end(), pred);

// Searching
auto it = std::search(v.begin(), v.end(), sub.begin(), sub.end());
auto it = std::adjacent_find(v.begin(), v.end());  // Find consecutive duplicates
```

### Modifying Sequence Operations

```cpp
// Copying
std::copy(src.begin(), src.end(), dest.begin());
std::copy_if(src.begin(), src.end(), std::back_inserter(dest), pred);

// Transforming
std::transform(v.begin(), v.end(), v.begin(), [](int x) { return x * 2; });
std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>{});

// Filling
std::fill(v.begin(), v.end(), value);
std::generate(v.begin(), v.end(), []() { return rand(); });
std::iota(v.begin(), v.end(), 0);  // Fill with 0, 1, 2, 3...

// Removing (erase-remove idiom)
v.erase(std::remove(v.begin(), v.end(), value), v.end());
v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());

// Replacing
std::replace(v.begin(), v.end(), old_val, new_val);
std::replace_if(v.begin(), v.end(), pred, new_val);

// Reversing
std::reverse(v.begin(), v.end());

// Rotating
std::rotate(v.begin(), v.begin() + n, v.end());  // Move first n elements to end

// Unique (remove consecutive duplicates)
v.erase(std::unique(v.begin(), v.end()), v.end());
```

### Sorting and Related

```cpp
// Sorting
std::sort(v.begin(), v.end());
std::sort(v.begin(), v.end(), std::greater<>{});  // Descending
std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.name < b.name; });

std::stable_sort(v.begin(), v.end());  // Preserves relative order of equal elements
std::partial_sort(v.begin(), v.begin() + n, v.end());  // Only sort first n

// Nth element (partition around nth)
std::nth_element(v.begin(), v.begin() + n, v.end());  // nth element in sorted position

// Checking sorted
bool sorted = std::is_sorted(v.begin(), v.end());

// Binary search (requires sorted range)
bool found = std::binary_search(v.begin(), v.end(), value);
auto it = std::lower_bound(v.begin(), v.end(), value);  // First >= value
auto it = std::upper_bound(v.begin(), v.end(), value);  // First > value
auto [lo, hi] = std::equal_range(v.begin(), v.end(), value);  // Range of equal elements
```

### Min/Max Operations

```cpp
// Single values
int m = std::min(a, b);
int m = std::max({a, b, c, d});  // Initializer list
auto [lo, hi] = std::minmax(a, b);

// In containers
auto it = std::min_element(v.begin(), v.end());
auto it = std::max_element(v.begin(), v.end());
auto [min_it, max_it] = std::minmax_element(v.begin(), v.end());

// Clamping
int clamped = std::clamp(value, low, high);
```

### Numeric Operations (`<numeric>`)

```cpp
// Sum/accumulate
int sum = std::accumulate(v.begin(), v.end(), 0);
int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<>{});

// Reduce (parallelizable version of accumulate, C++20)
int sum = std::reduce(v.begin(), v.end());

// Inner product (dot product)
int dot = std::inner_product(a.begin(), a.end(), b.begin(), 0);

// Partial sums (prefix sum)
std::partial_sum(v.begin(), v.end(), result.begin());

// Adjacent difference
std::adjacent_difference(v.begin(), v.end(), result.begin());

// Transform reduce (C++20)
auto result = std::transform_reduce(v.begin(), v.end(), init, binary_op, unary_op);
```

### Set Operations (on sorted ranges)

```cpp
std::vector<int> result;

// Union
std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

// Intersection
std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

// Difference (in a but not in b)
std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

// Symmetric difference (in a or b but not both)
std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

// Check if includes
bool includes = std::includes(a.begin(), a.end(), b.begin(), b.end());

// Merge sorted ranges
std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
```

### Heap Operations

```cpp
// Create heap (max-heap by default)
std::make_heap(v.begin(), v.end());

// Add element to heap
v.push_back(value);
std::push_heap(v.begin(), v.end());

// Remove largest from heap
std::pop_heap(v.begin(), v.end());
v.pop_back();

// Sort heap
std::sort_heap(v.begin(), v.end());

// Check if heap
bool is_heap = std::is_heap(v.begin(), v.end());
```

### Permutations

```cpp
// Next/previous permutation
std::next_permutation(v.begin(), v.end());
std::prev_permutation(v.begin(), v.end());

// Shuffle
std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});
```

## How to Suggest

When you detect an algorithm opportunity, present it like this:

> **STL Algorithm Suggestion**
>
> Instead of this manual loop:
>
> ```cpp
> for (auto it = v.begin(); it != v.end(); ++it) {
>     if (*it == target) return it;
> }
> ```
>
> Consider using `std::ranges::find`:
>
> ```cpp
> auto it = std::ranges::find(v, target);
> ```
>
> **Benefits:** More readable, expresses intent, potentially optimized.

## Common Patterns

### Erase

```cpp
// C++20
std::erase_if(v, [](int x) { return x % 2 == 0; });
std::erase(v, value);
```

### Transform + Back Inserter

```cpp
std::vector<std::string> names;
std::ranges::transform(people, std::back_inserter(names),
    [](const Person& p) { return p.name; });
```

### Copy If + Back Inserter

```cpp
std::vector<int> evens;
std::ranges::copy_if(v, std::back_inserter(evens),
    [](int x) { return x % 2 == 0; });
```

### Accumulate with Custom Operation

```cpp
// Concatenate strings
std::string result = std::accumulate(strings.begin(), strings.end(),
    std::string{}, [](const std::string& a, const std::string& b) {
        return a.empty() ? b : a + ", " + b;
    });
```

## C++20 Ranges

<IMPORTANT>
Use `std::ranges` and `std::views`. Do not use `std::ranges::to` (C++23). Copy a view into a container with the iterator-pair constructor or `std::ranges::copy`.
</IMPORTANT>

```cpp
std::ranges::sort(v);
auto it = std::ranges::find_if(v, [](int x) { return x % 2 == 0; });

auto even_view = v | std::views::filter([](int x) { return x % 2 == 0; });
std::vector<int> evens(even_view.begin(), even_view.end());

// C++23 — do not use
// auto evens = v | std::views::filter(...) | std::ranges::to<std::vector>();
```

## References

- [CppReference - Algorithms Library](https://en.cppreference.com/w/cpp/algorithm)
- [CppReference - Containers Library](https://en.cppreference.com/w/cpp/container)
- [CppReference - Numeric Library](https://en.cppreference.com/w/cpp/numeric)
- [C++ Core Guidelines - STL](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-stdlib)
