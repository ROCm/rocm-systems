# Transform

Catalog id: `transform`

A new container is built from each element of another.

## Implement this

```cpp
std::vector<std::string> names;
names.reserve(entries.size());
std::ranges::transform(entries, std::back_inserter(names), &entry::name);
```

Same result as a view copied into a container:

```cpp
auto ts = samples | std::views::transform(&sample::ts);
std::vector<std::uint64_t> stamps(ts.begin(), ts.end());
```

## Switch only if the result feeds one more loop

Iterate the view directly and skip the container:

```cpp
for (std::uint64_t ts : samples | std::views::transform(&sample::ts)) { emit(ts); }
```

## Pitfalls

- `std::ranges::to` is C++23. Copy a view with the iterator-pair constructor, or `std::ranges::copy` into a `back_inserter`.
- The iterator-pair constructor needs a common range. After `take_while` or other non-common views, pipe through `std::views::common` first.
- A view does not own its data. Never return a view over a local or a temporary container.
- A view is lazy. Modifying the source between creating the view and iterating it changes the result.
