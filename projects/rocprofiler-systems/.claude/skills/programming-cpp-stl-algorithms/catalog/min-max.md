# Min-max

Catalog id: `min-max`

The extreme element of a range, or a value bounded to a range.

## Implement this

```cpp
const auto latest = std::ranges::max_element(samples, {}, &sample::ts);
if(latest != samples.end())
{
    use(*latest);
}

const int workers = std::clamp(requested, 1, max_workers);
```

## Switch only if both ends are needed

```cpp
const auto [low, high] =
    std::ranges::minmax_element(samples, {}, &sample::ts);  // one pass
```

## Pitfalls

- `max_element` on an empty range returns `end()`. Check before dereferencing.
- `std::ranges::max(v)` on an empty range is undefined behavior.
- `const auto& extreme = std::max(left(), right());` dangles: `std::max` returns a reference to a temporary. Bind by value.
- `std::max(left, right)` with mixed types does not compile. Convert explicitly; `std::max<std::uint64_t>(left, right)` silently converts a negative `int`.
- `std::clamp(value, low, high)` requires `low <= high`.
