# Sequence container

Catalog id: `sequence-container`

A container type is chosen for a sequence. Contiguous memory wins on almost every access pattern.

## Implement this

```cpp
std::vector<event> events;
events.reserve(expected_count);

std::array<std::uint64_t, 4> counters{};  // size known at compile time, no heap

void
write_events(std::span<const event> events);  // parameter: accepts vector, array, C array
```

## Switch only if

| Condition | Container |
| --- | --- |
| Push and pop at both ends | `std::deque` |
| Splice between lists, or iterators must survive inserts in the middle | `std::list` |
| FIFO or LIFO with no other access | `std::queue` / `std::stack` (backed by `std::deque`) |

## Pitfalls

- `std::vector<bool>` is a packed bitset. It has no `bool&`, and two threads writing different elements race. Use `std::vector<std::uint8_t>` or `std::bitset<N>`.
- A `std::span` parameter does not own its data. Do not store it past the call.
- `reserve` before a known number of `push_back` calls. `resize` creates elements; `reserve` does not.
- `std::list` for "frequent middle inserts" is usually slower than a vector because of allocations and cache misses. Measure before switching.
