# Modify while iterating

Catalog id: `modify-while-iterating`

A container grows or shrinks while it is iterated. Know what each container invalidates before you write the loop.

## Implement this

Append, and process the new elements too: an index loop re-reads `size()` and never holds a reference.

```cpp
for (std::size_t i = 0; i < work.size(); ++i) {
    if (work[i] < 0) { work.push_back(-work[i]); }
}
```

Remove: `std::erase_if` (see `erase`).

## Switch only if the loop must stay a range-for

```cpp
std::vector<int> extra;
for (int w : work) {
    if (w < 0) { extra.push_back(-w); }
}
work.insert(work.end(), extra.begin(), extra.end());
```

## Pitfalls

| Container | Insert invalidates | Erase invalidates |
| --- | --- | --- |
| `std::vector`, `std::string` | everything if it reallocates, else at and after the insert point | at and after the erase point |
| `std::deque` | all iterators; references stay valid only at either end | everything unless at either end |
| `std::unordered_map` / `set` | iterators on rehash; references stay valid | only the erased element |
| `std::map`, `std::set`, `std::list` | nothing | only the erased element |

- `auto& first = v.front(); v.push_back(x);` leaves `first` dangling. Re-read after the insert, or `reserve` first.
- A range-for over a `std::vector` that calls `push_back` on it is undefined behavior.
- Iterating a `std::unordered_map` while another thread inserts is a data race, even when the keys differ.
