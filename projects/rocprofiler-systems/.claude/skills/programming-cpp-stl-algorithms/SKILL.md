---
name: programming-cpp-stl-algorithms
description: Use when writing or changing C++ that loops over a container, builds one container from another, finds, counts, sorts, filters, erases, deduplicates, sums, or takes the top-k of elements, mutates a container while iterating it, or chooses a container for lookup, per-id storage, or a priority queue
---

# C++ STL Algorithms and Containers

Match the operation in [catalog.md](catalog.md). Read only that row's example file and write its **Implement this** code.

**Core principle:** Name the operation, then write the algorithm that has that name. The loop body is the only code you write.

## Before writing

When the change adds a loop over a container, or a new container member:

1. Search `catalog.md` by the operation or container: `rg -n -i "erase|top|lookup" catalog.md`
2. Write `catalog: <id> | <choice>`.
3. Read that row's example file and apply its **Pitfalls** before the code compiles.
4. If the row's **switch only if** condition is true, write that choice instead.

## Principles

- Whole container: `std::ranges::` with a projection (`&entry::name`), not an iterator pair plus a lambda.
- A range-for that only performs a side effect per element is fine. Do not wrap it in `std::for_each`.
- `std::vector` is the default container. Another container needs a row's **switch only if** condition.
- Before you insert or erase while iterating, check the row `modify-while-iterating`.
- C++20 only. These are C++23: `std::ranges::to`, `std::ranges::fold_left`, `std::ranges::contains`, `std::views::zip`, `std::views::enumerate`, `std::views::join_with`, `std::string::contains`.

## Red flags

- A raw loop whose body is `if (...) out.push_back(...)`, `sum +=`, or `return true`
- `v.erase(std::remove_if(...), v.end())` or an `erase` call inside a loop
- `std::sort` of everything to read the first k elements
- `operator[]` on a map to read a value
- `const_cast` on a container element
- A `catalog:` line that does not match the code

## Example

Request: "Return the names of entries newer than the cutoff."

`catalog: filter | std::views::filter + transform into a vector`

```cpp
auto recent = entries | std::views::filter([cutoff](const entry& e) { return e.timestamp >= cutoff; })
                      | std::views::transform(&entry::name);
std::vector<std::string> names(recent.begin(), recent.end());
```
