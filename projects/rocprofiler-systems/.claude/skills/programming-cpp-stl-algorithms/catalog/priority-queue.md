# Priority queue

Catalog id: `priority-queue`

The smallest or largest element is always taken next.

## Implement this

```cpp
struct later_deadline {
    bool operator()(const task& a, const task& b) const { return a.deadline > b.deadline; }
};

using task_queue = std::priority_queue<task, std::vector<task>, later_deadline>;  // top() = earliest deadline

std::optional<task> pop_next(task_queue& queue) {
    if (queue.empty()) { return std::nullopt; }
    task next = queue.top();
    queue.pop();
    return next;
}
```

## Switch only if priorities change after insert, or any element can be removed

```cpp
std::set<std::pair<deadline_t, task_id>> queue;  // begin() is next; erase + insert to reprioritize
auto next = queue.extract(queue.begin()).value();
```

## Pitfalls

- The comparator is inverted: `std::priority_queue` keeps the element that compares *greatest* on top. Smallest-first needs `>`. Name the comparator after what ends on top.
- `top()` returns a `const&`. Copy it, then `pop()`. `std::move(const_cast<task&>(queue.top()))` mutates an element the heap still orders, and only works if `pop()` happens not to compare it.
- `top()` and `pop()` on an empty queue are undefined behavior.
- `std::priority_queue` cannot find or update an element. Do not work around it with a parallel lookup table; use the `std::set` form.
