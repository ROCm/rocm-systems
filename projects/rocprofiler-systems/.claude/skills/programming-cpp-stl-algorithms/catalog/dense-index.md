# Dense index

Catalog id: `dense-index`

Keys are dense integers 0..N-1: thread ids, device ids, enum values. The index is the key; no hashing, iteration in id order.

## Implement this

```cpp
class thread_event_store
{
public:
    explicit thread_event_store(std::size_t thread_count)
    : m_per_thread(thread_count)
    {}

    void add(std::size_t tid, event item) { m_per_thread.at(tid).push_back(item); }

    template <typename Func>
    void for_each(Func&& callback) const
    {
        for(const auto& events : m_per_thread)
        {
            for(const event& item : events)
            {
                callback(item);
            }
        }
    }

private:
    std::vector<std::vector<event>> m_per_thread;
};
```

## Switch only if ids are sparse or unbounded

Use `std::unordered_map<index, Value>` (see `lookup-table`).

## Pitfalls

- Size the outer vector once, before threads start. Resizing it while threads append to inner vectors is a data race and invalidates their references.
- Each thread may append only to its own inner vector without a lock.
- A `std::map<int, T>` keyed by a dense id costs a node allocation per key and a tree walk per access.
- Bounds: `.at(tid)` at the trust boundary, `[tid]` on a path already validated.
