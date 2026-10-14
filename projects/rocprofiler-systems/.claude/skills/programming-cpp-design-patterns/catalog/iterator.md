# Iterator

Catalog id: `iterator`

A collection is walked. Use the standard library.

## Implement this

```cpp
for(const element& item : collection)
{
    use(item);
}
```

A custom container provides `begin()` and `end()` returning normal iterators, then the same loop works.

## Pattern only if a custom container cannot provide standard iterators

```cpp
class cursor
{
public:
    [[nodiscard]] bool has_next() const { return m_index < m_items.size(); }

    [[nodiscard]] const element& next() { return m_items[m_index++]; }

private:
    const std::vector<element>& m_items;
    std::size_t                 m_index = 0;
};
```

Prefer `begin()` and `end()` over `has_next()` when the container can support them.
