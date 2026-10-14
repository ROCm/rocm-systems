# Iterator

Catalog id: `iterator`

A collection is walked. Use the standard library.

## Implement this

```cpp
for (const element& item : collection) {
    use(item);
}
```

A custom container provides `begin()` and `end()` returning normal iterators, then the same loop works.

## Pattern only if a custom container cannot provide standard iterators

```cpp
class cursor {
public:
    bool has_next() const { return index_ < items_.size(); }

    const element& next() { return items_[index_++]; }

private:
    const std::vector<element>& items_;
    std::size_t index_ = 0;
};
```

Prefer `begin()` and `end()` over `has_next()` when the container can support them.
