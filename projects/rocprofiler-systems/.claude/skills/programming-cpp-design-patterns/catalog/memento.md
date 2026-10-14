# Memento

Catalog id: `memento`

Object state is saved and restored. Copy the value when the state is already a value.

## Implement this

```cpp
state saved = current;
current     = saved;
```

## Pattern only if callers must not see the private representation

```cpp
class memento
{
    state m_state;
    friend class originator;
    explicit memento(state value)
    : m_state{ std::move(value) }
    {}
};

class originator
{
public:
    [[nodiscard]] memento save() const { return memento{ m_state }; }
    void                  restore(const memento& saved) { m_state = saved.m_state; }

private:
    state m_state;
};
```
