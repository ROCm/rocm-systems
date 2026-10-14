# Memento

Catalog id: `memento`

Object state is saved and restored. Copy the value when the state is already a value.

## Implement this

```cpp
state saved = current;
current = saved;
```

## Pattern only if callers must not see the private representation

```cpp
class memento {
    state state_;
    friend class originator;
    explicit memento(state value) : state_{std::move(value)} {}
};

class originator {
public:
    memento save() const { return memento{state_}; }
    void restore(const memento& saved) { state_ = saved.state_; }

private:
    state state_;
};
```
