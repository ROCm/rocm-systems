# State

Catalog id: `state`

Behavior depends on mode. An enum is the mode unless each state is its own type.

## Implement this

```cpp
enum class mode { idle, running };

void request(mode& current) {
    switch (current) {
    case mode::idle:
        current = mode::running;
        break;
    case mode::running:
        break;
    }
}
```

## Pattern only if each state is a type and transitions are numerous

```cpp
class state {
public:
    virtual ~state() = default;
    virtual void handle(context& ctx) = 0;
};

class context {
public:
    void set_state(std::unique_ptr<state> next) { state_ = std::move(next); }
    void request() { state_->handle(*this); }

private:
    std::unique_ptr<state> state_;
};
```
