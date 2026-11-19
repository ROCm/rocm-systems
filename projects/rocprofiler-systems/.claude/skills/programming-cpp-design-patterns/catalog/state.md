# State

Catalog id: `state`

Behavior depends on mode. An enum is the mode unless each state is its own type.

## Implement this

```cpp
enum class mode
{
    idle,
    running
};

void
request(mode& current)
{
    switch(current)
    {
        case mode::idle: current = mode::running; break;
        case mode::running: break;
        default:
            throw std::logic_error(
                fmt::format("mode {} not handled", static_cast<int>(current)));
    }
}
```

## Pattern only if each state is a type and transitions are numerous

```cpp
class state_interface
{
public:
    virtual ~state_interface()        = default;
    virtual void handle(context& ctx) = 0;
};

class context
{
public:
    void set_state(std::unique_ptr<state_interface> next) { m_state = std::move(next); }
    void request() { m_state->handle(*this); }

private:
    std::unique_ptr<state_interface> m_state;
};
```
