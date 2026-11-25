# Command

Catalog id: `command`

Work is stored to run later, or it must be undone. Store the call and the data that reverses it.

## Implement this

```cpp
struct action
{
    std::function<void()> execute;
    std::function<void()> undo;
};

// The caller keeps target alive while the returned action is stored.
[[nodiscard]] action
make_assign(int& target, int next)
{
    const int previous = target;
    return {
        [&target, next] { target     = next; },
        [&target, previous] { target = previous; },
    };
}
```

## Pattern only if the operation is logged, queued, or undone as its own object

```cpp
class command_interface
{
public:
    virtual ~command_interface() = default;
    virtual void execute()       = 0;
    virtual void undo()          = 0;
};
```

Prefer the `action` struct unless the command type itself must be named in an interface.
