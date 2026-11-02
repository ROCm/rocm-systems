# Mediator

Catalog id: `mediator`

Objects call one another in a web. Pass the one collaborator a function uses.

## Implement this

```cpp
void
on_click(panel& panel)
{
    panel.refresh();
}
```

## Pattern only if one coordinator replaces those cross-calls

```cpp
class mediator_interface
{
public:
    virtual ~mediator_interface()                                       = default;
    virtual void notify(component& sender, std::string_view event_name) = 0;
};
```

Components call `notify` instead of holding pointers to each other.
