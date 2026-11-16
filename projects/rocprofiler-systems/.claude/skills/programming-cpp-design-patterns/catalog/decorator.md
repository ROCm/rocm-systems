# Decorator

Catalog id: `decorator`

Behavior is added around an existing call. A function is enough until wrappers are stacked at runtime.

## Implement this

```cpp
void
operation(component_interface& object)
{
    object.operation();
    extra();
}
```

## Pattern only if combinations are stacked at runtime

```cpp
class decorator : public component_interface
{
public:
    explicit decorator(std::unique_ptr<component_interface> wrapped)
    : m_wrapped{ std::move(wrapped) }
    {}

    void operation() override
    {
        m_wrapped->operation();
        extra();
    }

private:
    std::unique_ptr<component_interface> m_wrapped;
};
```
