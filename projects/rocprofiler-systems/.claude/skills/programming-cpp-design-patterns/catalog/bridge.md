# Bridge

Catalog id: `bridge`

Two parts of a design seem to vary. Vary the one axis that actually changes.

## Implement this

```cpp
template <typename Implementation>
class abstraction
{
public:
    void operation() { m_impl.operation_impl(); }

private:
    Implementation m_impl{};
};
```

## Pattern only if both axes vary independently at runtime

```cpp
class abstraction
{
public:
    virtual ~abstraction() = default;
    explicit abstraction(std::unique_ptr<implementation> impl)
    : m_impl{ std::move(impl) }
    {}

    virtual void operation() { m_impl->operation_impl(); }

protected:
    std::unique_ptr<implementation> m_impl;
};
```
