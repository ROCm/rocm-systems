# Prototype

Catalog id: `prototype`

A copy is needed. Name the concrete type when the copy site knows it.

## Implement this

```cpp
widget copy = original;
```

## Pattern only if the copy site does not know the dynamic type

```cpp
class prototype_interface
{
public:
    virtual ~prototype_interface()                                           = default;
    [[nodiscard]] virtual std::unique_ptr<prototype_interface> clone() const = 0;
};

class widget final : public prototype_interface
{
public:
    [[nodiscard]] std::unique_ptr<prototype_interface> clone() const override
    {
        return std::make_unique<widget>(*this);
    }
};
```
