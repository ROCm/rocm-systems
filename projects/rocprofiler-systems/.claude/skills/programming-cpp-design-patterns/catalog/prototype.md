# Prototype

Catalog id: `prototype`

A copy is needed. Name the concrete type when the copy site knows it.

## Implement this

```cpp
widget copy = original;
```

## Pattern only if the copy site does not know the dynamic type

```cpp
class prototype {
public:
    virtual ~prototype() = default;
    virtual std::unique_ptr<prototype> clone() const = 0;
};

class widget final : public prototype {
public:
    std::unique_ptr<prototype> clone() const override {
        return std::make_unique<widget>(*this);
    }
};
```
