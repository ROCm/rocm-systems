# Visitor

Catalog id: `visitor`

An operation runs over a set of types. A closed set uses `std::visit`.

## Implement this

```cpp
using element = std::variant<element_a, element_b>;

void apply(element& value) {
    std::visit([](auto& item) { item.summarize(); }, value);
}
```

## Pattern only if the element hierarchy is already open and new operations arrive more often than new elements

```cpp
class visitor {
public:
    virtual ~visitor() = default;
    virtual void visit(element_a& element) = 0;
    virtual void visit(element_b& element) = 0;
};

class element {
public:
    virtual ~element() = default;
    virtual void accept(visitor& operation) = 0;
};

class element_a final : public element {
public:
    void accept(visitor& operation) override { operation.visit(*this); }
};
```
