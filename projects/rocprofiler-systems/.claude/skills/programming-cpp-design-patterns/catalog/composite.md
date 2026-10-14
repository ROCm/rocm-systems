# Composite

Catalog id: `composite`

A tree holds leaves and groups. Store the children in the node unless callers must share one interface.

## Implement this

```cpp
struct node {
    void operation() const {
        for (const node& child : children) child.operation();
    }

    std::vector<node> children;
};
```

## Pattern only if callers must use one interface for a leaf and a group

```cpp
class component {
public:
    virtual ~component() = default;
    virtual void operation() = 0;
    virtual void add(std::shared_ptr<component> child) { (void)child; }
};

class composite final : public component {
public:
    void add(std::shared_ptr<component> child) override {
        children_.push_back(std::move(child));
    }

    void operation() override {
        for (const auto& child : children_) child->operation();
    }

private:
    std::vector<std::shared_ptr<component>> children_;
};
```
