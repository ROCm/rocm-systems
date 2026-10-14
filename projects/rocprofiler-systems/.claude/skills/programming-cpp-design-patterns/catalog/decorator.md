# Decorator

Catalog id: `decorator`

Behavior is added around an existing call. A function is enough until wrappers are stacked at runtime.

## Implement this

```cpp
void operation(component& object) {
    object.operation();
    extra();
}
```

## Pattern only if combinations are stacked at runtime

```cpp
class decorator : public component {
public:
    explicit decorator(std::unique_ptr<component> wrapped) : wrapped_{std::move(wrapped)} {}

    void operation() override {
        wrapped_->operation();
        extra();
    }

private:
    std::unique_ptr<component> wrapped_;
};
```
