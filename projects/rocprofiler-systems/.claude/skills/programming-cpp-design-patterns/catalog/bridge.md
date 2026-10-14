# Bridge

Catalog id: `bridge`

Two parts of a design seem to vary. Vary the one axis that actually changes.

## Implement this

```cpp
template <typename Implementation>
class abstraction {
public:
    void operation() { impl_.operation_impl(); }

private:
    Implementation impl_{};
};
```

## Pattern only if both axes vary independently at runtime

```cpp
class abstraction {
public:
    virtual ~abstraction() = default;
    explicit abstraction(std::unique_ptr<implementation> impl) : impl_{std::move(impl)} {}

    virtual void operation() { impl_->operation_impl(); }

protected:
    std::unique_ptr<implementation> impl_;
};
```
