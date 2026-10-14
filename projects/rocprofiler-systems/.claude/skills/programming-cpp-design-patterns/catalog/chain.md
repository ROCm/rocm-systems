# Chain of Responsibility

Catalog id: `chain`

Several handlers may process one request. Call them in order. A handler that finishes the request returns `true`.

## Implement this

```cpp
using handler = std::function<bool(request&)>;

void dispatch(const std::vector<handler>& handlers, request& req) {
    for (const handler& handle : handlers) {
        if (handle(req)) return;
    }
}
```

## Pattern only if handlers are inserted at runtime and each one stores the next

```cpp
class handler {
public:
    virtual ~handler() = default;

    void set_next(std::shared_ptr<handler> next) { next_ = std::move(next); }

    virtual void handle(request& req) {
        if (next_) next_->handle(req);
    }

private:
    std::shared_ptr<handler> next_;
};
```
