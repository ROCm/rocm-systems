# Type Erasure

Catalog id: `type-erasure`

Unrelated types share an operation and have no common base. Do not add a base to types you cannot change.

## Implement this

Closed set. `std::variant` rejects every other type.

```cpp
using item = std::variant<sensor, image>;

void postprocess_all(std::vector<item>& items) {
    for (item& value : items) {
        std::visit([](auto& object) { object.postprocess(); }, value);
    }
}
```

One callback, not a whole object:

```cpp
std::function<void()> postprocess = [&sensor_value] { sensor_value.postprocess(); };
```

## Pattern only if the set is open and the container owns the value

Match this shape. Adapt the stored type and the method name; keep the private concept, the model, and `std::unique_ptr` ownership.

```cpp
class postprocessable {
public:
    template <typename T>
    explicit postprocessable(T object)
        : storage_{std::make_unique<model<T>>(std::move(object))} {}

    void postprocess() { storage_->postprocess(); }

private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual void postprocess() = 0;
    };

    template <typename T>
    struct model final : concept_t {
        explicit model(T object) : object_(std::move(object)) {}
        void postprocess() override { object_.postprocess(); }
        T object_;
    };

    std::unique_ptr<concept_t> storage_;
};

std::vector<postprocessable> items;
items.emplace_back(sensor{});
items.emplace_back(image{});
```

## Pattern only if the set is open and the caller keeps the objects

This view does not own the object. The caller keeps it alive.

```cpp
class postprocessable_view {
public:
    template <typename T>
    explicit postprocessable_view(T& object)
        : object_{&object}
        , postprocess_impl_{[](void* ptr) { static_cast<T*>(ptr)->postprocess(); }} {}

    void postprocess() { postprocess_impl_(object_); }

private:
    void* object_;
    std::function<void(void*)> postprocess_impl_;
};
```

`std::function` has allocation and indirect-call cost. On a hot path, store a function pointer with the same signature instead.
