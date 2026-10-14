# Type Erasure

Catalog id: `type-erasure`

Unrelated types share an operation and have no common base. Do not add a base to types you cannot change.

## Implement this

Closed set. `std::variant` rejects every other type.

```cpp
using item = std::variant<sensor, image>;

void
postprocess_all(std::vector<item>& items)
{
    for(item& value : items)
    {
        std::visit([](auto& object) { object.postprocess(); }, value);
    }
}
```

One callback, not a whole object:

```cpp
std::function<void()> postprocess = [&sensor_value] { sensor_value.postprocess(); };
```

## Pattern only if the set is open and the container owns the value

Match this shape. Adapt the stored type and the method name; keep the private `concept_interface`, the model, and `std::unique_ptr` ownership.

```cpp
class postprocessable
{
public:
    template <typename T>
    explicit postprocessable(T object)
    : m_storage{ std::make_unique<model<T>>(std::move(object)) }
    {}

    void postprocess() { m_storage->postprocess(); }

private:
    struct concept_interface
    {
        virtual ~concept_interface() = default;
        virtual void postprocess()   = 0;
    };

    template <typename T>
    struct model final : concept_interface
    {
        explicit model(T value)
        : object{ std::move(value) }
        {}
        void postprocess() override { object.postprocess(); }
        T    object;
    };

    std::unique_ptr<concept_interface> m_storage;
};

std::vector<postprocessable> items;
items.emplace_back(sensor{});
items.emplace_back(image{});
```

## Pattern only if the set is open and the caller keeps the objects

This view does not own the object. The caller keeps it alive.

```cpp
class postprocessable_view
{
public:
    template <typename T>
    explicit postprocessable_view(T& object)
    : m_object{ &object }
    , m_postprocess{ [](void* ptr) { static_cast<T*>(ptr)->postprocess(); } }
    {}

    void postprocess() { m_postprocess(m_object); }

private:
    void*                      m_object;
    std::function<void(void*)> m_postprocess;
};
```

`std::function` has allocation and indirect-call cost. On a hot path, store a function pointer with the same signature instead.
