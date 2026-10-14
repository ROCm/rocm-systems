# Builder

Catalog id: `builder`

Construction has many fields or steps. Independent fields do not need a builder.

## Implement this

An aggregate with defaults. Designated initializers follow member order.

```cpp
struct trace_config {
    std::size_t buffer_size = 1'048'576;
    bool compression = false;
    std::optional<std::string> label;
};

trace_config config{.buffer_size = 4096, .compression = true, .label = "startup"};
```

## Pattern only if order or cross-field checks must fail before the value exists

`build()` is the only way to obtain the value, and it rejects an invalid combination.

```cpp
class car_builder {
public:
    car_builder& set_engine(engine value) {
        engine_ = std::move(value);
        return *this;
    }
    car_builder& set_gps(bool enabled) {
        gps_ = enabled;
        return *this;
    }
    car build() const {
        if (gps_ && !engine_) throw std::invalid_argument{"gps requires an engine"};
        return car{engine_, gps_};
    }

private:
    std::optional<engine> engine_;
    bool gps_ = false;
};

auto made = car_builder{}.set_engine(v8).set_gps(true).build();
```
