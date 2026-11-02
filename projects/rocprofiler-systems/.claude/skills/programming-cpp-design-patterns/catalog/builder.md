# Builder

Catalog id: `builder`

Construction has many fields or steps. Independent fields do not need a builder.

## Implement this

An aggregate with defaults. Designated initializers follow member order.

```cpp
constexpr std::size_t k_default_buffer_size = 1'048'576;
constexpr std::size_t k_startup_buffer_size = 4096;

struct trace_config
{
    std::size_t                buffer_size = k_default_buffer_size;
    bool                       compression = false;
    std::optional<std::string> label;
};

trace_config config{ .buffer_size = k_startup_buffer_size,
                     .compression = true,
                     .label       = "startup" };
```

## Pattern only if order or cross-field checks must fail before the value exists

`build()` is the only way to obtain the value, and it rejects an invalid combination.

```cpp
class car_builder
{
public:
    [[nodiscard]] car_builder& set_engine(engine value)
    {
        m_engine = std::move(value);
        return *this;
    }
    [[nodiscard]] car_builder& set_gps(bool enabled)
    {
        m_gps = enabled;
        return *this;
    }
    [[nodiscard]] car build() const
    {
        if(m_gps && !m_engine) throw std::invalid_argument{ "gps requires an engine" };
        return car{ m_engine, m_gps };
    }

private:
    std::optional<engine> m_engine;
    bool                  m_gps = false;
};

const auto made = car_builder{}.set_engine(block).set_gps(true).build();
```
