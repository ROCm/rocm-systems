# Factory Method

Catalog id: `factory-method`

A type tag selects which object to create. Put that choice in one function.

## Implement this

Closed set of products. The caller visits the result; there is no product base.

```cpp
enum class exporter_kind
{
    perfetto,
    csv,
    rocpd
};

using exporter = std::variant<perfetto_exporter, csv_exporter, rocpd_exporter>;

[[nodiscard]] exporter
make_exporter(exporter_kind kind)
{
    switch(kind)
    {
        case exporter_kind::perfetto: return perfetto_exporter{};
        case exporter_kind::csv: return csv_exporter{};
        case exporter_kind::rocpd: return rocpd_exporter{};
        default: throw std::invalid_argument{ "unknown exporter" };
    }
}
```

Open product set, still one creation function:

```cpp
[[nodiscard]] std::unique_ptr<exporter>
make_exporter(std::string_view kind)
{
    if(kind == "perfetto") return std::make_unique<perfetto_exporter>();
    if(kind == "csv") return std::make_unique<csv_exporter>();
    if(kind == "rocpd") return std::make_unique<rocpd_exporter>();
    throw std::invalid_argument{ "unknown exporter" };
}
```

## Pattern only if subclasses must choose the product

```cpp
class creator_interface
{
public:
    virtual ~creator_interface()                                    = default;
    [[nodiscard]] virtual std::unique_ptr<product> create_product() = 0;
};

class perfetto_creator final : public creator_interface
{
public:
    [[nodiscard]] std::unique_ptr<product> create_product() override
    {
        return std::make_unique<perfetto_exporter>();
    }
};
```
