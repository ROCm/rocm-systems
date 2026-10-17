# Strategy

Catalog id: `strategy`

One of several algorithms is selected. A closed set is a `switch` or a function argument.

## Implement this

```cpp
enum class record_order
{
    by_name,
    by_time,
    by_size
};

void
order_records(std::vector<record>& records, record_order order)
{
    switch(order)
    {
        case record_order::by_name: std::ranges::sort(records, {}, &record::name); return;
        case record_order::by_time: std::ranges::sort(records, {}, &record::time); return;
        case record_order::by_size: std::ranges::sort(records, {}, &record::size); return;
        default: throw std::invalid_argument{ "invalid record order" };
    }
}
```

A compile-time swap is policy-based dependency injection. Follow `programming-cpp-policy-based-di`.

## Pattern only if the choice is runtime and carries state the caller should not see

```cpp
class strategy_interface
{
public:
    virtual ~strategy_interface() = default;
    virtual void execute()        = 0;
};

class context
{
public:
    void set_strategy(std::unique_ptr<strategy_interface> next)
    {
        m_strategy = std::move(next);
    }
    void run() { m_strategy->execute(); }

private:
    std::unique_ptr<strategy_interface> m_strategy;
};
```
