# Flyweight

Catalog id: `flyweight`

Many objects repeat the same immutable data. Share that data. Keep the per-object state at the use site.

## Implement this

```cpp
class flyweight_factory
{
public:
    [[nodiscard]] std::shared_ptr<const flyweight> get(const std::string& key)
    {
        auto [iter, inserted] = m_cache.try_emplace(key, nullptr);
        if(inserted) iter->second = std::make_shared<const flyweight>(key);
        return iter->second;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<const flyweight>> m_cache;
};
```

The position, tick, or other extrinsic value stays in the caller, not in `flyweight`.

## Pattern only if extrinsic state must stay outside a large shared object

Use the factory above. Do not copy the shared state into every object.
