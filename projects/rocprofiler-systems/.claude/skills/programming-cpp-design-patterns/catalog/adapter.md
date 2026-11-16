# Adapter

Catalog id: `adapter`

An existing API does not match the call the caller wants to make.

## Implement this

Translate at the call site.

```cpp
void
request(const adaptee& object)
{
    object.specific_request();
}
```

## Pattern only if the translation must replace the target type wherever it is used

```cpp
class adapter : public target
{
public:
    explicit adapter(adaptee& object)
    : m_adaptee{ object }
    {}

    void request() override { m_adaptee.specific_request(); }

private:
    adaptee& m_adaptee;
};
```

`m_adaptee` is non-owning. The caller keeps the adapted object alive.
