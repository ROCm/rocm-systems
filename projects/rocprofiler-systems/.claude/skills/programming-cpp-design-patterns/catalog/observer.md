# Observer

Catalog id: `observer`

Something must run after a change. One known listener is a parameter.

## Implement this

```cpp
class counter
{
public:
    explicit counter(panel& panel) noexcept
    : m_panel{ panel }
    {}

    void set_value(int value)
    {
        if(value == m_value) return;
        m_value = value;
        m_panel.refresh();
    }

private:
    panel& m_panel;
    int    m_value = 0;
};
```

One listener whose type is not known at the call:

```cpp
class counter
{
public:
    explicit counter(std::function<void()> on_change)
    : m_on_change{ std::move(on_change) }
    {}

    void set_value(int value)
    {
        if(value == m_value) return;
        m_value = value;
        m_on_change();
    }

private:
    std::function<void()> m_on_change;
    int                   m_value = 0;
};
```

## Pattern only if the listener set is open and changes at runtime

```cpp
class subject
{
public:
    void attach(observer& listener) { m_listeners.push_back(&listener); }

    void notify()
    {
        for(observer* listener : m_listeners)
            listener->update();
    }

private:
    std::vector<observer*> m_listeners;
};
```

The subject does not own the observers. The caller keeps them alive and detaches them before they die.
