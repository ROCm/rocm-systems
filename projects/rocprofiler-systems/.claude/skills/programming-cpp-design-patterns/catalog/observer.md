# Observer

Catalog id: `observer`

Something must run after a change. One known listener is a parameter.

## Implement this

```cpp
class counter {
public:
    explicit counter(panel& panel) noexcept : panel_{panel} {}

    void set_value(int value) {
        if (value == value_) return;
        value_ = value;
        panel_.refresh();
    }

private:
    panel& panel_;
    int value_ = 0;
};
```

One listener whose type is not known at the call:

```cpp
class counter {
public:
    explicit counter(std::function<void()> on_change) : on_change_{std::move(on_change)} {}

    void set_value(int value) {
        if (value == value_) return;
        value_ = value;
        on_change_();
    }

private:
    std::function<void()> on_change_;
    int value_ = 0;
};
```

## Pattern only if the listener set is open and changes at runtime

```cpp
class subject {
public:
    void attach(observer& listener) { listeners_.push_back(&listener); }

    void notify() {
        for (observer* listener : listeners_) listener->update();
    }

private:
    std::vector<observer*> listeners_;
};
```

The subject does not own the observers. The caller keeps them alive and detaches them before they die.
