# Abstract Factory

Catalog id: `abstract-factory`

Several related objects must be created together so the caller cannot mix families.

## Implement this

One function returns the whole family.

```cpp
struct widget_family {
    std::unique_ptr<button> button;
    std::unique_ptr<checkbox> checkbox;
};

widget_family make_light_widgets() {
    return {std::make_unique<light_button>(), std::make_unique<light_checkbox>()};
}
```

## Pattern only if more than one family is selected at runtime

```cpp
class gui_factory {
public:
    virtual ~gui_factory() = default;
    virtual std::unique_ptr<button> create_button() = 0;
    virtual std::unique_ptr<checkbox> create_checkbox() = 0;
};

class light_factory final : public gui_factory {
public:
    std::unique_ptr<button> create_button() override {
        return std::make_unique<light_button>();
    }
    std::unique_ptr<checkbox> create_checkbox() override {
        return std::make_unique<light_checkbox>();
    }
};
```
