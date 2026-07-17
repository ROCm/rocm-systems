# C++ Design Patterns Reference

Reference: [Refactoring.Guru - Design Patterns in C++](https://refactoring.guru/design-patterns/cpp)

## Creational Patterns

### Factory Method

```cpp
class Creator {
public:
    virtual std::unique_ptr<Product> create_product() = 0;
};
```

**Use when:** Subclasses decide which class to instantiate. Replaces conditional `new`.

### Abstract Factory

```cpp
class GUIFactory {
public:
    virtual std::unique_ptr<Button> create_button() = 0;
    virtual std::unique_ptr<Checkbox> create_checkbox() = 0;
};
```

**Use when:** Creating families of related objects that must be used together.

### Builder

```cpp
auto car = CarBuilder()
    .set_engine(v8)
    .set_wheels(4)
    .set_gps(true)
    .build();
```

**Use when:** Many constructor parameters or step-by-step construction.

### Prototype

```cpp
class Prototype {
public:
    virtual std::unique_ptr<Prototype> clone() const = 0;
};
```

**Use when:** Copying objects without coupling to concrete classes.

### Singleton

```cpp
class Singleton {
public:
    static Singleton& instance() {
        static Singleton instance;
        return instance;
    }
private:
    Singleton() = default;
};
```

**Use sparingly.** Only when exactly one instance is truly needed.

## Structural Patterns

### Adapter

```cpp
class Adapter : public Target {
    Adaptee* m_adaptee;
public:
    void request() override { m_adaptee->specific_request(); }
};
```

**Use when:** Integrating legacy code or third-party libraries with incompatible interfaces.

### Bridge

```cpp
class Abstraction {
protected:
    std::unique_ptr<Implementation> m_impl;
public:
    virtual void operation() { m_impl->operation_impl(); }
};
```

**Use when:** Both abstraction and implementation need to vary independently.

### Composite

```cpp
class Component {
public:
    virtual void operation() = 0;
    virtual void add(std::shared_ptr<Component>) {}
};
class Composite : public Component {
    std::vector<std::shared_ptr<Component>> m_children;
};
```

**Use when:** Tree structures (file systems, UI components, org charts).

### Decorator

```cpp
class Decorator : public Component {
protected:
    std::unique_ptr<Component> m_wrapped;
public:
    void operation() override {
        m_wrapped->operation();
    }
};
```

**Use when:** Adding responsibilities dynamically (streams, middleware).

### Facade

```cpp
class Facade {
    SubsystemA m_a; SubsystemB m_b; SubsystemC m_c;
public:
    void simple_operation() { m_a.op1(); m_b.op2(); m_c.op3(); }
};
```

**Use when:** Simple interface to complex subsystem.

### Flyweight

```cpp
class FlyweightFactory {
    std::unordered_map<std::string, std::shared_ptr<Flyweight>> m_cache;
public:
    std::shared_ptr<Flyweight> get_flyweight(const std::string& key);
};
```

**Use when:** Many objects share common state (text characters, game particles).

### Proxy

```cpp
class Proxy : public Subject {
    std::unique_ptr<RealSubject> m_real;
public:
    void request() override {
        if (!m_real) m_real = std::make_unique<RealSubject>();
        m_real->request();
    }
};
```

**Use when:** Lazy initialization, access control, logging, or caching.

## Behavioral Patterns

### Chain of Responsibility

```cpp
class Handler {
    std::shared_ptr<Handler> m_next;
public:
    virtual void handle(Request& req) { if (m_next) m_next->handle(req); }
};
```

**Use when:** Multiple handlers for requests (middleware, event handling).

### Command

```cpp
class Command {
public:
    virtual void execute() = 0;
    virtual void undo() = 0;
};
```

**Use when:** Undo/redo, queuing, or scheduling operations.

### Mediator

```cpp
class Mediator {
public:
    virtual void notify(Component* sender, const std::string& event) = 0;
};
```

**Use when:** Components have complex interdependencies.

### Memento

```cpp
class Originator {
public:
    Memento save() { return Memento{m_state}; }
    void restore(const Memento& m) { m_state = m.m_state; }
};
```

**Use when:** Save/restore functionality (undo, checkpoints).

### Observer

```cpp
class Subject {
    std::vector<Observer*> m_observers;
public:
    void attach(Observer* o) { m_observers.push_back(o); }
    void notify() { for (auto* o : m_observers) o->update(); }
};
```

**Use when:** Objects need to react to changes in other objects.

### State

```cpp
class Context {
    std::unique_ptr<State> m_state;
public:
    void set_state(std::unique_ptr<State> s) { m_state = std::move(s); }
    void request() { m_state->handle(*this); }
};
```

**Use when:** Object behavior depends on state (state machines).

### Strategy

```cpp
class Context {
    std::unique_ptr<Strategy> m_strategy;
public:
    void set_strategy(std::unique_ptr<Strategy> s) { m_strategy = std::move(s); }
};
```

**Use when:** Interchangeable algorithms (sorting, validation, formatting).

### Template Method

```cpp
class AbstractClass {
public:
    void template_method() { step1(); step2(); step3(); }
protected:
    virtual void step2() = 0;
};
```

**Use when:** Algorithm structure is fixed but some steps vary.

### Visitor

```cpp
class Visitor {
public:
    virtual void visit(ElementA& e) = 0;
    virtual void visit(ElementB& e) = 0;
};
class Element {
public:
    virtual void accept(Visitor& v) = 0;
};
```

**Use when:** Adding operations without modifying element classes.

## C++ Idioms

### Type Erasure (Non-owning View)

```cpp
class postprocessable_view {
public:
    template<typename T>
    explicit postprocessable_view(T& obj)
        : m_object{&obj}
        , m_postprocess_impl{[](void* ptr) { static_cast<T*>(ptr)->postprocess(); }}
    {}
    void postprocess() { m_postprocess_impl(m_object); }
private:
    void* m_object;
    std::function<void(void*)> m_postprocess_impl;
};
```

**Use when:** Polymorphism for unrelated types without inheritance (duck typing).

### Type Erasure (Owning)

```cpp
class postprocessable {
    struct concept_t {
        virtual ~concept_t() = default;
        virtual void postprocess() = 0;
    };
    template<typename T>
    struct model : concept_t {
        explicit model(T obj) : m_obj(std::move(obj)) {}
        void postprocess() override { m_obj.postprocess(); }
        T m_obj;
    };
    std::unique_ptr<concept_t> m_storage;
public:
    template<typename T>
    explicit postprocessable(T obj) : m_storage{std::make_unique<model<T>>(std::move(obj))} {}
    void postprocess() { m_storage->postprocess(); }
};
```

| Variant | Use When |
| --------- | ---------- |
| **Type View (non-owning)** | Objects live elsewhere, just need polymorphic access |
| **Owning Type Erasure** | Container should own the objects |
| **`std::variant`** | Fixed set of known types (prefer when possible) |
| **Virtual inheritance** | Types naturally form a hierarchy |

## Anti-Patterns

- **Singleton abuse** — don't use for everything; often a code smell
- **Over-engineering** — don't add patterns "just in case"
- **Pattern obsession** — simple code beats clever patterns
- **Wrong pattern** — verify the problem actually matches
