---
name: programming-cpp-design-patterns
description: C++ design patterns reference - suggests applicable patterns when code matches a problem that a pattern solves
---

# C++ Design Patterns Skill

Use this skill when implementing C++ code that could benefit from design patterns.

<IMPORTANT>
When you recognize a problem that matches a design pattern, **suggest the pattern to the user** with a brief explanation of why it fits.
</IMPORTANT>

Reference: [Refactoring.Guru - Design Patterns in C++](https://refactoring.guru/design-patterns/cpp)

## When to Suggest Patterns

Actively look for these problem indicators and suggest the matching pattern:

### Creational Patterns

| Pattern | Suggest When... |
| --------- | ----------------- |
| **Factory Method** | Code uses `new` with conditional logic to decide which class to instantiate |
| **Abstract Factory** | Need to create families of related objects that must be used together |
| **Builder** | Constructor has many parameters, or object requires step-by-step construction |
| **Prototype** | Need to copy objects without depending on their concrete classes |
| **Singleton** | Need exactly one instance with global access (use sparingly!) |

### Structural Patterns

| Pattern | Suggest When... |
| --------- | ----------------- |
| **Adapter** | Need to use a class with an incompatible interface |
| **Bridge** | Want to separate abstraction from implementation (vary both independently) |
| **Composite** | Working with tree structures where leaf and composite should be treated uniformly |
| **Decorator** | Need to add behaviors to objects dynamically without subclassing |
| **Facade** | Need to provide a simple interface to a complex subsystem |
| **Flyweight** | Need many similar objects that share common state to save memory |
| **Proxy** | Need to control access to an object (lazy loading, access control, logging) |

### Behavioral Patterns

| Pattern | Suggest When... |
| --------- | ----------------- |
| **Chain of Responsibility** | Multiple handlers could process a request, decided at runtime |
| **Command** | Need to parameterize objects with operations, queue operations, or support undo |
| **Iterator** | Need to traverse a collection without exposing its internal structure |
| **Mediator** | Many objects communicate in complex ways, creating tight coupling |
| **Memento** | Need to save and restore object state (undo/redo functionality) |
| **Observer** | Objects need to be notified when another object changes state |
| **State** | Object behavior changes based on internal state (like a state machine) |
| **Strategy** | Need to select algorithm at runtime, or have multiple ways to do something |
| **Template Method** | Algorithm structure is fixed but some steps vary in subclasses |
| **Visitor** | Need to add operations to objects without modifying their classes |

### C++ Idioms

| Pattern | Suggest When... |
| --------- | ----------------- |
| **Type Erasure** | Need polymorphism for unrelated types without inheritance (duck typing) |

## How to Suggest

When you detect a pattern opportunity, present it like this:

> **Design Pattern Suggestion: [Pattern Name]**
>
> I notice you're [describe the problem]. This is a good fit for the **[Pattern Name]** pattern.
>
> **Why it fits:**
>
> - [Reason 1]
> - [Reason 2]
>
> **Would you like me to implement it using this pattern?**

## Pattern Details

### Creational Patterns

#### Factory Method

```cpp
// Problem: Direct instantiation with conditional logic
if (type == "A") return new ProductA();
else return new ProductB();

// Solution: Factory Method
class Creator {
public:
    virtual std::unique_ptr<Product> create_product() = 0;
};
```

**Use when:** Subclasses should decide which class to instantiate.

#### Abstract Factory

```cpp
// Creates families of related objects
class GUIFactory {
public:
    virtual std::unique_ptr<Button> create_button() = 0;
    virtual std::unique_ptr<Checkbox> create_checkbox() = 0;
};
```

**Use when:** System should be independent of how products are created.

#### Builder

```cpp
// Problem: Complex constructor
Car(engine, wheels, seats, gps, sunroof, color, ...);

// Solution: Builder
auto car = CarBuilder()
    .set_engine(v8)
    .set_wheels(4)
    .set_gps(true)
    .build();
```

**Use when:** Object construction has many optional parameters or steps.

#### Prototype

```cpp
class Prototype {
public:
    virtual std::unique_ptr<Prototype> clone() const = 0;
};
```

**Use when:** Need to copy objects without coupling to their concrete classes.

#### Singleton

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

**Use sparingly:** Only when exactly one instance is truly needed (logger, config).

### Structural Patterns

#### Adapter

```cpp
// Make incompatible interface work with expected interface
class Adapter : public Target {
    Adaptee* m_adaptee;
public:
    void request() override {
        m_adaptee->specific_request();
    }
};
```

**Use when:** Integrating legacy code or third-party libraries.

#### Bridge

```cpp
// Separate abstraction from implementation
class Abstraction {
protected:
    std::unique_ptr<Implementation> m_impl;
public:
    virtual void operation() {
        m_impl->operation_impl();
    }
};
```

**Use when:** Both abstraction and implementation need to vary independently.

#### Composite

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

**Use when:** Working with tree structures (file systems, UI components, org charts).

#### Decorator

```cpp
class Decorator : public Component {
protected:
    std::unique_ptr<Component> m_wrapped;
public:
    void operation() override {
        m_wrapped->operation();
        // Add behavior
    }
};
```

**Use when:** Need to add responsibilities dynamically (streams, middleware).

#### Facade

```cpp
class Facade {
    SubsystemA m_a;
    SubsystemB m_b;
    SubsystemC m_c;
public:
    void simple_operation() {
        m_a.op1();
        m_b.op2();
        m_c.op3();
    }
};
```

**Use when:** Need simple interface to complex subsystem.

#### Flyweight

```cpp
class FlyweightFactory {
    std::unordered_map<std::string, std::shared_ptr<Flyweight>> m_cache;
public:
    std::shared_ptr<Flyweight> get_flyweight(const std::string& key);
};
```

**Use when:** Many objects share common state (text characters, game particles).

#### Proxy

```cpp
class Proxy : public Subject {
    std::unique_ptr<RealSubject> m_real;
public:
    void request() override {
        // Access control, lazy loading, logging, caching
        if (!m_real) m_real = std::make_unique<RealSubject>();
        m_real->request();
    }
};
```

**Use when:** Need lazy initialization, access control, logging, or caching.

### Behavioral Patterns

#### Chain of Responsibility

```cpp
class Handler {
    std::shared_ptr<Handler> m_next;
public:
    virtual void handle(Request& req) {
        if (m_next) m_next->handle(req);
    }
};
```

**Use when:** Multiple handlers for requests (middleware, event handling).

#### Command

```cpp
class Command {
public:
    virtual void execute() = 0;
    virtual void undo() = 0;
};
```

**Use when:** Need undo/redo, queuing, or scheduling operations.

#### Iterator

```cpp
class Iterator {
public:
    virtual bool has_next() = 0;
    virtual Element& next() = 0;
};
```

**Use when:** Traversing custom collections. (Prefer STL iterators in C++)

#### Mediator

```cpp
class Mediator {
public:
    virtual void notify(Component* sender, const std::string& event) = 0;
};
```

**Use when:** Components have complex interdependencies.

#### Memento

```cpp
class Memento {
    State m_state;
    friend class Originator;
};

class Originator {
public:
    Memento save() { return Memento{m_state}; }
    void restore(const Memento& m) { m_state = m.m_state; }
};
```

**Use when:** Need save/restore functionality (undo, checkpoints).

#### Observer

```cpp
class Subject {
    std::vector<Observer*> m_observers;
public:
    void attach(Observer* o) { m_observers.push_back(o); }
    void notify() {
        for (auto* o : m_observers) o->update();
    }
};
```

**Use when:** Objects need to react to changes in other objects.

#### State

```cpp
class State {
public:
    virtual void handle(Context& ctx) = 0;
};

class Context {
    std::unique_ptr<State> m_state;
public:
    void set_state(std::unique_ptr<State> s) { m_state = std::move(s); }
    void request() { m_state->handle(*this); }
};
```

**Use when:** Object behavior depends on state (state machines).

#### Strategy

```cpp
class Strategy {
public:
    virtual void execute() = 0;
};

class Context {
    std::unique_ptr<Strategy> m_strategy;
public:
    void set_strategy(std::unique_ptr<Strategy> s) { m_strategy = std::move(s); }
};
```

**Use when:** Need interchangeable algorithms (sorting, validation, formatting).

#### Template Method

```cpp
class AbstractClass {
public:
    void template_method() {
        step1();
        step2();  // Override this
        step3();
    }
protected:
    virtual void step2() = 0;
};
```

**Use when:** Algorithm structure is fixed but some steps vary.

#### Visitor

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

**Use when:** Need to add operations without modifying element classes.

## C++ Idioms

### Type Erasure (Type View)

Type erasure allows storing heterogeneous types in a container without inheritance.

**Suggest when:**

- Need a container of unrelated types that share a common interface (but don't inherit from a base class)
- Want polymorphism without virtual inheritance
- Working with types you can't modify (third-party, built-in)

```cpp
// Type-erased view - stores any type with a postprocess() method
class postprocessable_view {
public:
    template<typename T>
    explicit postprocessable_view(T& obj)
        : m_object{&obj}
        , m_postprocess_impl{[](void* ptr) {
              static_cast<T*>(ptr)->postprocess();
          }}
    {}

    void postprocess() { m_postprocess_impl(m_object); }

private:
    void* m_object;
    std::function<void(void*)> m_postprocess_impl;
};

// Usage: unrelated types, no common base class
struct sensor {
    void postprocess() { /* calibrate */ }
};

struct image {
    void postprocess() { /* apply filters */ }
};

// Store heterogeneous types in one container
std::vector<postprocessable_view> items;
sensor s;
image img;
items.emplace_back(s);
items.emplace_back(img);

for (auto& item : items) {
    item.postprocess();  // Calls correct implementation
}
```

**Key points:**

- **No inheritance required** - types just need matching method signature
- **Non-owning** - view references external objects (caller manages lifetime)
- **Runtime cost** - `std::function` has overhead; for hot paths use function pointers

**Owning variant** (owns the object):

```cpp
class postprocessable {
public:
    template<typename T>
    explicit postprocessable(T obj)
        : m_storage{std::make_unique<model<T>>(std::move(obj))}
    {}

    void postprocess() { m_storage->postprocess(); }

private:
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
};

// Owns the objects
std::vector<postprocessable> items;
items.emplace_back(sensor{});
items.emplace_back(image{});
```

**When to use which:**

| Variant | Use When |
| --------- | ---------- |
| **Type View (non-owning)** | Objects live elsewhere, just need polymorphic access |
| **Owning Type Erasure** | Container should own the objects |
| **std::variant** | Fixed set of known types (prefer this when possible) |
| **Virtual inheritance** | Types naturally form a hierarchy |

**Standard library examples:** `std::function`, `std::any`, `std::move_only_function` (C++23)

## Anti-Patterns to Avoid

- **Singleton abuse** - Don't use for everything; it's often a code smell
- **Over-engineering** - Don't add patterns "just in case"
- **Pattern obsession** - Simple code is better than clever patterns
- **Wrong pattern** - Make sure the problem actually matches

## References

- [Refactoring.Guru - Design Patterns in C++](https://refactoring.guru/design-patterns/cpp)
- [Gang of Four - Design Patterns Book](https://en.wikipedia.org/wiki/Design_Patterns)
