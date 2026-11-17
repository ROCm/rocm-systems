# C++ Design Patterns

Guidance for picking a shape, not a rule a reviewer enforces line by line. Read
it when a problem looks like it already has a name.

Reference: [Refactoring.Guru, design patterns in
C++](https://refactoring.guru/design-patterns/cpp).

Simple code beats a clever pattern, and a pattern added "just in case" is a cost
with no benefit. Add one when the problem is in front of you.

## Patterns already required elsewhere

Two shapes are not optional here, because another rule depends on them.

**Adapter.** A third-party C API gets wrapped in an interface of ours. This is
what makes the code testable at all, so [`testability.md`](testability.md)
requires it. `SdkWrapper` adapts the rocprofiler-sdk, `filesystem_wrapper_t`
adapts disk access, `EnvironCache` adapts the environment block.

```cpp
class Adapter : public Target {
    Adaptee* m_adaptee;
public:
    void request() override { m_adaptee->specific_request(); }
};
```

**Strategy, lookup table, command map, and `std::visit`.** These are the
replacements [`core.md`](core.md) names when a branch chain gets too long. The
rule and the thresholds live there.

## Creational

| Pattern | Fits when |
|---|---|
| Factory Method | A conditional decides which concrete class to build |
| Abstract Factory | Families of related objects must be used together |
| Builder | Construction has many optional parameters or several steps |
| Prototype | You need to copy an object without knowing its concrete type |

```cpp
// Factory Method: replaces `if (type == "A") return new ProductA();`
class Creator {
public:
    virtual ~Creator() = default;
    virtual std::unique_ptr<Product> create_product() = 0;
};

// Builder: replaces Car(engine, wheels, seats, gps, sunroof, colour, ...)
auto car = CarBuilder().set_engine(v8).set_wheels(4).build();
```

Singleton is deliberately absent. Process-lifetime state is governed by
[`callback-boundaries.md`](callback-boundaries.md), which is a rule.

## Structural

| Pattern | Fits when |
|---|---|
| Bridge | Abstraction and implementation must vary independently |
| Composite | A tree where leaves and nodes look the same to callers |
| Decorator | Behaviour is added at runtime instead of by subclassing |
| Facade | A complex subsystem needs one simple entry point |
| Flyweight | Many objects share the same immutable state |
| Proxy | Access needs controlling: lazy construction, caching, logging |

## Behavioural

| Pattern | Fits when |
|---|---|
| Chain of Responsibility | Several handlers might take a request |
| Command | Operations need queuing, scheduling, or undo |
| Iterator | A custom collection needs traversal. Prefer STL iterators |
| Mediator | Components are wired to each other too tightly |
| Memento | State needs saving and restoring |
| Observer | Something must react when another object changes |
| State | Behaviour depends on which state the object is in |
| Template Method | The algorithm shape is fixed, individual steps vary |
| Visitor | Operations get added without touching the element classes |

```cpp
// Template Method
class AbstractClass {
public:
    void run() { step1(); step2(); step3(); }
protected:
    virtual void step2() = 0;
};
```

## Type erasure

Polymorphism across unrelated types that share a call signature but no base
class.

```cpp
// Non-owning view over anything with postprocess()
class PostprocessableView {
public:
    template <typename T>
    explicit PostprocessableView(T& obj)
        : m_object{&obj}
        , m_postprocess{[](void* p) { static_cast<T*>(p)->postprocess(); }}
    {}

    void postprocess() { m_postprocess(m_object); }

private:
    void* m_object;
    std::function<void(void*)> m_postprocess;
};
```

The view does not own the object, so the caller controls its lifetime. An owning
variant holds a `unique_ptr` to an internal `Concept` base with a templated
`Model` derived from it.

| Option | Use when |
|---|---|
| `std::variant` | The set of types is fixed and known. Prefer this |
| Virtual inheritance | The types genuinely form a hierarchy |
| Type-erased view | Objects live elsewhere, you only need polymorphic access |
| Owning type erasure | The container should own the objects |

`std::function` allocates and dispatches indirectly, so keep type erasure off
the hot path. See [`performance.md`](performance.md).
