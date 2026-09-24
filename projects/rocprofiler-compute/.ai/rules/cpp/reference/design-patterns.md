# Reference: Design Patterns

Reference material, not a rule. Nothing here is enforced in review. Read it when
a problem looks like it already has a name.

Reference: [Refactoring.Guru, design patterns in
C++](https://refactoring.guru/design-patterns/cpp).

Before reaching for any of this: simple code beats a clever pattern, and a
pattern added "just in case" is a cost with no benefit. Add one when the problem
is actually in front of you.

## Creational

| Pattern | Fits when |
|---|---|
| Factory Method | A conditional decides which concrete class to build |
| Abstract Factory | Families of related objects must be used together |
| Builder | Construction has many optional parameters or several steps |
| Prototype | You need to copy an object without knowing its concrete type |

Singleton is deliberately absent from this table. Process-lifetime state in this
project is governed by [`../callback-boundaries.md`](../callback-boundaries.md),
which is a rule, not a suggestion.

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

## Structural

| Pattern | Fits when |
|---|---|
| Adapter | An existing interface is the wrong shape |
| Bridge | Abstraction and implementation must vary independently |
| Composite | A tree where leaves and nodes should look the same to callers |
| Decorator | Behaviour is added at runtime instead of by subclassing |
| Facade | A complex subsystem needs one simple entry point |
| Flyweight | Many objects share the same immutable state |
| Proxy | Access needs controlling: lazy construction, caching, logging |

Adapter is the one that shows up most here. `SdkWrapper` in
`src/lib/rocprofiler_compute_tool/sdk_wrapper.h` is an adapter over the
rocprofiler-sdk C API, which is also what makes it mockable.

```cpp
class Adapter : public Target {
    Adaptee* m_adaptee;
public:
    void request() override { m_adaptee->specific_request(); }
};
```

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
| Strategy | Algorithms are interchangeable at runtime |
| Template Method | The algorithm shape is fixed, individual steps vary |
| Visitor | Operations get added without touching the element classes |

```cpp
// Strategy
class Strategy {
public:
    virtual ~Strategy() = default;
    virtual void execute() = 0;
};

class Context {
    std::unique_ptr<Strategy> m_strategy;
public:
    void set_strategy(std::unique_ptr<Strategy> s) { m_strategy = std::move(s); }
};

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
the hot path. See [`../performance.md`](../performance.md).
