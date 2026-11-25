# C++ Design Pattern Catalog

Search this file, then open **only** the matching example and implement the section that applies.

```bash
rg -n -i "symptom or pattern" catalog.md
```

Ids: `factory-method` `abstract-factory` `builder` `prototype` `singleton` `adapter` `bridge` `composite` `decorator` `facade` `flyweight` `proxy` `chain` `command` `iterator` `mediator` `memento` `observer` `state` `strategy` `template-method` `visitor`

| id | search | symptom | first choice | pattern only if | example |
| --- | --- | --- | --- | --- | --- |
| factory-method | factory, new, make_unique, switch, type tag | A type tag selects which object to create | One function. Closed set: return `std::variant`. Open set: return `std::unique_ptr` to the product base | Subclasses must choose the product | [example](catalog/factory-method.md) |
| abstract-factory | family, theme, matching products | Several related objects must be created together | One function returning a struct of those objects | More than one family is selected at runtime | [example](catalog/abstract-factory.md) |
| builder | constructor, many parameters, optional fields, stepwise | Construction has many fields or steps | Aggregate with default member initializers | Order or cross-field checks must fail before the value exists | [example](catalog/builder.md) |
| prototype | clone, copy, concrete class | A copy is needed without naming the concrete class | Ordinary copy or a `clone()` on the known type | The copy site does not know the dynamic type | [example](catalog/prototype.md) |
| singleton | singleton, instance(), global, logger, config, one instance | Callers want one object reachable from many places | Pass the object as a parameter | It is process-lifetime, mutable, and tests can replace the instance. A request for global access is not enough | [example](catalog/singleton.md) |
| adapter | adapter, incompatible, legacy, third-party | An existing API does not match the call site | A function that translates the call | The translation must replace the target type wherever it is used | [example](catalog/adapter.md) |
| bridge | bridge, abstraction, implementation, two axes | Two parts of a design vary | Vary the one real axis with a template or function | Both axes vary independently at runtime | [example](catalog/bridge.md) |
| composite | composite, tree, leaf, children | Leaf and group are treated the same | A node holding `std::vector` of children | Callers must use one interface for both | [example](catalog/composite.md) |
| decorator | decorator, wrap, middleware, add behavior | Behavior is added around an object | A function wrapper around the call | Combinations are stacked at runtime | [example](catalog/decorator.md) |
| facade | facade, subsystem, simplify | A caller makes a long sequence of subsystem calls | One function that sequences the calls | Many callers need that same surface | [example](catalog/facade.md) |
| flyweight | flyweight, share, intrinsic, intern, many objects | Many objects repeat the same immutable data | Intern the shared data in a map | Extrinsic state must stay outside the shared object | [example](catalog/flyweight.md) |
| proxy | proxy, lazy, access control, cache, remote | Access needs a check, delay, or stand-in | Call the real object, with the check at the call site | The stand-in must have the subject's interface | [example](catalog/proxy.md) |
| chain | chain of responsibility, handler, middleware | Several handlers may process one request | A `std::vector` of functions, called in order | Handlers are inserted at runtime and may stop the chain | [example](catalog/chain.md) |
| command | command, undo, redo, queue, schedule | Work is stored or undone | Store the call and its reverse as two lambdas | The operation is logged, queued, or undone as its own object | [example](catalog/command.md) |
| iterator | iterator, traverse, collection | A collection is walked | Standard iterators and ranges | A custom container cannot provide them | [example](catalog/iterator.md) |
| mediator | mediator, many objects, coupling | Objects call one another in a web | Pass the collaborator a function actually uses | One coordinator replaces those cross-calls | [example](catalog/mediator.md) |
| memento | memento, snapshot, undo, checkpoint, restore | Object state is saved and restored | Copy the value | Callers must not see the private representation | [example](catalog/memento.md) |
| observer | observer, notify, listener, callback, subscribe | Something must run after a change | One collaborator parameter, or one `std::function` | The listener set is open and changes at runtime | [example](catalog/observer.md) |
| state | state, state machine, mode, transition | Behavior depends on mode | An enum and `switch` | Each state is a type and transitions are numerous | [example](catalog/state.md) |
| strategy | strategy, algorithm, policy, interchangeable | One of several algorithms is selected | Closed set: `switch` or a function argument. Compile-time swap: `programming-cpp-policy-based-di` | The choice is runtime and carries state the caller should not see | [example](catalog/strategy.md) |
| template-method | template method, skeleton, steps, hook | The steps are fixed and some steps vary | A function that takes the varying steps as callables | The skeleton must be inherited and steps overridden | [example](catalog/template-method.md) |
| visitor | visitor, double dispatch, operations, closed set | An operation runs over a set of types | Closed set: `std::visit` | The element hierarchy is already open and new operations arrive more often than new elements | [example](catalog/visitor.md) |
