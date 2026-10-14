---
name: programming-cpp-design-patterns
description: Use when writing or changing C++ that creates objects conditionally, adds a class hierarchy, shares one operation across types, notifies collaborators, or is asked for a singleton, factory, strategy, observer, or other global instance
---

# C++ Design Patterns

Match the symptom in [catalog.md](catalog.md). Read only that row's example file and implement the section that applies.

**Core principle:** Pick the design while writing the type. The example file shows the shape to implement.

## Before writing

When the change adds a global instance, a base class, a creation branch, or a callback list:

1. Search `catalog.md` by the symptom or pattern name.
2. Write `catalog: <id> | <choice>`.
3. Read that row's example file and no other pattern file.
4. Implement its **Implement this** section. If the row's **pattern only if** condition is true, implement **Pattern only if** instead.
5. The catalog line must name the choice the code implements.

Search: `rg -n -i "singleton|variant|callback" catalog.md`

## Singleton

A request for global access does not satisfy the singleton row. Pass the object. The sample is [catalog/singleton.md](catalog/singleton.md).

| Excuse | Reality |
| --- | --- |
| "The user asked for a singleton" | Global access is the symptom, not the requirement |
| "Every translation unit needs it" | Pass the object into the function that uses it |
| "I checked the catalog" | The `catalog:` line names the choice the code implements |

## Red flags

- `instance()` or a function-local static added for convenient access
- A `catalog:` line that does not match the code
- A hierarchy added while the matching row's first choice is a language or standard-library facility
- Code that does not follow the opened example's ownership and call shape

## Example

Request: "Make the logger a singleton so every translation unit can reach it."

`catalog: singleton | pass the object as a parameter`

```cpp
void
run_worker(logger& log);
```

## Common mistakes

| Code being written | Read |
| --- | --- |
| Independent optional fields | [catalog/builder.md](catalog/builder.md) |
| One known listener | [catalog/observer.md](catalog/observer.md) |
| Closed enum of algorithms | [catalog/strategy.md](catalog/strategy.md) |
| Closed set of types | [catalog/type-erasure.md](catalog/type-erasure.md) |
| Open set, same operation, no shared base | [catalog/type-erasure.md](catalog/type-erasure.md) |
