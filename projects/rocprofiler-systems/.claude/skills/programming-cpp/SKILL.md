---
name: programming-cpp
description: Use when writing or modifying C++ — adding or changing a function, class, or header, including a change that "just needs to be quick"
---

# C++ Core Guidelines

Hold these while writing the code, not after it exists. They are the guidelines.
When a change needs a specific technique, load the project skill named below
before writing that part.

Hard rules for this repository (C++20 only, formatting, casts, no timemory) are
in `.claude/rules/cpp-conventions.md`. Follow them.

## Focus while developing

**Less code.** Remove first, then add. Before a new function, type, branch, or
parameter, delete what the change makes unnecessary and reuse what is already
there. Write only what the current use needs. Delete a helper, parser, or guard
written for a case nobody has hit. Leave the file simpler, and no larger, than
you found it.

**Performance.** The hot path is the code that runs per sample or per event.
Keep allocation, copies of large objects, virtual calls, type-erased callables,
shared-ownership copies, and locks off that path. Pass large inputs by
reference to const and small, cheap values by value. Size a container up front
when the count is known. Measure before making the code harder to read.

**Easy to extend.** One type, one responsibility. Put the part that will grow —
a new source, a new device, a new format — behind a seam, so adding it adds a
type instead of another branch in the core. Keep parsing, math, and state
changes as pure functions over values, and keep files, clocks, the environment,
system calls, and GPU APIs at the edge. A reader can then test and reuse the
core without the effect.

**Easy to read.** The type and the name say what the value is. Prefer immutable
data. Return a result instead of writing through an output argument. Leave a
function early when a precondition fails, instead of nesting the rest. A
comment states only the why the code cannot show.

**Testable through its API.** Write the unit test together with the code, not
after. Keep the public surface minimal: expose only what a user of the module
needs, and keep everything else encapsulated inside the module. Test through
that public API only. Never make a member public, add an accessor, or add a
hook just so a test can reach it; if a behavior cannot be tested through the
API, the design needs a seam for its dependency, not a wider interface.

## Respect in every change

- A failure the caller must handle is part of the return type. Never turn a
  failed read into a zero or a default that looks like real data.
- When a library throws, treat it as a report that the call may be wrong.
  Check the arguments and the preconditions before writing a handler. Catch
  only an error the program has a real response to; swallowing it hides a misuse.
- Ownership is visible: a value first, then unique ownership, then shared.
  A raw pointer or reference never owns. Wrap a resource in the same step that
  acquires it.
- Mark what does not change as const. Use a class only when the type has an
  invariant to protect; otherwise a struct.
- Prefer a check at compile time over the same check at run time, when the
  inputs are already known then.
- Match the naming, error style, and helpers of the file being edited. Do not
  copy a local habit that breaks the points above.

## Load the matching skill

Read the skill in the right column before writing that part. Each one holds the
form; this skill does not repeat it.

| You are about to | Skill |
| --- | --- |
| Move work to compile time: a constant, a lookup table, a branch on a type | programming-cpp-constexpr |
| Hide a system call, file, clock, or third-party API so the code is testable without a virtual call | programming-cpp-policy-based-di |
| Choose a container, or replace a hand-written loop with a standard algorithm | programming-cpp-stl-algorithms |
| Add a class hierarchy, a creation branch, a shared operation, or a singleton, factory, or strategy | programming-cpp-design-patterns |
| Add or rename a type, file, or namespace | programming-cpp-naming-rules |
| Write the unit test for the code | testing-gtest-gmock |
