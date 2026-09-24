# C++ Core Rules

The [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
are the reference for anything this file does not cover.

## Language version

C++17 is the baseline. `src/lib/CMakeLists.txt` sets it for the whole tree.

One exception: `torch_trace_collector` builds as C++20, because libtorch
requires it. Its `CMakeLists.txt` raises the standard for that target only.
Outside that target, do not use concepts, ranges, `std::span`, `std::format`,
coroutines, `consteval`, `constinit`, or `[[nodiscard("message")]]`.

For a span, look for an existing view type in the tree first. If there is none,
pass a pointer and a size, or an iterator pair.

## Interfaces

- Make interfaces explicit and strongly typed.
- Never transfer ownership through a raw pointer or reference. A raw `T*` or
  `T&` is non-owning.
- Do not pass an array as a single pointer.
- Avoid non-const globals and singletons. The one place we allow
  process-lifetime state is covered in [`callback-boundaries.md`](callback-boundaries.md).

## Functions

- A function performs a single logical operation.
- Take `T*` or `T&` rather than a smart pointer, unless the function is about
  ownership.
- Pass cheap-to-copy types by value, everything else by `const&`.
- Pass in-out parameters by non-const reference.
- Prefer a return value to an out parameter. To return several values, return a
  struct or a tuple.
- Use `unique_ptr<T>` to transfer ownership, `shared_ptr<T>` to share it.

## Classes

- Use `class` when there is an invariant to hold. Use `struct` when the members
  vary independently.
- A function is a member only if it needs direct access to the representation.
- Minimize what is exposed. Avoid trivial getters and setters.
- Rule of 0/5: if you define or `=delete` any of the copy, move, or destructor
  functions, handle all of them. `EnvironCache` in
  `src/lib/rocprofiler_compute_tool/environ_cache.h` is an example.
- A base class destructor is either public and virtual, or protected and
  non-virtual.
- A virtual function declares exactly one of `virtual`, `override`, or `final`.
- Initialize members in declaration order. Prefer default member initializers
  and initialization over assignment in the constructor body.
- Do not call virtual functions from constructors or destructors.

## Resources

- Manage every resource with RAII.
- No raw `new`, `delete`, `malloc`, or `free`. Use `make_unique` and
  `make_shared`.
- Prefer scoped objects. Do not heap-allocate without a reason.
- Prefer `unique_ptr` over `shared_ptr` unless ownership is genuinely shared.
- Perform at most one explicit allocation per statement, and hand the result to
  a manager object immediately.

## Errors

- Decide the error strategy for a component up front.
- Throw to signal that a function cannot do its job. Use exceptions only for
  errors, never for control flow.
- Throw purpose-built exception types, not built-in ones.
- Destructors, deallocation, `swap`, and exception copy or move must never fail.
- Do not catch in every function. Minimize explicit `try`/`catch`.
- Exceptions must not escape a host callback. See
  [`callback-boundaries.md`](callback-boundaries.md).

## Early return

Guard clauses instead of nesting.

```cpp
// Bad
std::optional<Result> process(const Input& in) {
    if (in.is_valid()) {
        if (auto data = fetch_data(in)) {
            if (!data->empty()) {
                return compute(*data);
            }
        }
    }
    return std::nullopt;
}

// Good
std::optional<Result> process(const Input& in) {
    if (!in.is_valid()) return std::nullopt;

    auto data = fetch_data(in);
    if (!data) return std::nullopt;
    if (data->empty()) return std::nullopt;

    return compute(*data);
}
```

Refactor a branch chain once it passes one of these: three or more `if`/`else`
branches, five or more `switch` cases, three levels of nesting, or any
`dynamic_cast` chain. Replace it with polymorphism, a lookup table, a command
map, `std::variant` plus `std::visit`, or `if constexpr`. Two simple branches
are fine as they are.

## const

Mark everything `const` that does not change: locals, parameters, member
functions that do not modify the object.

## constexpr

If a value or function can be evaluated at compile time, it should be. Constants
are `static constexpr` members or namespace-scope `constexpr`, as in
`CsvCountersWriter::kFileSuffix`.

A `constexpr` function runs at compile time only when its arguments are
constants, and silently falls back to runtime otherwise. Assigning the result to
a `constexpr` variable makes the compiler prove it:

```cpp
constexpr int square(int x) { return x * x; }

constexpr int a = square(8);   // compile time, or it fails to compile
int b = square(n);             // runtime, n is not constexpr
```

That is the C++17 way to get what `consteval` gives in C++20. `consteval` and
`constinit` are not available.

`if constexpr` branches at compile time, and the branch not taken is not
compiled at all. Prefer it to tag dispatch or SFINAE. It is only valid inside a
template.

```cpp
template <typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>)            return val * 2;
    else if constexpr (std::is_floating_point_v<T>) return val * 2.0;
    else static_assert(always_false<T>, "unsupported type");
}
```

A `constexpr` function must stay pure: no I/O, no global state. That also makes
it testable with `static_assert`, which catches the bug before the binary
exists.

## noexcept

Mark a function `noexcept` when it cannot throw. Move constructors, move
assignment, `swap`, and simple getters should all be `noexcept`. Move operations
being `noexcept` is what lets containers move instead of copy.

## Attributes

| Attribute | Use when |
|---|---|
| `const` | The value or object will not be modified |
| `constexpr` | It can be evaluated at compile time |
| `noexcept` | The function is guaranteed not to throw |
| `[[nodiscard]]` | Ignoring the return value is likely a bug |
| `[[maybe_unused]]` | Deliberately unused, to silence the warning |
| `[[fallthrough]]` | A `switch` case falls through on purpose |

`[[nodiscard]]` belongs on factory functions, error codes, `try_lock`, and
`find`. `[[maybe_unused]]` belongs on parameters used only in some build
configurations and on variables used only in assertions.

## C++17 features to use

- `auto` where the type is obvious, structured bindings, class template argument
  deduction.
- `if constexpr` for compile-time branching.
- `if` and `switch` with an initializer.
- `std::optional`, `std::variant`, `std::string_view`.
- `inline` variables in headers, fold expressions, `constexpr` lambdas.
- `std::filesystem`.

## Checklist

- [ ] No raw `new` or `delete`, no raw owning pointers
- [ ] RAII for every resource, ownership is clear
- [ ] `const` correct
- [ ] `constexpr` on compile-time constants and functions
- [ ] `noexcept` on move operations, `swap`, and simple getters
- [ ] `[[nodiscard]]` on factories and error codes
- [ ] Guard clauses instead of nesting
- [ ] Virtual destructor on base classes, Rule of 0/5 followed
- [ ] No magic numbers, no surprising implicit conversions
