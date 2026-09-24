# C++ Comments and Documentation

Two different things:

| Kind | Purpose | Form | Lives |
|---|---|---|---|
| Documentation | Tells a caller how to use an API | Doxygen block, `/** @param @return @throws */` | Above a public API |
| Inline comment | Explains one non-obvious line | `// short why` | Beside the line |

Code that needs neither is the goal. Reach for either only when the code does
not answer a question a competent reader will have.

## When to write a Doxygen block

Add one when any of these is true and the answer is not already in the function
name and signature:

- The function is part of a public API consumed by another module.
- Units, ownership, or threading are not obvious. "nanoseconds, monotonic",
  "caller takes ownership", "safe to call from any thread".
- The function can fail in a way the return type does not advertise.
- There is a precondition the type system does not express, such as "input must
  be sorted".

Tags: `@param`, `@return`, `@throws`, `@note`, `@warning`, `@see`,
`@deprecated`, `@code` / `@endcode`.

```cpp
/**
 * Sends data over the connection.
 * @param data Pointer to the buffer. Not owned.
 * @param size Number of bytes to send.
 * @return Number of bytes actually sent.
 * @throws ConnectionError If the peer has closed.
 */
[[nodiscard]] size_t send(const void* data, size_t size);
```

## When to write an inline comment

Only when all three hold:

1. The reader cannot get the intent from the code.
2. It will still be true a year from now, after refactors.
3. It is not already in the commit message, the PR description, the tracker, or
   another file.

If one of those fails, drop the comment.

A comment describes the code as it stands. It is not a place to record what you
tried first, what you rejected, or how the code used to look. That belongs in
the commit message.

```cpp
// Good. Each of these carries something the code does not say.
i++;                       // skip header row
timeout *= 2;              // exponential backoff
result |= 0x80;            // protocol sets the MSB on the negative flag
{                          // caller holds m_mutex
    state.value = compute(...);
}
// workaround for FOO-1234; remove once bar.so >= 2.5 ships
auto handle = legacy_open(path, /*safe=*/true);
```

```cpp
// Bad. Each one restates the code. Delete.
i++;                        // increment i
int sum = a + b;            // add a and b
std::vector<int> numbers;   // vector of numbers
m_count = 0;                // initialize count to zero
```

## Things that need no comment at all

- Trivial getters and setters.
- Self-describing predicates: `is_empty()`, `has_value()`.
- Language idioms: `std::move(x)`, `std::make_unique<T>()`.
- Operators with their conventional meaning.
- Constructors and destructors that do what RAII implies.
- A single-use internal helper whose name already says what it does.
- Test bodies. `TEST_F` already names the case.

```cpp
// Noise
/**
 * Returns the size.
 * @return The size.
 */
size_t size() const { return m_size; }

// Fine
size_t size() const { return m_size; }
```

## Long comment blocks

Multi-line banners at the top of a file, above a test case, or between helpers
usually retell something the diff, the test name, or the commit message already
says. Delete by default:

- "Background" or "Test strategy" preambles in test files.
- Banners separating helpers whose names already say which case they cover.
- Comments citing `file:line` somewhere else. Line numbers rot.
- "This used to be broken because X" above the fix.
- Decorative dividers like `// === Helpers ===`.

Keep:

- Hidden invariants. "caller holds m_mutex", "must be called once at startup".
- A workaround with a ticket reference and the condition for removing it.
- Unit, ownership, or threading notes the type system does not enforce. The
  comment above `EnvironCache` in
  `src/lib/rocprofiler_compute_tool/environ_cache.h` is one: it says why the
  class bypasses `getenv()`, which nothing in the code shows.
- Doxygen on public APIs.

If removing a comment would not confuse a competent reader a year from now, it
is noise.
