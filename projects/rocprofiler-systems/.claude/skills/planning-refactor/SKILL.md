---
name: planning-refactor
description: Planning skill for refactoring - improves existing code following best practices, reduces duplication, improves readability and testability
---

# Refactoring Planning

Use this skill when improving EXISTING code without changing its external behavior.

<IMPORTANT>
**Prerequisites:** Invoke `planning-base` skill first if not already loaded. It provides the core planning phases (0-5).

Follow all base planning rules, plus the refactoring-specific rules below.

**Mandatory:** Invoke and apply ALL relevant skills during refactoring:

- `code-smells` - Identify WHAT to fix (smell detection)
- `refactoring-techniques` - HOW to fix it (60+ techniques with examples)
- `programming-cpp` - C++ best practices, performance, testability
- `programming-cpp-design-patterns` - Suggest applicable patterns
- `programming-cpp-stl-algorithms` - Replace loops with STL algorithms
- `programming-cmake-best-practices` - Modern CMake practices
</IMPORTANT>

## Refactoring Goals

Every refactoring should aim for one or more of these:

| Goal | Description |
| ------ | ------------- |
| **Best Practices** | Apply C++ Core Guidelines, modern C++17 features |
| **Reduce Duplication** | Extract common code, apply DRY principle |
| **Improve Readability** | Better naming, smaller functions, clearer intent |
| **Testability** | Dependency injection, policy-based design, pure functions |
| **Performance** | Compile-time computation, cache-friendly, zero-cost abstractions |

## Refactoring-Specific Analysis

### Code Smell Detection

During analysis, actively look for these code smells:

**Duplication:**

- Copy-pasted code blocks
- Similar functions with minor differences
- Repeated patterns that could be templates

**Complexity:**

- Long functions (> 30 lines)
- Deep nesting (> 3 levels)
- God classes (too many responsibilities)
- Long parameter lists

**Poor Abstractions:**

- Raw loops instead of STL algorithms
- Manual memory management instead of RAII
- Inheritance where composition fits better
- Missing interfaces for testability

**C++ Specific:**

- Missing `const`, `constexpr`, `noexcept`
- Unnecessary copies (missing move semantics)
- Raw pointers for ownership
- C-style casts

### Design Pattern Opportunities

Check if any design pattern fits:

- **Strategy** - Multiple algorithms/behaviors
- **Factory** - Complex object creation
- **Adapter** - Interface mismatch
- **Decorator** - Adding behavior dynamically
- **Template Method** - Algorithm skeleton with variable steps

### STL Algorithm Opportunities

Replace manual loops with:

- `std::find`, `std::find_if` - searching
- `std::transform` - mapping
- `std::accumulate` - reducing
- `std::copy_if` - filtering
- `std::any_of`, `std::all_of` - predicates

## Changelog Consideration

After analysis, ASK the user:

> "Should this refactoring be added to the changelog?
>
> - Yes: If it's a significant improvement users should know about
> - No: If it's internal cleanup with no user impact
>
> Add to changelog? (yes/no)"

If yes:

```markdown
## Changelog Entry

### Changed
- <Brief description of what was refactored and why>
```

## Plan File Format

Save to `planning/refactor-<name>.md`:

```markdown
# Refactor: <Component/Area Name>

## Goal
<What is being refactored and why>

## Code Smells Identified
- [ ] <Smell 1>: <Location and description>
- [ ] <Smell 2>: <Location and description>

## Refactoring Strategy

### Best Practices to Apply
- <Guideline 1>
- <Guideline 2>

### Design Patterns to Use
- <Pattern>: <Where and why>
(or "No patterns needed")

### STL Algorithms to Apply
- Replace <loop description> with <algorithm>
(or "No algorithm replacements")

## Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Verify behavior unchanged (tests pass)
- [ ] Verify no new warnings/errors

## Testability Improvements
- <How code will be more testable>
(or "Already testable")

## Changelog
<Entry if requested, or "Not added to changelog">

## Notes
<Any additional context or decisions>
```

## Example

**User request:** "Refactor the parser module - it's hard to test and has lots of duplication"

**Planning output:**

```markdown
# Refactor: Parser Module

## Goal
Improve testability and reduce code duplication in the parser module

## Code Smells Identified
- [ ] Duplication: `parse_int()` and `parse_float()` share 80% of code
- [ ] Untestable: `parser` class creates its own `file_reader` internally
- [ ] Long function: `parse_expression()` is 150 lines with deep nesting
- [ ] Raw loops: Manual iteration in `parse_tokens()` instead of STL

## Refactoring Strategy

### Best Practices to Apply
- F.2: Functions should perform single logical operation
- I.1: Make interfaces explicit (inject dependencies)
- Use `std::string_view` for read-only string params
- Add `[[nodiscard]]` to parsing functions

### Design Patterns to Use
- **Strategy**: Extract number parsing into policy template
- **Dependency Injection**: Inject `i_reader` interface for testability

### STL Algorithms to Apply
- Replace token search loop with `std::find_if`
- Replace validation loop with `std::all_of`

## Tasks
- [ ] Extract `i_reader` interface from `file_reader`
- [ ] Inject reader dependency into `parser` constructor
- [ ] Create `parse_number<T>` template to unify int/float parsing
- [ ] Split `parse_expression()` into smaller functions
- [ ] Replace manual loops with STL algorithms
- [ ] Add `const`, `noexcept`, `[[nodiscard]]` where appropriate
- [ ] Verify all existing tests still pass

## Testability Improvements
- Parser can now be tested with mock reader
- Smaller functions are easier to unit test
- Pure parsing functions can be tested in isolation

## Changelog
### Changed
- Refactored parser module for improved testability and maintainability

## Notes
- Keep public API unchanged
- Existing tests must pass without modification
```

## Safety Rules

<IMPORTANT>
Refactoring MUST NOT change external behavior:

1. **Run tests before AND after** each refactoring step
2. **Small incremental changes** - one smell at a time
3. **Commit frequently** - easy to rollback if something breaks
4. **If tests fail** - revert and analyze before proceeding
</IMPORTANT>

## Test Plan

After COMPLETING the refactoring:

1. **Create test plan file:** `planning/testplan-<refactor-name>.md`
2. Invoke `testing-testplan` skill for the template
3. Document verification that behavior is unchanged
4. Include regression checks for all affected functionality

## Test Consideration

Then ASK the user:

> "Refactoring is complete. Would you like to add/improve unit tests?
>
> - The code is now easier to test due to dependency injection
> - Are there functions not covered by tests?"

If user agrees:

1. Invoke the testing skill: `testing-gtest-gmock` (C++) or `testing-pytest` (Python)
2. Follow the test planning process from that skill
3. Write tests ONE BY ONE, waiting for user approval after each test
4. Update test plan file with test status

## Pull Request

After tests (if any), ASK the user:

> "Ready to create a Pull Request. Should I proceed?"

If yes, invoke `git-create-pull-request` skill and create PR with:

- **Motivation** - Why refactoring was needed (code smells, testability, etc.)
- **Technical Details** - What was changed and design decisions
- **Test Plan** - Verification that behavior is unchanged
