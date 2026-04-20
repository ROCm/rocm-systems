# Python Coding Style — AI-Authoritative Rules

> These rules are the single authoritative source for AI coding assistants generating or modifying
> Python code in this project. They take precedence over general conventions.
> Full rationale and worked examples: [`PYTHON_CODING_STYLE.md`](../../PYTHON_CODING_STYLE.md)

---

## Functions

- Every function must do exactly **one** thing. If "and" appears in its description, split it.
- For any non-trivial or multi-step logic, prefer extracting a named helper over keeping an anonymous
  inline block, even if the helper is used only once. This does **not** apply to trivially readable
  single expressions or single standard-library calls.
- Do not reimplement logic that already exists in a helper — find it and call it.
- Do not produce nested functions unless the inner function is genuinely private to the outer scope
  and not reusable.
- Never mix I/O with computation in the same function. Separate them.

## Naming

- Use descriptive names. Never shorten for brevity.
- The smaller the scope of a function, the more specific its name must be.
- The larger the scope of a variable, the more specific its name must be.
- Do not use single-letter names, abbreviations, or vague names (`data`, `info`, `result`, `tmp`,
  `val`) except within a very tight, obvious loop.

## Helper Function Extraction

**These are strict requirements, not suggestions.**

- **Always extract** when: logic is repeated, an expression benefits from a name, or a block mixes
  abstraction levels.
- **Always prefer** combining existing helpers in new ways over modifying them (open-closed
  principle).
- **Never extract** when: the function name adds no information over reading the code directly, or
  the operation is a single standard-library call.

## Nesting

- Maximum **2 levels** preferred; **3 levels** is the hard limit.
- Use guard clauses (early returns/continues) to eliminate nesting rather than adding `else` branches.
- Invert conditions when possible — prefer `if not condition: return` over wrapping the body in
  `if condition: { ... }`.
- If reaching the nesting limit, extract the inner block into a named function.

## Abstraction Levels

- Every statement in a function must operate at the **same abstraction level**.
- High-level orchestration functions call named helpers — they do not contain raw file I/O, string
  manipulation, or byte operations inline.
- If adding a low-level operation to a high-level function, extract it into a helper first.

## Code Organization

Module structure order (strictly):
1. Module docstring
2. Imports (stdlib → third-party → local), sorted within each group
3. Constants
4. Public functions
5. Private helpers (prefixed `_`)
6. Classes

Additional rules:
- Public functions appear **before** private helpers in every file.
- Use `is None` / `is not None` — never `== None` or `!= None`.

## What NOT to Generate

- Wrapper functions whose body is a single, already-readable expression.
- Functions whose name merely repeats what the code obviously does (`def get_len(x): return len(x)`).
- Logic that already exists elsewhere in the codebase — search first, reuse second, write last.
