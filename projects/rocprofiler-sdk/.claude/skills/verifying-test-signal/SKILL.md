---
name: verifying-test-signal
description: "Use when a test suite reports green — a first run, a run after switching branches, a hand-rolled or out-of-tree harness, or tests added to code that already shipped. Catches suites that pass because nothing was compiled, nothing was asserted, or the assertion could not have failed."
---

# Verifying Test Signal

A passing run is evidence only if it could have failed. A suite that cannot fail is worse than no suite: it argues against the bug you are looking for.

**Iron Law:**

```
A GREEN SUITE PROVES NOTHING UNTIL YOU HAVE WATCHED IT GO RED
```

Violating the letter of this rule is violating its spirit.

## When to Use

- A suite passes first try, or right after a change you expected to break it
- You are running a custom build script rather than the project's ctest targets
- You are testing existing behavior, so failing-test-first is unavailable
- A test asserts a predicate or activation state rather than an observable result

## Harness Integrity

The compiler's exit status must reach the shell, or the suite tests whatever was on disk before.

| Rule | Why |
|------|-----|
| `set -eo pipefail`, and keep compiler invocations out of pipelines and command substitutions whose status you discard | `OBJS="$OBJS $(compile ...)"` where `compile` ends in `\| tail -1` yields `tail`'s status, so `set -e` never fires and errors scroll by on stderr |
| `rm -f` each object before compiling, and use a distinct object directory per branch | Otherwise a failed compile leaves the previous branch's object in place and the link succeeds against stale code |
| Read the build log, not just the test summary | The summary is downstream of a build you never confirmed happened |

## Retroactive RED by Mutation

When the code already exists you cannot write the failing test first. Mutation is the equivalent, and it is not optional:

1. Break exactly one production decision — invert a condition, drop a special case, make a gate always admit.
2. Run the suite. Require that *exactly* the intended tests fail, and that tests covering other behavior still pass.
3. Revert and confirm green.

Nothing failed? The test never reads production code. Something unexpected failed? Either it is over-coupled, or you found a second behavior worth naming.

## Vacuity Traps

| Shape | Why it cannot fail | Fix |
|-------|--------------------|-----|
| A predicate asserted from a "nothing configured" state | Trivially satisfied whatever production does | Assert it in a fixture where the subsystem is genuinely active |
| A test that restates the production expression inline | Test and production drift apart independently, each self-consistent | Name the predicate once in a header; test and production both call it |

## Rationalizations

| Excuse | Reality |
|--------|---------|
| "The tests pass" | Against which build? Confirm the objects were rebuilt this run |
| "It built fine" | A silent link against stale objects also prints nothing |
| "The assertion is obviously true" | Then it is not a test. Make it possible to violate |
| "I only moved code, so behavior is unchanged" | Moved code changes which predicate the caller consults. Mutate and see |
| "Writing tests first is impossible here" | True, and irrelevant. Mutation gives you the RED you skipped |

## Red Flags

- A test count with no compile lines above it
- A test name and the production symbol it covers share no call path
- Adding a test to an existing file and never seeing that file's suite fail
