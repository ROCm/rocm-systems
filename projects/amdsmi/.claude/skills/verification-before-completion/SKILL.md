---
name: verification-before-completion
description: "Verification checklist before marking work complete. Use when: finishing a fix, completing a review finding, validating a build, confirming a refactor."
---

# Verification Before Completion

Ensures fixes and changes are actually validated before declaring success. Prevents "it compiles so it works" thinking.

## Core Principle

Never mark work done based on absence of errors. Verify the **positive** — confirm the expected behavior exists.

## Verification Checklist

Before marking any task complete, verify:

### 1. Build Verification
- [ ] Code compiles without new warnings
- [ ] All existing tests still pass
- [ ] Package builds successfully (if applicable)

### 2. Behavioral Verification
- [ ] The fix actually addresses the reported problem (not just a related symptom)
- [ ] Expected output is produced (not just "no error")
- [ ] Edge cases considered: empty input, null handles, max values, concurrent access

### 3. Regression Check
- [ ] No existing functionality broken by the change
- [ ] API contract preserved (same inputs → compatible outputs)
- [ ] No new warnings from static analysis or formatters

### 4. Integration Verification
- [ ] Changes work across all affected layers (C → Python → CLI if applicable)
- [ ] Both system-installed and pip-installed paths work (for Python changes)
- [ ] Both RPM and DEB packaging still work (for build changes)

## Anti-Patterns

| Anti-Pattern | Reality |
|-------------|---------|
| "It compiles" | Compilation proves syntax, not correctness |
| "Tests pass" | Tests may not cover the changed path |
| "No errors in output" | Silent failures are the worst failures |
| "Works on my machine" | Check both install contexts |
| "The diff looks right" | Run the code, don't just read it |

## When to Use

- After fixing a review finding
- After completing a build/install cycle
- After refactoring code
- Before marking a subagent task as done
