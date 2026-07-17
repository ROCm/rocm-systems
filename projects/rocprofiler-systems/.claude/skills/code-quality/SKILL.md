---
name: code-quality
description: "Use when reviewing code for anti-patterns, performing code quality audits, assessing technical debt, applying refactoring techniques, or when another skill (pr-review, planning-refactor) needs smell detection and fix guidance. Combines smell detection (WHAT) with refactoring techniques (HOW) into a single workflow."
---

# Code Quality: Smell Detection + Refactoring

Unified skill for detecting code smells and applying the correct refactoring technique. See `catalog.md` for the full reference of 22 smells and 60+ techniques with code examples.

<IMPORTANT>
- Detect smells first, then prescribe fixes from the catalog
- Always report findings in the structured output format below
- Cross-reference: full smell definitions and refactoring steps are in `catalog.md`
</IMPORTANT>

## When to Use

| Context | Trigger |
| --------- | --------- |
| PR review | Invoked by `pr-review` Code Smells Agent |
| Refactoring planning | Before `planning-refactor` to identify targets |
| Code quality audit | User says "find anti-patterns", "code quality", "technical debt" |
| Applying a fix | User says "Extract Method", "Move Field", or names any refactoring technique |
| Quick fix | User wants to refactor specific code immediately |

## When NOT to Use

| Context | Use Instead |
| --------- | ------------- |
| Running linter/clang-tidy tools on files | `static-analysis` |
| Planning the overall refactor session | `planning-refactor` |
| General C++ style/safety rules | `programming-cpp` |

## Severity Scoring

| Severity | Score | Criteria | Action |
| ---------- | ------- | ---------- | -------- |
| Critical | 100 | Causes bugs, crashes, security issues | Block merge |
| Must Fix | 80 | Significantly harms maintainability | Fix before merge |
| Should Fix | 50 | Reduces code quality noticeably | Fix soon |
| Nitpick | 20 | Minor improvement opportunity | Optional |

## Detection Thresholds

| Smell | Threshold | Severity |
| ------- | ----------- | ---------- |
| Long Method | >50 lines | 50 |
| Long Method | >100 lines | 80 |
| Large Class | >500 lines | 50 |
| Large Class | >1000 lines | 80 |
| Long Parameter List | >4 params | 50 |
| Long Parameter List | >6 params | 80 |
| Deep Nesting | >3 levels | 50 |
| Deep Nesting | >5 levels | 80 |
| Duplicate Code | >10 lines identical | 50 |
| Feature Envy | >3 external getters | 50 |
| Message Chain | >3 calls | 50 |
| Shotgun Surgery | >5 classes touched | 80 |
| God Class | Multiple responsibilities | 80 |

## Output Format

Report all findings in this table:

```markdown
| File:Line | Smell | Category | Severity | Fix |
| ----------- | ------- | ---------- | ---------- | ----- |
| handler.cpp:120-195 | Long Method (75 lines) | Bloater | Should Fix (50) | Extract Method: split into validateRequest(), processData(), buildResponse() |
| config.cpp:45 | Magic Number | Bloater | Should Fix (50) | Replace Magic Number: `constexpr int MAX_RETRIES = 42;` |
```

## Quick Reference: Smell-to-Fix Map

### Bloaters

| Smell | Primary Fix | Alternative Fixes |
| ------- | ------------- | ------------------- |
| Long Method | Extract Method | Decompose Conditional, Replace Temp with Query, Replace Method with Method Object |
| Large Class | Extract Class | Extract Subclass, Extract Interface |
| Primitive Obsession | Replace Data Value with Object | Replace Type Code with Class/Subclasses, Introduce Parameter Object |
| Long Parameter List | Introduce Parameter Object | Preserve Whole Object, Replace Parameter with Method Call |
| Data Clumps | Extract Class | Introduce Parameter Object, Preserve Whole Object |

### Object-Orientation Abusers

| Smell | Primary Fix | Alternative Fixes |
| ------- | ------------- | ------------------- |
| Switch Statements | Replace Conditional with Polymorphism | Replace Type Code with Subclasses/State/Strategy |
| Temporary Field | Extract Class | Replace Method with Method Object, Introduce Null Object |
| Refused Bequest | Replace Inheritance with Delegation | Extract Superclass |
| Alt Classes w/ Different Interfaces | Rename Method | Move Method, Extract Superclass |

### Change Preventers

| Smell | Primary Fix | Alternative Fixes |
| ------- | ------------- | ------------------- |
| Divergent Change | Extract Class | Extract Superclass/Subclass |
| Shotgun Surgery | Move Method/Field | Inline Class |
| Parallel Inheritance Hierarchies | Move Method/Field | Collapse one hierarchy |

### Dispensables

| Smell | Primary Fix | Alternative Fixes |
| ------- | ------------- | ------------------- |
| Comments (what, not why) | Extract Method / Rename Method | Extract Variable, Introduce Assertion |
| Duplicate Code | Extract Method | Pull Up Method, Form Template Method, Extract Superclass |
| Lazy Class | Inline Class | Collapse Hierarchy |
| Data Class | Move Method (into class) | Encapsulate Field/Collection |
| Dead Code | Delete | Remove Parameter, Inline Class |
| Speculative Generality | Collapse Hierarchy | Inline Class/Method, Remove Parameter |

### Couplers

| Smell | Primary Fix | Alternative Fixes |
| ------- | ------------- | ------------------- |
| Feature Envy | Move Method | Extract Method (move envious part) |
| Inappropriate Intimacy | Move Method/Field | Hide Delegate, Extract Class |
| Message Chains | Hide Delegate | Extract Method + Move Method |
| Middle Man | Remove Middle Man | -- |
| Incomplete Library Class | Introduce Foreign Method | Introduce Local Extension |

## Workflow

1. **Scan** -- Read changed/target files and identify smells using thresholds above
2. **Classify** -- Assign category, severity score, and specific smell name
3. **Prescribe** -- Look up fix in the smell-to-fix map; consult `catalog.md` for step-by-step technique
4. **Report** -- Output structured findings table
5. **Apply** (if requested) -- Follow the technique steps from `catalog.md` to implement the fix

## Integration

| After finding... | Next step |
| ------------------ | ----------- |
| Smells during PR review | Continue `pr-review` with findings |
| Smells needing a plan | Invoke `planning-refactor` |
| Need to understand code first | Invoke `exploration-explore-code` |
| C++ specific patterns | Cross-check with `programming-cpp` |
| Tests needed after refactoring | Use `testing-gtest-gmock` or `testing-pytest` |

## References

- Full catalog: `catalog.md` (co-located)
- [Refactoring.Guru - Code Smells](https://refactoring.guru/refactoring/smells)
- [Refactoring.Guru - Techniques](https://refactoring.guru/refactoring/techniques)
- Martin Fowler, *Refactoring: Improving the Design of Existing Code*
