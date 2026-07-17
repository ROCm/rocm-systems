---
name: code-smells
description: Detect code smells based on refactoring.guru catalog - use when analyzing code for anti-patterns and refactoring opportunities
---

# Code Smells Detection

Comprehensive catalog of code smells for detecting anti-patterns and identifying refactoring opportunities. Based on the [Refactoring.Guru Code Smells Catalog](https://refactoring.guru/refactoring/smells).

<IMPORTANT>
This skill is a **reference catalog** for code smell detection. It is used by:
- `pr-review` Agent 3 (Code Smells Agent) during PR review
- `planning-refactor` when identifying improvement opportunities
- Standalone code quality analysis

When detecting smells, report findings in structured format with:

- File:Line location
- Smell name and category
- Severity score
- Suggested refactoring
</IMPORTANT>

## When to Use

| Context | Trigger |
| --------- | --------- |
| **PR Review** | Invoked by `pr-review` Code Smells Agent |
| **Refactoring Planning** | Before `planning-refactor` to identify targets |
| **Code Quality Audit** | User asks to analyze code quality or find anti-patterns |
| **Technical Debt Assessment** | Identifying areas needing improvement |

## Severity Levels

| Severity | Score | Impact | Examples |
| ---------- | ------- | -------- | ---------- |
| **Critical** | 100 | Causes bugs, crashes, security issues | Feature Envy causing null dereference |
| **Must Fix** | 80 | Significantly harms maintainability | God Class, Shotgun Surgery |
| **Should Fix** | 50 | Reduces code quality | Long Method, Duplicate Code |
| **Nitpick** | 20 | Minor improvement opportunity | Lazy Class, Comments |

## Output Format

When reporting code smells, use this structured format:

```markdown
| File:Line | Smell | Category | Severity | Suggested Refactoring |
| ----------- | ------- | ---------- | ---------- | ---------------------- |
| handler.cpp:120-195 | Long Method (75 lines) | Bloater | Should Fix (50) | Extract Method: split into validateRequest(), processData(), buildResponse() |
| config.cpp:45 | Magic Number | Bloater | Should Fix (50) | Replace Magic Number: `const int MAX_RETRIES = 42;` |
```

---

## Category 1: Bloaters

Code that has grown to excessive proportions, making it hard to work with.

### Long Method

**Threshold:** >10 lines warrants questions, >50 lines is definite smell

**Signs:**

- Method contains too many lines of code
- You need to scroll to see the whole method
- Method does multiple things

**Why it's bad:**

- Hard to understand and maintain
- Hides duplicate code
- Makes testing difficult
- Long methods accumulate like "Hotel California" - easy to add, hard to remove

**Detection criteria:**

```text
- Lines > 50: Should Fix (50)
- Lines > 100: Must Fix (80)
- Method does more than one thing (multiple responsibilities)
- Requires extensive comments to explain
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Extract Method | Split into smaller methods with descriptive names |
| Replace Temp with Query | Local variables interfering with extraction |
| Introduce Parameter Object | Many parameters passed between methods |
| Decompose Conditional | Complex if/else chains |
| Replace Method with Method Object | When extraction is too complex |

---

### Large Class

**Threshold:** Too many fields, methods, or lines (>500 lines warrants review)

**Signs:**

- Class contains excessive fields/methods/lines
- Class has multiple unrelated responsibilities
- You struggle to summarize what the class does in one sentence

**Why it's bad:**

- Cognitive overload (too many attributes to track)
- Code duplication within the class
- Violates Single Responsibility Principle

**Detection criteria:**

```text
- Lines > 500: Should Fix (50)
- Lines > 1000: Must Fix (80)
- >10 public methods: Review for SRP violation
- >15 fields: Likely doing too much
- Multiple unrelated method groups
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Extract Class | Separate distinct behavioral components |
| Extract Subclass | Behavior used only in some cases |
| Extract Interface | Define contract for operations |
| Duplicate Observed Data | GUI classes with domain logic |

---

### Primitive Obsession

**Signs:**

- Using primitives instead of small objects (money, phone numbers, ranges)
- Constants encoding information (`USER_ADMIN_ROLE = 1`)
- String constants as field names in arrays
- Using `int` for IDs instead of typed wrapper

**Why it's bad:**

- No type safety (passing wrong int to function)
- Related operations scattered across codebase
- No place to add validation or behavior
- Duplicate code for handling the primitive

**Detection criteria:**

```text
- Multiple functions operating on same primitive group: Should Fix (50)
- Constants simulating types: Should Fix (50)
- No domain objects for business concepts (money, date ranges, coordinates): Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Replace Data Value with Object | Group primitives into dedicated classes |
| Introduce Parameter Object | Primitives in method parameters |
| Preserve Whole Object | Passing extracted values instead of object |
| Replace Type Code with Class/Subclasses/State | Constants simulating types |
| Replace Array with Object | Using arrays with string keys |

---

### Long Parameter List

**Threshold:** >3-4 parameters

**Signs:**

- Method has more than 3-4 parameters
- Parameters are passed through multiple method calls
- Hard to remember parameter order
- Boolean parameters for mode switching

**Why it's bad:**

- Difficult to understand method signature
- Easy to pass wrong arguments
- Often indicates the method is doing too much
- Creates tight coupling

**Detection criteria:**

```text
- 4-5 parameters: Should Fix (50)
- 6+ parameters: Must Fix (80)
- Boolean "flag" parameters: Should Fix (50)
- Parameters from same object: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Replace Parameter with Method Call | Parameter is result of another call |
| Preserve Whole Object | Passing multiple fields from same object |
| Introduce Parameter Object | Group related parameters |

**When to ignore:** Creating dependencies between classes may be worse than long parameter list.

---

### Data Clumps

**Signs:**

- Same group of variables appears in multiple places
- Parameters that always travel together
- Removing one variable makes others meaningless

**Why it's bad:**

- Code duplication
- No single place for operations on the data
- Easy to pass incomplete groups

**Detection criteria:**

```text
- Same 3+ parameters in multiple method signatures: Should Fix (50)
- Same fields grouped in multiple classes: Should Fix (50)
- Database connection parameters passed separately: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Extract Class | For class fields |
| Introduce Parameter Object | For method parameters |
| Preserve Whole Object | Pass object instead of extracting values |

---

## Category 2: Object-Orientation Abusers

Incomplete or incorrect application of object-oriented programming principles.

### Switch Statements

**Signs:**

- Complex `switch` operator or `if-else` chain based on type
- Same switch logic scattered across multiple methods
- Adding a new case requires changes in many places

**Why it's bad:**

- Violates Open/Closed Principle
- Duplicate switch logic across codebase
- Easy to forget updating all switches
- Rare use of `switch` is hallmark of good OOP

**Detection criteria:**

```text
- Switch on type/enum with behavior: Should Fix (50)
- Same switch in multiple places: Must Fix (80)
- Switch with >5 cases with different behavior: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Replace Conditional with Polymorphism | When switch determines behavior |
| Replace Type Code with Subclasses | Type codes with distinct behavior |
| Replace Type Code with State/Strategy | Behavior varies at runtime |
| Replace Parameter with Explicit Methods | Simple value-based switches |
| Introduce Null Object | Null checks in conditionals |

**When to ignore:** Factory patterns legitimately use switch to create objects.

---

### Temporary Field

**Signs:**

- Fields only used in certain circumstances
- Fields remain empty/null most of the time
- Field existence confuses readers

**Why it's bad:**

- Developers expect fields to always be meaningful
- Requires null checks scattered through code
- Hides actual dependencies

**Detection criteria:**

```text
- Field used in <25% of methods: Should Fix (50)
- Field only set in one method and used in another: Should Fix (50)
- Fields for algorithm-specific data: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Extract Class | Move field and related code to separate class |
| Replace Method with Method Object | Algorithm needs many temp fields |
| Introduce Null Object | Replace null checks with null object pattern |

---

### Refused Bequest

**Signs:**

- Subclass uses only some parent methods/properties
- Inherited methods throw exceptions or do nothing
- Subclass doesn't follow Liskov Substitution Principle

**Why it's bad:**

- Confusing hierarchy (Dog inherits from Chair?)
- Breaking substitutability
- Indicates wrong abstraction

**Detection criteria:**

```text
- Override methods with empty body or exception: Should Fix (50)
- Subclass ignores >50% of parent interface: Should Fix (50)
- Inheritance for code reuse, not "is-a": Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Replace Inheritance with Delegation | Wrong hierarchy, use composition |
| Extract Superclass | Create proper abstraction for shared behavior |

---

### Alternative Classes with Different Interfaces

**Signs:**

- Two classes do the same thing with different method names
- Parallel implementations that could be unified
- Duplicate functionality discovered during review

**Why it's bad:**

- Code duplication across classes
- Confusion about which class to use
- Maintenance burden

**Detection criteria:**

```text
- Classes with same purpose, different interface: Should Fix (50)
- Duplicate logic in differently-named methods: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Rename Method | Align method names to common interface |
| Move Method | Consolidate implementations |
| Extract Superclass | Share common behavior |
| Delete redundant class | After consolidation |

**When to ignore:** Classes in different libraries with independent versioning.

---

## Category 3: Change Preventers

Issues that require changes in multiple places when a single change is needed.

### Divergent Change

**Signs:**

- Single class needs changes for multiple unrelated reasons
- Adding new feature requires changing many methods in one class
- Class has multiple "areas" of responsibility

**Why it's bad:**

- Single class handles too many concerns
- Any change risks breaking unrelated functionality
- Violates Single Responsibility Principle

**Detection criteria:**

```text
- Class changes for >2 unrelated reasons: Should Fix (50)
- Class has distinct "sections" of functionality: Should Fix (50)
- Methods naturally group into separate concerns: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Extract Class | Separate each concern into its own class |
| Extract Superclass/Subclass | When classes share some behavior |

---

### Shotgun Surgery

**Signs:**

- Single change requires edits to many different classes
- Adding a feature means touching 5+ files
- Opposite of Divergent Change

**Why it's bad:**

- Easy to miss a required change
- High chance of introducing bugs
- Expensive maintenance

**Detection criteria:**

```text
- Feature addition touches 5+ classes: Must Fix (80)
- Single responsibility scattered across classes: Must Fix (80)
- "Overzealous" previous separation causing this: Must Fix (80)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Move Method/Field | Consolidate scattered responsibility |
| Inline Class | After consolidation, remove empty shells |

---

### Parallel Inheritance Hierarchies

**Signs:**

- Creating subclass in one hierarchy requires subclass in another
- Class prefixes match across hierarchies
- Mirrored class structures

**Why it's bad:**

- Duplicate maintenance
- Growing complexity as hierarchies expand
- Easy to forget creating matching class

**Detection criteria:**

```text
- Two hierarchies with matching structures: Should Fix (50)
- Adding to one requires adding to another: Should Fix (50)
- Prefixes like "XHandler/XFactory": Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Move Method/Field | Make one hierarchy reference the other |
| Eliminate redundant hierarchy | After consolidation |

**When to ignore:** If deduplication makes code uglier, keep the parallel structure.

---

## Category 4: Dispensables

Pointless and unneeded elements whose absence would make code cleaner.

### Comments

**Signs:**

- Method filled with explanatory comments
- Comments describe WHAT instead of WHY
- Comments are outdated or incorrect
- Comments compensate for unclear code

**Why it's bad:**

- "The best comment is a good name for a method or class"
- Comments can become outdated and misleading
- Indicates code needs refactoring, not explaining

**Detection criteria:**

```text
- Comment explains what code does (not why): Nitpick (20)
- Commented-out code: Should Fix (50)
- Comment contradicts code: Must Fix (80)
- TODOs without ticket references: Nitpick (20)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Extract Variable | Complex expression needs explanation |
| Extract Method | Use comment text as method name |
| Rename Method | Method name should make comment unnecessary |
| Introduce Assertion | Document required state |

**When acceptable:** Comments explaining WHY (design decisions, complex algorithms, non-obvious constraints).

---

### Duplicate Code

**Signs:**

- Two code fragments look almost identical
- Same logic in different methods/classes
- Copy-paste programming

**Why it's bad:**

- Changes must be made in multiple places
- Easy to miss updating all copies
- Increases maintenance burden

**Detection criteria:**

```text
- Identical code blocks: Should Fix (50)
- Similar code blocks (>80% same): Should Fix (50)
- Copy-paste with minor modifications: Should Fix (50)
- Same algorithm implemented differently: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Extract Method | Within single class |
| Pull Up Method | In sibling subclasses |
| Form Template Method | Similar but not identical algorithms |
| Extract Superclass | Across unrelated classes |
| Extract Class | When hierarchy inappropriate |
| Consolidate Conditional Expression | Duplicate conditions |

---

### Lazy Class

**Signs:**

- Class doesn't do enough to justify existence
- Class was reduced by refactoring and is now minimal
- Class created for anticipated features that never came

**Why it's bad:**

- Every class costs maintenance
- Adds complexity without value
- Cognitive overhead for developers

**Detection criteria:**

```text
- Class with <3 methods: Nitpick (20)
- Class that just delegates to another: Nitpick (20)
- Subclass with minimal additions: Nitpick (20)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Inline Class | Merge into another class |
| Collapse Hierarchy | Merge minimal subclass with parent |

**When to ignore:** Class represents future expansion point.

---

### Data Class

**Signs:**

- Class contains only fields and getters/setters
- No behavior, just data container
- Other classes operate on its data

**Why it's bad:**

- Violates encapsulation
- Logic operating on data is scattered
- Can't enforce invariants

**Detection criteria:**

```text
- Class with only fields + accessors: Should Fix (50)
- No methods operating on own data: Should Fix (50)
- Other classes doing Feature Envy on this class: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Encapsulate Field | Make fields private |
| Encapsulate Collection | For collections |
| Move Method | Bring behavior from clients into class |
| Remove Setting Method | After class gains behavior |
| Hide Method | For overly permissive accessors |

---

### Dead Code

**Signs:**

- Variable, parameter, field, method, or class no longer used
- Code after return/throw/break
- Conditional branches that never execute

**Why it's bad:**

- Clutters codebase
- Confuses developers
- May be accidentally activated

**Detection criteria:**

```text
- Unused variable: Nitpick (20)
- Unused method/function: Should Fix (50)
- Unreachable code: Should Fix (50)
- Commented-out code: Should Fix (50)
- Unused class: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Delete | Remove unused code |
| Remove Parameter | Unused parameters |
| Inline Class/Collapse Hierarchy | Unused classes |

---

### Speculative Generality

**Signs:**

- Unused class, method, field, or parameter
- Hooks for future features that never came
- Overly abstract code for simple requirements

**Why it's bad:**

- "Just in case" code adds complexity
- YAGNI - You Aren't Gonna Need It
- Makes code harder to understand

**Detection criteria:**

```text
- Unused abstraction: Nitpick (20)
- Parameter only used in tests: Nitpick (20)
- Abstract class with single implementation: Nitpick (20)
- Methods that only delegate: Nitpick (20)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Collapse Hierarchy | Unnecessary abstract classes |
| Inline Class | Unnecessary delegation |
| Inline Method | Unused methods |
| Remove Parameter | Unused parameters |

**When to ignore:** Framework code where users may need the hooks.

---

## Category 5: Couplers

Issues involving excessive coupling between classes or excessive delegation.

### Feature Envy

**Signs:**

- Method accesses data of another object more than its own
- Method is more interested in another class
- Heavy use of another object's getters

**Why it's bad:**

- Logic is in wrong place
- Violates encapsulation
- Makes refactoring harder

**Detection criteria:**

```text
- Method uses >3 getters from another object: Should Fix (50)
- Method primarily operates on another class's data: Should Fix (50)
- After data move, operations didn't follow: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Move Method | Move entire method to target class |
| Extract Method | Move only envious part |

**When to ignore:** Strategy/Visitor patterns intentionally separate behavior from data.

---

### Inappropriate Intimacy

**Signs:**

- Class uses internal fields/methods of another class
- Two classes spend too much time delving into each other
- Bidirectional dependencies

**Why it's bad:**

- Tight coupling
- Hard to change one class without affecting the other
- Reduced reusability

**Detection criteria:**

```text
- Classes accessing each other's private/protected: Must Fix (80)
- Bidirectional association: Should Fix (50)
- Deep knowledge of another class's internals: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Move Method/Field | Put code where data lives |
| Extract Class | Separate shared parts |
| Hide Delegate | Reduce direct knowledge |
| Change Bidirectional to Unidirectional | Simplify association |
| Replace Delegation with Inheritance | When intimacy is with subclass |

---

### Message Chains

**Signs:**

- Series of calls like `a.b().c().d()`
- Client depends on navigation structure
- Changes to intermediate classes affect client

**Why it's bad:**

- Client coupled to entire chain structure
- Any chain change requires client updates
- Violates Law of Demeter

**Detection criteria:**

```text
- Chain of 3+ method calls: Should Fix (50)
- Chain navigation through multiple objects: Should Fix (50)
- "Train wreck" code: a.getB().getC().getD(): Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Hide Delegate | Shorten chain at each step |
| Extract Method + Move Method | Get what you need from chain start |

**Caution:** Overly aggressive hiding creates Middle Man smell.

---

### Middle Man

**Signs:**

- Class delegates most work to another class
- Class exists only to forward calls
- No substantial independent behavior

**Why it's bad:**

- Unnecessary indirection
- Adds complexity without value
- Often results from over-applying Hide Delegate

**Detection criteria:**

```text
- Class where >50% methods just delegate: Should Fix (50)
- Class with only pass-through methods: Should Fix (50)
- Gradual functionality migration left empty shell: Should Fix (50)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Remove Middle Man | Call delegate directly |

**When to ignore:**

- Proxy pattern (intentional delegation)
- Decorator pattern
- Hiding dependencies for testing

---

### Incomplete Library Class

**Signs:**

- Library doesn't meet your needs
- Can't modify library code
- Working around library limitations

**Why it's bad:**

- Workarounds scattered across codebase
- Duplicate solutions to same problem
- Fragile code depending on library internals

**Detection criteria:**

```text
- Multiple workarounds for same library limitation: Should Fix (50)
- Helper functions for library functionality: Nitpick (20)
```

**Refactoring:**

| Technique | When to Use |
| ----------- | ------------- |
| Introduce Foreign Method | Add few methods |
| Introduce Local Extension | Add substantial functionality |

---

## Quick Reference: Detection Thresholds

| Smell | Threshold | Severity |
| ------- | ----------- | ---------- |
| Long Method | >50 lines | Should Fix (50) |
| Long Method | >100 lines | Must Fix (80) |
| Large Class | >500 lines | Should Fix (50) |
| Large Class | >1000 lines | Must Fix (80) |
| Long Parameter List | >4 params | Should Fix (50) |
| Long Parameter List | >6 params | Must Fix (80) |
| Deep Nesting | >3 levels | Should Fix (50) |
| Deep Nesting | >5 levels | Must Fix (80) |
| Magic Numbers | Any unexplained literal | Should Fix (50) |
| Duplicate Code | >10 lines identical | Should Fix (50) |
| Feature Envy | >3 external getters | Should Fix (50) |
| Message Chain | >3 calls | Should Fix (50) |
| Shotgun Surgery | >5 classes touched | Must Fix (80) |
| God Class | Multiple responsibilities | Must Fix (80) |

## Integration with Other Skills

| Scenario | Next Skill |
| ---------- | ------------ |
| Found smells during PR review | Continue `pr-review` with findings |
| Found smells, need refactoring | Invoke `planning-refactor` |
| Need to understand code first | Invoke `exploration-explore-code` |
| Ready to apply specific technique | Invoke `refactoring-techniques` for step-by-step guidance |
| Ready to fix Long Method | Use Extract Method from `refactoring-techniques` |
| Ready to fix Feature Envy | Use Move Method from `refactoring-techniques` |

**Cross-Reference:** This skill identifies WHAT to fix. The `refactoring-techniques` skill shows HOW to fix it with 60+ techniques and code examples.

## References

- [Refactoring.Guru - Code Smells](https://refactoring.guru/refactoring/smells)
- [Refactoring: Improving the Design of Existing Code](https://martinfowler.com/books/refactoring.html) by Martin Fowler
