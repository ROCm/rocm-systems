---
name: planning-architecture
description: Plan new or redesigned system architecture - define components, boundaries, interactions, and technical decisions before implementation
---

# Architecture Planning

Use this skill when designing NEW architecture or planning REDESIGN of existing architecture.

<IMPORTANT>
**Prerequisites:** Invoke `planning-base` skill first if not already loaded. It provides the core planning phases (0-5).

Follow all base planning rules, plus the architecture-specific rules below.

**Mandatory:** Invoke `programming-cpp-design-patterns` during Phase 4 (detailing the architecture). Consider which patterns apply to the component interactions and extension points being designed.

**Key distinction:**

- `planning-architecture` = design what WILL BE built (this skill)
- `review-architecture` = document what already EXISTS
</IMPORTANT>

## When to Use

- Designing a new system, module, or component from scratch
- Redesigning or migrating existing architecture
- Making decisions that affect multiple components or modules
- Choosing between architectural patterns or technologies
- Defining system boundaries, APIs, or integration points

## Phase 1: Determine Scope

Use `AskUserQuestion` to ask:

> "What kind of architecture work is this?"

- **Greenfield** - Design a new system or component from scratch
- **Redesign** - Change the architecture of something that already exists

Wait for answer before proceeding.

### If Redesign: Understand Current State

Before designing the new architecture, understand what exists:

1. Invoke `review-architecture` or `exploration-explore-code` to analyze the current system
2. Identify what works well (keep), what is broken (fix), and what is missing (add)
3. Document the constraints: what cannot change (APIs, data formats, external dependencies)

## Phase 2: Gather Requirements

Identify and confirm with the user:

| Category | Questions |
| ---------- | ----------- |
| **Functional** | What must the system do? What are the core capabilities? |
| **Non-functional** | Performance targets, scalability needs, reliability requirements |
| **Performance** | Is this on a hot path? How frequently is it called? Is latency critical? |
| **Constraints** | Technology stack, team expertise, timeline, budget, legacy compatibility |
| **Integration** | What external systems must it connect to? What protocols/formats? |

### Performance-Sensitive Design

If the system is on a hot path or performance-critical, this changes the polymorphism strategy:

| Technique | Overhead | When to Use |
| ----------- | ---------- | ------------- |
| **Virtual interfaces** | vtable indirection, heap allocation, cache-unfriendly | Rare calls, set of types unknown at compile time |
| **std::variant + std::visit** | Zero indirection, stack-allocated, cache-friendly | Fixed set of known types (preferred for hot paths) |
| **Templates / Policy-based design** | Zero overhead, resolved at compile time | Types known at compile time, maximum performance |
| **CRTP** | Zero overhead, compile-time polymorphism | Static dispatch with base class behavior |

<IMPORTANT>
Always ask whether the code is on a hot path. If yes, prefer compile-time polymorphism (templates, std::variant, CRTP) over virtual interfaces. Virtual dispatch has vtable indirection and prevents inlining — avoid it in performance-critical code.
</IMPORTANT>

Present findings and ask: **"Are these requirements complete, or should I add/change anything?"**

Wait for answer before proceeding.

## Phase 3: Propose Alternatives

<IMPORTANT>
ALWAYS present at least 2 architectural alternatives. Never propose a single solution without showing what was considered and rejected.
</IMPORTANT>

For each alternative, describe:

| Aspect | What to Cover |
| -------- | --------------- |
| **Overview** | High-level description of the approach |
| **Components** | What modules/services/layers exist |
| **Interactions** | How components communicate |
| **Trade-offs** | Pros and cons relative to requirements |
| **Risk** | What could go wrong, what is hard to change later |
| **Effort** | Relative complexity of implementation |

Present alternatives in a comparison table:

```markdown
## Architectural Alternatives

### Alternative A: [Name]
[Description]

### Alternative B: [Name]
[Description]

### Comparison

| Criterion | Alternative A | Alternative B |
| ----------- | --------------- | --------------- |
| Meets functional requirements | Yes/Partial/No | Yes/Partial/No |
| Performance | [assessment] | [assessment] |
| Complexity | [Low/Medium/High] | [Low/Medium/High] |
| Extensibility | [assessment] | [assessment] |
| Risk | [assessment] | [assessment] |
```

Ask: **"Which direction do you prefer? Or should I explore a different approach?"**

Wait for answer before proceeding.

## Phase 4: Detail the Chosen Architecture

After the user picks a direction, flesh out the design. Cover these sections one at a time, getting user approval between each:

### 4.1 Class Diagram

Generate a Mermaid class diagram showing all components, their interfaces, relationships, and ownership. Present it for user approval before continuing.

### 4.2 Component Breakdown

Define each component:

- Responsibility (what it does and does NOT do)
- Public interface (what it exposes to other components)
- Internal structure (sub-components if any)
- Ownership boundaries (what data/state it owns)

### 4.3 Interactions and Data Flow

For each pair of interacting components:

- Direction of communication (who calls whom)
- Protocol or mechanism (function call, message queue, API, event)
- Data exchanged (types, formats)
- Error handling (what happens when communication fails)

### 4.4 Extension Points

Where and how the system can be extended:

- Plugin interfaces or abstract base classes
- Configuration-driven behavior
- Feature flags or conditional compilation
- Module boundaries designed for future expansion

### 4.5 Migration Plan (Redesign Only)

If redesigning existing architecture:

- Incremental migration steps (never big-bang)
- What can be changed independently
- Backward compatibility during transition
- Rollback strategy for each step
- Data migration approach if applicable

<IMPORTANT>
Present each section (4.1-4.5) individually. Wait for user approval before moving to the next section. Revise if the user requests changes.
</IMPORTANT>

## Phase 5: Risk Analysis

Identify architectural risks:

| Risk | Impact | Likelihood | Mitigation |
| ------ | -------- | ------------ | ------------ |
| [risk description] | [High/Medium/Low] | [High/Medium/Low] | [how to address] |

Focus on:

- Decisions that are hard to reverse later
- Single points of failure
- Scalability bottlenecks
- Coupling that will make future changes expensive
- Technology bets that may not pay off

## Phase 6: Create Implementation Roadmap

Break the architecture into implementable work items:

1. **Foundation first** - Core abstractions, interfaces, base components
2. **Critical path next** - The most important or risky pieces
3. **Incremental delivery** - Each step produces something usable or testable
4. **Tests alongside** - Each component should be testable in isolation

Map work items to PRs following `planning-base` PR size guidelines.

## Plan File Format

Save to `planning/architecture-<name>.md`:

```markdown
# Architecture: <System/Component Name>

## Goal
<What is being designed and why>

## Type
<Greenfield / Redesign>

## Requirements
### Functional
- <Requirement 1>
- <Requirement 2>

### Non-Functional
- <Requirement 1>

### Constraints
- <Constraint 1>

## Alternatives Considered

### Alternative A: <Name>
<Summary and why chosen/rejected>

### Alternative B: <Name>
<Summary and why chosen/rejected>

**Chosen:** Alternative <X> because <reason>

## Architecture Design

### Components
| Component | Responsibility | Interface |
| ----------- | --------------- | ----------- |
| <name> | <what it does> | <what it exposes> |

### Interactions
| From | To | Mechanism | Data |
| ------ | ---- | ----------- | ------ |
| <component> | <component> | <how> | <what> |

### Extension Points
- <Where and how the system can be extended>

## Migration Plan (if redesign)
- [ ] Step 1: <description>
- [ ] Step 2: <description>

## Risks
| Risk | Impact | Mitigation |
| ------ | -------- | ------------ |
| <risk> | <level> | <how> |

## Implementation Roadmap
### PR 1: <Title>
**Scope:** <What's included>
- [ ] Task 1
- [ ] Task 2

### PR 2: <Title>
**Scope:** <What's included>
- [ ] Task 1
- [ ] Task 2

## Notes
<Decisions, trade-offs, open questions>
```

## Common Mistakes

| Mistake | Fix |
| --------- | ----- |
| Jumping to a single solution without alternatives | Always present at least 2 alternatives with trade-offs |
| Designing without understanding requirements | Gather and confirm requirements first |
| Big-bang migration plan | Break redesign into incremental steps with rollback |
| Over-engineering for hypothetical future needs | Design for known requirements, add extension points only where likely |
| Detailing all sections at once | Present one section at a time, get approval between each |
| Skipping risk analysis | Hard-to-reverse decisions need explicit risk assessment |
| Ignoring current state in redesign | Always analyze what exists before proposing changes |

## Rationalizations to Reject

| Excuse | Reality |
| -------- | --------- |
| "The answer is obvious, no need for alternatives" | Even obvious choices benefit from documenting what was rejected and why |
| "We'll figure out migration later" | Migration is the hardest part of redesign. Plan it upfront. |
| "Requirements are clear from context" | Confirm anyway. Assumptions are the root of architectural drift. |
| "It's a small system, no need for formal planning" | Small systems grow. Early decisions are the hardest to change later. |

## Integration with Other Skills

| Before This Skill | Use |
| ------------------- | ----- |
| Understanding existing code | `exploration-explore-code` |
| Documenting current architecture | `review-architecture` |

| After This Skill | Use |
| ------------------ | ----- |
| Implementing planned features | `planning-feature` |
| Refactoring toward new architecture | `planning-refactor` |
| Creating test plans | `testing-testplan` |
