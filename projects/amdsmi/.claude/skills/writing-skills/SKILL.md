---
name: writing-skills
description: "Use when creating or editing a SKILL.md file under .claude/skills/. Defines the TDD-for-documentation discipline, structure, and CSO rules that every amd-smi skill must follow."
---

# Writing Skills — amd-smi

Skills are reusable reference guides agents load on demand. They are NOT narratives, NOT one-off notes, NOT project-specific runbooks (those go in `CLAUDE.md` or repo memory).

**Iron Law:** NO SKILL WITHOUT A FAILING PRESSURE TEST FIRST.

If you have not watched an agent fail without the skill, you do not know what the skill needs to teach. Same RED-GREEN-REFACTOR cycle as TDD, applied to documentation.

## When to Create a Skill

| Create when | Don't create when |
|-------------|-------------------|
| Technique applies across multiple amd-smi tasks | One-off solution |
| Same mistake keeps recurring across sessions | Standard practice already in `CLAUDE.md` |
| Pattern needs judgment (not enforceable by lint/regex) | Mechanical rule — automate it instead |
| Captures hard-won amd-smi-specific knowledge | Pure VS Code/git/bash trivia |

## Skill Types

- **Technique** — concrete steps (e.g., `amdsmi-build-install`, `systematic-debugging`)
- **Discipline** — enforces a rule under pressure (e.g., `test-driven-development`, `verification-before-completion`)
- **Reference** — lookup tables, API mappings, command catalogs (e.g., `personal-bash-deploy`)

## File Layout

```
.claude/skills/<skill-name>/
  SKILL.md          # required
  <supporting>.md   # only for heavy reference (100+ lines) or reusable scripts
```

**Naming:** `kebab-case`, verb-first or gerund preferred. Prefix with `amdsmi-` only if the skill is amd-smi-domain-specific (build, test runner, packaging). Generic workflow skills (TDD, debugging, planning) stay unprefixed.

## SKILL.md Template

```markdown
---
name: skill-name
description: "Use when [specific triggering symptoms]. [What problem it addresses]."
---

# Skill Name

[1-2 sentence purpose. Core principle.]

## When to Use

- [Concrete symptom or trigger]
- [Another trigger]

**Don't use when:** [counter-cases]

## [Core content — table, checklist, or step-by-step]

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| ... | ... |
```

## Description Field Rules (CSO — Critical for Discovery)

The `description` is what determines whether an agent loads the skill. Get it wrong and the skill is invisible.

| ❌ Bad | ✅ Good |
|-------|--------|
| `"For TDD work"` (vague) | `"Use when implementing any feature or bugfix, before writing implementation code"` |
| `"Write test → watch fail → minimal code → refactor"` (summarizes workflow — agent will skip the body) | `"Use when implementing any feature or bugfix, before writing implementation code"` |
| `"I help with debugging"` (first person) | `"Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes"` |
| `"For async tests"` (no symptoms) | `"Use when tests have race conditions, timing dependencies, or pass/fail inconsistently"` |

**Rules:**
- Start with `"Use when ..."`
- Third person, ≤ 500 chars
- Describe **triggers/symptoms**, NOT the workflow
- Include error-message keywords, tool names, file patterns agents would search for
- If amd-smi-specific, say so explicitly

## Cross-Referencing Other Skills

```
**REQUIRED SUB-SKILL:** Use the `test-driven-development` skill before implementation.
**REQUIRED BACKGROUND:** You MUST understand `systematic-debugging` before using this.
```

Never `@`-link — that force-loads the file and burns context.

## Token Budget

- Frequently-loaded skills: < 200 words
- Other skills: < 500 words
- Heavy reference: separate file, linked by relative path

Run `wc -w SKILL.md` before committing.

## Anti-Patterns

| Anti-Pattern | Why Bad |
|--------------|---------|
| Narrative ("In session 2026-04-12 we found...") | Not reusable |
| Multi-language examples (JS + Python + Go) | Maintenance burden, dilutes signal |
| Restating project conventions already in `CLAUDE.md` | Duplication, drift |
| Workflow summary in description | Agent follows description, skips body |
| `applyTo` frontmatter on a skill | Skills are on-demand — `applyTo` belongs on `.claude/rules/*.md` and `.github/instructions/*.md` |

## Pressure-Test Before Committing

1. **RED:** Give a fresh agent the trigger scenario without the skill. Record what it does wrong and the rationalizations it uses verbatim.
2. **GREEN:** Write the minimal skill addressing those specific failures. Re-run — agent should now comply.
3. **REFACTOR:** Find new rationalizations, add explicit counters (rationalization table, red-flags list). Repeat until bulletproof.

## Discipline-Skill Hardening

If the skill enforces a rule (TDD, verification, root-cause-first), add:

- **The Iron Law** — single bold rule at the top
- **Rationalization table** — every excuse you've seen → reality counter
- **Red Flags list** — phrases that mean "STOP, you're rationalizing"
- **"Violating the letter is violating the spirit"** — cuts off the spirit-vs-letter loophole

## Skill Creation Checklist

- [ ] Ran pressure test without skill — documented baseline failure
- [ ] `name` is kebab-case, no special chars
- [ ] `description` starts with "Use when", lists symptoms, no workflow summary
- [ ] Single excellent example (not multi-language)
- [ ] Rationalization table (if discipline skill)
- [ ] `wc -w` under target
- [ ] No duplication with `CLAUDE.md` or other skills
- [ ] Re-ran pressure test with skill — agent complies
