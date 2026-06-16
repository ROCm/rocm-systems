---
name: brainstorming
description: "Use before any creative work — designing a new amdsmi_* API, adding a CLI command, building a feature, or modifying behavior. Explores intent, requirements, and design before any code is written."
---

# Brainstorming — amd-smi

Turn ideas into approved designs through structured dialogue. Surface assumptions, propose alternatives, write the spec. **No code until the spec is approved.**

<HARD-GATE>
Do NOT invoke any implementation skill, edit any source file, or create any wrapper/CLI binding until you have presented a design and the user has explicitly approved it. "Simple" changes are where unexamined assumptions cause the most rework — every task gets a design, even if the design is three sentences.
</HARD-GATE>

## Checklist

Complete in order. Use the todo tool to track each step.

1. **Explore project context** — read relevant files, check recent commits in the affected area, identify cascade layers touched (see `project-layout` rule)
2. **Ask clarifying questions one at a time** — purpose, constraints, success criteria. Prefer multiple-choice. Never bundle multiple questions in one message.
3. **Propose 2–3 approaches** with tradeoffs and your recommendation
4. **Present the design** in sections scaled to complexity; get approval after each section
5. **Write the spec** to `docs/dev/specs/YYYY-MM-DD-<topic>-design.md` and commit
6. **Self-review the spec** — placeholder scan, internal consistency, scope, ambiguity. Fix inline.
7. **User reviews the written spec** — wait for approval before proceeding
8. **Hand off to `writing-plans`** — that is the ONLY next skill

## What Every amd-smi Spec Must Cover

Beyond the generic design sections, an amd-smi spec must address:

| Section | Why It Matters |
|---------|----------------|
| **API cascade impact** | Which layers change: header → `amd_smi.cc` → wrapper → interface → CLI → docs |
| **Backward compatibility** | Public C API is contract — additions safe, signature changes are breaking |
| **Error path** | New `amdsmi_status_t` values? Mapping to `AmdSmiException`? |
| **Test plan** | C++ GTest, Python unit, Python integration, CLI — which suites need new tests |
| **Hardware dependency** | Requires GPU? Specific ASIC? Document the test environment |
| **Build/packaging impact** | New CMake option? New file in install set? RPM + DEB both? |
| **Changelog entry** | What goes in `CHANGELOG.md` under the next version |

## Scope Check

If the request touches multiple independent subsystems (e.g., new metric + new CLI subcommand + new logging mode), flag this immediately. Decompose into independent sub-projects, each with its own spec → plan → implementation cycle. Don't refine details of a feature that needs splitting first.

## Working in Existing Code

- Follow established patterns. Read three similar implementations before proposing a new one.
- If a file you must touch has grown unwieldy (1000+ lines, mixed responsibilities), include a targeted refactor in the design — the way a careful engineer improves code they're already in.
- Don't propose unrelated refactoring. Stay focused.

## Design Principles

- **One question at a time** — never bundle
- **YAGNI ruthlessly** — strip speculative features from the design
- **Explore alternatives** — propose 2–3 approaches before settling
- **Smaller units > flexibility flags** — if you find yourself adding "configurability that wasn't requested", split into smaller files instead
- **Incremental approval** — section-by-section, not all-at-once

## Spec Self-Review (Inline, No Subagent)

Before asking the user to review, look at the spec with fresh eyes:

1. **Placeholders** — any `TBD`, `TODO`, vague requirements? Fix them.
2. **Internal consistency** — do sections contradict? Does architecture match feature descriptions?
3. **Scope** — single implementation plan, or needs decomposition?
4. **Ambiguity** — could any requirement be read two ways? Pick one, make it explicit.
5. **Cascade completeness** — every layer in the cascade addressed?

Fix inline. Move on.

## User Review Gate

After self-review:

> "Spec written and committed to `<path>`. Please review it and let me know if you want changes before I write the implementation plan."

Wait for explicit approval. If changes requested, fix and re-run self-review. Only proceed once approved.

## Red Flags — STOP and Re-Brainstorm

- "This is too simple to need a design"
- Jumping to "what file should I edit"
- Bundling multiple questions
- Proposing one approach without alternatives
- Skipping the user-review gate

**All of these mean: stop, return to the checklist.**

## After Approval

Invoke the `writing-plans` skill. Do NOT invoke any other skill from here.
