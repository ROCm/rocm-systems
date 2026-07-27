---
description: "Use when: writing or editing code comments, doxygen, help text, or any in-repo prose in amd-smi; deciding how much to comment; reviewing comment verbosity; referencing JIRA tickets in code."
---
# Output Conventions

Code-quality conventions for anything written into the amd-smi repository
(comments, doxygen, help text, docs). For commit and PR wording, use the
`amdsmi-commit-and-pr-conventions` skill instead.

## Assume Expert Readers

Write for a developer already familiar with the codebase. Do not explain what
the code plainly says or restate language/library basics.

## Comment Brevity

- Comment the **why**, not the **what**. If the code is readable, it needs no
  narration.
- Explain only a non-obvious root cause or a decision that isn't visible in the
  code.
- One or two lines is the norm. If a comment needs a full paragraph, that is a
  signal to stop and reconsider, or to ask before writing it.
- Never let comments outweigh the code they describe.

## No Ticket References in Code

- Do not put `ROCM-NNNNN`, `SWDEV-NNNNNN`, or `AILITOOLS-NNN` in code comments.
- Do not put internal labels ("regression guard", sprint names) in code.
- Reference the behavior or root cause. Tickets belong in the PR `JIRA ID`
  section only (see `amdsmi-commit-and-pr-conventions`).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Paragraph-length comment explaining how the code works | One line on the root-cause why, or none |
| `// Fixes ROCM-12345` above a change | Drop it; describe the behavior |
| Comment restating the next line in English | Delete it |
| Tutorial-style doxygen for an obvious getter | State the contract, nothing more |
