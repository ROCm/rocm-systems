# Output Conventions

Code-quality conventions for anything written into the rocr-runtime repository
(comments, doxygen, help text, docs). For commit and PR wording, follow standard
git conventions for the rocr-runtime project.

## Assume Expert Readers

Write for a developer already familiar with HSA and low-level GPU programming.
Do not explain what the code plainly says or restate C/C++ basics.

## Comment Brevity

- Comment the **why**, not the **what**. If the code is readable, it needs no
  narration.
- Explain only a non-obvious root cause or a decision that isn't visible in the
  code.
- One or two lines is the norm. If a comment needs a full paragraph, that is a
  signal to stop and reconsider, or to ask before writing it.
- Never let comments outweigh the code they describe.

## No Ticket References in Code

- Do not put `SWDEV-NNNNNN`, `JIRA-NNN`, or other ticket IDs in code comments.
- Do not put internal labels ("regression guard", sprint names) in code.
- Reference the behavior or root cause. Tickets belong in commits/PRs only.

## HSA API Documentation (Doxygen)

- All public HSA API functions **must** have doxygen comments
- Document all parameters, return values, and error codes
- Document thread-safety and memory ownership
- Document HSA spec version requirements for new APIs
- Keep doxygen comments concise — focus on contracts, not implementation

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Paragraph-length comment explaining how the code works | One line on the root-cause why, or none |
| `// Fixes SWDEV-12345` above a change | Drop it; describe the behavior |
| Comment restating the next line in English | Delete it |
| Tutorial-style doxygen for an obvious function | State the contract, nothing more |
