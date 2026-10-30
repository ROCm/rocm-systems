---
applyTo: "projects/amdsmi/**"
excludeAgent: "cloud-agent"
---

# amd-smi review rules

Apply these rules when reviewing changes to files under `projects/amdsmi/`. They add to your normal review: still check every change against the code for correctness and report those problems too.

## Check claims against the code first

- CHANGELOG.md: quote each new entry and check every claim against the diff and the current code. The entry must describe exactly the shipped behavior. Flag an entry that reports a fix, feature or test that the code does not contain.
- Comments, docs and help text must agree with the code. Flag contradictions and stale examples.

## In-repo prose (code comments, doxygen, help text, docs, CHANGELOG.md)

- Flag ticket IDs such as `ROCM-12345`, `SWDEV-123456` or `AILITOOLS-123`. Tickets belong only in the JIRA ID section of the PR description.
- Flag internal labels such as "regression guard" or sprint names. Describe the behavior instead.
- Flag the word "utilize"; amd-smi prose uses "use".

## CHANGELOG.md

- Entries are user-facing. Flag internal function, class, member or variable names, enum constants, signal or abort details, and source-file internals. Public API and CLI names are fine.
- Each entry is a short bold summary plus at most one concise bullet.
