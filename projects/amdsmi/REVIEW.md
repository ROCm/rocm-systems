# amd-smi review rules

Apply these rules when reviewing changes to files under `projects/amdsmi/`.

## In-repo prose (code comments, doxygen, help text, docs, CHANGELOG.md)

- Flag ticket IDs such as `ROCM-12345`, `SWDEV-123456` or `AILITOOLS-123`. Tickets belong only in the JIRA ID section of the PR description.
- Flag internal labels such as "regression guard" or sprint names. Describe the behavior instead.
- Flag the word "utilize"; amd-smi prose uses "use".

## CHANGELOG.md

- Entries are user-facing. Flag internal function, class, member or variable names, enum constants, signal or abort details, and source-file internals. Public API and CLI names are fine.
- Each entry is a short bold summary plus at most one concise bullet.
