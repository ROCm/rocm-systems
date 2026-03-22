# Forbidden patterns

- Rewriting entire files or large unrelated regions “for clarity.”
- Generic refactors (“make it more flexible,” new abstraction layers) without a stated need and approval.
- Renaming widely used symbols, CLI flags, or CSV/column semantics without migration plan and tests.
- Adding dependencies to work around a small local problem.
- Silent behavior changes in profiling, analysis, or roofline output without tests and changelog consideration.
- Nondeterministic tests (wall clock, unordered iteration over unordered sets for golden output).
