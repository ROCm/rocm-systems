# AI Agent Guidelines — rocprofiler-compute

## Python Code Style

Read and follow **[`.ai/rules/python-style.md`](.ai/rules/python-style.md)** before
generating or modifying any Python code. These rules cover function design, naming,
nesting, and code organization.

## Ruff and Tooling

All code in `src/` must pass Ruff checks. Read **[`.ai/rules/ruff-tooling.md`](.ai/rules/ruff-tooling.md)**
for enforced rules including type annotations, f-strings, and `pathlib` usage.

## Git Workflows

For git (clone, fetch, pull, push): do **not** use classic GitHub PATs (`ghp_*`,
tokens embedded in remote URLs, or pasted credentials). Prefer **`gh auth login`**
so GitHub auth is handled via the CLI. Use fine-grained or environment-provided
tokens only when the host already supplies them—not as a substitute with classic PATs.

When asked to commit changes, follow **[`.ai/rules/commit-workflow.md`](.ai/rules/commit-workflow.md)**
for staging, commit message conventions, pre-commit hook handling, and branch safety.

When asked to create a pull request, follow **[`.ai/rules/pr-workflow.md`](.ai/rules/pr-workflow.md)**
for PR template inference, JIRA handling, formatting, and repo identification.
