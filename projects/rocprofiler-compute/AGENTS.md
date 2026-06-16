# AI Agent Guidelines — rocprofiler-compute

## README

**[`README.md`](README.md)** is the project's user-facing overview.

## Contributing

**[`CONTRIBUTING.md`](CONTRIBUTING.md)** is the project's contributor guide.

## Python Code Style

Read and follow **[`.ai/rules/python-style.md`](.ai/rules/python-style.md)** before
generating or modifying any Python code. These rules cover function design, naming,
nesting, and code organization.

## Ruff and Tooling

All code in `src/` must pass Ruff checks. Read **[`.ai/rules/ruff-tooling.md`](.ai/rules/ruff-tooling.md)**
for enforced rules including type annotations, f-strings, and `pathlib` usage.

## Git Workflows

Prefer the **`gh` CLI** for all GitHub interactions (pull requests, issues,
reviews, and authenticated git operations) over any MCP server or tool that
relies on classic PATs (`ghp_*`), tokens in remote URLs, or pasted credentials.
If `gh` is not authenticated, ask the user to run `gh auth login` rather than
supplying a token yourself.

When asked to commit changes, follow **[`.ai/rules/commit-workflow.md`](.ai/rules/commit-workflow.md)**
for staging, commit message conventions, pre-commit hook handling, and branch safety.

When asked to create a pull request, follow **[`.ai/rules/pr-workflow.md`](.ai/rules/pr-workflow.md)**
for PR template inference, JIRA handling, formatting, and repo identification.

## Skills

Reusable agent workflows live under **[`.ai/skills/`](.ai/skills/)** with
tool-specific shims in `.claude/commands/`, `.github/prompts/`, and
`.cursor/commands/`.
