# AI Agent Guidelines — rocprofiler-compute

## Python Code Style

Read and follow **[`.ai/rules/python-style.md`](.ai/rules/python-style.md)** before
generating or modifying any Python code. These rules cover function design, naming,
nesting, and code organization.

## Ruff and Tooling

All code in `src/` must pass Ruff checks. Read **[`.ai/rules/ruff-tooling.md`](.ai/rules/ruff-tooling.md)**
for enforced rules including type annotations, f-strings, and `pathlib` usage.

## Git Workflows

When asked to commit changes, follow **[`.ai/rules/commit-workflow.md`](.ai/rules/commit-workflow.md)**
for staging, commit message conventions, pre-commit hook handling, and branch safety.

When asked to create a pull request, follow **[`.ai/rules/pr-workflow.md`](.ai/rules/pr-workflow.md)**
for PR template inference, JIRA handling, formatting, and repo identification.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
