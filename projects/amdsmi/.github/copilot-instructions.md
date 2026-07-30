# Copilot Instructions — AMD SMI

See [CLAUDE.md](../CLAUDE.md) for project overview, critical rules, and behavioral guidelines.
See [.github/CONTRIBUTING.md](CONTRIBUTING.md) for PR workflow and coding standards.

## Quick Reference

Critical rules (wrapper regeneration, `develop` target branch, pre-commit, version location, lint exclusions) live in [CLAUDE.md](../CLAUDE.md). This file adds only the agent settings below.

## Agent Settings

- **`agent-max-nesting-depth: 3`** — the deepest a subagent chain may nest.
  Subagents may dispatch their own subagents, but only while within this budget.
  Depth is counted from the first orchestrator (Planning / Development / Review),
  which runs at depth 1. An agent running
  at the max depth is a **leaf** — it must do the work itself and must not
  dispatch further. The mechanism lives in the `amdsmi-agent-handoff` skill ("Nesting
  Budget"). To change the cap, edit the number here — it is the single source of
  truth. (VS Code does not enforce this natively; the agents honor it by
  convention.)
