# AI framework roadmap (reserved)

Templates for **future** work. Until adopted, these are **placeholders** only.

## Status

| Topic | Status |
|-------|--------|
| DAG skill orchestration | Reserved |
| MCP integration (standardized) | Reserved |
| Subagent specialization | Reserved |

## DAG orchestration (future)

**Intent:** DAG of skill nodes with dependencies; align with AgentSkillOS / arXiv [2603.02176](https://arxiv.org/abs/2603.02176).

- **Graph format:** _TBD (e.g. under `.ai/guide/dags/`)_
- **Runner:** _CLI / hook / CI — TBD_
- **Checklist when ready:** schema version; example DAG; human vs agent invocation; optional `ai_dev_guide` / CI check.

## MCP integration (future)

- **Allowed servers:** _TBD_
- **Config:** _TBD (e.g. `.mcp.json` local vs committed)_
- **Secrets:** env only; see [`.ai/rules/security.md`](rules/security.md).
- **Checklist:** document in `CONTRIBUTING.md` or [`.ai/rules/tools_policy.md`](rules/tools_policy.md); review rule for MCP deps.

## Subagent specialization (future)

- **Roles:** _table: role → paths → primary skill — TBD_
- **Tool policy:** per-role restrictions beyond [`.ai/rules/tools_policy.md`](rules/tools_policy.md).
- **Checklist:** align with vendor docs; link roles to [`.ai/skills/index.md`](skills/index.md).

## Promotion

When implemented, move content into real files and **shrink** this doc to pointers.
