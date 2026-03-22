# Advanced harness (reserved — not implemented)

Use this file as a **template** for future work. Until each section is adopted, treat content as **design placeholders** only.

---

## Status

| Topic | Status | Owner action |
|-------|--------|----------------|
| DAG skill orchestration | **Reserved** | Fill “Implementation checklist” when adding automation |
| MCP integration | **Reserved** | Fill server list + policy when MCP is standardized for this repo |
| Subagent specialization | **Reserved** | Fill roles when using Claude Code / tool-specific subagents in CI or locally |

---

## DAG skill orchestration (future)

**Intent:** Represent multi-step tasks as a **directed acyclic graph** of skill nodes (dependencies, artifacts), aligned with ecosystem-scale skill papers (e.g. AgentSkillOS / arXiv [2603.02176](https://arxiv.org/abs/2603.02176)).

### Reserved template

- **Graph format:** _(e.g. YAML/JSON under `.ai/harness/dags/` — TBD)_
- **Runner:** _(CLI / hook / CI job — TBD)_
- **Validation:** Each node maps to an existing `.ai/skills/*.md`; edges declare required outputs.

### Implementation checklist (when ready)

- [ ] Define schema version and one example DAG (e.g. `fix_bug` → `write_test` → `optimize_performance`).
- [ ] Document how humans vs agents invoke the runner.
- [ ] Add `ai_dev_harness` or CI check for schema validity.

---

## MCP integration (future)

**Intent:** Curated Model Context Protocol servers for docs, issues, or internal APIs—**least privilege**, audited.

### Reserved template

- **Allowed servers:** _(list names + purpose — TBD)_
- **Config location:** _(e.g. committed `.mcp.json` vs local only — TBD)_
- **Secrets:** Never commit tokens; use env + [`.ai/rules/security.md`](../rules/security.md).

### Implementation checklist (when ready)

- [ ] Document servers in `CONTRIBUTING.md` or `.ai/harness/tools-policy.md`.
- [ ] Add review rule: MCP-added dependencies require approval.

---

## Subagent specialization (future)

**Intent:** Dedicated subagents (e.g. “trace parser only”, “CMake only”) with **narrow tool allowlists** and **scoped context**.

### Reserved template

- **Roles:** _(table: role name → allowed paths → primary skill — TBD)_
- **Tool policy:** Per-role bash/git restrictions beyond global [tools-policy.md](tools-policy.md).
- **Product mapping:** _(Claude Code agents / Cursor modes / other — TBD)_

### Implementation checklist (when ready)

- [ ] Align with vendor docs (e.g. Claude Code subagents).
- [ ] Link each role to `.ai/skills/index.md` entries.

---

## When to promote a section

Move content from here into real files (skills, `tools-policy.md`, `security.md`, CI) and **shrink** this section to a pointer once implemented.
