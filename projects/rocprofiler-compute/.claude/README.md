# Claude Code wiring (rocprofiler-compute)

**Claude-specific** pieces live here. The **shared** four-layer guide (Context → Skills → Tools → Hooks) is documented for **all models** in [`.ai/guide/workflow.md`](../.ai/guide/workflow.md).

## What is Claude-only vs shared

| Layer | Shared (all agents) | Claude-only |
|-------|---------------------|-------------|
| Context | Each tool’s brain file points at `.ai/` — see [workflow.md](../.ai/guide/workflow.md) | [`CLAUDE.md`](../CLAUDE.md) |
| Skills | [`.ai/skills/`](../.ai/skills/), [`.ai/guide/taxonomy.md`](../.ai/guide/taxonomy.md) | — |
| Tools | [`.ai/rules/tools_policy.md`](../.ai/rules/tools_policy.md) | — |
| Hooks | pre-commit + `scripts/ai_dev_guide.py` | [settings.json](settings.json), [hooks/bash_guard.py](hooks/bash_guard.py) |

## Relation to AgentSkillOS

[AgentSkillOS](https://github.com/ynulihao/AgentSkillOS) / arXiv [2603.02176](https://arxiv.org/abs/2603.02176): hierarchical skills + composition — implemented in repo via [`.ai/guide/taxonomy.md`](../.ai/guide/taxonomy.md) + `.ai/skills/*.md`.

## Files here

| File | Purpose |
|------|---------|
| [settings.json](settings.json) | `PreToolUse` → `bash_guard.py` |
| [rules/claude-guide.md](rules/claude-guide.md) | Claude rule injection (points at shared `.ai/` paths) |
| [capabilities.md](capabilities.md) | Stub → `.ai/guide/taxonomy.md` |
| [tools-policy.md](tools-policy.md) | Stub → `.ai/rules/tools_policy.md` |
| [hooks/bash_guard.py](hooks/bash_guard.py) | Destructive-pattern guard for `Bash` tool |

## References

- Claude Code hooks: [Hooks reference](https://code.claude.com/docs/en/hooks)
- Full agent map: [`.ai/README.md`](../.ai/README.md)
