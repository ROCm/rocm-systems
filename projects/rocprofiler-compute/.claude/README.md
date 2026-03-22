# Claude Code wiring (rocprofiler-compute)

**Claude-specific** pieces live here. The **shared** four-layer harness (Context → Skills → Tools → Hooks) is documented for **all models** in [`.ai/harness/multi_model.md`](../.ai/harness/multi_model.md).

## What is Claude-only vs shared

| Layer | Shared (all agents) | Claude-only |
|-------|---------------------|-------------|
| Context | Each tool’s brain file points at `.ai/` — see [multi_model.md](../.ai/harness/multi_model.md) | [`CLAUDE.md`](../CLAUDE.md) |
| Skills | [`.ai/skills/`](../.ai/skills/), [`.ai/harness/capabilities.md`](../.ai/harness/capabilities.md) | — |
| Tools | [`.ai/harness/tools-policy.md`](../.ai/harness/tools-policy.md) | — |
| Hooks | pre-commit + `scripts/ai_dev_harness.py` | [settings.json](settings.json), [hooks/bash_guard.py](hooks/bash_guard.py) |

## Relation to AgentSkillOS

[AgentSkillOS](https://github.com/ynulihao/AgentSkillOS) / arXiv [2603.02176](https://arxiv.org/abs/2603.02176): hierarchical skills + composition — implemented in repo via [`.ai/harness/capabilities.md`](../.ai/harness/capabilities.md) + `.ai/skills/*.md`.

## Files here

| File | Purpose |
|------|---------|
| [settings.json](settings.json) | `PreToolUse` → `bash_guard.py` |
| [rules/claude-harness.md](rules/claude-harness.md) | Claude rule injection (points at shared harness paths) |
| [capabilities.md](capabilities.md) | Stub → `.ai/harness/capabilities.md` |
| [tools-policy.md](tools-policy.md) | Stub → `.ai/harness/tools-policy.md` |
| [hooks/bash_guard.py](hooks/bash_guard.py) | Destructive-pattern guard for `Bash` tool |

## References

- Claude Code hooks: [Hooks reference](https://code.claude.com/docs/en/hooks)
- Full agent map: [`.ai/README.md`](../.ai/README.md)
