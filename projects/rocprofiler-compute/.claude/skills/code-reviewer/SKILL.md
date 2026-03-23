---
name: code-reviewer
description: Review rocprofiler-compute diffs/PRs — correctness, security, profiling/GPU, layering, tests (uses shared .ai skill).
---

# Code reviewer (Claude)

When this skill is invoked, follow the shared playbook (canonical content):

1. [`.ai/skills/code_review.md`](../../../.ai/skills/code_review.md) — priorities, comment types, domain checks, output shape.
2. [`AGENTS.md`](../../../AGENTS.md) — agent entry and four-layer guide.
3. [`CLAUDE.md`](../../../CLAUDE.md) — project brain and mandatory rules links.

**Focus:** GPU/profiling correctness, package layering (profiler vs analysis), profile-mode overhead, security from [`.ai/rules/security.md`](../../../.ai/rules/security.md).

**Human checklist:** [`.ai/review/checklist.md`](../../../.ai/review/checklist.md).
