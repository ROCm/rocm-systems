# rocprofiler-compute agent skills

User-facing Agent Skills for **profiling** and **analyzing** AMD GPU workloads
with `rocprof-compute`. They teach an AI agent how to collect hardware
counters (rocpd), attribute ROCTX ranges, find occupancy / memory / stall
bottlenecks, and emit txt/csv/db reports.

| Skill | When to use |
|---|---|
| [profile/SKILL.md](profile/SKILL.md) | Collect counters, roofline, ROCTX, experimental PC sampling |
| [analyze/SKILL.md](analyze/SKILL.md) | Interpret a workload directory and produce reports |

Each skill directory follows the same layout as
`projects/rocprofiler-sdk/skills/pc-sampling/`: `SKILL.md`, `skill-card.md`,
and `evals/evals.json` for trigger evaluation each ROCm release.

These are **not** the contributor workflows under `.ai/skills/` (code review,
rebase).
