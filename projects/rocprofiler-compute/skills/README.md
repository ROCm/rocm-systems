# rocprofiler-compute agent skills

User-facing Agent Skills for **profiling** and **analyzing** AMD GPU workloads
with `rocprof-compute`. They teach an AI agent how to collect hardware
counters, attribute PyTorch operators from experimental torch-trace workloads,
find occupancy / memory / stall bottlenecks, and emit reports.

| Skill | When to use |
|---|---|
| [profile/SKILL.md](profile/SKILL.md) | Collect counters, roofline, experimental PyTorch traces, experimental PC sampling |
| [analyze/SKILL.md](analyze/SKILL.md) | Interpret a workload directory and produce reports |

Each skill directory contains `SKILL.md`, `skill-card.md`, `references/` for
progressive disclosure, `scripts/` for deterministic checks, and
`evals/evals.json` for trigger evaluation. This adapts the packaged-skill
pattern used by `projects/rocprofiler-sdk/skills/pc-sampling/`.

Release results and pending hardware checks are recorded in
[VALIDATION.md](VALIDATION.md).

These are **not** the contributor workflows under `.ai/skills/` (code review,
rebase).

## Maintenance

Treat the skills as release artifacts. For each ROCm release:

1. Verify commands and deprecations against `src/argparser.py` and
   `CHANGELOG.md`.
2. Update the GPU/OS/API matrix and version dependencies in each skill's
   references.
3. Run trigger evaluations and helper-script tests on supported release
   environments.
4. Validate at least one recommended full workflow and one named fallback on
   representative supported hardware.
5. Record the tested ROCm build and hardware in the release validation
   results.

Changes to profiling behavior, output layout, supported architectures, or
experimental status must update the corresponding skill in the same product
release.
