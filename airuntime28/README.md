# AIRUNTIME-28: non-temporal stores in the blit copy kernel

Does a non-temporal store hint in `__amd_rocclr_copyBuffer` make device-to-device copies
faster on MI450? Measured on gfx1250.

Short answer: not on its own. It is worth having for the concurrent case — a cache-sensitive
kernel running alongside large copies — and only if such a workload can be named. Access width
matters about two hundred times more than any temporal hint, and the width sacrifice the ticket
assumed was never required.

| | |
|---|---|
| [REPORT.md](REPORT.md) | the decision, the numbers, the risks. Start here. |
| [METHOD.md](METHOD.md) | controls, statistics, confounders, and what is not controlled |
| [FINDING-gl2-residency.md](FINDING-gl2-residency.md) | a separate finding with a larger prize: nothing in GL2 survives a kernel dispatch on this part |
| [REPRODUCE.md](REPRODUCE.md) | how to run any of it |
| [CHANGELOG.md](CHANGELOG.md) | every claim an earlier revision made and this one withdraws |
| [daily-sync-script.md](daily-sync-script.md) | 40 seconds of it, out loud |

```
src/common/       the measurement core: one definition of each concept
src/experiments/  one thin main per question
remote/           build.sh, run_all.sh, and the inspection scripts
results/<stamp>/  one self-describing result set per run, with provenance
```

Run everything with `remote/run_all.sh`; see [REPRODUCE.md](REPRODUCE.md). The report's tables
are regenerated from a result set's `rows.tsv` rather than hand-copied.

## Where this lives

| | |
|---|---|
| the CLR change | commit `81e65d6bbb` on [`users/victzhan/AIRUNTIME-28-nt-blit`](https://github.com/ROCm/rocm-systems/tree/users/victzhan/AIRUNTIME-28-nt-blit), based on `develop` |
| this artifact | [`users/victzhan/AIRUNTIME-28-investigation`](https://github.com/ROCm/rocm-systems/tree/users/victzhan/AIRUNTIME-28-investigation/airuntime28) under `airuntime28/`. **Never merged** - it is a durable home for the measurements, not product code. |
| PR description | [PR-BODY.md](PR-BODY.md), kept in the repo so it can be regenerated rather than retyped |

The remote copy is pushed from here with `git archive --prefix=airuntime28/`, so this working
copy is the source and that branch is the published form. Update it by re-pushing rather than
by editing it in place, or the two will drift.