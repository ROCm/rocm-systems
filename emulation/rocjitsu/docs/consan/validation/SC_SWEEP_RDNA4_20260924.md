# RDNA4 SuperCollider pass over remaining yellow rows

September 24, 2026. This pass uses unchanged workloads, native rocprofv3 allowlists,
reviewed faults, and the existing 6/8 threshold. Each row first tries `sleep_wave`
with maximum 15. Rows below the bar then try the same delay on **load replays only**
(`RJ_CONSAN_SC_DELAY_READS_ONLY=1`). Store readback instrumentation remains enabled.
Green requires a matching passing clean comparator, eight admitted/reached trials,
complete coverage, and healthy before/after GPU checks. Scatter-reduce remains out
of scope and is not rerun. Existing green rows are not part of this calibration.

| Workload | Delay selection | Clean | Detections | Qualification |
| --- | --- | --- | ---: | --- |
| `qwen-prefill` | all | Pass | 0/8 | Below 6/8 |
| `qwen-prefill` | reads | Pass | 0/8 | Below 6/8 |

Completed 2 batches. Full results and frozen prospective specifications:
`/home/benoit/workspace/consan-validation/sc-sweep-20260924`.

The companion `results.json` records coverage/health/control audits and paths to raw
clean and individual fault results. Failure to qualify a new setting does not erase
a stronger historical measurement at another setting; the status table keeps the
strongest qualified clean comparison, with a link to this complete sweep.

Hook SHA-256: `359c3f22adb001e185f67026211c204e369c7d36c6744c0601915e1d59522e17`.
