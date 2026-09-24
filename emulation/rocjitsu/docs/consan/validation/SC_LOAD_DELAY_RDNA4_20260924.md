# RDNA4 SuperCollider uniform load replay delay

September 24, 2026. This pass uses unchanged workloads, native rocprofv3 allowlists,
reviewed faults, and the existing 6/8 threshold. Two prospective follow-ups use
uniform sleep=1 on **load replays only**
(`RJ_CONSAN_SC_DELAY_READS_ONLY=1`). Store readback instrumentation remains enabled.
Green requires a matching passing clean comparator, eight admitted/reached trials,
complete coverage, and healthy before/after GPU checks. Scatter-reduce remains out
of scope and is not rerun. Existing green rows are not part of this calibration.

| Workload | Delay selection | Clean | Detections | Qualification |
| --- | --- | --- | ---: | --- |
| `pytorch-rdna4-compiled-softmax` | uniform1-reads | Pass | 3/8 | Below 6/8 |
| `tp1-decode-combined` | uniform1-reads | Pass | 0/8 | Below 6/8 |

Completed 2 batches. Full results and frozen prospective specifications:
`/home/benoit/workspace/consan-validation/sc-sweep-20260924/uniform1`.

The companion `results.json` records coverage/health/control audits and paths to raw
clean and individual fault results. Failure to qualify a new setting does not erase
a stronger historical measurement at another setting; the status table keeps the
strongest qualified clean comparison, with a link to this complete sweep.

Hook SHA-256: `359c3f22adb001e185f67026211c204e369c7d36c6744c0601915e1d59522e17`.

## Conclusion

Neither prospective follow-up qualifies: compiled softmax detected **3/8** and
TP1 decode **0/8**. Both clean comparators passed, all 16 trials were admitted
and reached, and completeness/health checks passed. The table retains the
stronger historical observations of 4/8 and 2/8 respectively. These results do
not establish a sensitivity gain from the new opt-in load-only delay; it remains
an experimental control, disabled by default.

The frozen specification and plan were prepared before either run. Compiled
softmax uses the existing same-value-write allowance in both clean and fault
runs; TP1 decode does not. The implementation, test results, broader sweep,
and two new wave-delay qualifications are documented in the
[main sweep report](SC_SWEEP_RDNA4_20260924.md).
