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
| `pytorch-torch-mode` | all | Pass | 0/8 | Below 6/8 |
| `pytorch-torch-mode` | reads | Pass | 0/8 | Below 6/8 |
| `tp1-prefill` | all | Pass | 8/8 | Green |
| `tp1-decode-combined` | all | Pass | 0/8 | Below 6/8 |
| `tp1-decode-combined` | reads | Pass | 0/8 | Below 6/8 |
| `pytorch-rdna4-compiled-softmax` | all | Pass | 1/8 | Below 6/8 |
| `pytorch-rdna4-compiled-softmax` | reads | Pass | 3/8 | Below 6/8 |
| `pytorch-rdna4-llm-topk` | all | Pass | 0/8 | Below 6/8 |
| `pytorch-rdna4-llm-topk` | reads | Pass | 0/8 | Below 6/8 |
| `tp2-family` | all | Pass | 0/8 | Below 6/8 |
| `tp2-family` | reads | Pass | 0/8 | Below 6/8 |
| `pytorch-torch-histc` | all | Pass | 0/8 | Below 6/8 |
| `pytorch-torch-histc` | reads | Pass | 0/8 | Below 6/8 |
| `llama-rdna4-rms-norm` | all | Pass | 8/8 | Green |
| `d128-block` | all | Pass | 0/8 | Below 6/8 |
| `d128-block` | reads | Pass | 0/8 | Below 6/8 |
| `d128-pressure` | all | Pass | 0/8 | Below 6/8 |
| `d128-pressure` | reads | Pass | 0/8 | Below 6/8 |
| `wmma-attention` | all | Pass | 0/8 | Below 6/8 |
| `wmma-attention` | reads | Pass | 0/8 | Below 6/8 |
| `streamk-arrival` | all | Pass | 0/8 | Below 6/8 |
| `streamk-arrival` | reads | Pass | 0/8 | Below 6/8 |
| `tree-atomic-or` | all | Pass | 0/8 | Below 6/8 |
| `tree-atomic-or` | reads | Pass | 0/8 | Below 6/8 |
| `jakub-attention` | all | Pass | 0/8 | Below 6/8 |
| `jakub-attention` | reads | Pass | 0/8 | Below 6/8 |

Completed 28 batches. Full results and frozen prospective specifications:
`/home/benoit/workspace/consan-validation/sc-sweep-20260924`.

The companion `results.json` records coverage/health/control audits and paths to raw
clean and individual fault results. Failure to qualify a new setting does not erase
a stronger historical measurement at another setting; the status table keeps the
strongest qualified clean comparison, with a link to this complete sweep.

Hook SHA-256: `359c3f22adb001e185f67026211c204e369c7d36c6744c0601915e1d59522e17`.

## Findings

Two previously yellow rows qualify with wave-dependent sleep, maximum 15:
**TP1 prefill (8/8)** and **llama.cpp RMS norm (8/8)**. Both use delays on
loads and stores. Their exact prospective variants are preserved in
`tests/dbi/consan/consan_validation_faults_gfx1201.json`; the fault IDs end in
`-sweep-wave15-all`. Including the five previously green rows, this brings
SuperCollider to seven green rows. Thirteen in-scope rows remain yellow, plus
the out-of-scope scatter-reduce row.

All 28 configurations passed their matching clean runs and coverage/health
checks. The 224 fault trials were admitted and reached; the recorded detection
counts were independently checked against the individual trial results.
These are sensitivity misses, not rejected runs. Existing stronger historical
results remain visible in the status table.

The new optional load-only delay (`RJ_CONSAN_SC_DELAY_READS_ONLY=1`) lets writers
progress without adding the selected delay to their readbacks. It retains both
load and store instrumentation and comparison logic. The control defaults off;
ordinary SuperCollider timing and the Default engine are unchanged. This is a
small scheduling experiment, not additional memory-model coverage. With
wave-dependent sleep it produced **no additional green rows**. Compiled softmax
observed 3/8 versus 1/8 with delays on both kinds of access; that small difference
alone does not establish a reliable improvement.

Two prospectively specified follow-ups use uniform sleep=1 on loads only for
compiled softmax and TP1 decode. This gives every load replay a nonzero delay,
including waves assigned zero by the wave-dependent setting. See the
[uniform load-delay follow-up](SC_LOAD_DELAY_RDNA4_20260924.md).

## Implementation checks and reproduction

The implementation is local commit `add637cc3e3`. Tests cover delay filtering
across timing modes and retention of both load and store patches. The full
ConSan/ObservationPlan test run passed **981 tests**, with two opt-in benchmarks
skipped (`tests-final.log` under the artifact root above). No hook rebuild
occurred during the GPU sweep.

Source the host's `rdna4-20260923/env-current.sh`, select the same generated
native allowlists, and use the validation runner with `--target gfx1201`.
For the two new qualifying rows, run a `supercollider` clean comparator with
`CONSAN_VALIDATION_SC_DELAY_MODE=sleep_wave`,
`CONSAN_VALIDATION_SC_DELAY=15`, and
`CONSAN_VALIDATION_SC_DELAY_READS_ONLY=0`, then the corresponding checked-in
`-sweep-wave15-all` fault variant. The frozen `plan.json`, `spec.json`, and
`run_sweep.py` preserve the full sweep, including unsuccessful settings. Use
fresh artifact directories for reruns rather than overwriting this evidence.
