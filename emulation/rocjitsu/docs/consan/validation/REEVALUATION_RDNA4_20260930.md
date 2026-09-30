# RDNA4 regression review — September 30, 2026

The reevaluation retains every initial result. Controls use the September 26
hook (`8b1f90c5fb3`) and current hook (`72a843d3b61`), unchanged inputs and
settings, matching fresh clean/native checks, and fresh fault inventories.
All reported trials passed admission, reach, instrumentation coverage, mutation
installation, and GPU health gates. Each batch contains eight trials.

| Configuration | Historical | Initial current | Fresh old / current controls | Resolution |
| --- | ---: | ---: | ---: | --- |
| Top-k Default, high preset | 7/8 | 5/8 | 6/8 / 6/8 | Keep initial yellow; version effect not reproduced. |
| Compiled-softmax SuperCollider, sleep=1 | 8/8 | 5/8 | 6/8 / 7/8 | Keep initial yellow; version effect not reproduced. |
| CLIP SuperCollider, sleep=2 | 8/8 | 2/8 | 5/8 / 1/8 | Requalify with read-only delays, below. |
| WMMA attention SuperCollider, sleep=15 | 5/8 | 0/8 | 0/8 / 0/8 | Keep yellow; version effect not reproduced. |

Small matched batches do not prove equivalence. Passing repeats do not replace
initial misses at unchanged settings. Top-k detected 11/16 across its two
current batches, illustrating the high preset's marginal qualification here.

CLIP remained lower in a reverse-order comparison: current-first detected 3/8,
old-second 5/8. This warrants mitigation rather than dismissing the initial drop.
All 16 historical/current patched images per workload for CLIP, compiled-softmax,
and WMMA are byte-identical after normalizing only the logged report-buffer
addresses. Each masked word was checked against the actual allocation address;
all other bytes participate in the full-image hash. SuperCollider's host report
allocation/read/cleanup registry is also unchanged. These checks rule out changed
emitted instructions and changed report-registry source in these samples, but
not effects of host timing or runtime conditions. The exact cause is not proven.

Increasing CLIP's delay alone did not qualify: sleep=4 detected 5/8 and sleep=3
2/8. With sleep=2 and `RJ_CONSAN_SC_DELAY_READS_ONLY=1`, detection reached 8/8;
a separately run confirmation also detected 8/8. Both had fresh matching
clean/native checks and inventories. The new ledger selects the confirmation
and labels the changed setting. This is a tested calibration mitigation, not
a source-code fix. All earlier failures remain in the evidence.

The completed campaign contains 320 initial fault trials plus 112 follow-up
trials: 16 top-k controls, 48 SuperCollider controls, 16 reverse-order CLIP
controls, and four eight-trial CLIP calibration/confirmation batches. The
45 initial clean configurations comprise 43 applicable passes and two confirmed
global-only scatter scope limitations. Fourteen additional clean configurations
passed for the controls and calibration. Qualification remains statistical;
it does not establish exhaustive detection.

The benchmark reevaluation also investigated two apparent slowdowns. hipBLASLt
now checks all 261 hazards instead of stopping after retaining 32 diagnostics;
that additional correctness work explains much of its increased startup cost.
Fresh decode controls did not consistently reproduce the historical execution
slowdown. The old control's native reference drifted −28.1%, so its overhead
ratios were rejected. The benchmark ledger retains the original complete
five-workload matrix and records the control limitations.

Raw evidence is in
[/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930).
Key records include `qualification-audit.json`, `detection-comparison.json`,
`topk-controls/comparison.json`, the audits under `detection-controls/`,
`clip-followup-audit.json`, `selected-requalifications.json`,
`sc-image-comparison.json`, `sc-host-report-comparison.json`,
`benchmark-qualification-audit.json`, and `decode-controls-audit.json`.
