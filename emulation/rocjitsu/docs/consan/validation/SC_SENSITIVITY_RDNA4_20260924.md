# Bounded SuperCollider sensitivity experiments on RDNA4

September 24, 2026. The aim is a small improvement useful to several workloads,
keeping existing inputs, exact faults, native kernel allowlists, clean controls,
and the 6/8 detection threshold. The retained opt-in `sleep_wave` mode with
maximum 15 turns **two rows green**: FP16 matmul improves from 2/8 to **8/8**,
and FP8 matmul from 4/8 to **7/8**, with passing matched clean controls. These
experiments do not reinterpret numerical fault manifestations as SuperCollider
detections.

## Complete the first observation before delaying

The first candidate adds an LDS completion wait before the replay delay in the
ordinary LDS path. Previously, both original and duplicate accesses could be
queued before the final completion wait. Direct-to-LDS and D16 paths already
wait before the delay for their own data dependencies.

| Workload | Controls | Clean | Detection | Numerical fault manifestation |
| --- | --- | --- | --- | --- |
| Qwen-vocabulary top-k | sleep=1; completion wait | Pass | 0/8 | 8/8 |
| Compiled softmax | sleep=1; completion wait; same-value policy | Pass | 4/8 | 2/8 |

Top-k's eight trials were admitted and reached. The added wait alone did not
close its sensitivity gap. Compiled softmax matches its prior 4/8 result. The completion-wait candidate
was dropped without changing the tests or the production replay sequence.

The candidate changed exact instruction layouts, producing 37 initial test
failures. Since it showed no detection gain, it was dropped instead of updating
those tests. The original replay sequence is restored.

Local artifacts, including the exact source patch and hook hash:
`/home/benoit/workspace/consan-validation/sc-completion-20260924/`.

## Wave-dependent replay delay

The retained change varies replay delay using a bounded portion of the
resident-wave hardware ID. Uniform sleeps can preserve relative timing;
varying the delay can make peer-wave writes land between the two observations.
It reuses the already-reserved scalar comparison scratch before that register
holds saved VCC, preserving guest SCC, VCC, EXEC and neighboring registers.

### Wave-delay results

`sleep_wave` is an opt-in RDNA4 mode. The maximum is one of
1/3/7/15/31/63/127 sleep units (64 clocks per unit). It extracts the upper bits
of the resident-wave identity, including SIMD placement, into the existing
scalar VCC-save scratch before that scratch becomes live. A hardware-register
read, dependency delay, and `s_sleep_var` leave guest condition registers intact.
No register allocation, evidence format, memory-model, or default-mode changes
are required. Flat accesses save VCC after the delay in this mode.

| Workload | Controls | Clean | Detection | Numerical fault manifestation |
| --- | --- | --- | --- | --- |
| Compiled softmax | sleep_wave, max=127; same-value policy | Pass | 1/8 | 0/8 |
| Compiled softmax | sleep_wave, max=3; same-value policy | Pass | 0/8 | 0/8 |
| Production FP8 matmul | sleep_wave, max=15 | Pass | **7/8** | 0/8 |
| Production FP16 matmul | sleep_wave, max=15 | Pass | **8/8** | 0/8 |
| Native histogram | sleep_wave, max=15 | Pass | 0/8 | 0/8 |

The softmax results show that asymmetric delays can suppress manifestations
instead of exposing them; a larger delay is not necessarily better. The existing
controls retain their original instruction sequences. The ConSan suite passes
979 tests, with two opt-in benchmarks skipped. New execution tests cover bounded
placement-dependent delay, active and inactive waves, both SCC values, preserved
VCC/EXEC and neighboring registers, and invalid target/count rejection.

Wave-delay artifacts and prospective specifications:
`/home/benoit/workspace/consan-validation/sc-wave-20260924/`.

All seven batches (56 fault trials) were admitted, reached, complete, and healthy,
with matching clean controls. Both matmul rows reach the unchanged qualification
bar at max=15. Histogram and compiled softmax remain yellow; this setting is a
useful addition for some workloads, not a universal sensitivity improvement.
Eight-trial qualification does not establish a precise long-run detection rate.
No all-row calibration or minimum-delay search was attempted.

## Reproduction and audit

The new setting is:

```sh
RJ_CONSAN_MODE=supercollider
RJ_CONSAN_SC_DELAY_MODE=sleep_wave
RJ_CONSAN_SC_DELAY=15
```

For the validation runner, use `CONSAN_VALIDATION_SC_DELAY_MODE=sleep_wave` and
`CONSAN_VALIDATION_SC_DELAY=15` for the matching clean comparator. The qualified
matmul fault specifications use `barrier-drop-initial-tile-publication-sc-wave-15`
with the corresponding workload ID and `--profile supercollider`. Each uses
8 trials and the unchanged 6/8 acceptance threshold.

The artifact directory contains `audit.py` and `audit.json`, checking matching
clean/fault controls and hook SHA-256, admitted/reached trial counts, complete
coverage, GPU health, and reported detection totals for every completed batch.
All wave-delay experiments use hook SHA-256
`d34d1aad76da798b37920b2ff4e85f7d5013c7d632f9f0f06498cad5e0f22e92`.
The implementation is committed as `466abc5cd79`, with additional flag-preservation
coverage in `c735f7ac8f0`. The qualification artifacts preserve their original
experimental fault IDs; repository fault IDs are shortened for reuse.
