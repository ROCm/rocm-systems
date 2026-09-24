# Bounded SuperCollider sensitivity experiments on RDNA4

September 24, 2026. The aim is a small improvement useful to several workloads,
keeping existing inputs, exact faults, native kernel allowlists, clean controls,
and the 6/8 detection threshold. These experiments do not reinterpret numerical
fault manifestations as SuperCollider detections.

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

## Next candidate: vary replay delay by resident wave

If completing the first observation is insufficient, test an opt-in delay based
on a bounded portion of the resident-wave hardware ID. Uniform sleeps can
preserve relative timing; varying the delay may make peer-wave writes land
between the two observations. Reuse the already-reserved scalar comparison
scratch before it holds the saved VCC value. Preserve guest SCC, VCC, EXEC,
registers, memory effects, and the existing default timing policy. This is a
hypothesis until clean and fault trials demonstrate a gain.

### Wave-delay results

`sleep_wave` is an opt-in RDNA4 prototype. The maximum is one of
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
| Native histogram | sleep_wave, max=15 | Pass | 0/8 | 0/8 |

Longer asymmetric delays can suppress the manifestation instead of exposing it.
A shorter max=3 experiment was selected next. All named existing controls retain
their original instruction sequences; their 977 tests pass. Two additional
execution/validation tests cover placement-dependent bounded delay, preserved
guest flags and neighboring scratch, and rejection of unsupported targets or
unbounded counts.

Wave-delay artifacts and prospective specifications:
`/home/benoit/workspace/consan-validation/sc-wave-20260924/`.

FP8 matmul reaches green at max=15, compared with the prior uniform sleep=4
result of 4/8. All eight trials were admitted and reached. Histogram is the next
prospectively selected workload at the same max=15 setting.

Histogram did not improve at max=15. FP16 matmul is the final selected workload
at that same setting; this is a bounded search, not an all-row calibration.
