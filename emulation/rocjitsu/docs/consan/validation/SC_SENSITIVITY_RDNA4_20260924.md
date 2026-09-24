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

Top-k's eight trials were admitted and reached. The added wait alone did not
close its sensitivity gap. Compiled softmax is the second selected workload;
its existing same-value-write policy and sleep=1 controls are retained.

This candidate changes exact instruction layouts; the initial suite has 37
failures. It is an experiment, not a qualified source change. Updating those
tests is conditional on demonstrating sufficient end-to-end benefit.

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
