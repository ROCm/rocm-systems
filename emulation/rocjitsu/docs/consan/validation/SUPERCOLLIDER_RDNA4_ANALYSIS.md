# RDNA4 SuperCollider investigation — September 24, 2026

The September 23 revalidation has 3 green SuperCollider cells, compared with
16 in the historical ledger (commit `41c6a6e5e46`). These colors do not establish
an across-the-board detector regression: the fault contracts and some workloads
changed. This investigation separates those changes from detection behavior.

## Historical greens included expected misses

Retained July 16 artifacts under
`/home/benoit/workspace/consan-validation/final-tip-640e575-faults*`
contain accepted SuperCollider profiles with `detector_policy: not_detected`,
`detections: 0`, and `trials: 1` for D128 block, D128 pressure, WMMA attention,
Jakub, Stream-K arrival, and tree atomic-OR. Stream-K and tree include both
atomic-order and atomic-scope weakening. The current contract requires at least
6 detections in 8 admitted/reached trials. These six historical greens therefore
cannot support a claim that these workloads previously detected faults reliably.

Other retained bundles contain genuine SuperCollider mismatch reports, including
Qwen, TP1 prefill/decode, TP2, and CLIP. Those accepted profiles each had one
trial. Some earlier attempts expected a miss and were rejected *because a fault
was detected*; later specifications changed that expectation. This is another
reason to compare raw evidence rather than the old `accepted` flag.

## Faults changed, even where the code object did not

| Workload | Historical positive fault | September 23 fault |
| --- | --- | --- |
| Qwen | `_initializer_11_dispatch_0_transpose_1024x3072_f32`, ELF `de4a0fc370f12a36` | `main$async_dispatch_0_reduction_5x1024_f32`, ELF `5d29deb0b917dbeb` |
| TP1 prefill | matmul dispatch 2, `.text+0xaa4` | matmul dispatch 13, `.text+0x9f24` |
| TP1 decode | matmul dispatch 30, `.text+0x14c30` | attention dispatch 24, `.text+0x133a0` |
| TP2 | matmul dispatch 127, `.text+0x10c64` | matmul dispatch 27, `.text+0x9b70` |
| CLIP | batch matmul dispatch 11, `.text+0x562c` | matmul dispatch 3, `.text+0x1b98` |

TP1, TP2, and CLIP retain the respective code-object identities across these
comparisons. Qwen's old initializer is absent from the current native allowlist.
A regression claim needs the same executed binary, fault, controls, and comparable
trials. The existing comparison changes several of these simultaneously.

## Prospective experiment

Rerun the historical TP2 dispatch-127 barrier removal using the current hook,
current workload and allowlist, a matching clean run, delay=0, and 8 fault trials
with a 6/8 threshold. The pristine ISA confirms the exact signal/wait pair between
cooperative tile stores and consuming LDS loads. Retain the dispatch-27 miss;
an additional positive fault would not qualify that missed fault.

Artifacts and the frozen prospective specification:
`/home/benoit/workspace/consan-validation/sc-deep-dive-20260924/`.
