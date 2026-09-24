# RDNA4 atomic-publication investigation — September 24, 2026

## Finding

The recorded Default reds for `streamk-arrival` and `tree-atomic-or` are
**false-positive publication conflicts caused by incomplete atomic ordering
support in ConSan**. These are two descriptions of the same problem. The clean
binaries have the synchronization needed for the reported accesses; ConSan does
not carry that ordering to all their sampled LDS access records.

This finding concerns the reported pairs in these exact binaries, not a proof
that every access in either workload is race-free on every target. The cells
remain red until the detector is fixed and clean/fault qualification passes.

## Evidence from the actual workloads

Fresh inspection runs with the unchanged hook reproduce 8 and 12 conflicts,
respectively. Both numerical tests pass. All 20 overlapping cross-wave access
pairs were reconstructed from the access records, including the four tree
examples omitted by the report's example limit. **Every pair has at least one
access without attached synchronization metadata.** No overflow, malformed
sync evidence, or dropped windows accounts for the reports.

The Stream-K counter uses an agent-scope acquire/release atomic increment. The
last arrival, identified by the returned counter value, reads the earlier
producer's partials. In the pristine `56e5c14547caddca` code object, `.text+0x10200`
(VA `0x12700`) is the application `global_atomic_add_u32`, using `SCOPE_DEV`.
It is preceded by `s_wait_storecnt 0` and `s_wait_loadcnt_dscnt 0`, and followed
by `s_wait_loadcnt 0` and `global_inv` at device scope. The reported LDS writes
precede publication; the consuming loads follow acquisition.

The tree's producers use release atomic ORs, and the consumer uses an
acquire/release OR after observing all producer bits. The OR operations form an
unbroken RMW release sequence: the final acquiring RMW observes publication from
all preceding producers, not just one. In code object `e4db9470d11020fa`, the
producer is `.text+0x23e80` (VA `0x26380`), preceded by
`s_wait_storecnt_dscnt 0`; the consumer is `.text+0x105ec` (VA `0x12aec`), followed
by `s_wait_loadcnt 0` and device-scope `global_inv`. Both use `SCOPE_DEV`.

The associated source tests are in the external `hip-moi` repository:
`tests/instrumented/028_rdna4_wmma_streamk_arrival_counter_test.hip` and
`029_rdna4_wmma_streamk_tree_atomic_or_test.hip`. Their own bookkeeping adds
other atomics; those must not be confused with the application counter/bitmask.
The ISA completion waits cover the wave's preceding LDS operations. The
lane-zero producer/consumer accesses alone are sufficient to exhibit the bug.

For the ordering basis, see LLVM's
[AMDGPU backend memory-model sequences](https://llvm.org/docs/AMDGPUUsage.html#memory-model)
and [AMDGPU memory model](https://llvm.org/docs/AMDGPUMemoryModel.html).
LDS is shared within a workgroup; an agent-scope synchronization operation
covers both waves. A completed release/acquire publication orders all covered
preceding/following accesses, not just the nearest access instruction.

## The implementation loses those relationships

In `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_sync.inc`, the atomic
association pass explicitly constrains **each directional half to one selected
access window**. It proposes nearby access/atomic associations, then discards
competing windows. `Candidate::selected_slot` and `release_selected_slot` retain
one acquire window and at most one release window.

`hooks/consan/rj_hsa_dbi_sync.h` joins pending acquire/release metadata only to
those exact slots. Its `atomic_pair_orders_same_workgroup` predicate requires
valid metadata on both records, matching atomic address/size, compatible epoch,
and adequate scope. Missing metadata therefore prevents suppression even when
the actual program has a valid publication edge.

In the Stream-K inspection, only the last partial store and first partial load
carry the application counter metadata. Another load carries a different
bookkeeping atomic; the remaining accesses have no atomic attachment. This
cannot represent publication of all eight values. The tree needs publication
from multiple producer waves as well as coverage of multiple access sites.

## Independent minimal reproduction

`tests/dbi/consan/hip_consan_atomic_publication_probe.hip` uses two waves with
only lane zero active for both LDS accesses and the atomic. It has no hip-moi
bookkeeping and requires no implicit communication between lanes. Each writer
stores its partial, performs an agent-scope acq_rel increment, and the final
arrival loads both partials. This is a direct release/acquire publication proof.

| Variant | Native numerical result (3 runs) | Current Default/max (3 runs) |
| --- | --- | --- |
| One value per producer, acq_rel | Pass | 1 conflict each |
| Eight values per producer, acq_rel | Pass | 2 conflicts each |
| Eight values, acq_rel plus workgroup barrier | Pass | Clean, zero conflicts |
| Eight values, relaxed counter | Pass in these runs | 2 conflicts each |
| One/eight values, acq_rel plus extra device fence | Not separately rerun natively | 1/2 conflicts each; numerical pass |

The numbers count sampled overlapping records, not all source-level races.
The compiler combines some of the volatile generic stores. The relaxed arm is
not made correct by its numerical passes. Its synchronization contract is
weaker and must remain detectable after a fix.

Strict runs keep `RJ_CONSAN_FORBID_DIAGNOSTICS=1` and reject the acq_rel arms
with exit 89. Separate inspection runs set it to zero to permit numerical
verification and retain report data; they do **not** qualify a clean cell.
Those inspection runs also pass numerically. The minimal one-value ISA has
`s_wait_dscnt 0` before the device-scope RMW, followed by completion and
invalidation, yet its emitted metadata labels the RMW acquire-only. This exposes
a second limitation: recognition of an LDS-only release must account for waits
already satisfied or separated by scalar bookkeeping. An extra source fence did
not remove the false positive.

## Artifacts and current status

`/home/benoit/workspace/consan-validation/atomic-publication-20260924/` contains
sources, binaries, `run.py`, strict `results.json`, inspection
`observe-results.json`, original/patched code objects, disassembly, fresh
workload logs, and `audit_pairs.py` / `pair-audit.json`. GPU `rocminfo` and the HIP
smoke test pass after the experiments.

The hook SHA-256 remains
`69566b53a48b3d65aa1835b2ae206c4f3cc5799952dcdb34560dfc5fe328e803`.
No detector behavior was changed for these measurements.
