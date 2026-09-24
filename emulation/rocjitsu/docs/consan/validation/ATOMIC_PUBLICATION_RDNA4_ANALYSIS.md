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
sync evidence, or dropped windows accounts for the reports. Stream-K's ranges
are `[1024,1056)` in four-byte pieces; tree's are `[0,16)`, `[1024,1040)`, and
`[2048,2064)`. With 32 lanes × eight floats per subgroup, these are all lane-zero
partial elements. Both sides of the reported communication therefore use the
same lane that executes publication/acquisition; the classification does not
rely on assuming that an atomic in one lane orders unrelated lanes.

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

## What a correct fix needs

### 1. Recognize the actual completion requirements

`consan_sync_analysis.cpp::exact_release_wait_boundary` scans a contiguous
suffix of waits immediately before the atomic. The target wait classifier marks
store/flat-store waits as release boundaries but does not mark an LDS-only wait
as such. The minimal program has completed its generic store earlier, then
executes scalar/EXEC bookkeeping and a final `s_wait_dscnt 0`. Its release is
real, but this local pattern loses it.

Recognition should prove completion of the relevant earlier memory operations
on each admitted path, using their address spaces and target counters. A
standalone LDS wait is sufficient for preceding LDS accesses only when that
path proof is available. It must not indiscriminately establish completion of
outstanding global stores. Similarly, an unrelated earlier wait across a loop
or a path with later stores is insufficient. This belongs in target wait
semantics plus bounded CFG/dataflow analysis, with explicit support decisions.

### 2. Replace single-window attachment with publication state

The implementation cannot repair the original workloads just by recognizing
one more wait. It already recognizes their application atomic sequences.
One release must cover all sampled writes sequenced before it, and one acquire
must cover all sampled reads sequenced after it. Multiple independent atomic
objects can contribute ordering to the same access; bookkeeping atomics must
not overwrite the application publication.

A concrete representation is a per-owner logical access sequence plus retained
atomic publication events:

- Every sampled access carries its owner and dynamic sequence, in addition to
  the existing dispatch, workgroup, generation, and allocation identity.
- A release publishes the owner's preceding access frontier for that atomic
  object. An acquire merges the frontier of the publication it actually
  observed. Later accesses carry or reference that acquired frontier.
- RMW events preserve release-sequence ancestry. The tree consumer must inherit
  all three producer frontiers. A single last-owner field cannot represent it.
- Barriers and report-generation changes delimit/reset the relevant state;
  hash collisions, lost events, and uncertain joins remain explicitly
  incomplete, never permission to suppress a conflict.

This would change the report ABI/model, atomic and access emission, pending
joins, and host conflict analysis. For bounded kernels, an alternative is to
retain complete per-event access-window association lists and resolve their
ordering graph on the host. The required semantics are the same: many accesses
per event, multiple contributing events per access, and dynamic execution
identity. Merely removing the association pass's deduplication is insufficient;
its selected-slot fields and emitters still retain only one window.

### 3. Require evidence that the acquire observed the release

The current `SyncMetadata` has address, size, role, scope, outcome class, and
epochs. It has no observed atomic value, modification identifier, or release
sequence lineage. The current host predicate matches roles and addresses; that
alone is not a general happens-before proof. An acquire that read an initial or
older value must not inherit a later publication to the same address.

A generalized implementation therefore needs a validated observation witness:
for example, captured RMW old/new values and a proven unambiguous modification
chain, or a correctly synchronized per-object publication protocol that binds
the guest atomic transition to its metadata. Repeated values/ABA and concurrent
publishers must not be guessed through. Non-returning producer ORs are an
explicit design case: obtaining their modification identity needs additional
instrumentation or a proven protocol-specific witness. Ordering only sideband
records, without binding them to the guest operation, is insufficient.

This is a prerequisite for safely broadening suppression, not evidence that
such a false negative occurred in the current two workloads. Their monotone
counter/bitmask protocols give a direct ordering proof for this investigation.

### 4. Qualification for the implementation

| Case | Required behavior |
| --- | --- |
| One publication, multiple preceding writes and following reads | Clean without adding a workgroup barrier |
| Either wave is last arrival | Same clean result |
| Three producer release ORs followed by acquiring OR | All producer partials ordered |
| Two atomic objects interleaved with bookkeeping | Correct object retained for each publication |
| LDS-only release completion; earlier completed stores plus scalar bookkeeping | Recognized when the path proves completion |
| Pending global store not drained by an LDS-only wait | No invented release proof for that store |
| Relaxed publication or insufficient scope | Conflicts remain detectable |
| Acquire observes the initial value; later release to the same address | No retroactive synchronization |
| RMW release sequence, repeated values, CAS failure, loop iterations | Correct lineage or explicit incomplete evidence |
| Different dispatch/workgroup/generation, collisions or lost records | No cross-identity joins or silent clean result |

Then rerun both unchanged workload clean controls and their meaningful weakened
scope/order faults with the existing 6/8 bar. The added barrier probe is a
localization control, not a proposed workload repair. Changing the workloads to
add a barrier would remove the atomic-publication behavior that these tests
are intended to exercise.

## Implementation progress

The host publication proof is implemented in
`hooks/consan/rj_hsa_dbi_publication.{h,cpp}` (commit `33c59de6606`). It resolves
observed RMW transitions independently of physical record order, propagates
release sequences across producers and publication through multiple objects,
and checks per-owner dynamic access order. Missing observations, duplicate
values/ABA, disconnected modification chains, overlapping object identities,
and inconsistent execution order return explicit incomplete evidence.

Sixteen focused tests cover that contract; all 261 hook unit tests pass. The
build and test evidence is in `implementation-build.log` and
`host-publication-tests.log` in the artifact directory below.

This component is not yet connected to device reports and does not change the
current red cells. Remaining work is to emit and validate atomic observation
records and access sequence identities, improve completion recognition, connect
the proof to conflict analysis, and run the clean/fault qualification matrix
above. The device capture must establish the complete-transition precondition;
setting that flag on the existing address/role metadata would be unsound.

## Reproducing the minimal probe

Use the current ROCm environment, an artifact directory outside the source tree,
and the current hook. For example, from the repository root:

```sh
source /home/benoit/workspace/consan-validation/rdna4-20260923/env-current.sh
hipcc --offload-arch=gfx1201 -O2 -DCOUNT=1 \
  emulation/rocjitsu/tests/dbi/consan/hip_consan_atomic_publication_probe.hip \
  -o /tmp/consan-publication
/tmp/consan-publication
HSA_TOOLS_LIB="$CONSAN_VALIDATION_HOOK" \
HSA_TOOLS_ROCPROFILER_V1_TOOLS=1 \
RJ_CONSAN_MODE=default RJ_CONSAN_PRESET=max RJ_CONSAN_LOG=1 \
RJ_CONSAN_KERNEL_ALLOWLIST=_Z11publicationPiS_ \
RJ_CONSAN_FORBID_DIAGNOSTICS=1 /tmp/consan-publication
```

The current strict detector run exits 89 with a conflict. Recompile with
`-DCOUNT=8 -DBARRIER=1` for the clean barrier control or `-DCOUNT=8 -DRELAXED=1`
for the deliberately weakened ordering control. This is a manual regression
probe, not a passing test added to the default test suite.

## Artifacts and current status

`/home/benoit/workspace/consan-validation/atomic-publication-20260924/` contains
sources, binaries, `run.py`, strict `results.json`, inspection
`observe-results.json`, original/patched code objects, disassembly, fresh
workload logs, and `audit_pairs.py` / `pair-audit.json`. GPU `rocminfo` and the HIP
smoke test pass after the experiments.

The hook SHA-256 remains
`69566b53a48b3d65aa1835b2ae206c4f3cc5799952dcdb34560dfc5fe328e803`.
No detector behavior was changed for these measurements.
