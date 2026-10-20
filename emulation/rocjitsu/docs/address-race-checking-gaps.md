# Address race checking gaps

This page lists known gaps in the core address race checker that
`--race-checking=on` enables, described in
[address race diagnostics](memory-wait-diagnostics.md#address-race-diagnostics).
Each entry says what the checker misses or misreports, why, and what closing the
gap would take. The entries come from reading the checker
([`memory_race_checker.cpp`](../lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_race_checker.cpp))
and its call sites in
[`compute_unit.cpp`](../lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp),
from the [mutation harness](../tests/memory-diagnostics/shaders/README.md), and
from running the
[gfx950 Gluon tutorial kernels](https://github.com/ROCm/gfx950-gluon-tutorials).
The list is not exhaustive.

A clean run means that no unordered, conflicting pair of accesses was observed on
the paths the run executed, within the limits below. It does not prove that a
kernel is race-free.

## Summary

| Gap | Effect |
|---|---|
| [In-flight LDS operations at a barrier](#in-flight-lds-operations-at-a-barrier-barrier-hygiene) | Not reported unless another wave touches the bytes, by design |
| [Cross-workgroup synchronization](#cross-workgroup-synchronization) | False reports |
| [Atomic loads and stores](#atomic-loads-and-stores) | False reports |
| [LDS synchronization other than barriers](#lds-synchronization-other-than-barriers) | False reports |
| [gfx1250 LDS barrier cells](#gfx1250-lds-barrier-cells) | False reports |
| [Split barriers](#split-barriers) | Missed races |
| [Global accesses within a workgroup](#global-accesses-within-a-workgroup) | Missed races |
| [Accesses outside one dispatch](#accesses-outside-one-dispatch) | Missed races |
| [Checkpoint-restored LDS accesses](#modes-and-lifecycle) | Missed races |
| [Cluster multicast into peer LDS](#cluster-multicast-into-peer-lds) | Missed races |
| [Limits inherited from the functional model](#limits-inherited-from-the-functional-model) | Missed races and false reports |
| [Dynamic coverage](#dynamic-coverage) | Missed races |
| [Global shadow budget](#global-shadow-budget) | Missed races once the budget is reached |
| [LDS history size](#lds-history-size) | Unbounded memory |
| [Finding limits](#finding-limits) | Hidden findings |
| [Finding content](#finding-content) | Harder triage |
| [Modes and lifecycle](#modes-and-lifecycle) | Selectable coverage, reset history |
| [Performance](#performance) | Run time and memory |
| [Validation](#validation) | Untested or stale coverage |

## In-flight LDS operations at a barrier (barrier hygiene)

Barrier hygiene here means draining a wave's outstanding LDS operations before
it arrives at a barrier, for example with `s_waitcnt lgkmcnt(0)` or
`s_wait_dscnt 0x0` ahead of `s_barrier`. The checker does not require it: a wave
that reaches a barrier with LDS operations in flight is not reported for that
alone.

A barrier orders only operations whose completion the issuing wave has
established, through its waits or through counter capacity. An operation still
in flight when the barrier completes is carried past it and stays unordered with
the other members' later accesses until a barrier that follows its completion.
It is reported in only two cases:

- Another wave of the workgroup makes a conflicting access to the same bytes
  before that later barrier. The finding recommends a completion wait, which is
  the hygiene fix.
- The issuing wave itself touches the bytes through a different completion
  class before its wait, for example by reading the destination of an async
  direct-to-LDS or tensor-DMA write with a DS load.

Ordinary DS operations from one wave complete in order, so the wave's own later
DS accesses never conflict with its in-flight DS operations. Nothing is reported
for this sequence unless a different wave accesses the slot before the next
barrier that follows the write's completion:

```text
ds_write_b32 v0, v1     ; this wave's own slot
s_barrier               ; no s_waitcnt lgkmcnt(0) first
ds_read_b32  v2, v0     ; same wave, same slot: ordered
```

There are two reasons not to report the barrier itself:

- If no other wave touches the bytes, no schedule produces a different result,
  so there is no race to report. The two `lds_war_pattern` mutants that remove a
  pre-barrier wait are equivalent mutants for this reason: every thread reads
  only its own slot. The harness exempts them in `_KNOWN_PASSING_MUTANTS`, and
  moving the final read to another wave's slot makes the same mutants report RAW
  findings.
- Multi-buffered pipelines keep asynchronous copies in flight across barriers
  on purpose: a direct-to-LDS load or tensor-DMA transfer into the next buffer
  overlaps the barrier that guards the current one. Reporting every in-flight
  operation at a barrier would flag those kernels on every iteration.

What the default gives up:

- Fragile code goes unreported. Code that is correct only because no other wave
  touches the bytes becomes racy when a later change makes one do so. The
  checker reports that change, but only on a run that executes the new access.
- The check is dynamic. If the other wave's access happens only for some inputs
  or paths, runs that don't execute it report nothing, whereas a hygiene rule
  would flag the barrier on every run.
- Mutation testing needs curation. Removing a pre-barrier wait can produce an
  equivalent mutant, and each one needs a manual exemption. Exemptions are
  maintained by hand and can go stale.

If hygiene reporting is wanted, it should be kept separate from race findings
and be off by default. Possible forms:

- An opt-in dynamic report. When a barrier completes, the checker already visits
  each member's pending operations to carry them past it, so it could report
  ordinary DS operations that cross a barrier, once per barrier and producer PC.
  Direct-to-LDS and tensor-DMA writes should be excluded, because crossing
  barriers is their normal use. Expect reports on hand-pipelined kernels that
  issue the next stage's DS writes before the barrier that guards the current
  stage.
- A static rule in `rj_waitcheck` that flags a DS producer whose counter is not
  drained on some path to `s_barrier`. It would cover every path at no
  simulation cost, with the same false positives.
- A separate mutant class in the harness for removed pre-barrier waits, judged
  by the hygiene rule rather than by race detection.

## Synchronization the checker does not model

These gaps produce reports for correctly synchronized code. Apart from a wave's
own program order and waits, the only ordering the checker models is a barrier
between waves of one workgroup, and it applies that ordering only to LDS.

### Cross-workgroup synchronization

Global accesses from different workgroups of one dispatch are compared without
any ordering between them. Two accesses that overlap, where at least one writes
and they are not both atomic read-modify-writes, are reported whatever
synchronizes them. Correct code that orders workgroups through memory is
therefore reported, for example:

- release/acquire flag handoffs between producer and consumer workgroups;
- cooperative-groups grid synchronization;
- stream-K or split-K fix-up that passes partial tiles through a workspace;
- decoupled look-back scans.

Intentional benign races are reported as well, such as many workgroups storing
the same value to a "found" flag.

Closing this gap needs happens-before for global memory: per-workgroup vector
clocks, release and acquire edges from atomics and fences with their scopes, and
the cache write-back and invalidate instructions (for example `buffer_wbl2` and
`buffer_inv`) that implement those fences on each target. Without the cache
part, a handoff that is ordered but misses an invalidate would be accepted.

### Atomic loads and stores

AMDGPU has no separate atomic load or atomic store instruction. Compilers lower
them to ordinary loads and stores, such as `global_load_*`, `global_store_*`,
`ds_read*` and `ds_write*`, adding cache-policy bits where the scope needs them.
Only read-modify-write atomics reach the checker as atomic. A flag that one
workgroup sets with an atomic store or `atomicExch`, and another polls with an
atomic load, is therefore reported in addition to the data the flag protects.
The same applies to flags in LDS.

Recognizing atomic loads and stores would need metadata from the compiler, or a
cache-policy heuristic that would also match `volatile` and nontemporal
accesses.

### LDS synchronization other than barriers

Within a workgroup, only barriers order LDS accesses between waves, because they
are the only operation that merges the checker's vector clocks. Waves that hand
off LDS data through a flag, a counter or a spin-wait on an LDS atomic, such as
producer and consumer waves in a warp-specialized kernel, are reported. As with
global memory, the flag accesses themselves can be reported, because atomic
loads and stores look like plain DS accesses.

### gfx1250 LDS barrier cells

gfx1250 adds LDS barrier cells: 64-bit LDS words that hold a phase and a pending
count. Three operations arrive at a cell:

- `ds_atomic_barrier_arrive_rtn_b64`;
- `ds_atomic_async_barrier_arrive_b64`, an arrive tracked on the asynchronous
  operation counter;
- a tensor-DMA transfer whose descriptor requests an arrive when the transfer
  completes.

The checker sees the first two as ordinary LDS atomics. It does not see the
tensor-DMA arrive at all, because the transfer writes the cell directly rather
than through a routed memory instruction. Waiting for a cell's phase creates no
ordering, so data handed off through a cell is reported when a wave other than
its producer reads it.

The checker also learns that an operation completed only from the issuing
wave's waits and counter capacity. A tensor-DMA or async copy that the issuing
wave never waits for, because consumers rely on the cell instead, stays in
flight for the rest of the workgroup's life. Every other wave's read of its
destination is then reported, even after later barriers.

Modeling cells means recording each arrive as a release, including the
tensor-DMA arrive, treating a wave that observes the completed phase as
acquiring them, and marking the operations an arrive covers as complete.

## Races the checker does not see

### Split barriers

On targets with split barriers, `s_barrier_signal` does not block; only
`s_barrier_wait` does. The checker applies a barrier's ordering once, when the
last member signals, by merging every member's current vector clock. It does not
record where each member signaled or waited. Two kinds of access in the window
between a member's signal and its wait are therefore treated as ordered when
they are not:

- An access a member makes after signaling but before the barrier completes.
  Other members' accesses after the barrier count as ordered after it.
- An access a member makes after the barrier completes but before its own
  `s_barrier_wait`. It counts as ordered after the other members' accesses from
  before the barrier.

Whether a race in that window is reported depends on how the simulator happens
to interleave the waves. Compiled `__syncthreads()` normally places the signal
and the wait next to each other, so the gap matters for code that schedules work
between them. Closing it means taking a snapshot of each member's clock at its
signal and applying the merged clock to each member at its wait.

### Global accesses within a workgroup

The global shadow never compares an access with records from its own
workgroup, so global hazards inside one workgroup are not reported. An example
is waves of a workgroup that exchange partial results through a global workspace
without a barrier between the write and the read. Same-wave global accesses are
not compared either: where a target does not keep a wave's loads and stores to
one address in order, a missing wait between them is not reported. The
memory-wait checker still reports register-level hazards on the same
instructions.

Closing this gap means applying the workgroup's barrier ordering to global
memory as well as LDS, together with the visibility rules that make one wave's
global write visible to another: cache-policy bits, cache invalidates, and
threadgroup-split mode, where a workgroup's waves can run on different compute
units.

### Accesses outside one dispatch

Each GPU/SoC owns one checker. Its global history is keyed by the SoC-wide
dispatch ID and discarded when that dispatch retires. Accesses are therefore
not compared with:

- other dispatches, including concurrent kernels on other queues and AQL packets
  in one queue that may overlap because they lack the barrier bit;
- host reads and writes of fine-grained or coherent memory while a kernel runs;
- SDMA copies and other DMA engines;
- other GPUs and other processes, including memory shared through IPC.

Accesses to one physical page through two different virtual mappings are not
compared with each other.

### Cluster multicast into peer LDS

A direct-to-LDS load with a cluster multicast mask writes the LDS of every
selected workgroup in the cluster. The checker records the write only in the
issuing workgroup's LDS history, and only when that workgroup is selected.
Writes into peer workgroups' LDS are not recorded. A peer that reads the data
before the copy completes is missed, and so is a WAW between two multicasts into
one peer buffer. The legacy race-detector plugin warns once that it ignores peer
writes; the core checker says nothing.

Closing this gap means recording each peer write in the peer workgroup's LDS
history with the issuing wave's completion token, and letting cluster barriers
order waves across workgroups. Today a cluster barrier orders each workgroup's
members only among themselves.

### Limits inherited from the functional model

- The simulator routes a whole FLAT instruction by its first requesting lane.
  When that lane is global, lanes that address the shared (LDS) aperture are
  recorded as global accesses at their aperture addresses. Those addresses are
  the same in every workgroup, so two workgroups that each use their own LDS
  this way can be reported as a cross-workgroup race. The simulator also sends
  every lane of such an instruction through one pipeline.
- On gfx1250, FLAT lanes in the upper half of the shared aperture (direct data
  share, DDS) are routed to the LDS pipeline, but their translated addresses lie
  outside the workgroup's LDS allocation, so the checker drops them. DDS
  accesses are not checked.

### Dynamic coverage

Only the accesses that a run executes are checked. Paths, inputs, problem sizes
and launch configurations that the run does not exercise are not covered, so
races that need a tail tile, a particular grid size or a data-dependent branch
need a run that takes them.

The simulator also runs one schedule. Apart from the split-barrier window above,
LDS checking compares accesses by happens-before rather than by the order they
ran, and global checking ignores order, so a race between two executed accesses
is found whichever one runs first. Control flow that depends on timing, such as
spin-waits, work claimed through atomics, or which workgroup arrives last at a
fix-up step, follows only the outcome this schedule produces.

## Bounded state

### Global shadow budget

`race_checking_global_shadow_mb` (512 MiB by default) bounds the global shadow's
allocated storage. Once it is reached, new accesses are still checked against
retained records but are no longer retained, so a later access that conflicts
only with an unretained one is missed. The checker then prints a warning and
sets `coverage_degraded` in the JSON report.

- The default is too small for realistic shapes. All 20 gfx950 Gluon tutorial
  kernels reached it in the historical, final P2, and final P3 campaigns at the
  shapes in [Performance](#performance), while a 4 GiB budget retained every
  access for all of them.
- One atomic checker-wide cap is shared by all 256 page-hash shards, so a busy
  shard can use capacity that idle shards do not need. Reservations cannot
  overshoot the cap, and retiring a dispatch releases its exact modeled charge
  for immediate reuse.
- The cap charges 1 KiB page slabs, 16-byte records, 16-byte interned
  provenance sites, and open-addressed page/site directories. Site reuse can
  make the 16-byte record smaller than the old 24-byte inline-provenance
  record, while sparse pages and low site reuse still incur slab and directory
  slack. The exact amplification is therefore workload- and shard-distribution
  dependent; the cap is not a process-RSS limit.
- Allocator metadata, small per-dispatch owner objects, slab-descriptor vectors,
  mutexes, and fixed checker caches are not charged. The JSON report exposes
  successful and failed reservations, releases, slab/directory growth, and
  dispatch release/peak values so this boundary is explicit.
- The warning prints once per checker (one checker per GPU/SoC), and
  `coverage_degraded` remains a single flag for that GPU's whole report. P0
  statistics identify retained and rejected bytes by shard for the eight most
  recent completed dispatches with address-race work, but the cumulative
  report does not retain per-dispatch detail for every earlier dispatch.

Sizing the default from the dispatch's footprint or from host memory, bounding
the excluded owner/allocator overhead, and retaining explicit coverage status
for each dispatch would address the remaining gaps.

### LDS history size

LDS history is outside the global budget and has no limit or warning of its own.
Each workgroup keeps one 24-byte record per wave, access kind and touched LDS
dword, plus a dense index over the touched window. A barrier that every active
wave joins drops the history; a named barrier drops only the records that every
active wave has already observed. A tile that every wave reads costs one record
per reading wave: eight waves reading a 64 KiB tile hold 3 MiB until the next
full barrier, in every resident workgroup. Part of the added memory in the Gluon
measurements is this history.

## Reporting

### Finding limits

- At most 64 distinct findings are kept at a time, across the dispatches in
  flight. A finding is distinguished by space, kind, dispatch, producer PC,
  consumer PC and address. One racy instruction pair that touches many
  addresses, such as a loop over a tile, therefore produces a separate finding
  for each starting address. It can fill the limit and hide every other pair in
  that dispatch. A dispatch's findings are released when it retires, so later
  dispatches have room again.
- The "further diagnostics are suppressed" warning prints once per GPU/SoC
  checker. When a later dispatch reaches the limit, nothing is printed; only the
  JSON `suppressed` count keeps rising.
- The JSON report keeps at most 1,024 memory-wait and 1,024 memory-race
  findings. Its shared `suppressed` counter includes either vector overflowing
  and races rejected by the 64-key cap. `observed` counts findings delivered to
  the report plus explicitly suppressed race events; repeated race keys
  deduplicated before reporting are not counted.

Keying findings by instruction pair, and reporting the address range each one
covers, would let one racy pair use one slot.

### Finding content

- PCs are the raw virtual addresses of the instructions. Findings carry no kernel
  name, code-object offset or instruction text, so mapping them to source needs
  the code object's load address. The legacy plugin prints kernel names and
  disassembly.
- LDS addresses include the workgroup's LDS allocation base on its compute unit,
  which is not printed. Subtract it to get the kernel's LDS offset.
- A newer access by the same wave (LDS) or workgroup (global), of the same kind
  and covering the same bytes, replaces the older record. The reported producer
  can therefore be a later access than the one that first raced.
- Once records of one access kind from two workgroups cover a global byte,
  accesses of that kind by further workgroups are checked but not recorded. A
  finding names one of the racing workgroups, not all of them.
- For global findings, which access is the producer, and whether the finding is
  RAW or WAR, reflect the order in which the simulator ran two unordered
  accesses. On hardware either can come first.

## Modes and lifecycle

- Top-level `race_checking` and `--race-checking=on|off` override every CU.
  Without either, each CU's `memory_race_diagnostics` policy applies. Enabled
  CUs still share one checker and one address history for the whole GPU/SoC.
- Top-level `race_checking_checks` and `--race-checking-checks` select `global`,
  `lds-same-wave`, `lds-cross-wave`, `direct-to-lds`, and `tensor-dma`
  families. The last two add transfer sources to selected address domains and
  cannot be used alone. Omission selects all families. The selector is intended
  for overhead attribution and for deployments that deliberately accept the
  documented loss of unselected coverage.
- The two `--race-checking` overrides are refused with `--attach`. In DBT guest
  mode they are also refused when the guest config names a host
  `simulator_config`; set `race_checking` and `race_checking_checks` in that
  config instead. `--wait-checking` has the same rules.
- Address history is not saved in checkpoints and is reset on restore, so a race
  whose first access predates the restore is missed. In addition, the checkpoint
  format does not preserve a resident wave's LDS allocation base and size.
  Restored waves default to a zero-sized allocation, and the core checker clips
  their later LDS ranges to empty. Post-restore LDS races are therefore outside
  coverage even when both conflicting accesses occur after the restore. Global
  checking remains enabled conservatively when selected.
- Checkpoints preserve the effective family selection and global shadow budget,
  but not the diagnostic report path; a VM restored through
  `rj_vm_restore_checkpoint` does not write a JSON diagnostic report.
- One report is written per GPU. GPU 0 uses the configured
  `memory_diagnostic_report` path; additional GPU `N` uses `<path>.gpuN`.
- Enabling the legacy race-detector plugin together with core checking
  duplicates reports and adds callback overhead.

## Performance

See the dedicated
[address race checker performance analysis](address-race-checking-performance-analysis.md)
for the full comparison, hotspot analysis, memory-amplification calculation,
optimization roadmap and sign-off protocol.

The campaigns below answer different questions. The 20-kernel campaign measures
a heterogeneous realistic workload mix on the checker before the final
SoC-shared ownership and lifecycle refactor. The synthetic campaigns isolate
memory-bookkeeping cost. A targeted representative rerun supplies a limited
same-kernel comparison. Percentages are relative to the same binary with race
checking off unless stated otherwise. These results are current through
2026-10-20.

### Pre-final-refactor realistic campaign

These figures came from 20
[gfx950 Gluon tutorial kernels](https://github.com/ROCm/gfx950-gluon-tutorials),
run with a Release build and `configs/gfx950_mi355x.json` (256 compute units).
The GEMMs were 4096 × 4096 with K = 4096 for fp16 and 8192 for fp8 and MXFP4.
FMHA used 16 heads, sequence length 4096 and head dimension 128, about 1/63 of
the tutorial's benchmark size. Each configuration used two processes with two
timed dispatches each; each kernel value is the median of the four pooled
dispatch times.

| Configuration | Time per dispatch, geomean (range) | Added peak memory |
|---|---|---|
| Off, compared with contemporaneous `develop` | 0.97×, within noise | Not measured |
| On, 512 MiB default budget | +20.4% (+4% to +52%) | 0.7 to 1.3 GiB |
| On, 4 GiB budget | +26.2% (+3% to +73%) | 1.4 to 2.1 GiB |

Peak memory with race checking off was 2.0 to 2.2 GiB. Checking adds a roughly
constant 0.5 to 1.9 s per dispatch at these shapes. The relative cost at the
default budget is therefore highest for kernels that are cheap to simulate (FMHA
+49%, fp16 GEMMs +24%) and lowest for expensive ones (fp8 GEMMs +12%, MXFP4
GEMMs about +6%).

All 20 kernels' default-budget configurations warned that retention degraded;
none of the 4 GiB configurations did. The campaigns produced no race findings.

These measurements are the historical realistic baseline, but they are not a
clean before/after measurement of the final refactor. In particular, the older
checker was owned per command processor while the current checker is shared by
all command processors and CUs of a GPU/SoC. The final P3 rerun below supersedes
them for current overhead.

### Pre-P0/P1 synthetic baseline

Four memory-intensive gfx950 workloads (stream, LDS reduction, GEMM and
histogram) were run against the post-redesign, pre-P0/P1 Release binary with
checking on and off. The small set used 262,144 elements (GEMM 128); the large
set used 1,048,576 elements (GEMM 256). The geomeans deliberately exclude a
compute-only control:

| Execution | Small memory workloads | Large memory workloads |
|---|---:|---:|
| Default dispatch threading | +37.6% | +54.1% |
| Serial dispatch | +41.1% | +48.4% |

The compute-only control was +1.2% with default threading and -2.8% serial,
consistent with noise. In the less noisy serial comparisons, the current
race-off path was -0.03% versus `develop` across the five small workloads and
+1.9% across the four large workloads. This is evidence that the enabled path,
not a general disabled-path regression, causes the slowdown.

Combining those serial baselines, race-on was +40.7% on the small memory set and
+51.2% on the large set versus `develop`. Those are the source of the earlier
"about 41% / 51%" headline; the +41.1% / +48.4% on/off values above isolate the
pre-P0/P1 checker's incremental cost.

These numbers are intentionally not comparable to the 20-kernel geomean: the
synthetics maximize accesses per unit of unrelated simulation work. The earlier
realistic campaign already contained individual kernels at +52% with the
default budget and +73% at 4 GiB.

An earlier stabilized serial pass measured about +37% on the four small memory
workloads and +30% when the compute control was included; the latest all-five
geomean was +31%. The memory-only spread is another reason to treat the
loaded-host synthetic figures as ranges rather than a refactor regression.

### P1 synthetic rerun

The final P1 binary, the same binary with checking off, and original `develop`
were compared in one pinned, rotating nine-repetition matrix. All 216 cases
passed and no checked run degraded coverage:

| Comparison | Small memory workloads | Large memory workloads |
|---|---:|---:|
| P1 race-off / `develop` | +10.4% (-4.1% to +17.1%) | -0.4% (-5.9% to +3.1%) |
| P1 race-on / race-off | +26.3% (+17.0% to +43.3%) | +41.0% (+34.7% to +43.0%) |
| P1 race-on / `develop` | +39.4% (+26.1% to +48.2%) | +40.4% (+32.3% to +42.2%) |

The small set remained bimodal; its +10.4% race-off point estimate is not a
resolved disabled-path regression because the interval includes zero. The
longer large set is stable. The latest +39.4% / +40.4% race-on-versus-`develop`
result is directionally lower than the older +40.7% / +51.2% headline, but
shared-host variation prevents attributing that whole difference to P1.

### Current mutation-harness sweep

A separate serial gfx1250 sweep used 21 small mutation shaders, five processes
per variant, and ordinary wait checking in both variants. Enabling address race
checking added 3.7% geomean whole-process wall time (bootstrap 95% interval
about +1.9% to +5.4%) and did not exhaust the global shadow. These CI-sized
kernels show the low-cost end of the workload range; their whole-process timing
is not comparable to the Gluon per-dispatch measurements.

### Earlier two-kernel representative rerun

FMHA v4 and A16W16 v9 were rerun on an earlier checker build with one process per
variant and three timed dispatches; the table uses their medians. The 512 MiB
runs emitted the coverage warning; the 4 GiB runs did not. This status comes
from raw stderr: the benchmark helper's `degraded` field still matched the old
warning text and incorrectly remained false.

| Kernel | Race-off | 512 MiB overhead | 4 GiB overhead | Added peak RSS, 512 MiB / 4 GiB |
|---|---:|---:|---:|---:|
| FMHA v4 | 2.844 s | +42.3% | +50.9% | 1.43 / 1.58 GiB |
| A16W16 v9 | 3.731 s | +29.7% | +34.8% | 1.36 / 2.38 GiB |
| Two-kernel geomean | — | +35.9% | +42.6% | — |

For the same two kernels, medians from the pre-final-refactor campaign give
+49.2% at the default budget and +53.4% at 4 GiB. The earlier rerun is therefore
directionally faster, not evidence that the refactor increased overhead. It is
still not a controlled redesign A/B: the runs were collected at different
times, and the earlier rerun has only one process per variant. The final P3
campaign below supersedes it for current overhead.

### P0/P1 fixed-work instrumentation

The optional JSON report now includes bounded per-dispatch counters for ranges,
dwords, shadow work, allocations, lock acquisition, barriers, cleanup, and
retained or rejected bytes.

The original P0 harness accidentally enabled reporting for names ending in
`-no-report`; its no-report and report-overhead conclusions are invalid. A
corrected, explicit four-variant P0/P1 matrix found P1 was 2.0% faster on the
small set and 0.7% faster on the large set without a report, with both
intervals including zero. With a report, P1 was 4.2% and 4.9% faster; only the
large interval excluded zero. Report collection added 7.4% for P0 and 2.9% for
P1 on the large set. All 288 runs passed with no rejected shadow bytes.

The counters give stronger optimization priorities than the software profile
alone. Global-only stream and histogram dispatches created thousands of LDS
workgroup states. GEMM generated 37.82 million dword candidates and rejected
28.31 million (74.9%) as duplicates. Large dispatches inserted 1.05 to 7.67
million records, GEMM visited 10.11 million records, and LDS reduction incurred
205,675 vector growths. Serial map, state, and global-shard lock acquisition
averaged 26 to 28 ns including timing overhead, with no sign of contention.

P1 now creates no LDS state for zero-LDS stream and histogram workgroups.
Batched setup cuts clock-matrix growth by 75%; cached state/wave bindings cut
large LDS/GEMM workgroup lookups by 90% to 98%; and the aligned GEMM fast path
skips its 74.9% overlapping prefix before dword expansion. A one-workgroup
global dispatch records no race work at all. `dword_candidates` now excludes
prefixes skipped before expansion, while `dword_duplicates_skipped` includes
them. The generic path and exact `dword_visits` semantics are unchanged.

### P2 global-shadow result

P2 replaces global page hash nodes and relocatable record vectors with
per-dispatch flat page/site directories and bounded slabs. It also replaces 256
isolated budget shares with one exact checker-wide modeled cap. Dispatch
cleanup returns its charge immediately, and a bounded saturation cache avoids
repeating known-failed growth attempts without skipping retained-record checks.

In the controlled P2/P1 matrix, P2 without JSON collection was 16.5% slower on
the noisy small set and 11.1% slower on the stable large set. With collection,
it was 1.3% and 10.6% slower. Exact site interning is the main new hot path:
large stream was 29.1% slower, while site-reusing GEMM changed by 1.0%.
Compact 8-byte site-directory entries and removal of duplicate site probes and
per-record tracking atomics recovered seven percentage points from the first
large-set P2 result.

The layout improves storage for the site-heavy large stream: report-run peak
RSS fell from 330.6 MB on P1 to 315.6 MB on P2, and its modeled peak was 105.7
MB. A separate 512-MiB/4-GiB control had zero rejected bytes in all 80 cases;
the large-set timing differed by 1.1%. Thus the P2/P1 timing comparison is at
equal full coverage, not an artifact of earlier budget exhaustion.

The final realistic rerun used the same 20 Gluon kernels and shapes as the
historical campaign, with five rotating processes and two timed dispatches per
variant. P2 race-off was +1.4% versus `develop` (-0.8% to +2.5%), which does not
resolve a disabled-path regression. Race-on was +19.5% versus race-off at the
default budget (+18.2% to +21.5%) and +20.0% with full 4 GiB coverage (+18.6%
to +22.1%). Against `develop`, the corresponding geomeans were +21.1% and
+21.6%.

All 400 processes passed with matching output and no finding. Every
default-budget race-on process degraded retention; no 4 GiB process did.
Full/default timing differed by +0.4% (-0.4% to +1.3%). Added dispatch-window
peak RSS was 0.58--1.12 GiB at the default and 0.83--1.21 GiB at 4 GiB,
compared with 0.7--1.3 GiB and 1.4--2.1 GiB in the historical campaign.

### P3 compact-access result

P3 represents a regular access from consecutive active lanes as one affine
`(base, stride, size, first lane, count)` descriptor. Aligned descriptors use
compact dword runs and in-flight LDS operations keep the descriptor through
barrier carry. Sparse, descending, irregular, element-masked, DS2,
per-lane-clipped, and tensor-DMA accesses retain explicit ranges. Unaligned
affine accesses keep exact generic byte-mask iteration.

A pinned rotating P3/P2/`develop` matrix ran all eight serial synthetics nine
times. All 288 processes passed with full retained coverage. P3 race-on versus
P2 race-on was +0.2% on both the small and large sets; the bootstrap intervals
(-7.3% to +5.5% and -2.3% to +2.9%) include zero. No elapsed-time change
resolves.

The representation itself is active: on the largest stream and LDS reduction,
97.0% and 99.1% of logical lane ranges were affine. LDS reduction kept 131,072
pending operations compact and removed 122,880 range-vector growths. GEMM was
entirely explicit, however. More importantly, P2 and P3 performed exactly the
same dword visits and record insertions/scans on every synthetic. P3 therefore
establishes a correctness-tested compact boundary but does not yet batch the
per-dword shadow work or exact site interning that dominates runtime.

The final P3 realistic campaign repeated all 20 Gluon kernels with five
rotating processes and two timed dispatches per variant. Race-off was +0.2%
versus `develop` (-1.0% to +2.3%). Race-on was +21.2% versus race-off at the
default budget (+19.4% to +22.9%) and +21.8% with full 4 GiB coverage (+19.8%
to +23.1%); the corresponding comparisons with `develop` were +21.4% and
+22.0%. Full/default timing differed by +0.5% (-0.9% to +1.4%).

All 400 processes passed with matching output and no finding. Every
default-budget process degraded retention and no 4 GiB process did. Added
dispatch-window peak RSS was 0.58--1.08 GiB at the default and 0.84--1.20 GiB
at 4 GiB. Normalized to each campaign's `develop`, P3 differs from P2 race-on
by only +0.2% at the default and +0.3% at 4 GiB. The direct synthetic P3/P2
comparison is also flat, so no P3 runtime gain or regression resolves.

### Post-P3 lock-scope audit

The synchronization audit keeps only shared-state and simulator-ordering work
inside race-check locks. JSON formatting and file I/O use a revisioned snapshot
outside the report-state mutex. Warning logs, report updates, and user reporter
callbacks run after checker locks are released. Dispatch cleanup detaches
workgroup/global state under its map or shard lock, then destroys slabs,
aggregates statistics, and returns budget charge outside it. Command-processor
report flushes are deferred until after the queue lifecycle mutex, including
direct VM faults and failed queue-close paths.

The CU wave-state lock still covers each instruction's simulated issue,
scoreboard, barrier, and wave lifetime. The workgroup-state mutex still protects
LDS clocks and records, and one global-shard mutex still protects the records
being traversed and updated. Those are shared resources, not local preparation.
Parallel workgroup completion also skips a report-pending atomic exchange; the
worker join performs retirement and flushing.

Four direct lock-scope/P3 matrices found no resolved timing change:

| Geomean audited/P3 | Small set | Large set | All eight |
|---|---:|---:|---:|
| Serial, no JSON | -1.0% (-8.7% to +5.6%) | +0.2% (-1.9% to +2.4%) | -0.4% (-4.5% to +3.0%) |
| Threaded, no JSON | +0.4% (-4.4% to +4.7%) | +0.3% (-5.0% to +4.4%) | +0.4% (-3.4% to +3.2%) |
| Serial, JSON statistics | -1.5% (-10.7% to +16.5%) | +1.7% (-0.4% to +4.1%) | +0.1% (-4.8% to +8.9%) |
| Threaded, JSON statistics | -1.4% (-4.4% to +4.1%) | -0.8% (-7.8% to +6.0%) | -1.1% (-4.7% to +3.3%) |

All 864 processes passed with expected output and full retained coverage.
Serial measured acquisitions averaged about 26--35 ns. Under threaded JSON
instrumentation, workgroup-state locks remained about 30--37 ns and map locks
about 97--171 ns, but global-shard acquisitions ranged from 0.30 to 5.81
microseconds. The audit therefore removes avoidable blocking and deadlock
exposure without resolving elapsed time. Global contention is real for some
threaded workloads, but its remaining critical section is the required
per-dword shadow traversal rather than reporting, cleanup, or local work.

### Selectable-family overhead

The current checker can time and deploy its address domains and transfer
sources separately through `race_checking_checks`. A full-coverage matrix ran
all eight relevant selector combinations over the 20 Gluon kernels, with five
rotating processes and four timed dispatches per process:

| Selection | Geomean overhead versus race-off |
|---|---:|
| `global` | +2.22% (+1.14% to +3.02%) |
| `lds-same-wave` | +2.55% (+1.43% to +3.47%) |
| `lds-cross-wave` | +6.13% (+5.01% to +7.15%) |
| Both regular LDS families | +6.19% (+5.18% to +7.21%) |
| All address domains without direct-to-LDS | +8.99% (+7.66% to +9.96%) |
| Direct-to-LDS source increment | +12.67% (+11.55% to +13.72%) |
| `all` | +22.81% (+21.44% to +23.62%) |

All 800 processes passed with matching per-kernel digests, no finding and no
coverage degradation. Adding same-wave LDS after cross-wave LDS was only
+0.05% (-0.86% to +1.07%), but the prerequisite cross-wave family was already
+6.13%; the combined option is not below 2%.

The gfx950 tensor-DMA source increment was +0.80% (+0.13% to +1.46%), but
gfx950 executes no tensor-DMA instruction, so that is a dormant-source noise
control. A dedicated gfx1250 workload measured +7.46% (+1.76% to +9.66%) for
the active source with cross-wave LDS and +20.89% (+15.08% to +29.39%) with
all address domains. Its same-wave-only upper bound was +3.97%.

The acceptance criterion requires both the estimate and 95% upper bound to be
at most 2% on a workload that actively executes the check. No independently
useful active family qualifies. The
[performance analysis](address-race-checking-performance-analysis.md#selectable-family-overhead-and-2-decision)
contains the synthetic, focused-source, and per-kernel evidence.

### Software profile

Hardware `perf` sampling was unavailable (`perf_event_paranoid=4`), so a
userspace SIGPROF sampler was used on the exact pre-P1 Release binary. After
normalizing to samples that resolved inside `librocjitsu.so`, direct
race-checker functions accounted for:

| Workload | Direct race-check code | Largest resolved costs |
|---|---:|---|
| Stream | 27% | `global_access` 22% |
| LDS reduction | 27% | global access 7%, LDS record 6%, `lds_access` 5% |
| GEMM | 40% | `lds_access` 14%, global access 7%, LDS record 7%, range collection about 6% |

These are lower bounds for checker cost because allocator and mutex time that
resolves to libc is not attributed to the caller. The samples identify
per-dword shadow traversal, LDS history updates and lane/range preparation as
hot paths. They do not measure lock wait directly; the post-P3 counter matrices
above provide that measurement and show workload-dependent global-shard
contention without a resolved aggregate timing change.

### Measurement limits

The shared measurement host was heavily loaded (load averages roughly 400–560
during parts of the synthetic/profile campaign, about 101 during the
representative rerun, 44–47 during the final P1 matrices, 43–70 during the
final P2 Gluon matrix, and 42–49 when the final P3 matrix started on 384
logical CPUs). The selector campaign started and ended near 46. A
sign-off run needs reserved cores, fixed CPU policy, interleaved on/off variants
from one binary and at least five processes per variant. It must record coverage
warnings and peak RSS with time: a faster capped run can simply have retained
less history. gfx1250, wave32, and multi-GPU remain unprofiled. The post-P3
matrix now measures gfx950 threaded lock contention, but its unpinned intervals
remain wide.

## Validation

The final selector build passes the 174-test complete core race/wait/config and
checkpoint selection, the 250-test combined legacy/core race selection, the
CLI selector CTest, and all 41 mutation-tool unit tests. The gfx950 matrix
tested 52 mutants and the gfx1250 matrix tested 83; both had clean baselines,
no inconclusive run, no false positive, and only the two known
`lds_war_pattern` equivalent misses.

Configuration and execution tests cover every family spelling, canonical
formatting, source-without-domain rejection, omitted-selector compatibility,
checkpoint round-trip, CLI restrictions, family-specific findings, disabled
direct/tensor sources, and LDS-only dispatch finalization. All 2,126 selector
performance cases passed: 576 general synthetics, 540 focused active paths,
210 direct/tensor source cases, and 800 Gluon processes. None emitted a finding
or coverage warning, and every Gluon kernel had one digest across all selectors.

The earlier focused threaded probe still passes AddressSanitizer and
UndefinedBehaviorSanitizer with leak detection. ThreadSanitizer binaries build,
but its runtime cannot initialize on this host. The selector change did not
alter the shadow layouts exercised by those probes.

- The mutation harness has two mutation kinds: removing one wait instruction,
  and rewriting a load as a store. It never removes a barrier, which is the usual
  cause of a cross-wave LDS race. Core execution tests cover no synchronization,
  a barrier with an in-flight write, and a completion wait followed by a barrier.
  The wait mutants of `lds_reduce.hip` and `fa_barrier_epoch.hip` add harness
  coverage, the latter on gfx1250 only.
- Only `global_byte_eviction.hip` exercises cross-workgroup global races in the
  harness, through its load-to-store mutants. Core execution tests also compare
  ordinary cross-workgroup stores with atomic read-modify-writes and verify that
  dispatch retirement releases both LDS and global history.
- No race has been injected into a realistic kernel. The Gluon runs show that
  those kernels produce no false reports, and measure the cost, but they do not
  show that a race in a kernel of that size is found, in particular after the
  global shadow degrades.
- Budget tests cover degradation at zero, the exact 21,888-byte first-retention
  boundary, checker-wide shard skew, concurrent admission without overshoot,
  transactional record/site failures, partial-dispatch cleanup and immediate
  reuse. Separate tests cover packed provenance, directory/slab growth, and
  saturation-cache hits and invalidation.
- Compact-representation tests compare affine and explicit global findings
  across pages, verify unaligned byte masks, overlap and broadcast first-lane
  provenance, compact pending LDS carry across repeated barriers, fallback
  execution, accounting, and JSON serialization.
- Report tests cover a combined single-GPU file, non-clobbering when unused,
  flush at ordinary grid retirement, and LDS-only statistics finalization when
  global checking is unselected. There is no end-to-end multi-GPU test of the
  `.gpuN` report paths.
- There are no tests for split barriers with work between the signal and the
  wait, for LDS barrier cells, or for cluster multicast into peer workgroups.

## Suggested order of work

Correctness and coverage:

1. Raise or auto-size the default global budget, and report degradation per
   dispatch.
2. Key findings by instruction pair, and add kernel names and code-object
   offsets to findings.
3. Serialize or reconstruct resident LDS allocation metadata at checkpoint
   restore so post-restore LDS checking resumes with empty history.
4. Add barrier-removal mutants to the mutation harness.
5. Model gfx1250 LDS barrier cells, including the tensor-DMA arrive.
6. Apply split-barrier ordering at each member's own signal and wait.
7. Record cluster multicast writes in the peer workgroups' LDS history.
8. Add happens-before for global memory. This removes the largest class of false
   reports, and is a prerequisite for checking global accesses within a
   workgroup.

Performance, preserving the same findings and retained coverage:

P1 added zero-LDS and one-workgroup no-work paths, LDS-only completion tokens,
aligned overlapping-prefix skipping, batched workgroup setup, and cached
lifetime-safe state/wave bindings. P2 added per-dispatch global slabs, flat
directories, packed records and sites, checker-wide exact accounting, immediate
budget reuse, and the saturation cache. P3 added affine global/LDS descriptors,
aligned dword runs, compact pending LDS operations, explicit fallback, and
representation counters.

1. Batch the global-read and deferred-LDS-write effects of direct-to-LDS, which
   adds 12.7% to the realistic geomean, and reuse one prepared provenance/site
   context across both effects.
2. Consume affine descriptors in page/shard and dense-LDS batches so compact
   runs reduce shadow lookups, callbacks, and record-insertion overhead rather
   than only range materialization. Add fallback-reason counters before
   extending classification to GEMM's currently explicit patterns.
3. Stable-bucket and coalesce compatible explicit ranges by page/dword while
   retaining the generic path for gathers, overlap, unaligned accesses and
   per-lane diagnostics.
4. Reuse barrier scratch and index pending LDS operations by range so accesses
   visit only potentially overlapping incomplete operations.
5. Reduce optional report-collection work only if its remaining cost matters
   for profiling runs; report snapshots and file I/O are already outside the
   state lock, and collection is absent when no report is configured.
6. Evaluate compact owner/conflict summaries and epoch-based LDS history with
   differential tests. Threaded lock-wait instrumentation shows material,
   workload-dependent global-shard contention, but the lock-scope audit is
   timing-neutral. Batch more shadow work per acquisition before changing lock
   topology.
