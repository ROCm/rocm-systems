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
| [Cluster multicast into peer LDS](#cluster-multicast-into-peer-lds) | Missed races |
| [Limits inherited from the functional model](#limits-inherited-from-the-functional-model) | Missed races and false reports |
| [Dynamic coverage](#dynamic-coverage) | Missed races |
| [Global shadow budget](#global-shadow-budget) | Missed races once the budget is reached |
| [LDS history size](#lds-history-size) | Unbounded memory |
| [Finding limits](#finding-limits) | Hidden findings |
| [Finding content](#finding-content) | Harder triage |
| [Modes and lifecycle](#modes-and-lifecycle) | Unavailable modes, reset history |
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
  kernels reached it at the shapes in [Performance](#performance), while a 4 GiB
  budget retained every access for all of them.
- The budget is split evenly across 256 shards, chosen by page hash. A shard
  that fills its share degrades coverage even if other shards have room.
- For densely accessed data, the shadow costs about 4.25 bytes of page index per
  dword (1,088 bytes per 1 KiB page) plus 24 bytes per retained record. Data
  that one workgroup writes keeps one record per dword, and data that many
  workgroups read keeps two. The shadow is therefore about 7 times the size of
  the global data a dispatch writes, and 13 times the size of the shared data it
  reads.
- The warning prints once per checker (one checker per GPU/SoC), and
  `coverage_degraded` is a single flag for that GPU's whole report, so neither
  says which dispatch lost coverage.

Sizing the default from the dispatch's footprint or from host memory, sharing
one budget across the shards, and reporting degradation per dispatch would
address these.

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
- The `--race-checking` override is refused with `--attach`. In DBT guest mode
  it is also refused when the guest config names a host `simulator_config`; set
  `race_checking` in that config instead. `--wait-checking` has the same rules.
- Address history is not saved in checkpoints and is reset on restore, so a race
  whose first access predates the restore is missed.
- Checkpoints preserve the global shadow budget, but not the diagnostic report
  path; a VM restored through `rj_vm_restore_checkpoint` does not write a JSON
  diagnostic report.
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
SoC-shared ownership and lifecycle refactor. The current synthetic campaign
isolates memory-bookkeeping cost. A targeted current rerun supplies a limited
same-kernel comparison. Percentages are relative to the same binary with race
checking off unless stated otherwise. These results are current through
2026-10-15.

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

These measurements remain the broadest realistic baseline, but they are not a
clean before/after measurement of the final refactor. In particular, the older
checker was owned per command processor while the current checker is shared by
all command processors and CUs of a GPU/SoC.

### Current synthetic microbenchmarks

Four memory-intensive gfx950 workloads (stream, LDS reduction, GEMM and
histogram) were run against the current Release binary with checking on and off.
The small set used 262,144 elements (GEMM 128); the large set used 1,048,576
elements (GEMM 256). The geomeans deliberately exclude a compute-only control:

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
current checker's incremental cost.

These numbers are intentionally not comparable to the 20-kernel geomean: the
synthetics maximize accesses per unit of unrelated simulation work. The earlier
realistic campaign already contained individual kernels at +52% with the
default budget and +73% at 4 GiB.

An earlier stabilized serial pass measured about +37% on the four small memory
workloads and +30% when the compute control was included; the latest all-five
geomean was +31%. The memory-only spread is another reason to treat the
loaded-host synthetic figures as ranges rather than a refactor regression.

### Current mutation-harness sweep

A separate serial gfx1250 sweep used 21 small mutation shaders, five processes
per variant, and ordinary wait checking in both variants. Enabling address race
checking added 3.7% geomean whole-process wall time (bootstrap 95% interval
about +1.9% to +5.4%) and did not exhaust the global shadow. These CI-sized
kernels show the low-cost end of the workload range; their whole-process timing
is not comparable to the Gluon per-dispatch measurements.

### Current representative rerun

FMHA v4 and A16W16 v9 were rerun on the current checker with one process per
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
+49.2% at the default budget and +53.4% at 4 GiB. The current rerun is therefore
directionally faster, not evidence that the refactor increased overhead. It is
still not a controlled redesign A/B: the runs were collected at different
times, and the current rerun has only one process per variant.

### Software profile

Hardware `perf` sampling was unavailable (`perf_event_paranoid=4`), so a
userspace SIGPROF sampler was used on the exact current Release binary. After
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
hot paths. They do not measure lock wait directly and do not establish that
contention is the dominant cost.

### Measurement limits

The shared measurement host was heavily loaded (load averages roughly 400–560
during parts of the synthetic/profile campaign and about 101 during the final
rerun, on 384 logical CPUs), so the exact percentages are directional. A
sign-off run needs reserved cores, fixed CPU policy, interleaved on/off variants
from one binary and at least five processes per variant. It must record coverage
warnings and peak RSS with time: a faster capped run can simply have retained
less history. gfx1250, wave32, multi-GPU and measured lock-contention scaling
remain unprofiled.

## Validation

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
- The budget unit test covers degradation at zero and checkpoint round-tripping
  at a nonzero value. The per-shard split and nonzero capacity boundary are
  exercised only by the Gluon measurements.
- Report tests cover a combined single-GPU file, non-clobbering when unused and
  flush at ordinary grid retirement. There is no end-to-end multi-GPU test of
  the `.gpuN` report paths.
- There are no tests for split barriers with work between the signal and the
  wait, for LDS barrier cells, or for cluster multicast into peer workgroups.

## Suggested order of work

Correctness and coverage:

1. Raise or auto-size the default global budget, and report degradation per
   dispatch.
2. Key findings by instruction pair, and add kernel names and code-object
   offsets to findings.
3. Add barrier-removal mutants to the mutation harness.
4. Model gfx1250 LDS barrier cells, including the tensor-DMA arrive.
5. Apply split-barrier ordering at each member's own signal and wait.
6. Record cluster multicast writes in the peer workgroups' LDS history.
7. Add happens-before for global memory. This removes the largest class of false
   reports, and is a prerequisite for checking global accesses within a
   workgroup.

Performance, preserving the same findings and retained coverage:

1. Instrument lane ranges, distinct dwords, records walked, allocations, budget
   loss and lock-wait time so later changes have fixed-work explanations.
2. Add exact no-work paths for zero-LDS workgroups and one-workgroup global
   dispatches, and attach completion tokens only to effects whose LDS history
   consumes them.
3. Stable-bucket and coalesce compatible lane ranges by page/dword, with
   aligned, disjoint and aggregate-in-bounds fast paths; keep the generic path
   for gathers, overlap, unaligned accesses and per-lane diagnostics.
4. Batch workgroup registration, size vector clocks once, and cache a
   lifetime-safe workgroup state and wave index instead of repeating map lookup
   and linear wave search.
5. Reuse barrier scratch and index pending LDS operations by range so accesses
   visit only potentially overlapping incomplete operations.
6. Improve global locality and memory amplification with dispatch arenas,
   flatter page storage, balanced budget use and compact records.
7. Consider affine page-run descriptors and epoch-based LDS history only after
   the lower-risk paths are measured. Change lock topology only if lock-wait
   instrumentation shows contention is material.
