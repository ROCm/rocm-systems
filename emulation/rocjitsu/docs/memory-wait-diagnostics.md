# Memory wait diagnostics

Memory-result diagnostics are enabled by default. Functional execution computes
memory results eagerly; a core, per-wave scoreboard separately tracks whether a
register result is known complete.
Reading or overwriting an outstanding destination prints a `memory-wait` warning
with the issuing PC, consuming PC, register, counter, and sufficient wait
threshold. Execution continues with the eager value. The first conflicting access
reports the producer and recovers its readiness to avoid cascading warnings. Each CU
prints at most 16 warnings during its lifetime, then one suppression notice.
This budget is per CU, so N active CUs can print up to 17 × N lines for the
same kernel. The budget stays local so a faulty kernel on one CU does not hide
diagnostics from another CU or device. Repeated sites can exhaust this budget
and hide distinct later hazards; the printed set is not exhaustive. Per-CU
diagnostic counts continue after printing is suppressed. Fix reported waits and
rerun to expose later sites. For an unfiltered investigation, a host debugger
breakpoint at `ComputeUnitCore::report_memory_wait` observes every reported
hazard before the output limit is applied.

Registers in diagnostics use the scoreboard's physical numbering. On targets
with an accumulator bank, `v256+n` names `aN` (for example, `v288` is `a32`).

If warnings appear incorrect or checking slows a workload, disable both ordinary
and XCNT checks for a launch without editing the config or rebuilding:

```sh
rocjitsu --config /path/to/config.json --wait-checking=off -- <command>
```

With Mirage, use `mirage run --wait-checking=off -- <command>`. The flag works
with profiles or `--config` and composes with `--cpu-thread-budget`.

Use `--wait-checking=on` for ordinary checks only, or `--wait-checking=all` to include
gfx1250 XCNT. These modes override both per-CU settings on every GPU. Omitting the
flag preserves JSON settings and defaults. See [configuration](configuration.md#memory-wait-diagnostics)
for persistent global overrides and launch-mode restrictions.

Without a global override, memory-result diagnostics default to
`memory_wait_diagnostics=warn`. To disable them per CU, add this entry to each
`compute_unit` node's `config` array:

```json
{"key": "memory_wait_diagnostics", "value": "off"}
```

`warn` and `off` are the accepted values. Invalid values reject the configuration.
This is a core simulator feature and requires no plugin. XCNT replay-source
diagnostics remain independently disabled by default, as described below.

## XCNT replay-source diagnostics

On gfx1250, the core also warns when a memory instruction's source register is
overwritten before address translation is known complete. Hardware may need that
original value for XNACK replay. This checks scalar address operands and VMEM
address/data operands, including EXEC and dynamically selected VGPR banks. Reading
a protected source again is allowed. `xcnt-wait` warnings have their own limit of
16 per CU and do not consume the ordinary `memory-wait` warning budget.

XCNT tracking and warnings default to `off`. Without a global `wait_checking`
override, enable them independently of memory-result checks by adding this entry
to each `compute_unit` node's `config` array:

```json
{"key": "xcnt_diagnostics", "value": "warn"}
```

The accepted values are `warn` and `off`; invalid values reject the configuration.
This setting has no effect on other architectures. It controls diagnostics, not
hardware XNACK support. A global `wait_checking` override takes precedence over
both per-CU keys. Without one, setting `memory_wait_diagnostics=off` leaves an
explicitly enabled XCNT check active. Use `--wait-checking=off` to disable both.

VMEM coverage is qualified for `MODE.REPLAY_MODE=1` (bit 25), the multi-group mode
selected by LLVM at kernel entry. VMEM replay sources in single-group mode are
not checked; scalar replay sources are checked in either mode. This is an explicit
coverage boundary, not a claim that single-group kernels are replay-safe.

Within the qualified model, X waits retire an ordered VMEM queue prefix. SMEM
translation is unordered and requires a zero wait. A zero KM wait also releases
SMEM sources. VMEM completion waits map completed instructions back to their X
queue positions and release the proven translation prefix, including mixed loads
and stores. Completion counters retain their independent positions.
The checker accounts for implicit drains at branches, register-control operations,
messages and barriers, SMEM/VMEM transitions, and ordered VMEM destination reuse.
It protects source dwords and executed vector lanes, without retaining payloads or
simulating faults. Warnings describe a possible replay failure, even when the eager
execution's numerical results are correct. Atomic replay ordering and memory
visibility are outside this register-source check.

## Coverage

The checker tracks scalar and vector register results from both memory pipelines and
inline producers, including DS permutations and message returns. Store operands are
checked when they consume a pending result. The issuer checks each instruction
before execution, using decoded register numbers and the wave's current EXEC,
register-bank and addressing state. Counter-only producers
contribute to partial waits even without a register result. Implicit SCC consumers and
overwrites (including HWREG aliases), M0 and EXEC message results, wave-sized VCC
accesses, and private address uses of FLAT_SCRATCH are checked too. Preserved scalar
halves and inactive vector lanes are not treated as consumed inputs. EXEC is consumed
even when its eagerly computed value disables all vector lanes.

The shared target model handles CDNA1 through CDNA5 and RDNA1 through RDNA4, including
RDNA3.5. Each ordered completion class retains the suffix requested by a wait.
Scalar memory results require a zero wait. A mixed counter can prove an ordered
result ready only using younger operations in that same completion class. The
[counter coverage document](memory-wait-counter-coverage.md) details producer families, multi-counter operations,
instruction units, and exceptions.

Finite counter capacity also proves completion. For example, admitting a 64th
operation to a six-bit counter forces the oldest of 63 outstanding operations in
one ordered class to complete. Admission is checked before the new instruction
reads its registers. Counter-only operations count, but unordered younger operations
cannot establish FIFO completion of an older result. This does not infer readiness
of scalar-memory, GDS, or legacy CDNA FLAT results from backpressure.

Completion dependencies survive ordinary branches and CU scheduling; branches
implicitly drain XCNT. State is reset when a wave slot is freed and is not serialized
in checkpoints. The ordinary checker does not compare memory addresses. The opt-in
core address checker described below reuses its completion positions to cover those
hazards.
A warning describes a possible missing wait under
delayed completion or replay even though eager execution already has a value. The
feature does not model GPU latency.

## False negatives and false positives

A clean run does not prove that all required waits are present, even within one
wave. Memory effects execute eagerly, so a global or LDS write followed by a read
can appear correct even when a memory-ordering wait is missing. Direct/async LDS
transfers, tensor DMA footprints, cache coherence and accesses from other waves are
outside this register-dependency check. Not every store/read pair needs a wait:
ordinary same-wave DS accesses are ordered, and
CDNA5 VMEM stores and loads to the same global address stay ordered too.

Warnings can also be false positives. The checker does not model instruction
latency or distance: it cannot recognize a dependency made safe by
architecturally sufficient instruction spacing. Counter backpressure is modeled
only where the target's counter capacity and completion ordering prove readiness;
unqualified completion classes remain conservative. The checker does not simulate
hardware occupancy or select a latency at which memory becomes visible.

These are known limits, not an exhaustive list. Investigate warnings against the
target's ordering rules; use `--wait-checking=off` to suppress both warning classes
when needed.

## Address race diagnostics

`--race-checking=on` or top-level `"race_checking": "on"` enables the core
`memory-race:` checker on every CU. Without either global override, a CU opts in
with `{"key":"memory_race_diagnostics","value":"warn"}`; `off` is the default
and the only other accepted value. Enabling it selects all check families by
default. Use `--race-checking-checks=<list>` or top-level
`"race_checking_checks": "<list>"` to select a comma-separated subset:

- `global` checks regular vector and scalar global accesses across workgroups.
- `lds-same-wave` checks accesses against the wave's pending LDS operations.
- `lds-cross-wave` retains LDS history and applies resolved barrier ordering.
- `direct-to-lds` includes direct-to-LDS operations in the selected domains.
- `tensor-dma` includes tensor-DMA operations in the selected domains.

The last two are source switches and require at least one of the first three
address domains. `all` selects every family. The selector does not turn on the
master switch by itself, and omission preserves the all-family behavior.
Selecting a subset deliberately removes diagnostic coverage; it is primarily
an attribution and controlled-deployment mechanism. Current active-path
measurements do not establish any independently useful family below a 2%
overhead bound; see the
[family performance decision](address-race-checking-performance-analysis.md#selectable-family-overhead-and-2-decision).

The checker reports same-wave overlapping LDS
RAW/WAR/WAW while the producer completion token remains pending, and cross-wave
LDS conflicts not ordered by the exact resolved barrier member set. It also
compares vector global, scalar-memory, and tensor-DMA accesses between
workgroups in the same dispatch. Scratch/private accesses and
atomic-vs-atomic pairs are excluded. Compiler-lowered atomic loads and stores
still look like ordinary accesses and remain a
[known false-positive class](address-race-checking-gaps.md#atomic-loads-and-stores).
Ordinary same-class DS issue ordering is honored; direct-to-LDS writes use their
load obligation and tensor transfers use `tensorcnt`.

Tensor ranges come from exact in-bounds elements, including descriptor padding.
Direct-to-LDS destinations use routed per-lane addresses and cluster participant
selection. Diagnostics identify both sites, waves and lanes and recommend a wait
or barrier. The global budget is `race_checking_global_shadow_mb` (512 MiB by
default). It is one atomic cap shared by all 256 shards of the GPU checker and
charges the modeled payload of global page/site directories and
page/record/site slabs. Allocator metadata, small owner objects, slab-descriptor
vectors, mutexes, and fixed checker caches are outside the cap, so it is not an
RSS limit. If another access cannot be retained within the cap, the checker
warns once and sets `coverage_degraded`; the access is still checked against
retained history.
Separately, console output keeps at most 64 distinct race keys at a time across
the live dispatches of one GPU/SoC checker and warns once when further keys are
suppressed. LDS history is outside the global budget: it lives only while its
workgroup is resident and grows with the workgroup's waves and LDS footprint.

One `MemoryRaceChecker` is shared by every command processor and CU of a
GPU/SoC. It keys workgroup state by the SoC-wide dispatch ID and the grid-wide
workgroup ID. Each memory instruction is checked once with all of its lane
ranges. A zero-LDS workgroup creates no LDS checker state. For a nonzero-LDS
workgroup, startup registers the known wave span in one batch and sizes its
vector-clock matrix once. A 64-shard map locates that state; each wave then
caches a generation-checked shared state handle and wave index. Normal accesses
skip repeated map lookup and linear wave search but still take the workgroup's
state mutex. Different workgroups do not serialize on that mutex.

Each workgroup keeps one record per wave, access kind and LDS dword: a newer
access by the same wave subsumes an older one. A barrier that every active wave
joins drops the history. A barrier orders only completed accesses: one still in
flight, such as a multi-buffered async copy, stays unordered with the other
members until a barrier that follows its completion, so only touching its bytes
early is reported.

The global shadow has 256 hash shards. An access holds one shard mutex at a time
as it visits pages, so pages that hash to the same shard can serialize. Each
active dispatch in a shard owns deletion-free open-addressed page and site
directories plus bounded slabs: 16 pages, 1,024 records, or 256 sites per full
slab. A 16-byte record names a 16-byte interned `(pc, wave, lane)` site and is
tagged with the accessing workgroup. A dword's bytes keep at most one record per
workgroup and two other workgroups per access kind. That is enough to detect any
later conflicting workgroup, so read-shared data such as kernel arguments or
GEMM inputs does not grow the shadow with the number of readers. Dispatch
retirement destroys only that dispatch's slabs and directories and immediately
returns their charge to the checker-wide budget. A grid containing exactly one
workgroup bypasses global checking entirely because no other workgroup can
supply a comparable record. Checkpoint restore lacks the original grid size and
conservatively leaves this bypass disabled.

The checker limits each mutex to the state it owns. The 64 workgroup-map
shards protect lookup, insertion, and removal only; a workgroup's mutex protects
its wave clocks, pending LDS operations, and LDS records. The global path holds
one of its 256 shard mutexes only while locating or changing that shard's
dispatch/page/site/record state. Finding callbacks, warning logs, diagnostic
report updates, statistics aggregation, object destruction, and global-budget
release happen after the relevant checker lock is released. Dispatch cleanup
first detaches shared objects and then destroys them outside the map or shard
critical section.

The CU wave-state mutex is separate and intentionally remains held while an
instruction is issued and its scoreboard, barrier, and wave-lifetime state are
updated. This establishes simulator ordering; it does not protect local range
preparation or report output. Parallel workgroup completion queues only its
shared completion record. The command processor drains those records and
flushes any completed-dispatch report after releasing its queue lifecycle
mutex.

Set top-level `memory_diagnostic_report` to write combined core wait and address
findings as JSON. One report object is shared by every CP and CU of a GPU and is
flushed whenever a dispatch retires or is dropped. Findings accumulate across
dispatches in the process, and each flush rewrites the complete retained report.
The report copies a revisioned snapshot while holding its state mutex, then
formats JSON and performs file I/O outside that mutex. A separate writer mutex
orders concurrent flushes without blocking finding producers on formatting or
the filesystem. Direct terminal VM faults, failed graceful queue closes, and
normal retirement all reach an after-lock flush boundary.
A configuration-only SoC that never flushes and records nothing does not replace
an existing file; the first dispatch flush writes a report even when its
`findings` array is empty. GPU 0 writes the configured path. In a multi-GPU
config, each additional GPU `N` writes `<path>.gpuN`.

The version-1 JSON object contains `findings`, `observed`, `suppressed` and
`coverage_degraded`. It also contains `race_statistics`, whose cumulative
`totals` and eight most recent completed dispatches with address-race work
measure the checker's fixed work. `last_dispatch` repeats the newest entry for
convenience. Counters cover accesses and lane ranges, candidate and
visited dwords, shadow pages and records, vector growth, workgroup lookup,
pending operations, barriers, cleanup, and retained or rejected global-shadow
bytes. Global-shadow counters also cover budget reservation attempts, failures
and releases; page, record and site slab/directory growth; site lookup,
interning and reuse; and saturation-cache hits, inserts and invalidations. Each
recent dispatch includes `global_shadow_bytes_released` and the checker-wide
`global_shadow_peak_bytes` observed when it retired. Recent dispatches include
only nonempty global-shard entries so retention and lock acquisition time can
expose skew without retaining an unbounded per-dispatch history.

Regular consecutive-lane accesses can remain in an affine descriptor instead
of materializing one `LaneRange` per lane. `affine_accesses` and
`affine_lane_ranges` count those instructions and their logical lane ranges;
`affine_dword_runs` and `affine_dword_visits` count their compact aligned runs
and the dwords still sent to the shadow. `affine_pending_ops` counts compact
in-flight LDS operations, while `explicit_accesses` counts instructions that
use the generic range representation. These fields classify representation,
not coverage: both paths preserve the same byte masks and shadow visits.

`dword_candidates` counts dwords that enter scalar expansion. For aligned
overlapping lane ranges, the checker can skip a prefix while it is still
guaranteed to duplicate the recent-dword cache; those dwords increment
`dword_duplicates_skipped` without incrementing `dword_candidates`.
`dword_visits` retains its exact meaning: distinct dword/mask combinations
actually sent to the shadow.

`retained_bytes` is the cumulative amount successfully charged when a modeled
directory or slab grew; it is neither the current charge nor process RSS.
`rejected_bytes` records the minimum modeled payload associated with failed
admissions. `budget_bytes_released` is cumulative, while
`global_shadow_bytes_released` is the charge returned by one dispatch and
`global_shadow_peak_bytes` is a checker-wide high-water mark. Fields ending in
`lock_wait_ns` include the timed lock acquisition itself as well as any wait.
They measure only checker workgroup-map, workgroup-state, and global-shard
locks, not the CU wave-state, command-processor queue, report-state, or writer
mutexes.
Statistics collection and steady-clock calls are enabled only when this JSON
report is configured. Console-only checking still takes the common
instrumentation branch, but does not update counters or call the clock; its
measured overhead is recorded in the
[performance analysis](address-race-checking-performance-analysis.md).

Each finding identifies its checker, kind, resource, producer and consumer PCs,
and dispatch/workgroup/wave context; race entries also include the byte range,
lanes and recommended action. The report retains at most 1,024 `memory-wait`
and 1,024 `memory-race` entries. `observed` counts findings delivered to the
report plus race events explicitly suppressed by the 64-key cap; repeated race
keys deduplicated before reporting are not counted. The shared `suppressed`
counter includes either kind overflowing its 1,024-entry vector and races
rejected by the 64-key console cap.

JSON output is optional. Without `memory_diagnostic_report`, enabled checks
still print `memory-wait:`, `xcnt-wait:` and `memory-race:` warnings to the
normal log.

Address history is not serialized in checkpoints and is reset on restore, while
the configured global shadow budget is preserved. Resident LDS allocation
metadata is not serialized either: restored waves default to a zero-sized LDS
allocation, so their later LDS address ranges are outside checker coverage.
Global accesses remain checked conservatively when the `global` family is
selected. The effective family selection is checkpointed, but the report path is not
checkpointed, so a VM restored through `rj_vm_restore_checkpoint` does not write
a JSON diagnostic report. The legacy race-detector plugin remains supported,
but enabling both produces overlapping reports and a warning based on the
effective per-CU settings; prefer the core mode for new address-race workflows.

[Address race checking gaps](address-race-checking-gaps.md) lists the checker's
known false positives, false negatives, limits and current performance evidence.

## Access planning and cost

The scoreboard retains register dependencies, not memory payloads or cached access
plans. Ordinary elementwise instructions probe decoded register numbers directly.
A 1,280-byte shadow per wave covers physical VGPRs 0–1023 and scalar/special
registers. Separate result and replay bits share each byte. A negative probe skips
the detailed checker; a possible overlap checks the pending records' lane and byte
masks. The shadow is conservative across overlapping records and counter retirement.

The wave stores VGPR bank selection in one byte. An issued dependency records the
physical register number, so a later bank change cannot redirect that dependency.
VCC is wave state with separate low/high word identities. Unordered scalar memory
results require a zero wait; they cannot be made ready by an unrelated partial wait.

Exceptional instructions use shared addressing and lane-selection helpers before
execution. Relative registers and permutations can require M0 or selector values;
FLAT domain selection requires address values. Those control inputs are checked
before being read for planning. Planning does not fire execution-plugin callbacks.
The checks and producer registration precede execution lane loops and async MMA
publication. Only the issuing thread accesses the scoreboard: register accessors,
MMA workers, observer snapshots and memory writeback do not check it.

Lane masks distinguish divergent consumers and pending results across waterfall
iterations. Qualified sub-dword forms also distinguish consumed and written bytes,
without treating preservation of the other bytes as a read. Other sub-dword forms
can conservatively check a whole register. Encoded memory address/data sources keep
their dependencies even when an invalid descriptor or an out-of-bounds address
suppresses the memory effect.

Detailed records are allocated on the first tracked issue and reused across wave
slot activations. Retirement clears affected shadow bytes and restores overlapping
live records. Diagnostic formatting occurs only on a hazard. With both settings
disabled, the issuer skips the checker; execution-plugin callbacks remain independent.
