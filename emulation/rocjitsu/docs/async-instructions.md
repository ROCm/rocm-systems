# Asynchronous MMA execution

Functional simulation can issue an expensive MMA instruction to a persistent
helper while the issuing CPU executes independent instructions from the same
wave. A register scoreboard keeps reads and writes ordered. The wave drains
all pending work before leaving its bounded issue window or returning to CU
scheduling.

The matrix handlers currently execute synchronously through rj-arith: no handler
is eligible for helper offload. Effective helper allocation is therefore zero,
even when a preset or explicit setting requests helpers. The retained adapter
and configuration described below can support future eligible handlers.

## Thread policy

The retained execution-thread allocation is **E + sum(D - 1) + H**: engine
threads E, inclusive dispatch width D for each SoC, and H helpers shared by the
VM's CUs. The dispatch pool accepts concurrent XCD submissions; each caller
executes its own CUs while helpers can execute eligible MMA instructions.

`async_helper_threads` extends the existing allocation tables. Omission or -1
selects the configured policy; zero disables helpers. Explicit requests range
from 0 to 128 and take priority over automatic helper selection only when an
eligible handler exists. The requested value is retained for configuration and
checkpoint serialization; the current effective helper count remains zero.
Automatic selection uses CPU affinity and the target's allocation table.

The table shows current effective allocations, including suppression of helpers:

| Budget / affinity | gfx950 E/D/H | gfx1250 E/D/H | Total gfx950 / gfx1250 |
|---:|---:|---:|---:|
| 1 | 1/1/0 | 1/1/0 | 1 / 1 |
| 2 | 1/2/0 | 1/2/0 | 2 / 2 |
| 4 | 2/3/0 | 1/4/0 | 4 / 4 |
| 8 | 2/7/0 | 2/7/0 | 8 / 8 |
| 16 | 8/9/0 | 8/9/0 | 16 / 16 |
| 24 | 8/17/0 | 8/17/0 | 24 / 24 |
| 32 | 8/17/0 | 8/25/0 | 24 / 32 |
| 40 | 8/17/0 | 8/33/0 | 24 / 40 |
| 48 and above | 8/17/0 | 8/33/0 | 24 / 40 |

A budget between entries selects the lower entry; 12 selects 8. Explicit
`cpu_thread_budget` limits the combined allocation. Larger hosts retain the
highest table entry. Custom JSON allocations and explicit knobs remain
available for workloads that benefit from more workers.

The retained server presets request helpers after reaching their engine/dispatch
ceiling. While no handler is eligible, higher budgets retain 8/17/0 on gfx950 and
8/33/0 on gfx1250. Re-enabling a handler would make helper allocation effective
again; it would also require renewed validation of overlap and thread budgets.

Desktop and MI210 presets stop at 1/24/0; CDNA3 stops at 8/17/0.
Two- and four-GPU presets keep **1/1/0** for RCCL. Mirage embeds these native
tables in its rocjitsu backend and exposes the helper option alongside the
engine, dispatch and total-budget options.

```sh
rocjitsu --config configs/gfx950_mi350x.json --thread-budget-table
rocjitsu --config configs/gfx950_mi350x.json --cpu-thread-budget 64 --thread-budget-table
```

## Execution and ownership

Each issue window has a bounded queue and a 512-register dependency map:
256 VGPRs and a separate 256-entry accumulator bank. Read-after-write,
write-after-read and write-after-write conflicts wait for the relevant jobs;
shared read-only inputs remain independent. Sources are not copied. Lazy
register storage is materialized before publishing jobs to helpers.

Capacity acquisition never waits. If the shared pool is full or contended,
the issuer executes inline after resolving its dependencies. Completed MMA
jobs release dependencies independently; a slow earlier job does not hold an
unrelated completed result. The issuer polls or joins jobs and destroys decoded
instructions on their allocator's owning thread. Admission plans retain code
snapshots and decisions; their temporary decodes never escape a scan.

Each issuer uses a stable, hashed starting position when selecting a free
helper, preferring reuse across consecutive jobs. This preference does not pin
threads or guarantee cache locality. Other free slots remain available, and
pools larger than 64 helpers retain first-word priority. Selection still falls
back inline after a bounded number of failed reservation attempts.

Both publication and completion use release/acquire synchronization. Helpers
inherit the issuer's floating-point environment. A bounded warm wait precedes
blocking through standard atomic wait/notify operations.
Unexpected handler failures are propagated after joining published work.
Precise asynchronous exception rollback is not supported.

Helper resources belong to the VM and are created lazily. Separate VMs have
separate pools; CUs and GPUs in one VM share its pool. Fork children reject
inherited helper state and use synchronous execution. Pool lifetime tests
cover this fallback; arbitrary forked application workloads are not qualified.

## Eligibility and fallback

The previously eligible dense MFMA and WMMA families now execute
synchronously through rj-arith. No current instruction family uses the helper
pool; the configuration and adapter remain available for future integration.
The adapter rules below apply when an instruction family opts back in.
Both targets use [cached lookahead admission](mma-admission.md) to require an
independent MMA that can remain on the issuer. Scaled instructions require
vector or inline scale operands; scalar-register scales execute synchronously.
Encoding filters bypass ineligible instructions before queue construction.

The adapter requires functional execution, the target wave size, full EXEC,
ordinary register addressing, and no active debugger or trap handler. An ISA
property enables the adapter only for supported families. Other architectures
and clocked execution keep their ordinary issue paths.

Ordinary memory instructions continue through the existing memory pipelines.
Their register footprints still participate in scoreboard checks. Control
flow, barriers, unsupported instructions, relevant execution-state changes and
window boundaries drain pending MMA work before proceeding.

Observer plugins must explicitly support async issue notifications.
Throughput and kernel logging do; architectural observers retain synchronous
execution. All instruction callbacks run on the issuer. Register callbacks
can occur concurrently on helpers. Throughput keeps instruction counts and
dispatch wall time but marks overlapping execution-duration metrics invalid.

## Checkpoints

Checkpoints retain the helper request and configured triples, then resolve
automatic choices for the receiving host. Helper fields are appended to the
existing schema. Older checkpoints retain zero helpers, including those with
an allocation table and legacy automatic-dispatch checkpoints. Outstanding
jobs never survive an issue window and are not serialized.

## Diagnostics

JSON configuration is the control surface; async environment overrides are not
supported. The issue window is bounded to 32 instructions, lookahead examines
up to eight following instructions, and at most seven helper jobs are pending
per wave. Compile-time VM logging includes admission and queue counters.
The throughput plugin records offloaded instructions as untimed instructions;
it retains valid dispatch and process wall-time measurements.

## Validation

Unit tests cover register hazards, helper saturation, independent retirement,
exceptions, floating-point state, instruction-cache invalidation, observer
fallback, VM ownership and legacy checkpoint compatibility. Functional tests
compare mixed MMA/memory instruction streams with serial execution.

The [TSan concurrency campaign](../fuzz/execution/README.md#threadsanitizer-concurrency-campaign)
currently checks concurrent issuer execution with helpers disabled. Its success
criterion is no TSan reports, not nonzero offload counters. Helper-path coverage
requires an eligible handler and verified nonzero offload.

For performance comparisons use the same workload and total thread budget,
reporting dispatch and process wall time
separately. CPU simulator measurements do not establish physical GPU behavior.
