# Asynchronous MMA execution

Functional simulation can issue an expensive MMA instruction to a persistent
helper while the issuing CPU executes independent instructions from the same
wave. A register scoreboard keeps reads and writes ordered. The wave drains
all pending work before leaving its bounded issue window or returning to CU
scheduling.

The gfx950 and gfx1250 single-GPU presets enable this path when their selected
thread granule includes helpers. Other shipped targets and multi-GPU presets
keep helpers disabled. This accelerates host execution; it does not model GPU
MMA latency or validate application memory hazards.

## Thread policy

The retained execution-thread allocation is **E + sum(D - 1) + H**: engine
threads E, inclusive dispatch width D for each SoC, and H helpers shared by the
VM's CUs. The dispatch pool accepts concurrent XCD submissions; each caller
executes its own CUs while helpers can execute eligible MMA instructions.

`async_helper_threads` extends the existing allocation tables. Omission or -1
selects the configured policy; zero disables helpers. Explicit counts range
from 0 to 128 and take priority over the automatic budget. Automatic selection
uses CPU affinity, keeping engine/dispatch cost at most 32. Helpers are additive;
the shipped single-GPU server presets stop at 48 total threads.

| Budget / affinity | gfx950 E/D/H | gfx1250 E/D/H | Total execution threads |
|---:|---:|---:|---:|
| 1 | 1/1/0 | 1/1/0 | 1 |
| 2 | 1/2/0 | 1/2/0 | 2 |
| 4 | 2/3/0 | 1/4/0 | 4 |
| 8 | 2/7/0 | 2/7/0 | 8 |
| 16 | 8/9/0 | 8/9/0 | 16 |
| 24 | 8/17/0 | 8/17/0 | 24 |
| 32 | 8/25/0 | 8/25/0 | 32 |
| 34 | 8/25/2 | 8/25/2 | 34 |
| 36 | 8/25/4 | 8/25/4 | 36 |
| 40 | 8/25/8 | 8/25/8 | 40 |
| 48 and above | 8/25/16 | 8/25/16 | 48 |

A budget between entries selects the lower entry; 12 selects 8. Explicit
`cpu_thread_budget` limits the combined allocation. Affinity of 64 still selects
48: the measurements do not justify more than 16 preset helpers. Custom JSON
allocations and explicit knobs remain available for experiments.

Both targets preserve the synchronous engine/dispatch allocations through 32.
Reserving helpers within that budget regressed Gluon FP16 by displacing CU
workers, even with cached independence admission. Adding helpers above 32
avoids that tradeoff. `async_helper_threads: 0` disables helpers while retaining
8/25/0 on larger hosts. No workload classification is needed in the selector.

The 32-thread CPU cap is a conservative resource policy, not a universal
throughput optimum. Extra dispatch workers can outperform extra helpers on
CU-rich workloads such as large-grid Gluon FP16. Helpers offer their largest
gains when few workgroups limit CU parallelism. The preset keeps the existing
CPU allocation and spends additional affinity on this complementary path;
helpers are created lazily, only when eligible MMA reaches the adapter.

Desktop and MI210 presets retain 1/D/0 through D=32; CDNA3 retains the landed
mixed E/D table with H=0. Two- and four-GPU presets keep **1/1/0** for RCCL.
Mirage embeds these native tables in its RocJITsu backend and exposes the helper
option alongside the engine, dispatch and total-budget options.

```sh
rocjitsu --config configs/gfx950_mi355x.json --thread-budget-table
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

The default instruction families are dense FP16, FP8/BF8 and scaled/unscaled
f8f6f4 MFMA on gfx950, and f32-output FP16/BF16 K32, FP8/BF8 K64/K128,
dense FP4 and scaled 16x16x128 f8f6f4 WMMA on gfx1250.
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

For performance comparisons use the same workload and total thread budget,
reporting dispatch and process wall time
separately. CPU simulator measurements do not establish physical GPU behavior.
