# Asynchronous MMA execution

Functional simulation can issue an expensive MMA instruction to a persistent
helper while the issuing CPU executes independent instructions from the same
wave. A register scoreboard keeps reads and writes ordered. The wave drains
all pending work before leaving its bounded issue window or returning to CU
scheduling.

The gfx950, gfx1250 and gfx1201 single-GPU presets enable this path when their
selected thread granule includes helpers. Other shipped targets and multi-GPU
presets keep helpers disabled. This accelerates host execution; it does not model GPU
MMA latency or validate application memory hazards.

RDNA3, RDNA3.5 and RDNA4 support helpers for wave32 and wave64 F32-output
FP16/BF16 K16 WMMA, including the bit-exact arithmetic paths. gfx1201 selects
24 dispatch threads plus eight helpers at a budget of 32; gfx1100 and gfx1151
keep helpers opt-in. Partial-EXEC waves execute synchronously. See
[RDNA WMMA execution](rdna-wmma.md) for SIMD and its effect on helper allocation.

## Thread policy

The retained execution-thread allocation is **E + sum(D - 1) + H**: engine
threads E, inclusive dispatch width D for each SoC, and H helpers shared by the
VM's CUs. The dispatch pool accepts concurrent XCD submissions; each caller
executes its own CUs while helpers can execute eligible MMA instructions.

`async_helper_threads` extends the existing allocation tables. Omission or -1
selects the configured policy; zero disables helpers. Explicit counts range
from 0 to 128 and take priority over the automatic budget. Automatic selection
uses CPU affinity and the target's allocation table. gfx1151, gfx1201 and older CDNA
presets stop at an engine/dispatch cost of 24; gfx1100 uses up to 32 dispatch
threads. gfx1201 adds eight helpers at a budget of 32. gfx950 adds eight helpers for
32 total threads; gfx1250 uses up to 40 engine/dispatch threads and eight
helpers for 48 total.

| Budget / affinity | gfx950 E/D/H | gfx1250 E/D/H | Total gfx950 / gfx1250 |
|---:|---:|---:|---:|
| 1 | 1/1/0 | 1/1/0 | 1 / 1 |
| 2 | 1/2/0 | 1/2/0 | 2 / 2 |
| 4 | 2/3/0 | 1/4/0 | 4 / 4 |
| 8 | 2/7/0 | 2/7/0 | 8 / 8 |
| 16 | 8/9/0 | 8/9/0 | 16 / 16 |
| 24 | 8/17/0 | 8/17/0 | 24 / 24 |
| 32 | 8/17/8 | 8/25/0 | 32 / 32 |
| 40 | 8/17/8 | 8/33/0 | 32 / 40 |
| 48 and above | 8/17/8 | 8/33/8 | 32 / 48 |

A budget between entries selects the lower entry; 12 selects 8. Explicit
`cpu_thread_budget` limits the combined allocation. Larger hosts retain the
highest table entry. Custom JSON allocations and explicit knobs remain
available for workloads that benefit from more workers.

Both server targets add helpers after reaching their engine/dispatch ceiling.
Lower-budget comparisons on IREE, Gluon and Tensile favor retaining general
CU workers over reserving helper slots. Helpers are created lazily when
eligible MMA reaches the adapter. Disabling helpers retains 8/17/0 on gfx950
and 8/33/0 on gfx1250 at larger budgets. These defaults limit CPU use while
retaining matrix overlap; they are not universal throughput optima.

gfx1100 stops at 1/32/0, gfx1201 at 1/24/8, and gfx1151 and MI210 at
1/24/0; CDNA3 stops at 8/17/0. The desktop entries through 24 threads are
unchanged. Disabling gfx1201 helpers retains 1/24/0.
Two- and four-GPU presets keep **1/1/0** for RCCL. Mirage embeds these native
tables in its rocjitsu backend and exposes the helper option alongside the
engine, dispatch and total-budget options.

```sh
rocjitsu --config configs/gfx950_mi355x.json --thread-budget-table
rocjitsu --config configs/gfx950_mi355x.json --cpu-thread-budget 64 --thread-budget-table
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
The RDNA paths use the same admission and register dependency checks.

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
