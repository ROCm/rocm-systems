# Cached MMA lookahead admission

The [MMA scoreboard](async-instructions.md) uses bounded lookahead for CDNA4
MFMA and CDNA5 WMMA. An instruction is offloaded only when the scan
finds another independent MMA that the issuing thread can execute. Instruction
size alone cannot predict profitability. An isolated or dependent MMA pays helper
handoff and completion costs without useful overlap.

A useful cost criterion is
`min(MMA execution time, independent issuer work) > handoff + completion + bookkeeping`.
The instruction allowlist supplies measured handler-cost qualification; cached
lookahead establishes independent issuer work. Admission does not measure wall
time on every dynamic instruction. On the development host, dense K32/K64/K128
WMMA handlers take roughly 31/61/121 microseconds, versus about one microsecond
for a warm empty handoff. Immediate submission and joining of a real MMA adds
several microseconds, including wakeup and register-cache transfer costs.

This cost criterion assumes spare CPU capacity. A fixed helper allocation also
reduces the threads available for CU dispatch. Independent pairs can speed up
while an entire kernel slows down because its non-MMA phases have fewer workers.
The target allocation tables therefore require whole-kernel, equal-budget
measurements in addition to handler and handoff microbenchmarks.

## Implementation

Each optional async CU adapter owns a cache of admission plans keyed by PC,
VMID and register-bank bounds. Both successful and rejected plans are cached.
The scan stops at a dependency, unsupported instruction, control-flow boundary
or the edge of the current code page. It never executes instructions, changes
the PC, notifies observers or emits a diagnostic. The runtime scoreboard
continues to check dependencies and available helper capacity.

A plan collects an independent group, offloads its prefix and reserves the
final MMA for the issuer. Restricting the scan to pairs sacrificed throughput
on wider independent groups. The earliest outstanding issuer reservation wins,
so subsequent admissions cannot keep moving all useful work into the pool.

I-cache invalidation increments an epoch. After an epoch change, a plan checks
its saved code bytes before reuse; unchanged code keeps plans across dispatches.
Changed bytes rebuild the plan. Debugger and fetchability checks remain in the
ordinary issue path. A scan destroys its temporary decoded instructions before
returning, on the same thread and while their backing bytes remain alive. Plans
retain only code snapshots and the reserved issuer offset, so they can survive
CU migration between dispatch workers without retaining decoder allocations.

**Only the speculative admission decision is cached.** Ordinary execution
still decodes each issued instruction and retains its existing mutable state
and ownership. For a fixed scan bound and unchanged code, speculative decoding
and plan construction scale with distinct admission sites per participating CU,
not dynamic loop iterations. Overlapping cold scans can decode the same bytes
again. Byte validation repeats after I-cache invalidation. The cache has a
bounded entry count; filling it clears retained plans before adding a new site.
Replacing an existing site's code requires no eviction. Entries remain local
to the CU.

## Configuration

Server presets add 2, 4, 8 or 16 helpers above the existing 32-thread
engine/dispatch allocation, selected automatically when CPU affinity permits.
The largest preset uses 48 total threads. Set `async_helper_threads: 0` to
keep execution synchronous, or `cpu_thread_budget: 32` to limit the total.

Every selected CDNA4 MFMA and CDNA5 WMMA family requires cached admission,
including dense K32 and scaled 16x16x128 WMMA. Lookahead inspects at most eight
following instructions. There are no async environment-variable controls.

Compile-time VM logging distinguishes speculative decodes, plan construction,
plan hits, epoch validation, accepted/rejected admissions and reserved issuer
MMAs. Counters flush at wave halt; the printed entry count is the cache's current
size, so summing it across flushes overcounts storage.
