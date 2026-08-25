# Introduction: what kernel replay is, and what makes it hard

## 1. The constraint that creates the problem

A GPU's performance monitoring hardware is a fixed, small resource. Each hardware block has a
handful of counter registers, and a counter register can be programmed to observe exactly one event
at a time. On CDNA2, CDNA3 and CDNA4 alike the budget per block is: 8 in the sequencer (SQ), 4 in the
L2 cache (TCC), 4 in the vector L1 (TCP), 6 in the shader-processor interconnect (SPI), 2 each in the
texture addresser and data units (TA, TD), and 2 in the graphics register bus manager (GRBM). These
numbers have not grown across three generations. Requesting more events from a block than it has
registers is not slow — it is impossible, and the hardware reports a hard error.

Meanwhile, the questions people ask of a profiler have grown. Almost every useful derived metric is a
ratio or a rate assembled from several raw events: an arithmetic intensity needs both flop counts and
byte counts, a cache hit rate needs both hits and total accesses, an occupancy estimate needs
sequencer state and wave-launch counts. A modest-looking request such as "give me the memory
hierarchy summary" expands into dozens of raw events spread across blocks whose registers are already
oversubscribed several times over.

So the shape of the problem is a mismatch between two numbers. A tool needs `k` observations of a
kernel; the hardware yields `m < k` observations per execution. Closing that gap requires either more
registers, which is not on offer, or more executions.

## 2. Three ways to get more executions, and what each one gives up

The gap can only be closed by running something more than once, and the design space is determined by
*what* gets re-run.

**Re-run the application.** Launch the whole program once per counter group. This is the traditional
answer and it is trivially correct: each run is a fresh process with fresh memory, so no state has to
be reconstructed. It costs the entire application runtime multiplied by the number of groups, and it
is only meaningful if the application is deterministic — if the second run takes a different code
path, executes a different number of kernels, or converges in a different number of iterations, then
the counter groups describe different work and combining them is meaningless. For an iterative solver
with a data-dependent exit condition, or anything with atomics-order-dependent results, that
condition fails silently.

**Re-run nothing; rotate the groups across dispatches.** Give dispatch 1 the first counter group,
dispatch 2 the second, and so on, then aggregate statistically. This is what
`rocprof-compute --iteration-multiplexing` does. Cost is one application run, and it is immune to
every memory-reconstruction problem discussed below because it never reconstructs anything. What it
gives up is per-dispatch correlation: two counter groups are never observed on the same execution, so
any derived metric combining them is a ratio of averages over different work. That is perfectly
acceptable for a steady-state loop where every iteration is statistically identical, and unusable for
a kernel whose behavior varies across invocations — which is exactly the kernel most worth
investigating.

**Re-run one dispatch, in place.** Execute the same kernel `N` times inside a single application run,
reconstructing its inputs before each execution, and let the application observe only one of them.
This is **kernel replay**. Cost is `N` × the kernel's own time plus `N` × the cost of reconstructing
its inputs — small relative to the application when the kernel is a small part of a long run. It is
the only one of the three that yields several counter groups from *the same execution of the same
kernel on the same data*, which is what makes a derived metric a statement about one event rather
than a ratio across different events.

What kernel replay gives up, in exchange, is the correctness that comes for free in the other two:
it has to reconstruct state, and it can be wrong about it. That is the entire subject of this
document.

## 3. The claim kernel replay has to make

Stripped of implementation, kernel replay asserts something specific about `N` executions of a
kernel:

> Given the same kernel, the same launch geometry, and the same arguments, if the memory the kernel
> reads is restored to the same contents before each execution, then each execution performs the same
> work — so the hardware events observed in pass *i* and pass *j* describe the same computation and
> may be combined.

Two things about this are worth stating plainly, because both are load-bearing and neither is
obvious.

First, the claim is about **observational equivalence, not identity**. The passes are not the same
execution and cannot be. Later passes see warm caches, a different TLB state, different clock
residency, and a different scratch layout. The claim is not that these are identical but that they do
not perturb the events being counted enough to matter — an assumption which is *true for most events
and false for precisely the events that measure cache and memory behavior*. A profiler that replays a
kernel and reports its L2 hit rate is reporting the hit rate of a kernel running on a cache that a
previous identical execution has already warmed. This is a limit of the method rather than a bug in
any implementation, and it is worth being explicit about it, because it is invisible in the output.

Second, the claim quantifies over "the memory the kernel reads" — which nobody knows. A profiler
cannot see a kernel's memory footprint without analyzing its machine code, so it cannot restore
exactly what the kernel touches. It has to restore a superset it *can* enumerate and hope the kernel's
footprint fits inside it. Everything difficult about kernel replay reduces to the gap between those
two sets:

- **Memory in the restored set but not the kernel's footprint** costs time and nothing else. This is
  the cost problem: a snapshot of an application's whole resident footprint to replay a kernel that
  touches one array.
- **Memory in the kernel's footprint but not the restored set** produces wrong answers with no
  diagnostic. This is the correctness problem, and it is the one that matters, because the failure is
  silent: the kernel's inputs differ between passes, the counters differ accordingly, and the profiler
  reports them as though they described the same work.

The second case is not hypothetical or rare. Whether an allocation lands inside the restorable set is
decided by *which allocator the application used*, not by anything about the kernel. An application
built against a stream-ordered allocator, an expandable-segment allocator, or a managed-memory
allocator can have its entire working set outside the restorable set while every individual step of
the replay reports success. Determining which real applications are in that position, and what to do
about it, is the substance of this study.

## 4. What the implementation actually does

The mechanism follows from the claim. To reconstruct a dispatch's inputs, the SDK intercepts the
dispatch at the point where it enters the hardware queue and expands it, on the submitting thread,
into a sequence:

1. **Establish exclusivity.** Take a per-agent writer lock, so no other dispatch on this GPU can run
   inside the window; ordinary dispatches take the reader side. Different GPUs use different locks, so
   replays on different agents proceed concurrently.
2. **Quiesce the device.** Fence against work already in flight on the submitting queue, then wait
   for every other queue on the agent to go idle — the lock stops new dispatches but cannot un-submit
   work that is already queued.
3. **Capture.** Copy every tracked device allocation owned by this agent, plus every module-scope
   `__device__` / `__constant__` variable, into host memory. This is the "restorable set" of §3.
4. **Run the passes.** For each pass: notify the tool, submit the dispatch with its completion signal
   suppressed, wait for the pass's records to be delivered, ask the tool whether to continue, and
   restore the captured image before the next one. The last executed pass is deliberately *not*
   restored, so the application receives the memory its kernel actually produced.
5. **Signal once.** Fire the application's original completion signal after the final pass, so the
   application observes exactly one execution regardless of how many actually ran.

There is no replay worker thread and no persistent state. Everything happens synchronously inside the
interception of one packet write.

Two properties of this design are worth noting because they shape the analysis. Because replay is a
callback-tracing service rather than a counter-collection mode, it is not tied to counters at all —
the same mechanism serves timing, PC sampling or thread trace, and a tool decides per pass which of
its services are active. And because it is per-dispatch, its cost and its correctness are both
properties of the individual dispatch, so the same application can have kernels that replay soundly
and cheaply alongside kernels that cannot be replayed at all.

## 5. What it costs

The cost model is short enough to state exactly. For one replayed dispatch with `P` passes, a
restorable footprint of `F` bytes, an effective host-link bandwidth of `B`, and a kernel time of
`T_k`:

```
    cost ≈ P · T_k  +  P · F / B
           └ passes ┘  └ capture + restores ┘
```

The second term usually dominates, and by a wide margin. `T_k` is typically microseconds to
milliseconds; `F` is the application's resident footprint, which on a modern accelerator can be tens
of gigabytes. The term that decides whether replay is usable is therefore `F / B`, and neither factor
is what one might assume:

- `F` is not the kernel's footprint but everything enumerable, which for a framework-based
  application means the whole allocator arena.
- `B` is not the link's headline bandwidth. The snapshot destination is ordinary unpinned host
  memory, so each region's transfer drives a pin / IOMMU-map / DMA / unmap cycle rather than a single
  large pinned transfer. Published measurements of the equivalent path put this in the low
  single-digit GB/s, not the tens.

Aggregating over an application, replay is worth its cost when the replayed dispatches' total
overhead stays below the application replay it displaces:

```
    K · F / B  ≲  T_app · (P − 1) / P
```

for `K` replayed dispatches. Read as a design constraint rather than an equation, it says: replay
wins when few dispatches are replayed and the footprint is small, and loses when either grows. That
is the opposite of the intuition that a per-dispatch method should scale better than a whole-program
one, and it is why kernel filtering is not a convenience feature but a precondition.

## 6. What this study establishes

This document set answers four questions, in order of how much they affect the roadmap.

**Which applications can be replayed soundly?** Decided by allocator provenance rather than by
anything about the kernels, which is not where one would look first. §2 gives the decision procedure;
§4 applies it to named production codes across HPC, AI, drug discovery, finance and robotics.

**What does it cost, really?** §1 derives the cost model above and measures its terms rather than
assuming them. §7 gives the experiments, each with an explicit pass/fail oracle, including the
preflight measurements that must happen before any of the cost claims can be trusted.

**Where is it provably correct, and where merely usually correct?** §6 states the state model and the
proof obligations the mechanism must discharge, and identifies which ones the current implementation
discharges, which it discharges under assumptions, and which it does not discharge at all. §3 covers
the same ground for multi-GPU, peer access, device partitioning, collectives and MPI, where the
per-agent isolation argument rests on application behavior more than it appears to.

**Is it the right mechanism at all?** §11 compares it to Nsight Compute's kernel replay and CUPTI's
checkpoint API, to AMD's own earlier attempts, and to iteration multiplexing — which is not a
strawman alternative but a cheaper, MPI-safe method that is sound under precisely the allocators
that defeat replay. §10 asks how much of the design transfers to other vendors' hardware and to
non-GPU accelerators. §5 covers the natural extension to ranges and graphs, and §8 and §9 collect the
design trade-offs, the lessons, and the prioritized recommendations.

The short version of the conclusion, stated up front so the rest can be read against it: the
mechanism is sound and the implementation is careful, but its domain of applicability is narrower
than the feature's framing suggests, and the boundary of that domain is currently invisible to users.
The highest-value work is not making replay faster or more general — it is making the boundary
visible, so that an application outside it is told so rather than handed numbers.
