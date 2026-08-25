# 8. Design evaluation against real-world use cases

*Written to be dropped into the internal Confluence design document as its "Real-world use case
evaluation" section. It is deliberately structured as an evaluation rather than a description: each
subsection states what we were trying to learn, what the evidence says, and what design position
follows.*

---

## 8.1 Why this section exists

The design document up to this point describes a mechanism: intercept a dispatch, lock the agent,
drain, snapshot device memory, re-execute N times, restore between passes, signal once. That
description is accurate and it is not sufficient to decide anything, because every interesting
question about kernel replay is a question about the *application it is pointed at*:

* Does the application allocate the kernel's outputs where the tracker can see them?
* Is one dispatch the unit the user wants a number for?
* Is the application's resident footprint 200 MB or 150 GB?
* Does anything other than an AQL queue write the buffers during the window?
* Can the application tolerate its GPU going quiet for several seconds?

The answers differ by orders of magnitude across the workload classes we care about, and they point
at *different designs*, not at one design with a longer limitations list. This section makes that
explicit so that scope decisions are made against evidence rather than against the mechanism's
current shape.

### 8.1.1 What "evaluated against" means here

For each use-case class we asked four things:

1. **What is the user's actual profiling question?** Not "collect these counters" — the reason they
   want the counters.
2. **What fraction of the work is even eligible** under the mechanism's gates (single packet, single
   dispatch, not a graph)?
3. **Is replay sound for that work**, in the specific sense of §6: is the kernel's write set inside
   the restored set, and is the window actually isolated?
4. **Is replay economic**, per the cost model $K \cdot F \lesssim B \cdot T_{app}\frac{P-1}{P}$ of
   §1.3.2?

A use case only "works" when all four line up. Most of the interesting findings are cases where
three of the four line up and the fourth kills it — and in several of those the fix is a different
replay *scope*, not a repair to the current one.

---

## 8.2 The use-case classes we evaluated against

These are grouped by the *shape* of the GPU work rather than by scientific domain, because the shape
is what determines the verdict. Domain-by-domain verdicts are in §4; this table is the design-facing
summary.

| Class | Representative workloads | GPU work shape | Footprint per GPU | Eligible fraction | Verdict |
|---|---|---|---|---|---|
| **C1. Dense-kernel HPC** | LAMMPS/Kokkos, LULESH, miniFE, stencil and finite-element codes | many launches of a few kernels, one or two streams, `hipMalloc`-provenance arrays | 10 MB – few GB | high | **works today**; the design's best case |
| **C2. Library microbenchmarking** | rocBLAS/hipBLASLt GEMM, MIOpen convolutions, BabelStream | isolated kernels, tiny surrounding state | small | high, except multi-kernel solutions | **works today**; also the least valuable, since these are easy to profile by other means |
| **C3. Large-footprint AI training** | transformer pre-training and fine-tuning, PyTorch eager or compiled | thousands of launches per step, caching allocator, collectives, sometimes graphs | 20 – 170 GB | medium in eager, near zero under graph capture | **blocked on economics** (footprint) and **soundness** (allocator mode), not on the core logic |
| **C4. Graph-captured inference serving** | batched LLM decode, compiled inference steps | almost all steady-state work inside graph launches | 10 – 170 GB | ≈ 0 | **blocked on scope**: needs graph replay (§5.1A) |
| **C5. Collective-bound distributed** | data/tensor/pipeline-parallel training, GPU-aware-MPI halo exchange | collectives interleaved with compute; peer progress required | large | eligible but unsound | **hard carveout**; today it hangs then aborts the job |
| **C6. Long-setup, short-kernel-of-interest** | docking and screening pipelines, MD after system preparation, inference after a long warm-up, simulation after mesh generation and partitioning | expensive setup, then a kernel someone wants to optimize | small to medium | high | **the strongest value case**, because application replay pays the setup $P$ times and kernel replay pays it once |
| **C7. Real-time / closed-loop** | robotics perception and control, streaming media, low-latency trading inference | fixed deadlines, sensor or network DMA into device buffers | small | eligible | **offline-only**; the window violates deadlines, and non-AQL DMA writers break isolation |
| **C8. Pure-device-state simulation** | quantum circuit state-vector simulation, some Monte Carlo engines | one enormous device array, no host interaction in steady state | up to all of HBM | high | **sound and useless without write-set restore**: the footprint *is* the state |

Two observations from this table drive the rest of the section.

**The classes where replay is sound are not the classes where it is valuable.** C1 and C2 work today,
but C1 codes launch the same kernel thousands of times — which means the cheapest correct way to
collect several counter groups is to collect different groups on different *iterations* and never
snapshot anything, and C2 is trivially profiled by running the microbenchmark again. The classes
where replay would be transformative are C3, C4 and C6, and of those only C6 works today.

**The blockers are three different kinds of problem, and they need three different responses.** C3 is
blocked by *cost* (a performance engineering problem: §9 R12–R14). C4 is blocked by *scope* (a feature
problem: graph replay). C5 is blocked by *semantics* (a carveout that must be detected, not fixed).
Conflating them into one "limitations" list is what makes the roadmap hard to argue about.

---

## 8.3 Research questions

These are the questions where we do not currently have the data to make a confident design decision,
stated with what evidence would settle each. Every one is answerable with a bounded experiment on a
single node, and several are answerable with the matrix in §7.

**RQ1 — What fraction of dispatches in production workloads passes the eligibility gate?**
The gate requires a single-packet, single-dispatch, non-graph submission. We do not know the real
distribution. *Evidence:* count dispatches with `--kernel-trace` and compare against replayed
dispatch ids, per workload in §7.3–§7.4. *Why it matters:* if the answer is 95% for HPC and 5% for
inference serving, the graph-replay work in §5.1A moves from "nice" to "the only way this feature
reaches the AI workloads it was justified by".

**RQ2 — How large is a kernel's write set relative to the agent's resident footprint?**
This ratio is the entire payoff of write-set-only restore. *Evidence:* instrument a device-side page
hash before and after a dispatch across the §7 workload suite and report the distribution of
(bytes written ÷ tracked bytes). *Why it matters:* if the median is 1%, R13 is a 100× win and should
be built before anything else; if it is 60%, R13 is not worth the complexity and the answer is
reachable-set narrowing (R12) alone.

**RQ3 — How much does replay perturb the measurement?**
Separate from correctness (§1.4). *Evidence:* collect one counter group with replay and the same
group without replay, on the same dispatch, and compare; then compare pass 0 against pass k within a
replay. *Why it matters:* if pass-to-pass drift on cache metrics is 5%, we document it; if it is 50%,
derived metrics that combine groups from different passes are not meaningful and cache control (R10)
becomes a correctness feature rather than a refinement.

**RQ4 — Is iteration multiplexing statistically equivalent to replay for iterative codes?**
For a code that launches the same kernel with the same shape thousands of times, collecting group A
on iteration *i* and group B on iteration *i+1* costs nothing and needs no snapshot. Replay's claim
to superiority is that it guarantees *identical inputs*, whereas multiplexing only gives
*statistically similar* ones. *Evidence:* collect the same metric set both ways on a C1 workload and
compare distributions; look specifically at whether the variance across iterations exceeds the
variance across replay passes. *Why it matters:* this is the single biggest threat to the value
proposition for HPC. If multiplexing is equivalent for C1, then replay's justified scope narrows to
C3/C4/C6 — which is a defensible and much clearer story, and it changes which optimizations matter.

The competitor's terms are now precisely known, which sharpens the question rather than settling it.
`rocprof-compute --iteration-multiplexing` needs roughly 50 dispatches per kernel to cover its ~15
counter subsets, keys kernel identity by name (optionally plus launch parameters), and **imputes**
values where coverage is short. So the boundary is quantitative: replay wins where a kernel of
interest executes fewer than ~50 times, or where kernel names are not stable across dispatches
(large AI models dispatching per layer break the identity key — AMD documents this). It also wins
where same-dispatch counter *correlation* is the point rather than aggregate rates. Everywhere else —
iterative HPC codes launching the same kernel thousands of times, which is most of C1 —
multiplexing is cheaper, MPI-safe, and, because it never touches memory, **sound under exactly the
allocators that make replay silently wrong** (§4.1). That last point is the strongest form of this
threat and it was not visible before §4: multiplexing is not merely a cheaper alternative for C1, it
is the *correct* one for Kokkos-on-ROCm-6.x, where replay produces wrong numbers.

**RQ5 — How often do real applications allocate kernel outputs outside the tracked set?**
*Evidence:* wrap the managed, stream-ordered and virtual-memory allocation entry points and report
untracked device-visible bytes per run, alongside tracked bytes, across the §7 suite and across
allocator configurations. *Why it matters:* it converts §2.2's argument from "this could happen" into
a coverage number per workload, and it tells us whether the untracked-coverage decline rule (R6) will
fire on 5% or 80% of real runs.

**RQ6 — Can a conservative reachable-set closure be made sound in practice?**
Narrowing the snapshot to what the kernel can reach requires discovering device pointers from the
kernarg segment and, transitively, from captured regions. *Evidence:* prototype the closure, compare
its result against the full-agent footprint and against a device-hash-derived ground-truth write set,
and count how often the closure misses a written region. *Why it matters:* R12 is the largest
single cost reduction available, and its risk is entirely "did we miss a pointer". A measured miss
rate of zero across the suite, plus a conservative fallback, is what makes it shippable.

**RQ7 — Is there a read-only fast path?**
A kernel whose write set does not intersect the tracked set needs no restore at all — replay becomes
free. Kernel argument metadata in the code object may indicate read-only pointer arguments, and a
device hash can confirm emptiness after pass 0. *Evidence:* measure what fraction of dispatches in
the suite are read-only with respect to tracked memory (reduction kernels writing only a small
accumulator are nearly free too). *Why it matters:* a cheap, high-confidence fast path that removes
the dominant cost for a subset of dispatches, with no carveout implications.

**RQ8 — Where is the crossover between device-resident and host-resident snapshots?**
*Evidence:* implement both and sweep footprint. *Why it matters:* it decides the default policy for
R14 and it determines how much free HBM the feature should be willing to consume.

**RQ9 — What do users actually want the unit of repetition to be?**
Dispatch, graph, annotated range, or application. *Evidence:* the fastest route is to prototype
user-driven range replay (§5.1B), which is small, and put it in front of two or three internal
consumers alongside per-dispatch replay. *Why it matters:* it is the difference between building
API-capture range replay (large) and exposing a pass loop (small).

---

## 8.4 Design questions and their trade-offs

Each question is stated with its options, the trade-off axis, and a recommended position. These are
the decisions the design document should record explicitly, because most of them are currently
implicit in the code.

### DQ1 — What is the snapshot scope?

| Option | Cost | Risk | Complexity |
|---|---|---|---|
| Whole agent (today) | proportional to the process's entire resident footprint | lowest — cannot miss a region the kernel touches | lowest |
| Reachable set from kernargs (+ transitive closure) | proportional to the kernel's operands | may miss a region reached by a pointer the closure did not find | moderate |
| Write set only, discovered by hashing after pass 0 | proportional to bytes actually written | needs a correct first pass and page-granular hashing | moderate |

**Trade-off:** conservatism against cost, and the conservative choice is the one that makes the
feature unusable on the workloads it was justified by. **Recommended position:** reachable set with a
conservative fallback to whole-agent, then write-set restore layered on top; keep whole-agent as an
explicit "paranoid" mode for validating the narrower modes. Note the ordering matters: reachable-set
narrowing helps *every* workload including read-only kernels, while write-set restore only helps the
restore half.

### DQ2 — Where does the snapshot live?

Unpinned host memory (today), pinned host memory, device memory, or device-first with host spill.
**Trade-off:** speed against HBM pressure. A device-resident snapshot runs at HBM bandwidth — roughly
two orders of magnitude faster than the current path — but consumes HBM in a process that may already
be near capacity, which is exactly the C3/C8 case. **Recommended position:** device-first with host
spill, and pinned host buffers when spilling. This is also what the closest prior art does, which is
a reasonable signal.

### DQ3 — Abort or decline on failure?

**Trade-off:** a decline risks the user silently getting no data; an abort risks killing a
multi-hour job. **Recommended position:** decline, always, with a machine-readable reason recorded in
the output. The current use of a fatal abort for drain timeouts converts every unsupported-kernel
encounter into a job kill, which is the one outcome strictly worse than not profiling. The single
exception is a partially applied restore, where continuing would corrupt application data — that must
stay fatal, and it already is.

### DQ4 — What is the isolation granularity?

Per HSA agent (today), per physical device, per peer-access clique, or process-global.
**Trade-off:** parallelism against correctness. Per-agent maximizes concurrent replays and is wrong
whenever peer access is enabled or the device is partitioned into multiple agents that share memory
and caches. **Recommended position:** keep per-agent as the *lock*, but make eligibility conditional:
decline when peer-exposed allocations are live, and widen the lock to the physical device when the
agent is one partition of several in use. Peer-clique windows are the correct long-term answer and
only become affordable once DQ1 and DQ2 are resolved.

### DQ5 — What is the cache and clock policy?

None (today), flush between passes, warm-up pass discarded, or user-selectable.
**Trade-off:** comparability against realism. Flushing makes passes comparable *to each other*,
which is what you need to combine counter groups — but it makes them all equally unlike the
production execution. Not flushing makes pass 0 unlike passes 1..P−1, which is worse, because it
introduces a systematic bias that the user cannot see. **Recommended position:** flush by default,
expose the policy, and record which policy produced the output.

### DQ6 — Is the pass count a cap or a minimum?

Today it is a hard cap and the continue callback can only break early; there is already a TODO
noting that treating it as a minimum would be useful. **Trade-off:** predictability against
adaptivity. Adaptive pass counts enable "keep going until the counter variance is below a threshold",
which is genuinely useful for noisy metrics, but it makes the cost unbounded. **Recommended
position:** keep the cap as the default, allow an explicit "minimum with a ceiling" mode, and require
the ceiling.

### DQ7 — Where does the pass loop live?

In the SDK (today), or in the application via a range API.
**Trade-off:** transparency against correctness and cost. An SDK-driven loop needs no source change
and cannot handle anything requiring external progress; an application-driven loop needs two lines of
annotation and handles collectives, MPI, real-time pipelines and graph-captured work, while making
the snapshot optional. **Recommended position:** both, with the range API prioritized higher than its
size suggests, because it is the only path to C5 and it dissolves rather than mitigates the cost
model.

### DQ8 — What is the output contract?

Today: P records sharing one dispatch id, distinguished by `replay_pass`.
**Trade-off:** fidelity against consumer simplicity. Sharing the dispatch id is right — it is one
logical dispatch — but it breaks any consumer that assumes dispatch ids are unique per record, and it
leaves open how a derived metric combining counters from different passes should be attributed.
**Recommended position:** keep the shared id, document it as a contract, and add the per-dispatch
provenance fields from R11 (footprint, timings, cache policy, canary consistency) so a consumer can
decide whether combining passes is legitimate for that row.

### DQ9 — Opt-in granularity

All eligible dispatches (today), or require a kernel filter above a footprint threshold.
**Trade-off:** ease of use against pathological cost. Defaulting to "replay everything" in a process
with a 150 GB footprint produces a run that never finishes, and the user's only clue is that it is
slow. **Recommended position:** admission control with a projected-cost diagnostic that names the
remedy.

### DQ10 — Detection versus documentation for unsound cases

**Trade-off:** engineering effort against silent wrong answers. Several unsound cases are cheaply
detectable (peer pointers in kernargs, collective kernel names, untracked allocation volume,
concurrent async copies if the copy APIs are intercepted). Others are not detectable in-process at all
(another process writing IPC memory, NIC RDMA). **Recommended position:** detect everything
detectable and decline; for the remainder, rely on the verify mode of R9 rather than on documentation,
because documentation does not protect a user who did not read it.

### DQ11 — Graph handling

Decline (today), replay the graph as a unit, or expand the graph into individual dispatches.
**Trade-off:** expansion would maximize per-kernel detail but requires re-deriving each node's
dependencies and defeats the purpose of graphs; replaying the graph as a unit is simple and gives
range-level numbers. **Recommended position:** replay the graph as a unit, declining graphs with
allocation nodes or collective kernels.

### DQ12 — Should device-memory checkpointing live in the profiler at all?

This is the architectural question the design document should confront directly. Snapshotting and
restoring device memory, tracking allocations, draining queues and handling partition and peer
topology is *runtime* functionality, not profiler functionality. On the comparable platform the
equivalent capability ships as part of the profiling and driver layers rather than being
re-implemented by each tool. Keeping it in ROCprofiler-SDK means the profiler carries an allocation
tracker that must stay correct against every future allocator change in HIP, and every new
allocation path (stream-ordered pools, virtual memory mapping, whatever comes next) silently
degrades the profiler's soundness with no signal.
**Trade-off:** shipping velocity now against a recurring correctness liability. **Recommended
position:** ship the profiler-side implementation, and open the conversation with the runtime team
about a supported "save/restore device state for this agent" primitive with allocation-provenance
awareness. The tracker's blind spots are all consequences of guessing at the runtime's allocation
behaviour from outside it.

---

## 8.5 How the trade-offs resolve differently per application class

The same design question gets different answers depending on the workload, which is the main argument
for making these policies configurable and for recording which policy was used.

| Class | Snapshot scope | Residence | Cache policy | Pass count | Failure policy | Right replay scope |
|---|---|---|---|---|---|---|
| C1 dense HPC | reachable set is enough; whole-agent is tolerable | host is fine at these sizes | flush | small $P$, kernel-filtered | decline | per-dispatch, or **iteration multiplexing instead** |
| C2 library benchmarks | reachable set | either | flush | any | decline | per-dispatch |
| C3 AI training | write set, mandatory | device-first, mandatory | flush | small $P$, aggressive filtering | decline | per-dispatch for eager; graph for compiled |
| C4 inference serving | write set | device-first | flush | small $P$ | decline | **graph** |
| C5 collective-bound | n/a | n/a | n/a | n/a | **detect and refuse** | **application-driven range** |
| C6 long setup | reachable set | either | flush | large $P$ is affordable here | decline | per-dispatch |
| C7 real-time | reachable set | device-first | flush | small $P$ | decline, and refuse when deadlines are declared | offline bench harness, not the live system |
| C8 pure device state | **write set is the only option** | device-first impossible (the state fills HBM) → host, pinned | flush | small $P$ | decline | per-dispatch with write-set restore |

Reading the table by column is instructive: the *cache policy* answer is the same everywhere, so it
should simply be the default. The *snapshot scope* and *residence* answers vary by two orders of
magnitude in their consequences, so they must be policies with good defaults and diagnostics. And the
*right replay scope* column is different for four of the eight classes, which is the strongest
argument in this document that a single per-dispatch mechanism cannot be the whole feature.

---

## 8.6 Lessons learned

Including from the basic case, where no carveout applies.

**L1 — The unit of profiling value is rarely a single dispatch.** Users ask "why is my attention block
slow", "why does my timestep take 40 ms", "where does my decode latency go". A per-dispatch mechanism
answers a narrower question than the one being asked, and the gap is filled by graphs and ranges
rather than by more per-dispatch fidelity. This is not a criticism of the implementation; it is the
reason §5 exists.

**L2 — Provenance decides soundness, not semantics.** We spent effort reasoning about which *kernels*
are safe to replay. The productive framing turned out to be: which *allocator* produced the buffers
the kernel writes. A kernel doing a floating-point atomic accumulation into a `hipMalloc` buffer is
perfectly safe; the identical kernel writing a stream-ordered pool allocation is silently wrong. That
reframing is what makes the §2 decision procedure short.

**L3 — Isolation is only as strong as the narrowest chokepoint you intercept.** The AQL write
interceptor is an excellent chokepoint for *kernel dispatches* and it sees nothing of SDMA copies,
peer-agent kernels, host stores to mapped memory, or NIC RDMA. Building an isolation guarantee on a
single chokepoint means the guarantee holds exactly for the traffic that flows through it. Every
remaining isolation gap in §6.3 is an instance of this one lesson.

**L4 — A bounded wait followed by a fatal abort is the wrong default for observability code.** It was
chosen for good reasons — hanging silently is worse than dying loudly — but the third option, declining
and continuing, is better than both and was not on the menu when the bound was written.

**L5 — Unobservability and measurement fidelity are separate correctness properties, and only the
first one was specified.** The mechanism is careful and largely correct about the application not
being able to *tell* that the kernel ran P times. It says nothing about whether the numbers collected
in pass 3 describe the same execution conditions as pass 0. For a profiler, the second property is the
product.

**L6 — A conservative design can be safe and still unusable.** Snapshotting the whole agent cannot
miss anything, which is why it was the right first implementation, and it scales with a quantity
(process resident footprint) that has nothing to do with the kernel being measured. Safety chosen
along the wrong axis costs three orders of magnitude on the workloads that matter most.

**L7 — What cannot be proven must be measured at runtime.** Two of the six proof obligations in §6
are unverifiable by construction from inside the mechanism. The only honest response is a runtime
check: a canary counter in every group, and an optional repeated-pass verification. This is worth
more than any additional documentation, because it turns an assumption into a per-run measurement
that the user sees.

**L8 — Getting the lock *shape* right early paid off; getting its *granularity* right did not.** The
reader/writer split is a genuinely good decision: ordinary dispatches stay concurrent with each other
and only pay a shared lock, and the writer window is what needs exclusivity. The granularity — one
lock per HSA agent — was fixed before peer access and device partitioning were considered, and both
break it. The shape survives; the granularity needs to become a policy.

**L9 — Making replay a callback-tracing domain with a per-dispatch, per-agent pass count was the
right call.** It decoupled replay from counter collection, allowed any per-dispatch service to use
it, and made heterogeneous nodes work without a global pass count. This is the part of the design
that should be preserved unchanged as new scopes are added, and it is why adding graph replay is a
small change rather than a redesign.

**L10 — Beta features need decline telemetry from the first day they exist.** Every gate in the
current implementation logs a one-shot warning and continues. That is the right behaviour and the
wrong observability: a user running a real application cannot tell how much of their workload was
skipped, and neither can we. The eligibility fraction is the most important number about this
feature and we currently cannot report it.

---

## 8.7 Future improvements in the basic case, with no carveouts involved

This subsection deliberately excludes everything about managed memory, collectives, MPI, graphs,
peer access, partitions and async copies. Assume the friendliest possible workload: **one GPU, one
stream, one thread, all buffers from `hipMalloc`, no other actors, kernel write set entirely
tracked.** In that basic case the mechanism is *sound*. It is still leaving most of its value on the
table, in ways worth listing separately from the carveout discussion because none of them requires
resolving a semantic question.

### 8.7.1 Cost, in the basic case

1. **Narrow the snapshot to the kernel's operands.** Even in the basic case, `snap()` copies every
   tracked allocation on the agent, not the three buffers the kernel uses. For a 4 MB kernel in a
   2 GB process this is a 500× overcharge, with no correctness benefit in the basic case at all.
2. **Do not restore what was not written.** A basic-case reduction kernel writes 4 KB and gets a full
   footprint restore between every pass.
3. **Skip the snapshot entirely for read-only kernels.** A kernel that writes nothing in the tracked
   set needs no snapshot and no restore. Replay becomes free. This is detectable after one pass with a
   device hash, and possibly statically from kernel argument metadata.
4. **Use a transfer path that can actually DMA.** The snapshot destination is an unpinned
   `std::vector`, so the copy cannot be a device-to-host DMA into it. Pinning the destination, or
   keeping the image in device memory, is a pure win in the basic case.
5. **Stop issuing one blocking copy per region, serially.** Basic-case processes still have dozens to
   hundreds of tracked regions, and each one pays full latency with no overlap. Batching small regions
   and issuing copies concurrently is straightforward and the existing perf test already sees the
   fixed per-inventory offset this creates.
6. **Overlap what can be overlapped.** Counter readout for pass *i* and the restore preparing pass
   *i+1* are independent for most of their duration.
7. **Offer iteration multiplexing as an alternative mode.** In the basic case of an iterative
   application launching the same kernel repeatedly, collecting a different counter group on each
   launch achieves the same end result with *zero* snapshot cost. It is a weaker guarantee (similar
   inputs rather than identical ones), and for many uses it is enough. Not offering it means users pay
   the strong guarantee's price whether or not they need it.

### 8.7.2 Measurement quality, in the basic case

8. **Flush caches between passes, or say that you did not.** In the basic case, pass 0 runs after a
   full-footprint device-to-host read and passes 1..P−1 run immediately after a full host-to-device
   write. Cache-sensitive counters are biased by pass index. Nothing about carveouts is involved.
9. **Offer a discarded warm-up pass.** Standard practice everywhere else in performance measurement,
   and free to implement given the pass loop already exists.
10. **Address clock and power state.** The window stalls the agent for seconds, then runs the same
    kernel P times back to back. Clock behaviour at pass 0 and pass 7 differ. At minimum record the
    clocks; better, expose a clock-control policy.
11. **Report the canary.** One counter present in every group, compared across passes, per dispatch.
    In the basic case it should always agree — which is exactly why a disagreement is worth an
    error: in the basic case, a disagreement means either kernel nondeterminism or a bug in the
    mechanism, and both are things we want to hear about.
12. **State the serialization effect.** In the basic case with one stream this costs nothing
    semantically, but the measured kernel still runs with the whole cache hierarchy to itself and no
    concurrent work. If the application normally overlaps that kernel with anything, the replayed
    measurement is of a different regime. Recording "measured under exclusive agent access" in the
    output prevents a class of misinterpretation.

### 8.7.3 Robustness and ergonomics, in the basic case

13. **Fix the address-reuse hole.** Even a basic-case, single-stream application has a host thread
    that may free and allocate during the window, because the allocation wrappers are outside the
    replay lock by design. A generation counter closes it.
14. **Decline instead of aborting.** A basic-case application with a slow kernel and an unlucky
    system hiccup can exceed a 5-second drain slice. Twelve of those and the process dies.
15. **Fail fast on reentrancy rather than hanging.** A tool whose callback launches a kernel is a
    basic-case programming error that currently produces an unbounded hang with no diagnostic.
16. **Add admission control with a projected cost.** In the basic case the user's mistake is usually
    "I asked for eight counter groups on every dispatch of a program with a 30 GB footprint". Telling
    them the projected cost and the remedy is more useful than being slow.
17. **Report footprint, timings and declines per dispatch.** Today the only way to learn the snapshot
    size is to read info-level log lines. This is the data both users and we need most.
18. **Make the output contract explicit.** P records share one dispatch id. Downstream consumers need
    to be told, and derived metrics that combine counter groups from different passes need a stated
    attribution rule.
19. **Document that indefinite loops require the continue callback**, which the implementation already
    enforces with a warning, and consider whether the pass-count-as-minimum variant is worth the
    unbounded cost it permits.

The point of separating this list from the carveout discussion: **even for the friendliest possible
workload, the largest single improvement available is a 2–3 order of magnitude cost reduction that
requires no new semantics, no new interception, and no resolution of any open correctness question.**
That should be the top of the roadmap, ahead of extending coverage to harder workload classes, because
extending coverage multiplies a cost that is currently 100× too high.

---

## 8.8 Open risks

| Risk | Consequence if unaddressed | Mitigation |
|---|---|---|
| Allocator drift in HIP | new allocation paths silently move memory out of the tracked set, degrading soundness with no signal | untracked-byte accounting (RQ5) plus a runtime-owned save/restore primitive (DQ12) |
| Iteration multiplexing is equivalent for HPC | the feature's value narrows to AI and long-setup cases, after the effort was justified by HPC | settle RQ4 early |
| Graph adoption increases | the eligible fraction of modern workloads trends toward zero | graph replay (DQ11) |
| Job kills in production | one abort on a large distributed run and the feature will not be enabled again | DQ3, immediately |
| Silent wrong numbers | the worst possible outcome for a measurement tool, and currently reachable without any warning | canary and verify mode (L7) |

## 8.9 Recommended positions, collected

For the design document's decision log:

1. Snapshot the reachable set, fall back to whole-agent, add write-set restore afterwards.
2. Device-resident snapshot with pinned host spill.
3. Decline and record; never abort except on a partially applied restore.
4. Keep the reader/writer lock shape; make its granularity a policy driven by peer exposure and
   partition topology.
5. Flush caches between passes by default; record the policy.
6. Pass count remains a cap by default.
7. Build user-driven range replay early; it is small and it is the only route to collectives and MPI.
8. Keep the shared dispatch id; add provenance fields to the output.
9. Require kernel filtering above a projected-cost threshold.
10. Detect everything detectable; use runtime verification for the rest.
11. Replay graphs as units rather than excluding them.
12. Open the runtime-owned checkpoint primitive conversation now, not after the tracker's third
    allocator surprise.
