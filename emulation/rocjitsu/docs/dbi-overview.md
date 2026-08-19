# DBI Overview

**What this document is.** A high-level description of where rocjitsu's dynamic
binary instrumentation (DBI) is going, why it is shaped the way it is, and which
decisions are settled. It is written above the level of the source tree (i.e.,
no file paths, no class names, no signatures) so that it stays accurate as
`develop` moves.

**What this document is not.** Not an implementation plan, and not a description
of what exists today. For the current state of the implementation read
`dbi-design.md`; where the two disagree, that document describes reality and this
one describes intent.

It also does not design any client of the framework. The concurrency sanitizer
motivates this work and appears throughout as a worked example, but its detection
algorithm, shadow state, and report format are deliberately absent, and no
separate document specifies them. The bar here is that someone building the
**framework** should find the framework's contracts answered.

Requirements are tiered throughout:

- **Committed**: the design targets this.
- **Must remain possible**: not being built, but the architecture must not
  foreclose it. A much weaker and cheaper obligation, and the distinction is
  load-bearing: do not implement one of these because it appears here.
- **Open**: a real decision not yet made, listed so it gets made deliberately
  rather than by accident.

---

## 1. Purpose and position

### The problem

Hardware performance counters tell you that something happened, not who did
it. A counter reports a bank conflict; it does not name the instruction or the
access pattern responsible. The attribution question is the one users actually
have, and the one counters structurally cannot answer.

Binary instrumentation answers it by inserting code that observes the program as
it runs. On AMDGPU this has historically been hard enough that it did not exist.
rocjitsu changes the economics: the machine-readable ISA foundation built for
simulation also provides the decode, analysis, and encode machinery
instrumentation needs. Instrumentation is not a capability bolted onto the
simulator; it is a second consumer of the same substrate.

### The first client

A **concurrency sanitizer** to detect data races in GPU kernels. It is the
milestone that justifies the work and it sets the priority: collect a memory
address stream at runtime, move it to the host, analyse it there.

The framework must be more general than its first client. Function and basic
block counters, branch analysis, and timing tools are all in the intended range.

**Two race detectors already exist in this project, and neither is what is
described here.** One is a simulator plugin: it does not use binary
instrumentation, it relies on the simulator's total observability, and it was
written for that environment. The other is a prototype, usable but not
production code and not a design to reproduce. Neither is the intended
architecture; neither is superseded by this document.

### Where this sits

- **Versus hardware counters.** Complementary. Counters are cheap, aggregate,
  unattributed. DBI is expensive, precise, attributed. A user reaches for DBI
  once a counter has told them a problem exists.
- **Versus Luthier and Dyninst.** Those are general-purpose GPU instrumentation
  frameworks with far more device-side capability. This is deliberately the
  constrained option: no external dependencies, a fixed catalogue of probes,
  streaming with host-side analysis. If a user needs arbitrary device-side
  logic, those are the right tools. Not a gap to be closed.
- **Versus rocjitsu's DBT.** Same substrate, opposite failure semantics (§2.4).

### Users

**Everyday users**: Mainly AMD engineers and AMD customers. Run a provided tool and
read a report. They write nothing. Instrumentation must be transparent: launch
the application under the tool, or attach to a running one, and get output. No
separate step to produce an instrumented binary, no rebuild, no relink.

**Advanced users**: Primarily AMD engineers that build tools on the framework.
Their surface is:

> **Site selection, probe choice, and host-side analysis, not probe authoring.**

That sentence is the boundary of the whole project. Advanced users say what to
instrument, pick from a catalogue of provided probes, and write the interesting
logic on the host, where there is a real language, real memory, and no register
pressure. Authoring probes requires a compiler, a stable ABI, and unbounded
device-side capability which is all out of scope (§9).

**Open:** whether AMD authors tools on customers' behalf or customers write their
own. This decides whether the advanced-user API is a supported product surface or
an internal API that happens to be usable. It need not be decided now, but it
must not be decided by neglect. The cheap hedge is to design as if external users
will touch it while committing to nothing.

---

## 2. The model

### 2.1 Variants and dispatch-time selection

The center of the design is not "instrument a kernel." It is **kernel variants
with dispatch-time selection**; instrumentation is one variant-producing
transform among several.

A variant is code plus a kernel descriptor, resident in device memory. Nothing
binds a dispatch to a particular variant except a pointer in the dispatch packet.
Therefore:

- **Producing a variant** is a pure code-object-to-code-object transform,
  normally performed when a code object is loaded, though nothing requires that
  timing and §8.2 depends on being able to produce one later.
- **Selecting a variant** happens per dispatch, by rewriting the dispatch packet
  as it is submitted. The cost is a table lookup.

That separation is what makes per-dispatch behavior possible without
per-dispatch instrumentation: sampling, instrumented-versus-original comparison,
routing dispatches to different buffers. It requires only that the framework
observe and edit dispatch packets.

**Committed:** instrumentation at load time; variant selection at dispatch time;
queue interception as a mandatory component, not an optimisation.

### 2.2 The forward-looking loop

**Must remain possible:** a tool that runs a kernel many times, observes it
through instrumentation, synthesises a new variant (a rewritten algorithm, not
merely an observed one), loads it at runtime, selects it for subsequent
dispatches, and instruments that to verify the result.

This is profile-guided runtime specialisation. It is not being built. It is
recorded because it imposes exactly one architectural requirement, and that
requirement is cheap to preserve:

> **Transforms are code-object-to-code-object functions, closed under
> composition.**

If that holds, synthesising and instrumenting a derived variant is the same
machinery applied twice. If instrumentation is instead written against "the
application's code object" rather than "a code object", the capability is
foreclosed. The same invariant is what lets instrumentation compose with binary
translation (§6.8).

### 2.3 Observational versus transformational

Two classes of transform, sharing all machinery, differing in their obligations:

- **Observational** transforms must not change what the program computes.
  Instrumentation is observational, and that is what makes it safe to enable
  without auditing the result.
- **Transformational** transforms deliberately change behavior. Correctness then
  depends on the transform being right, not on the framework being transparent.

Only observational transforms are committed. Transformational ones are must
remain possible, and should require explicit opt-in when they arrive.

### 2.4 DBI is not DBT

They share a substrate and have opposite failure semantics. Designers should not
import DBT's instincts.

| | DBT | DBI |
|---|---|---|
| Nature | Substitutive: replaces code so the app runs at all | Additive: inserts code alongside the original |
| Failure | Binary. One untranslatable instruction and you have nothing | Graded. A missed site costs coverage, not correctness |
| Needs | Completeness, caching | Coverage accounting, sampling |

---

## 3. Invariants

These hold regardless of implementation. Breaking one is a change to the plan,
not to the code.

1. **Transparency.** Instrumented code computes what the original computed.
   Architectural state the guest can observe, including the lane execution mask,
   condition codes, and mode bits, is bit-identical wherever the guest can
   observe it.

   This extends to state that is easy to overlook because it is not a register
   the guest names. **Probe memory operations perturb the guest's
   outstanding-memory counters**, and guest code performs arithmetic against its
   own count of outstanding operations; leaving those disturbed violates this
   invariant in a way that manifests as data corruption rather than a fault.
   Handling it is the framework's responsibility, not the probe's (§6.6).

2. **No silent loss of coverage.** Every instrumentation request resolves to
   instrumented, or to rejected with a reason. Every dropped record is counted.
   Every unsampled dispatch is accounted for. A tool must always be able to tell
   the user what it did **not** observe. A sanitizer that silently loses data is
   a sanitizer that silently lies.

3. **Transforms compose.** Code object in, code object out, no assumption about
   provenance.

4. **Instrumentation does not allocate device memory during execution** (§4).
   Buffers are obtained ahead of time by the host. One deliberate exception,
   which should be understood rather than glossed: raising the scratch
   reservation causes the runtime to allocate more scratch at dispatch. That is
   an allocation on instrumentation's behalf, it scales with occupancy, and it
   can fail. When it fails the application's queue is not degraded, it is
   stopped. It is acceptable because it is bounded and requested at
   instrumentation time rather than during execution. Note the mechanism: the
   dispatch packet's segment sizes are what the hardware honors, so raising
   the reservation means rewriting the packet, not the kernel descriptor
   (§6.5).

5. **The framework does not require serializing the application.** No mechanism
   in the design may depend on kernels running one at a time; that would change
   the behavior being measured and is very hard to walk back.

   This is a constraint on the framework, not a promise about every run. A
   sufficiently heavyweight tool can induce serialization as a consequence of
   what it asked for; most concretely, a large enough scratch request causes
   the runtime to admit one scratch dispatch at a time. The heavier the tool,
   the more it perturbs the application; that is inherent, and acceptable. What
   is not acceptable is it happening invisibly, so induced serialization is
   detected and reported (§7).

6. **The device never waits on the host for an individual record** (§5.5, noting
   the deliberate exception in lossless mode).

7. **Extension points are source-compatible, not binary-compatible.** Probes,
   client tools, and the framework are built and shipped together. A project-wide
   position, not a DBI-specific one.

---

## 4. The memory model

**The defining constraint: instrumentation gets a small, fixed scratch allocation
for spilling and makes no other device memory allocations.**

Accepted consequences:

- The framework does not track device memory or interfere with how the
  application uses it. Instrumentation is not in the way.
- **No on-device aggregation.** Everything is streamed to the host and reduced
  there. Tools that would naturally accumulate on the device still work (a basic
  block counter emits a record per execution instead) but they pay for it in
  link bandwidth (§7).
- The implementation is dramatically simpler.

That bill is the accepted trade and should not be re-litigated without new
information.

Buffers carrying data to the host are obtained from the system by the host,
ahead of time. "No device allocation" is a rule about who allocates and when, not
a claim that device-visible memory does not exist.

**Supportable, but not planned:** letting a tool supply device buffers of its
own, which would restore on-device aggregation and with it an efficient counter.
It is a generalisation of the same mechanism and needs no new delivery machinery
(§5.4). Recorded so nothing forecloses it. There is no present reason to build
it, and it carries a cost the current model avoids: tool buffers occupy cache and
capacity, perturbing the memory behavior of the application under study.

---

## 5. The data path

### 5.1 Where the buffer lives

**Committed: pinned, fine-grained host memory, for correctness and live
visibility rather than throughput.** This is not a claim that host memory is the
fastest place for a GPU to write; it very often is not.

The model requires the host to read records while the kernel producing them is
still running. That is what makes streaming analysis, bounded buffer reuse, and
draining possible at all. Host-resident fine-grained memory provides it directly:
the device writes, the host reads, and the platform guarantees coherence between
them with no third party involved.

Device-local memory can be moved to the host at high rate, since a copy engine
beats a host thread reading across the link. But that is a transfer of a region
whose extent must be known first, not a live view of memory the device is
concurrently writing. Getting live visibility that way means adding a scheduling
mechanism, a shared engine the application may itself be using, and a third
participant in the coherence argument. The choice is to keep the data path direct
and its correctness easy to state, and to pay for that in write throughput.

On parts where host and device share physical memory this distinction largely
dissolves; the reasoning above is about discrete parts.

### 5.2 The record-write contract

What a probe must do to write a record, and what that costs. Everything below
follows from the placement decision above, and the rest of §5 depends on it.

**A record must become visible to a host reader in bounded time.** Ordinary
global stores may sit in cache indefinitely, so the write path must use
system-scope-coherent stores. This is a correctness requirement, not an
optimisation: without it the drainer sees nothing and the flow control of §5.5 is
meaningless. What that means in instructions varies considerably by generation,
and on some the penalty is severe.

**The write is expensive at the source, not only at the link.** Fine-grained host
memory is mapped uncached, so writes do not coalesce the way device-local writes
do and a scattered per-lane pattern is costly where it is issued. This is the
main reason §7's arithmetic describes a ceiling rather than a rate the device
will sustain, and why the visible symptom of over-instrumenting is a slow kernel
rather than a full buffer.

**Claiming a slot requires an atomic the platform may not provide.** The slot
reservation that gives §5.6 its ordering is a device-side atomic into host
memory, and that depends on the host interconnect supporting atomic operations
from the device. Where it does not, fine-grained access may not be granted at
all. Availability is queryable, must be queried rather than assumed, and needs a
fallback where it is absent.

Together these bound how much a probe can afford to write, which is why record
density (§5.3) is treated as a performance property rather than a formatting one.

### 5.3 Record format

**Committed: a heterogeneous, self-describing record stream.** Every record
carries a tag identifying its layout and the host demultiplexes. New probe types
are added by adding record kinds, not by breaking the format.

**The framework defines the envelope; the tool defines the payload.** The
framework guarantees the stream is self-describing and that certain context is
obtainable by a probe: which dispatch, workgroup, and wave it is running in,
which lanes were active, which site it was attached to. What a probe records is
an agreement between it and its host consumer; a field no consumer reads need not
be filled, and identity should be treated as available-if-wanted rather than
mandatory. The distinction that matters is between what a probe **can know**,
which is a framework contract since a probe cannot report its dispatch unless the
framework makes that reachable, and what it **writes down**, which is a tool
decision.

Two granularities, both supported:

- **Wave-granular**: one record per wavefront per event, carrying the lane mask.
  Far cheaper in bandwidth and atomics. Lossy under divergence if it records only
  a representative lane's data.
- **Lane-granular**: one record per active lane. Complete, proportionally more
  expensive.

Wave-granular is the expected default, with lane-granular available where
fidelity demands it. Because the stream is typed, both coexist.

**Record density is a performance feature, not a formatting detail.** Every byte
is link bandwidth (§7), so density is a property of each record kind rather
than of the format. A low-rate diagnostic kind can afford generous fixed
metadata. A high-volume kind cannot, the obvious case being a per-lane address
stream, and it needs either a much larger useful payload per record or a
compressed encoding such as a base address plus per-lane deltas. The tagged-stream
design exists precisely so metadata chosen for one kind is not imposed on another.

### 5.4 How a probe finds the buffer

Three mechanisms can deliver the buffer address to a probe.

**Baked as an immediate** into the instrumented code. The framework knows the
address at instrumentation time and materialises it. Nothing is loaded or
negotiated, which makes it easiest to implement. Its limitation is structural:
the address becomes part of the code image, and an image is shared by every
dispatch of that kernel. Fine when a kernel runs once or its dispatches never
overlap; it cannot express a per-dispatch buffer, so concurrent dispatches must
share one.

**Passed through the kernel argument segment.** The dispatch packet points at a
host-written argument buffer, which the framework can substitute with an extended
copy carrying its own pointer alongside the application's arguments. The most
robust of the three, and the only one per-dispatch by construction, since the
argument buffer is itself per-dispatch. It costs a scalar load. This presumes the
kernel receives an argument pointer at all; kernels taking no arguments may not,
in which case one must be enabled, subject to the shifting and the ceiling
described below.

Nothing stops the framework from pointing the packet at a buffer of its own.
Three conditions govern whether that buffer is a valid one, and each is a silent
corruption if missed:

- **Where it comes from.** Argument buffers must be allocated from the memory the
  runtime designates for that purpose, and must satisfy the kernel's declared
  alignment. An ordinary allocation that happens to be device-visible is not
  sufficient.
- **What it contains.** The application's arguments are followed by implicit
  arguments the toolchain adds and the kernel expects to find. The extended copy
  reproduces both and appends after them; writing over the implicit region
  corrupts a kernel that never declared those arguments and cannot be blamed for
  reading them.
- **How long it must live.** Until the dispatch completes, which the framework
  has no direct way to observe. A dispatch packet carries at most one completion
  signal, and applications frequently leave it unset, so the framework cannot
  simply attach one to learn when reclaiming is safe. It must either introduce
  its own synchronisation into the submission stream or infer completion from
  the queue's own progress. Both work; the choice should be deliberate, because
  reclaiming early corrupts a dispatch still reading the buffer.

**Read from the dispatch packet.** A wave can be given a pointer to its own
dispatch packet in a preloaded scalar pair. This does not carry the buffer
address, since the packet is a fixed hardware-defined structure with nowhere to
put one, so it reaches the buffer indirectly via the argument buffer the packet
names, which is strictly more work than reading the argument pointer directly.
Its value is as a fallback for kernels not compiled to receive an argument
pointer, and as a source of per-dispatch identity records may want anyway.

Enabling a preloaded register a kernel did not already have is possible but not
free. These registers are assigned positionally, meaning those enabled are laid
out in a fixed order, so turning one on shifts everything after it, and the
dispatch pointer sits early, ahead of the argument pointer and the workgroup
identifiers. The shift does **not** require rewriting guest instructions: a short
prologue copying displaced values back down costs one move per value, and the
translation path already does exactly this. The binding constraint is elsewhere.
Hardware initialises only a small fixed number of these registers, so on a kernel
that already consumes that budget, insertion is not expensive, it is unavailable.

The kernel descriptor is not a candidate: it is hardware-consumed dispatch
configuration, every field architecturally defined, and a wavefront cannot read it
as data.

**The lean is toward the kernel argument segment**, because per-dispatch binding
is what the rest of the design wants. A pool of buffers allocated up front and
bound per dispatch satisfies the memory model while still allowing concurrent
instrumented kernels (invariant 5). A per-code-object buffer with a baked
immediate remains a legitimate first implementation, and the choice is
recoverable: it changes where the value comes from, not what probes receive.

Two optimisations, neither required:

- **Hold the pointer in a reserved scalar pair for the kernel's lifetime.** A
  framework-inserted prologue at kernel entry, which §8.3's required
  initialisation instrumentation means must exist regardless, loads it once and
  passes it to probes as an argument, removing the per-site load while keeping
  per-dispatch binding. It costs two scalar registers held for the whole kernel,
  which the framework can create by growing the allocation (§6.5, tier 3) rather
  than hoping to find dead ones. Whether that costs occupancy is
  generation-dependent.
- **Fold the address into the code** where the buffer genuinely is per code
  object, collapsing back to the baked immediate.

### 5.5 Flow control and loss

The device must know how much space remains, or it will overwrite records the
host has not read, silently and undetectably. This aggregate flow control, a
single shared consumed-index, is required, and it is **not** per-record
acknowledgement.

Per-record acknowledgement, meaning the device waiting for the host to confirm
each record, is **rejected**. It puts host latency in the wave's critical path and
makes the host a liveness dependency. Request/response transports designed for
low-volume device-to-host calls are the wrong shape for a high-volume one-way
stream and should not be proposed for this path. They are reasonable for
development-time debugging output, which is not on the measurement path.

Two properties hold in both modes below.

**Loss respects the boundaries of an event.** Where one occurrence of an
instrumented event produces several records, most clearly one per active lane
of a wavefront executing one instruction, those records all land or none do.
Dropping part of a group is worse than dropping all of it: a consumer computing
an aggregate over a wavefront's addresses gets a wrong answer rather than a
missing one, and cannot tell the difference. Drop granularity is the event, not
the record.

**Drops are counted per record kind.** Loss is not uniformly harmful. For a race
detector, losing an access record costs a missed race, an incomplete answer.
Losing a synchronization record costs a fabricated one, because the ordering
edge that would have explained an apparent conflict is gone. Per-kind accounting
is what lets a client distinguish "I saw less than everything" from "I may be
about to report something untrue", and refuse to answer in the second case.

**Lossy (default).** Increment the dropped count and skip the store. Never
blocks, bounded overhead, records are a sample.

**Lossless.** The wave waits for space. No data loss, at real cost:

- The host drainer becomes a **liveness dependency of the GPU kernel**, a
  permanent property of the mode, and the reason lossy is the default.
- It requires a drainer on an **independent thread**. If the drainer is the
  application thread and that thread is blocked in a synchronous kernel launch,
  the result is deadlock: the kernel cannot finish until the host drains, and the
  host cannot drain until the kernel finishes.
- It requires a **bounded wait with a defined escalation**. There is no
  per-dispatch watchdog that simply kills a slow kernel; what a spinning wave
  actually endangers is preemption. The driver must be able to remove the
  queue from the hardware, to evict memory, to schedule another process, or at
  teardown, and a wave that cannot be preempted turns that into a timeout,
  which can escalate to a device reset affecting other processes on the same
  device. The likelier everyday outcome is a wedged process rather than a
  reset, but the multi-tenant hazard is real and on shared systems argues
  against the mode by itself. Two obligations follow: the spin must use an
  instruction that leaves the wave preemptible rather than a tight busy loop,
  and on reaching the bound the framework either degrades to dropping (recording
  the transition) or aborts with a diagnostic. The bound is a hard requirement,
  not a quality-of-implementation matter.
- It is **not viable for cooperative or persistent kernels**, where all
  workgroups are resident simultaneously and no wave can retire to make room. A
  full buffer there is a deadlock the escalation bound can only convert to an
  abort.
- It **heavily perturbs timing**, changing the interleaving of the program under
  study. Harmless for happens-before analysis, fatal for anything inferring
  behavior from timing. A per-client judgement.

**Supportable, but not planned:** reserving capacity for protected record kinds
so low-priority records drop first while high-priority ones are always admitted.
A middle ground costing one comparison on the low-priority path. Not part of the
plan; a client that cannot tolerate loss should use lossless mode.

A flat fill-once buffer that stops when full is a legitimate simpler starting
point. It shares the record format; only space reclamation differs.

### 5.6 Ordering and time

**Records are totally ordered by their position in the buffer.** Producers claim
a slot by atomic increment, so the slot index is itself a sequence number,
costing nothing beyond the reservation that was happening anyway.

Note carefully what that does and does not give a consumer. It is the order in
which probes claimed slots, not the order in which instrumented instructions
executed, and emphatically not a time base. It is enough to recover the relative
order of records from a single wave, which is what reconstructing that wave's
behavior requires. Anything stronger must be built by the client from data it
records itself.

**There is no timestamp requirement.** A concurrency sanitizer establishes
happens-before from the program's **synchronization structure**, meaning
barriers, atomics, and fences, not from time. That has a direct consequence for
probe design: some probes must observe an instruction's outcome, such as an
atomic's return value or success mask, not merely that it executed (§6.3).

**Must remain possible:** timing tools, which do need a clock in the record.
Nothing in the format should assume time is absent, but a clock is payload, and
adding one is a tool's decision rather than a framework change.

---

## 6. Instrumentation

### 6.1 Site selection

**The durable identity of a site is (code object identity, offset).** A virtual
address is the runtime projection of that identity, valid only for one load of
one image. Anything persisted uses the durable form; a live API operating on a
running process may use addresses. The framework maintains the mapping.

The low-level selection API is offset-based and complete. Every convenience layer
compiles down to it, so layers can be added, replaced, or supplied by users
without touching the core:

- **by symbol or kernel name.**
- **by source location**, where line information is present, and frequently it
  is not: release builds ship without it, vendor libraries are stripped, and some
  of the kernels users most want attributed are hand-written assembly with no
  source to point at. Source attribution is best-effort, not a guarantee. Where
  available, resolution belongs on the host: a record identifies its site and the
  host resolves it afterwards. Baking file and line into records spends scarce
  link bandwidth on data identical for every occurrence of a site and derivable
  from what the host already holds.
- **by basic block**, including block exits as first-class sites rather than
  only entries. Any analysis bracketing a region needs a probe on every path out
  of it, early exits included, or its pairing is incomplete.
- **by predicate over decoded instructions**, such as "all global loads" or "all
  atomics". The layer that matters most to the first client, and the decode and
  instruction-property machinery to evaluate it already exists. The properties
  such a predicate needs, being access size, read versus write, and address
  space, are recoverable from the instruction encoding, so this layer does not
  depend on debug information.

Selection must compose with the runtime filtering of §7. A user asking for "these
instructions, in this kernel, only for these workgroups, only for this range of
dispatches" is expressing one intent through two mechanisms: sites are chosen at
instrumentation time, dispatches at submission time, anything finer inside the
probe.

Selection can fail for a given site. Per invariant 2, failure is reported with a
reason, never silently.

### 6.2 Relocation and instrumentability

**Instrumentability is a relocation problem.** Nearly any instruction can be
instrumented. Instrumentation displaces code, and the instructions that resist it
are exactly those whose meaning depends on their own address:

- **Instructions that read or write the program counter.** A program-counter read
  yields the address of the following instruction; if that feeds arithmetic that
  later computes a branch target, moving the code changes the result. Either the
  dependent instructions are adjusted for the move, or a different instruction in
  the same block is chosen.
- **Branches.** If the anchor is a branch, instrumentation must execute before
  the original branch does, and a relative branch that moves must have its
  displacement re-encoded to still reach its target.

These are not exotic cases needing separate machinery. Whatever relocation
strategy is chosen, instructions move and this bookkeeping is required. It is the
discipline binary translation already enforces:

> **Anything naming a code or data address is a fixup, never a final value
> written at emission time.**

DBI inherits that rule wholesale. One consequence: the relocation bookkeeping
required for correctness is the mapping from instrumented addresses back to
original ones. Other tools need that mapping (§8.1); maintaining it is not extra
work undertaken on their behalf.

**A rewritten image must remain a valid one.** The loader is not a byte pipe. It
checks that the image targets the agent exactly, both architecture and the
feature flags that distinguish otherwise-identical targets, along with segment
sizes and alignment, relocations, and the metadata note. A mismatch is a rejected
load, not a degraded one.

The metadata note deserves particular attention, because it is not merely
descriptive. **It, rather than the kernel descriptor, is where the runtime
answers questions about a kernel's resource requirements**, and those answers
are what the application's runtime uses to populate dispatch packets. An
instrumented image whose note still describes the original therefore does not
merely misreport; it causes packets to be built for the wrong kernel. This is the
same hazard as §6.5's, seen from the other end.

#### Relocation strategy (OPEN)

How instrumented code is laid out is **not decided**. It is recorded rather than
deferred silently, because the choice determines what the emitter must be capable
of and is expensive to revisit.

The forcing constraint: relative control flow reaches roughly ±128 KB, and that
bound applies symmetrically to the branch out to a trampoline and the branch
back. Absolute-target control flow has unlimited reach but takes its target from
a register pair, and the sequence to set one up is far too long to substitute for
a single instruction.

**This is not a tail case.** The workloads this framework serves are large,
heavily unrolled, heavily fused kernels, many per application. Anchors beyond
reach are the normal condition.

1. **A single cave appended after the existing code.** Exactly one instruction is
   substituted in place and everything else appended, so no instruction moves
   and every original branch keeps its distance. The single-slot substitution is
   a hard constraint, not an incidental one: overwriting a longer span risks
   clobbering an instruction that is itself a branch target. Only site-to-cave
   reach binds, and that is where it eventually fails. Note that "nothing moves"
   is true of code and not of the container: growing the code section displaces
   what follows it in the image, including the kernel descriptors, unless the
   added space is given its own loadable segment. **The current short-term
   answer.**
2. **Islands interleaved during a rewrite of the existing code section.** Reach is
   solved by layout, at the cost of moving everything.
3. **A second code section** holding relocated, instrumented copies of whole
   functions, with the original section's entries redirected across. Trampolines
   become unconstrained: there is room to spend instructions on reaching, and
   probe bodies could be inlined into the relocated function, reversing the
   not-specialised-per-call-site consequence of §6.7.

A two-hop scheme through a minimal nearby stub is **rejected**: prior binary
translation work established it as insufficient.

Strategies 2 and 3 both relocate whole functions, meaning branch targets and
inlined data, and both, by moving and growing code, can push a function's own
internal branches out of reach. Both therefore require the emitter to relax an
out-of-range relative branch into a longer reaching sequence. That is a genuine
cost of choosing either, and it does not apply to strategy 1.

Distinguishing 2 from 3: strategy 3 leaves the original section byte-identical to
compiler output, so original offsets stay valid for anything else reading the
image, such as a debugger or another tool's samples (§8.1). Strategy 2 moves
everything. Against strategy 3: code size, and instruction-cache pressure from a
second copy larger than the original, which is precisely the pressure large and
numerous kernels already create.

One argument is **not** available to either side: preserving an uninstrumented
variant. The original code object's bytes exist on the host under any strategy
and can be loaded as a separate variant, so no layout choice forecloses running
uninstrumented (§2.1).

### 6.3 Probe placement

A probe runs **before** or **after** the instruction it is attached to, per
probe:

- **Before**: reads the instruction's source operands. Address capture works
  this way.
- **After**: reads the instruction's results. Capturing an atomic's return value
  or success mask requires it.

### 6.4 Execution mask policy

The instrumented sequence has three regimes with respect to the lane execution
mask. Only one is a policy choice.

1. **Save and restore window, all lanes enabled. Mandatory.** The probe may run
   under a wider mask than the guest, or alter the mask itself, and can therefore
   clobber physical registers of lanes the guest had inactive. The save must have
   captured those lanes for the restore to put them back; saving under the
   guest's mask would leave them unsaved and silently corrupted. Note that with
   all lanes enabled, a save spilling to memory issues addresses for lanes the
   guest had masked off, which is correct for register preservation but is a real
   interaction with the spill-destination choice.
2. **Probe body, policy.** See below.
3. **The guest instruction, the guest's original mask. Mandatory.** Invariant 1.

The policy in (2) is best expressed not as "restore or not, for speed" but as a
declaration about the probe:

> **Does this probe's body operate on guest lane state?**

Probes reading guest vector state or emitting per-lane data must run under the
guest's mask; with all lanes enabled they would read garbage from inactive lanes
and emit junk records. Probes whose body is uniform, such as a counter or a
timestamp, can run with all lanes enabled and save the restore.

Note the difference between **the mask as data** (a probe recording which lanes
were active) and **the mask as control** (a probe wanting its own work
predicated). The former is available from the saved copy; whether it is free
depends on where the save allocator put it (§6.5).

Wave sizes differ across targets and modes. Probe declarations are wave-size
agnostic; the framework materialises the correct widths.

### 6.5 Register pressure and spilling

Two facts make this tractable, and both are static: the probe's register
footprint is known because the framework has the probe, and the guest's live
registers at the insertion point are known from liveness analysis. Their
intersection is the minimal set that must be preserved; the complement of the
guest's live set is the pool of available destinations.

That yields a cost hierarchy, each tier cheaper at run time and more work at
instrumentation time than the next:

1. **Registers dead at the site.** Cheapest, and it does not change the kernel's
   register allocation at all.
2. **Rewrite the probe to fit what is available.** Register renaming on a compiled
   probe is the same decode-analyse-emit machinery instrumentation already
   requires, so this is not a new capability class. Tuple alignment must be
   respected, and architecturally fixed registers cannot be renamed.
3. **Grow the kernel's register allocation.** Costs occupancy. Requires reading
   and modifying the kernel descriptor's resource fields.
4. **Spill to scratch.** Costs memory latency and draws on the small fixed
   budget. Many small kernels reserve no scratch at all, in which case this tier
   does not exist until the framework creates it, establishing scratch state the
   kernel never had rather than merely enlarging a reservation.

**Growing a memory resource is a dispatch-packet edit, not a descriptor edit.**
The segment sizes the hardware honors come from the dispatch packet; the sizes
recorded in the code object are what the application's runtime reads in order
to fill that packet. Raising a reservation therefore means editing the packet at
submission, and if the recorded sizes are left stale the two disagree. The
failure is not a fault: accesses beyond the provisioned region are silently
discarded or read as zero. This is the single easiest way to corrupt a guest
while believing instrumentation is transparent.

Two further consequences of growing scratch specifically. A large enough request
causes the runtime to admit one such dispatch at a time, which is the induced
serialization invariant 5 permits but requires be reported. A request that cannot
be satisfied does not degrade the dispatch, it stops the application's queue.
Both argue for treating tier 4 as genuinely last-resort rather than as a
comfortable fallback.

This hierarchy is why the memory model of §4 is survivable: most spilling should
never touch memory.

Two things about where the pressure lies, both generation-specific and to be
resolved against the target's occupancy model rather than assumed:

- **Whether scalar-register pressure costs anything varies by generation.** On
  some, scalar registers are allocated per wave at fixed size and do not enter
  the occupancy calculation; on others they are drawn from a shared pool and do.
  Effort spent easing scalar pressure, probe rewriting in particular, pays off
  very differently per target, and may not pay off at all.
- **Vector register growth costs occupancy broadly**, and is the one
  instrumentation decision that can visibly change application performance. Where
  a separate accumulation register file exists physically, unused accumulation
  registers can be a cheap spill destination; where the file is unified with the
  vector file they are not free, because the split is part of the kernel's
  declared allocation and changing it can itself change occupancy.

**The quality of instrumentation is bounded by the quality of the liveness
analysis.** Conservative liveness means more spilling means more overhead, and
anything defeating control-flow reconstruction degrades to worst-case spilling.
The binary analysis layer is load-bearing, not incidental.

### 6.6 Outstanding memory operations

Instrumentation must reckon with the guest's in-flight memory operations. Two
independent hazards.

**Spilling a register that is the destination of an in-flight load** saves a
stale value. The load then lands, and the restore overwrites it with the stale
copy, corrupting the guest irrespective of any other bookkeeping.

**Inflating the guest's outstanding-operation counters** breaks relaxed waits.
A wait for zero outstanding operations is harmless; instrumentation only makes
the guest wait longer. But compilers routinely emit waits permitting some
operations to remain outstanding, and if instrumentation has added to that count
such a wait is satisfied too early, and the guest reads a register whose load has
not landed. This is silent data corruption, and it is the more dangerous of the
two because nothing about it looks like a fault.

Two hardware properties constrain the response. The counters cannot be read or
restored, so there is no save-and-restore. And a probe cannot wait for only its
own operations: the counters are counts rather than tags, and the probe's
operations are the newest, so waiting for the count to fall to any threshold
retires the oldest entries first, which are the guest's.

**The framework therefore drains outstanding operations on entry to the
instrumented sequence.** It is required anyway for spill correctness, and it
disposes of the relaxed-wait hazard at the same time: once the guest has nothing
outstanding, every subsequent guest wait is satisfiable and its data has
definitively arrived. Operations the probe leaves outstanding when the guest
resumes can then only make the guest wait longer, never less. A separate drain
before the kernel terminates ensures record writes cannot race termination.

**This is a framework obligation, not a probe one.** The probe is context-free and
shared across every site; only the framework knows the guest's state at a given
site. A probe declares the memory operations it issues; the framework emits the
waits. Counters are split and named differently across generations, so this is
per-counter and target-aware.

Two refinements are available later and are not required for correctness:
narrowing or eliminating the drain where analysis proves no in-flight destination
overlaps the spill set and no relaxed wait follows; and, rather than draining,
adjusting the guest's subsequent wait thresholds by the probe's known operation
count. The latter is exactly semantics-preserving and free at runtime, but must
contend with control-flow merges where a wait is reachable from both instrumented
and uninstrumented paths.

**The cost is easy to underestimate.** An entry drain destroys memory-latency
hiding at every instrumented site. On a memory-bound kernel with a probe on every
access it is plausibly a larger contribution to overhead than the record traffic
§7 treats as the governing constraint.

### 6.7 Probes

Probes are standalone compiled objects, built as part of rocjitsu, made resident
in the instrumented image, and called.

**What a probe may not do.** Committed constraints, which the framework verifies
rather than trusts:

- **No scratch.** A function that spills requires valid private-segment state,
  and the guest's scratch belongs to the guest.
- **No local data share.** It is allocated per workgroup at dispatch and the
  guest owns all of it.
- **No nesting or recursion.**
- **No state across invocations**, which follows from the first two. An analysis
  that would naturally open a region and close it, such as timing a block or
  pairing an entry with an exit, cannot hold the opening value anywhere. It
  emits a record at each end and the host pairs them. That costs a second record
  and buys the guarantee that probes are stateless and re-entrant.

**How probes are reached.**

- **Called, not inlined**, via a call primitive taking an absolute target from a
  register pair, so the call itself is unconstrained by range. The branch from
  the site into its trampoline is constrained; see §6.2.
- **Therefore not specialized per call site.** Record kind, granularity, and
  per-site constants are passed rather than folded in. Code size stays flat;
  argument setup is paid at every site. This is why the argument-passing
  convention carries weight.
- **Placement is free.** A probe need not live in the instrumented kernel's code
  section, only be resident, executable, and reachable, which absolute
  addressing guarantees. Co-location is a lifetime and loader-simplicity choice.

**The probe ABI.**

- **Internal and versioned in lockstep**, because probes ship with the
  framework, so drift cannot happen in the field.
- **Register footprint is inferred, not declared.** The framework decodes the
  probe and computes which registers it reads, writes, and disturbs, the same
  analysis it already performs on guest code. A declaration cannot be guaranteed
  for a compiled probe: the compiler owns register allocation, so an annotation
  is a claim about an artifact that a different compiler version or optimization
  decision can invalidate unnoticed. The failure is silent and severe, since the
  framework then preserves the wrong registers and corrupts the guest.
  Declarations are trustworthy only where the author fixes the registers, which
  means hand-written assembly.
- **Inference yields clobbers only.** Where the probe's arguments and return
  address live is fixed by the calling convention of the toolchain that built
  it, not by one this framework invents, so calls must conform to that
  convention.
- **Must remain possible: user-authored probes.** The blocker is that the
  call-site sequence must match the probe's calling convention, which is only a
  blocker if that sequence is hand-written per probe. The requirement is
  therefore narrow and cheap:

> **The calling convention must be data the framework consumes, a probe
> descriptor giving footprint, argument locations, placement, and mask policy,
> never assumptions baked into the emitter.**

**How probes are built.**

- **Compiled from source, for the targets a build needs.** A device compiler is
  therefore a prerequisite of instrumentation, and that dependency should fall
  on instrumentation rather than the project as a whole: someone building only
  the simulator or the translator should not need one.
- **The target set is configuration, not a fixed matrix.** Hardware executes one
  target, so building for the local system is the sensible default; the
  cross-target case arises under the simulator, and cross-compilation is a
  matter of asking the toolchain for a different target rather than having that
  hardware present. The buildable set is bounded by the toolchain's vintage, so
  a target this framework supports may be unbuildable on an older compiler.
- **Wave size is a genuine variant axis in a way target count is not.** Where a
  target supports more than one wave size, the probe must match the guest
  kernel's wave size, because mask widths and lane semantics differ. Those
  targets need a probe per wave size, selected at instrumentation time.
- **Testing covers mechanisms, not the full matrix.** What needs per-target
  coverage is each distinct mechanism a probe embodies; probes sharing a
  mechanism and differing only in payload do not each require every target.

### 6.8 Composition with translation

Instrumentation must be able to run **after** binary translation, because the
translated code is what executes. This follows from invariant 3 and costs nothing
if transforms are a pipeline behind a single interception point; it is a fight if
each transform is an independent hook competing for the same mechanism.

One consequence for the analysis layer: instrumentation cannot assume it is
looking at compiler output. It may be looking at machine-generated translation
output, with different register-pressure characteristics.

---

## 7. Performance

### The governing arithmetic

**The device's capacity to generate records exceeds the host link's capacity to
carry them by two to three orders of magnitude on discrete parts.** No amount of
drainer optimisation changes that. Where host and device share physical memory
the gap narrows substantially; the design should not assume the discrete case is
the only case.

What that gap describes is a ceiling, not a rate the device will sustain. Because
the buffer is uncached host memory (§5.2), the write path throttles the producer
directly: the observable effect of over-instrumenting is waves stalling on their
own record writes, not an orderly queue draining at link speed. The practical
consequence is the same, produce fewer records, but the symptom a user reports
will be a slow kernel rather than a full buffer.

The ordering of bottlenecks is: device produces ≫ link carries ≥ host
processes. Post-processing matters because it must not become a new bottleneck
below the link, but making it faster buys nothing beyond the link ceiling.

What buys throughput is **producing fewer records**:

- **Sampling.** Dispatch-granular selection supplies the most practically
  important controls the framework offers: **instrument only a named kernel,
  only dispatches in a numbered range, or only within a time window.** These are
  not conveniences; they are what makes the framework applicable to workloads
  that cannot be shrunk.

  The decision is a table lookup, but the mechanism is not free. Observing
  packets requires interposing on submission, which routes every dispatch in the
  process through the framework, including those sampling declines to
  instrument, and adds latency to each. Applications issuing very many small
  dispatches pay this whether or not they are being sampled. It is the price of
  admission for everything in §2.1, and it should be counted once, honestly,
  rather than assumed away.

  Finer granularity, such as per workgroup, per wave, or every Nth execution of
  a site, happens inside the probe and costs cycles there instead. The two
  mechanisms should not be conflated.
- **Filtering in the probe.** The earliest point volume can be reduced. It costs
  cycles rather than memory, so it survives the memory model. This is the
  design's main lever and should be a first-class probe capability.
- **Denser records** (§5.3).

One caution against reading link bandwidth as the only constraint: draining the
guest's outstanding memory operations at each site (§6.6) removes latency hiding,
and on memory-bound kernels can cost more than the records do. Record volume
dominates what the host sees; the drain dominates what the device pays.

### Overhead budget

**There is no fixed overhead target.** Overhead must instead be bounded,
predictable, and reported. A number would be meaningless: cost is dominated by
what the user chose to instrument, and occupancy effects make it nonlinear, since
a probe costing a handful of cycles can more than halve throughput if register
growth crosses an occupancy threshold.

The framework reports, per run:

- wall-clock for instrumented and uninstrumented dispatches, per kernel. The
  baseline comes from dispatches that ran the original variant, which sampling
  supplies for free; a dispatch cannot be re-run to obtain one, since kernels
  mutate memory. **Under full instrumentation there is no in-run baseline** and
  the application must be run again without instrumentation.
- occupancy change. Occupancy is not determined by register counts alone: local
  data share per workgroup, workgroup size, and wave-slot limits also bind, and
  for kernels limited by those an occupancy figure inferred from register deltas
  is simply wrong. It must be computed against the full occupancy model or
  reported as the estimate it is.
- records produced, drained, and dropped
- sites requested, instrumented, and rejected
- **whether instrumentation induced serialization**. Per invariant 5, a
  heavyweight tool may cause the runtime to admit dispatches one at a time, and
  a user comparing timings needs to know that happened (e.g., by tracking how
  many dispatches are in flight at once, which the framework can count because
  it already observes every submission, and comparing that against the
  uninstrumented dispatches of the same run). This is only in question when the
  framework grew a memory resource, so it costs nothing the rest of the time.

A user who sees "3.2× slower, 41% of records dropped, occupancy 8→5" can make an
informed decision. A user who sees a slow run and a clean report cannot.

**Representative small inputs are the expected usage** where users have that
option, the standard bargain for heavyweight dynamic analysis, and better stated
up front than treated as an apology.

It must not be the only answer, because for a large class of users it is not
available. High-performance-computing workloads often exhibit the behavior of
interest only at full scale; machine-learning workloads are profiled on a real
training step of a real model, with no smaller version that reproduces the bug.
For those users the workable substitute is **narrowing in dispatch space rather
than input space**: one kernel, a range of dispatches, or a window of execution,
at full scale. That comes free from dispatch-granular selection, which is why
that mechanism deserves prominence.

---

## 8. Runtime integration

### 8.1 How a tool gets into the process

The dynamic linker can preload a library before the process begins, which is how
rocjitsu's driver-level interposer gets in. Separately, the GPU runtime hands its
dispatch table to registered tool libraries during its initialisation. The
second is the natural home for instrumentation, since it operates in terms of
code objects and queues rather than driver calls. Note that runtime
initialisation is triggered by the application, so a tool library is loaded
before the application does any GPU work, not before it starts running.

**Open: which registration path.** The runtime offers two, an older one keyed
off an environment variable and a newer one in which the runtime hands its table
to a separately registered tool library. They are not complementary: the runtime
attempts the newer path first and, when it succeeds, does not consult the older
one at all. A tool using the older mechanism therefore only runs when the newer
is disabled, and disabling it is what removes every other profiling tool's
access to the runtime, which is why the two cannot currently coexist. This is
structural, not a bug in any one tool's setup. The framework must choose the
newer path to be a well-behaved co-resident, and that choice has not been made.

Interception is installed by modifying entries in the runtime's dispatch table.
Two interception points matter:

- **Code object load**, before the loader places the image. Where instrumentation
  happens.
- **Queue creation and packet submission**, so the framework can observe and edit
  dispatch packets. Where variant selection, sampling, and buffer binding happen.

**What is and is not safe at tool-load time.** Agents and their memory pools
already exist by then, so enumerating them is not the hazard. The real ones are
that re-entering runtime initialisation from inside the tool's entry point
deadlocks, and that table entries take effect the moment they are written,
including for the runtime's own internal calls, before the tool has finished
setting itself up. Deferring agent-scoped work to a later interception is
defensible, but as a sequencing choice rather than because agents are unavailable.

**Multiple hooks must compose.** A user instrumenting a workload that also
requires translation needs both at once, and a mechanism in which tools displace
one another silently is not acceptable. This applies equally to tools this project
does not own: an existing profiler in the same process wants the same dispatch
table, the same queue interception, and the same code-object-load notification.

**Chaining is mandatory.** A tool receiving a dispatch table must retain the
previous entries, call them, and propagate their results. The table is the live
one and tools mutate it in place, so capturing it and dropping what was there is
a defect, not a strategy: it silently disables whatever was installed first. The
framework must also tolerate being handed a table more than once; refusing a
second registration is as wrong as capturing.

**Not every hook is chainable, and those that are not must be named.** Some
runtime facilities accept exactly one registrant and reject the second, such as a
system-event handler or a queue-creation notifier. For those, composition is
unavailable and first writer wins, so the framework must decide whether to claim
them at all, and fail visibly rather than silently when it cannot. "Hooks
compose" is a rule about the dispatch table, not a property of every extension
point the runtime offers.

**The framework must not disable facilities other tools depend on.** Turning off a
runtime mechanism to simplify one's own path is expedient, and where another tool
relies on that mechanism to function at all, it silently removes that tool's
entire capability. Any such measure must be conditional on the user not wanting
the other tool, and must be visible.

**An obligation to external tools.** Instrumentation rewrites the code object
before it is loaded, so what other tools observe is the instrumented image. Every
offset they report, whether a sampled program counter or a trace address, refers
to instrumented code rather than the image the user built. Worse, resource figures
read from the loaded image are the instrumented register and memory footprints,
so any occupancy figure derived from them is wrong.

Two requirements follow. They are requirements rather than courtesies, because
without them enabling instrumentation silently corrupts another tool's output:

- **The instrumented-to-original mapping must be published**, in a form usable by
  a tool that observes the instrumented image but never consumes this framework's
  records. The bookkeeping already exists (§6.2); the delivery mechanism is open.
- **The original resource footprint must remain available** alongside the
  instrumented one, so consumers can report what the application would have used.

**On joining data with another tool.** Identifiers another tool mints, being its
dispatch, queue, agent, and code-object identities, cannot be reconstructed from
outside it and must be received from it. The only identifier an outside component
can reliably contribute is one it supplies to that tool in advance. A framework
wanting its output correlated must therefore participate in that tool's
correlation scheme rather than inventing a parallel one.

**Live integration into another tool's data model may be out of reach.** Feeding
records directly into another tool's buffers requires that tool to expose an
ingestion path and an extensible record taxonomy. Where it does not, the route is
to emit alongside and join afterwards, a limitation to state plainly rather than
design around, and lifting it is a request to the other project.

#### Publishing the mapping (OPEN)

The requirement is settled and forced from outside; the delivery mechanism is
not. Candidates:

- **Carried in the records themselves.** Each record already names its original
  site, so anything consuming this framework's output is self-mapping. This
  covers our consumers completely and other tools not at all, which is precisely
  the shape of the problem.
- **A side-car artifact** per instrumented code object. Straightforward and
  universally consumable, at the cost of a file nobody asked for, a lifetime to
  manage, and a convention every consumer must learn.
- **Debug information in the instrumented image.** Consumers already know how to
  read it and no new convention is needed; awkward because the images most in
  need of mapping are frequently the ones with no debug information to extend.
- **Through the observing tool's own notification path**, if it grows a field for
  a rewritten image. Cleanest for the consumer, entirely outside this project's
  control, so it cannot be the only plan.

The choice is unusually consumer-driven and should be made with whoever must read
it: a mechanism no consumer adopts satisfies the requirement on paper and not in
fact.

### 8.2 User experience

The everyday user's invocation is the existing launcher with a configuration
file, running an unmodified application. DBI is a configuration section and a
hook, not a new tool, and there is no separate instrument-then-run step.

**Attaching to a running application.** The committed meaning is **activating a
hook already present in the process**, not injecting into a process that never
had one. Injection is out of scope; the ecosystem already provides a registration
path that makes the dormant-hook form workable.

What attach reaches is asymmetric, and the asymmetry is accepted rather than
solved:

- **Code objects are reachable retroactively.** Instrumentation happens at load,
  so by the time a tool attaches the interesting code objects are loaded, but
  nothing must be modified in place. An instrumented variant of an already-loaded
  code object can be built and then named in subsequent dispatches. This is §2.1
  doing work it was not specifically designed for.
- **Queues are reachable only prospectively.** Submission interception is
  established when a queue is created. The obstacle is not merely that an
  existing queue was missed: there is no way to enumerate the queues a process
  already owns, so a late-arriving tool cannot find them even in principle. It
  sees the doorbell traffic of queues it never observed being created and cannot
  attribute it. Dispatches on those queues cannot be redirected.

**This limits attach to applications that create queues during their run.** An
application creating its queues once at startup and running for hours is
precisely the case that motivates attaching and precisely the case attach reaches
nothing. That is a real gap, accepted knowingly; a submission interception point
not requiring ownership of queue creation is not currently known to exist.

Note that existing tooling in this project uses "attach" for something else
entirely: launching a new process against an already-running service. The
collision will mislead people, and the user-facing name for this capability
should not repeat it.

**Output format is a tool decision, not a framework constraint.** The everyday
user's default is a human-readable report in a file, because that is what an
everyday user wants. A tool whose consumers are other tools may emit
machine-readable output instead, and for anything sitting alongside existing
profiling data, that output needs identifiers joining to the rest of the
ecosystem's, since data that cannot be correlated to a kernel trace is stranded.
Raw record spooling is a debugging mode: at scale it is terabytes.

### 8.3 Failure at instrumentation time

**Best-effort, with required sites.**

- Attempt everything the user asked for; where a site cannot be instrumented,
  warn and continue.
- **Collect, then decide.** Accumulate all outcomes before evaluating success. A
  user with ten problems should see ten, not one per run.
- A tool may mark instrumentation as **required**, initialisation and
  finalisation being the obvious cases, where partial application is meaningless.
  If required instrumentation fails, instrumentation fails.
- If instrumentation fails, the application exits with a clear diagnostic.

The failure mode to avoid above all others is **silently running uninstrumented
and producing a clean report**, which a user cannot distinguish from a clean run.

**This policy does not vary with when the failure is discovered.** Code objects
load throughout a run, including device libraries, runtime-compiled modules, and
lazily loaded ones, so failures are not a startup-only phenomenon, and there is
no meaningful line between "before execution begins" and "during it". A required
instrumentation point that cannot be satisfied means the tool cannot do the job
it was asked to do, as true at the three-hour mark as at the first dispatch.
Continuing would only produce output the user cannot rely on.

The counterpart obligation is that **the user is told how much was actually
instrumented**: sites requested, instrumented, and rejected, and which code
objects were covered. Best-effort coverage is only defensible if its extent is
visible.

### 8.4 Failure at run time

**Host process death**, meaning the application crashes or is killed. End-of-run
reporting never happens and in-process analysis state is lost. An application
under a sanitizer is disproportionately likely to crash, so this is the common
case. Mitigations, in increasing order of cost:

- **Incremental output**, which streaming analysis wants anyway.
- **Signal handling** to flush, subject to async-signal-safety constraints.
- **An out-of-process drainer**, with the buffer in shared memory, so the
  application's death takes neither the data nor the analysis with it. The
  durable answer, and rocjitsu already has comparable multi-process
  memory-sharing machinery.

**Device faults.** The runtime surfaces queue errors, and a memory fault is
reported with the faulting agent, the faulting virtual address, and flags
describing the reason, including whether the report is imprecise. What is not
reported is a program counter, so the fault alone does not identify the
instruction. Absent a handler the default behavior is a diagnostic message and
an abort from a runtime thread. The hardware trap mechanism used by the GPU
debugger could in principle expose a faulting wave's state, and the runtime has a
device core-dump path worth leaning on; both are must-remain-possible, not
committed.

Two capabilities are worth deliberate design:

- **The last records before a fault are diagnostic.** An address stream plus a
  fault is a far better crash report than the fault alone, exactly the
  attribution that motivates the project, and the more valuable because the
  fault carries no program counter of its own. It is genuinely best-effort
  rather than nearly free: the faulting wave's in-flight stores and unflushed
  caches are lost, so what survives is what had already reached host memory.
- **Distinguishing an application fault from a probe fault.** Instrumentation bugs
  manifest as device faults, and without a way to tell them apart the framework
  will be blamed for application bugs and vice versa. Probe writes target known
  address ranges, which makes this tractable.

Both run into host-process death: a device fault usually wedges the queue and the
process is terminated. Same answer, incremental output or out-of-process.

---

## 9. Non-goals

Deliberate scope decisions, not missing features.

- **No on-device aggregation or reduction.** The device streams; the host
  reduces. A consequence of §4, and a statement about efficiency rather than
  capability: a basic block counter is implementable, it simply emits a record
  per execution instead of accumulating. Such tools work; they cost more here
  than they would elsewhere.
- **No arbitrary user-authored probe logic.** No compiler, no code generation from
  source. Probes ship with rocjitsu. Hand-written assembly probes are a
  possibility, not a commitment.
- **No general-purpose instrumentation framework.** Luthier and Dyninst are
  building those. This is the constrained, streaming option.
- **No dependencies outside ROCm**, and no third-party instrumentation library.
  The one prerequisite worth naming is the device compiler needed to build
  probes (§6.7). It is part of ROCm rather than external, but it is packaged
  separately from the runtime and it is needed only at build time, so a machine
  that only runs instrumented applications does not need it.
- **AMDGPU targets only.**
- **No completeness guarantee.** Sampling and dropping are architectural.
- **No binary-stable ABI** for probes or client tools.
- **Not always-on.** An analysis tool for representative inputs, not production
  telemetry.
- **Not a debugger, and not a replacement for hardware counters.**

---

## 10. Simulator and hardware

**Both are targets. Neither is a fallback for the other.**

Hardware sets the rules. Every constraint in this document, including the memory
model, the probe restrictions, occupancy cost, and cache coherence, exists
because of silicon. The simulator is the low-friction validation path: everything
is observable and instrumentation is comparatively easy.

Because the instrumentation step is a pure code-object transform it is identical
for both. The runtime integration around it is not: buffer placement,
dispatch-packet rewriting, and drain scheduling differ between an environment the
framework controls entirely and one where it must negotiate with a real runtime
and real hardware. The shared part is nonetheless large enough to give a
differential-testing story, in that the same instrumentation on both targets
should produce consistent results, and divergence is a bug in one of them.

---

## 11. Open questions and known gaps

### Open questions

Each is discussed in context above. Everything else in this document is either
committed or explicitly deferred; these five are the decisions still to make.

1. **Buffer scope** (§5.4): per code object, per queue, or per dispatch. The
   delivery mechanism leans to the kernel argument segment; what remains open is
   the scope it is asked to express. Decidable within the project.
2. **Relocation strategy for large kernels** (§6.2): how sites beyond
   relative-branch reach are handled. Expected to be the common case, not a tail
   case, for the workloads this framework targets. The candidates and their
   trade-offs are written up; what is wanted is a choice among them.
3. **How the instrumented-to-original mapping is published** (§8.1): the
   requirement is settled, the delivery mechanism is not. Consumer-driven, so it
   wants the tools that must read the mapping represented in the decision.
4. **Advanced-user API status** (§1): supported product surface, or internal API
   that happens to be usable. This is a question about who the framework is for,
   and answering it may need someone who owns that relationship.
5. **Which runtime registration path the framework uses** (§8.1): the older
   environment-keyed mechanism it uses today, or the newer one. The two are
   mutually exclusive in practice, and only the newer permits coexistence with
   other profiling tools.

### Known gaps

Subjects a reader may reasonably expect that this document does not address.
Listed so their absence is visibly deliberate.

- **Scale-out.** Multiple agents, multiple queues, multiple processes. Buffers
  are agent-scoped; nothing here says who drains what when a node has eight
  devices and eight ranks.
- **Code objects that are not a single static image**, meaning runtime-compiled
  and runtime-loaded modules, bundles, and images already instrumented.
- **Kernel entry and exit instrumentation**, which §8.3's required initialisation
  and finalisation sites presuppose and §5.4's prologue depends on.
- **Which memory regions are in scope** for instrumentation.
- **Validation strategy** beyond the differential testing of §10.
- **Determinism and reproducibility** of a sampled run.
- **A security and threat model.** A framework that rewrites code before it is
  loaded and shares memory with a host process has one; nothing here states what
  it assumes about the code it instruments or the process it runs in.
- **Buffer sizing guidance.** How large a buffer should be for a given workload,
  and how a user is meant to arrive at that number, which matters given that
  overflow policy is a first-class design element.
