# ConSan Production Design

**RocJitsu** is this repository's GPU binary-analysis, simulation, and rewriting
project. **ConSan** is its concurrency sanitizer for AMD GPU memory shared by a
workgroup. ConSan inspects and rewrites final native GPU machine code before the
ROCm runtime loads it.

This file is the output of **Stage 1**, the design-only stage defined in
[PLAN.md](PLAN.md). It is ready for review. **Stage 2** means the subsequent
implementation stage; this file does not authorize Stage 2 by itself. The
**destination** is the proposed long-lived internal structure. The
**implementation-week endpoint** is the smaller part of that destination that
can realistically be established in approximately one week. The
**incremental route** is the ordered set of independently working changes that
gets there without replacing the whole system at once.

Here a **target** means one AMD `gfx` architecture/ABI variant for which
RocJitsu must analyze or generate machine code. The required targets are CDNA3
(`gfx942`), CDNA4 (`gfx950`), CDNA5 (`gfx1250`), RDNA3 (`gfx1100`), and RDNA4
(`gfx1201`). **CDNA** and **RDNA** name AMD GPU architecture families; the
numbers distinguish generations. The four retained ConSan execution paths are
defined below before their internal design is discussed.

## Reading model and terminology

This section defines GPU-execution and ConSan-specific terms used in the rest
of the document. Ordinary C++, compiler, testing, and software-engineering
terms keep their usual meanings.

### Program and GPU execution terms

- **Host-side** work runs in the CPU process; **device-side** work runs on the
  GPU. A **kernel** is a host-launchable GPU entry point. A **dispatch** is one
  launch of a kernel; it contains workgroups whose threads can cooperate
  through local memory.
- A **wave** is the hardware group of lanes that execute one instruction
  together. A **lane** is one participating GPU thread within that wave. The
  **execution mask**, represented by AMD's `EXEC` register, says which lanes
  are active for the current instruction.
- A gfx1250 **workgroup cluster** is a hardware-visible group of workgroups that
  can use cluster-scoped facilities. Its cluster coordinate is distinct from
  the x/y/z coordinate of a workgroup inside an ordinary dispatch.
- A **code object** is the AMD ELF object containing native GPU instructions,
  kernel descriptors, symbols, and resource metadata. The **guest** is the
  application code being instrumented; **tool code** is code ConSan inserts.
- A **kernarg region** is dispatch-specific memory holding a kernel's explicit
  arguments and runtime-provided implicit arguments. Copying or extending it
  requires preserving the ABI layout and keeping the copy alive through the
  dispatch. The source sometimes shortens this term to **kernarg**.
- The current ConSan **hook** is an HSA tool library loaded through
  `HSA_TOOLS_LIB`. It intercepts code-object and dispatch operations so ConSan
  can install rewritten code, bind reports, and manage their lifetimes. An HSA
  **agent** is the runtime object representing one GPU.
- **DBI** means dynamic binary instrumentation: inspecting an already compiled
  binary and adding observations without recompiling the application's source.

The AMD register and memory terms used later have these meanings:

- **SGPR** means scalar general-purpose register, whose value is shared by a
  wave. **VGPR** means vector general-purpose register, which has a value per
  lane. **AccVGPR** is accumulator register storage used by matrix operations;
  some targets allocate it separately and others unify it with VGPR storage.
- `VCC` is AMD's vector condition register. A **HWREG** is a named hardware
  state register accessed by special instructions; `HW_ID` and `HW_ID1` expose
  hardware execution identity on applicable targets. **TTMP** registers are
  privileged temporary scalar registers that some target paths use under
  constrained conditions.
- **LDS** (local data share) is low-latency memory shared by the waves in a
  workgroup. HSA metadata calls its allocation the **group segment**. An
  **address space** is a category of memory with distinct addressing and
  sharing rules, such as global, LDS, or private memory. **FLAT** instructions
  use a generic address space that can represent multiple categories; a
  **group-FLAT** access is a FLAT instruction that analysis has established may
  or must address LDS.
- **Coherence** is the memory-system guarantee governing when host and device
  observers agree that a write is visible. ConSan report storage needs the
  coherence and atomic properties promised by its publication protocol.
- **Private** or **scratch** memory is storage private to a lane. A
  **dynamic-stack** kernel computes some of that storage dynamically instead
  of using only a fixed descriptor size. `VSCRATCH` is target state used to
  address scratch memory.
- A wave is **resident** while it occupies GPU execution resources and may run.
  **Occupancy** is the number of waves that can be resident concurrently given
  the registers, LDS, scratch, and other resources a kernel requires.
- A **wait counter** tracks outstanding asynchronous memory operations. A wait
  instruction blocks until a selected counter condition holds. Instrumentation
  must not accidentally wait for guest operations earlier than the guest did,
  nor save a register before an outstanding guest load has finished writing it.
- `M0`, VGPR **bank mode**, preload registers, cache controls, and branch reach
  are target mechanics: respectively special scalar state, the target's VGPR
  indexing mode, values supplied at kernel entry, memory-cache behavior, and
  the maximum distance an encoded branch can reach. They are not sanitizer
  semantics.

### Concurrency and ConSan terms

**Evidence** is information ConSan retains to support or qualify a sanitizer
conclusion. A **report** is the bounded memory layout through which device code
and host code transport or retain that evidence.

- An **owner** is the execution identity to which an access is attributed, such
  as a lane or wave within a dispatch and workgroup. A **scope** states which
  owners a synchronization operation can order.
- **Synchronization** is behavior that orders memory operations. A **barrier**
  coordinates a participating group. **Release** ordering publishes earlier
  work; **acquire** ordering makes suitably published work visible to later
  operations. An **atomic** performs an indivisible memory operation and may
  carry acquire or release ordering. A **fence** orders memory operations
  without itself transferring the ordinary data.
- A **happens-before edge** is a supported proof that one relevant operation is
  ordered before another. An **epoch** is a compact logical generation used by
  ConSan to represent progress through supported synchronization. It is not a
  wall-clock timestamp.
- A ConSan **concurrency model** is the explicit set of owners, access
  conflicts, and synchronization rules that an engine recognizes. It is a
  deliberately bounded sanitizer contract, not a claim to model every possible
  GPU ordering mechanism.
- A **memory access** is a load, store, or atomic operation over a byte range.
  Two accesses **conflict** when their byte ranges overlap, at least one writes,
  and they belong to different relevant owners without a supported
  happens-before edge. A detected conflict is evidence of a possible race
  under the selected ConSan model; the exact claim depends on the engine.
- **Memory provenance** is the analysis explaining which address space or
  object a computed pointer can refer to. **Confidence** records whether that
  conclusion is exact, conservative, ambiguous, or unsupported; downstream
  code may weaken a claim but may not silently strengthen it.
- A **site** is an original instruction location considered for observation.
  A **physical site** is the one instruction offset at which bytes can be
  inserted. A **semantic site** is one meaning associated with that
  instruction, such as one of two address ranges or one member of a logical
  synchronization sequence. Several semantic sites can therefore share one
  physical site. A **two-address instruction** performs one native operation
  over two independently meaningful byte ranges.
- **Static** means a fact or site known from the binary before execution;
  **dynamic** means one fact or event from a particular runtime execution. A
  site is **applicable** when its semantic domain belongs to the selected
  engine's contract. It is **admitted** when policy requests instrumentation
  for it after capability/confidence checks.
- **Instrumentation** is the inserted behavior that observes a site. A
  **probe** is one inserted observation sequence. **Before** and **after** say
  whether a probe executes immediately before or immediately after the guest
  instruction. An after-probe is required when the observation needs the
  guest instruction's result, such as whether a compare-and-swap atomic
  succeeded. Compare-and-swap writes a new value only when the old value equals
  an expected value.
- **Decoding** converts instruction bytes into an instruction representation.
  **Classification** converts that target-specific instruction into a
  target-independent meaning such as “32-bit LDS store” or “release atomic.”
  **Normalization** expresses different target spellings that have the same
  sanitizer meaning in one shared form; the resulting fact is **normalized**.
  An **inventory** is the immutable collection of those program facts before
  any engine chooses what to instrument.
- **Policy** chooses intended behavior from semantic facts and user settings.
  **Mechanism** is how that already chosen behavior is implemented. For
  example, “observe this release atomic” is policy; choosing registers and
  emitting target instructions is mechanism.
- **Lowering** converts a target-independent probe request into target-native
  instructions and resource effects. A **resource** is limited machine capacity
  such as registers, LDS, scratch, report memory, or code bytes. An **effect**
  states what a probe reads, writes, clobbers, waits for, or allocates. A
  **clobber** is machine state that a probe overwrites; it must be restored when
  the guest can observe it.
  **Emission** writes those instructions.
  **Placement** chooses legal locations for them and connects branches and
  relocated guest instructions. A **patch** is a concrete change to the code
  object. A **cave** is unused or newly provided instruction space; a
  **trampoline** is a branch sequence that reaches instrumented code; a
  **relay** is an intermediate branch used when the final destination is out of
  direct reach.
- A transformation is **transactional** when it either produces one complete,
  validated replacement or leaves the original unchanged. It never exposes an
  image containing only some of the planned patches.
- ConSan evidence includes access records, synchronization records, sampled
  windows, shadow state, mismatch markers, and diagnostics. A **snapshot** is a
  host-owned copy of report state taken at a defined lifecycle point. A
  **schema** defines the
  named fields, types, versions, and meaning of such data independently of one
  concrete instance. An **identifier** or **ID** is a value used to name and
  associate entities without treating their storage address as identity.
  **Attribution** is the association from evidence back to its original site
  and runtime owners. **Publication** is the
  protocol by which the GPU makes a complete record visible; a **commit** field
  or operation distinguishes complete data from a partially written record.
  **Malformed** evidence violates its schema or publication protocol. **Stale**
  evidence belongs to an older generation or lifetime. **Torn** evidence was
  observed with only part of a multi-field publication visible.
- **Capacity** is the finite amount of state reserved for a kind of evidence.
  **Saturation** or **overflow** means execution needed more than that capacity.
  **Loss** means relevant evidence was not retained; a record is **dropped**
  when it is one particular item that could not be retained. These conditions
  affect what a run is allowed to claim.
- **Coverage** records which relevant static sites were applicable, admitted,
  successfully instrumented, selected at runtime, and represented by valid
  evidence. **Completeness** states whether the declared observation contract
  completed without a coverage or evidence gap. **Trust evaluation** combines
  coverage, loss, malformed-data, and analysis facts. A **verdict** is its typed
  conclusion, such as conflict observed, no conflict in the retained bounded
  evidence, incomplete, unsupported, or failed. A **diagnostic** is the
  user-facing explanation rendered from that conclusion.
- A **shadow** is metadata that summarizes prior observed accesses. An
  **exact-shadow** algorithm directly updates this metadata for every admitted
  dynamic access within its declared capacity. “Exact” does not mean that
  unsupported sites or capacity loss disappear from the trust calculation.
- **Sampling** selects only a declared subset of dynamic executions. A sampled
  conflict is useful evidence; absence of a conflict is statistically
  inconclusive. A **selector** is the rule that chooses that subset. A
  **sticky marker** is a device-visible flag that, once set by a mismatch,
  remains set until the host collects it.

ConSan exposes two algorithm families, historically called **flavors**. The
Memory-Ordering Instrumentation flavor, abbreviated **MOI**, has three
**engines**, meaning three ways to retain and interpret the same broad memory-
ordering vocabulary:

- **Record/Replay** records bounded device events and reconstructs supported
  ordering and conflicts on the host.
- **Sampled** retains selected causal windows and scans them on the host.
- **Inline Shadow** updates shadow and ordering state and makes supported-form
  conflict decisions on the device.

The **SuperCollider** flavor delays and repeats a supported load, or reads a
store back, then sets a sticky marker if the value changed. It detects value
instability and deliberately does not claim to reconstruct happens-before.
This document sometimes calls all four choices **modes** or **execution paths**
when their historical flavor-versus-engine distinction is irrelevant.
Later dense tables may abbreviate Record/Replay as **RR** and SuperCollider as
**SC**; prose uses the full names where space permits.

**Ordinary instrumentation** means one of those four sanitizer modes observing
the application without intentionally changing its result. **Fault injection**
intentionally removes or corrupts a selected correctness mechanism so a
validation test can demonstrate detector sensitivity. A **perturbation**
intentionally changes timing or ordering to make a latent failure more likely
to appear. Fault injection and perturbation are explicitly requested
validation operations, not ordinary instrumentation.

The current prototype passes configuration and work-in-progress state through
two especially large structures. `ConSanOptions` begins as requested options
but is also populated with selected registers, concrete report addresses, and
other derived choices. `ConSanResult` is both the accumulating work state and
the final result, containing semantic facts, patches, resources, warnings,
runtime layout, and status. Calling them **mutable** means that many phases edit
the same instances. The proposed named types—`ProgramInventory`,
`ObservationPlan`, `EvidenceRequirements`, `ResourcePlan`, `TransformResult`,
and `RunVerdict`—separate those responsibilities; their exact contracts are
defined in section 5.2 before the design relies on their internal fields. For
example, policy consumes an immutable `ProgramInventory` and returns a new
`ObservationPlan`; it cannot delete an awkward inventory site, and lowering
must return a typed rejection rather than silently changing policy's decision.

The document uses the following provisional C++ type names. They are listed
here so a name is meaningful on first use; section 5.2 gives the full contract.

| Name | Local meaning |
| --- | --- |
| `ConSanRequest` | Supported user choices about which ConSan behavior to request; it contains no selected registers, patch bytes, or device addresses. |
| `TransformPolicy` | Limits and policy for changing a code object, such as allowed image growth and placement work. |
| `RuntimePolicy` | Policy for activation, installation failure, and process-level runtime lifetimes or capacity. |
| `ConSanDebugOverrides` | Focused-test and early-development controls that are intentionally not part of the supported product request. |
| `MutationRequest` | A separate request to corrupt or perturb a program for validation; ordinary sanitizer observation is not a mutation. |
| `TargetProfile` | Immutable ISA and ABI facts for the selected GPU target and kernel. |
| `RuntimeCapabilities` | Facts queried from the actual runtime backend and GPU agent, such as usable memory and atomic/coherence properties. |
| `ProgramInventory` | Immutable facts found in the original code object, including stable site identities and decoded access/synchronization meaning. |
| `SiteDecision` | Engine policy's typed decision that a semantic site is not applicable, unsupported for a reason, or admitted. |
| `ProbeIntent` | A target-neutral description of one observation that lowering should implement at a physical site. |
| `ObservationPlan` | The complete set of site decisions, admitted probe intents, required engine state, and evidence needs for one transformation. |
| `CoverageLedger` | The joined record of applicability, admission, lowering success, runtime selection, evidence loss, and analysis completeness. |
| `EvidenceRequirements` | Address-free description of the report schema, capacity, memory properties, scope, and loss rules an engine needs. |
| `BoundRuntimeResources` | Concrete allocations, device addresses, generations, scopes, and lifetimes that satisfy evidence requirements. |
| `ResourcePlan` | Immutable assignment of registers, spills, segment growth, and other resources to admitted probe intents. |
| `LoweringOutcome` | Per-site result saying that lowering succeeded or failed for a typed resource or placement reason. |
| `TransformResult` | Static result of attempting the code-object transformation, including validated bytes, mapping, coverage, and typed status. |
| `RunVerdict` | Conclusion formed after runtime evidence is decoded and analyzed. |
| `TargetClassifier` | Narrow target-dependent operation that maps decoded native instructions to shared semantic forms. |
| `TargetLowerer` | Narrow target-dependent operation that maps probe intents and assigned resources to native instructions. |
| `ConSanOptions` | Temporary mutable compatibility-lowering state. Its production constructor preserves each typed input as a base subobject and derives only lowerer-internal controls; default construction exists only for focused mechanism tests. |
| Compatibility lowering operations | Explicit temporary boundary containing current probe emission, resource, and placement machinery after semantic policy moves out of it. The former wrapper class has been deleted. |

These names describe responsibilities, not a requirement that each become a
class. Most should be plain structures, enum variants, and pure functions.

## 1. Decisions in brief

1. ConSan's production core is a sequence of immutable, typed stage contracts,
   not the present mutable `ConSanOptions`/`ConSanResult` state machine and not
   four target-by-engine implementations.
2. A stable semantic inventory is built once. It distinguishes one physical
   instruction site from the access ranges or synchronization events that the
   instruction represents. Engine policy consumes that inventory but cannot
   modify it.
3. Engine policy produces a target-neutral observation plan and a coverage
   ledger. Lowering reports outcomes back to that ledger; it does not decide
   after the fact which semantic sites mattered.
4. Target-independent sanitizer meaning is shared across all five targets.
   Target differences end in `TargetProfile` data or in narrow instruction
   classification and lowering adapters. A target name in engine policy is a
   design error.
5. Report sizing, runtime buffer requirements, concrete buffer binding, device
   evidence decoding, host analysis, and user-facing verdicts are separate
   components. A report address is never a planning option.
6. SuperCollider is not forced into the MOI ordering abstraction. The three MOI
   engines share semantic access and ordering vocabulary, but deliberately
   differ in what evidence they retain and where the conflict decision occurs.
7. Fault injection and perturbation are validation tools. They may consume the
   same semantic inventory, but they are not sanitizer engines and do not shape
   the production observation pipeline.
8. The one-week implementation endpoint is a production control-plane spine,
   meaning a working sequence of host-side decision stages:
   immutable request, target and runtime capability facts, semantic inventory,
   engine/site plan, coverage ledger, evidence requirements, explicit pipeline
   results, and a narrow legacy-lowering adapter. The roughly 92,000 lines of
   current instrumentation and HSA-hook machinery cannot credibly be rewritten
   in one week; that work is not hidden inside this milestone.
9. Migration is component-by-component. Each slice first establishes a
   contract, adapts the current implementation behind it, switches named
   consumers, runs proportional tests, and deletes the displaced path. There is
   no parallel rewrite and no final mode-wide activation switch.
10. The design leans toward DBI's intended client/framework boundary. A
    **client/framework boundary** separates ConSan-owned sanitizer policy from
    generally reusable DBI mechanics. DBI's near-term **host-streaming model**
    has device probes publish records and performs complex processing on the
    host; it cannot yet express every stateful on-device mode. Record/Replay can
    meet DBI first without making the other modes block it or lose their
    requirements.

## 2. Scope and evidence

### 2.1 Scope of this design

This document has two horizons:

- **Internal destination:** the components and dependency direction ConSan
  should retain while its current lowering remains in RocJitsu.
- **Implementation-week endpoint:** the subset that can be established through
  small cutovers in approximately one week, before the DBI design discussion.

The endpoint is intentionally not described as a completed DBI migration. The
current `HSA_TOOLS_LIB` hook remains active, current device probe bodies remain
behind an adapter, and the current report ABI remains valid until separately
migrated. Here the **report ABI** is the exact byte layout and publication rules
that let inserted GPU probes and the host report reader interpret the same
memory. The endpoint is valuable because it makes sanitizer semantics,
requirements, coverage, and framework mechanics separately explainable and
testable.

### 2.2 Evidence inspected

A **prologue** below means tool code inserted at kernel entry. A
**status ledger** is one of the target E2E-status tables. A **semantic oracle**
is the expected ConSan behavior used to decide whether a test passed—for
example, exact program output plus the presence or absence of a particular
diagnostic.

The design was derived from the current source rather than from filenames:

- the top-level composition and validation path in `consan.cpp` and
  `consan_composition.inc`;
- the complete MOI path rooted at the current C++ function
  `try_patch_consan_moi`, including access,
  prologue, barrier, atomic, fence, report, resource, placement, and model code;
- the independent SuperCollider LDS and group-FLAT implementation;
- the HSA hook's configuration, transform admission, report allocation,
  replacement lifetime, executable binding, dispatch segment growth, teardown,
  replay, sampled scan, and coverage reporting;
- `DESIGN.md`, which documents current behavior; `FLAVORS.md`, which compares
  the four execution paths; `CAPABILITIES.md`, which records the normative
  target/engine/form matrix; `VALIDATION.md`, which defines E2E procedures; the
  five target status ledgers; and `PLAN_DEVICE_TESTS.md`, which records device-
  test goals and provenance;
- the host tests under `tests/patch/consan`, the HSA-hook tests, and the paired
  device suite under `tests/dbi/consan/device`;
- the checked-in current DBI design and Allyson Cauble-Chantrenne's
  forward-looking DBI overview in PR 10407, together with her recorded
   clarifications.

At this revision the ConSan patch directory contains about 81,500 lines and the
ConSan HSA-hook directory about 10,750 lines. Large responsibilities are
concentrated in 7,143 lines of MOI placement, 4,630 lines of validation, 4,382
lines of SuperCollider LDS lowering, 3,819 lines of synchronization analysis,
and several 2,000--3,800-line per-engine emission fragments. These measurements
are not a deletion target by themselves; they demonstrate why a one-week
flag-day rewrite would be neither reviewable nor credible.

The `consan-device` label identifies checked-in tests that execute instrumented
device code. The Slice 0 baseline records 3,501 such CTest rows: 2,908
simulator rows and 593 physical rows. The five target-labelled totals are 572
for gfx942, 1,185 for gfx950 including physical coverage, 538 for gfx1100, 548
for gfx1201, and 658 for gfx1250. Counts are evidence of breadth, not a coverage
metric; the behavioral pairing and semantic oracle matter more. The exact
contract inventory, structural measurements, and qualification result are in
[PRODUCTION_BASELINE.md](PRODUCTION_BASELINE.md).

## 3. Current implementation map

This section describes the prototype as it exists. The following local terms
make the implementation map precise:

- A **transform prefix** is work performed before the implementation branches
  into a particular ConSan mode. To **compose** transformations is to apply
  multiple individually described transformations in a controlled order and
  validate their combined result.
- `AMDGPU` is LLVM's ELF name for AMD GPU machine-code objects. A code-object
  **container** here is a text-bearing kernel or function region decoded as a
  unit.
- A **candidate** is an inventory item not yet finally admitted. The retired
  prototype used a second **site-disposition** record to repeat whether a site
  was supported, unsupported, or not applicable. Production now records that
  answer only in typed policy decisions. To **canonicalize aliases** is to
  choose one physical-site representative when several analysis paths refer to
  the same instruction while preserving every equivalent source-container
  name.
- **Engine-effective options** are the current mutable option values after the
  selected engine's defaults and derived choices have been applied. A
  **patch kind** is the prototype enum category attached to one emitted edit. Current
  code sometimes infers semantic success by counting these mechanism categories.
  **Telemetry** is optional debugging/measurement data such as registers,
  offsets, and patch counts; it is not a behavioral result.
- A **resource plan** chooses registers, spills, descriptor growth, and other
  storage needed by probes. **Liveness** analysis determines whether an
  original program value may still be needed at a program point. A register is
  **dead** there when overwriting it cannot affect later guest behavior. A
  **spill** temporarily saves a register value to private memory and restores
  it later.
- An **inline substitution** fits replacement instructions at the original
  site. A **dispatcher** routes execution from many sites to their probe bodies.
  A **reservoir** is a region holding several relay or probe bodies.
  **Patch geometry** means the concrete offsets, sizes, caves, relays, and branches of
  the rewritten image.
- A **kernel-entry prologue** is instrumentation run once when each wave enters
  a kernel. **Persistent state** is probe state captured there and kept for use
  at later sites. **Transient state** exists only while one probe runs.
- A **dynamic instance** is one runtime execution of a static instruction site.
  A report **generation** distinguishes reuse of the same storage by different
  executable or dispatch lifetimes. A **watchpoint** is a sampled record of an
  address range and owner. A **causal window** is the bounded group of sampled
  accesses and synchronization facts analyzed together. A **causal snapshot**
  is a bounded view of ordering state retained with an access so a later
  device-side check can decide whether owners were ordered. A report **bank** is
  one independently indexed region of a bounded report.
- A bounded **identity table** maps compact report indices to dispatch, owner,
  or site identities. **Pending-acquire state** records that a sampled acquire
  is waiting to be associated with the matching publication/window before host
  analysis treats it as ordering evidence.
- **First-light evidence** means that a bounded slot preserves the first
  qualifying observation for a static identity rather than an exhaustive event
  history. A **token** is a compact identifier representing a recorded ordering
  fact.
- A **code-object reader** is the HSA callback object from which the runtime
  reads bytes. An **executable** is the runtime-loaded object created from one
  or more code objects. A **dispatch packet** is the command structure submitted
  to a GPU queue to launch a kernel. **Teardown** is the cleanup performed when
  an executable or the tool is destroyed.
- **Fine-grained memory** is host-allocated memory that the GPU can access with
  the coherence and atomic properties required by the report protocol.
  A **late-bound** value, such as a report address, is supplied after semantic
  planning rather than baked into the original request.
- **Fail-open** runtime policy loads or preserves the original program when
  instrumentation cannot be installed. **Fail-closed** policy refuses the
  operation instead. An **interposer** is hook code placed between the
  application and a runtime API. A **registry** tracks related runtime objects
  and their lifetimes.
- To **admit a transform** is to allow rewriting under current per-object and
  process resource ceilings. A hook **test guard** is a test-mode assertion
  about expected transformation/evidence; it must not become production
  semantic policy.
- **Pristine** means the unmodified input image. A **transformational** operation
  is allowed to change program semantics; ordinary observation is not. A
  **manifestation** is the incorrect value, diagnostic, hang, or other behavior
  that an intentional mutation makes visible.
- An **automatic report** is one whose requirements are derived by ConSan and
  whose storage is allocated by the runtime hook, rather than one whose address
  and layout the user supplies explicitly.
- A lowering **retry** repeats lowering after a required late-bound resource,
  such as an automatically allocated report address, becomes available. It
  must use the same semantic inventory and plan as the earlier sizing pass.

### 3.1 Common transform prefix

The current C++ function `try_patch_consan_impl` is the top-level ordinary/
mutation transformation implementation. It currently owns all of the following:

1. recognize mutation and perturbation combinations and recursively compose
   pristine planning, mutation, instrumentation, and final validation;
2. parse and validate the AMDGPU ELF, copy kernel/function metadata into the
   result, identify the target, and create the decoder;
3. decode every container, build memory-site statistics, relay group-FLAT
   pointer provenance across calls, reattribute pre-applied code, and prune
   unreachable inferred ranges;
4. when requested, build fault sites, synchronization events, synchronization
   sequences, barrier lifecycles, execution owners, ordinary release/acquire
   associations, fence candidates, perturbation candidates, and fault
   annotations;
5. dispatch either to the MOI pipeline or to SuperCollider lowering; and
6. apply mutations, compose patch stages, validate the final ELF, and encode the
   install outcome in the same result object.

The useful ordering and ownership analysis is shared work. The recursion,
mutable option refinement, patch bytes, runtime ABI fields, warnings, and
validation telemetry are not part of that semantic inventory.

### 3.2 SuperCollider path

The ordinary SuperCollider path inventories native LDS and group-FLAT access
sites, canonicalizes physical aliases, allocates coverage records, and then
uses separate large LDS and FLAT lowerers. A probe preserves the guest access,
delays, repeats a load or reads a store back, compares values, and writes a
sticky mismatch marker. Placement includes inline substitutions, local caves,
dense dispatchers, relay reservoirs, register selection, spill handling, and
transactional ELF replacement.

SuperCollider does not build a happens-before relation. Barrier and atomic
semantics enter only through the separate perturbation/fault-validation path.
A mismatch is evidence of value instability, not proof of a data race; a
same-value race may be invisible.

### 3.3 Shared MOI prefix

Record/Replay, Sampled, and Inline Shadow currently share a second pipeline:

1. derive engine-effective options and clear prior lowering state from the
   mutable result;
2. project MOI access candidates, including normalized byte ranges, from the
   access intents already admitted by policy and their matching inventory
   sites; emitters do not re-decode range geometry from pristine bytes;
3. attach gfx1250 VGPR-bank state where needed;
4. consume typed access and synchronization decisions and their probe intents;
5. repeatedly build resource plans while automatic owner, dispatch, EXEC-save,
   persistent state, private state, spill, and per-owner assignments are chosen;
6. invoke engine-specific access and synchronization lowerers;
7. add the common kernel-entry prologue as applicable;
8. publish every access and synchronization lowering outcome directly from
   resource/placement state into the typed coverage ledger; and
9. summarize registers, patches, coverage, warnings, report inventory, and
   transform outcome into `ConSanResult`.

`ConSanOptions` contains user policy, validation-only controls, concrete report
addresses, test overrides, internally selected registers, resource decisions,
and transient facts. `ConSanResult` contains the semantic inventory, resource
plans, concrete patch geometry and bytes, runtime ABI facts, coverage,
diagnostics, telemetry, and final outcome. Rebuilding plans while mutating both
objects makes phase ownership and stale-state invariants difficult to state.

### 3.4 Record/Replay

Device probes publish bounded access records plus admitted barrier, atomic, and
fence evidence. Automatic layouts use report-wide dispatch and access-identity
tables; capacity and bank saturation are explicit. Kernel entry captures the
dispatch/workgroup/owner state that guest code may later reuse. At teardown the
host snapshots committed records, repairs attribution where necessary, replays
them through the exact-shadow and ordering model, and produces attributed
conflicts or an incomplete verdict.

Here a **semantic reference** is the most directly inspectable implementation
of the shared MOI ordering and conflict rules, used as an oracle when comparing
other engines. Record/Replay is that reference; it is not an exhaustive trace.
The current automatic path uses the bounded first-light evidence defined at the
start of this section, so a clean replay is inconclusive if dynamic identities,
tables, or record capacities saturate.

### 3.5 Sampled

Device probes deterministically select dynamic instances and publish compact,
generation-qualified watchpoints, causal windows, synchronization metadata,
and pending-acquire state. Access, barrier, and atomic evidence must choose
compatible banks from the same dispatch/workgroup identity. The host validates
snapshots, joins pending synchronization, scans retained windows, and reports a
sampled conflict or an explicitly statistical/incomplete miss. An expert
immediate check exists, but is not the ordinary engine contract.

Sampling currently reduces retained evidence more than execution overhead: all
eligible sites still enter substantial common dispatch logic before an
unselected instance exits. That performance trait is not a semantic contract.

### 3.6 Inline Shadow

Device probes update versioned exact-shadow cells and bounded release,
acquired-token, and causal-snapshot tables, then decide supported-form
conflicts on the device. Barriers advance epoch state; selected atomics publish
or acquire ordering tokens. The host validates and summarizes already-produced
diagnostic records. Entry-captured persistent state and resident-wave ownership
are essential because the original launch inputs are ordinary guest state after
entry.

"Exact" means exact for admitted access and ordering forms within declared
capacity. Unsupported forms and dropped diagnostics remain outside the claim.

### 3.7 HSA runtime path

The current hook performs responsibilities that must become separate contracts:

- parse environment policy and preserve the existing rule that merely loading
  the tool activates default Record/Replay;
- admit transforms under per-object and process memory ceilings;
- intercept code-object readers and executable loads;
- run an inventory pass, plan and allocate a fine-grained automatic report,
  then retry MOI lowering with the concrete late-bound buffer;
- create and retain replacement reader storage until the executable lifetime
  makes release safe;
- join replacement symbols and kernel objects to required private/group segment
  growth and edit dispatch packets;
- bind report generations to executable lifetimes;
- on executable destruction or tool teardown, snapshot reports, replay or scan
  evidence, free storage, combine static and dynamic completeness, render
  diagnostics, and enforce test guards; and
- select fail-open, fail-closed, load-original, load-replacement, or reject
  behavior.

These are real product requirements. Their current co-location in a 5,600-line
interposer and a 2,000-line report registry is not the required design.

### 3.8 Validation-only mutation path

Fault injection discovers one reviewed semantic mutation against pristine code,
applies it to a separate image, inventories and instruments the mutated image,
and independently validates the composition. Barrier/atomic perturbations may
be composed with SuperCollider to expose a manifestation. This path is allowed
to be transformational; ordinary sanitizer instrumentation is not.

The production design treats mutation as a client of semantic inventory and a
consumer of the same transactional patch facilities. Mutation controls do not
belong in `ConSanRequest`, and engine code must not branch on them.

## 4. Behavioral and architectural invariants

An invariant is a rule that every valid implementation and migration state must
preserve. **Behavioral** invariants constrain what the application or user can
observe. **Instrumentation** invariants constrain the inserted probes and code-
object transformation. **Runtime/evidence** invariants constrain allocation,
publication, lifetime, and interpretation after loading. **Architectural**
invariants constrain where target differences may appear.

Terms used repeatedly below are:

- **Static** information is known from the code object before execution.
  **Dynamic** information arises from a particular runtime execution.
- **Transparency** means ordinary instrumentation preserves the guest program's
  result and all guest-visible machine state. A **clobber** is state a probe
  overwrites temporarily and must preserve when the guest can observe it.
- A memory operation is **relaxed** when it does not itself request a stronger
  ordering guarantee. Instrumentation must not strengthen such an operation in
  a way that changes which guest synchronization completes.
- A **fingerprint** is a content-derived identifier used to verify that two
  plans refer to the same input. A **mapping** relates an original instruction
  address to its location after transformation.
- A **scope** on a runtime object says whether it belongs to one code object,
  executable, dispatch, workgroup, or process. Its **lifetime** is the interval
  during which every consumer is allowed to use it.
- A **typed failure** is a named machine-readable reason, not a warning string.
  A renderer may turn it into prose, but control flow must use the typed value.
- An **in-flight load** has been issued but has not yet completed writing its
  destination register. To **drain** a counter is to wait until the relevant
  outstanding operations complete. A **zero-scratch** kernel has no usable
  scratch backing under its current descriptor/runtime state, so a spill cannot
  assume storage exists.
- A **raw address** is only a numeric device pointer, without an allocation
  handle, scope, generation, or lifetime proof. The x/y/z **workgroup tuple** is
  the complete three-dimensional coordinate. A **hash collision** occurs when
  different identities produce the same compact hash. **Idempotent teardown**
  means repeating cleanup has the same safe result and never frees an object
  twice.
- **Register granularity** is the target's allocation rounding unit. A
  **differential behavior** is an unexplained difference between simulator and
  physical execution of the same transformed-code contract.

### 4.1 Semantic invariants

1. **Stable identity.** A physical instruction site is identified by code-object
   identity plus original text offset. Logical ranges or events at the same
   instruction add a typed member identity; they do not pretend to be separate
   physical insertion points.
2. **One canonical physical site.** Aliases discovered through kernel/function
   ownership or multiple semantic interpretations are canonicalized before
   engine policy. One instruction cannot be patched twice by accident.
3. **Immutable evidence.** Inventory facts are never erased or rewritten to
   make an engine or lowerer succeed. Derived plans refer to inventory IDs.
4. **Confidence never strengthens silently.** A conservative, ambiguous, or
   unsupported decode cannot become exact merely because a downstream engine
   wants it. Every exclusion has a typed reason.
5. **Common vocabulary, distinct decisions.** MOI engines share access ranges,
   owners, dispatch/workgroup identity, epochs, barriers, atomics, fences, and
   scopes. Their evidence-retention and conflict-decision algorithms remain
   distinct. SuperCollider shares access facts but makes no ordering claim.
6. **Exact application output.** Normal instrumentation preserves the original
   program's results. Validation mutation is a separately requested,
   transformational operation.
7. **Honest cleanliness.** Unsupported sites, rejected lowering, unsampled
   dispatches, saturation, overflow, malformed evidence, or dropped ordering
   records cannot be summarized as a clean complete run.

### 4.2 Instrumentation invariants

1. The guest instruction executes exactly once with its original operands and
   execution mask, except for an explicitly selected validation mutation.
2. Probe placement is semantically before or after the guest instruction.
   Address capture normally uses before; an atomic return value or success mask
   requires after.
3. Every guest-visible register, condition, execution-mask, mode, and memory
   ordering state is preserved. Probe and spill memory operations cannot make a
   later relaxed guest wait complete too early.
4. Registers that may still receive an in-flight guest load are not saved as
   stale spill values. Required drains and waits are derived from explicit probe
   effects and target counter semantics.
5. Kernel-entry ABI values used after arbitrary guest code are copied to owned
   persistent state at entry. A later probe never rereads a launch register
   whose lifetime ended.
6. Resource growth is reflected consistently in the rewritten code object and
   in each dispatch packet that hardware executes. Dynamic stack and zero-
   scratch cases fail with a reason rather than relying on accidental state.
7. Planning is transactional. No consumer sees partial patch bytes, partial
   descriptor edits, or a report plan that corresponds to a different semantic
   inventory. The emitted ELF is reparsed and validated before installation.
8. Every requested semantic site ends in a typed static outcome: not
   applicable, unsupported, admitted then resource-rejected, admitted then
   placement-rejected, or instrumented. Warning strings are renderings, not
   the contract.

### 4.3 Runtime and evidence invariants

1. Report storage is host-allocated before device execution and is accessible
   with the memory and atomic/coherence properties required by the selected
   engine.
2. A concrete buffer binding has an explicit code-object/executable/dispatch
   scope, generation, size, layout, and lifetime. It is never inferred from a
   raw address alone.
3. A record has a commit protocol. The host cannot consume partially published
   fields, stale generations, or an entry from a different dispatch/workgroup
   as current evidence.
4. Dispatch identity and the complete x/y/z workgroup tuple are kept distinct
   from owner identity. gfx1250 cluster identity is an additional coordinate,
   not a hash collision accepted as identity.
5. Bounded state has a typed capacity and loss contract. Access loss can miss a
   conflict; synchronization loss can fabricate one and therefore requires a
   stronger trust response.
6. Static coverage, runtime selection, dynamic evidence loss, diagnostic
   capacity, and analysis verdict remain separately observable and are combined
   only by the final trust evaluator.
7. Runtime installation policy is separate from transform correctness.
   `Unchanged`, `NoApplicableSites`, `Unsupported`, `Invalid`, and
   `ModifiedValid` are not aliases. `Unchanged` means no replacement was
   produced without claiming why; `NoApplicableSites` means the inventory found
   nothing the selected mode should observe; `Unsupported` means a required
   contract could not be provided; `Invalid` means input or output validation
   failed; and `ModifiedValid` means a changed image passed final validation.
   Fail-open/fail-closed policy chooses what the runtime does with that status;
   it does not rewrite the status.
8. Replacement bytes, report buffers, copied kernargs, symbol bindings, and
   dispatch adjustments remain alive until the last runtime consumer is done;
   teardown is idempotent and accounts for failures.

### 4.4 Architecture invariants

`rj_code_arch_t` is the current C++ enum that names a RocJitsu GPU target;
`gfx*` means any literal target identifier beginning with `gfx`.

1. Engine policy never switches on `gfx*` or `rj_code_arch_t`.
2. Semantic forms are coarser than mnemonics. A target adapter maps decoded ISA
   spellings to forms and exact operands; the shared capability matrix decides
   whether the engine contract applies.
3. Wave size is a per-kernel fact, not inferred solely from target family.
4. Register granularity, separate versus unified AccVGPR allocation, preload
   layout, HWREG availability, wait-counter spelling, branch/call reach, VGPR
   bank state, cluster state, scratch ABI, and instruction encoding terminate
   at target-profile or target-lowering boundaries.
5. Simulator and hardware execute the same transformed code-object contract.
   Runtime buffer placement and queue integration may differ, and differential
   behavior is a defect to explain rather than an expected fork.

## 5. Destination architecture

### 5.1 Dependency direction

**Dependency direction** answers “which component is allowed to know about
which other component?” An arrow `A -> B` below means that `B` consumes an
output or contract from `A`; it does not mean that `A` may call back into `B` or
mutate `B` later. A plus sign joins several required inputs. “Optional” means a
transformation that stops after inventory/planning can omit a concrete runtime
binding, while a transform that embeds an address must supply one.

Names used in the graph that are not already defined data contracts are:

- `EnginePolicy`: pure decision logic that maps the request, target
  capabilities, and semantic inventory to an observation plan.
- `ResourceRequests`: the register, state, segment, and lifetime needs declared
  by the chosen probe intents before any concrete register is selected.
- `RuntimeCoordinator`: host logic that checks runtime capabilities, allocates
  and binds report storage, retains it for the required lifetime, and returns
  evidence after execution. A narrow HSA or simulator **backend adapter**
  performs the backend-specific API calls.
- `ResourcePlanner`: logic that combines requests with liveness and capability
  facts to choose legal concrete resources.
- `PlacementTransaction`: the all-or-nothing operation that lays out lowered
  fragments, fixes branch targets, edits descriptors, and validates the final
  code object.
- `ValidatedTransform`: replacement bytes that have passed structural and
  target validation. `StaticMapping`: the stable relation between original
  sites and their locations in those replacement bytes.
- `EvidenceDecoder`: validation and conversion from report bytes to typed
  events. `EngineAnalyzer`: the mode-specific algorithm that interprets those
  events. `TrustEvaluator`: shared logic that combines analysis with coverage
  and loss. `DiagnosticRenderer`: edge logic that converts the typed verdict
  and site mapping into stable text or structured output.

The production dependency graph is:

```text
Configuration text/environment -> immutable request/policy values
Target/code metadata             -> TargetProfile
TargetProfile + original image   -> ProgramInventory

ConSanRequest + TargetProfile + ProgramInventory
    -> EnginePolicy -> ObservationPlan

ObservationPlan -> CoverageLedger
ObservationPlan -> EvidenceRequirements
ObservationPlan -> ResourceRequests

EvidenceRequirements + RuntimeCapabilities
    -> RuntimeCoordinator -> BoundRuntimeResources
ResourceRequests + TargetProfile + RuntimeCapabilities
    -> ResourcePlanner -> ResourcePlan

ObservationPlan + ResourcePlan + optional BoundRuntimeResources
    -> TargetLowerer + PlacementTransaction
    -> ValidatedTransform + StaticMapping
    -> CoverageLedger

BoundRuntimeResources -> EvidenceDecoder -> EngineAnalyzer
CoverageLedger + analysis -> TrustEvaluator -> DiagnosticRenderer
```

Arrows are one-way. Host analyzers depend on semantic and evidence schemas, not
on patch geometry. Engine policy depends on semantic inventory, not on liveness
or instruction encoders. Target lowering receives an already-decided probe
intent, not permission to reinterpret applicability. The runtime coordinator
binds requirements through a narrow backend adapter and owns lifetimes; it does
not select sanitizer semantics.

No virtual class hierarchy is implied. Prefer plain immutable structures, typed
variants, and pure functions. Introduce an interface only when there are real
independent implementations or when an old/new migration seam requires one. A
generic wrapper around duplicated code, with no common invariant or algorithm,
is not a useful component.

### 5.2 Core data contracts

#### `ConSanRequest`

`ConSanRequest` is immutable user/product policy. A **semantic domain** is a
category of behavior—ordinary accesses, barriers, atomics, or fences.
**Provenance policy** says how conservative pointer facts, especially
group-FLAT candidates, affect eligibility. **Bounded-evidence policy** sets
supported capacity choices without choosing an address.
**Completeness/trust policy** says which typed gaps make a run incomplete or
unacceptable. A
**stable expert control** is an intentionally documented user option, not every
tuning field that happens to exist in the prototype. The **supported surface**
is the set of options and behavior the project promises to maintain for users.

It contains:

- selected engine;
- enabled semantic domains and provenance policy;
- sampling and bounded-evidence policy;
- evidence completeness/trust policy that belongs to the sanitizer;
- stable expert controls that are deliberately part of the supported surface.

It excludes concrete registers, report addresses, resolved layouts, per-owner
assignments, test-only filters, fault mutation, and patch placement. A separate
`ConSanDebugOverrides` carries focused-test and early-development controls. A separate
`MutationRequest` carries fault/perturbation policy. Parsing environment values
is outside all three types. `TransformPolicy` owns image-growth, placement-work,
and transformation resource ceilings. `RuntimePolicy` owns activation,
installation requirements, fail-open/fail-closed behavior, and process-level
lifetime/capacity policy.

A **resolved layout** contains exact offsets and sizes after planning. A
**ceiling** is a hard upper bound, such as maximum added image bytes or process
report memory. Separating these policies prevents an environment parser or
runtime failure rule from changing which instructions are semantically
relevant.

#### `TargetProfile`

`TargetProfile` contains immutable architectural facts selected once for a code
object and refined with per-kernel ABI facts where necessary. A
**semantic instruction form** is the normalized meaning used by engine policy,
such as an LDS store or workgroup barrier, rather than one mnemonic spelling.

The profile contains capabilities and quantities, not engine policy: wave
sizes, register allocation/granularity, separate or
unified accumulation storage, preload availability and layout, scratch/private
ABI, architectural LDS and addressing limits, execution-mask width,
branch/call forms and reach, wait-counter families, address widths, cluster
support, VGPR bank-mode support, and supported semantic instruction forms.

The code-object profile is immutable. “Refined with per-kernel facts” means
deriving an immutable per-kernel view, not editing the shared profile.

Encoding functions do not live in the profile. `TargetClassifier` extracts
semantic operands from decoded instructions and `TargetLowerer` emits reviewed
target sequences. A **semantic operand** is the role of an instruction field,
such as address, stored value, returned old value, width, scope, or ordering.
These are the only target-dependent behavior interfaces.

#### `RuntimeCapabilities`

An **execution backend** is either the physical HSA runtime path or the
RocJitsu simulator path.

`RuntimeCapabilities` contains facts discovered from that selected execution
backend and agent rather than
inferred from `gfx*`: host/device-visible memory properties, atomic/coherence
support, allocation limits, runtime segment limits, and executable/dispatch
binding facilities. Resource and evidence planning consume this value
explicitly. A simulator and physical agent may have different runtime
capabilities while using the same `TargetProfile`.

Keeping these facts separate prevents “this is gfx950” from being mistaken for
“this particular gfx950 agent offers a suitable memory pool.”

#### `ProgramInventory`

`ProgramInventory` is an immutable description of the original code object.
Here **CFG ownership** associates shared helper code with the kernels that can
reach it. **Dynamic LDS** is extra per-dispatch group-segment storage in
addition to a kernel's fixed LDS use.

An **execution-owner proof** is the analysis evidence for attributing shared or
called code to a kernel/owner domain. A **barrier lifecycle group** relates the
events that initialize, arrive at, wait on, or reuse one logical barrier. An
**ordinary release/acquire association** relates a data
load/store and separate cache/fence instructions that together implement
publication without one native ordered atomic. An **inventory exclusion** is a
typed fact explaining why analysis could not classify an otherwise relevant
item; it is retained rather than erased. An **analysis limitation** records a
fact the inventory builder could not prove, even when there is no single
excluded instruction.

`PhysicalSiteId` is a tuple: code-object identity prevents offsets from two
objects colliding, and original text offset locates the instruction before any
patch changes addresses. An **ordinal** is a small stable index distinguishing
members with the same physical location. `SemanticSiteId` adds the semantic
domain and ordinal to the physical tuple.

The inventory contains:

- fingerprint, target, text sections, kernels/functions, descriptors, entry
  points, resource metadata, dynamic-stack/dynamic-LDS facts, and control-flow-
  graph (CFG) ownership;
- `PhysicalSiteId = (code_object_id, original_text_offset)` plus container and
  owner references;
- decoded access ranges, access kind, address-space/provenance, operand facts,
  and semantic confidence;
- synchronization events and logical sequences, operation, scope, memory role,
  dynamic outcome requirements, and exact member identities;
- execution-owner proofs, barrier lifecycle groups, and ordinary
  release/acquire associations; and
- typed inventory exclusions and analysis limitations.

One instruction can yield several `SemanticSiteId`s by adding a domain and
member/range ordinal to its `PhysicalSiteId`. Patch placement remains keyed by
the physical site. This makes two-address LDS forms and multi-event barrier
sequences explicit without duplicating insertion points. It also lets coverage
speak about two logical ranges while placement still emits at most one physical
probe there.

Inventory contains no selected engine, report layout, registers, patch bytes,
or mutable warning list. Fault candidate discovery may produce a separate view
over inventory; it does not extend the core inventory with live mutation state.

#### `ObservationPlan`

`ObservationPlan` is the engine's complete target-neutral statement of what one
transform intends to observe. Every probe observes every lane active at its
guest instruction, and every admitted probe is part of the requested engine
contract: losing one during lowering is a coverage gap. An **evidence kind**
names the logical record or state update the probe must produce. An **evidence
schema** defines the typed fields and meaning of an evidence kind independently
of its concrete byte offsets.

The plan contains:

- a `SiteDecision` for every relevant semantic site;
- admitted `ProbeIntent`s keyed to physical sites, with before/after position,
  guest values required, and evidence kind;
- engine state requirements such as entry identity, epoch, sampling selector,
  exact shadow, ordering tables, or sticky marker;
- the logical evidence kinds from which capacity inputs are derived; and
- relationships between intents, such as one physical probe covering both
  halves of a logical barrier sequence.

`SiteDecision` is one of `NotApplicable(reason)`, `Unsupported(reason)`, or
`Admitted(intent_ids)`. Lowering cannot change it to supported. A distinct
`LoweringOutcome` later records `Instrumented`, `ResourceRejected(reason)`, or
`PlacementRejected(reason)`.

`NotApplicable` means the site's semantic form does not belong to the selected
engine contract. `Unsupported` means it does belong but a stated capability or
analysis requirement is missing. `Admitted` means policy has requested one or
more intent identifiers. `Instrumented` means all required machine-code work
for that intent was placed. `ResourceRejected` means no legal register/state
assignment was found. `PlacementRejected` means resources existed but no legal
transactional code layout was found. These states deliberately separate a
policy answer from a mechanism result.

`ProbeIntent` is a closed semantic variant, not a bag of target words. Initial
variants include redundant access observation, access record, sampled access,
exact-shadow access, synchronization record, sampled synchronization,
epoch advance, inline ordering publication/acquire, entry-state capture, and
termination flush. Engine-specific fields stay inside the matching variant.

A **closed semantic variant** means that the allowed intent kinds are an
explicit finite list checked by the compiler. **Entry-state capture** copies an
ABI value while it is still valid at kernel entry. **Termination flush**
publishes any evidence that must be finalized before a wave or kernel ends.
Neither term prescribes target instructions; lowering decides those.

#### `CoverageLedger`

To **join without conflating** means to associate facts through stable site,
intent, dispatch, and evidence identifiers while keeping each axis separately
queryable. For example, “the static site was instrumented” does not imply “this
dispatch was selected,” and “the dispatch was selected” does not imply “all
records were published.” The ledger retains all three facts.

`CoverageLedger` joins, without conflating:

1. inventory applicability and typed exclusions;
2. engine admission decisions;
3. resource and placement outcomes per physical and semantic site;
4. dispatch selection and unsampled counts;
5. published, dropped, saturated, malformed, and stale evidence by kind; and
6. diagnostic capacity and host-analysis completeness.

The ledger is the source of truth for coverage and trust. Human-readable
warnings and logs render it. Exact patch counts, cave locations, or selected
register numbers are optional mechanism telemetry and are never substituted
for it.

#### `EvidenceRequirements` and `BoundRuntimeResources`

A **logical delivery contract** says whether evidence is code-object-wide,
executable-wide, or per-dispatch and which memory, coherence, and atomic
properties publication needs. A **lifetime token** proves that the underlying
allocation remains alive while a transform, executable, dispatch, or evidence
reader can still use it.

`EvidenceRequirements` is a pure output of request, observation plan, and
capacity policy. It describes engine schema, required bytes/alignment/memory
properties, scope, capacities, loss classes, generation rules, a logical
delivery contract, and any private/group-segment additions. It contains no
address or allocation handle. Keeping requirements address-free permits
inventory and policy to run before allocation and permits the same plan to be
rebound to another dispatch generation.

The runtime coordinator satisfies it through a narrow backend adapter and
returns `BoundRuntimeResources`, which
adds concrete allocation handles, device-visible addresses, generation,
dispatch/code-object scope, layout version, and lifetime token. Rebinding does
not require rebuilding semantic inventory. A transform that embeds a concrete
address consumes the bound value in a distinct late-binding lowering pass.

#### `ResourcePlan`

A scalar/vector **tuple width** is the number of adjacent SGPRs or VGPRs needed
for one logical value. **Fixed architectural state** is a named register or
mode that cannot be freely allocated. An **outstanding-counter effect** says
which target wait counters a probe's own memory operations increment or wait
on. **Renaming** moves a still-live guest value to another legal register so
the original register can be used. **Descriptor growth** raises declared
kernel resource usage consistently with the actual new allocation.

Each probe intent declares semantic requirements rather than chosen registers:

- guest values and results it reads;
- persistent and transient scalar/vector tuple widths;
- clobbers, fixed architectural state, execution-mask policy, and required
  before/after position;
- memory operations and outstanding-counter effects;
- lifetime: site, wave, kernel, workgroup, or dispatch; and
- acceptable strategies: dead registers, renaming, descriptor growth, private
  spill, or not applicable.

The resource planner combines those requirements with liveness, owners,
`TargetProfile`, `RuntimeCapabilities`, and the logical evidence-delivery
contract to produce immutable per-owner/per-site assignments. Concrete buffer
addresses do not affect semantic admission or resource strategy; they enter
only the explicitly late-bound lowering step. Lowering consumes the assignment
exactly; it does not mutate configuration and ask the planner to try again
implicitly. Alternative strategies and rejection reasons remain typed.

“Consumes the assignment exactly” means the lowerer either emits using the
chosen registers/state or returns a typed inconsistency; it cannot silently pick
a different scratch register and make the recorded plan false. If several
strategies are legal, the planner records the alternatives and the reason one
was selected.

#### `TransformResult` and `RunVerdict`

**Telemetry** is optional implementation measurement, such as chosen
registers, patch counts, or cave sizes. It helps debugging but is not the
behavioral contract. A **typed contract failure** is a
`ConSanContractIssue` stored by the pipeline stage that rejected its input;
lowering and final-validation errors retain diagnostic text.

`TransformResult` contains the validated replacement bytes when present,
static original-to-instrumented mappings, resource/segment requirements,
coverage ledger, typed transform status, typed contract failures, and
contextual lowering errors. Semantic
inventory and debug telemetry are optional separately owned artifacts rather
than the current roughly 600-line result structure shared by every caller.
A transform result is static: it says what happened while rewriting, not
whether a conflict occurred during a later execution.

The implementation records six independently meaningful contracts in a fixed
`ConSanPipelineStage` array: configuration, target/runtime capabilities,
program inventory, observation plan, evidence requirements, and runtime
binding. The array position is the stage identity; `TransformResult` owns the
collision-aware pristine `ConSanCodeObjectId` once. Lowering, final validation,
and completion are not second copies of the result outcome. `Deferred` means a
valid address-free result awaits runtime binding; `NotApplicable` means the
selected mode or result needs no value from that stage. Neither is silently
presented as `Completed`. Request, capability, and binding failures live as the
`ConSanContractIssue` of their owning stage, so control flow never parses
diagnostic text. Lowering and final-validation diagnostics live once in the
shared transform artifacts.

`TransformResult` owns immutable inventory, observation plan, coverage,
replacement bytes, status, diagnostics, mutation detail, resource plans, and
emitted patch geometry. The HSA coordinator consumes these values directly;
there is no public raw-lowering-result view. Only an address-free MOI sizing
result privately retains the unmodified semantic state required by the bound
retry, and ordinary or completed transforms never retain that state. Large
replacement images and every published artifact therefore remain
single-owned.

`RunVerdict` is produced only after runtime evidence. It distinguishes:

- conflict or SuperCollider mismatch observed;
- no conflict observed in retained evidence, with complete execution of the
  declared bounded or sampled contract (not a claim of global race freedom);
- no conflict observed but statistically or dynamically incomplete;
- transform/runtime unsupported;
- malformed or untrustworthy evidence; and
- application or instrumentation failure.

Presentation strings, exit policy, and testing assertions consume this type.

The second verdict does not mean “race-free.” It means every site and runtime
event promised by the declared bounded or sampled policy completed, and no
conflict appeared in the evidence that policy retains. The third verdict says
the absence of a conflict cannot even make that narrower complete-within-
contract claim because selection, capacity, or validity was incomplete.

### 5.3 Components and ownership

**Ownership** means that one component is the authoritative place where a
decision, invariant, or state transition is implemented. “Consumes / produces”
names its immediate input and output contract; it does not permit reaching into
another component's private state. An **isolation-test contract** says what can
be proved by constructing only that component's direct inputs. A
**documentation contract** says which facts a maintainer must be able to learn
without reverse-engineering the implementation.

The “five-target disposition” column uses three ideas:

- **Shared unchanged:** one semantic algorithm is compiled once and receives
  no target name. Target facts may already have been normalized in its input.
- **Shared with capability data or a narrow adapter:** one algorithm remains
  shared, but data supplies quantities/legal forms or a small target operation
  classifies/emits native instructions.
- **Genuinely target-specific:** machine-code behavior differs because an ISA
  or ABI fact differs. The row must name that fact and keep the difference at a
  target boundary.

The isolation-test column deliberately mixes semantic table/property tests with
mechanism-specific byte goldens, lifecycle state-machine tests, and placement
rollback tests. Each belongs at the boundary that owns the corresponding
contract.

| Component | Owns | Consumes / produces | Five-target disposition | Isolation-test contract | Documentation contract |
| --- | --- | --- | --- | --- | --- |
| Configuration parser | Environment/API spelling, defaults, validation | text → `ConSanRequest`, `TransformPolicy`, `RuntimePolicy`, debug and mutation requests | Shared unchanged | table-driven parsing, conflicting/ignored options, load-only default activation | supported options, defaults, deprecations |
| Target profile | Architectural and per-kernel ABI facts | target/code metadata → `TargetProfile` | Shared table; each target supplies data | complete five-target fact matrix and impossible-combination rejection | reason for every differing fact |
| Runtime-capability discovery | Agent/backend allocation, coherence, atomic and binding facts | HSA/simulator query → `RuntimeCapabilities` | Shared fact model; narrow backend query adapter | physical/simulator fixtures and impossible-property rejection | provenance and meaning of every runtime fact |
| Target classifier | ISA spelling to semantic operands/forms | decoded instruction → typed access/synchronization facts or reason | Narrow implementation per encoding family; no engine policy | golden decode/property cases for every distinct mechanism | semantic form and confidence mapping |
| Inventory builder | Stable sites, ranges, synchronization sequences, ownership, address recipes | code object + profile → `ProgramInventory` | Shared algorithms using classifier | synthetic CFG/ownership/sequence cases on all applicable targets | vocabulary, identity, confidence invariants |
| Engine policy | Applicability and semantic observation intent | request + inventory + target profile → `ObservationPlan` | Shared unchanged; capability data is input | per-engine semantic matrices, multi-range/sequence coalescing, no target names | each mode's claims and deliberate exclusions |
| Coverage ledger | Static/dynamic completeness and typed reasons | decisions + lower/runtime outcomes → coverage/trust inputs | Shared unchanged | exhaustive state-transition and evidence-loss tests | meaning of clean, incomplete, unsupported |
| Evidence planner | Engine-specific retention semantics and bounded capacities | observation plan + capacity policy → requirements | Shared arithmetic; closed typed alternative per engine | overflow/boundary/layout-independent capacity tests | retention and completeness semantics per alternative |
| Resource planner | Register/state/spill/segment assignments | intents + liveness + target/runtime capabilities → plans | Shared search; target costs/legality are profile/adapter facts | dead/grow/spill/reject cases for each mechanism | hierarchy, lifetime, occupancy effects |
| Target lowerer | Exact probe and entry/exit sequences | intent + resources + binding → target words/fixups/effects | Narrow target implementation; share algorithms and builders | byte goldens only for distinct encodings plus state-preservation tests | ISA/ABI reason for every branch |
| Placement transaction | Relocation, caves/islands, fixups, descriptors, final ELF | lowered fragments + original image → validated transform | Shared unchanged; target branch forms supplied by lowerer | branch reach, multiple sites, rollback, large image, final reparse | layout strategy and transactional guarantees |
| Runtime coordinator | Allocation, replacement and executable lifetime, dispatch binding | requirements + runtime capabilities + HSA/DBI runtime → bound resources/evidence snapshots | Shared lifecycle; backend adapter differs between simulator/hardware | mocked lifecycle, concurrent objects/dispatches, failure cleanup | ownership/lifetime/state machine |
| Evidence decoder | Commit/generation/layout validation | bytes + typed evidence alternative → evidence or malformed reason | Shared unchanged | torn/stale/overflow/malformed snapshots | publication and validation protocol |
| Engine analyzers | Record/Replay replay, Sampled scan, SuperCollider/Inline collection | typed evidence + semantic metadata → analysis facts | Shared unchanged | model/reference traces and bounded-loss cases | algorithm and limitations per engine |
| Trust evaluator | Final supported-contract claim | coverage + evidence + analysis → `RunVerdict` | Shared unchanged | truth table for loss, rejection, saturation and conflicts | user meaning of every verdict |
| Diagnostic renderer | Stable user-facing attribution and summaries | verdict + site mapping → text/JSON | Shared unchanged | schema/golden tests only for supported output surface | versioned output fields |
| Mutation pipeline | Reviewed validation corruption/perturbation | mutation request + inventory → separate transformed image/proof | Shared planner; target mutation emitter is narrow | exactly-one selection, pristine identity, composition/rollback | explicitly transformational scope |

“Shared unchanged” does not mean every target executes identical instruction
bytes. It means those bytes were produced beyond the shared semantic component's
boundary.

### 5.4 Component rules

- Components exchange immutable values or explicit builders whose mutability is
  confined to construction. A component may not reach back and edit its input.
- Each component owns its diagnostic reasons as enums/structured data. Rendering
  lives at the edge.
- Stable semantic types live below engines. Engine-specific ABI types live with
  their evidence schema, not in target lowerers. “Below engines” means in a
  dependency layer that every engine may consume but no engine owns.
- Target adapters expose facts and operations, not a general escape hatch that
  accepts an engine enum and an options blob. An **options blob** is an
  unstructured or over-broad parameter collection whose fields let the adapter
  rediscover policy.
- An isolation test should normally construct the component's immediate input.
  Requiring an ELF fixture for pure capacity arithmetic or a physical GPU for
  site admission signals a missing boundary.
- Every component document states purpose, non-responsibilities, input/output,
  invariants, target variation, failure categories, and tests. File comments
  explain encodings and local proofs; this document explains system ownership.

### 5.5 Worked example: one LDS store under Record/Replay

This example is illustrative; exact field names remain a Stage 2 coding choice.
It shows why the stages are separate.

1. The original code object contains a native LDS store at original text offset
   `0x120`. The target classifier decodes its target-specific operand fields and
   returns the semantic fact “four-byte LDS write whose address comes from this
   VGPR.” It does not decide whether Record/Replay wants the site.
2. The inventory builder assigns a `PhysicalSiteId` containing the code-object
   identity and `0x120`. It creates an access `SemanticSiteId` containing that
   physical ID, the access domain, and range ordinal zero. The completed
   `ProgramInventory` cannot later be edited.
3. Record/Replay engine policy sees a supported LDS write and produces
   `Admitted(intent_7)` plus an access-record `ProbeIntent`. The intent states
   that the address, width, write kind, active lanes, owner, and site identity
   must be recorded before the guest store. It names no SGPR, VGPR, report
   address, or target instruction.
4. The evidence planner includes one access-record slot and its publication/loss
   rules in `EvidenceRequirements`. Separately, the intent yields resource
   requests for temporary registers and for persistent dispatch/workgroup/owner
   identity.
5. The runtime coordinator checks that the selected agent offers the required
   fine-grained coherent memory and atomics, allocates the report, and returns a
   `BoundRuntimeResources` value containing the concrete device address and
   lifetime. The resource planner uses liveness and the target/runtime facts to
   assign legal registers and any spill or descriptor growth.
6. The target lowerer emits native instructions for the intent using exactly
   those assigned resources and the bound address. The placement transaction
   puts the probe before `0x120`, preserves the original store exactly once,
   resolves branches/fixups, updates metadata, and validates the final ELF. It
   records `Instrumented` for the intent in the coverage ledger.
7. At runtime the probe publishes a committed access record. After completion,
   the evidence decoder rejects stale, torn, wrong-generation, or malformed
   data and converts valid bytes to a typed access event. Record/Replay analysis
   combines it with other access and synchronization events to find conflicts.
8. The trust evaluator combines that analysis with the coverage ledger. It may
   conclude “conflict observed,” “no conflict in complete retained bounded
   evidence,” or “incomplete.” Only then does the diagnostic renderer create
   user-facing output.

If classification had been ambiguous, the inventory would retain that
confidence. If policy required a form unavailable on this target, it would
produce `Unsupported(reason)`. If registers or placement failed, the original
`Admitted` decision would remain true while `LoweringOutcome` recorded the
mechanism failure. No later stage would delete the site and make the run look
clean.

## 6. Cross-architecture design

**Cross-architecture sharing** means one semantic algorithm serves every
applicable target while target facts arrive as data or narrow classification/
emission calls. A **capability matrix** is a table indexed by target, engine,
and semantic form that states supported, unsupported, or not applicable with a
reason. To **fork an engine** is to create a separate target-selected copy of
its semantic algorithm; capability data must not do that. An
**access-only atomic** is treated as a conflicting load/store but supplies no trusted
happens-before edge. An **ordered atomic** additionally participates in the
supported release/acquire model.

### 6.1 What is shared

The following have one implementation for gfx942, gfx950, gfx1100, gfx1201,
and gfx1250:

- site identity, physical alias canonicalization, semantic confidence, access
  range overlap, and ownership vocabulary;
- synchronization event/sequence construction once target operands have been
  classified;
- engine applicability, mode policy, coverage state transitions, report
  requirement arithmetic, evidence decoding, replay/scan models, trust
  evaluation, and diagnostics;
- resource-search strategy and placement transaction; and
- runtime lifetime and failure state machines above the simulator/HSA backend.

Capability data can remove a form from a target, but must not fork the engine.
For example, cluster barriers and ordered LDS atomics are forms available on
CDNA5, while relaxed LDS atomic accesses are access-only on the admitted CDNA
targets. That fact belongs in the capability/profile matrix consumed by the
same engine policy.

### 6.2 Target mechanics used below

- Instruction formats, opcodes, operands, offset scaling, and cache/order/scope
  bits differ by target and may change the semantic classification or emitted
  encoding.
- A **two-address** instruction touches two independently meaningful byte
  ranges. An **asynchronous transfer** starts data movement whose completion is
  tracked separately. A **direct-to-LDS** form moves data directly into LDS.
  A **transpose** form rearranges data as part of the operation. These forms
  matter when they change address classification, operand capture, or required
  waits; otherwise they are workload context around ordinary classified sites.
- **MFMA** (matrix fused multiply-add), **WMMA** (wave matrix multiply-
  accumulate), and **SWMMAC** (scaled wave matrix multiply-accumulate) are AMD
  matrix instruction families. **FP8** is an eight-bit floating-point format.
  **TDM** here means tensor data movement. These instructions may create
  register pressure or accompany LDS traffic without themselves becoming
  ConSan sites.
- **System/preloaded SGPRs** carry runtime-supplied kernel-entry values. A
  **launch payload** is the set of entry values describing the dispatch and
  workgroup. **Identity capture** copies the required values before guest code
  may reuse their registers.
- A **call form** transfers control with a return convention. `s_call` is a
  gfx1250 scalar call form used by relevant lowering paths. A **literal64** form
  embeds a 64-bit immediate value in instruction data. **Split counters** means
  a target tracks categories of outstanding memory work in separate wait
  counters.
- **High-bank LDS** means LDS addressing that reaches target-provided storage
  beyond the lower bank range handled by narrower encodings. A **packed** form
  represents multiple subvalues in one operand. A **VGPR bank transition**
  changes the target's current VGPR indexing mode and must be restored.

These are mechanics, not a license to duplicate engine policy. Where a term is
only workload context, the device tests exercise it because it changes code
shape or resource pressure around actual ConSan sites.

### 6.3 Narrow target boundaries

Target-specific code is justified only for the following mechanics:

1. **Instruction classification.** Raw formats, opcode fields, mnemonic
   aliases, address/data/result operands, offset scaling, cache/order/scope
   bits, native LDS/two-address/asynchronous/tensor forms, and instruction size.
2. **Kernel ABI and resources.** Wave size, SGPR/VGPR granularity, AccVGPR
   separation or unification, system/preloaded SGPR layout, scratch/private
   addressing, dynamic-stack conventions, group-segment limits, and metadata
   fields that must change with allocation.
3. **Identity capture.** HW_ID/HW_ID1 and related fields, available launch
   payload, dispatch identity source, x/y/z workgroup coordinates, resident-wave
   identity, and gfx1250 cluster-workgroup identity.
4. **Probe emission.** Target instruction encodings, EXEC/VCC width, waits and
   cache operations, spill sequences, branch/call/fixup forms, and gfx1250 VGPR
   bank transitions.
5. **Target-only facilities.** gfx1250 clustered dispatch/barriers and ordered
   LDS atomics are semantic forms. High-bank LDS is an addressing and lowering
   difference. Tensor/matrix data movement and asynchronous transfers are
   workload context unless a reviewed classifier deliberately exposes a
   sanitizer-relevant semantic form; their mere presence does not create one.

Even here, the algorithm is shared when the operation is the same. “Save the
guest mask, run under the declared mask, publish one record atomically, drain
probe memory, restore state” is common; only the builder operations differ.

### 6.4 Per-target disposition

In this table, **wave32** and **wave64** mean a kernel executes with 32 or 64
lanes per wave. **Exclusive coverage** names a test context that exists only on
that target; it does not imply a separate engine.
**Physical differential execution** runs the same contract in simulation and
on installed hardware and investigates any difference. “gfx9-style scratch”
means the scratch-addressing
ABI inherited by the CDNA3 target family. A **wide** instruction form transfers
more bytes or register operands than the ordinary scalar-width form. A
**scaled matrix** form applies encoded scale factors as part of a matrix
operation. “gfx12 encoding” names instruction-encoding machinery shared by
gfx1201 and gfx1250; it does not imply that their ABI or capabilities are the
same.

| Target | Shared semantic path | Facts/mechanics supplied by target boundary | Genuine exclusive coverage |
| --- | --- | --- | --- |
| gfx942 / CDNA3 | All four engines and common MOI model | wave64 profile, CDNA3 encodings/waits, descriptor/register/AccVGPR rules, hardware identity and launch payload, gfx9-style scratch | no engine fork; CDNA-family native matrix/direct-to-LDS mechanisms where applicable |
| gfx950 / CDNA4 | Same as gfx942, including physical differential execution | wave64 profile, CDNA4 encodings/cache and wait details, descriptor/register/AccVGPR rules, hardware identity, CDNA4 transpose and large-LDS capabilities | native CDNA4 transpose and other CDNA4-only ISA forms; physical runtime backend coverage |
| gfx1100 / RDNA3 | Same four semantic engines | wave32 kernel profile, RDNA3 encodings, preload/identity and scratch rules, unified VGPR/AccVGPR resource facts where applicable | RDNA3-native WMMA/packed forms; no invented substitute for unavailable packed atomics |
| gfx1201 / RDNA4 | Same four semantic engines | wave32 profile, gfx12/RDNA4 encodings and split counters, TTMP/preload/identity details, VSCRATCH and unified resource rules | RDNA4 FP8/WMMA and target-native wide forms |
| gfx1250 / CDNA5 | Same four semantic engines | wave32 profile, gfx12/CDNA5 encodings, HW_ID1 plus launch/dispatch identity encodings, VGPR-bank mode, wider LDS addressing, cluster coordinate, CDNA5 scratch/resource rules, `s_call`/literal64 forms | cluster semantics and ordered LDS atomics; high-bank LDS lowering; TDM/asynchronous/SWMMAC/scaled-matrix workload contexts where they exercise classified sites |

The table names families to explain mechanisms, not to create family-level
engine implementations. gfx1201 and gfx1250 may share an encoding builder while
having different ABI and capability rows. gfx942 and gfx950 may share an
algorithm while still selecting distinct generated encoders.

### 6.5 Review rule for target branches

Every production target conditional must answer all of these questions in a
nearby comment or profile definition:

1. Which ISA/ABI fact differs?
2. Why can it not be represented as data or an existing target-builder call?
3. Which other targets share the mechanism?
4. Which focused host test and device contract exercise it?
5. What would be deleted if RocJitsu/DBI acquired the general facility?

An engine-name plus target-name conditional is presumptively misplaced. An
exception requires a semantic capability unique to that target, not merely a
different encoding.

## 7. Engine relationships and contracts

An engine **contract** states which events it observes, which ordering facts it
supports, where it decides a conflict, what evidence it retains, and what an
absence of diagnostics is allowed to mean. To **normalize** an access is to
express target-specific operands as one byte-range/kind representation. A
**predicate** is a boolean rule; the conflict predicate returns true exactly
when the modeled facts constitute a conflict. **Attribution** associates a
record or diagnostic with its original code object, site, dispatch, workgroup,
and owner. An atomic's **dynamic outcome** is a result known only after it runs,
such as compare-and-swap success. Its **memory role** says whether it acts as an
access, release, acquire, or a combination. An **association** is a proven link
between separate instructions that together implement one ordering pattern.

The high-level contracts are:

| Mode | Device action | Retained evidence | Where the useful decision occurs | What no diagnostic means |
| --- | --- | --- | --- | --- |
| Record/Replay | Publish bounded access and synchronization records. | Committed event snapshot plus loss/capacity facts. | Host replay. | No conflict appeared in valid retained events; completeness depends on bounded coverage. |
| Sampled | Select dynamic identities and publish bounded causal windows. | Selected watchpoints and matching synchronization metadata. | Host scan, except optional experimental device checks. | Statistically inconclusive even when retained data is valid. |
| Inline Shadow | Update shadow/order state and check conflicts. | Device state plus bounded attributed diagnostics. | Device probe. | No supported-form conflict was emitted within declared capacity; unsupported/lost coverage remains separate. |
| SuperCollider | Delay and repeat/read back an observation, then compare. | Sticky mismatch marker. | Device comparison. | No observed value instability; it is not a race-freedom statement. |

### 7.1 Shared semantic core

The MOI core defines:

- normalized LDS byte ranges and access kinds;
- physical site, code object, container, dispatch, x/y/z/cluster workgroup, and
  owner identities;
- epoch and logical sequence identity;
- workgroup and cluster barrier events;
- ordered atomic operations, dynamic outcomes, scopes, memory roles, and
  addressed ordinary fence associations; and
- the conflict predicate: overlapping range, at least one write, distinct
  owners in the applicable ownership domain, and no supported ordering edge.

The core does not define how many events must be retained, where state lives,
or where a conflict is decided. Those are engine contracts.

An **ownership domain** is the set of owners compared for one memory scope, for
example lanes or waves within the relevant workgroup/cluster. A normalized
range provides start byte and width, so overlap does not depend on the original
mnemonic.

SuperCollider reuses normalized access sites, target lowering, resource
planning, coverage, and runtime binding, but not the MOI conflict or ordering
model.

### 7.2 Record/Replay policy

A **site token** is the compact report value that maps back to a stable semantic
site. A dispatch/access **table** is a bounded indexed region used to retain
first-light identities. Invalid tokens, collisions, or exhausted table entries
must remain visible rather than being discarded as if no event occurred.

- Request before-access records and before/after synchronization records as
  required to capture operands and dynamic outcomes.
- Preserve exact dispatch/workgroup/owner/site attribution for each committed
  record.
- Perform ordering and conflict analysis on the host.
- Treat record capacities, dispatch/access table saturation, invalid site
  tokens, lost synchronization evidence, and malformed commits as explicit
  trust inputs.
- Retain its role as semantic reference for shared MOI conflict behavior while
  documenting that bounded first-light capture is not exhaustive.

Record/Replay is the first DBI migration because its complex processing is
host-side. That scheduling fact does not make its report table or current
first-light policy the abstract parent of other engines.

### 7.3 Sampled policy

A **selector** is the deterministic function deciding whether a dynamic
identity participates in a sampled window. Deterministic means the same seed
and identity produce the same answer, making a run reproducible. A **collision**
means two distinct identities compete for one bounded bank/slot. A **drop** is
evidence that could not be published. Both affect completeness.

- Select a bounded subset of dynamic access identities under an explicit,
  reproducible sampling policy.
- Attach barrier/atomic synchronization evidence only when it names the same
  causal window; never join banks solely by static site.
- Scan retained windows on the host, with any immediate device signal treated
  as an optional optimization.
- A conflict is meaningful; absence is statistically inconclusive. Selection,
  collisions, drops, and malformed synchronization remain visible.

The intended production boundary permits the selector to move earlier and be
computed once per workgroup without changing inventory, evidence semantics, or
host analysis.

### 7.4 Inline Shadow policy

A **versioned shadow cell** stores access metadata plus a generation/version so
old state is not mistaken for the current dispatch. An **ordering table** stores
bounded release/acquire tokens used by the supported happens-before model. An
**attributed diagnostic** carries enough stable identity to map the device's
decision back to the original program site and owners.

- Maintain exact supported-form shadow state keyed by LDS cell and qualified by
  dispatch/workgroup/owner/epoch.
- Apply supported ordering updates and conflict checks on the device.
- Emit bounded structured diagnostics for decisions already made; host work is
  validation, attribution, and summarization rather than replay.
- Require persistent entry-captured state and explicit table/capacity
  completeness.

Inline Shadow is not re-expressed as a stream of all accesses merely to fit
near-term DBI. It is preserved behind the same target-neutral semantic plan and
runtime-requirement boundary, making the later framework gap concrete.

### 7.5 SuperCollider policy

The **original guest access exactly once** rule distinguishes program behavior
from observation: a guest load or store still has its one original semantic
effect. The later repeated load or store readback is tool evidence and must not
be confused with executing the guest instruction twice.

- Select supported load/store sites using normalized access inventory.
- Execute the original guest access exactly once, delay, perform the supported
  redundant observation, compare, and set a sticky marker on mismatch.
- Report physical/static coverage and marker capacity/lifetime honestly.
- Do not infer peers, happens-before, or race certainty from a mismatch.

The observation often needs both before- and after-instruction values and a
target-correct readback sequence. It can use common placement and state
preservation without inheriting MOI epochs or ordering tables.

### 7.6 Deliberate non-unifications

A **non-unification** is an explicit decision not to force different semantics
behind one implementation merely because their names sound related. A common
report **envelope** is a shared header/publication protocol around an engine-
specific payload. A **union layout** reserves fields/capacity for every engine
in one largest structure even though only one alternative is active. An
**abstract parent** is a base interface that claims its children implement the
same operation; that claim is wrong when the modes answer different questions.

- There is no hypothetical generic `Engine::detectRace()` method spanning SuperCollider and
  MOI; they answer different questions.
- There is no single “shadow” representation spanning RR host replay, Sampled
  windows, and Inline device state; they have different boundedness and
  publication semantics.
- A common report envelope does not require one union layout sized for all
  engines.
- Fault injection is not a fifth engine.
- The mode names do not justify four copies of decoding, site identity,
  resource policy, placement, or target encoders.

## 8. Test and documentation contract

The test map distinguishes semantic units (sanitizer meaning), mechanism units
(encoding/resource/placement details), HSA-hook integration, paired device
contracts, and E2E/physical qualification. A **target-exclusive mechanism** is
an ISA/ABI facility absent from the other targets, not merely a test that was
first written for one target.

### Standing type documentation and unit-test policy

ConSan follows a **type-first contract policy** throughout the production
migration. Every named enum, structure, class, variant, or type alias that is
introduced or materially changed must carry a generous type-level comment. The
comment explains the semantic concept represented by the type, its invariants,
ownership or lifetime when relevant, sentinel and failure states, and the
responsibilities that deliberately remain outside it. A reader should be able
to understand the component's vocabulary primarily from its types instead of
reconstructing it from control flow or scattered field comments.

Every reasonably testable unit behavior owned by such a type must also have a
focused host unit test. This includes construction and validation, enum
iteration and naming, mappings between typed domains, lookup and derivation,
boundary arithmetic, target matrices, derived predicates, and invalid or
sentinel inputs. Passive target data receives an independent expected-value row
for every supported architecture. Compile-time assertions and generated-
documentation comparisons supplement these tests; they do not replace focused
unit tests with precise failure names. Behavior observable only in device code
instead receives the narrowest applicable paired correct/incorrect device
contract, in addition to any host-testable planning or encoding boundary.

No migration slice is complete merely because a new type compiles or is
indirectly exercised by a large integration test. Its type comment and focused
unit contract land with the type. When a type later gains functionality, the
same change extends its focused tests. Tests assert the type's semantic
contract and owned invariants, not incidental representation choices belonging
to another component.

### 8.1 Test tiers

| Tier | What it proves | Preservation rule during migration |
| --- | --- | --- |
| Semantic host units | Site classification, synchronization sequences, ownership, capability, range conflict, report arithmetic, replay/scan/trust models | Highest internal-contract priority. Port to the new component input before deleting the old implementation. |
| Mechanism host units | Register choices, spill/wait sequences, patch bytes, relay/placement, descriptor edits, ELF rollback | Gate the component that currently owns the mechanism. Preserve the invariant, but revise implementation-specific expected values when the reviewed boundary changes. |
| HSA-hook units | Default activation, option parsing, memory admission, report allocation, reader/replacement/executable lifetime, dispatch growth, report teardown, failure policy | Preserve observable lifecycle behavior; migrate mocked tests to runtime-coordinator inputs rather than retaining a monolithic hook. |
| Paired device contracts | Correct workload has exact output and no diagnostic; adjacent incorrect workload changes the relevant ordering/publication and produces the declared semantic signal | Primary behavioral oracle. Run on every applicable emulated target and periodically on physical gfx950. Do not weaken to make a refactor pass. |
| Target-specific device contracts | Real target-exclusive mechanism under the same correct/incorrect contract | Required for genuine ISA/ABI capability branches; typed not-applicable elsewhere. |
| Physical and E2E qualification | Runtime/memory properties, generated workloads, scale and pressure not faithfully reducible to a checked-in fixture | Qualification gate at proportionate checkpoints, not the inner loop for every small cutover. Extract a quick regression whenever an investigation finds a missing contract. |

The workload names used in the coverage summary below describe test provenance,
meaning the real workload idiom
that motivated a small checked-in contract:

- **Framework lifecycle** covers load, launch, unload, and repeated/concurrent
  use. **Graph replay** means replaying a previously captured launch graph.
- **Subword aliasing** means byte or halfword accesses overlap inside a larger
  word. **Mixed owners** means shared helper code is reachable from kernels with
  different resource or ownership facts.
- **Stream-K** is a matrix-multiplication scheduling style in which workgroups
  publish and consume partial progress. **GEMM** means general matrix-matrix
  multiplication. **Matrix staging** moves tiles through LDS before matrix
  instructions consume them.
- **Top-K** selects the largest or smallest K values. **MoE** means mixture of
  experts, a model structure that routes items among expert computations. An
  **optimizer** updates model parameters. **Continuous batching** adds/removes
  requests while a serving loop continues.
- **RCCL** is AMD's collective communication library. RCCL-style partial
  barriers model synchronization shapes seen in collective kernels without
  checking a large RCCL binary into this suite.

These names do not create product-specific ConSan branches. They identify
control-flow, resource-pressure, access, and synchronization idioms that the
implementation must continue to handle.

The device suite's common workloads cover framework lifecycle, graph replay,
dynamic stack and mixed owners, subword aliasing, barriers, atomics/fences,
Stream-K-style publication, GEMM/matrix staging, sort/Top-K, optimizer/MoE,
continuous batching, RCCL-style partial barriers, and large/heterogeneous code
objects. Target extensions cover CDNA MFMA/direct-to-LDS, CDNA4 transpose and
large LDS, RDNA WMMA/FP8, and gfx1250 cluster/high-bank plus
TDM/asynchronous/SWMMAC workload contexts. This provenance is why the paired tier is
more valuable than a rectangular count alone.

### 8.2 Contract map by component

“Regression-10622 activation tests” below means the focused tests originally
added for the reported hook-activation/lifecycle behavior: merely loading the
tool must retain the supported default while unrelated hook use does not acquire
unintended side effects. An **all-five-target capability projection** applies
the same semantic policy inputs to each target profile and compares the typed
disposition. A **wrong-engine snapshot** carries a schema/engine tag different
from the reader selected to decode it and must be rejected.

| Component | Existing evidence to retain | Focused tests required at its new boundary |
| --- | --- | --- |
| Request/configuration | HSA hook configuration and regression-10622 activation tests | construct `ConSanRequest` directly; table-test defaults, invalid combinations, and separation from mutation/debug options |
| Target profile/classifier | analysis, capability manifest, engine conformance and target-specific host tests | one complete profile row per target; semantic-form goldens per distinct encoding; wave-size and ABI fixtures |
| Runtime capabilities | HSA memory-pool admission, report allocation and simulator/runtime fixtures | typed physical/simulator capability fixtures; reject missing coherence, atomic, allocation and binding properties before lowering |
| Program inventory | analysis, synchronization, fault, ordinary, relay/provenance tests | immutable inventory snapshots by semantic ID; physical alias, multi-range, owner, confidence, sequence and address-recipe tests |
| Engine policy | MOI common, RR, Sampled, Inline, SuperCollider and conformance tests | pure site-decision matrices independent of registers and bytes; one all-five-target capability projection |
| Coverage/trust | site-disposition assertions, hook static coverage, diagnostic guards | exhaustive admission/lowering/runtime loss truth table; no-warning-string assertions |
| Evidence planner/decoder | report-plan, ABI/model and hook snapshot tests | pure requirements/layout bounds; stale/torn/generation/wrong-engine/malformed inputs |
| Resource planner | resource and per-engine pressure tests | typed request-to-plan tests for dead/grow/spill/reject, persistent/transient lifetime, dynamic stack, wave and target costs |
| Lowering/placement | current patch-kind, exact-word, spill, relay and final-validation tests | keep target byte goldens only at adapter; add target-neutral intent-to-effect tests and transactional multi-site tests |
| Runtime coordinator | HSA-hooks lifecycle, memory-ceiling, report registry, dispatch segment tests | explicit state-machine tests for allocation/bind/load/unload/reload/concurrency/rollback; simulator and HSA adapters share conformance cases |
| Engine analyzers/verdict | Record/Replay model, Sampled synchronization/model, Inline Shadow model, report summary tests | typed evidence traces for ordered/unordered, loss and incomplete cases; output schema separate |
| Mutation pipeline | fault and composition tests plus E2E fault manifests | pristine-ID selection, exactly-one, mutation-before-inventory, rollback and separate transformational status |

### 8.3 Tests that may entrench the prototype

To **entrench** the prototype is to make an accidental internal choice look
mandatory because a test asserts it outside the component that owns it.
`ConSanPatchKind` is the current enum identifying concrete emitted patch
categories. `scratch_vgpr` and `resolved_moi_*` are current fields recording
selected scratch registers and derived MOI choices. A trampoline offset is a
concrete placement location. A **raw word** assertion compares encoded machine-
instruction integers directly. These are legitimate lowerer/placement oracles
at their owning boundary, but usually not semantic policy oracles.

The current host tests contain roughly 1,150 `ConSanPatchKind` references,
1,050 `scratch_vgpr` references, 680 `resolved_moi_*` references, 440
trampoline-offset references, 350 resource-plan references, and hundreds of raw
word and explicit SGPR/VGPR assertions. These are not automatically bad: exact
encoding, state preservation, and placement tests are necessary. They become
prototype entrenchment when used to require a particular mechanism outside the
component that owns it.

Suspect assertions include:

- exact patch kind or patch count where several transparent placements satisfy
  the same intent;
- a particular scratch register, cave offset, relay topology, or plan-iteration
  order rather than legal non-overlap and state preservation;
- full raw-word sequences asserted in engine-policy tests rather than in the
  target lowerer;
- the current broad `ConSanOptions` mutation order, automatic-choice warning
  text, or fields of the giant `ConSanResult`;
- current report offsets or table multiplicities that are not a supported
  external ABI; and
- performance workarounds such as the present Sampled dispatch path presented
  as semantics.

The following are nonnegotiable behavioral contracts:

- exact application results and canaries, where a **canary** is a guard value
  whose unexpected change reveals corruption;
- correct/incorrect workload adjacency and the intended semantic difference;
- diagnostic class and attribution at the supported product boundary;
- no false diagnostic on the correct member;
- stable semantic site identity and supported target/form dispositions;
- guest-state transparency, report publication validity, honest coverage and
  incomplete verdicts; and
- externally supported configuration and output fields.

When a test blocks a slice, classify it before editing:

1. identify the assumption it asserts;
2. find the owning component and first-principles invariant;
3. decide whether the assertion is product behavior, that component's required
   mechanism, or accidental prototype shape;
4. add or strengthen the behavioral/contract assertion before removing the
   accidental one; and
5. make the test-contract change reviewable in the same slice.

### 8.4 Per-slice gates

Every slice runs:

1. the new component's focused host tests;
2. the pre-existing host tests for the moved responsibility;
3. the relevant paired device scenarios on all five applicable simulator
   targets, not only the architecture where the code was first noticed; and
4. broader host and device suites proportional to the affected seam.

The fast inner loop may omit serialized physical gfx950 runs, but the omission
is recorded and discharged periodically, after native-sensitive changes, and
at the implementation-week endpoint. E2E rows are selected when the slice
touches a behavior not faithfully represented in the checked-in tier. No
individual green row proves a cross-target or cross-engine boundary.

### 8.5 Documentation as a component contract

During Stage 2, this document changes with the architecture. Each landed
component gets a short design section or file beside the code or primary design
document it explains. It contains:

- purpose and non-responsibilities;
- input/output types and ownership;
- invariants and typed failures;
- target variation and why it is unavoidable;
- isolation tests and device contracts; and
- temporary adapter and its deletion issue, if any.

A deletion issue for a temporary adapter is a tracked obligation, not
permission to leave an unowned compatibility path indefinitely.

`DESIGN.md` and `FLAVORS.md` remain the product/algorithm documentation.
`CAPABILITIES.md` remains generated from the executable target/engine/form
contract. **Executable contract** means the same typed tables are consumed or
checked by code/tests and rendered into documentation, rather than maintaining
an independent prose claim. Implementation comments explain local proofs and ISA encodings, not
historical workload anecdotes. This keeps the code navigable for human
reviewers and future automated coding agents without requiring either to
reconstruct the architecture from `.inc` inclusion order.

## 9. Scoped implementation-week endpoint

### 9.1 Required state at the end of the week

The week establishes the production **control plane** and cuts production
callers over to it. A **production caller** is a path used by ordinary ConSan
execution, not only by a test or old/new comparison harness:

1. the HSA path constructs immutable `ConSanRequest`, `TransformPolicy`, and
   `RuntimePolicy` values; mutation and debug controls are separate;
2. all five targets resolve through one reviewed `TargetProfile` table and
   narrow classifier interface;
3. simulator and HSA backends expose one typed `RuntimeCapabilities` contract
   rather than letting target names stand in for memory or binding properties;
4. the transform builds one immutable `ProgramInventory` with stable physical
   and semantic identities;
5. all four modes express access applicability through engine policy and one
   coverage ledger; the three MOI modes likewise express barrier, atomic, and
   fence applicability there;
6. report sizing and buffer properties are emitted as address-free
   `EvidenceRequirements`, then bound explicitly by the current HSA runtime;
7. a small top-level pipeline passes typed stage outputs in one direction and
   returns a split `TransformResult`;
8. current probe/resource/placement code is reachable only through named
   compatibility operations that consume the new plan and report typed
   lowering outcomes; and
9. each new component has focused tests and the design documentation described
   above.

At that endpoint, engine policy, target support, coverage, and runtime
requirements are reviewable without reading patch emitters. DBI owners can see
exactly what framework inputs ConSan needs. Current lowerers may still use an
internal **compatibility projection**, meaning an adapter-produced legacy view
of the new typed data, but no new semantic policy is allowed
inside it and its remaining responsibilities are inventoried for later
component migrations. “Inventoried” means each retained responsibility and
consumer is explicitly listed rather than hidden under “legacy.”

### 9.2 Explicitly outside this one-week endpoint

- rewriting or relocating every existing probe body;
- replacing the current branch-only relay, cave, resource, or spill machinery;
- changing the current report ABI or user-visible diagnostic semantics;
- migrating to DBI probes, variants, streams, or the future common hook;
- optimizing Sampled's runtime selector;
- solving DBI's large-kernel relocation or all-target enablement;
- restructuring files solely to make the source tree look layered; and
- removing the legacy-lowering adapter before its individual access,
  synchronization, prologue, resource, and placement responsibilities have production
  replacements.

These exclusions prevent a names-and-files cleanup from concealing a second
flag-day rewrite. They do not weaken the final destination in section 5.

### 9.3 Quantitative review baseline

At the start and end of Stage 2 record:

- total ConSan patch and hook lines;
- fields in public request, internal compatibility options, inventory, plan,
  and result types;
- references to target IDs outside profile/classifier/lowerer code;
- engine-specific copies of site admission and coverage logic;
- direct hook dependencies on patch geometry; and
- remaining consumers of the compatibility mechanism projection.

The week succeeds by establishing dependency direction and deleting displaced
policy, not by moving the same lines behind new filenames. New type or adapter
growth must be offset by deletion or justified as a temporary measured seam.

## 10. Fully incremental migration sequence

The slices below are the implementation order. A slice is complete only when
its cutover and deletion are done. If its seam grows beyond the stated
boundary, split the slice before coding further.

Every slice uses the same fields:

- **Current responsibility** names exactly what the prototype owns before the
  slice, so the migration cannot quietly expand.
- **New boundary and contract** names the replacement responsibility and its
  typed input, output, invariants, and failure states.
- **Temporary seam and consumers** says how unmigrated code uses the new value,
  and lists the production/test code that must switch. Shape conversion is
  allowed; a seam may not make new semantic decisions.
- **Test gate** names the focused and behavioral evidence required before
  production use switches.
- **Cutover and deletion** names the exact switch and the prototype code or
  fields removed in the same slice. Deletion is how the design prevents a
  permanent old/new fork.
- **Prerequisite** names an earlier contract on which this slice is allowed to
  depend. It is not a vague scheduling preference.

The **contract-test inventory** maps each proposed component to the tests that
guard it. The generated **capability manifest** is the machine-checked
target/engine/form table from which `CAPABILITIES.md` is rendered. During a
temporary seam, old/new comparison asserts **semantic equality**: the same
stable sites, facts, and decisions even if container shape or field order
differs. A **field-inventory test** fails if an adapter acquires an unmapped
field. An
**inventory fingerprint check** proves that an early sizing pass and a later
bound lowering pass refer to identical semantic input. A **named barrier** is a
target barrier using an explicit hardware barrier identifier rather than the
ordinary whole-workgroup barrier. A **dual-comparison path** temporarily runs
old and new derivations for equivalence; it must be deleted after cutover.
The current **ABI retry** is the two-pass behavior in which an early transform
derives a report layout and a later transform repeats lowering with a concrete
buffer. An **orchestrator** is the small host function that invokes the stages
in dependency order and passes each typed result to its declared consumers. A
**stage-order test** proves that the orchestrator invokes stages in
the declared dependency order and passes matching fingerprints.
**Design reconciliation** compares the implemented endpoint with this document
and resolves any temporary deviation instead of silently redefining the design.

### Slice 0: freeze the reviewed baseline

- **Current responsibility:** Behavior is distributed across host tests,
  device pairs, E2E ledgers, and warning/patch-shape assertions.
- **New boundary and contract:** Add a small contract-test inventory naming the
  host and paired-device gates for each component and record the initial
  quantitative baseline from section 9.3. Correct any discovered test that
  lacks a behavioral oracle before relying on it.
- **Temporary seam and consumers:** None; this slice changes tests/docs only.
- **Test gate:** Complete ConSan host suite, all five simulator device targets,
  and physical gfx950. Record any E2E gate that cannot be run in the timebox.
- **Cutover and deletion:** Baseline becomes the reviewed Stage 2 reference;
  delete redundant tests only when their unique assertion is named elsewhere.
- **Prerequisite:** Reviewed Stage 1 document.

The frozen Slice 0 artifact is
[PRODUCTION_BASELINE.md](PRODUCTION_BASELINE.md).

### Slice 1: centralize `TargetProfile`

**Implementation status (2026-08-25): complete.** The immutable five-target
table and its pure code-object/per-kernel lookups live in
`consan_capability_contract.h`. Target admission, semantic-form availability,
wave and EXEC rules, descriptor allocation and accumulator models, dispatch
and workgroup identity facilities, scratch and group limits, branch/call
facilities, wait families, and bank/cluster facilities now share that source.
The profile does not select an engine or encode instructions. ConSan callers
no longer use `instrumentation_builder.h`'s encoding-routing predicates as
product admission policy; the generic builder keeps those predicates for its
non-ConSan consumers.

The dedicated `capability_contract_test.cpp` suite owns the five-row host
matrix and focused tests for profile validation, lookup and derivation, enum
iteration/naming/domain mapping, masks, predicates, boundary arithmetic,
disposition matrices, and invalid typed inputs. It also cross-checks generic
scratch encodings and branch/call boundaries. The generated capability
manifest, the complete ConSan host suite, and a correct/incorrect simulator
pair on each supported target form the completed cutover gate. The deletion
audit has one profile table and no remaining ConSan references to the generic
admission/family predicates or duplicated `CDNA3 || CDNA4` / `RDNA4 || CDNA5`
family tests; exact single-target branches remain in classifiers and lowerers.

- **Current responsibility:** Architecture predicates and constants are spread
  through analysis, resource planning, prologues, access/synchronization lowerers,
  validation, and `instrumentation_builder.h`, the current file that routes
  calls to target-specific instruction builders.
- **New boundary and contract:** Introduce the immutable profile and pure lookup
  for all five targets. Move only factual predicates first: admitted target,
  capability forms, wave/EXEC rules, allocation granularity, AccVGPR model,
  preload/identity availability, scratch/group limits, branch/call and bank/
  cluster facilities.
- **Temporary seam and consumers:** Existing helpers delegate to the profile;
  emitters still receive the current `arch` target-enum parameter and continue
  using target builders.
- **Test gate:** Five-row profile matrix, generated capability-manifest check,
  existing architecture analysis/resource tests, and one correct/incorrect
  device pair per target.
- **Cutover and deletion:** Switch one factual helper family at a time, then
  delete its duplicate predicates. No engine policy moves in this slice.
- **Prerequisite:** Slice 0.

### Slice 2: split immutable requests and runtime facts from implementation state

**Implementation status (2026-08-25): complete.** The documented request,
transform, runtime, debug, mutation, capability-requirement, and bound-resource
contracts live in `consan_request_contract.h.inc`. The environment parser now
constructs those contracts as the public base values of `HookConfig` and runs
their deterministic cross-field validation once. Production code receives the
contracts separately and by const reference after construction; the two
values that genuinely change during a load, `MutationRequest` and
`BoundRuntimeResources`, are copied into explicitly named working values.

The HSA adapter now performs one region walk per load to produce
`RuntimeCapabilities`. Automatic SuperCollider and MOI report paths state and
validate their capability requirements before allocation, and the typed
maximum workgroup-LDS fact replaces the former separate group-region query.
The production transformer entry accepts the separated contracts. The
temporary `ConSanOptions` compatibility state constructs itself directly from
those contracts only at the remaining prototype lowering, retry, and inventory
seams; its raw transform overload is internal to focused mechanism tests. The
former production hook block that manually copied approximately ninety policy,
mutation, register, and report fields and the later standalone compatibility
adapter have both been deleted. The hook-test transform override now receives
the same separated typed inputs as production, so inventory and retry tests no
longer force the hook to reconstruct a `ConSanOptions` value. Its raw return is
only a synthetic lowerer fixture and is immediately published into the typed
result before loader policy can observe it.

The focused `request_contract_test.cpp` host suite covers defaults, value
semantics, all validation and sentinel branches, mode conflicts, sampling and
register boundaries, every mutation predicate, pristine mutation projection,
both physical and simulator capability fixtures, every missing required
runtime fact, all resource lifetimes, issue naming, fresh legacy adapter input,
and typed-entry equivalence. The hook suite retains load-only Record/Replay
activation, fail-open/fail-closed, parser, allocation, and lifecycle coverage.
The completed cutover gate is the full ConSan host suite, all HSA-hook units,
the capability-manifest check, all 2,908 simulator device contracts on all
five targets, and a physical gfx950 correct/incorrect pair for each of the four
engines.

- **Current responsibility:** `ConSanOptions` mixes product configuration,
  mutation/debug controls, late-bound report data, and derived register/
  placement state. HSA/simulator capability facts are queried or inferred at
  their individual use sites, sometimes through target assumptions.
- **New boundary and contract:** Add `ConSanRequest`, `TransformPolicy`,
  `RuntimePolicy`, `ConSanDebugOverrides`, `MutationRequest`, and
  `BoundRuntimeResources`. Add `RuntimeCapabilities` as the typed output of a
  narrow HSA/simulator query adapter. The configuration parser constructs the
  request/policy values and validates cross-field policy once; it does not
  manufacture runtime capabilities.
- **Current compatibility seam and consumers:** `ConSanOptions` constructs one
  fresh mutable lowerer state from the separated typed values only at the raw
  lowering call. Its old direct overload remains internal for focused mechanism
  tests. Production resource/report consumers use the typed runtime and binding
  subobjects rather than parallel projections.
- **Test gate:** Direct request parsing/default/conflict tests, especially
  load-only Record/Replay activation and fail-open/closed separation; existing
  hook configuration tests; physical/simulator capability fixtures and missing-
  property rejection; smoke paired device rows for all four modes.
- **Cutover and deletion:** Switch the production HSA caller to the new request.
  Switch the moved report/resource decisions to `RuntimeCapabilities`. Delete
  hook-side field-by-field policy duplication and the corresponding repeated
  agent queries; remove mutation, report-address, and test fields from every
  new component signature. The adapter remains, with a field inventory test
  preventing new additions.
- **Prerequisite:** Slice 1 for target validation.

### Slice 3A: stable code-object and access inventory

**Implementation status:** complete. `ProgramInventory` now owns the decoded
text-section, kernel, function, and normalized access inventory in shared const
storage. `ConSanResult` owns that inventory, and every consumer reads its
immutable container accessors directly; the former duplicate mutable
containers and inherited compatibility spans have been deleted. Access
candidate discovery and SuperCollider's coverage denominator consume
normalized inventory without rediscovering semantics.

`ConSanCodeObjectId` retains the existing `fnv1a64:<hex>` diagnostic spelling
but combines it with byte size and an independently seeded reverse digest for
identity equality. Consequently, a collision in the public fingerprint cannot
silently merge physical or semantic site identities. `PhysicalSiteId` combines
that content identity with original text offset. Symbol aliases therefore
share a physical identity, while `ConSanProgramContainerRef` remains separate
attribution evidence. Each normalized logical access range receives a
`SemanticSiteId` with an explicit range ordinal, including the two ranges of
two-address LDS instructions.

The focused `program_inventory_test.cpp` host suite exhaustively covers every
new enum/name contract, identity validity and collision handling, immutable
copy/move lifetime, builder facts and const container access, native LDS,
direct-to-LDS, subword and every supported two-address spelling, all FLAT
provenance mappings and raw operands, every typed inventory exclusion,
physical alias identity, and equivalence with a real decoded code object.
Existing alias-failure tests also guard the normalized SuperCollider coverage
path. The slice gate includes all ConSan host and hook tests, the capability
manifest, and representative paired access device contracts on all five
simulated targets.

- **Previous responsibility:** Kernel/function metadata, LDS/FLAT sites,
  candidate ranges, pointer provenance, owner references, and patch state share
  `ConSanResult`.
- **Implemented boundary and contract:** `ProgramInventory` owns
  code-object/container facts, `PhysicalSiteId`, access `SemanticSiteId`,
  normalized access ranges, provenance/confidence, and typed exclusions.
  Construction remains backed by the current decoder and provenance analysis.
- **Retired temporary seam:** The read-only `LegacyInventoryView` projection
  was deleted in Slice 4F. Candidate, resource, lowering, hook, and test
  consumers now read the owning `ProgramInventory` directly.
- **Test gate:** Analysis/provenance/physical-alias/two-address/subword tests,
  immutable-ID tests, capability conformance, common access device pairs on all
  five targets, and target-exclusive access pairs.
- **Completed cutover and deletion:** Access candidate discovery and
  SuperCollider coverage use the new inventory. Duplicate access/container
  vector ownership, mutation, and compatibility projection have been deleted
  from `ConSanResult`.
- **Prerequisite:** Slices 1 and 2.

### Slice 3B: synchronization and ownership inventory

**Implementation status:** complete. `ProgramInventory` now owns sync events,
logical sequences, owner annotations, barrier lifecycles, ordinary
release/acquire associations, and MOI fence candidates. Its
`SynchronizationInventoryView` is an immutable query value;
`ConSanResult` neither inherits nor stores a second copy of those spans. Only
semantic analysis receives `SynchronizationInventoryBuildView`, making every
remaining write capability explicit in a producer signature.

Each decoded event has a content-qualified `SemanticSiteId`. Every sequence
and lifecycle group preserves its ordered typed member IDs in parallel with a
temporary string adapter for lowerers and diagnostics. Sequence association
merges the typed IDs for barrier pairs, atomic/cache patterns, ordinary
release/acquire patterns, and gfx1250 barrier lifecycles. Owner annotation and
all derived recipes are completed before the immutable view is published.
Fault and engine code consume that view read-only.

`ProgramInventoryBuilder` can create a deep-copied analysis revision from a
published inventory. The original storage remains immutable, making bounded
component reanalysis and adversarial corruption tests explicit rather than
letting callers mutate a published result. Focused host tests prove const view
types, empty and populated projections, copy/move lifetime, complete projection
of all synchronization-derived record kinds, revision isolation, and stable
event/sequence/lifecycle IDs on a real gfx1250 code object. The completed gate
also passed all 1,232 ConSan host tests, 85 ConSan hook tests, 564 paired
barrier/atomic/fence/Stream-K/cluster simulator rows across all five targets,
and 102 corresponding physical-gfx950 rows. Sampled D128 E2E validation stayed
complete on physical gfx950 (128/128 accesses and 117/117 barriers) and
RocJitsu-emulated gfx1250 (18/18 accesses and 8/8 barriers).

- **Previous responsibility:** Sync events, sequences, owner annotation,
  barrier lifecycles, ordinary release/acquire association, address recipes,
  and fence candidates were interleaved with fault and engine setup in the
  mutable result.
- **Implemented boundary and contract:** `ProgramInventory` owns immutable typed
  synchronization/ownership data and stable member identities. Semantic
  inventory construction is independent of the selected downstream MOI
  engine; optional CFG products remain explicit requests.
- **Current temporary seam and consumers:** Current lowerers and mutation
  planners read `SynchronizationInventoryView`. String event identities remain
  only as a compatibility projection adjacent to authoritative typed member
  identities. Fault sites and mutation plans remain separate result-owned
  inventories and cannot modify sanitizer semantics.
- **Test gate:** Sync/ordinary/fault inventory tests, owner and shared-helper
  tests, barrier/atomic/fence device pairs on every applicable target, plus
  gfx1250 cluster sequence tests.
- **Completed cutover and deletion:** Fault planning and current engine
  consumers read the immutable projection. Duplicate semantic synchronization
  vector ownership and option-dependent reset paths have been deleted from
  `ConSanResult`.
- **Prerequisite:** Slice 3A.

### Slice 4A: access policy and coverage ledger

**Implementation status:** complete. The pure
`plan_consan_access_observation` boundary consumes immutable access inventory
and produces one explicit `ConSanSiteDecision` per semantic access range plus
one coalesced `ConSanProbeIntent` per physical instruction. It handles all four
engines, native LDS, direct-to-LDS, group FLAT, relaxed LDS atomic access,
target capability, provenance confidence, container filters, synchronization
reservation, inventory exclusions, and physical aliases without mutating its
inputs. Equivalent aliases share one intent while retaining every source name;
conflicting aliases fail closed with a typed reason.

`ConSanCoverageLedger` owns copies of the plan's immutable decisions and
intents and records a typed lowering outcome for every admitted intent. MOI
candidate discovery is now restricted to admitted physical intents, and both
MOI and SuperCollider project their current resource and placement outcomes
back into the ledger. SuperCollider's physical coverage list remains a
temporary adapter. Both MOI access and synchronization site-disposition lists
have been deleted: admitted candidates come only from typed decisions and
intents, final lowering outcomes are published directly to those intents, and
runtime coverage reads the pipeline's typed ledger after the compatibility
result has been moved aside.

The focused `observation_plan_test.cpp` suite covers every enum and stable
name, invalid sentinels, plan-local identities, construction invariants,
lookup, deep-copy lifetime, ledger transitions, physical-outcome adaptation,
all four engine mappings, multi-range coalescing, every typed inventory
limitation, capability and mnemonic rejection, family/provenance/filter/sync
exclusions, relaxed atomics, alias agreement and conflict, determinism, and
input immutability. Engine integration tests additionally require a real
SuperCollider patch and current MOI lowerers to publish their final typed
outcomes.

- **Previous responsibility:** SuperCollider and each MOI engine independently
  filtered access candidates, appended dispositions, later inferred success
  from patch kinds, and rendered warnings.
- **Implemented boundary and contract:** One pure engine-policy function owns
  access applicability and target-neutral intent admission for all four modes.
  One result-owned ledger joins those decisions to typed mechanism outcomes.
- **Current temporary seam and consumers:** Current candidate, resource, and
  placement implementations still consume legacy shapes translated from the
  admitted intents. Their results are translated back into
  `ConSanLoweringOutcomeKind`; only the physical SuperCollider coverage list
  remains as a compatibility projection.
- **Test gate:** Pure four-engine policy matrices, physical-site coalescing,
  the complete common paired scenario set on all simulator targets,
  wide/group-FLAT and target-specific access contracts, then physical gfx950
  access pairs.
- **Completed cutover and deletion:** Native LDS, direct-to-LDS, group FLAT,
  multi-range, and relaxed-atomic access applicability now enter every engine
  through the common plan. Duplicate per-container physical dispositions were
  collapsed. Exact target lowering classifiers, candidate payloads, and legacy
  coverage renderers remain until the resource/placement and evidence slices
  can replace them without a global switch.
- **Prerequisite:** Slice 3A.

The completed gate passed 1,414 ConSan host tests (plus two intentionally
skipped external-object benchmarks), 98 ConSan hook/runtime tests, all 2,908
simulator device rows, and 142 access-focused physical-gfx950 rows. The
`d128-block` E2E row remained accepted in all four engines on physical gfx950.
RocJitsu-emulated gfx1250 remained accepted in SuperCollider, Sampled, and
Inline Shadow with complete static coverage; Record/Replay retained its
already-published yellow status, with 18/18 accesses and 8/8 barriers statically
covered but incomplete bounded runtime metadata.

### Slice 4B: barrier policy

- **Completed boundary and contract:** `plan_consan_barrier_observation` is the
  single pure admission boundary for all engines. It consumes immutable shared
  synchronization events and sequences, returns a barrier-only plan fragment,
  and expresses disabled tracking, mutation-only SuperCollider behavior,
  container/runtime exclusions, target scope support, malformed encodings,
  ambiguous or incomplete sequences, adjacent redundant full barriers, and
  physical-alias contradictions as typed decisions. Record/Replay receives one
  event-record intent per admitted instruction. Sampled and Inline receive one
  completion intent covering every semantic event in a qualified signal/wait
  sequence.
- **Completed types and composition:** `ConSanBarrierPolicyReason`,
  `ConSanBarrierSiteDecision`, `ConSanBarrierPolicyRequest`, and
  `ConSanBarrierPolicyResult` document the barrier contract locally.
  `ConSanObservationPlan::append` transactionally rebases plan-local intent IDs,
  allowing independently testable access and barrier policy fragments to form
  one immutable plan. `ConSanCoverageLedger` now owns both decision families.
- **Completed temporary seam:** MOI and SuperCollider append barrier policy to
  their access plan before constructing the ledger. Current MOI resource and
  lowering candidates may only narrow physical insertion sites admitted by
  barrier intents. Final resource/placement results publish directly through
  the intent ledger; the hook's stable machine-readable diagnostic schema is
  rendered from typed decisions, lowering entries, and resource plans rather
  than from a second internal record.
- **Completed test gate:** Focused units cover every enum spelling, request and
  result validity, plan composition and rollback, ledger ownership, all four
  engine contracts, paired-sequence coalescing, disabled/filter/runtime
  exclusions, workgroup and gfx1250 cluster capability boundaries, malformed
  encodings, redundant barriers, every Sampled qualification fact, incomplete
  and ambiguous sequences, aliases, determinism, and immutability. Existing
  integration tests now assert typed Record/Replay, Sampled, and Inline plans
  and lowering outcomes. The checked-in device gate retains its paired
  ordered-tile, double-buffer, RCCL, lifecycle, Stream-K, and gfx1250 cluster
  coverage on every applicable target.
- **Completed cutover and retained compatibility seam:** Workgroup, named, and
  cluster barrier admission no longer originates in engine-local disposition
  scans. The current lowerers still translate admitted intents through their
  operand-rich candidate structures and reuse the shared Sampled qualification
  predicate; those structures remain until the resource/placement slices can
  replace them component by component.
- **Prerequisite:** Slices 3B and 4A.

The completed host gate passed 1,428 ConSan tests plus two intentionally
skipped external-object benchmarks, and all 98 ConSan hook/runtime tests. All
2,908 emulator device rows passed in 87.38 seconds of wall time. The
physical-gfx950 barrier and ordering subset passed all 132 rows. The physical
gfx950 `d128-block` E2E row was accepted for all four engines, with complete
barrier coverage (Record/Replay and Inline 119/119; Sampled 117/117). The
RocJitsu-emulated gfx1250 row accepted baseline, SuperCollider, Sampled, and
Inline. Its Record/Replay result retained the pre-existing yellow condition:
static coverage was complete (18/18 accesses and 8/8 barriers), while the
bounded runtime report remained incomplete with `metadata-full`; this slice did
not regress or conceal that limitation.

### Slice 4C: atomic and fence policy

- **Completed boundary and contract:** A pure atomic/fence policy now consumes
  immutable synchronization inventory and creates typed before/after intents,
  dynamic-result requirements, stable association identities, and per-engine
  capability dispositions. `ConSanAtomicPolicyReason`,
  `ConSanFencePolicyReason`, `ConSanFenceAssociation`, and
  `ConSanDynamicResultRequirement` retain every exclusion as machine-readable
  evidence instead of making warning strings part of control flow.
- **Completed cutover:** Ordered native atomics and ordinary-memory/fence
  sequences are qualified once by policy. Candidate retention and the coverage
  ledger consume the published decisions; they no longer rediscover
  eligibility in each engine. The policy uses synchronization-
  derived scope for ordinary sequences, admits the exact gfx1250 buffer form
  supported by Record/Replay lowering, and distinguishes guest synchronization
  from barriers introduced by ConSan itself.
- **Retained compatibility seam:** Current Record/Replay, Sampled, and Inline
  lowerers still translate admitted intents through operand-rich candidate
  structures. Those structures continue to own register allocation,
  instruction placement, and target words until the resource and lowering
  slices replace them component by component.
- **Completed test gate:** Focused type units exhaustively check enum sets and
  stable names, association identity validity and ordering, candidate-derived
  predicates, decision/intent invariants, append/rebase behavior, typed error
  outcomes, target matrices, dynamic-result mappings, exact encoding and
  operand boundaries, ambiguity, and deterministic non-mutating policy.
  Checked-in device contracts retain compare-and-swap, atomic store, FLAT
  atomic, Stream-K, tree-reduction atomic OR, target global-memory/cache,
  fence-publication, and gfx1250 ordered-LDS coverage.
- **Prerequisite:** Slices 3B and 4A.

### Slice 4D: delete the MOI site-disposition compatibility ledger

- **Completed deletion:** `ConSanSiteDispositionRecord`, its disposition and
  lowering enums, all four stringifiers, the `ConSanResult` field, and the MOI
  synchronization producer/adapter pass have been deleted. No production C++
  control flow reads or writes the retired vocabulary.
- **Single sources of truth:** `ConSanObservationPlan` owns semantic admission;
  `ConSanCoverageLedger` owns per-intent lowering; resource plans own allocation
  failures. The HSA hook keeps its external `coverage_site` log schema as a
  rendering adapter over those typed values, so tooling compatibility does not
  require duplicate internal state.
- **Alias and aggregate contracts:** synchronization canonicalization retains
  every equivalent source-container name. A multi-intent decision is complete
  only when all of its intents are instrumented; resource or placement loss
  remains visible when another intent at the same semantic site succeeded.
- **Test gate:** focused policy/retry/partial-lowering units, the complete
  ConSan host suite, all HSA hook tests, and the complete simulator and physical
  device matrices.

The completed host gate passed 1,505 of 1,507 ConSan tests, with the remaining
two external-object benchmarks intentionally skipped, plus all 197 ConSan
hook/runtime tests and all 371 CPU-only validation-protocol tests. All 2,908
emulator device rows passed in 71.74 seconds of wall time. The complete 593-row
physical-gfx950 device matrix passed in 426.82 seconds of wall time.

### Slice 4E: delete the SuperCollider access-coverage projection

- **Completed deletion:** `ConSanScAccessCoverageKind`,
  `ConSanScAccessCoverageSite`, `sc_access_coverage_resolved`, and the duplicate
  physical-site vector have been deleted. SuperCollider lowerers now publish
  resource rejection directly to the shared coverage ledger, and final patch
  publication changes remaining admitted intents to instrumented or placement
  rejected without reconstructing support from a second representation.
- **Shared hook contract:** Static coverage for all four engines is aggregated
  from the same typed policy decisions and lowering entries. SuperCollider
  require-patch policy treats pending and placement-rejected admitted intents
  as required, excludes resource-rejected intents, fails closed on an invalid
  or missing plan, and accepts a valid plan with no applicable access sites.
- **Regression discovered by the device gate:** The first implementation
  conflated an invalid plan with a valid empty plan, causing runtime-only code
  objects to be rejected. The hook unit test now distinguishes admitted,
  placement-rejected, resource-rejected, invalid, and valid-empty states. The
  complete simulator matrix caught the omission before the slice was committed.
- **Deletion result:** Production loses 102 net lines. Tests stop constructing
  the private projection and instead assert target-neutral admission and typed
  lowering outcomes, leaving the complete tranche 155 lines smaller including
  its focused test changes and this documentation.
- **Completed gate:** 1,505 of 1,507 ConSan host tests passed, with the two
  external-object benchmarks intentionally skipped; all 194 current HSA-hook
  tests and all 371 CPU-only validation-protocol tests passed; all 2,908
  simulator device rows passed in 68.64 seconds of wall time; and the complete
  593-row physical-gfx950 device matrix passed in 444.01 seconds of wall time.

### Slice 4F: delete the legacy container-inventory projection

- **Completed deletion:** `LegacyInventoryView`, its three borrowed fields,
  `ProgramInventory::legacy_view()`, `ConSanResult` inheritance from the view,
  and the associated binding step have been deleted. Production and test
  consumers now name the owning `ProgramInventory` explicitly and read its
  immutable text-section, kernel, and function ranges.
- **Storage-independent contract:** The new accessors return const spans rather
  than references to the backing vectors. Consumers can iterate and index the
  immutable inventory without learning its storage type or acquiring mutation
  authority. This keeps a later storage reorganization local to the inventory
  component.
- **Regression discovered by the device gate:** An intermediate accessor
  returned a vector reference. Two policy consumers copied that vector into an
  `auto` local while retaining pointers into it, so the pointers became
  dangling when the local died. The checked-in gfx1250 full-bank Stream-K
  correct/incorrect pair exposed the resulting loss of atomic/fence ordering
  evidence. Returning spans restores the borrowing contract, and both paired
  workloads pass.
- **Deletion accounting:** The slice removes the compatibility type and its
  binding path, but spelling the real owner at every consumer adds 30 net
  production lines after formatting. This is an intentional, bounded cost:
  the source now has one owner and one immutable container API instead of a
  shorter inherited alias whose lifetime and authority were implicit.
- **Completed checked-in gate:** all 1,519 selected ConSan host tests and all
  86 focused HSA-hook tests passed. In the 2,908-row simulator matrix, 2,906
  rows passed under 64-way load; the correct/incorrect gfx950 large-LDS
  InlineShadow pair timed out, then both passed together in 57.86 seconds when
  rerun without unrelated contention. In the 593-row physical-gfx950 matrix,
  592 rows passed in the aggregate 501.60-second run; the sole timed-out
  Sampled independent-scalar-proofs row then passed alone in 1.21 seconds.

### Slice 4G: delete the synchronization-inventory result projection

- **Completed deletion:** `ConSanResult` no longer inherits
  `SynchronizationInventoryView`, copies five borrowed spans, or offers an
  install-and-rebind method. Inventory producers assign one immutable
  `ProgramInventory`; consumers request its synchronization view explicitly.
- **Retained clean boundary:** `SynchronizationInventoryView` remains the
  grouped, read-only input to barrier and atomic/fence policy. It has no owner
  state and cannot outlive the `ProgramInventory` storage it describes.
  `SynchronizationInventoryBuildView` remains the distinct analysis-only
  mutation capability.
- **Focused contract:** the inventory unit suite now asserts that
  `ConSanResult` is not derived from the view while continuing to prove const
  element types, complete projection, copy/move lifetime, and builder-revision
  isolation. This guards against reintroducing convenient duplicate result
  state.
- **Deletion result and immediate payoff:** production grows by nine formatted
  lines because consumers now name the real inventory and a few large
  functions retain a local immutable view. The paired follow-up is completed
  in Slice 4H, which deletes the roughly 185-line pre-plan auto-report
  inventory reconstruction that rescanned these same synchronization facts.
  No additional production adapter was needed. More importantly, the code has one
  synchronization owner, one mutation capability, and one immutable policy
  projection; result copying can no longer carry stale spans that require
  rebinding.
- **Completed checked-in gate:** all 1,519 selected ConSan host tests, all 86
  focused HSA-hook tests, all 2,908 simulator device rows, and all 593
  physical-gfx950 device rows passed. The final simulator and physical matrices
  took 75.32 and 441.91 seconds of wall time, respectively.

### Slice 4H: delete pre-plan auto-report inventory reconstruction

- **Completed deletion:** `inventory_consan_moi_auto_report` and its roughly
  185-line reconstruction of access, barrier, atomic, fence, owner-bank,
  Sampled, and InlineShadow capacities from prototype candidates, resource
  plans, patches, synchronization facts, and original object bytes have been
  deleted. The HSA runtime coordinator now binds the engine-specific
  `ConSanEvidenceRequirements` already published by `TransformResult`.
- **One authoritative path:** the hook neither reruns an evidence planner nor
  guesses requirements when an observation plan is absent. Record/Replay,
  Sampled, and InlineShadow expose their already-validated sizing inventory,
  ABI plan, and runtime requirements through the closed evidence variant. A
  malformed or missing MOI evidence contract is rejected under fail-closed;
  fail-open installs the pristine object without allocating a report.
- **Test cutover and guard:** mechanism tests call the production typed
  planners through a narrow test-only dispatcher that reads only immutable
  inventory and observation policy. Synthetic allocation/retry hook fixtures
  now carry valid engine-matching observation plans. Tests whose sole purpose
  was to preserve the deleted compatibility adapter are gone. A focused hook
  regression test supplies contradictory legacy candidates and patches with no
  observation plan and proves that they cannot resurrect automatic allocation
  or instrumentation.
- **Live mutation invariant:** pristine and live fault transforms compare the
  sizing inventories from their respective typed evidence contracts. The
  growth test now changes the live semantic plan rather than mutating
  post-lowering telemetry, so it guards the actual capacity contract.
- **Deletion result:** the formatted production source is 192 net lines
  smaller. Source plus tests is 146 net lines smaller; the test-only growth is
  the explicit hook contract guard and typed fixture setup. Including this
  expanded design record, the complete tranche remains 113 net lines smaller.
- **Completed checked-in gate:** 1,503 of 1,505 selected ConSan host tests
  passed, with two external-object benchmarks intentionally skipped, plus all
  100 focused HSA-hook tests. All 2,908 simulator device rows passed in 72.88
  seconds of wall time, and all 593 physical-gfx950 device rows passed in
  439.14 seconds. E2E validation is intentionally outside this deletion
  tranche.

### Slice 4I: delete legacy report-need reconstruction

- **Completed deletion:** `moi_inventory_needs_report_buffer` and
  `sc_inventory_needs_report_buffer` are gone. The hook no longer infers a
  report or marker requirement from prototype candidates, synchronization
  sites, or emitted patch kinds, and it no longer reruns the SuperCollider
  evidence planner. All four engines consume the evidence requirement already
  published by `TransformResult`.
- **Semantic binding contract:** the shared MOI sizing inventory now states
  whether any admitted access, barrier, atomic, or fence observation exists.
  Record/Replay, Sampled, and InlineShadow require a concrete binding only
  when that semantic inventory is nonempty and its ABI plan is complete. Thus
  a valid empty observation plan may retain a header-only ABI description
  without allocating a useless report; an incomplete plan with real
  observations remains a visible capacity failure rather than looking empty.
- **Fail-safe cutover:** a modified result with missing, malformed, or
  wrong-engine typed evidence cannot allocate from legacy telemetry or install
  unbound instrumentation. Fail-open loads the pristine object and fail-closed
  rejects it. Non-installable SuperCollider outcomes bypass marker allocation
  and retain their normal typed loader policy.
- **Engine isolation:** ignored report settings remain available to the
  configuration diagnostics but are projected out before runtime binding. An
  MOI buffer can therefore no longer make a SuperCollider transform appear
  bound to the wrong resource kind, and conversely an SC marker cannot bind an
  MOI requirement.
- **Focused contract gate:** unit tests cover every MOI semantic count,
  complete and capacity-limited requirements, valid empty plans for all four
  engines, malformed requirements, and pipeline binding-stage outcomes. Hook
  tests leave contradictory legacy patches/candidates in place and prove that
  missing evidence cannot allocate or install, valid empty evidence allocates
  nothing, foreign-engine resource settings are ignored, and both fail-open
  and fail-closed behavior remain explicit.
- **Deletion result:** the hook and pipeline shrink by 26 net production lines;
  the shared typed evidence API grows by 23 lines, leaving production three
  net lines smaller. Source plus the new focused regression contracts grows by
  181 net lines. The test growth is deliberate: it replaces implicit behavior
  of the deleted scans with independently falsifiable typed and lifecycle
  contracts.
- **Completed checked-in gate:** 1,504 of 1,506 selected ConSan host tests
  passed, with two external-object benchmarks intentionally skipped, plus all
  102 focused HSA-hook tests. All 2,908 simulator device rows passed in 69.80
  seconds of wall time. The immediately preceding Slice 4H physical gate
  passed all 593 gfx950 rows; the next periodic physical gate remains due
  after the following deletion tranche. E2E validation remains outside this
  work.

### Slice 4J: make semantic-architecture admission an inventory invariant

- **Completed deletion and ownership cutover:** The free
  `consan_result_has_resolved_semantic_arch` compatibility helper is gone.
  `ProgramInventory::has_resolved_semantic_arch()` now owns the invariant
  because the inventory itself records both whether semantic analysis was
  attempted and which architecture supplied those semantics. Both pristine
  inventory admission and final transform admission query that typed owner;
  they no longer infer an analysis-stage invariant from the mutable mechanism
  result surrounding it.
- **Explicit pass-through behavior:** A parse-only unsupported-target
  inventory does not require a semantic architecture and remains safe to pass
  through unchanged. Once an architecture-dependent stage marks the
  requirement, an invalid architecture is rejected, and assigning a concrete
  semantic architecture resolves the invariant. The direct inventory test and
  both hook paths guard all three states.
- **Typed HSA fixture repair:** The complete hook gate exposed one stale
  synthetic test fixture left by Slice 4I. Its InlineShadow case had reused a
  Record/Replay plan without the matching immutable LDS inventory. The fixture
  now supplies an engine-correct exact-shadow intent and access inventory.
  The same test now consistently proves that Record/Replay, Sampled, and
  InlineShadow prefer fine-grained report memory regardless of region order
  and reject a coarse-only region that cannot satisfy their typed coherence
  requirement.
- **Deletion result:** Production source is one net line smaller: seven lines
  of result-level compatibility API were replaced by a six-line inventory
  invariant, while the two production callers are line-neutral. The focused
  test contract grows by 41 net lines to model the real InlineShadow input and
  all three engines' negative coarse-only capability cases.
- **Completed checked-in gate:** The complete RocJitsu host binary passed
  5,600 of 5,604 tests with four intentional skips; the complete HSA-hook
  binary passed all 197 tests. All 2,878 currently registered simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 75.19
  seconds of wall time. E2E validation remains outside this deletion work.

### Slice 4K: delete legacy static-coverage reconstruction

- **Completed deletion and ownership cutover:** The HSA hook now computes its
  static coverage summary exclusively from `TransformResult::coverage_ledger`.
  The duplicate scans over MOI candidates, resource plans, synchronization
  inventories, and emitted patch kinds are gone, including the Sampled and
  InlineShadow special cases that attempted to reconstruct semantic
  applicability after lowering.
- **Strict-installation contract:** `RJ_CONSAN_REQUIRE_PATCH` now asks the
  typed observation plan whether any semantic probe was requested and the
  coverage ledger whether any requested probe was instrumented. A structural
  prologue or dispatcher cannot satisfy the guard, resource- or placement-
  rejected probes remain visible failures, and a valid empty plan requires no
  instrumentation. Legacy candidates and patch mnemonics have no vote.
- **One semantic denominator:** Coverage groups typed semantic decisions by
  resource kind and original physical site, then reads each cited intent's
  durable lowering outcome. Unsupported, resource-rejected, placement-
  rejected, instrumented, and expert-limit states therefore use the same
  denominator as policy and lowering instead of an architecture-specific hook
  approximation.
- **Test cutover:** The hook regression covers pending, resource-rejected, and
  instrumented MOI plans for Record/Replay, Sampled, and InlineShadow, plus the
  existing SuperCollider cases. The obsolete hook-local B96 mnemonic test was
  deleted because architecture-aware B96 admission is already covered at the
  shared capability and lowerer boundaries and in paired device contracts.
  Fault-reservation fixtures now publish explicit typed plans, while process-
  memory fixtures publish a valid empty plan and no longer fabricate candidate
  or resource-plan telemetry unrelated to memory accounting.
- **Deletion result:** Production source is 134 net lines smaller. Source plus
  tests is 167 net lines smaller; test deletions remove assertions about the
  discarded reconstruction while retaining the typed behavioral guards.
- **Completed checked-in gate:** The complete RocJitsu host binary passed
  5,600 of 5,604 tests with four intentional skips; the complete HSA-hook
  binary passed all 196 tests. All 2,878 simulator device rows across gfx942,
  gfx950, gfx1100, gfx1201, and gfx1250 passed in 71.72 seconds of wall time.
  E2E validation remains outside this deletion work.

### Slice 4L: replace flat mutation telemetry with one typed outcome

- **Completed ownership cutover:** `ConSanMutationOutcome` now owns the fault
  and perturbation request, plan, and emission tallies plus the stable applied
  fault identity. The compatibility lowerer constructs this value once and
  publication moves it into `TransformResult`, clearing the compatibility
  payload instead of retaining a second authoritative copy. The seven flat
  counter/identity members of `ConSanResult` are gone.
- **Runtime cutover:** dry-run load selection, ambiguity rejection, strict
  exactly-one enforcement, fault and perturbation summaries, and post-load
  fault-reservation commit all consume the typed outcome. The hook no longer
  reads compatibility mechanism state for mutation counts or identity.
  Detailed candidate, plan, and patch records remain on the temporary seam
  because they describe lowering mechanics rather than this result-level
  contract.
- **Composition boundary:** the intermediate corruption stage is explicitly a
  fault-only stage carrying a `ConSanMutationTally` and logical identity. It
  cannot overwrite perturbation facts produced by the subsequent
  instrumentation stage. The complete host gate caught and thereby guarded
  this distinction across atomic-address, atomic-order, atomic-scope,
  barrier-drop, and barrier-move compositions.
- **Focused contract gate:** unit tests cover empty, unique, ambiguous, and
  applied tallies; fault/perturbation separation; value semantics;
  deterministic mutation entry; and the single-owner publication invariant.
  Existing composition tests cover retained pristine plans, stable identities,
  dry-run proofs, final validation, and rollback, while hook and device tests
  cover selection and installation behavior.
- **Size result:** production source grows by 48 net lines and source plus
  tests by 89 net lines. The growth is the documented typed value and its
  focused contracts; it replaces seven unstructured fields and deletes four
  hook-side uses of compatibility state. Subsequent slices should reuse
  this value rather than add another projection.
- **Completed checked-in gate:** the complete RocJitsu host binary passed
  5,602 of 5,606 tests with four intentional skips; the complete HSA-hook
  binary passed all 196 tests. All 2,878 simulator device rows across gfx942,
  gfx950, gfx1100, gfx1201, and gfx1250 passed in 69.15 seconds of wall time;
  all 587 physical-gfx950 device rows also passed, accounting for 435.53
  seconds of aggregate CTest test time. E2E validation remains outside this
  deletion work.

### Slice 4M: publish one typed kernel-dispatch contract

- **Completed ownership cutover:** `ConSanDispatchRequirements` is now the
  sole transform-to-runtime contract for affected kernel symbols, fixed and
  dynamic private-segment growth, group-segment growth, and semantic probe
  attribution. Publication reduces all lowering records to one sorted,
  name-unique entry per kernel. The value owns no HSA handles; the hook adds
  executable and symbol lifetimes only after the replacement loads.
- **Semantic attribution instead of patch taxonomy:** An instrumented kernel
  is derived from `ConSanCoverageLedger` entries whose typed lowering outcome
  is `Instrumented`, joined to execution owners in `ProgramInventory`. The HSA
  hook's 46-case patch-kind classifier, resource-plan lookup, and kernel-range
  rescan are deleted. Adding a new patch encoding can therefore no longer
  silently make a valid semantic probe invisible at dispatch time.
- **Segment binding instead of runtime reconstruction:** Low-level patch
  records still supply segment-growth facts at the lowering/publication seam,
  where explicit descriptor owners are preferred and the bounded legacy
  text-range fallback occurs once. The hook consumes only the reduced typed
  contract when binding loaded symbols. Its decision to intercept dispatch
  packets now comes from the contract's dynamic-frame requirement rather than
  a second scan over compatibility patches.
- **Fail-safe lifetime rules:** Non-installable results cannot retain dispatch
  requirements, runtime binding demotion clears them, and discarding a
  replacement clears the typed value together with the temporary mechanism
  state. Validation rejects empty names, empty per-kernel payloads,
  unsorted/duplicate entries, and a dynamic addend larger than the published
  private-segment bound.
- **Focused contract gate:** Unit tests cover empty and attribution-only
  values, ordering and uniqueness, segment bounds, value semantics, packet
  interception, deterministic publication, maximum aggregation, explicit
  multi-owner attribution, access and synchronization execution owners, the
  bounded unowned-patch fallback, malformed result rejection, and runtime
  discard. Hook tests exercise typed dynamic-private interception, symbol
  binding and cleanup, and real evidence binding for zero-record diagnostics.
- **Size result and next deletion:** Production grows by 95 net lines and
  source plus tests by 341 net lines. This is the documented typed boundary
  and its independently falsifiable contract, while 79 net hook lines and the
  runtime's entire patch-taxonomy dependency disappear. The only remaining
  production use of `legacy_mechanism()` is the broad transform-detail
  logging/telemetry block; deleting that final observer is the next
  compatibility-seam slice rather than adding another projection.
- **Completed checked-in gate:** The complete RocJitsu host binary passed
  5,604 of 5,608 tests with four intentional skips, and the complete HSA-hook
  binary passed all 196 tests. All 2,878 simulator device rows across gfx942,
  gfx950, gfx1100, gfx1201, and gfx1250 passed in 74.92 seconds. A focused
  physical-gfx950 gate passed all 44 dynamic-private, mixed-owner, and
  dispatch-identity rows across the engines in 13.78 seconds. E2E validation
  remains outside this deletion work.

### Slice 4N: delete hook rendering of prototype placement counters

- **Completed deletion:** The HSA hook no longer renders SuperCollider flat
  selection, discarded branch work, LDS relay routing, MOI relay routing,
  reservoir allocation, or planner work counters from the compatibility
  mechanism. These were exact implementation-work measurements, not semantic
  coverage, evidence, mutation, capacity, or user-diagnostic contracts.
- **Contract audit:** No validation parser, device oracle, or fault-inventory
  collector consumes these five records. The only consumers outside the
  lowerers were two hook tests that manufactured hundreds of arbitrary counter
  values to preserve the exact prototype log spelling; those tests are
  deleted. Stable patch timing, resource summaries, fault/site identities,
  coverage, evidence loss, and analysis-verdict records remain unchanged.
- **Retained component tests:** Relay-router and lowerer tests continue to use
  the internal counters to prove bounded planning work, exhaustion reporting,
  deterministic routing, reservoir reuse, and placement behavior. Removing a
  cross-layer debug rendering therefore does not discard the tests that
  falsify the underlying algorithms.
- **Deletion result:** Production source is 127 net lines smaller and source
  plus tests is 432 net lines smaller. No replacement projection, renderer, or
  compatibility type was added.
- **Completed checked-in gate:** The selected ConSan host suite passed 1,508
  of 1,510 tests with two external-object benchmarks intentionally skipped,
  and the complete HSA-hook suite passed all 194 tests. All 2,878 simulator
  device rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in
  65.52 seconds. The immediately preceding dispatch-contract slice exercised
  the affected runtime seam on 44 physical-gfx950 rows; this log-only deletion
  does not alter device execution. E2E validation remains outside this work.

### Slice 4O: delete per-candidate and chosen-register debug records

- **Completed deletion:** The hook no longer walks the compatibility result to
  print every perturbation candidate/plan, mutation-composition access plan,
  composite patch proof, raw MOI candidate, accepted resource plan, or chosen
  persistent register. These records exposed candidate container layout,
  decoded operands, exact scratch registers, and the temporary composition
  algorithm rather than a sanitizer behavior or stable diagnostic.
- **Preserved supported output:** The typed fault and perturbation mutation
  summaries remain. So do the fault-site and synchronization identities,
  coverage/trust records, report/evidence records, patch timing, aggregate MOI
  resource summary, resource-alternative chronology, and aggregated resource
  failures consumed by validation or qualification tooling. The complete hook
  gate caught the latter's qualification-level contract; its renderer was
  restored rather than weakening that test.
- **Device-contract audit:** The CDNA kernarg-preload/private-state device pair
  additionally required the exact text `automatic_private_epoch=true`. Both
  behavioral members already exercised that private-state path successfully,
  use stride one to remove the incidental dispatch-identity consumer, and run
  with strict `require_patch`. The mechanism-string oracle was therefore
  deleted while the cross-architecture correct/incorrect workload and its
  semantic output/diagnostic assertions remain intact.
- **Deletion result:** Production source is 208 net lines smaller. Source plus
  checked-in tests is 218 net lines smaller; the three added test lines explain
  why the remaining device oracle is behavioral rather than register-shaped.
  No replacement debug schema or projection was introduced.
- **Completed checked-in gate:** The selected ConSan host suite passed 1,508
  of 1,510 tests with two external-object benchmarks intentionally skipped,
  and the complete HSA-hook suite passed all 194 tests. All 2,878 simulator
  device rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in
  67.92 seconds. This output-only slice does not alter physical device
  execution. E2E validation remains outside this work.

### Slice 4P: delete raw program-inventory dump records

- **Completed ownership cutover:** Every hook-side inventory query now reads
  `TransformResult::program_inventory`, including owner attribution,
  synchronization records, kernel/function summary counts, and the parsed
  code-object summary. The hook no longer reaches through the compatibility
  result to obtain an alternate view of the same immutable program facts.
- **Completed deletion:** The per-section, per-kernel, per-function, per-LDS-
  site, and per-FLAT-site dumps are gone. They repeated the complete decoded
  inventory in a hook-specific textual schema, exposed raw operands and
  preflight mechanics, and had no supported parser or behavioral test
  consumer. Retaining them would make the runtime adapter a second inventory
  presentation layer and preserve a large compatibility-result traversal.
- **Preserved contracts:** The compact program summary, semantic fault and
  synchronization records used by validation tooling, typed coverage and
  coverage-site records, resource qualification summaries, patch proof, and
  analysis verdict remain. This deletion therefore removes decoded-structure
  debugging rather than sanitizer behavior, trust evidence, or supported
  qualification output.
- **Deletion result:** Production source is 177 net lines smaller. No new
  renderer, projection, compatibility type, or replacement output schema was
  added.
- **Completed checked-in gate:** The selected ConSan host suite passed 1,508
  of 1,510 tests with two external-object benchmarks intentionally skipped,
  and the complete HSA-hook binary passed all 194 tests. All 2,878 simulator
  device rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in
  72.16 seconds. This output-only cutover does not alter device execution;
  physical qualification remains covered by the immediately preceding
  physical tranche. E2E validation remains outside this work.

### Slice 4Q: delete dry-run composite-proof reconstruction

- **Prototype mechanism removed:** A SuperCollider fault/perturbation dry run
  used to invoke a second, live full transformation solely to reverse-engineer
  one selected access and a `ConSanCompositeProof` from emitted patch records.
  Neither value affected the dry-run outcome or a later live transform. Their
  hook rendering was deleted in Slice 4O, leaving only mechanism tests as
  consumers of this expensive recursive path.
- **Completed deletion:** The recursive live transform, patch-kind-to-access
  reconstruction, translated-anchor proof builder, `ConSanAccessPlan`,
  `ConSanCompositeProof`, their two compatibility-result fields, and retry
  cleanup are gone. A dry run now performs only the requested semantic
  inventory and mutation planning; it no longer executes hidden live
  instrumentation to manufacture an otherwise unobservable proof object.
- **Behavioral test replacement:** The affected tests now exercise the actual
  live barrier, atomic-order, and atomic-scope compositions. They require a
  validated replacement, exactly one applied fault and perturbation, the
  expected fault rewrite, the perturbation probe, the access probe, and a
  structurally valid modified ELF. The existing rollback test continues to
  prove that an unreachable carried perturbation produces no replacement or
  applied mutation. Thus the retained oracle covers composition behavior
  rather than the deleted patch-derived representation.
- **Deletion result:** Production source is 228 net lines smaller and the
  focused tests are 28 net lines smaller, for 256 net deleted lines. No
  replacement proof type or compatibility seam was introduced.
- **Completed checked-in gate:** The selected ConSan host suite passed 1,508
  of 1,510 tests with two external-object benchmarks intentionally skipped,
  and the complete HSA-hook binary passed all 194 tests. All 2,878 simulator
  device rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in
  72.16 seconds. The deleted path existed only during a dry-run request and
  never supplied device execution, so the immediately preceding physical
  tranche remains the proportional physical gate. E2E validation remains
  outside this work.

### Slice 4R: delete unsupported inventory telemetry

- **Completed deletion:** The hook no longer emits raw parse-visit flags,
  count-only fault/destination/synchronization inventory records, every
  decoded synchronization-event record, or every barrier-lifecycle group.
  Those lines had no documented or checked-in consumer and duplicated facts
  already owned and unit-tested by immutable `ProgramInventory`. The patch-end
  and configuration records retain outcome, warning, patch, and policy facts
  without consulting `visited_code_object`.
- **Retained semantic output:** Fault sites, barrier-move destinations, logical
  synchronization sequences, fault plans and summaries, compact program
  summary, typed coverage, resource qualification, patch proof, and runtime
  analysis remain. The hook gate specifically protects the compact target and
  architecture names, including a parsed-but-unsupported target; that record
  was retained and now reads the typed inventory unconditionally instead of
  depending on `parsed_code_object`.
- **Deletion result:** Production source is 93 net lines smaller. This also
  removes all hook reads of the two raw parse-progress flags. No replacement
  log schema or compatibility value was added.
- **Completed checked-in gate:** The complete HSA-hook binary passed all 194
  tests, including the target-name and unsupported-target contracts. All
  2,878 simulator device rows across gfx942, gfx950, gfx1100, gfx1201, and
  gfx1250 passed in 68.27 seconds. The preceding full ConSan host gate remains
  applicable because this slice changes only hook rendering. Physical and E2E
  execution are unaffected and remain outside this proportional gate.

### Slice 4S: delete cached resource-summary state

- **Single authoritative representation:** Resource plans and emitted patches
  are the transform facts. Their aggregate counts are now calculated by the
  pure shared resource-component function `summarize_consan_resource_plans`.
  `ConSanResult` no longer carries a second mutable summary that had to be
  cleared and recomputed at four control-flow exits and could become stale
  whenever either authoritative collection changed.
- **Completed consumer cutover:** Final validation derives its unsupported-plan
  decision from the shared query, the hook derives the existing resource log
  at its reporting boundary, and mechanism tests derive assertions from the
  same immutable inputs. The hook alternative-telemetry fixture now constructs
  the actual plan alternatives rather than manually setting unrelated cached
  counters.
- **Focused contract gate:** New resource-component tests cover every register
  allocation source, every alternative outcome, selected-to-vetoed
  normalization for an unsupported plan, planned and emitted spill-byte
  aggregation, ignored non-spill patches, and empty input. Existing pipeline
  tests continue to verify that real transforms produce the expected plans and
  emitted spill mechanics.
- **Size and ownership result:** Production source is one net line smaller.
  Source plus tests grows by 75 net lines because the previously untested
  hidden aggregation gained two focused behavioral tests. More importantly,
  one compatibility-result field, its reset, four eager mutation sites, and
  the private pipeline helper are deleted; the retained calculation now has
  one documented, independently testable owner.
- **Completed checked-in gate:** The ConSan host gate passed 1,510 of 1,512
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 64.95
  seconds. This representation-only cutover does not change emitted device
  code; the periodic physical gate remains applicable. E2E validation remains
  outside this work.

### Slice 4T: delete parser-progress compatibility flags

- **Immutable fact ownership:** Structural AMDGPU parsing is now the
  `ProgramInventory::code_object_parsed()` fact. The builder publishes it only
  after successful parsing, independently of whether the parsed target maps to
  a supported semantic architecture. Copies, moves, retries, the production
  pipeline, and hook diagnostics therefore observe it through the same
  immutable program owner as target, architecture, and code-object identity.
- **Completed deletion:** `ConSanResult::parsed_code_object` and
  `visited_code_object` are gone, along with every assignment and retry check.
  “Visited” merely restated that a C++ function had been entered; pristine
  identity already proves that an inventory belongs to the supplied bytes.
  Early configuration/capability rejection also no longer fabricates a
  compatibility inventory solely to carry that progress bit: its typed code-
  object identity remains in `TransformResult`, while its inventory stage is
  correctly absent.
- **Contract-test correction:** Inventory tests cover absent, pre-parse, parsed,
  copied, and moved values. Pipeline tests now assert that a configuration
  rejection leaves inventory absent and that a successful split result owns a
  parsed inventory. Parse-failure and valid-but-unsupported-target tests assert
  the immutable fact directly. Tests that merely asserted entry into
  `try_patch_consan` lost that mechanism-shaped assertion while retaining
  outcome, identity, diagnostic, and transactional-result checks.
- **Deletion result:** Production source is 15 net lines smaller and source
  plus tests is 26 net lines smaller. The compatibility result loses two fields
  and early rejection loses its private inventory builder; no parallel parse
  state or adapter was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,510 of 1,512
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 75.05
  seconds. This state-ownership cutover does not change emitted device code;
  the periodic physical gate remains applicable. E2E validation remains
  outside this work.

### Slice 4U: delete branch-only placement-failure telemetry

- **Behavioral contract:** A branch-only body that cannot be placed is already
  represented by the absence of its instrumentation patch and a specific
  warning explaining the placement or routing failure. Successful transforms
  are represented by their emitted patch plans and final-validation outcome.
  Those are the facts consumed by production and exposed to users.
- **Completed deletion:** `ConSanResult` no longer carries
  `moi_branch_only_placement_failure_count`, and the four MOI lowerers no longer
  maintain the counter at seven separate failure exits. The field influenced
  no production decision and ceased to be hook output in Slice 4N, so retaining
  it would preserve only a mutable prototype implementation detail.
- **Contract-test correction:** Successful placement tests continue to check
  concrete emitted patches, chosen routes, reservoir use where relevant, and
  final validation. The unrouteable-body test continues to check that the
  transform stays unmodified, emits no access-record patch, and reports the
  `has no first-hop relay` warning. Ten assertions of the redundant counter
  were deleted. The separate routing work telemetry remains for now because it
  still guards batch construction and bounded-search complexity; it is not
  conflated with this outcome counter.
- **Deletion result:** Production source is 10 lines smaller and source plus
  tests is 20 lines smaller. No replacement state or adapter was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,510 of 1,512
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 73.97
  seconds. This telemetry-only deletion does not change emitted device code;
  the periodic physical gate remains applicable. E2E validation remains
  outside this work.

### Slice 4V: delete telemetry for transactionally discarded routing

- **Transactional contract:** A speculative FLAT direct-reservoir retry either
  commits its selected patches, routes, placement reservations, and reservoir
  footprint together or retains none of them. Failed speculative work is not a
  transform result and is not part of the supported observability contract.
- **Completed deletion:** The FLAT lowerer no longer publishes a losing retry's
  placement count, routing aggregate, or reservoir inventory. It no longer
  snapshots, subtracts, and stores those values before rolling the retry back.
  With no remaining consumer for telemetry subtraction, the shared 24-field
  delta helper, its parallel member-name schema, its formatting function, and
  the empty-routing helper are also gone. The router's per-plan outcome and
  retained aggregate work counters remain available for bounded-work and
  selected-route tests.
- **Contract-test correction:** The losing-retry test is now explicitly a
  transaction rollback test. It checks that the result is unmodified, owns no
  patches or retained reservoir footprint, selects no branch-only site, and
  preserves the first-pass failure outcome. Successful-retry tests continue
  to check emitted reservoir patches, relay ownership, retained footprint,
  final validation, and reuse of an earlier selected route. Two tests that
  existed only to exercise the deleted delta/formatter mechanism were removed.
- **Deletion result:** Production source is 151 net lines smaller and source
  plus tests is 213 net lines smaller. No replacement state or compatibility
  adapter was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,508 of 1,510
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 68.90
  seconds. This discarded-state deletion does not change emitted device code;
  the periodic physical gate remains applicable. E2E validation remains
  outside this work.

### Slice 4W: delete cached Sampled barrier applicability counts

- **Single authoritative representation:** Sampled barrier admission and
  lowering are represented by typed barrier-site decisions in the observation
  plan and coverage ledger. A successfully emitted sync-metadata patch carries
  its exact `covered_sync_event_count`. Together these facts distinguish
  admitted, instrumented, resource-rejected, and uncovered events without a
  parallel aggregate.
- **Completed deletion:** `ConSanResult` no longer carries
  `sampled_barrier_applicable_event_count`. The Sampled lowerer no longer
  increments the cache before owner/window analysis, subtracts from it on one
  rejection path, or maintains an otherwise unreachable 32-bit overflow
  diagnostic. No production consumer read the value.
- **Contract-test correction:** Tests with typed observation-plan or coverage-
  ledger assertions retain those stronger checks. Tests focused on control-
  flow placement now inspect `covered_sync_event_count` on the emitted sync
  patch rather than the cached result-wide count. Access, barrier, multi-owner,
  resource-rejection, cluster-scope, conditional, and loop cases all remain
  covered.
- **Deletion result:** Production source is 16 lines smaller and source plus
  tests is 21 net lines smaller. No replacement state or adapter was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,508 of 1,510
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 68.13
  seconds. This cached-state deletion does not change emitted device code; the
  periodic physical gate remains applicable. E2E validation remains outside
  this work.

### Slice 4X: delete mirrored owner and dispatch allocation flags

- **Authoritative allocation facts:** The selected owner and dispatch-ID
  registers are the resolved SGPR/VGPR optionals, while emitted prologue and
  instrumentation patches prove their use. Resource plans and allocation
  warnings retain the reason and source when that distinction is relevant.
  A second boolean saying that the same resolved register was automatic is not
  needed by lowering, validation, the hook, or diagnostics.
- **Completed deletion:** `ConSanResult` no longer carries
  `moi_owner_sgpr_automatic`, `moi_dispatch_id_sgprs_automatic`, or
  `moi_dispatch_id_vgprs_automatic`. Automatic placement still records its
  internal state in the live lowering options where later code needs it, but
  it no longer copies that state into a write-only result flag at seven exits.
- **Contract-test correction:** Tests continue to check the resolved register
  values, mutually exclusive scalar/vector dispatch representations, emitted
  initialization and instrumentation code, final validation, and allocation
  warnings. Ten assertions that only restated how those registers were chosen
  were deleted.
- **Deletion result:** Production source is 13 lines smaller and source plus
  tests is 23 lines smaller. No replacement state or adapter was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,508 of 1,510
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 69.52
  seconds. This result-shape deletion does not change emitted device code; the
  periodic physical gate remains applicable. E2E validation remains outside
  this work.

### Slice 4Y: delete mirrored persistent-state allocation flags

- **Behavioral state contract:** Persistent vector state, persistent scalar
  state, owner-private state, and the transient EXEC-save window are proved by
  their resolved assignments and by the private offsets, descriptor updates,
  prologues, instrumentation sequences, resource plans, and diagnostics that
  consume them. The live lowering options retain internal automatic-selection
  markers only for as long as subsequent lowering decisions require them.
- **Completed deletion:** `ConSanResult` no longer carries the write-only
  `moi_persistent_vgprs_automatic`, `moi_persistent_sgprs_automatic`,
  `moi_private_epoch_automatic`, or `moi_exec_save_sgprs_automatic` flags.
  Twenty placement exits no longer mirror option state into those fields. The
  EXEC retry no longer snapshots and restores a boolean whose only purpose was
  preserving that telemetry.
- **Contract-test correction:** Across Record/Replay, Sampled, InlineShadow,
  Inline atomic, and common multi-owner tests, resolved register tuples,
  private-state offsets, exact patch kinds and encodings, descriptor growth,
  warnings, resource plans, owner exclusions, and final validation continue to
  distinguish every allocation fallback. One hundred twenty-five assertions
  that only restated the removed booleans were deleted; no behavioral test case
  or device workload was removed.
- **Deletion result:** Production source is 30 lines smaller and source plus
  tests is 155 lines smaller. No replacement state or adapter was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,508 of 1,510
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 67.75
  seconds. This result-shape deletion does not change emitted device code; the
  periodic physical gate remains applicable. E2E validation remains outside
  this work.

### Slice 4Z: delete the FLAT selection aggregate

- **Single authoritative result:** A FLAT site is represented by its immutable
  inventory identity, typed coverage-ledger outcome, emitted patches, and any
  diagnostic explaining rejection. Successful relay and reservoir behavior is
  visible in the actual patch and retained-route records. A parallel aggregate
  of candidate categories, relay words, missing resources, routing totals, and
  selected counts does not add a behavioral contract.
- **Completed deletion:** `ConSanResult` no longer carries
  `ConSanFlatSelectionTelemetry`. The FLAT lowerer no longer maintains its
  candidate-category counts, routing and placement counters, relay-word total,
  missing-VCC and missing-scratch counts, selected count, or final reservoir
  snapshot. Speculative direct-reservoir retry rollback now uses the retained
  candidate boundary it actually needs instead of snapshotting the aggregate.
  Partial-selection diagnostics report the precise live rejection counters
  rather than referring users to deleted structured telemetry.
- **Coverage-ledger correction:** Removing the aggregate exposed that two
  resource-failure exits did not publish a typed per-site outcome. A shared
  SuperCollider helper now marks every intent for the rejected physical site
  `ResourceRejected` when VCC-save or scratch/spill resources cannot be
  encoded. This is a stronger and architecture-neutral result than retaining a
  FLAT-specific result-wide count.
- **Contract-test correction:** Tests continue to inspect emitted FLAT patches,
  relay routes, retained reservoir footprint, retry commit and rollback, and
  final validation. Rejection tests now assert exact `Instrumented` and
  `ResourceRejected` ledger outcomes and the precise warning reason. Assertions
  that merely mirrored the deleted aggregate were removed, and no behavioral
  test case or device workload was removed.
- **Deletion result:** Production source is 76 net lines smaller and source
  plus tests is 127 net lines smaller. No compatibility adapter or replacement
  aggregate was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,508 of 1,510
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 67.20
  seconds. This result-shape deletion does not change emitted device code; the
  periodic physical gate remains applicable. E2E validation remains outside
  this work.

### Slice 4AA: keep carried-composition phase state out of results

- **Phase-local contract:** Validation of an intermediate instrumentation image
  over a fault-mutated input may consume the carried pristine perturbation plan.
  That is a property of the internal finalization call, not a persistent fact
  about the transform result. Public validation always proves a complete result
  from its public inventory and composed-stage proof.
- **Completed cutover:** `ConSanResult` no longer carries the mutable
  `carried_composite_instrumentation_stage` phase flag. The composition
  coordinator passes the internal validation purpose directly to finalization,
  and only the carried intermediate stage can select that path. Complete
  results therefore cannot leak, retain, or have callers alter an internal
  phase marker.
- **Retained behavioral proof:** Existing atomic and barrier fault-composition,
  carried perturbation, rollback, corruption, and public revalidation tests
  exercise both validation purposes. The checked-in device workloads continue
  to cover fault-only and ordinary complete-result validation. No assertion was
  weakened or removed for this cutover.
- **Size result:** Production and test line counts are unchanged: the internal
  parameter replaces the public field and its reads without adding a
  compatibility adapter. The result shape and ownership boundary are simpler.
- **Completed checked-in gate:** The ConSan host gate passed 1,508 of 1,510
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 67.35
  seconds. The periodic physical gate remains applicable. E2E validation
  remains outside this work.

### Slice 4AB: remove the public raw-result compatibility view

- **Completed artifact ownership:** `TransformResult` now directly owns the
  emitted patch inventory, resource plans, fault sites, barrier-move
  destinations, and fault plans. These are the retained inputs to supported
  patch-proof, resource-qualification, mutation dry-run, and automatic-report
  metadata behavior. The HSA coordinator and report registry consume those
  typed fields directly instead of reaching through a second result object.
- **Completed compatibility deletion:** The public `legacy_mechanism()` API is
  gone, as are all production and ordinary-test callers. Ordinary and completed
  transforms no longer retain a hidden `ConSanResult` after publication. At
  this slice, the address-free MOI report-sizing result alone retained a
  private unmodified retry inventory; Slice 5S later deletes that final broad
  capsule. Runtime demotion clears the one published patch inventory rather
  than synchronizing two public representations.
- **Focused contract gate:** Pipeline tests independently cover publication of
  all five artifact families, direct/retry artifact parity, mutation-detail
  publication, patch removal during runtime demotion, and rejection when an
  ordinary result is incorrectly passed to the inventory retry. The complete
  hook gate protects the existing diagnostic and automatic-report mapping
  output through the new owner.
- **Size and ownership result:** Production grows by 27 net lines and source
  plus tests by 70 net lines. The growth is the generously documented public
  ownership and focused retry/artifact contract; it deletes the unrestricted
  compatibility accessor and avoids retaining the much larger raw result in
  every ordinary transform. No new public adapter or duplicate artifact
  collection was introduced.
- **Completed checked-in gate:** The ConSan host gate passed 1,509 of 1,511
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 60.82
  seconds. The periodic physical gate remains applicable. E2E validation
  remains outside this work.

### Slice 4AC: delete the mirrored LDS replay-limit signal

- **Single authoritative result:** The shared bounded-planning meter owns LDS
  convergence work and exhaustion, while the convergence loop also requires
  successful termination before emission. A second value embedded in retained
  reservoir-footprint telemetry cannot add another supported outcome.
- **Completed deletion:** `ConSanBranchOnlyReservoirTelemetry` no longer carries
  `lds_replay_limit_reached_count`, and the LDS lowerer no longer copies the
  planning meter's exhaustion bit into it. No production or test consumer read
  the mirror. Reservoir telemetry is again only retained footprint; planning
  telemetry is only bounded-work behavior.
- **Retained behavioral proof:** Thirty-one focused reservoir, routing, and
  convergence tests continue to prove bounded exhaustion, transaction rollback,
  used and unused footprint, recursive routing, and successful convergence.
  No test or observable behavior was weakened.
- **Deletion result:** Production source is nine lines smaller. No replacement
  field, adapter, or test-only mechanism was added.
- **Completed checked-in gate:** The ConSan host gate passed 1,509 of 1,511
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 72.80
  seconds. This mirror deletion does not change emitted device code; the
  periodic physical gate remains applicable. E2E validation remains outside
  this work.

### Slice 4AD: derive unused reservoir footprint

- **Single authoritative inventory:** Retained relay-reservoir footprint is
  completely described by planned count/bytes and used count/bytes. Unused
  footprint is their difference, not independently mutable transform state.
- **Completed deletion:** `ConSanBranchOnlyReservoirTelemetry` no longer stores
  `unused_reservoir_count` or `unused_appended_bytes`. The shared accumulator no
  longer updates them and MOI barrier composition no longer recomputes them
  after changing the used totals. The value is now four counters with no
  derived mirrors.
- **Contract-test correction:** The shared reservoir test still proves the
  exact unused count and bytes by subtraction. Record/Replay proves a retained
  unused reservation, while Inline Shadow and SuperCollider prove planned
  footprint bounds used footprint and retain their exact emitted-use checks.
  These assertions cover behavior without testing redundant storage.
- **Deletion result:** Production source is 11 lines smaller and source plus
  tests is 13 lines smaller. No accessor, cache, or compatibility field replaces
  the deleted values.
- **Completed checked-in gate:** The ConSan host gate passed 1,509 of 1,511
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 75.50
  seconds. This derived-state deletion does not change emitted device code; the
  periodic physical gate remains applicable. E2E validation remains outside
  this work.

### Slice 4AE: delete aggregate branch-only routing work

- **Phase-owned observations:** Branch-only relay routing retains work counts
  for relay qualification, fallback setup, feasibility scanning, and route-
  optimization search and scanning. Each counter names the phase whose bound
  it can validate. The router's direct outcome continues to expose its
  internal aggregate while a plan is being solved, where that value is used by
  the focused complexity tests.
- **Completed deletion:** `ConSanBranchOnlyRoutingTelemetry` no longer retains
  `search_work_count` or `scan_work_count`. The former was an
  undifferentiated feasibility-solver implementation detail and the latter was
  the sum of three already-retained phase counters. Neither value had a
  production reader. The result recorder no longer copies either aggregate.
- **Contract-test correction:** The result-level routing test continues to
  prove each exact phase count and every failure and strategy count. The RDNA4
  far-body behavior test now proves qualification work plus actual fallback or
  feasibility-scan work instead of only observing the deleted totals. Direct
  router tests retain their aggregate bounded-work, deterministic-cost, and
  limit-exhaustion assertions because those values remain meaningful inside
  the solver contract.
- **Deletion result:** Production source is four net lines smaller and source
  plus tests is six net lines smaller. No compatibility accessor or derived
  aggregate replaces the deleted fields.
- **Completed checked-in gate:** The focused branch-only router and RDNA4
  far-body gate passed all 91 tests. The ConSan host gate passed 1,509 of 1,511
  tests with two external-object benchmarks intentionally skipped, and the
  complete HSA-hook binary passed all 194 tests. All 2,878 simulator device
  rows across gfx942, gfx950, gfx1100, gfx1201, and gfx1250 passed in 61.44
  seconds. This result-only deletion does not change emitted device code; the
  periodic physical gate remains applicable. E2E validation remains outside
  this work.

### Slice 4AF: retain only integration-level routing telemetry

- **Ownership boundary:** `BranchOnlyRelayPlanOutcome` is the authoritative
  per-call solver result. Its focused tests cover return and reservation
  failures, phase exhaustion, invariant fallback, pristine-relay rejection,
  and optimization quality directly. A completed transform retains only the
  much smaller observation set needed to prove integration behavior: batched
  invocation shape, exercised fallback tier, selected rejection classes, and
  phase work.
- **Completed deletion:** `ConSanBranchOnlyRoutingTelemetry` no longer copies
  nine per-call incident fields that had no policy, diagnostic, or integration
  consumer. Recording no longer mirrors return or reservation failures, phase
  and optimizer exhaustion, invariant failures, pristine occupancy rejection,
  or excess optimization claims. MOI and LDS lowering no longer make no-op
  reservation-failure recording calls after commits fail.
- **Retained contract:** Integration tests still detect accidental per-site
  inventory construction through exact pair and plan-call counts, prove the
  fallback and optimizer paths used by generated layouts, and retain the
  rejection classes they consume. The router's unit suite still proves every
  removed fact on the direct outcome. The narrowed recorder has direct unit
  coverage for every retained failure and rejection class.
- **Deletion result:** Production source is 66 net lines smaller and source
  plus tests is 75 net lines smaller. No replacement result type, adapter, or
  derived copy was added.
- **Completed checked-in gate:** All 90 focused branch-only router tests pass.
  The ConSan host gate passed 1,509 of 1,511 tests with two external-object
  benchmarks intentionally skipped, and the complete HSA-hook binary passed
  all 194 tests. All 2,878 simulator device rows across gfx942, gfx950,
  gfx1100, gfx1201, and gfx1250 passed in 65.49 seconds. Emitted code and
  routing decisions are unchanged; the periodic physical gate remains
  applicable. E2E validation remains outside this work.

### Slice 5A: Record/Replay evidence requirements

- **Completed boundary and contract:** The pure Record/Replay evidence planner
  consumes only `ConSanObservationPlan` and
  `ConSanRecordReplayCapacityPolicy`. It returns a typed alternative that
  intrinsically denotes bounded-first-light retention, completeness-invalidating
  evidence loss, and executable lifetime, plus runtime requirements and exact
  address-free ABI sizing input and result. The current heterogeneous
  report-layout planner is an implementation detail of that value; neither type
  owns an address or allocation.
- **Completed production cutover:** Automatic Record/Replay sizing counts
  access ranges, barriers, atomics, and fences from admitted typed intents.
  The HSA hook validates the requirements' coherent host/device memory, atomic
  publication, allocation-size, and executable-binding facts before
  allocation, then passes the existing layout/address through the late-bound
  compatibility adapter. Production sizing no longer rescans resource plans,
  site dispositions, or emitted patches.
- **Historical compatibility seam, now deleted:** This slice temporarily let
  hook unit tests inject synthetic `ConSanResult` values that predated
  `ObservationPlan`. Slice 4H migrated those fixtures to typed plans and
  deleted the fallback projection, so production and tests now share one
  evidence-sizing path.
- **Completed test gate:** Focused type tests exhaustively cover
  construction-reason, layout-outcome, and layout-reason enums; explicit
  capacity policies; empty and mixed intent plans; multi-range
  physical-intent limits; target-neutral record-dimension mapping; invalid and
  foreign plans; malformed intent payloads; capacity failure; every runtime
  fact; cross-type invariants; value semantics; deterministic non-mutation;
  and a deliberately contradictory legacy result proving that valid plans are
  authoritative. Existing report-boundary/overflow, hook allocation/retry,
  multi-object, lifecycle, and paired device tests remain the integration
  gate.
- **Prerequisite:** Slices 2 and 4A--4C.

The completed host gate passed 1,461 of 1,463 ConSan tests, with two
external-object benchmarks intentionally skipped, plus all 98 ConSan
hook/runtime tests. All 2,908 emulator device rows passed in 75.94 seconds of
wall time. Twelve representative physical-gfx950 Record/Replay access,
publication, atomic, fence, Stream-K, and tree-reduction rows passed. The
physical gfx950 `d128-block` Record/Replay E2E row remained fully accepted with
128/128 accesses and 119/119 barriers. RocJitsu-emulated gfx1250 retained its
known yellow result: static coverage remained complete at 18/18 accesses and
8/8 barriers, while bounded replay metadata filled before dynamic analysis
could complete.

### Slice 5B: Sampled, Inline, and SuperCollider evidence requirements

- **Completed boundary and contract:** Sampled, InlineShadow, and
  SuperCollider now have separate address-free evidence-requirement types
  under one closed variant. Each alternative intrinsically defines its
  retention and loss semantics and executable lifetime while carrying only
  varying runtime capabilities, capacities, and typed construction results.
  Sampled and InlineShadow retain their distinct ABI sizing inputs and plans;
  SuperCollider retains only its fixed sticky-marker size. No largest-union
  report contract exists.
- **Completed immutable derivation:** Sampled capacities are derived only from
  typed access, barrier, and atomic intents plus explicit capacity policy.
  InlineShadow joins exact-shadow intents to immutable access identities,
  original kernel-descriptor LDS declarations, and normalized native static
  offsets. Group-FLAT, dynamic LDS, and an encoded extent beyond the fixed
  declaration select the complete target aperture; missing required inventory
  facts fail closed with a typed reason. SuperCollider derives its zero-or-one
  marker requirement only from admitted redundant-access intents.
- **Completed production cutover:** Every valid production observation plan
  reaches the engine-specific typed planner before hook allocation. The hook
  consumes the resulting runtime requirements and late-binds the existing
  report or marker registry. Slice 4H subsequently deleted the old
  result/patch rescan and migrated the synthetic hook fixtures. ABI decoders
  and allocation mechanics remain implementation details for later slices.
- **Completed type and unit gate:** Focused tests cover neutral and directly
  constructed capacity policies; every construction/layout reason enum and
  invalid sentinel; every typed-alternative and intent-to-capacity mapping;
  expert limits; missing
  inventory relationships; descriptor/static-range/full-aperture selection;
  capacity failure; runtime capability admission; cross-type and
  schema-specific invariants; the closed variant; deterministic non-mutation;
  and contradictory legacy telemetry proving typed-plan authority. The
  normalized LDS-offset tests independently cover CDNA3, CDNA4, CDNA5, RDNA3,
  RDNA4, direct-to-LDS, and unreadable pristine bytes.
- **Prerequisite:** Slice 5A.

The completed host gate passed 1,484 of 1,486 ConSan tests, with two
external-object benchmarks intentionally skipped, plus all 85 selected ConSan
hook/runtime tests. All 2,908 simulator device rows passed in 78.23 seconds of
wall time, and a focused 20-row physical-gfx950 gate passed paired repeated-
dispatch, marker-pressure, full-bank Stream-K, and large-LDS cases for
Sampled, InlineShadow, and SuperCollider. Physical gfx950 `d128-block` E2E
validation accepted all four engines with 128/128 accesses; Record/Replay and
InlineShadow covered 119/119 barriers and Sampled covered 117/117.
RocJitsu-emulated gfx1250 accepted SuperCollider, Sampled, and InlineShadow
with 18/18 accesses and all applicable barriers. Record/Replay retained its
pre-existing yellow bounded-capacity condition: static coverage was complete
at 18/18 accesses and 8/8 barriers, but replay metadata filled before dynamic
analysis completed.

### Slice 5C: delete legacy evidence-capacity adapters

- **Direct typed ownership:** The typed pipeline now constructs each MOI
  capacity policy directly from the three values that own its semantics: the
  automatic-allocation ceiling in `ConSanRequest`, the optional expert access
  cap in `TransformPolicy`, and InlineShadow's LDS aperture in
  `RuntimeCapabilities`. Evidence planning no longer constructs the prototype
  `ConSanOptions` aggregate and then reads those values back out.
- **Completed deletion:** The three production `ConSanOptions`-to-capacity
  adapters and their declarations are gone, as is the adapter-only unit test.
  Mechanism-focused legacy tests construct the same narrow policies locally in
  test support until those tests migrate to the typed pipeline; production
  evidence headers and planners no longer depend on the legacy aggregate.
- **Contract test:** A pipeline test covers Record/Replay, Sampled, and
  InlineShadow with and without an explicit expert access cap. It compares the
  published requirement with direct planning from the typed request, policy,
  capabilities, immutable observation plan, and program inventory, including
  the allocation ceiling and InlineShadow aperture. Existing evidence-planner
  tests continue to cover every individual capacity input and outcome.
- **Deletion result:** Production source is 44 net lines smaller and source
  plus tests is 22 net lines smaller. The added test checks the production
  ownership boundary instead of retaining tests for deleted adapters.
- **Completed checked-in gate:** All 54 focused evidence-requirement and
  pipeline tests pass. The complete host gate passes 1,509 tests with the two
  expected benchmark-object skips; all 194 hook tests pass; and all 2,878
  simulator-device tests pass in 64.14 seconds of wall time. Evidence semantics
  and emitted device code are unchanged; E2E validation remains outside this
  work.

### Slice 5D: delete the raw retry option snapshot

- **Typed reconstruction:** Automatic-report retry now reconstructs its
  pristine lowering options from `ConSanRequest`, `TransformPolicy`,
  `RuntimePolicy`, `ConSanDebugOverrides`, `RuntimeCapabilities`, and an
  explicitly unbound resource value. The live `MutationRequest` and bound
  report allocation remain the only narrow changes applied by the existing
  retry contract.
- **Completed deletion:** `ConSanResult` no longer embeds an optional copy of
  the large mutable `ConSanOptions` aggregate. The initial MOI analysis no
  longer writes that snapshot, and raw mechanism retry receives its pristine
  options explicitly instead of treating a previous output as hidden input.
  The typed result privately retains only the one-bit extended-barrier shape
  choice that cannot yet be recovered from the other typed inputs.
- **Contract coverage:** The raw retry tests pass their original pristine
  configuration explicitly and continue to compare report-only, synchronization,
  malformed-inventory, disabled-fault, late-fault, extended-barrier, and
  fallback behavior with fresh transforms. Typed pipeline tests independently
  compare retained-inventory retry with a direct typed transform and verify the
  extended-barrier shape choice and clearing of premature runtime binding.
- **Size result:** Production line count is unchanged: typed reconstruction and
  the documented private shape bit replace the deleted snapshot plumbing.
  Object state is materially smaller and more narrowly owned because every
  ordinary `ConSanResult` no longer carries storage for an optional
  `ConSanOptions`.
- **Completed checked-in gate:** All 11 focused retry tests pass. The complete
  host gate passes 1,509 tests with the two expected benchmark-object skips;
  all 194 hook tests pass; and all 2,878 simulator-device tests pass in 60.74
  seconds of wall time. E2E validation remains outside this work.

### Slice 5E: delete mirrored transform-mode identity

- **Authoritative owners:** `ConSanRequest` remains the sole owner of requested
  flavor and MOI engine. Once semantic policy succeeds,
  `ConSanObservationPlan::engine` is the immutable identity shared by its
  decisions, intents, coverage ledger, evidence requirements, and retry
  validation.
- **Completed deletion:** The mutable `flavor` and `moi_engine` mirrors are gone
  from `ConSanResult`, along with every production and synthetic-fixture write.
  Exception paths no longer manufacture mode identity after a failed
  transform. Raw retry now checks the requested engine against the retained
  observation plan instead of comparing two mutable legacy enums.
- **Contract coverage:** The Record/Replay, Sampled, and InlineShadow inventory
  tests assert the selected observation-plan engine. Retry still rejects an
  engine mismatch, but the negative test now corrupts the authoritative plan
  identity. Pipeline publication and HSA-hook fixtures continue to cover
  request-to-result behavior without populating deleted mirrors.
- **Deletion result:** Production source is 13 net lines smaller and source plus
  tests is 34 net lines smaller. Each raw result also loses the two redundant
  enum fields.
- **Completed checked-in gate:** All 25 focused inventory and retry tests pass.
  The complete host gate passes 1,509 tests with the two expected
  benchmark-object skips; all 194 hook tests pass; and all 2,878
  simulator-device tests pass in 66.85 seconds of wall time. E2E validation
  remains outside this work.

### Slice 5F: delete legacy loader-admission policy

- **Single policy owner:** `TransformResult::install_action` is now the only
  implementation that maps a static transform outcome, final-validation fact,
  replacement storage, and fail-closed policy to a loader action. The HSA
  adapter already consumed that typed method; production had no remaining raw
  caller.
- **Completed deletion:** The duplicate
  `consan_install_action(const ConSanResult &, bool)` declaration and
  implementation are gone. Raw mechanism and fuzz tests retain their stronger
  structural assertions but no longer retest loader policy through the
  compatibility record. The redundant raw truth-table test is also gone.
- **Preserved contract coverage:**
  `InstallActionTruthTableUsesOnlySplitStaticResult` covers unchanged,
  unsupported, invalid, validated replacement, missing validation, missing
  bytes, fail-open, and fail-closed cases on the production result. Additional
  typed pipeline and hook tests exercise policy on real transformations.
- **Deletion result:** Production source is 17 lines smaller and duplicate
  test/fuzz code is 46 lines smaller.
- **Completed checked-in gate:** The complete host gate passes 1,508 tests with
  the two expected benchmark-object skips; all 194 hook tests pass; and all
  2,878 simulator-device tests pass in 70.03 seconds of wall time. The removed
  raw truth table accounts for the one-test decrease. E2E validation remains
  outside this work.

### Slice 5G: delete dead telemetry predicates

- **Completed deletion:** The unused planning-work emptiness predicate and the
  reservoir-telemetry emptiness predicate are gone. The latter had one test
  caller, which now asserts the telemetry value contract directly.
- **Deletion result:** Production headers are 10 lines smaller without changing
  telemetry storage or behavior. All ConSan, hook, and device targets rebuild,
  and the affected degraded-partial-route test passes. The immediately
  preceding complete checked-in gate remains applicable because no executable
  behavior changed.

### Slice 5H: remove raw results from the hook transform seam

- **Typed production boundary:** The HSA hook's transform override now has the
  same contract as the real transformer: it receives the immutable typed inputs
  and returns a `TransformResult`. A raw `ConSanResult` no longer crosses the
  hook transform boundary, and production no longer exposes a function that
  publishes an arbitrary raw mechanism record.
- **Narrow test fixture privilege:** Mechanism and HSA-hook tests still need to
  construct synthetic lowering artifacts while the prototype lowerer is being
  decomposed. A documented helper in the test tree alone may pass such a
  fixture through `TransformResult`'s real validation and stage-publication
  path. The helper is a friend only for that operation; it is not a production
  adapter or API.
- **Completed deletion:** The public raw-result publisher, its declaration,
  definition, and all production calls are gone. Retry calls the private typed
  publication operation directly, and each test override publishes its fixture
  before returning to the hook. The existing pipeline publication test still
  proves that joined coverage and segment-growth artifacts take the production
  validation path.
- **Deletion result:** Production source is 29 net lines smaller. The test-only
  fixture helper is 34 lines; including its call-site conversion, source plus
  tests grows by 11 net lines in exchange for removing the raw compatibility
  type from a production integration boundary.
- **Completed checked-in gate:** The complete host gate passes 1,508 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across gfx942, gfx950, gfx1100, gfx1201, and
  gfx1250 pass in 69.26 seconds of wall time. E2E validation remains outside
  this work.

### Slice 5I: delete duplicate patched-image rejection telemetry

- **Authoritative owners:** `ConSanPatchedImageGrowthBudget` owns the resolved
  total and remaining limits before replacement, and `TextReplacementResult`
  owns the exact transaction growth and failure outcome after replacement.
  The ConSan replacement helper combines those values once into the retained
  diagnostic.
- **Completed deletion:** `ConSanResult` no longer stores a second vector that
  copied the operation, input size, policy, existing growth, transaction
  growth, required total, and limit for policy rejections. The duplicate
  `ConSanPatchedImageGrowthRejection` type and both construction paths are
  gone. No production code read this telemetry.
- **Preserved contract coverage:** Focused budget tests still cover absolute
  and relative limits, saturation, invalid policy, and an already-exceeded
  image. Replacement tests retain exact diagnostics at relative and absolute
  boundaries and for malformed/allocation failures. The staged-growth test now
  asserts the complete diagnostic, including original input, accumulated
  first-stage growth, second-stage transaction growth, required total, and
  resolved limit, instead of inspecting the deleted copy.
- **Deletion result:** Production source is 30 lines smaller and source plus
  tests is 48 lines smaller. No replacement telemetry or adapter was added.
- **Completed checked-in gate:** All 13 focused growth tests pass. The complete
  host gate passes 1,508 tests with the two expected benchmark-object skips;
  all 194 HSA-hook tests pass; and all 2,878 simulator-device tests across the
  five supported targets pass in 64.71 seconds. E2E validation remains outside
  this work.

### Slice 5J: make the transform outcome authoritative for validation

- **Single state contract:** `ConSanTransformOutcome::ModifiedValid` means that
  replacement bytes completed structural and semantic final validation. No
  other outcome is validated or installable. The outcome and the presence of
  replacement bytes are therefore sufficient to validate a result and derive
  loader policy.
- **Completed deletion:** Both `ConSanResult` and `TransformResult` no longer
  carry a `final_validation_passed` boolean that duplicated the outcome. All
  writes, clears, copies, well-formedness checks, loader-policy branches, and
  staged-composition checks now use the authoritative enum. This removes the
  formerly representable contradiction “ModifiedValid but not validated.”
- **Contract-focused tests:** Existing final-validation assertions now check
  `ModifiedValid` directly. The typed install-action truth table retains the
  missing-replacement rejection case. The hook integration test that formerly
  fabricated the contradictory boolean state now proves rejection of a
  `ModifiedValid` fixture with missing replacement bytes. Retry equivalence
  tests no longer compare the same outcome twice.
- **Fuzzer maintenance:** The transform fuzzer now consumes the authoritative
  outcome and `ProgramInventory` identity. Its unmatched-wait oracle derives
  the original text-section file offset from `AmdGpuCodeObject`, replacing
  stale references to raw result fields deleted in earlier slices. The normal
  build does not configure fuzz targets, so the source was independently
  syntax-checked with the build's Clang flags and profile definition.
- **Deletion result:** Production source is 13 net lines smaller and source
  plus tests is 33 net lines smaller.
- **Completed checked-in gate:** The complete host gate passes 1,508 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 76.71
  seconds. E2E validation remains outside this work.

### Slice 5K: localize the Record/Replay access-to-sync handoff

- **Explicit stage contract:** `MoiRecordReplayAccessOutput` now carries the
  exact synchronization-island reservation and generated relay ranges from
  Record/Replay access placement to the immediately following synchronization
  lowering. Its lifetime is one MOI pipeline invocation; it is not transform
  output, telemetry, or a supported inspection interface.
- **Completed deletion:** `ConSanResult` no longer exposes four mutable fields
  used only as temporary communication between those two lowerings. Access
  placement publishes its narrow output directly, and barrier placement
  consumes it explicitly. Other engines receive an empty value rather than
  inheriting unrelated Record/Replay state.
- **Behavior-focused tests:** The access-heavy split-barrier test still
  requires every selected access and every split-barrier member to be emitted.
  The CDNA4 sparse/dense composition test still requires all 1,024 accesses,
  its barrier, branch-only routes, and the generated-code route frontier to
  succeed. Assertions that directly inspected the deleted reservation count
  and private periodic-bank spacing were removed: they duplicated those
  behavioral outcomes and unnecessarily made one placement algorithm part of
  the test contract.
- **Size result:** Production grows by 12 net lines for the documented typed
  stage value and explicit dependency, while source plus tests shrinks by 14
  net lines. Each ordinary `ConSanResult` loses an optional, two integers, and
  a vector. The aggregate has no independent behavior to unit-test; the two
  composition tests exercise both populated members through their real
  consumer.
- **Completed checked-in gate:** The two focused composition tests pass. The
  complete host gate passes 1,508 tests with the two expected benchmark-object
  skips; all 194 HSA-hook tests pass; and all 2,878 simulator-device tests
  across the five supported targets pass in 62.93 seconds. E2E validation
  remains outside this work.

### Slice 5L: derive staged composition from typed patch provenance

- **Single state contract:** A composite result is identified by the presence
  of both `Mutation` and `Instrumentation` patch phases. Final validation
  consumes those typed records directly; it no longer trusts a separately
  writable boolean that can contradict the inventory it purported to prove.
- **Completed deletion:** `staged_composition_validated`, its three production
  writes, and 17 repetitive test assertions are gone. Mutation-only results
  retain only Mutation patches, instrumentation-only results retain only
  Instrumentation patches, and the staged composer prepends its independently
  validated Mutation inventory to the Instrumentation inventory.
- **Stronger validation and contract test:** A lifecycle-leave rewrite is
  considered reinstrumented only when the covering Instrumentation patch also
  identifies a relocated guest instruction that decodes as a barrier. Merely
  fabricating an Instrumentation-phase accounting record cannot bypass the
  fixed-zero lifecycle proof. The composite corruption test now removes the
  Instrumentation phase from every patch and requires the resulting illegal
  same-phase overlap to be rejected.
- **Deletion result:** Production source is one net line smaller, source plus
  tests is 17 net lines smaller, and every raw result loses the duplicate
  boolean state.
- **Completed checked-in gate:** The two focused final-validation tests pass.
  The complete host gate passes 1,508 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878
  simulator-device tests across the five supported targets pass in 68.32
  seconds. E2E validation remains outside this work.

### Slice 5M: derive full workgroup-payload ownership

- **Single semantic owner:** Descriptor mutation and final validation now call
  one pure predicate that identifies when an emitted MOI patch makes its owner
  consume all three launch workgroup coordinates. The predicate distinguishes
  genuine MOI observation patches from independently composed mutation and
  malformed-barrier patches. It also states the architecture distinction
  directly: RDNA and CDNA5 observation bodies consume the firmware payload,
  while CDNA3/CDNA4 require a real entry capture backed by complete persistent
  state.
- **Completed deletion:** `ConSanResult` no longer retains a sorted vector of
  descriptor offsets that merely copied the writer's decision for a later
  validator. The descriptor writer resolves owners from the semantic predicate
  and the validator independently joins that same contract with each owning
  kernel's resource requirement. Workgroup registers remain resource facts;
  their mere presence no longer masquerades as proof that a descriptor payload
  transaction occurred.
- **Contract coverage:** A direct unit test covers owned and unowned patches,
  invalid targets, RDNA4 and CDNA5 observation bodies, CDNA3/CDNA4 complete and
  incomplete entry storage, MOI relay islands, SuperCollider exclusion, and a
  composed malformed-barrier abort. Existing descriptor and device tests still
  exercise the real writer/validator path, including full Record/Replay launch
  coordinates and CDNA4 InlineShadow persistent state.
- **Size result:** Production source grows by two formatted lines for the
  documented shared predicate, but every transform result loses a vector and
  its allocation. No replacement cache, flag, or telemetry field was added.
- **Completed checked-in gate:** The complete host gate passes 1,509 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 72.28
  seconds. E2E validation remains outside this work.

### Slice 5N: delete copied prologue-scratch assignments

- **Direct stage ownership:** Entry-prologue scratch assignments remain in the
  mutable lowering options that carry them from resource planning to prologue
  emission. They are transient mechanism state, not transform output,
  validation evidence, or user-visible telemetry.
- **Completed deletion:** `ConSanResult` no longer copies the complete
  assignment vector after each of five scalar-persistence selection paths or
  clears a sixth unused mirror on entry. No production consumer ever read the
  copy. Two tests no longer assert its size or selected register; they retain
  the stronger observable requirements that the correct scalar state,
  descriptor growth, access/barrier instrumentation, and owning entry
  prologues are actually emitted and finally validated.
- **Deletion result:** Production source is 12 lines smaller, source plus tests
  is 16 lines smaller, and every result loses another dynamically allocated
  vector. No replacement state or adapter was added.
- **Completed checked-in gate:** The two affected full-bank and multi-component
  tests pass. The complete host gate passes 1,509 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878 simulator-
  device tests across the five supported targets pass in 70.45 seconds. E2E
  validation remains outside this work.

### Slice 5O: delete the transient owner-SGPR mirror

- **Authoritative owners:** Explicit owner-SGPR requests remain in lowering
  options. Automatic InlineShadow owner computation uses the already-owned
  EXEC-save allocation and per-owner transient assignments. Those values feed
  emission directly; the completed transform does not need a separate scalar
  claiming that all owners happened to choose one register.
- **Completed deletion:** `ConSanResult::resolved_moi_owner_sgpr`, its reset and
  fallback copy, two early writes, and an 11-line reconciliation pass used only
  to publish the test observation are gone. No production consumer read the
  field. Tests with explicit input use that input; automatic-allocation tests
  use the authoritative EXEC-save result while continuing to match the emitted
  owner acquisition, nonzero bias, scalar backup/restore, and VALU transfer
  sequences.
- **Deletion result:** Production source is 17 lines smaller, source plus tests
  is 19 lines smaller, and each result loses the redundant optional scalar. No
  replacement field or test adapter was added.
- **Completed checked-in gate:** All five affected instruction-level tests
  pass. The complete host gate passes 1,509 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878 simulator-
  device tests across the five supported targets pass in 76.24 seconds. E2E
  validation remains outside this work.

### Slice 5P: delete uncalled hook diagnostic formatters

- **Narrow diagnostic ownership:** The hook retains only formatting functions
  that contribute to an emitted diagnostic or report. Stable shared names for
  typed ConSan vocabulary remain in the core, while hook-local names remain
  only where the hook actually presents hook-owned policy or mechanism state.
- **Completed deletion:** Seven hook-local functions with no callers are gone:
  the preflight-action, perturbation-kind, perturbation-edge,
  synchronization-event-kind, MOI-candidate-source, LDS-access-kind, and flat
  address-space-hint formatters. Four of these were complete enum switches;
  three merely forwarded to already-tested shared vocabulary. No diagnostic,
  configuration, test seam, or external API used them.
- **Deletion result:** Production source is 64 lines smaller. No replacement
  helper, compatibility adapter, or test-only observation was added. Existing
  shared-vocabulary tests continue to cover the names that are actually
  presented by live diagnostics.
- **Completed checked-in gate:** The complete host gate passes 1,509 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 67.44
  seconds. E2E validation remains outside this work.

### Slice 5Q: delete abandoned placement and logging implementations

- **Reachable placement surface:** The cave allocator retains its live generic
  multi-anchor selection and its live owner-constrained single-word selection.
  These are the two algorithms called by current lowering. It no longer also
  carries a third owner-constrained variable-size multi-anchor algorithm that
  no lowering path ever selected.
- **Completed deletion:** The uncalled
  `claim_reachable_from_all_for_owner` implementation and the uncalled hook
  `consan_log_level_enabled` predicate are gone. A whole search, reachability,
  owner-filtering, distance-ranking, and overlap-retirement path is therefore
  no longer presented as supported placement behavior. Logging continues to
  use its live emission path and atomic configured level directly.
- **Deletion result:** Production source is 51 lines smaller. No replacement
  behavior, adapter, state, or test hook was added; symbol-level reference
  auditing proved that neither function had a caller.
- **Completed checked-in gate:** The complete host gate passes 1,509 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 66.06
  seconds. E2E validation remains outside this work.

### Slice 5R: derive extended-barrier inventory from typed inputs

- **Semantic shape owner:** A pure typed predicate now decides whether program
  inventory must retain extended barrier pairs. Its inputs are exactly the
  observation request, debug override, and mutation request. Runtime bindings,
  target mechanics, and legacy lowering options cannot influence this semantic
  inventory choice.
- **Completed control-plane cutover:** The HSA automatic-report path no longer
  constructs a complete `ConSanOptions` merely to ask this question and no
  longer names `ConSanOptions` or `LegacyOptionsAdapter` at all. The cutover
  also exposed and deleted the raw-options mutation enable/disable overloads,
  their 11-entry pointer-to-member table, and its alias; no caller remained
  after hook mutation control moved to `MutationRequest`.
- **Contract coverage:** A focused truth-table test covers the four barrier
  mutations, unmatched-barrier abort, the Sampled-plus-barrier conjunction,
  and the negative Record/Replay and untracked-barrier cases. The existing
  pristine-inventory test still proves the resulting choice changes actual
  extended-pair inventory without installing a replacement.
- **Deletion result:** Production source is 25 physical lines smaller despite
  the documented typed predicate. No compatibility projection or replacement
  mutation table was introduced.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 64.97
  seconds. E2E validation remains outside this work.

### Slice 5S: delete the hidden MOI retry result capsule

- **Single artifact ownership:** The automatic-report retry now consumes the
  `ProgramInventory`, observation plan, coverage ledger, mutation artifacts,
  resource plans, and patch records already owned by `TransformResult`. It
  constructs a fresh compatibility-lowerer input explicitly; it neither stores
  a second broad result nor relies on the unspecified contents of moved-from
  vectors and aggregates after publication.
- **Completed compatibility deletion:** The private
  `optional<ConSanResult>` and the publication flag that requested its retention
  are gone. A single private provenance bit still distinguishes the named
  pristine-inventory entry point from an ordinary unbound transform. It grants
  the consumptive retry no data of its own, and runtime demotion clears it.
- **Contract coverage:** The typed retry parity test requires the reconstructed
  path to match direct lowering in observation plan, coverage, replacement
  bytes, patches, resources, fault sites, barrier-move destinations, and fault
  plans. A separate negative test proves that a superficially similar ordinary
  result remains ineligible without pristine-inventory provenance.
- **Deletion result:** Production source is two physical lines smaller, and
  each retained report-sizing result loses an entire optional
  `ConSanResult`—including all of its vectors and mutable mechanism state—in
  exchange for one boolean marker. No alternate retry inventory type was
  introduced.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 64.44
  seconds. E2E validation remains outside this work.

### Slice 5T: move failure policy entirely to installation

- **Policy-independent transformation:** Preflight now records `Blocked` when
  static code-object facts make a kernel unsafe to transform. `Blocked` is a
  transform fact that produces `Unsupported`; it never decides whether the
  pristine image is loaded or the code-object load is rejected. Only
  `TransformResult::install_action` combines the completed static outcome with
  runtime `fail_closed` policy.
- **Completed compatibility deletion:** `ConSanOptions::fail_closed`, its
  `LegacyOptionsAdapter` projection and field-inventory entry, and the
  adapter's entire `RuntimePolicy` parameter are gone. The old `Reject`
  preflight state and error-producing branch are replaced by the
  policy-neutral `Blocked` state and warning. Hook summary diagnostics report
  blocked transforms rather than presenting runtime rejection as an analysis
  fact.
- **Contract coverage:** A compile-time assertion prevents `fail_closed` from
  returning to raw lowering options. A focused preflight test exercises a
  decoded but unsupported DS operation and requires `Blocked` plus an
  `Unsupported` result. A pipeline test runs the same invalid image under both
  runtime policies, requires identical identities, stages, inventories,
  observation and coverage artifacts, outcomes, issues, warnings, and mutation
  facts, then proves that only their installation actions differ.
- **Deletion result:** Production source is 14 physical lines smaller despite
  adding the explanatory preflight type contract. No replacement option,
  policy adapter, or mirrored rejection flag was introduced.
- **Completed checked-in gate:** The complete host gate passes 1,512 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 67.86
  seconds. E2E validation remains outside this work.

### Slice 5U: delete copied branch-routing telemetry

- **Single planning-fact owner:** `BranchOnlyRelayPlanOutcome` remains the
  typed result of a direct-router planning attempt. It owns the selected
  strategy, route, planning-work counters, and structured failure when no
  route exists. Lowering no longer copies those facts into a second mutable
  event log attached to the broad transform result.
- **Completed compatibility deletion:** The result-wide MOI and LDS routing
  logs, their reservoir-footprint summaries, all record-and-aggregate helper
  functions, and the reservoir set's telemetry projection are gone. These
  fields had no runtime consumer; they exposed the chronology of the current
  routing implementation solely so integration tests could reconstruct it.
- **Contract coverage:** Direct-router tests continue to validate every field
  of the typed planning outcome, including randomized oracle agreement,
  deterministic selection, owner provenance, failures, strategies, and work
  limits. Integration tests continue to validate the behavior that survives a
  router replacement: emitted patch topology and bytes, warnings, rollback,
  resource limits, and deterministic output. Two tests whose only contract was
  the deleted telemetry representation are removed.
- **Deletion result:** Production source is 261 physical lines smaller, and
  tests are 171 physical lines smaller. No replacement result field, event
  recorder, aggregation API, or compatibility projection was introduced.
- **Completed checked-in gate:** The focused router and integration gate passes
  all 112 tests. The complete host gate passes 1,510 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878
  simulator-device tests across the five supported targets pass in 63.80
  seconds. E2E validation remains outside this work.

### Slice 5V: keep bounded-work measurements at planner boundaries

- **Semantic boundary:** Planning allowances remain explicit transform policy,
  and every bounded planner still fails without publishing a partial result
  when it exhausts that allowance. Exact units consumed are component-local
  algorithm measurements, not semantic facts about a completed ConSan
  transform.
- **Completed compatibility deletion:** `ConSanResult` no longer carries an
  eight-counter aggregate for direct-reservoir discovery, SOPP routing, LDS
  relay-layout replay, and LDS convergence. Lowering no longer threads that
  aggregate through four engines or accumulates measurements across nested
  planner calls.
- **Shared component contract:** The prior two-counter SOPP-specific telemetry
  shape is now the target-neutral `PlanningWorkMeasurement`. SOPP routing and
  direct-reservoir discovery expose that one shared type only at their focused
  planner APIs. Their unit tests still prove exact charging, saturation,
  exhaustion, accumulation, and exact-limit rollback without exporting those
  mechanics through the transform result.
- **Integration contract:** ConSan integration tests continue to require the
  observable consequences of bounded planning: explicit exhaustion errors, no
  partial patches, transactional rollback, deterministic replacement bytes,
  selected reservoir topology, and successful operation with normal limits.
- **Deletion result:** Production source is 24 physical lines smaller despite
  adding the documented shared measurement type; tests are 14 physical lines
  smaller. No replacement result field or cross-planner aggregator was added.
- **Completed checked-in gate:** The focused bounded-planning gate passes all
  30 tests. The complete host gate passes 1,510 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878
  simulator-device tests across the five supported targets pass in 68.62
  seconds. E2E validation remains outside this work.

### Slice 5W: make full active-EXEC observation an invariant

- **One contract, not a one-choice option:** Every target-neutral probe intent
  observes every lane active at its guest instruction. This is now stated by
  `ConSanProbeIntent` itself instead of encoded as a `lane_mask` field whose
  enum admitted exactly one valid value.
- **Completed speculative deletion:** The single-value
  `ConSanLaneMaskPolicy`, its iterable array, diagnostic formatter, validation
  helper, intent field, and every forced initializer are gone. No current
  engine selected or consumed an alternative; the abstraction existed only to
  reserve a possible future extension point.
- **Contract coverage:** Observation-plan validation still rejects every
  malformed property that can vary, and access, barrier, atomic, fence,
  evidence, pipeline, and hook tests still exercise the resulting intents.
  Device tests continue to cover actual lane-sensitive behavior; deleting an
  unvarying metadata field does not change emitted EXEC handling.
- **Deletion result:** Production source is 38 physical lines smaller and tests
  are 16 physical lines smaller. No boolean, enum replacement, or compatibility
  default was introduced.
- **Completed checked-in gate:** The focused observation-policy gate passes all
  105 tests. The complete host gate passes 1,510 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878
  simulator-device tests across the five supported targets pass in 67.46
  seconds. E2E validation remains outside this work.

### Slice 5X: make admitted-intent coverage intrinsically required

- **One coverage contract, not an unused grading axis:** Every intent admitted
  by current ConSan policy is part of the selected engine contract. Any pending,
  resource-rejected, or placement-rejected intent therefore makes static
  coverage incomplete; no current engine produces or consumes an optional
  enhancement intent.
- **Completed speculative deletion:** `ConSanProbeRequirement`, its iterable
  array, diagnostic formatter, validation helper, `ConSanProbeIntent` field,
  and every forced `Required` initializer are gone. The coverage-ledger query
  is now `all_intents_instrumented()` and checks every entry directly, so the
  former `BestEffort` branch cannot silently excuse lost instrumentation.
- **Contract coverage:** The ledger unit still proves that pending and rejected
  outcomes are incomplete and only an instrumented outcome is complete.
  Access, barrier, atomic/fence, evidence, pipeline, analysis, and hook tests
  continue to exercise the intents without asserting an unvarying metadata
  field.
- **Deletion result:** Production source is 46 physical lines smaller and tests
  are 16 physical lines smaller. No boolean, default, compatibility alias, or
  replacement grading policy was introduced.
- **Completed checked-in gate:** The focused observation-policy gate passes all
  105 tests. The complete host gate passes 1,510 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878
  simulator-device tests across the five supported targets pass in 70.89
  seconds. In the 587-row physical-gfx950 run, 585 rows passed and two rows hit
  the 60-second process timeout; those timeout-only rows passed immediately
  when rerun alone in 1.22 and 0.44 seconds. E2E validation remains outside
  this work.

### Slice 5Y: let evidence alternatives own their fixed semantics

- **One discriminant, not four copies:** The closed
  `ConSanEvidenceRequirements` variant already distinguishes Record/Replay,
  Sampled, InlineShadow, and SuperCollider. Each named alternative now
  intrinsically documents its retention model, completeness effect, and
  executable binding lifetime instead of storing mutable tags that can only be
  equal to constants.
- **Completed speculative deletion:** `ConSanEvidenceSchema`,
  `ConSanEvidenceBoundedness`, and `ConSanEvidenceLossSeverity`, their iterable
  arrays and formatters, sixteen repeated tag fields, the schema and delivery
  visitors, and constant-comparison validation branches are gone. Runtime
  binding still requires executable-lifetime resources directly, and every
  alternative still fails closed on evidence loss through its documented
  contract.
- **Contract coverage:** Tests require each engine to publish the correct
  concrete variant alternative and continue to corrupt every varying runtime,
  capacity, inventory, and layout relationship accepted by `well_formed()`.
  Variant ordinal and a type's agreement with redundant copies of its own name
  are no longer treated as product behavior.
- **Deletion result:** Production source is 184 physical lines smaller and
  tests are 46 physical lines smaller. No replacement discriminator, common
  base, tag accessor, or compatibility field was introduced.
- **Completed checked-in gate:** The focused evidence/pipeline gate passes all
  56 tests. The complete host gate passes 1,510 tests with the two expected
  benchmark-object skips; all 194 HSA-hook tests pass; and all 2,878
  simulator-device tests across the five supported targets pass in 72.41
  seconds. The complete physical-gfx950 gate was exercised in the immediately
  preceding tranche; this metadata-only slice does not alter lowering or device
  bytes. E2E validation remains outside this work.

### Slice 5Z: consume normalized access origin directly

- **One decoder fact, not a lowering reclassification:** `ProgramInventory`
  already records whether an admitted access originated from a native LDS
  instruction, a FLAT instruction, or a global instruction that writes
  directly to LDS. The temporary MOI candidate now retains that exact
  `ConSanAccessOrigin`; resource planning and emission no longer translate it
  into their own three-way source enum.
- **Completed compatibility deletion:** `ConSanMoiCandidateSource`, its
  formatter and formatter test, and the candidate's derived `direct_to_lds`
  boolean are gone. Native-versus-FLAT address width, gfx1250 high-bank
  capture, spill recovery, gfx12 address materialization, and direct-to-LDS
  relocation all branch on the normalized origin fact. The origin enum moved
  earlier in the assembled type declarations so both inventory and the
  temporary lowering operand use the same definition.
- **Contract coverage:** Host tests now assert the exact normalized origin for
  native LDS, FLAT, and direct-to-LDS candidates, together with FLAT
  address-space provenance where it affects strict or likely admission. They
  no longer test agreement between two copied tags. Existing paired device
  contracts exercise all three origins across the five supported targets.
- **Deletion result:** Production source is 29 physical lines smaller. Tests
  add 14 net lines because they replace one compatibility-tag assertion with
  the stronger origin and provenance contract where applicable. No helper
  predicate, compatibility alias, or second origin representation was added.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 98.12
  seconds. The complete physical-gfx950 gate was exercised in Slice 5X; this
  representation cutover preserves the selected access set and emitted bytes.
  E2E validation remains outside this work.

### Slice 5AA: reuse normalized access operands in MOI lowering

- **One operand record, not selected field copies:** `ProgramInventory`
  already retains the decoder operands needed by access policy and temporary
  compatibility lowering in `ConSanAccessOperandFacts`. The temporary MOI
  candidate now embeds that record directly. Planning and emission consume its
  named normalized fields rather than a parallel set of destination, address,
  data, and raw-encoding members.
- **Narrower non-access boundary:** The runtime workgroup-selection gate only
  needs the guest instruction bytes that it may execute on the unselected
  path. It now accepts that byte span directly. Atomic, barrier, and fence
  lowering no longer fabricate partially initialized `ConSanMoiCandidate`
  objects merely to pass a file offset and size through this helper.
- **Contract coverage:** Host tests continue to assert normalized register and
  raw-encoding facts for native LDS, FLAT, and direct-to-LDS accesses, now
  through the shared operand record. Existing paired device contracts cover
  relocation, runtime sampling gates, operand overlap, gfx12 FLAT address
  materialization, and gfx1250 high-bank/direct-to-LDS cases across the five
  supported targets.
- **Deletion result:** Production source is 16 physical lines smaller. Eleven
  candidate operand members, their individual projection assignments, and
  three fake access-candidate construction paths are gone. No compatibility
  accessor or second operand schema was introduced; test source has no net
  line growth.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 74.04
  seconds. The complete physical-gfx950 gate was exercised in Slice 5X; this
  representation cutover preserves the selected access set and emitted bytes.
  E2E validation remains outside this work.

### Slice 5AB: consume normalized program-container identity

- **One attribution record, not four parallel fields:** Every normalized
  access already names the `ConSanProgramContainerRef` through which it was
  decoded. The temporary MOI candidate now retains that typed record directly
  instead of separately copying a name, kernel/function boolean, entry offset,
  and gfx1250 cluster-ID-use flag. The independently selected execution-owner
  descriptor remains a lowering decision because a shared helper may execute
  for more than one kernel.
- **Typed kernel identity:** `ConSanProgramContainerRef::is_kernel()` derives
  dispatchability from the enum and is the sole boolean query used by access
  planning. Its kernel and function results have focused unit coverage; no
  stored boolean can disagree with the normalized kind.
- **Completed incidental deletion:** The container-kind iterable table and
  formatter had no production consumer; only their own mechanism test called
  them. They are deleted rather than moved with the real type. Ownership
  grouping, name filtering, entry-state lookup, shared-helper routing, and
  cluster-aware runtime gates all consume the normalized container record.
- **Contract coverage:** Host tests assert kernel and function attribution on
  the normalized record. Existing paired device contracts exercise shared
  helpers, cross-kernel dispatch identity, multi-owner routing, and gfx1250
  cluster workgroup identity across every applicable engine and target.
- **Deletion result:** Production source is 19 physical lines smaller. Test
  source adds three net lines for the typed kernel/function query and updated
  normalized-record assertions. No compatibility accessor, copied boolean, or
  second container schema was introduced.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 74.25
  seconds. The complete physical-gfx950 gate was exercised in Slice 5X; this
  attribution cutover preserves selected sites and emitted bytes. E2E
  validation remains outside this work.

### Slice 5AC: validate and plan from admitted inventory identities

- **Final validation follows the production contracts:** Inline Shadow's
  workgroup-exact-shadow validator now resolves an admitted
  `ExactShadowAccess` intent back to its immutable access-inventory site. It no
  longer searches the temporary MOI lowering-candidate projection to recover
  the decoded access width.
- **One admitted-site source for liveness planning:** Exact-shadow candidate
  offsets were inserted independently into three liveness/resource site sets
  even though every such offset is already present in the observation plan's
  admitted probe intents. The duplicate insertion loops and capacity
  accounting are gone; synchronization events, resource plans, admitted
  intents, and kernel entries remain the explicit site sources.
- **Contract coverage:** Existing host contracts require MOI candidates to be
  exactly the admitted access intents and exercise final exact-shadow
  validation, including a deliberately corrupted publication. The paired
  device suite exercises the resulting liveness, automatic transient-register,
  and exact-shadow validation behavior across all five targets.
- **Refactoring result:** Production source is size-neutral: 12 lines of
  candidate-projection lookup and duplicate bookkeeping were removed and 12
  lines of explicit intent-to-inventory resolution were added. This slice is a
  dependency-direction cutover rather than a line-count win: final validation
  and liveness site selection no longer depend on temporary access-candidate
  telemetry, and no helper, compatibility accessor, or second identity record
  was introduced.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 79.19
  seconds. In the physical-gfx950 sweep, 585 of 587 cases passed; the two
  module-lifecycle launches that overlapped relinking the hook DSO failed to
  load that tool, then both passed immediately from the stable build (3.73 and
  3.72 seconds). All 587 physical cases are therefore qualified. E2E validation
  remains outside this work.

### Slice 5AD: delete unconsumed access-fact formatters

- **No API for unreachable diagnostics:** The LDS-access-kind and FLAT
  address-space-hint string formatters had no production caller. Their only
  callers were assertions that tested those two functions in isolation; no
  diagnostic, hook, policy, planner, or lowerer consumed their output.
- **Completed prototype deletion:** Both declarations and switch
  implementations are gone, together with the three self-justifying test
  lines. The underlying typed facts remain part of normalized inventory and
  continue to drive policy and lowering; only the unused string projection was
  removed.
- **Deletion result:** Production source is 34 physical lines smaller and test
  source is three lines smaller. No replacement formatter, generic enum
  printer, or compatibility API was introduced.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 75.72
  seconds. Slice 5AC completed the periodic physical-gfx950 gate immediately
  before this non-device formatter deletion. E2E validation remains outside
  this work.

### Slice 5AE: reuse the complete normalized access record in MOI lowering

- **No parallel access schema:** `ConSanMoiCandidate` now extends the exact
  immutable `ConSanAccessInventorySite` selected by observation policy. It no
  longer copies container identity, origin, access kind, FLAT address-space
  hint, physical and file offsets, instruction size, decoded width, operand
  facts, or mnemonic into ten independently maintained fields. The temporary
  lowering value retains only state that is genuinely created after admission:
  the gfx1250 bank mode, one selected execution-owner descriptor, and the
  current emitters' compact nonnegative access ranges.
- **One normalized snapshot:** Candidate construction copies the complete
  inventory value once. Every engine, placement path, emitter, validator, and
  hook-test fixture reads the inherited normalized facts. A focused host
  contract compares the complete candidate base with its matching inventory
  site for Record/Replay, Sampled, and Inline Shadow, so adding a future
  inventory field cannot silently recreate a partial projection.
- **Completed incidental deletion:** The first-light emitter called a helper
  that converted decoded width to a byte count, rejected malformed widths,
  and then never consumed the returned count. Admitted normalized ranges are
  already the authority for emitted byte spans. The unused helper and its
  redundant check are deleted rather than attached to the new representation.
- **Deletion result:** Production source is 11 physical lines smaller despite
  the documented candidate boundary; source plus tests is eight lines smaller.
  No compatibility alias, fallback field, or second construction path was
  retained.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 69.50
  seconds. Slice 5AC completed the periodic physical-gfx950 gate; this
  representation cutover preserves the admitted sites and emitted byte ranges.
  E2E validation remains outside this work.

### Slice 5AF: lower directly from normalized access ranges

- **One range representation:** The temporary `ConSanMoiAccessRange` type and
  the candidate-owned `access_ranges` vector are deleted. Record/Replay,
  Sampled, and Inline Shadow now consume the `ConSanAccessRange` values already
  owned by the inherited inventory site. Candidate construction no longer
  allocates, filters, or copies a second range collection.
- **Semantic fact versus lowering interpretation:** Inventory preserves the
  optional signed displacement encoded by the original access. Native LDS
  emission consumes that decoded displacement directly. FLAT emission first
  materializes the instruction's complete effective address and therefore
  uses lowering displacement zero instead of applying the encoded immediate a
  second time. `ConSanMoiCandidate::lowering_offset` documents that narrow
  mechanism rule without storing another value.
- **Contract coverage:** A focused host test proves both sides of the lowering
  rule, including a negative FLAT semantic displacement and a scaled native
  two-address displacement. Existing two-address tests now compare the
  candidate's inherited range vector directly with inventory and exercise the
  lowering query. The complete device matrix continues to cover subword, wide,
  two-address, group-FLAT, CDNA, RDNA, and gfx1250 high-bank forms across every
  applicable engine.
- **Deletion result:** Production source is 17 physical lines smaller and
  source plus tests is 15 lines smaller. The second range type, its vector
  allocation, conversion loop, and duplicate range-equality scaffolding are
  gone; no cached offset or compatibility view replaces them.
- **Completed checked-in gate:** The complete host gate passes 1,511 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 65.70
  seconds. Slice 5AC completed the periodic physical-gfx950 gate; this cutover
  emits the same normalized byte widths and lowering displacements. E2E
  validation remains outside this work.

### Slice 5AG: delete SuperCollider's reverse access projection

- **One policy input:** SuperCollider support policy now asks whether the
  normalized `ConSanAccessInventorySite` can be lowered. It no longer rebuilds
  a `ConSanLdsSite` or `ConSanFlatSite`—including raw encoding, ownership, and
  identity fields that the yes/no query never consumed—after shared inventory
  has already normalized the same instruction.
- **Narrow remaining seam:** The support implementation is expressed in terms
  of semantic access shape and operands. The still-unmigrated SuperCollider
  emitters retain internal legacy-site forwarding overloads, but policy and its
  tests can no longer reach the legacy-site API. A later complete emitter
  cutover can delete those overloads with the legacy container vectors.
- **Contract coverage:** The cross-target D16 load and subword-store tests now
  query support through normalized inventory and still require valid emitted
  SuperCollider patches. The access-policy suite covers all four engines,
  multi-range admission, typed exclusions, alias coalescing, provenance,
  target capability, and deterministic non-mutation of inventory.
- **Deletion result:** Production source is nine physical lines smaller and
  source plus tests is three lines smaller. Forty-eight lines of reverse
  projection and the two public legacy-site queries are gone; no replacement
  access record or copied field set was introduced.
- **Completed checked-in gate:** The complete host gate passes 1,511 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 65.50
  seconds. Slice 5AC completed the periodic physical-gfx950 gate. E2E
  validation remains outside this work.

### Slice 5AH: lower SuperCollider from shared access admission

- **One admission authority:** The shared observation plan now performs
  physical-site alias validation, provenance filtering, and instruction-shape
  support classification once. Both SuperCollider lowerers derive their
  deterministic emission lists from the plan's admitted physical file offsets
  instead of independently repeating those decisions over the legacy site
  vectors.
- **Preserved execution ownership:** Identical kernel/function views of one
  physical instruction still merge their execution-owner descriptors before
  lowering. Conflicting aliases fail closed while constructing the shared
  inventory and plan, before either lowerer can observe an admitted offset.
  The lowerers retain legacy site records only for emitter and resource
  mechanics that have not yet moved to normalized accesses.
- **Contract coverage:** Existing focused tests require conflicting physical
  aliases to fail closed, identical aliases to patch only once for every
  owner, incomplete owner context to remain unsupported, and non-group aliases
  outside the selected FLAT provenance contract not to poison strict mode.
  Those tests pass through the new single-admission path for both FLAT and LDS.
- **Deletion result:** Production source is 85 physical lines smaller. The
  second pair of physical-site canonicalizers, duplicate applicability and
  support scans, semantic-equality helpers, and internal legacy-site support
  forwarding overloads are deleted; no parallel compatibility path replaces
  them.
- **Completed checked-in gate:** The complete host gate passes 1,511 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 73.75
  seconds. Slice 5AC completed the periodic physical-gfx950 gate. E2E
  validation remains outside this work.

### Slice 5AI: lower SuperCollider FLAT from normalized inventory

- **No FLAT compatibility record in the engine:** The SuperCollider FLAT
  lowerer now retains pointers to immutable `ConSanAccessInventorySite`
  records. Instruction identity, decoded width, provenance, operands,
  container attribution, and execution owners all come from that one
  inventory; the lowerer no longer copies, merges, or reads `ConSanFlatSite`
  values from kernel and function containers.
- **One alias decision:** Shared policy admits one physical instruction and
  rejects conflicting aliases. Lowering selects the first deterministic
  normalized representative by file offset; execution-owner analysis has
  already attached the complete owner set to every identical alias. A kernel
  descriptor remains an explicit owner of its own attributed record, and
  later resource plans can still contribute additional proven owners.
- **Debug-path convergence:** The destructive direct-FLAT-trap probe now also
  discovers candidates from normalized inventory. Its legacy support helper
  is deleted, so no remaining code in the FLAT lowerer reads the legacy FLAT
  vectors.
- **Contract coverage:** Focused tests cover subword and wide FLAT operations,
  strict and likely provenance, scratch overlap and spill selection, shared
  helper ownership, one physical patch for aliased kernels, inconsistent alias
  rejection, resource rejection, and the direct-trap debug path. Those tests
  exercise the normalized record directly; no new compatibility-oriented test
  was introduced.
- **Deletion result:** Production source is 13 physical lines smaller. The
  copied canonical FLAT-site record, duplicate container traversal and owner
  merge, and legacy direct-trap support helper are gone.
- **Completed checked-in gate:** The complete host gate passes 1,511 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 74.62
  seconds. Slice 5AC completed the periodic physical-gfx950 gate. E2E
  validation remains outside this work.

### Slice 5AJ: lower SuperCollider native LDS from normalized inventory

- **Complete access cutover:** The native-LDS lowerer now carries immutable
  `ConSanAccessInventorySite` pointers through owner validation, liveness and
  scratch planning, cave and relay selection, gfx1250 two-address expansion,
  instruction emission, patch publication, and coverage accounting. Together
  with Slice 5AI, no SuperCollider source reads `ConSanLdsSite`,
  `ConSanFlatSite`, or the legacy container access vectors.
- **One owner query:** A shared helper derives the complete, sorted execution
  owner descriptors from normalized analysis and explicit kernel attribution.
  FLAT and native-LDS lowering use that query instead of carrying independently
  merged owner vectors. Alias admission remains owned by shared policy; the
  lowerers select one deterministic normalized representative per physical
  file offset.
- **Normalized resource mechanics:** Scratch overlap, tuple alignment,
  descriptor-growth headroom, compare-source selection, subword handling, and
  two-address decoding now consume normalized operands and widths. The old
  `legacy_scratch_search_start` layer, two one-consumer scratch wrappers, and
  legacy two-address site overloads are deleted. Sync analysis calls the
  remaining mnemonic classifier directly.
- **Shared FLAT/LDS placement protection:** The native-LDS lowerer now builds
  its selected-container and future-FLAT protection indexes from normalized
  accesses too. This removes its last indirect dependency on the legacy FLAT
  and LDS vectors without changing test-only container selection semantics.
- **Contract coverage:** Focused tests cover complete and missing execution
  owners, aliased physical sites, cross-target tuple rules, subword and D16
  operations, wide and two-address forms, gfx1250 high-bank and expanded-offset
  forms, scratch spills, cave/relay routing, and coverage-ledger outcomes.
  Existing paired device workloads exercise the same native-LDS mechanisms on
  all five targets, so no compatibility-shape test was added.
- **Deletion result:** Production source is seven physical lines smaller
  despite replacing terse legacy fields throughout the complete lowerer. The
  copied canonical native-LDS record and container traversals are gone, and
  this slice introduces no fallback representation.
- **Completed checked-in gate:** The complete host gate passes 1,511 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,878 simulator-device tests across the five supported targets pass in 73.05
  seconds. An earlier cold-cache run timed out only the unrelated gfx950 Inline
  Shadow large-LDS pair; both passed in isolation before the clean full rerun.
  Slice 5AC completed the periodic physical-gfx950 gate. E2E validation remains
  outside this work.

### Slice 5AK: enrich execution ownership in normalized inventory

- **One enrichment target:** Control-flow analysis now writes execution-owner
  descriptors directly to `ConSanAccessInventorySite`. It no longer mutates
  the prototype LDS and FLAT container records and then asks the inventory
  builder to copy the same facts back into the normalized representation.
- **One normalized lifecycle:** Access decoding and normalization happen once,
  before synchronization and ownership analysis. Later stages enrich that
  stable inventory in place through a builder-only mutable capability; every
  published `ProgramInventory` view remains read-only. The duplicate final
  clear, allocation, container walk, instruction-range decode, and rebuild are
  deleted.
- **Shared-function trigger:** Ordinary SuperCollider requests ownership
  analysis by asking whether normalized access inventory contains a
  function-attributed site. The decision no longer depends on the prototype
  container vectors that its lowerers have already stopped consuming.
- **Contract coverage:** Direct-call and wide-literal indirect-call tests now
  assert owner recovery through normalized function accesses. Real-code-object
  inventory coverage requires every normalized kernel access to carry the
  owning kernel descriptor rather than comparing against a mutable prototype
  field. The builder-only normalization test still independently covers value
  preservation before the obsolete source field is deleted.
- **Deletion result:** Production source is six physical lines smaller. The
  two-container owner mutation loop and a complete second inventory rebuild
  are gone; the new mutable accessor is confined to construction and documents
  the only permitted post-decode enrichment.
- **Completed checked-in gate:** Focused inventory and shared-function
  ownership tests pass. The complete host gate passes 1,510 tests with the two
  expected benchmark-object skips; all 194 HSA-hook tests pass; and all 2,908
  simulator-device tests across the five supported targets pass in 64.55
  seconds. Slice 5AC completed the periodic physical-gfx950 gate. E2E
  validation remains outside this work.

### Slice 5AL: delete prototype access-owner state

- **One owner representation:** `ConSanLdsSite` and `ConSanFlatSite` no longer
  carry execution-owner descriptor vectors. Ownership is a control-flow fact
  of the normalized access inventory, not an instruction-decoder field and not
  state that engines may read from prototype containers.
- **No reverse copy:** Inventory normalization no longer copies owner state
  from either legacy record. Focused evidence and pipeline fixtures seed the
  normalized owner fact explicitly, while the pure decode-normalization test
  requires that ownership remain absent until the owner-analysis stage.
- **Deletion result:** Production source is six physical lines smaller. Two
  duplicate vectors, their comments, and both copy assignments are gone; no
  alias or compatibility accessor replaces them.
- **Completed checked-in gate:** The 73 focused inventory,
  evidence-requirement, and pipeline tests pass. The complete host gate passes
  1,510 tests with the two expected benchmark-object skips; all 194 HSA-hook
  tests pass; and all 2,908 simulator-device tests across the five supported
  targets pass in 59.84 seconds. Slice 5AC completed the periodic
  physical-gfx950 gate. E2E validation remains outside this work.

### Slice 5AM: derive LDS fault inventory from normalized accesses

- **One fault-analysis input:** Exact LDS-address fault discovery now consumes
  `ConSanAccessInventorySite`, including the direct-to-LDS distinction,
  normalized operands, decoded width, target support, and stable container
  attribution. It no longer scans either prototype container LDS vector.
- **Self-contained container attribution:** Normalized container references now
  retain their pristine text file offset as well as their executable entry
  offset. That fact lets gfx1250 fault analysis recover VGPR-MSB mode without a
  reverse lookup into mutable decode containers, and is independently covered
  by the inventory value-record test.
- **Debug-path convergence:** The destructive LDS-to-`endpgm` proof probe also
  selects a normalized native-LDS access and reads its physical identity and
  instruction width. Its only container query is the still-live kernel
  preflight status; no fault or placement source now accepts `ConSanLdsSite`.
- **Contract coverage:** The cross-target exact-address fault test continues to
  require stable discovery, identity selection, operand diagnostics, owner
  recovery, mutation, and final validation on CDNA3, CDNA4, RDNA4, and CDNA5.
  The gfx1250 two-address exclusion and MOI composition tests remain green,
  together with all focused inventory and proof-probe cases.
- **Temporary size cost and deletion point:** This cutover adds 23 production
  lines because normalized fault construction temporarily coexists with the
  container access vectors and their normalization bridge. It introduces no
  fallback. The same short series must next migrate provenance,
  reattribution, pruning, and decode publication, then delete both legacy
  access structs, all four container vectors, and the bridge; that deletion is
  the required payoff before this tranche is complete.
- **Completed checked-in gate:** All 217 focused inventory, fault, LDS, and
  proof-probe tests pass. The complete host gate passes 1,510 tests with the two
  expected benchmark-object skips; all 194 HSA-hook tests pass; and all 2,908
  simulator-device tests across the five supported targets pass in 68.06
  seconds. Slice 5AC completed the periodic physical-gfx950 gate. E2E
  validation remains outside this work.

### Slice 5AN: prune inferred ranges in normalized inventory

- **One pruning authority for accesses:** The zero-sized-symbol CFG pass now
  erases unreachable normalized access records directly. It no longer prunes
  the prototype LDS and FLAT vectors and relies on a later rebuild to express
  that decision.
- **Stable construction order:** Normalization now runs before inferred-range
  pruning. The pruned vector is the inventory published to policy, ownership,
  faults, and lowerers; no later access rebuild can restore a rejected tail.
  Non-access barrier, atomic, fence, and ordinary-memory records remain on
  their existing container path until their own inventory migrations.
- **Typed range provenance:** `ConSanProgramContainerRef` explicitly records
  whether its source symbol range was inferred from a zero size. The inventory
  value test covers that fact, and the pruning pass no longer reverse-looks up
  mutable containers to decide applicability.
- **Contract coverage:** Both final and bounded zero-sized-symbol tests still
  exclude unreachable Record/Replay accesses, while an explicitly sized
  unreachable tail remains conservatively inventoried. The focused inventory
  and inferred-range gate passes all 20 cases.
- **Temporary size cost and deletion point:** This step adds ten production
  lines and brings the short legacy-access-removal series to 33 net-added
  lines. That debt is bounded to the same tranche as Slice 5AM: migrate
  preapplied-range attribution and decode publication, then delete the legacy
  LDS/FLAT types, container vectors, and normalization bridge before declaring
  the tranche complete.
- **Completed checked-in gate:** The complete host gate passes 1,510 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,908 simulator-device tests across the five supported targets pass in 59.38
  seconds. Slice 5AC completed the periodic physical-gfx950 gate. E2E
  validation remains outside this work.

### Slice 5AO: reattribute staged accesses in normalized inventory

- **One staged-access authority:** Fault-composition ranges now transfer their
  existing access records to the dispatchable owner directly in the normalized
  inventory. Composition no longer moves legacy LDS or FLAT records among
  kernel and function containers.
- **Late-code merge:** The same builder operation normalizes instructions that
  exist only in an installed trampoline and merges them without duplicating a
  physical access already decoded through an overlapping symbol. A focused
  unit test covers owner transfer, preservation outside the staged range,
  native-LDS and FLAT additions, and duplicate suppression.
- **Single publication boundary:** Initial normalized publication now follows
  call-provenance relay and precedes staged-range reattribution. No later raw
  rebuild can overwrite the transferred owner.
- **Temporary size cost and deletion point:** This step adds 32 production
  lines and brings the short legacy-access-removal series to 65 net-added
  lines. The new builder operation replaces composition's legacy movement but
  still accepts a temporary decoded container. Decode must publish normalized
  records directly next; then the legacy LDS/FLAT types, all four container
  vectors, and the conversion bridge must be deleted before this tranche is
  complete.
- **Completed checked-in gate:** The focused inventory and representative
  preapplied composition gate passes all four cases. The complete host gate
  passes 1,511 tests with the two expected benchmark-object skips; all 194
  HSA-hook tests pass; and all 2,908 simulator-device tests across the five
  supported targets pass in 69.01 seconds. Slice 5AC completed the periodic
  physical-gfx950 gate. E2E validation remains outside this work.

### Slice 5AP: collapse legacy decoded access records

- **One access vocabulary:** The separate `ConSanLdsSite` and
  `ConSanFlatSite` records are deleted. Decode/provenance analysis now stages
  the same typed `ConSanAccessInventorySite` record that immutable inventory
  publishes, distinguished by its explicit origin and nested operand facts.
- **One container vector:** Kernel and function records each have one decoded
  access vector instead of parallel LDS and FLAT vectors. All decoder,
  provenance-relay, hook-summary, and test consumers use origin and normalized
  fields rather than selecting a representation by vector name.
- **Smaller completion boundary:** Inventory publication now completes one
  staged access shape with stable code-object identity, container attribution,
  address-space provenance, logical ranges, and typed exclusions. The two
  legacy conversion overloads and their field-by-field copies are gone.
- **Contract coverage:** Decoder tests now assert the shared typed fields
  directly, and the real-code-object inventory test checks that staged decoder
  facts survive completion into the immutable inventory. Existing range,
  provenance, exclusion, alias, direct-to-LDS, and policy tests continue to
  cover the completed record.
- **Deletion accounting:** This step deletes 54 net physical production lines,
  or 69 net non-comment code lines; the difference is the expanded construction
  contract on the shared type. The Slice 5AM--5AP series has therefore repaid
  its 65-line code scaffolding and is four non-comment code lines smaller than
  its starting point. It remains 11 physical lines larger while the two
  container staging vectors are still published alongside the completed
  inventory; consuming and deleting that staging seam is the next payoff.
- **Completed checked-in gate:** The complete host gate passes 1,511 tests with
  the two expected benchmark-object skips; all 194 HSA-hook tests pass; and all
  2,908 simulator-device tests across the five supported targets pass in 63.60
  seconds. Slice 5AC completed the periodic physical-gfx950 gate. E2E
  validation remains outside this work.

### Slice 5AQ: consume decoded access staging at publication

- **One published authority:** `ProgramInventory::access_sites()` is now the
  only access collection that survives inventory construction. Kernel and
  function access vectors are explicitly builder-only decode staging; the
  publication boundary completes their code-object identity, container,
  range, exclusion, and execution-owner facts, appends the completed records,
  and clears the staging vectors.
- **Incremental revisions remain sound:** Completed records are retained when
  an immutable inventory is deep-copied into a new builder revision. A revision
  can stage and publish a newly analyzed container without reconstructing
  earlier records, and repeated publication is idempotent because consumed
  staging cannot be appended twice.
- **Consumers use the contract:** The HSA summary and post-transform tests no
  longer inspect per-container staging. They consume the normalized published
  inventory and use its typed container attribution when separating kernel and
  function statistics.
- **Contract coverage:** Inventory tests prove staging consumption, repeated
  publication, preservation of an immutable source revision, append-only
  publication in a derived revision, and completion of every access fact from
  a real code object. Existing decoder, provenance, range, alias, policy, and
  device tests now read the same published facts used by production.
- **Deletion accounting:** Production changes add 37 and delete 36 physical
  lines. The one-line physical increase is documentation and the explicit
  consume operation, while the duplicate published state and all of its
  consumer paths are gone. The remaining per-container vectors have only a
  builder-staging lifetime; routing decode directly into a dedicated inventory
  construction capability is the next boundary that can delete them entirely.
- **Completed checked-in gate:** The host gate passes all 1,512 runnable tests
  with the two expected benchmark-object skips; all 194 HSA-hook tests pass;
  and all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 63.43 seconds. Slice 5AC completed the
  periodic physical-gfx950 gate. E2E validation remains outside this work.

### Slice 5AR: delete function-owned access staging

- **Single construction owner:** `ConSanFunctionInfo` no longer owns an access
  vector. Function decode and bounded call-provenance relay write normalized
  records into builder-owned inventory staging, while the function value keeps
  only container metadata and statistics. Kernel and function attribution use
  the same documented `ConSanProgramContainerRef` construction helpers.
- **Transactional relay:** Each call-provenance pass constructs a replacement
  function-access projection and publishes it only after the pass succeeds.
  Decode or relay exceptions restore both function metadata and the original
  access staging, preserving the existing all-or-nothing composition contract.
- **Deterministic publication:** Publication temporarily removes pending
  function records from the global staging collection, publishes kernel
  records in historical container order, and then completes the function
  records. Already-completed revision prefixes remain in place. A completed
  function record has a valid code-object identity, so repeated publication
  cannot consume it again.
- **Contract coverage:** Inventory and observation-plan tests construct
  function-owned access facts only through builder staging. They prove
  physical alias canonicalization across kernel and function attribution,
  deterministic order, conflict behavior after attribution changes, and
  preservation of a completed immutable revision.
- **Temporary size cost and next deletion:** This slice adds 30 net production
  lines and nine net test lines. The cost is explicit attribution and strong
  exception safety around a seam that previously relied on a mutable field.
  Deleting `ConSanKernelInfo::access_sites` next removes the final container
  staging vector and permits publication to become a single completion pass
  over builder-owned access facts.
- **Completed checked-in gate:** The host gate passes all 1,512 runnable tests
  with the two expected benchmark-object skips, and all 194 HSA-hook tests
  pass. The five-target simulator gate passed 2,907 of 2,908 rows in one
  64-way run; the sole gfx942 Sampled module-lifecycle row hit the shared
  60-second timeout under contention and passed alone in 0.43 seconds. Slice
  5AC completed the periodic physical-gfx950 gate. E2E validation remains
  outside this work.

### Slice 5AS: delete kernel-owned access staging

- **One access owner:** `ConSanKernelInfo` no longer owns an access vector.
  Kernel and function decode now construct `ConSanAccessInventorySite` records
  directly in the builder-owned program inventory. Container attribution is
  assigned when each record is created, so no later pass has to recover its
  owner from the vector that happened to contain it.
- **Single in-place publication pass:** Publishing the inventory now completes
  the code-object identity of each still-unpublished access record in place.
  Already-completed records are retained unchanged, making repeated publication
  idempotent without removing, sorting, or reinserting records according to
  temporary container ownership.
- **Bounded temporary decode state:** Function call-provenance relay still needs
  a function-local replacement vector while it computes an all-or-nothing
  result. That vector is an algorithm-local transaction, not another stored
  representation. Successful decode appends kernel records followed by
  function records in the historical deterministic order; decode failure also
  preserves the successfully decoded function prefix. Preapplied-range decode
  similarly passes a local decoded span directly to unique inventory
  reattribution.
- **Contract coverage:** The real decoder, inventory, observation-policy,
  evidence-requirement, pipeline, SuperCollider, and HSA-hook fixtures now stage
  access facts through the sole builder-owned collection. Focused inventory
  tests prove container attribution before publication, in-place identity
  completion, idempotent publication, physical-alias canonicalization,
  preapplied-range reattribution, deterministic order, and preservation of
  already-published revisions.
- **Deletion accounting:** Production changes add 78 and delete 85 physical
  lines, a net deletion of seven lines. Tests add 81 and delete 65 lines while
  replacing direct writes to the removed representation with its public
  construction contract and strengthening state-transition coverage.
- **Completed checked-in gate:** The host gate passes all 1,512 runnable tests
  with the two expected benchmark-object skips; all 194 HSA-hook tests pass;
  and all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 65.97 seconds. The periodic physical-gfx950
  gate also passes all 593 tests in 435.17 seconds. E2E validation remains
  outside this work.

### Slice 5AT: delete copied fence-event projections

- **One synchronization-event owner:** `ConSanMoiFenceCandidate` is now only a
  typed association between the authoritative fence and communication events.
  It no longer copies either event's string identity, container, kernel flag,
  instruction location and size, address source and displacement, or raw
  scope. Those facts remain owned by the immutable synchronization inventory.
- **Typed joins at policy and lowering boundaries:** Fence policy groups and
  filters associations through their `SemanticSiteId` references. Record/Replay
  lowering resolves those references once and carries a borrowed pointer to the
  authoritative fence event in its short-lived operand-rich adapter. Missing or
  stale references fail closed; no string identity is reconstructed as an
  alternate join key. `SynchronizationInventoryView::find_event` is the shared
  lookup contract, and lowering does not wrap it in another index.
- **Contract coverage:** Focused analysis, policy, inventory, fault, Inline
  Shadow, and Record/Replay tests now inspect fence and communication facts
  through the typed join. They cover successful and absent event lookup,
  qualified and unqualified associations, conflicting aliases, stale typed
  identities, buffer-address policy, and ordinary load/store communication.
  The cross-target device suite retains paired correct/incorrect fence and
  atomic publication contracts.
- **Deletion accounting:** Production changes add 110 and delete 123 physical
  lines, a net deletion of 13 lines. Tests add 73 and delete 67 lines while
  replacing assertions on copied fields with assertions on the authoritative
  event records and their typed associations.
- **Completed checked-in gate:** The host gate passes all 1,512 runnable tests
  with the two expected benchmark-object skips; all 194 HSA-hook tests pass;
  and all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 84.78 seconds. Slice 5AS completed the
  periodic physical-gfx950 gate. E2E validation remains outside this work.

### Slice 5AU: delete the unconsumed communication-address recipe

- **Dead projection removed:** `ConSanCommunicationAddressRecipe` and its two
  private enums had no production consumer. Semantic analysis nevertheless
  built one record per atomic or ordinary-memory sequence, and program
  inventory stored and copied those records. Only tests read the result. The
  type, storage, build capability, construction call, and all references are
  deleted rather than preserving an aspirational interface beside the live
  lowering path.
- **Duplicate analysis removed:** Recipe construction separately recovered
  communication operands, rebuilt a CFG liveness analysis, revalidated owner
  descriptors, selected a post-sequence scratch pair, and encoded a second
  support verdict. Actual Record/Replay fence lowering already derives its
  address and resource plan at the correct insertion point, including the
  pre-guest capture required when an ordinary acquire destroys its address.
  Removing the recipe therefore deletes redundant work and eliminates the risk
  that an unused verdict disagrees with the mechanism that executes.
- **Behavioral contracts retained:** Tests still prove ordinary release/acquire
  association, exact pre-guest capture and patch extent for destructive
  acquire sequences, atomic and fence address publication, liveness/resource
  rejection in the live planners, and paired correct/incorrect behavior on all
  five targets. Three host tests that asserted only fields of the unreachable
  recipe were deleted; mechanism-only recipe assertions embedded in otherwise
  behavioral tests were removed without weakening their live outcomes.
- **Deletion accounting:** Production changes add three and delete 316 physical
  lines, a net deletion of 313 lines. Tests delete 132 lines and add none. The
  inventory and design documentation now describe only synchronization facts
  that a production consumer uses.
- **Completed checked-in gate:** The host gate passes all 1,509 runnable tests
  with the two expected benchmark-object skips; all 194 HSA-hook tests pass;
  and all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 82.40 seconds. Slice 5AS completed the
  periodic physical-gfx950 gate. E2E validation remains outside this work.

### Slice 5AV: delete the pristine-mutation result box

- **Direct consumptive retry input:** The internal mutation retry now accepts
  its optional pristine `ConSanResult` directly. The removed
  `ConSanPristineMutationInventory` had no invariant, behavior, or independent
  ownership: it contained exactly one `ConSanResult`, existed only in the
  implementation file, and was immediately unwrapped after one move.
- **No replacement abstraction:** The caller still transfers the already-built
  pristine inventory exactly once, and the callee still resets only the
  mutation-planning state before selecting and applying the late-bound fault.
  Provenance and inventory-shape authorization remain at the typed pipeline
  boundary; another internal wrapper would not strengthen either contract.
- **Behavioral coverage:** Existing focused tests exercise live and rejected
  late-bound fault retries, report-only retries, extended barrier-pair
  preservation, pristine provenance rejection, and retry equivalence with a
  fresh transform. They cover the ownership path that changed without retaining
  a test for the deleted box itself.
- **Deletion accounting:** Production changes add three and delete eight
  physical lines, a net deletion of five lines. No test or production behavior
  is added, renamed, or removed.
- **Checked-in gate:** The focused retry/provenance gate passes all ten tests;
  the host gate passes all 1,509 runnable tests with the two expected
  benchmark-object skips; and all 172 HSA-hook and hook-lifecycle tests pass.
  All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 84.62 seconds. Slice 5AS completed the
  periodic physical-gfx950 gate. E2E validation remains outside this work.

### Slice 5AW: delete the non-operative barrier qualification table

- **Claims without a consumer removed:** The barrier-mutation qualification
  enums, record, and target/form lookup function were linked into the ConSan
  implementation but were read only by host tests. No request validation,
  semantic admission, mutation planning, final validation, lowering, runtime
  policy, or diagnostic consulted them. Their `Proven`, `DeferredA1`, and
  topology values therefore could neither enable safe behavior nor prevent an
  unqualified operation.
- **Executable contracts remain authoritative:** The retained barrier tests
  still exercise decoded lifecycle admission, pair-scope mutation planning,
  exact rewritten instructions, transactional rejection, and final validation.
  Checked-in device tests retain the cross-target correct/incorrect behavioral
  contracts. Qualification status belongs in validation documentation until a
  real policy consumer needs a typed input; an isolated production lookup table
  is not evidence.
- **Deletion accounting:** Production deletes 34 physical lines and adds none.
  Tests delete 40 lines and add none: one test existed only to inspect the dead
  table, while three live mutation tests lose only adjacent assertions about
  the same non-operative metadata.
- **Checked-in gate:** The host gate passes all 1,508 runnable tests with the
  two expected benchmark-object skips; all 172 HSA-hook and hook-lifecycle
  tests pass; and all 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass in 82.98 seconds. Slice 5AS remains
  the latest periodic physical-gfx950 gate; E2E validation remains outside this
  work.

### Slice 5AX: delete the descriptive legacy-projection inventory

- **One executable projection:** `LegacyOptionsAdapter::adapt` is the only
  authoritative typed-request-to-prototype projection. The removed
  `projected_field_names` array was a second, manually maintained prose summary
  of that function. Production never read it, and its test could prove only
  that fourteen arbitrary strings were nonempty and unique—not that the
  adapter projected the right fields or that a new field was reviewed.
- **Behavioral coverage retained:** The adapter tests still construct every
  configuration family, verify the projected values and derived Inline Shadow
  mode, cover every mutation and bound-resource family, and prove that each
  adaptation returns a fresh value without mutating typed inputs. The adapter
  source and these executable assertions are the deletion inventory until the
  compatibility projection itself disappears.
- **Deletion accounting:** Production adds one comment line and deletes 24
  physical lines, a net deletion of 23 lines. Tests add one renamed test line
  and delete nine mechanism-only lines, a net deletion of eight lines. No
  replacement table or metadata API is introduced.
- **Checked-in gate:** All five focused adapter tests pass. The host gate passes
  all 1,508 runnable tests with the two expected benchmark-object skips; all
  172 HSA-hook and hook-lifecycle tests pass; and all 2,908 simulator-device
  tests across `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250` pass in
  73.83 seconds. Slice 5AS remains the latest periodic physical-gfx950 gate;
  E2E validation remains outside this work.

### Slice 5AY: delete write-only automatic-allocation flags

- **Selected resources remain the state:** The removed
  `automatic_moi_exec_save_sgprs` and `automatic_moi_dispatch_id_vgprs` booleans
  were set or cleared after register placement but never read by production.
  The actual optional EXEC-save and dispatch-ID registers, component-local
  assignments, spill modes, persistence proofs, and published resolved choices
  remain authoritative wherever later planning or emission needs them.
- **No mechanism-only test setup:** Twelve host fixtures set the EXEC-save flag,
  but no implementation consumed it and no assertion inspected it. Those
  assignments therefore could not select a path or strengthen coverage. The
  fixtures retain all concrete register, spill, owner, patch-byte, and
  diagnostic assertions that exercise their intended paths.
- **Deletion accounting:** Production deletes fifteen physical lines, including
  both fields, their comments, and ten writes. Tests delete twelve no-op setup
  lines. No replacement state or derived predicate is needed.
- **Checked-in gate:** The host gate passes all 1,508 runnable tests with the
  two expected benchmark-object skips; all 172 HSA-hook and hook-lifecycle
  tests pass; and all 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass in 81.33 seconds. Slice 5AS remains
  the latest periodic physical-gfx950 gate; E2E validation remains outside this
  work.

### Slice 5AZ: delete copied growth-budget inputs

- **Budget outputs only:** `ConSanPatchedImageGrowthBudget` no longer copies the
  original and current image byte counts into its returned value. The resolver
  uses both inputs to compute the operative total limit, existing growth,
  remaining growth, and already-exceeded status, but no production consumer
  read the copied inputs afterward.
- **Computed behavior remains covered:** The growth-policy tests still exercise
  absolute and percentage resolution, staged original-input accounting, exact
  admission and rejection boundaries, saturation, and distinct malformed or
  allocation failures. Two assertions that merely compared returned copies to
  the arguments have no behavioral content and are deleted.
- **Deletion accounting:** Production deletes four physical lines and adds
  none; tests delete two input-echo assertions and add none. No accessor or
  alternate stored representation replaces them.
- **Checked-in gate:** All thirteen focused growth-policy tests pass. The host
  gate passes all 1,508 runnable tests with the two expected benchmark-object
  skips; all 172 HSA-hook and hook-lifecycle tests pass; and all 2,908
  simulator-device tests across `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and
  `gfx1250` pass in 79.32 seconds. Slice 5AS remains the latest periodic
  physical-gfx950 gate; E2E validation remains outside this work.

### Slice 5BA: delete unused inventory-exclusion prose

- **Typed reasons only:** `ConSanInventoryExclusion` now stores only the
  machine-readable reason that access policy consumes. Its removed `detail`
  string was populated with one fixed sentence per reason, never rendered or
  otherwise read by production, and duplicated the explanation already owned
  by the documented reason enum.
- **Semantic coverage retained:** Program-inventory tests still exercise every
  exclusion reason produced by normalization, the complete/incomplete-site
  contract, value equality, and the mapping from each reason into access
  policy. Removing a test assignment to an unobserved string does not weaken
  those behavioral checks.
- **Deletion accounting:** Production deletes ten physical lines and adds none:
  one string field and its comment plus five fixed prose initializers. Tests
  delete one write and add none. No diagnostic accessor or alternate prose
  representation replaces them.
- **Checked-in gate:** All 27 focused inventory and access-policy tests pass.
  The host gate passes all 1,508 runnable tests with the two expected
  benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests pass; and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 76.43 seconds. Slice 5AS remains the latest
  periodic physical-gfx950 gate; E2E validation remains outside this work.

### Slice 5BB: delete test-only enum catalogues

- **No compiled test-support API:** Five production enums no longer carry a
  parallel constexpr value array and string-name switch used only by one
  generic unit-test assertion. Semantic-site domains, access origins,
  address-space facts, provenance facts, and inventory-exclusion reasons are
  represented once by their documented enum declarations and are consumed as
  typed values by real inventory and policy code.
- **Behavioral enum coverage remains:** Focused inventory tests exercise all
  three semantic domains, all three access origins, every normalized address
  space and provenance outcome, and every exclusion reason through their real
  builders and consumers. The generic exhaustive-name helper remains only for
  fence associations, whose iterable catalogue is itself used to drive
  eligibility and atomic/fence policy tests.
- **Deletion accounting:** Production deletes 139 physical lines and adds none
  across the site-identity, decoded-code-object, and program-inventory types.
  Tests delete thirteen lines and add one renamed test declaration, a net
  deletion of twelve lines. No test-only replacement catalogue is introduced.
- **Checked-in gate:** All eighteen focused program-inventory tests pass. The
  host gate passes all 1,508 runnable tests with the two expected
  benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests pass; and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 72.13 seconds. Slice 5AS remains the latest
  periodic physical-gfx950 gate; E2E validation remains outside this work.

### Slice 5BC: share the typed transform policy with lowering

- **One policy representation:** The temporary mutable `ConSanOptions` now
  inherits the production `TransformPolicy` instead of redeclaring its image
  growth, patch-count, and four planning-work limits. Existing lowerer code
  therefore reads and mutates the actual typed policy subobject while the
  remaining raw members are limited to other not-yet-migrated inputs and
  lowering-selected state.
- **One bounded adapter operation:** `LegacyOptionsAdapter` copies the complete
  policy subobject in one value-semantic assignment. The removed seven-field
  projection could drift whenever policy gained a field; the existing policy
  and adapter tests prove defaults, both growth forms, every work limit, expert
  patch limits, fresh outputs, and unchanged typed inputs.
- **Deletion accounting:** Production adds 47 and deletes 65 physical lines, a
  net deletion of eighteen. Most additions relocate the existing documented
  `TransformPolicy` definition before its temporary derived lowerer state; no
  second policy type, accessor layer, or fallback remains. Tests are unchanged.
- **Checked-in gate:** All six focused policy and adapter tests pass. The host
  gate passes all 1,508 runnable tests with the two expected benchmark-object
  skips; all 172 HSA-hook and hook-lifecycle tests pass; and all 2,908
  simulator-device tests across `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and
  `gfx1250` pass in 74.68 seconds. Slice 5AS remains the latest periodic
  physical-gfx950 gate; E2E validation remains outside this work.

### Slice 5BD: share the typed transform request with lowering

- **One request representation:** The temporary mutable `ConSanOptions` now
  inherits the production `ConSanRequest` instead of redeclaring the engine,
  owner-source, provenance, probe, owner-tracking, sampling, delay, and report
  marker inputs. Existing lowerer code therefore reads and mutates the actual
  typed request subobject while the remaining raw members are limited to other
  not-yet-migrated inputs and lowering-selected state.
- **One bounded adapter operation:** `LegacyOptionsAdapter` copies the complete
  request subobject in one value-semantic assignment. It then normalizes only
  the lowerer's temporary nonoptional flavor and derives the InlineShadow
  workgroup-shadow flag. The removed field-by-field projection could drift as
  the request contract evolved; the request-contract and adapter tests exercise
  every request family and preserve fresh outputs and unchanged typed inputs.
- **Deletion accounting:** Production adds 70 and deletes 109 physical lines, a
  net deletion of 39. Most additions relocate the existing documented
  `ConSanRequest` definition before its temporary derived lowerer state; no
  second request type, accessor layer, or fallback remains. Tests are
  unchanged.
- **Checked-in gate:** All nine focused request-contract and adapter tests pass.
  The host gate passes all 1,508 runnable tests with the two expected
  benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests pass; and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 71.21 seconds. Slice 5AS remains the latest
  periodic physical-gfx950 gate; E2E validation remains outside this work.

### Slice 5BE: share typed debug overrides with lowering

- **One debug representation:** The temporary mutable `ConSanOptions` now
  inherits the production `ConSanDebugOverrides` instead of redeclaring probe,
  diagnostic, kernel-filter, and explicit-register controls. Existing lowerer
  code therefore reads the actual typed debug subobject. The base also retains
  hook-only runtime assertions without exposing them as lowering mechanics.
- **Only semantic translations remain:** `LegacyOptionsAdapter` copies the
  complete debug subobject in one value-semantic assignment, then translates
  only `test_force_vgpr_spill` and `test_force_private_epoch` to the lowerer's
  intentionally shorter internal selector names. The removed thirteen-field
  projection could drift whenever debug controls changed.
- **Deletion accounting:** Production adds 63 and deletes 94 physical lines, a
  net deletion of 31. Most additions relocate the existing documented
  `ConSanDebugOverrides` definition before its temporary derived lowerer state;
  no second debug type, accessor layer, or fallback remains. The adapter test
  adds seven lines to prove the entire inherited value, including hook-only
  controls, as well as the two explicit name translations.
- **Checked-in gate:** All seven focused debug-contract and adapter tests pass.
  The host gate passes all 1,508 runnable tests with the two expected
  benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests pass; and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 74.07 seconds. Slice 5AS remains the latest
  periodic physical-gfx950 gate; E2E validation remains outside this work.

### Slice 5BF: share the typed mutation request with lowering

- **One mutation representation:** The temporary mutable `ConSanOptions` now
  inherits the production `MutationRequest` instead of redeclaring fault,
  selection, proof, and SuperCollider-perturbation controls. The typed helper
  predicates and pristine-inventory projection remain attached to that single
  representation, so lowerer storage cannot drift from the behavior validated
  at the public boundary.
- **Only derived lowerer state remains:** `LegacyOptionsAdapter` copies the
  complete mutation subobject in one value-semantic assignment and derives
  only the expensive barrier-move destination-inventory switch. Internal retry
  guards, inventory-shaping state, and the marker immediate remain ordinary
  lowerer state because they are not caller mutation semantics.
- **Deletion accounting:** Production adds 81 and deletes 184 physical lines,
  a net deletion of 103. Most additions relocate the existing documented
  `MutationRequest` definition before its temporary derived lowerer state; no
  second mutation type, accessor layer, or fallback remains. The adapter test
  adds three and deletes forty lines: one complete typed-value comparison now
  covers every mutation field, including reservation timeout and load
  occurrence, while one separate assertion proves the derived inventory flag.
- **Checked-in gate:** All eleven focused mutation-contract and adapter tests
  pass. The host gate passes all 1,508 runnable tests with the two expected
  benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests pass; and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 72.87 seconds. The periodic physical-gfx950
  gate passes all 593 tests in 486.24 seconds. E2E validation remains outside
  this work.

### Slice 5BG: share typed runtime facts and bindings with lowering

- **One runtime representation:** The temporary mutable `ConSanOptions` now
  inherits `RuntimeCapabilities` and `BoundRuntimeResources`. The lowerer reads
  the queried workgroup-LDS aperture directly from the runtime-fact subobject
  and reads report addresses, layout, generation, dispatch identity, and
  lifetime scope directly from the binding subobject; it no longer owns renamed
  or partial copies of either contract.
- **Complete bounded copies:** `LegacyOptionsAdapter` assigns each complete
  typed value once. The capability test now gives every runtime fact a
  nondefault value and compares the inherited value as a whole; the binding
  test does the same for every resource field. This also makes fields unused by
  current lowering explicit rather than silently discarding them at the seam.
- **Deletion accounting:** Production adds 165 and deletes 180 physical lines,
  a net deletion of fifteen. Nearly all additions relocate the existing
  documented runtime enums and contracts before their temporary derived
  lowerer state. Tests add fourteen and delete thirteen lines, with the
  lowerer's old `moi_max_workgroup_lds_bytes` spelling replaced by the typed
  `max_workgroup_lds_bytes` name.
- **Checked-in gate:** All ten focused runtime-capability, binding, and adapter
  tests pass. The host gate passes all 1,508 runnable tests with the two
  expected benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests
  pass; and all 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass in 69.35 seconds. Slice 5BF remains
  the latest physical-gfx950 gate, with all 593 tests passing; E2E validation
  remains outside this work.

### Slice 5BH: delete the standalone options adapter

- **No compatibility wrapper:** `LegacyOptionsAdapter` is deleted from
  production and tests. Once every caller-owned input became an actual base
  subobject of the temporary lowerer state, the adapter contained no policy or
  validation responsibility and merely forwarded six values plus four derived
  controls.
- **Fresh-state construction has one owner:** `ConSanOptions` now has one
  explicit production constructor that copies the six typed inputs verbatim
  and derives only its nonoptional flavor, barrier-move inventory switch,
  InlineShadow workgroup mode, and two renamed test-path selectors. Default
  construction remains available for focused mechanism tests. The five former
  adapter tests are renamed as construction tests and continue to prove whole-
  value preservation, derived controls, caller immutability, and independence
  between constructed values.
- **Deletion accounting:** Production adds 28 and deletes 40 physical lines, a
  net deletion of twelve; tests add twenty and delete twenty-six, a net
  deletion of six. No production or test source names the deleted adapter.
- **Checked-in gate:** All five focused construction tests pass. Renaming them
  brings them under the broad `*ConSan*` host filter, which now passes all 1,513
  runnable tests with the two expected benchmark-object skips. All 172 HSA-hook
  and hook-lifecycle tests pass, and all 2,908 simulator-device tests across
  `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250` pass in 73.28 seconds.
  Slice 5BF remains the latest physical-gfx950 gate, with all 593 tests passing;
  E2E validation remains outside this work.

### Slice 5BI: delete the duplicate fault-retry request

- **One mutation vocabulary through retry:** `ConSanMoiInventoryRetryConfig`
  now carries the production `MutationRequest` directly. The deleted
  `ConSanFaultMutationRetryConfig` repeated nearly every fault switch,
  selection identity, index, proof opt-in, and address operand under shorter
  names, then maintained full conversions both from and back into the lowerer
  state.
- **Typed behavior replaces projection mechanics:** Retry asks the existing
  `MutationRequest::has_fault_mutation()` predicate whether a late fault is
  active, rejects its typed `fault_dry_run`, and assigns the complete mutation
  subobject only when applying a live late fault. The existing behavioral tests
  cover dry-run rejection, live atomic-fault equivalence to a fresh transform,
  unsatisfied exact selection, disabled and absent report-only paths, report
  rebinding, synchronization inventory, and malformed inventory. The removed
  projection-only test merely restated the deleted conversion code.
- **Fixed private marker protocol:** The markerless barrier-move fallback uses
  its private fixed `s_nop 42/43` protocol locally. Its mutable test-only option
  and retry field had no parser, hook, or supported request path and are
  deleted.
- **Deletion accounting:** Production adds fifteen and deletes 144 physical
  lines, a net deletion of 129; tests add four and delete sixty, a net deletion
  of 56. No production or test source names the deleted retry type or marker
  option.
- **Checked-in gate:** All thirteen focused mutation-contract and MOI-retry
  tests pass. The host gate passes all 1,512 runnable tests with the two
  expected benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests
  pass; and all 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass in 66.15 seconds. Slice 5BF remains
  the latest physical-gfx950 gate, with all 593 tests passing; E2E validation
  remains outside this work.

### Slice 5BJ: delete the partial report-binding retry projection

- **One runtime binding through retry:** `ConSanMoiInventoryRetryConfig` now
  carries the production `BoundRuntimeResources` value directly. The deleted
  `ConSanMoiReportRetryConfig` copied only five MOI fields under shorter names,
  omitted the binding lifetime and SuperCollider address, and required retry
  to project those fields back into the lowerer one at a time.
- **No partial reconstruction:** The typed pipeline passes its already-
  validated binding unchanged, retry assigns the complete binding subobject,
  and mechanism tests construct the same typed value. Existing retry
  equivalence tests cover ordinary Record/Replay access inventory, explicit
  synchronization layout, live late mutation, and malformed or mutable
  inventory; the construction-contract tests separately cover every binding
  member and value semantics.
- **Deletion accounting:** Production adds four and deletes twenty-seven
  physical lines, a net deletion of twenty-three. Tests add sixteen and delete
  thirty-six lines, a net deletion of twenty. No production or test source
  names the deleted report-retry type or its projection members.
- **Checked-in gate:** All fifteen focused retry, resource-binding, and options-
  construction tests pass. The host gate passes all 1,512 runnable tests with
  the two expected benchmark-object skips; all 172 HSA-hook and hook-lifecycle
  tests pass; and all 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass in 75.24 seconds. Slice 5BF remains
  the latest physical-gfx950 gate, with all 593 tests passing; E2E validation
  remains outside this work.

### Slice 5BK: delete renamed debug-control copies

- **One debug representation in lowering:** Spill-tier and private-epoch test
  selection now read the inherited `ConSanDebugOverrides::test_force_vgpr_spill`
  and `test_force_private_epoch` fields directly. The temporary lowerer no
  longer stores the same two caller inputs again under shorter names.
- **Tests exercise the production vocabulary:** Focused mechanism tests now set
  the typed debug fields that production hook parsing and typed construction
  use. The whole-value construction assertion already proves both fields cross
  the boundary unchanged, so two assertions of the deleted copies are removed.
- **Deletion accounting:** Production adds eight and deletes sixteen physical
  lines, a net deletion of eight; the changes outside the deleted members are
  direct field-name substitutions. Tests add 109 and delete 111 lines, a net
  deletion of two, almost entirely the same mechanical substitution.
- **Checked-in gate:** The host gate passes all 1,512 runnable tests with the
  two expected benchmark-object skips; all 172 HSA-hook and hook-lifecycle
  tests pass; and all 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass in 69.93 seconds. Slice 5BF remains
  the latest physical-gfx950 gate, with all 593 tests passing; E2E validation
  remains outside this work.

### Slice 5BL: delete the duplicate lowering flavor

- **One flavor value through lowering:** `ConSanOptions` now uses its inherited
  `ConSanRequest::flavor` directly. The deleted nonoptional member hid that
  request field under the same name and copied it during construction, making
  the apparent input depend on the static type through which it was read.
- **Construction states stay explicit:** The typed request retains
  `std::nullopt` as its validation-rejected construction sentinel. Focused
  mechanism tests still default to disabled instrumentation by setting the
  inherited request field to `ConSanFlavor::None` in the compatibility
  lowerer's default constructor. Production construction preserves the
  validated optional value unchanged instead of projecting it.
- **Deletion accounting:** Production adds five and deletes eight physical
  lines, a net deletion of three. No test or production source declares a
  second ConSan flavor field.
- **Checked-in gate:** All nine focused request and options-construction tests
  pass. The host gate passes all 1,512 runnable tests with the two expected
  benchmark-object skips; all 172 HSA-hook and hook-lifecycle tests pass; and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in 66.53 seconds. Slice 5BF remains the latest
  physical-gfx950 gate, with all 593 tests passing; E2E validation remains
  outside this work.

### Slice 5BM: give static transform artifacts one owner

- **One artifact subobject:** `ConSanTransformArtifacts` owns the immutable
  inventory, observation plan, coverage ledger, mutation sites and outcome,
  resource alternatives, patches, transform outcome, and warnings produced by
  one static transform. Compatibility lowering builds that subobject in place;
  `TransformResult` receives the same value by move rather than redeclaring
  every field and projecting them individually.
- **Retry consumes the same value:** The pristine-inventory retry now moves the
  complete typed artifact subobject back into lowering in one operation. It no
  longer reconstructs a hand-selected subset whose membership could drift
  from ordinary publication. Replacement bytes, runtime evidence, install
  state, engine-private candidates, and lowering failures remain outside the
  artifact contract because they have different owners.
- **Contract coverage:** The publication test now also fills and verifies
  mutation and warning state, in addition to its existing inventory, policy,
  coverage, fault, resource, patch, and dispatch-requirement checks. The full
  focused pipeline/retry set passes all 33 tests.
- **Deletion accounting:** Production adds 51 and deletes 75 physical lines, a
  net deletion of 24. The added lines are the documented common type; the
  deletion removes the second field inventory and both field-by-field transfer
  sequences. Tests add five lines.
- **Checked-in gate:** The host gate passes all 1,512 runnable tests with the
  two expected benchmark-object skips; all 172 HSA-hook and hook-lifecycle
  tests pass; and all 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass in 73.68 seconds. A 26-case physical-
  gfx950 cross-engine smoke matrix covering repeated dispatch identity, dense
  SuperCollider, full-bank Stream-K, and the large-LDS pipeline passes in
  23.06 seconds; the complete serialized physical matrix remains reserved for
  the final tranche gate. E2E validation remains outside this work.

### Slice 5BN: give the validated replacement image the same owner

- **One replacement-image value:** `ConSanTransformArtifacts::replacement`
  now owns the complete validated replacement code object. Compatibility
  lowering and `TransformResult` no longer declare separate byte vectors or
  move the image explicitly across their boundary. The generic RocJitsu
  instrumentation result keeps its independent `elf_bytes` field because it
  belongs to a different API.
- **Publication is exact by construction:** Moving the common artifact
  subobject publishes the replacement together with its patch inventory,
  outcome, and warnings. The focused publication test fills a recognizable
  replacement and verifies that the exact bytes arrive in the typed result;
  existing result-validation, installation, retry-equivalence, determinism,
  and runtime-discard tests continue to exercise its lifecycle.
- **Deletion accounting:** Implementation files add 161 and delete 162
  physical lines, a net deletion of one. Nearly all changed lines are the
  mechanical replacement of two former field names by the one shared name;
  structurally, the slice removes two vector declarations and one explicit
  transfer while adding one documented shared field. Tests add 728 and delete
  719 physical lines, likewise dominated by the mechanical name change, plus
  the exact publication assertion.
- **Checked-in gate:** All 1,524 ConSan host tests and all 172 HSA-hook tests
  pass. Of the 2,908 simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250`, 2,906 pass in the parallel matrix; the
  two gfx950 large-LDS InlineShadow cases reach their 60-second timeout after
  starting in the most heavily contended wave, then both pass in isolation in
  54.44 seconds. Slice 5BM's 26-case physical-gfx950 cross-engine smoke remains
  the physical gate for this artifact-ownership tranche; the complete
  serialized physical matrix remains reserved for the final gate. E2E
  validation remains outside this work.

### Slice 5BO: keep runtime dispatch identity out of static output

- **Input remains input:** The runtime-provided MOI report dispatch identity
  is no longer copied into mutable `ConSanResult`. Lowering still reads the
  value from `BoundRuntimeResources`; final validation now receives the same
  expected value explicitly instead of recovering a caller input from the
  transform's output record.
- **Negative contract coverage:** The literal-dispatch full-pressure test now
  validates the completed replacement with the original dispatch identity and
  proves that changing only the expected identity makes final validation
  reject the image. Two assertions that merely checked the deleted copy are
  removed.
- **Deletion accounting:** Implementation files add 31 and delete seventeen
  physical lines, a net addition of fourteen. The slice deletes the duplicate
  field and its assignment; the additional lines make the dependency explicit
  through ordinary, retry, and mutation-composition finalization paths rather
  than hiding it in mutable output state. Tests add six and delete two lines.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in 69.32
  seconds. Slice 5BM's 26-case physical-gfx950 cross-engine smoke remains the
  physical gate for this tranche; the complete serialized physical matrix
  remains reserved for the final gate. E2E validation remains outside this
  work.

### Slice 5BP: give each typed failure one control-plane owner

- **One owner for contract failures:** A configuration, target/runtime
  capability, or runtime-binding failure now exists only as the typed
  `contract_issue` of its owning `ConSanPipelineStageState`. The separate
  `TransformResult::configuration_issue` copy and the duplicate `Contract`
  entry in `ConSanTransformIssue` are deleted. Callers no longer have to prove
  that three representations of the same failure agree.
- **Diagnostic issues keep one narrower role:** `ConSanTransformIssue` now
  retains contextual failures from inventory, observation planning, evidence
  planning, runtime binding, compatibility lowering, and final validation.
  Its invariant is simply a valid category, owning stage, and nonempty detail;
  typed request/capability failures never require a caller to parse that
  detail because the stage record already owns their enum value.
- **Contract coverage:** Focused tests continue to reject every malformed
  stage enum, status, identity, and typed contract payload. Configuration and
  backend failures now assert the stage-owned enum and the absence of a
  redundant diagnostic issue, while contextual issue tests cover every
  remaining invariant.
- **Deletion accounting:** Implementation files add 29 and delete 55 physical
  lines, a net deletion of 26. Tests add sixteen and delete 28 lines, a net
  deletion of twelve. This more than repays Slice 5BO's temporary fourteen-line
  implementation increase.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in 73.34
  seconds. Slice 5BM's 26-case physical-gfx950 cross-engine smoke remains the
  physical gate for this control-plane-only tranche; the complete serialized
  physical matrix remains reserved for the final gate. E2E validation remains
  outside this work.

### Slice 5BQ: use the owning stage as the issue classifier

- **Delete the parallel taxonomy:** `ConSanTransformIssueKind` and its naming
  and iteration machinery are deleted. Only its `LegacyLowering` value had an
  actual producer; the other five values were speculative scaffolding. Every
  issue already carries a `ConSanPipelineStage`, so the second enum could only
  disagree with the durable pipeline contract that owns the failure.
- **Smaller invariant:** A contextual transform issue is now exactly an owning
  stage plus nonempty diagnostic detail. Typed contract failures remain in the
  stage record itself, as established by Slice 5BP. Runtime verdicts remain in
  their separate runtime model; no semantic distinction is lost.
- **Deletion accounting:** Implementation files add one and delete 63 physical
  lines, a net deletion of 62. Tests delete nineteen lines. No replacement
  abstraction or fallback is added.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in under 70
  seconds. Slice 5BM's physical-gfx950 smoke remains the physical gate for this
  control-plane-only tranche; the complete serialized physical matrix remains
  reserved for the final gate. E2E validation remains outside this work.

### Slice 5BR: share lowering errors without a projection type

- **One error vector:** Fatal lowering and final-validation diagnostics now
  live in `ConSanTransformArtifacts` beside the outcome and warnings they
  qualify. `ConSanResult` and `TransformResult` no longer declare separate
  error representations, and publication moves the complete artifact value
  without rebuilding one wrapper per string.
- **Typed failures stay typed:** Request, capability, and binding failures
  remain the `ConSanContractIssue` of their owning stage. The shared string
  vector is only for contextual mechanism failures, all of which arise inside
  the lowering/final-validation boundary. Empty diagnostics remain invalid.
- **Runtime and tests consume the owner:** The HSA coordinator logs the shared
  error strings directly. Pipeline tests assert the same invalid-object,
  retry-provenance, determinism, and fail-open/fail-closed behavior through
  that value, including rejection of an empty diagnostic.
- **Deletion accounting:** Implementation files add nine and delete 37
  physical lines, a net deletion of 28. Tests add 31 and delete 46 lines, a
  net deletion of fifteen. The per-error conversion loop and the entire
  `ConSanTransformIssue` projection type are gone.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  physical gate for this representation-only tranche; the complete serialized
  physical matrix remains reserved for the final gate. E2E validation remains
  outside this work.

### Slice 5BS: keep raw working state out of the hook test boundary

- **Artifacts are the publication input:** `TransformResult`'s private
  publication seam now accepts only `ConSanTransformArtifacts`. A completed
  lowerer value converts to that base value before publication; private
  candidates, resolved registers, warning checkpoints, and the historical
  `modified` bit cannot cross the boundary accidentally.
- **Hook tests exercise the production contract:** The HSA transform override,
  its queued/live-fault fixtures, and all helper builders now use shared static
  artifacts. Its two direct transformation assertions enter
  `transform_consan`, and the hook test no longer includes the raw-lowering
  header. A stale `moi_candidates` fixture was deleted because runtime report
  planning consumes the typed observation/coverage contract instead.
- **Narrow test privilege:** `TransformResultTestAccess` can publish synthetic
  static artifacts through the real stage and binding validation, but it can
  no longer accept `ConSanResult`. This preserves focused hook lifecycle tests
  without creating a test-only raw-mechanism bypass.
- **Deletion accounting:** Implementation files add six and delete seven
  physical lines, a net deletion of one. Tests add 100 and delete 114 lines, a
  net deletion of fourteen; most changed lines are the fixture type cutover.
  No fallback or second representation is retained.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  physical gate for this test/control-plane tranche; the complete serialized
  physical matrix remains reserved for the final gate. E2E validation remains
  outside this work.

### Slice 5BT: derive modification state from owned artifacts

- **Delete the parallel mutable bit:** `ConSanResult::modified` is gone.
  Candidate modification state is derived from the typed outcome and emitted
  patch inventory, so a finalized result cannot publish a boolean that
  disagrees with its replacement contract. `mark_modified()` moves only an
  unchanged candidate to `ModifiedValid`; it cannot erase an `Unsupported` or
  `Invalid` decision made by a more authoritative analysis step.
- **Keep construction and publication semantics distinct:** While the
  compatibility lowerer is assembling a result, patch telemetry still reveals
  byte changes that must be validated or rolled back even if a later stage has
  already classified the attempt as unsupported. Finalization clears those
  bytes and patches on every non-installable result. A focused unit test covers
  unchanged, modified, unsupported, invalid, and staged-patch cases.
- **Share active-image selection:** Eight MOI access, barrier, atomic,
  prologue, sampled, and pipeline paths now use one `active_moi_bytes` helper
  instead of independently selecting between pristine and staged bytes. This
  both pays for the derived-state API and gives incremental lowering one
  definition of its current image.
- **Deletion accounting:** Implementation files add 71 and delete 79 physical
  lines, a net deletion of eight. Tests add 821 and delete 802 lines, a net
  addition of nineteen; almost every changed test line is the mechanical
  field-to-query cutover, with the focused state-invariant test providing the
  new coverage. No compatibility field or fallback remains.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in about 71
  seconds. Slice 5BM's physical-gfx950 smoke remains the physical gate for
  this artifact-state tranche; the complete serialized physical matrix remains
  reserved for the final gate. E2E validation remains outside this work.

### Slice 5BU: remove raw lowering from the production boundary

- **Typed production output:** The remaining compatibility lowerer is declared
  by `consan_lowering.h` as `lower_consan`, whose result is only
  `ConSanTransformArtifacts`. Both ordinary publication and the pristine MOI
  inventory pass use this boundary. Recursive fault/perturbation composition
  also re-enters it and explicitly resumes private composite working state;
  mutable planner fields cannot escape with its result.
- **Delete the legacy entry:** `consan_legacy_lowering.h` and the historical
  `try_patch_consan` symbol are gone. The complete mutable working result has no
  library-header declaration. Seventeen mechanism-test translation units name
  their bypass `test_lower_consan`, and the transform fuzzer declares the
  complete entry locally, making every remaining non-production use explicit.
- **Deletion accounting:** Implementation files add 44 and delete 40 physical
  lines, a net addition of four. This replaces a 25-line raw-result header with
  an 18-line artifact-only header; the small balance is the explicit typed-to-
  private conversion used by recursive composition. Tests add 1,222 and delete
  1,204 lines, a net addition of eighteen; almost all of that diff is the
  mechanical call-site rename and its formatting, not duplicated behavior.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in about 67
  seconds. Slice 5BM's physical-gfx950 smoke remains the physical gate for this
  tranche; the complete serialized physical matrix remains reserved for the
  final gate. E2E validation remains outside this work.

### Slice 5BV: give candidate rollback one artifact operation

- **One owned rollback operation:** `ConSanTransformArtifacts` now owns
  `discard_candidate_modification()`, the single operation that removes a
  candidate replacement image together with the patch telemetry describing
  it. The operation deliberately leaves outcome classification, diagnostics,
  semantic inventory, and mutation facts to the caller that knows why the
  candidate was rejected.
- **Delete parallel clearing:** Composition, final validation, MOI dispatch-ID
  fallback, SuperCollider proof failure, and both runtime-binding rejection
  paths no longer independently clear the two vectors. A focused unit test
  proves both the removal and the retained-state contract. No fallback clearing
  path or second representation remains.
- **Deletion accounting:** Implementation files add eighteen and delete
  thirty-two physical lines, a net deletion of fourteen. The focused test adds
  twelve lines.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in about 66
  seconds. Slice 5BM's physical-gfx950 smoke remains the periodic physical gate;
  the complete serialized physical matrix remains reserved for the final gate.
  E2E validation remains outside this work.

### Slice 5BW: give report geometry one canonical representation

- **One address-free layout:** `ConSanMoiReportBufferLayout` now carries its
  evidence engine and is the sole representation of an MOI allocation's exact
  capacities, offsets, and byte extent. Planning, late runtime binding, report
  initialization, and lowering all exchange that same value. The duplicate
  `ConSanMoiReportLayoutOverride` and its field-by-field copy and comparison
  routines are deleted.
- **One engine owner:** `ConSanMoiAutoReportPlan` no longer repeats the engine
  already owned by its layout. Report-header construction and runtime
  allocation likewise derive the evidence protocol from the layout rather
  than accepting a second engine argument that could disagree. Late binding
  reconstructs and replans candidate geometry, then accepts only exact
  canonical equality for the requested engine and registered allocation size.
- **Contract coverage:** Unit tests exercise complete, incomplete, invalid,
  inconsistent, malformed-offset, wrong-engine, undersized-allocation, and
  noncanonical-capacity cases. Evidence-requirement tests mutate the canonical
  layout engine directly, preserving cross-type mismatch coverage after the
  duplicate plan field is removed.
- **Deletion accounting:** Implementation files add 186 and delete 335
  physical lines, a net deletion of 149. Most additions are the 64-line
  standalone canonical type moved out of an include fragment; most deletions
  remove the 55-field duplicate representation and manual conversion and
  equality code. Tests add 116 and delete 103 lines, a net addition of thirteen
  for the focused consistency contract; the remaining test diff is mechanical
  terminology and type migration.
- **Checked-in gate:** All 1,524 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in 68.78
  seconds. Slice 5BM's physical-gfx950 smoke remains the periodic physical
  gate; the complete serialized physical matrix remains reserved for the final
  gate. E2E validation remains outside this work.

### Slice 5BX: retain only independent pipeline contracts

- **Six real contracts:** The pipeline retains configuration, target/runtime
  capabilities, program inventory, observation plan, evidence requirements,
  and runtime binding. The former `LegacyLowering`, `FinalValidation`, and
  `Complete` entries merely repeated `TransformResult::outcome`; they are
  deleted rather than renamed into permanent scaffolding.
- **Fixed identity and ownership:** Stage state is a fixed array indexed by the
  stage enum. Array position owns stage identity and `TransformResult` owns its
  code-object identity once, eliminating the variable-length ordering and nine
  repeated collision-aware identities. Invalid enum lookups still fail closed,
  and each position independently validates its legal status and typed issue.
- **Behavioral test correction:** Tests continue to cover every stage/status
  enum, issue ownership, malformed status, deterministic publication, and all
  cross-type result invariants. Tests that removed/reordered vector elements or
  corrupted a copied per-stage code-object ID are deleted because those
  malformed representations can no longer be constructed, not because the
  invariant was weakened.
- **Deletion accounting:** Implementation files add forty-two and delete
  102 physical lines, a net deletion of sixty. Tests add eighteen
  and delete forty-two lines, a net deletion of twenty-four. No compatibility
  alias, synthetic terminal stage, or fallback vector representation remains.
- **Checked-in gate:** The focused 25-test pipeline contract suite, all 1,524
  ConSan host tests, all 172 HSA-hook tests, and all 2,908 simulator-device
  tests across `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250` pass.
  The simulator matrix completes in 68.56 seconds. Slice 5BM's physical-gfx950
  smoke remains the periodic physical gate; E2E validation remains outside
  this work.

### Slice 5BY: make immutable inventory own its joins

- **One query owner:** `ProgramInventory` now resolves kernel descriptors and
  exact kernel/function names, while `SynchronizationInventoryView` resolves
  typed or named events and unique event-to-sequence membership. Exact absence
  returns null; duplicate names retain code-object order for compatibility;
  ambiguous sequence membership fails closed.
- **Deleted result coupling:** The raw-`ConSanResult`
  `find_consan_sync_sequence_for_event` API, the private
  `kernel_for_descriptor` helper, a second named-event helper, and the
  corresponding engine-local joins are deleted. Record/Replay, Sampled,
  Inline Shadow, SuperCollider, mutation, validation, report planning, and the
  typed pipeline all query the same immutable owner.
- **Focused contract coverage:** Inventory tests cover empty, exact, absent,
  duplicate-name, descriptor, typed-event, named-event, and unique-sequence
  queries. Existing ambiguity coverage proves that two sequences containing
  the same event remain a hard failure rather than an arbitrary first match.
- **Deletion accounting:** Implementation files add 206 and delete 226
  physical lines, a net deletion of twenty. Tests add forty-two and delete
  nine lines. No compatibility wrapper preserves either removed query API.
- **Checked-in gate:** All 1,526 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in 69.23
  seconds. Slice 5BM's physical-gfx950 smoke remains the periodic physical
  gate; E2E validation remains outside this work.

### Slice 5BZ: share dense access topology and occupied text

- **One topology contract:** `MoiDenseCandidatePartition` now owns the
  target-independent facts used by Record/Replay, Sampled, and Inline Shadow:
  typed kernel/function owner identity, original candidate index, anchor
  ordering, dispatcher-capacity splitting, and common SOPP branch span. Engine
  callers still supply their real dispatcher capacity and retain their own
  relay width, scalar ABI, eligibility, and emitted evidence semantics.
- **One relocation-exclusion contract:** `MoiOccupiedTextRanges` canonicalizes
  unsorted, adjacent, and overlapping half-open pristine-text ranges and owns
  overlap queries. Access-engine relay-host searches and barrier dense fallback
  now use it instead of four local merge/search implementations. The shared
  owner collector protects every access anchor in a container plus its native
  barrier, fence, and atomic sites; Record/Replay additionally protects its
  qualified ordinary-memory ordering sequences.
- **Focused contract coverage:** Unit tests cover empty/reversed range removal,
  adjacency coalescing, half-open boundaries, owner-kind separation, anchor
  sorting, preservation of the caller's candidate index, explicit dispatcher
  capacity, unlimited capacity, and mandatory branch-span splitting. Existing
  cross-target engine and device tests continue to exercise the resulting
  physical routes.
- **Deletion accounting:** Implementation files add 212 and delete 248 physical
  lines, a net deletion of thirty-six. Tests add seventy-two lines. The three
  engine-local owner/group implementations and four local occupied-range
  normalizers are deleted without a compatibility wrapper.
- **Checked-in gate:** All 1,528 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass in approximately 65 seconds. Slice 5BM's
  physical-gfx950 smoke remains the periodic physical gate; E2E validation
  remains outside this work.

### Slice 5CA: make dense relay hosts one shared claim

- **One host-claim mechanism:** Sampled and Inline Shadow now call
  `claim_moi_dense_owner_relay_host` for typed kernel/function resolution,
  owner-local access/synchronization exclusion, scalar-bootstrap liveness
  proof, relocation decoding, and reservation. Their engine code retains only
  the policy that decides whether a group needs a host and the evidence ABI
  that determines its size.
- **No cross-group alias:** Every successful relocated host is inserted into
  one claim set before the next group is searched. This closes Sampled's prior
  opportunity to reuse the same original instructions for two capacity-split
  groups and makes its ownership rule match Inline Shadow and Record/Replay.
  The existing 65-access CDNA4 test now requires at least two relocated hosts,
  proves they do not consume any access anchor, and proves their ranges are
  pairwise disjoint.
- **Type-owned topology facts:** `MoiDenseCandidateGroup` now owns anchor
  projection, neighboring-entry overlap, and minimum explicit-anchor size.
  Focused unit assertions cover those operations; all three access engines
  consume them instead of repeating the loops.
- **Deletion accounting:** Implementation files add 101 and delete 118
  physical lines, a net deletion of seventeen. Tests add twenty-three lines.
  Neither old engine-local host search survives behind a wrapper.
- **Checked-in gate:** All 1,528 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The simulator matrix completes in
  approximately 67 seconds. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CB: share dense synchronization and atomic topology

- **One heterogeneous route partition:** `MoiDenseRouteSite` and
  `MoiDenseRouteIndexGroup` carry only typed kernel/function ownership,
  pristine-text anchors, and caller-owned indices. The common partitioner now
  supplies deterministic owner ordering, dispatcher-capacity splitting, and
  SOPP branch-span splitting to access, barrier, sampled-atomic, and
  inline-atomic planners without forcing their semantic candidate types or
  evidence policy into one representation.
- **Deleted planner copies:** Sampled barrier, Inline Shadow barrier, Sampled
  atomic, and Inline Shadow atomic planners no longer build local owner maps,
  sort their own anchors, or reproduce the common branch-span rule. Three
  synchronization/atomic relocation paths also use `MoiOccupiedTextRanges`
  instead of retaining private range normalization and lookup algorithms.
- **Focused contract coverage:** A direct unit test separates same-named
  kernels and functions, proves deterministic anchor order and caller-index
  retention, exercises explicit and unlimited capacity, forces branch-span
  splitting, and covers empty input. Existing cross-target dense access,
  barrier, and atomic tests exercise each engine-specific consumer.
- **Deletion accounting:** Implementation files add 146 and delete 177
  physical lines, a net deletion of thirty-one. Tests add thirty lines. The
  replaced owner/group and occupied-range algorithms are deleted rather than
  retained behind compatibility adapters.
- **Checked-in gate:** All 1,528 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CC: close the raw MOI retry boundary

- **Typed ownership in both directions:** The runtime-bound MOI retry now
  accepts and returns `ConSanTransformArtifacts`. Only its internal lowering
  body reconstructs `ConSanResult`; the production pipeline and retry-focused
  tests can no longer transport private candidates or resolved-register
  mirrors across that boundary.
- **Deleted false warning provenance:** The private
  `moi_stage_warning_begin` field and its validation/reset paths are deleted.
  The production projection had never transported that field into a retry, so
  it could not preserve the distinction its comment claimed. Retry now
  explicitly replaces all unbound-attempt diagnostics while immutable
  inventory, policy, coverage, and mutation artifacts carry the semantic
  result.
- **Test boundary:** A compile-time assertion pins the retry's artifact-only
  signature. Existing typed-pipeline, exact fresh-versus-retry image,
  synchronization-inventory, late-fault, and malformed-inventory tests cover
  the behavior. The optional external-object benchmark no longer reads the
  private MOI candidate vector merely to print diagnostic counters.
- **Deletion accounting:** Implementation files add fifteen and delete
  twenty-three physical lines, a net deletion of eight. Tests add and delete
  twenty-one lines. No compatibility overload retains the raw input or output
  signature.
- **Checked-in gate:** All 1,528 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CD: share dense access route emission

- **One route emitter:** Record/Replay and Sampled access groups now call
  `emit_moi_dense_access_group` for entry-island construction,
  optional relocated-entry-host routing, dispatcher comparisons and targets,
  SCC restoration, dispatcher telemetry, host entry rewriting, and original
  call-anchor rewriting. Evidence construction and engine admission remain
  separate because they express different sanitizer semantics.
- **Policy is explicit, not copied:** The only intentional route difference is
  whether equal spill-backed PC and call-return assignments form a collapsed
  route on explicit-key targets. Record/Replay and Sampled pass that decision
  explicitly to the common mechanism. The old owner-local emission copies and
  an unreachable PC-key subtraction branch are deleted rather than retained
  behind wrappers.
- **Focused route coverage:** Existing tests exercise both instantiations on
  RDNA4, CDNA4, and CDNA5, including explicit and call-return keys, relocated
  hosts, capacity partitioning, spill-backed scalar layouts, call-key capture,
  SCC preservation, and large kernels.
- **Deletion accounting:** Implementation files add 271 and delete 562
  physical lines, a net deletion of 291. No tests are added because the
  existing focused matrix already directly exercises every shared-route branch
  and both engine policy inputs.
- **Checked-in gate:** All 1,528 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CE: fold Inline Shadow into dense access route emission

- **All access engines share the mechanism:** Inline Shadow now joins
  Record/Replay and Sampled in `emit_moi_dense_access_group`. Entry-island and
  optional relocated-host installation, dispatcher-arm ordering and storage,
  SCC-preserving direct or indirect targets, dispatcher telemetry, host-entry
  rewriting, and original-anchor rewriting have one implementation.
- **Real ABI differences remain visible:** Inline Shadow retains its own
  scalar-ABI selection, SCC-tagged explicit route key, indirect-PC dependency
  wait, eight-word return island, and dispatcher-to-barrier reuse telemetry.
  Record/Replay and Sampled retain their seven-or-eight-word spill layout and
  explicit collapsed-spill policy. These are named branches in the shared
  route mechanism rather than copied engine bodies; evidence emission and
  admission remain engine-owned.
- **Focused route coverage:** Existing RDNA4, CDNA4, and CDNA5 tests cover
  one-word call relays, relocated entry hosts, ordinary and tagged explicit
  keys, aliased key/SCC state, SCC restoration before the evidence body,
  called-function hosts, and large composed layouts. The complete focused
  Record/Replay and Sampled route matrix remains green as well.
- **Deletion accounting:** Implementation files add 171 and delete 406
  physical lines, a net deletion of 235. The 329-line Inline Shadow emission
  copy is deleted without a compatibility wrapper or fallback.
- **Checked-in gate:** All 1,528 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CF: make access shape one policy/lowering contract

- **One mnemonic-shape vocabulary:** Access policy and all MOI emitters now
  consume the same directly tested predicates for single-range native LDS,
  two-range native LDS plus its byte scale, and supported FLAT access forms.
  Provenance, target operand legality, and engine evidence semantics remain
  separate decisions.
- **No policy/lowering drift:** The duplicate access-policy and compatibility-
  lowering whitelists are deleted. Public capability queries delegate to the
  same internal contract, so no third classification path remains.
- **Focused contract coverage:** Direct unit coverage pins all four two-range
  scales, unsupported single-range input, representative FLAT load/store
  forms, and rejected FLAT/global forms. Existing B96 boundary tests exercise
  RDNA3, RDNA4, CDNA3, CDNA4, and CDNA5, while access-policy tests cover all
  four engines and typed unsupported decisions.
- **Deletion accounting:** Implementation files add 95 and delete 141 physical
  lines, a net deletion of 46. Tests add sixteen lines.
- **Checked-in gate:** All 1,530 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CG: give two-address LDS shape one typed owner

- **One complete form record:** `NativeLdsTwoAddressForm` now owns direction,
  per-address element width, and encoded-offset byte scale for all sixteen
  LLVM-style and native two-address LDS spellings. Program inventory,
  instruction-width analysis, placement, access policy, and every MOI lowerer
  consume that record or a direct projection of it.
- **Deleted partial classifiers:** Inventory no longer carries private copies
  of the mnemonic set, width inference, and scale table. Placement no longer
  carries separate read, write, suffix-width, and scale lists, and analysis no
  longer special-cases the eight stride-64 spellings to avoid mistaking
  address spacing for transfer width. Fault-injection admission remains
  separate because mutation capability is not the same contract as decoded
  access shape.
- **Focused contract coverage:** The direct unit test now enumerates all
  sixteen forms and checks direction, element width, and scale, plus a rejected
  ordinary LDS form. Existing inventory and gfx1250 lowering tests check that
  the shared facts produce two stable semantic ranges and safe scratch use.
- **Deletion accounting:** Implementation files add 163 and delete 184
  physical lines, a net deletion of twenty-one. The new foundational header
  replaces the formerly MOI-private table while making the complete shared
  contract available during inventory construction.
- **Checked-in gate:** All 1,530 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CH: remove false dependencies on compatibility lowering state

- **Typed artifacts are sufficient for ordinary consumers:** Access, fault,
  synchronization, SuperCollider, MOI emission, placement, descriptor, and
  coverage helpers that read or update only production transform artifacts now
  accept `ConSanTransformArtifacts` directly. The compatibility result is no
  longer their accidental API merely because the current top-level lowerer
  happens to derive from that value.
- **A patch owns its payload requirement:** Full-workgroup-ID descriptor
  mutation now consumes the observation engine and the emitted patch. CDNA
  entry-capture patches already record their selected persistent workgroup
  SGPRs or private offsets; the deleted fallback to a result-wide resolved-SGPR
  mirror could disagree with a per-owner patch and made an unrelated mutable
  field part of descriptor policy.
- **Focused contract coverage:** The direct payload-requirement test covers
  RDNA4, CDNA5, CDNA4, CDNA3, invalid targets, non-MOI patches, indirect
  islands, register-backed and private-backed CDNA entry capture, engine
  exclusion, and missing ownership using only patch-local facts.
- **Boundary accounting:** Production `ConSanResult` references fall from 200
  to 85. The signature narrowing and explicit engine argument add 139 and
  delete 132 physical implementation lines after formatting, a temporary net
  increase of seven lines; it introduces no state or wrapper and is the direct
  prerequisite for deleting the remaining three focused workspaces.
- **Checked-in gate:** All 1,530 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CI: make scalar-persistent state an emitted patch contract

- **The prologue owns its ABI:** Every MOI owner/epoch entry prologue now
  records the exact persistent owner, epoch, optional compact workgroup key,
  and exact Record/Replay workgroup-coordinate SGPRs that it initializes. The
  same patch records the complete scalar allocation high-water mark already
  used to grow its owning kernel descriptor.
- **No transform-wide scalar mirror:** Placement no longer publishes four
  copies of its selected scalar state into `ConSanResult`, the main lowerer no
  longer resets or backfills those copies, and final validation consumes only
  the emitted prologue contract. Atomic transaction validation likewise uses
  the access patch's private-state marker or its owning prologue's scalar ABI
  instead of consulting mutable lowering state.
- **Tests inspect durable behavior:** The existing scalar-placement matrix now
  derives its assertions from emitted prologue metadata. This preserves exact
  coverage of automatic and explicit selection, collision avoidance, CDNA and
  RDNA placement, exact workgroup tuples, and emitted instruction operands
  while no longer entrenching a temporary result representation.
- **Deletion accounting:** Implementation files add 46 and delete 71 physical
  lines, a net deletion of 25. Four `ConSanResult` fields and all of their
  publication plumbing are gone; production references to the compatibility
  type fall from 85 to 79.
- **Checked-in gate:** All 1,530 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CJ: make admitted access candidates a local lowering projection

- **Temporary state has lexical ownership:** The admitted MOI access vector is
  now built after policy and owned by the one `try_patch_consan_moi` attempt.
  Placement and the three access engines receive a read-only span only for the
  calls that consume it; the vector is no longer retained in `ConSanResult`.
- **Stable resource identity replaces pointer arithmetic:** Access resource
  plans are resolved by their typed site kind and unique original-text anchor.
  The deleted implementation derived a candidate index by subtracting raw
  object addresses from a result-owned vector, unnecessarily coupling plan
  lookup to that vector's allocation and lifetime.
- **Tests assert durable contracts:** Candidate-oriented tests now reconstruct
  the admitted set from immutable access inventory plus observation-plan
  intents, and inspect normalized inventory records rather than a lowering
  scratch type. The few tests of gfx1250 VGPR-bank mode derive that fact from
  the pristine bytes exactly as lowering does. Dedicated adapter tests remain
  only where candidate-only lowering behavior itself is the unit under test.
- **Deletion accounting:** Implementation files add 96 and delete 101 physical
  lines, a net deletion of five. More importantly, the result loses another
  mutable vector and production references to `ConSanResult` fall from 79 to
  76 while all candidate consumers now expose their actual dependency.
- **Checked-in gate:** All 1,530 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CK: bound perturbation planning to one lowering attempt

- **Candidates and plans are not transform output:** Synchronization analysis
  now writes perturbation candidates and selected plans into a focused
  `ConSanPerturbationPlanningState`. The state lives only across candidate
  construction, selection, emission, and the final validation of that lowering
  attempt; `ConSanResult` no longer retains either vector.
- **Composite evidence crosses one explicit boundary:** Fault/perturbation
  composition carries the pristine planning state only while translating its
  chosen edge. The instrumented intermediate image passes that same temporary
  evidence explicitly to final validation, including the case where mutation
  intentionally removed the boundary that reinventory would otherwise find.
  Durable results remain the mutation tally and the emitted patch's semantic
  identity, owner, source anchor, and composition facts.
- **Tests inspect planner state deliberately:** Planner-focused unit tests use
  an explicit test-only inspection parameter. Ordinary callers and final
  transform consumers cannot observe or depend on the workspace. Existing
  tests still cover stable selection, rejection reasons, bounded controls,
  byte emission, rollback, and pristine-to-mutated composition.
- **Structural accounting:** Two more mutable vectors leave `ConSanResult`.
  Explicit lifetime and intermediate-validation plumbing adds 155 and deletes
  100 implementation lines, a temporary net increase of 55; this is a
  prerequisite to deleting the compatibility result rather than a line-count
  reduction on its own.
- **Checked-in gate:** All 1,530 ConSan host tests and all 172 HSA-hook tests
  pass. All 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. Slice 5BM's physical-gfx950 smoke remains the
  periodic physical gate; E2E validation remains outside this work.

### Slice 5CL: derive owner-local persistent VGPR state from prologues

- **One owner, one emitted initialization contract:** Final validation now
  resolves an exact-shadow patch's owner-local dispatch VGPR from that owner's
  entry-prologue patch. It no longer consults a parallel vector populated by
  register placement before any bytes are emitted.
- **The reporting view is derived:** Tests that examine the per-owner
  allocation matrix reconstruct it from prologue owners and their recorded
  owner, epoch, workgroup, and dispatch registers. This retains focused
  coverage of mixed private/register components, AccVGPR boundaries, dynamic
  stacks, and owner-specific dispatch fallbacks without making planning state
  part of the transform result.
- **Deletion accounting:** `ConSanResult` loses its persistent-VGPR-assignment
  vector and placement no longer publishes or clears it. Implementation files
  add eight and delete nine physical lines, a net deletion of one; the larger
  benefit is removing one of the final two owner-indexed compatibility
  workspaces.
- **Checked-in gate:** All 1,530 ConSan host tests pass. The simulator-device
  and hook gates remain unchanged from Slice 5CK and are rerun at the next
  mechanism boundary; E2E validation remains outside this work.

### Slice 5CM: make vector-persistent state patch-local

- **Initializers and consumers name the same ABI:** A shared metadata helper
  records owner, epoch, compact workgroup key, and exact workgroup-coordinate
  VGPRs on every relevant entry prologue and Inline atomic consumer. Release-
  transaction validation now reads the exact atomic patch instead of a
  code-object-wide result mirror.
- **Persistent state is distinguished from entry scratch:** Patch metadata
  says whether its VGPR tuple is the lasting mechanism ABI and whether that ABI
  is owner-local. This prevents a scalar-persistent prologue's temporary entry
  VGPR carrier from masquerading as vector persistence, while retaining hybrid
  vector-owner/scalar-workgroup configurations.
- **Tests consume durable contracts:** Global and per-owner test reporting
  helpers derive their views only from emitted patches. Existing placement and
  instruction tests continue to cover explicit and automatic registers,
  code-object-wide versus owner-local selection, scalar and hybrid fallbacks,
  descriptor growth, collision avoidance, and all five target architectures.
- **Deletion accounting:** Four optional/aggregate fields and their reset,
  publication, and fallback logic leave `ConSanResult`. Implementation files
  add 27 and delete 31 physical lines, a net deletion of four. Only transient
  scalar assignments and dispatch-selection mirrors remain in the
  compatibility result.
- **Checked-in gate:** All 1,530 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. E2E validation remains outside this work.

### Slice 5CN: delete the compatibility result

- **One transform artifact:** The empty `ConSanResult` subclass is deleted.
  Production lowering, validation, composition, fuzzing, and mechanism tests
  now pass `ConSanTransformArtifacts` directly; no wrapper can acquire another
  parallel result field later.
- **Typed allocation contract:** `ConSanTransformArtifacts` owns one immutable
  `ConSanMoiRegisterAllocation`: the code-object-wide transient scalar ABI,
  owner-component overrides, and the code-object-wide persistent dispatch
  VGPR pair. These are placement results shared by multiple emitted patches,
  so they are not reconstructed from patch telemetry or copied onto every
  patch. Patch-local persistent state remains on the patch that emits or
  consumes it.
- **Planning state remains temporary:** Global and owner-local register
  selections still live in the mutable internal options only while placement
  and emission need them. Lowering freezes them into the typed allocation;
  tests and final validation consume that durable value directly.
- **Deletion accounting:** Implementation files add 220 and delete 257
  physical lines after formatting, a net deletion of 37. The change deletes the final four
  register mirrors, their reset/snapshot/publication plumbing, and all
  production references to the compatibility type. Test declarations move to
  the real artifact type and retain focused placement coverage.
- **Checked-in gate:** All 1,532 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across `gfx942`, `gfx950`, `gfx1100`,
  `gfx1201`, and `gfx1250` pass. The 26-case physical-gfx950 cross-engine
  smoke also passes; E2E validation remains outside this deletion work.

### Slice 5CO: delete retained-inventory shape coupling

- **Fresh late-fault transform:** The address-free MOI sizing inventory has one
  shape determined solely by its typed request. A fault supplied only after
  runtime allocation now takes the ordinary fresh transform path from the
  pristine bytes. The rare validation-only path no longer forces every normal
  inventory to retain extended barrier pairs or move destinations.
- **Deleted compatibility state:** `qualify_extended_barrier_pairs`,
  `moi_retry_preserves_extended_barrier_pairs_`, their hook argument, and the
  inventory-shape comparison/fallback protocol are gone. The pass-through
  composite-result wrapper is gone as well.
- **Allocation and validation repair:** The completed transient allocation is
  frozen once as `ConSanMoiRegisterAllocation`, not copied onto every patch.
  Dispatch-prologue validation recognizes the complete ordered capture/ABI-
  repair sequence and the shared workgroup-payload predicate now covers
  persistent SGPR, persistent VGPR, and private-memory coordinate tuples.
- **Regression contracts:** Focused host tests prove fresh late-fault retry
  equivalence for exact and incomplete extended-barrier pairs, typed allocation
  equality and owner override resolution, and all three persistent workgroup
  representations. The existing paired dynamic-private-stack device workload
  guards the corrected dispatch-prologue validation on gfx942 and gfx950.
- **Accounting:** Relative to Slice 5CN, implementation files add 216 and
  delete 162 physical lines (net +54). The retry deletion is net-negative; the
  tranche as a whole is temporarily positive because restoring strict semantic
  validation required replacing lossy patch reconstruction and adding the
  complete ordered-sequence proof. This is correctness repair, not a new
  compatibility layer.
- **Checked-in gate:** All 1,532 ConSan host tests, all 172 HSA-hook tests, all
  2,908 simulator-device tests across the five targets, and the 26-case
  physical-gfx950 cross-engine smoke pass. E2E validation remains outside this
  deletion work.

### Slice 5CP: delete the partial retry capsule

- **One bound lowerer state:** The internal retry now receives one
  `ConSanOptions` already constructed from the request, mutation, runtime
  capabilities, and bound resources. It no longer accepts a second object and
  mutates its resource and fault base subobjects after construction.
- **No bound-pristine state:** The pristine inventory entry has no
  `BoundRuntimeResources` parameter and constructs/publishes the explicit
  unbound value itself. Callers cannot supply an address only for the entry to
  clear five fields again.
- **Completed deletion:** The one-consumer `ConSanMoiInventoryRetryConfig`, its
  optional absent-versus-empty mutation state, all construction sites, and the
  now-meaningless equality test for those two shapes are gone.
- **Accounting:** Implementation files add 32 and delete 50 physical lines
  (net -18); the complete tranche adds 68 and deletes 140 (net -72), including
  tests simplified to pass the exact live lowerer state.
- **Checked-in gate:** All 1,531 ConSan host tests, all 172 HSA-hook tests, and
  all 2,908 simulator-device tests across the five targets pass. E2E
  validation remains outside this deletion work.

### Slice 5CQ: delete recursive fault application

- **One analysis, one exact mutation:** Fault planning already produces an
  exact mutation plan and complete pristine semantic inventory. Both ordinary
  and fault/perturbation composition now apply that plan directly to an
  independently owned artifact instead of recursively invoking the complete
  ConSan lowerer to rediscover and apply the same mutation.
- **Preserved composition proof:** A selected perturbation still retains the
  pristine inventory while its stable site, sequence, and owner identities are
  translated across the mutation. Fault-only transforms move the inventory
  directly; composite transforms retain a second owned view only for this
  explicit proof, not for another decode or analysis pass.
- **Completed deletion:** The mutable `faults_preapplied` recursion guard and
  all of its planning, mutation, instrumentation, and late-retry assignments
  are gone. The special recursive fault-application branch is gone as well.
- **Accounting:** Implementation files add 15 and delete 24 physical lines
  after formatting, a net deletion of nine, while also removing one full
  decode/analyze/finalize pass from composite fault execution.
- **Checked-in gate:** All 114 applicable focused fault, perturbation, and
  composition host tests pass, as do all 1,531 ConSan host tests, all 172
  HSA-hook tests, and all 2,878 currently generated simulator-device tests
  across `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`. E2E
  validation remains outside this deletion work.

### Slice 5CR: keep Sampled operand recovery patch-local

- **One owner for a candidate decision:** Whether an overlapping Sampled LDS
  address is recovered from the VGPR spill is selected while planning one
  access and stored on its `PlannedSampledPatch`. Planning and both final
  emission paths now pass that value directly to the Sampled watchpoint
  builder.
- **Completed deletion:** The code-object-wide
  `moi_sampled_spill_backed_operand_recovery` mutable option, its planning
  publication, and its two final-emission reconstructions are gone. The
  builder can no longer consume a stale decision left by another candidate.
- **Accounting:** Implementation files add 17 and delete 25 physical lines
  after formatting, a net deletion of eight.
- **Checked-in gate:** All 168 Sampled and spill-backed focused host tests pass,
  including CDNA4 full-pressure address recovery and gfx1250 dense routing.
  All 1,531 ConSan host tests, all 172 HSA-hook tests, and all 2,878 generated
  simulator-device tests across the five supported targets also pass. E2E
  validation remains outside this deletion work.

### Slice 5CS: keep MOI access overlap patch-local

- **One access, one overlap fact:** Each Record/Replay and Sampled planned
  access already records whether its selected scratch spill overlaps guest
  operands; Inline Shadow derives the same fact from its per-patch spill. All
  three root access emitters now receive that exact value directly.
- **Completed deletion:** The code-object-wide
  `moi_spill_overlaps_guest_operands` option, three planning publications, and
  six final-emission reconstructions are gone. Scratch-overlap admission and
  spilled-address recovery can no longer depend on whichever candidate most
  recently mutated a copied global options object.
- **Accounting:** Implementation files add 30 and delete 42 physical lines
  after formatting, a net deletion of twelve.
- **Checked-in gate:** All 152 focused spill, overlap, full-pressure, and
  dynamic-stack host tests pass across Record/Replay, Sampled, and Inline
  Shadow. All 1,531 ConSan host tests, all 172 HSA-hook tests, and all 2,878
  generated simulator-device tests across the five supported targets also
  pass. E2E validation remains outside this deletion work.

### Slice 5CT: delete the Inline Shadow rollout mode

- **One patch, one shadow representation:** Every eligible Inline Shadow
  access attempts the collision-free workgroup-local LDS representation. Its
  planned patch owns the optional successful layout; subword, dynamic-LDS,
  spill-backed, and no-fit cases retain their explicit external exact-shadow
  fallback. Emission consumes that layout and its existing local-cell
  parameters directly.
- **Completed deletion:** The mutable `moi_inline_workgroup_shadow` rollout
  marker, constructor derivation, candidate reconstructions, owner-indexed
  shadow lookup, external-only full-aperture capacity preflight, and atomic-only
  rollout warning are gone. Atomic-only resource sizing now uses the semantic
  access-inventory fact instead of disabling a mode flag.
- **Contract audit:** Local-shadow, no-fit, dynamic-LDS, atomic-only, and CDNA4
  spill-reload host contracts remain. The spill test now counts the one local
  cell's publication and diagnostic reloads instead of the unreachable two-
  cell legacy external representation. The test that required rejection by
  the deleted external-only capacity preflight is gone.
- **Accounting:** Implementation files add 11 and delete 51 physical lines
  after formatting, a net deletion of 40. Tests add three and delete 70
  physical lines, a net deletion of 67.
- **Checked-in gate:** All 1,530 ConSan host tests, all 172 HSA-hook tests, and
  all 2,878 generated simulator-device tests across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250` pass. The 26-case physical-gfx950
  cross-engine smoke also passes. E2E validation remains outside this deletion
  work.

### Slice 5CU: keep scalar-prologue scratch attempt-local

- **One explicit stage handoff:** Persistent-state placement returns the
  descriptor-indexed entry VGPR scratch assignments to the current MOI
  lowering attempt. The two possible owner/epoch prologue emission points
  consume that same vector directly.
- **Completed deletion:** `ConSanOptions` no longer owns or copies
  `moi_prologue_scratch_vgpr_assignments`. Per-site Inline caches no longer
  clear an unrelated code-object vector, and the one-consumer lookup wrapper
  is gone. The assignment type remains the focused contract between placement
  and prologue emission.
- **Accounting:** Implementation files add 24 and delete 28 physical lines
  after formatting, a net deletion of four.
- **Checked-in gate:** All 35 focused prologue, scalar-persistence, full-VGPR,
  and entry-scratch host tests pass. All 1,530 ConSan host tests, all 172
  HSA-hook tests, and all 2,878 generated simulator-device tests across the
  five supported targets also pass. E2E validation remains outside this
  deletion work.

### Slice 5CV: use one typed fault-presence contract

- **Single decision:** Staged mutation composition now uses
  `MutationRequest::has_fault_mutation()`, and requested-fault cardinality now
  comes from `MutationRequest::fault_mutation_count()`. Presence and count share
  one enumeration of the eleven switches; the lowerer-local counting helper
  and composition's independent presence predicate are gone.
- **Contract coverage:** The request-contract unit tests enumerate every fault
  kind, assert unit cardinality, and distinguish faults from SuperCollider
  perturbations.
- **Accounting:** After formatting the previously nonconforming fault-lowering
  include, implementation files add 46 and delete 47 physical lines, a net
  deletion of one. The semantic change deletes the independent presence
  predicate and the complete lowerer-local cardinality helper. Tests add two
  assertions. No production behavior or existing test expectation changes.
- **Checked-in gate:** All 115 focused request, fault, mutation, and composition
  tests pass. All 1,530 ConSan host tests, all 172 HSA-hook tests, all 2,878
  generated simulator-device tests across the five supported targets, and the
  26-case physical-gfx950 cross-engine smoke also pass. E2E validation remains
  outside this deletion work.

### Boundary audit before shared component cutovers

The remaining low-reference mutable fields are not duplicate switches:

- `patched_image_growth_input_bytes` retains the pristine root-image size while
  the second half of fault/instrumentation composition inventories an already
  mutated and potentially grown intermediate image;
- automatic-allocation markers distinguish a caller override from a placement
  decision and therefore control whether later placement may relocate or
  exclude that state;
- persistent-state markers record liveness and spill proofs that cannot be
  recovered from a selected register number; and
- dense-router and Inline-access facts are code-object semantic inventory used
  by resource sizing, placement, validation, and emission.

Deleting these facts locally would either recompute them from an invalid
intermediate view or scatter new parameters across the prototype lowerer. Their
next deletion boundary is the typed transform/resource context and component
cutover described above. The low-hanging tranche therefore ended at the
principled edge of the deeper redesign; subsequent slices cross that edge by
moving complete shared responsibilities, not by contorting individual flags
for nominal line-count reductions.

### Slice 5CW: share the planned-access placement contract

- **One cross-engine handoff:** `MoiPlannedAccessPatch` is the target-neutral
  placement-to-emission state shared by Record/Replay, Sampled, and Inline
  Shadow. `MoiPlannedReplayAccessPatch` adds the relocated-guest mechanics
  shared only by Record/Replay and Sampled. Engine-local plan types now contain
  only evidence indices, sampling gates, shadow layouts, and other genuinely
  engine-specific values.
- **Completed deletion:** The three copies of candidate, placement,
  entry-island, dense-router, displaced-guest, scratch/spill, private identity,
  and branch-only-route fields are gone. The second Record/Replay/Sampled copy
  of island sizing, private workgroup state, relocated-guest position, and
  overlap handling is gone as well. No compatibility projection or fallback
  remains.
- **Sharing and target boundary:** The common types contain no architecture
  discriminator and no instruction encoding. CDNA5 high-bank state is data in
  the replay plan; its native interpretation remains in target emission. This
  slice removes representation duplication across three engines without
  merging their evidence semantics.
- **Test policy:** The extracted types are passive stage contracts with no
  independent algorithm, so getter/equality tests would only test the language.
  Their behavior is covered by the existing host placement tests and paired
  device contracts for all three engines.
- **Accounting:** Implementation files add 56 and delete 74 physical lines, a
  net deletion of eighteen. Across the four affected implementation files,
  nonblank lines fall by twenty and comment-excluded code lines fall by 37.
  No target discriminator is added or moved, and no tests are added or deleted.
- **Checked-in gate:** All 1,530 ConSan host tests, all 172 HSA-hook tests, all
  2,878 generated simulator-device tests across the five supported targets,
  and the 26-case physical-gfx950 cross-engine smoke pass. E2E validation
  remains outside this deletion work.

### Slice 6: explicit pipeline and result cutover

- **Completed boundary:** `transform_consan` now owns the ordinary typed entry,
  and `transform_consan_with_mutation` makes validation-only mutation and
  perturbation composition explicit. Both return a split `TransformResult`
  with one immutable input identity, the complete ordered stage record,
  immutable inventory and policy artifacts, an engine-specific address-free
  evidence variant, replacement bytes, typed stage failures, contextual
  lowering errors, and static outcome.
- **Binding contract:** An unbound but complete report/marker requirement is
  explicitly `Deferred`. A concrete binding is accepted only when runtime
  capabilities satisfy the schema, lifetime scope matches, the engine-specific
  address is present, and the MOI allocation covers the exact planned byte
  count. Capacity-limited requirements remain well-formed and explicitly
  `Unsupported` at binding rather than acquiring an arbitrary address.
- **Production cutover:** `transform_consan` and
  `transform_consan_with_mutation` are the public typed entries, and the HSA
  production transform caller enters this pipeline. Installation policy is
  derivable from `TransformResult`; an rvalue compatibility projection remains
  for the hook lifecycle code that has not yet migrated. Hook test overrides
  retain their deliberately narrow legacy seam.
- **Temporary seam and remaining responsibility:** `TransformResult`
  construction is the sole typed-to-`ConSanOptions` entry. The prototype parser/inventory,
  lowerer, resource/placement machinery, finalizer, ABI retry, and recursive
  staged-mutation mechanics still execute inside that named boundary. The
  pipeline republishes their already-produced inventory and observation values
  in logical dependency order; it does not claim that those internal calls are
  yet physically separate. The temporary private retry capsule described here
  is deleted by Slice 5S; current results retain only typed/public artifacts
  and one provenance bit for the consumptive retry operation.
- **Focused type/unit gate:** Fifteen direct pipeline tests cover exhaustive
  enum iteration and naming, stage-record and issue validation, malformed
  cross-type relationships, configuration/capability short-circuiting,
  immutable fingerprints, all four evidence variants, absent/complete/wrong-
  schema/undersized bindings, installation truth tables, deterministic input
  preservation, exact split/mechanism agreement, and distinct ordinary/mutation entry
  points. The complete ConSan host gate passed 1,498 tests with two intentional
  external-object benchmark skips; all 98 selected hook and transform-memory
  tests passed.
- **Completed device and E2E gate:** The complete five-target simulator matrix
  passed all 2,908 cases in 85.82 seconds. Twenty focused physical-gfx950
  correct/incorrect contracts passed, including repeated-dispatch identity,
  dense SuperCollider, full-low-bank Stream-K, and the large-LDS pipeline. The
  physical gfx950 `d128-block` E2E row passed all four engines with exact
  coverage. The RocJitsu-emulated gfx1250 row retained its published state:
  SuperCollider, Sampled, and Inline Shadow passed with exact coverage;
  Record/Replay retained complete 18/18 access and 8/8 barrier static coverage
  and the already-published `metadata-full` limitation.
- **Slice-7 deletion boundary:** Remove both overloads from the public
  `consan.h` surface, keep the mutable-options implementation in an explicitly
  internal header for mechanism tests and fuzzing, and retain recursive
  mutation only inside the compatibility lowering implementation.
- **Prerequisite:** Slices 1--5B.

### Slice 7: endpoint cleanup and design reconciliation

- **Completed public boundary:** `consan.h` no longer declares any transform
  overload. Production code includes `consan_pipeline.h` and receives
  `TransformResult`; the compatibility lowerer exposes only artifact output
  through `consan_lowering.h`. Its complete mutable working entry has no
  library-header declaration. The redundant typed-to-`ConSanResult` overload
  and its temporary comparison test are deleted. The stronger pipeline
  projection test retains exact compatibility coverage.
- **Completed production routing:** Every ordinary or mutated installable HSA
  transform enters the typed pipeline. The pre-allocation MOI inventory pass
  and its runtime-bound retry both enter explicitly named typed operations and
  return `TransformResult`. The HSA
  coordinator no longer stores, publishes, or passes a mutable `ConSanResult`
  between the two phases. One documented extended-barrier preservation fact
  remains, and its focused unit test proves that it controls only incomplete
  extended-barrier-pair inventory. Raw mutable options remain reachable only
  inside the lowerer, 17 explicit mechanism-test translation units, and the
  transform fuzzer; they are not a production API.
- **Control-plane endpoint:** Configuration, capability validation, immutable
  program identity, observation policy, evidence requirements, runtime-binding
  status, static outcome, and installation policy have typed owners outside the
  prototype. The compatibility lowering may translate those values into
  mechanism shape, but may not originate target-neutral admission, report sizing,
  runtime allocation, or trust policy.
- **Honest deviations from the end-state diagrams:** The HSA adapter consumes
  `TransformResult` directly and automatic allocation retry is typed at that
  boundary. The compatibility result wrapper is gone, but the logical pipeline
  stages still wrap a monolithic artifact-producing lowerer rather than
  distinct parser, resource,
  emitter, and finalizer calls. `consan.h` still exposes prototype data
  structures used by mechanism helpers even though it exposes no transform
  function. These are bounded migration seams, not claims that the end state
  has been reached.

The remaining lowerer responsibilities are explicit:

| Area | Responsibility still inside the legacy boundary | Next deletion boundary |
| --- | --- | --- |
| Shared decode and inventory | ELF parsing, decoded events, CFG/liveness attachment, owner discovery, and artifact telemetry in one lowerer | Return `ProgramInventory` directly from a standalone inventory component |
| SuperCollider | Check/trap probe selection, register/resource choices, relays, sticky-marker lowering, and patch telemetry | Lower typed access intents and an optional bound marker through shared resource and emitter interfaces |
| Record/Replay | Access/synchronization probe bodies, pristine-inventory retry, replay ABI lowering, and applied-patch telemetry | Migrate first to the DBI-compatible host-processing path using typed evidence and runtime bindings |
| Sampled | Causal-window helpers, sampling-specific device state, synchronization helpers, and ABI lowering | Reuse shared intent/resource stages; retain only the engine's sampling and evidence semantics |
| Inline Shadow | Workgroup shadow layout, owner/epoch state, inline conflict checks, and device report lowering | Isolate the on-device state backend behind the same observation/resource boundaries while DBI support evolves |
| Mutation and composition | Recursive dry-run/live composition, pristine mutation inventory, perturbation composition, and exact-one application telemetry | Make mutation an explicit consumer of immutable inventory and a separate transform composition layer |
| Shared mechanism | Register allocation, spilling, wait preservation, branch reservoirs/relays, code-object growth, patch application, ABI retry, and final ELF validation | Extract resource planning, placement/emission, and transactional finalization in that order, shared by every engine and target |
| HSA lifecycle compatibility | The runtime coordinator and automatic allocation/retry path consume `TransformResult`; reader/executable metadata still reads a bounded legacy mechanism view | Extract the remaining patch geometry and registration telemetry into narrowly owned values, then delete the mechanism view |

- **Next component order:** First extract shared decode/inventory. Second separate
  shared resource planning from engine-specific evidence semantics, beginning
  with Record/Replay because it is the near-term DBI client. Third isolate
  placement/emission and transactional finalization. At each boundary, delete
  the corresponding derived `ConSanOptions` state and monolithic lowering code,
  and extend the focused type tests before proceeding.

The first deletion in that order is complete: `ProgramInventory` is now the
only owner of target identity, semantic architecture, metadata trust,
malformed-note count, and the semantic-classification-required state. The five
formerly duplicated `ConSanResult` members and their independent mutation
paths have been deleted. The same cutover subsequently deleted the duplicated
input byte count and fingerprint; retries and runtime reports now consume the
one `ConSanCodeObjectId` shared by `TransformResult` and `ProgramInventory`.
MOI access lowering now likewise receives candidates only by projecting
admitted access intents back to their normalized operand facts. The duplicate
legacy support classifier, physical-alias canonicalizer, and post-hoc intent
filter have been deleted, leaving one semantic admission decision across
Record/Replay, Sampled, and Inline Shadow.
Parser construction and decoded-container population remain inside the legacy
boundary and are the next inventory responsibility to extract.
- **Completed final gate:** All 1,502 selected ConSan host tests ran: 1,500
  passed and the two external-object benchmarks skipped as intended. All 98
  hook/transform-memory tests and all 16 focused pipeline tests passed. The
  generated capability document matched its typed manifest. The 2,908-case
  five-target simulator command retained every registered case and semantic
  coverage; one gfx942 Sampled case timed out under the 64-way aggregate run
  and then passed alone in 0.46 seconds. Twenty focused physical-gfx950 pairs
  passed. The native gfx950 `d128-block` E2E row again passed all four engines
  at 128/128 accesses and the expected barrier counts. RocJitsu-emulated
  gfx1250 again passed SuperCollider, Sampled, and Inline Shadow at 18/18
  accesses and all applicable barriers; Record/Replay retained the published
  yellow state with 18/18 accesses, 8/8 barriers, and `metadata-full` dynamic
  incompleteness. `git diff --check` is clean.
- **Prerequisite:** Slice 6.

### 10.1 Rules while executing the sequence

- Land each slice separately. Do not accumulate Slices 3--6 behind a flag.
- If a comparison path finds a semantic mismatch, first decide whether the new
  contract or current behavior is correct; do not automatically preserve the
  prototype.
- A compatibility adapter may translate shape, never invent policy or silently
  drop an unsupported item.
- If a component cannot cut over with the rest of ConSan still using legacy
  lowering, its boundary is too large and must be split.
- Deletion is part of the slice. A follow-up deletion issue is acceptable only
  for the explicitly named final legacy lowerer, whose consumers are outside
  this week's scope.
- Stop and revise this sequence if implementation evidence invalidates a
  dependency or makes the one-week endpoint unrealistic.

## 11. DBI compatibility and requirements ledger

### 11.1 Governing interpretation

In this section, **current** means implemented in the checked-in DBI code;
**intended** means confirmed design direction that may not yet be implemented;
**missing** means ConSan requires a facility for which neither a usable
implementation nor a settled complete contract exists. A **framework** owns
reusable instrumentation mechanics. A **client**, here ConSan, supplies
sanitizer-specific event selection, evidence meaning, analysis, and diagnostics.
A shared RocJitsu **substrate** is lower-level code used by the framework and
possibly other transformations.

**Observational instrumentation** preserves guest semantics and only collects
evidence. A transformation that intentionally changes guest behavior is not
observational, even if its purpose is to validate a sanitizer.

**DBT** means dynamic binary translation: rewriting executable code into
another executable form while it is loaded or run. It differs from DBI's
observational purpose, although both may transform code objects.
**Translated code** is the output of such an earlier DBT transformation.
**Transform composability** means two transforms can be applied in sequence while preserving
valid code, resource metadata, and original-address attribution. A
**common hook** is one runtime interposition layer that safely hosts multiple tools
instead of each tool independently replacing runtime callbacks.

The checked-in `dbi-design.md` describes implementation reality. The
[forward-looking DBI overview](https://github.com/ROCm/rocm-systems/pull/10407)
describes intent. This design targets the latter directionally without claiming
its facilities exist.

Allyson's clarifications settle the following for planning:

- near-term DBI processing is host-only, so production Record/Replay proceeds
  first and does not wait for on-device modes;
- production ConSan remains in RocJitsu, as it is today;
- before- and after-instruction instrumentation are intended framework
  capabilities;
- keep the current `HSA_TOOLS_LIB` integration until DBI owns a ready common
  hook;
- per-dispatch buffer binding is intended;
- instrumentation of already translated DBT code has no current use case and
  is deferred; preserve transform composability without making it a milestone;
  and
- test placement can follow the ownership boundaries in this document and be
  shuffled later.

The internal design therefore adopts DBI-shaped inputs—semantic site requests,
before/after placement, explicit probe effects, target-neutral resource
requirements, address-free buffer requirements, per-site outcomes, and
coverage—without importing an unfinished API or creating a permanent ConSan
facsimile of the framework. A **facsimile** here means a ConSan-owned duplicate
of generally reusable DBI machinery with different names.

### 11.2 Existing, intended, and missing facilities

Terms used in the comparison table are:

- To **preflight** a transform is to check every requested site's feasibility
  before editing bytes. **All-or-nothing** rejects the entire request if any
  site fails. **Graded per-site outcomes** permit each request to report
  instrumented or a typed rejection independently, subject to the client's
  coverage policy.
- An **offset request** names an instruction by byte offset. A selection
  **predicate** describes a property to match. **Symbol selection** names a
  function/kernel; **block selection** names a basic block, a straight-line CFG
  region with one entry. `BeforeInst` is the current DBI enum value requesting
  placement before an instruction.
- A **probe body** is the machine code implementing an observation. A compiled
  **no-op probe** is a body that deliberately performs no useful observation
  and is used to validate the invocation machinery. A **call envelope** is the
  framework-generated save/pass/call/restore sequence around a body. A probe
  **catalogue** is the closed, reviewed set of bodies clients may request. A
  **calling convention** defines argument locations, clobbers, return behavior,
  and preserved state.
- A probe is **stateless** when no tool-owned value persists between calls.
  **On-device aggregation** combines many dynamic observations in persistent
  device-visible state instead of streaming each observation individually.
- A code **variant** is one alternative executable form of the same input, such
  as original and instrumented. **Load-time production** creates variants while
  loading a code object. **Dispatch-time selection** chooses which variant one
  launch uses. **Queue interception** observes or edits dispatch packets as
  they enter an HSA queue.
- A record **stream** is an ordered transport of independently published
  records from probes to the host. **Heterogeneous** means it carries multiple
  record kinds. **Self-describing** means each record/envelope identifies its
  kind, version, and size. The **envelope** is the common header/protocol around
  a client payload. A **payload** is the ConSan-specific content.
  **Flow control** bounds producer/consumer pressure; **stream reservation** claims
  space for a record; **per-kind loss** reports which record category was
  dropped rather than only a total count.
- **Extended kernargs** add tool binding values to a correctly copied kernarg
  region without changing the application's argument interpretation.
  **Completion-safe lifetime** keeps that copy and every referenced buffer
  alive until GPU execution and host evidence consumption are finished.
- **Hook chaining** composes multiple runtime tools instead of allowing the last
  installed hook to displace the others. **Failure containment** prevents an
  instrumentation/tool failure from corrupting unrelated runtime state and
  applies the declared fail-open/fail-closed policy.
- An **original mapping** relates transformed locations and records to original
  code locations. **Compiler-clean input** means an untouched compiler-produced
  code object; accepting arbitrary valid input images means not relying on that
  assumption after another transform has run.
- The **instrumentor** is the current DBI framework component that selects
  insertion locations, constructs probe calls, preserves resources, and writes
  the transformed code object.
| Concern | Current DBI implementation | Intended/confirmed direction | ConSan action or requirement |
| --- | --- | --- | --- |
| Decode/CFG/liveness | Reusable decoder, CFG and liveness; current instrumentor consumes them | Shared RocJitsu substrate | Inventory and resource components depend on semantic/property interfaces, not ConSan raw decoders |
| Transactional ELF patching | Preflights all sites then performs one patch; current single-text cave model | Code-object-to-code-object transforms, composable variants | Reuse final validation now; require arbitrary input images and explicit original mapping later |
| Site selection | Offset requests, `BeforeInst`, all-or-nothing rejection | predicate/symbol/block selection and per-site graded outcomes | ConSan supplies required semantic IDs; every rejection returns a reason |
| Before/after | Before only today | Both intended and unblocked | Required for addresses before an access and atomic outcomes after it |
| Probe bodies | Copied compiled no-op probe, fixed call envelope | Fixed verified catalogue, compiled per target/wave size; host-side analysis | Record/Replay needs typed access/synchronization probes and context arguments; descriptor/calling convention must be data |
| Device state | Probe call is stateless; no LDS/persistent state | Near-term no on-device aggregation | Compatible with RR; future framework model needed for Sampled/SC/Inline |
| Register/resource handling | Liveness, some SGPR/VGPR/AccVGPR spills; limited growth/zero-scratch/targets | Framework owns transparent state preservation and reports occupancy effects | ConSan intents declare effects/lifetimes; DBI must cover all five targets, dynamic stack, entry state and segment edits |
| Outstanding memory | Current probe envelope drains supported counters | Framework obligation, target-aware | Probe descriptor declares operations; DBI preserves guest relaxed-wait semantics before/after every intent |
| Large-kernel placement | Appended cave with relative branch reach | Relocation strategy remains open; large kernels are normal | Current ConSan dense/relay behavior supplies tests and requirements, not code to copy wholesale |
| Stream/record ABI | Not the current production path | Heterogeneous self-describing host stream with per-kind loss | RR payloads fit an envelope; ordering-record loss must be distinguishable from access loss |
| Buffer binding | Current ConSan bakes a code-object report address | Per-dispatch binding intended, likely via extended kernargs | Require exact allocation/memory properties, copy of explicit+implicit kernargs, and completion-safe lifetime |
| Variant selection | Not the current instrumentor path | Load-time production, dispatch-time selection and queue interception | Needed for runtime sampling, original/instrumented comparison, and buffer routing |
| Runtime hook | ConSan has a private HSA tool/interceptor | DBI eventually owns composable common hook | Keep HSA adapter narrow; expose runtime requirements so ownership can move without changing engines |
| Coverage | Current instrumentor is all-or-nothing | requested/instrumented/rejected, unsampled and dropped all visible | Coverage ledger is the client/framework join; no clean verdict without it |
| Original mapping | Current patch telemetry is internal | mapping must be published; delivery is open | Transform result retains stable semantic/original mapping independently of cave layout |
| Simulator/hardware | Current DBI sim scope is CDNA3/CDNA4/RDNA4 | both first-class, same transform | Require gfx942/gfx950/gfx1100/gfx1201/gfx1250 and physical gfx950 qualification |
| Translation composition | Shared substrate but not a completed integration contract | overview says transforms compose | Defer translated-code instrumentation per current use case; never assume compiler-clean input in stable contracts |

### 11.3 Record/Replay DBI requirements

**Wave-granular** context is one value per wave; **lane-granular** context may
differ for each active lane. A probe's **context** is the site, execution
identity, active mask, arguments, and bound resources it can observe.
**Event-atomic publication** means that a logical event requiring several physical
records is either committed as a complete group or reported lost; the host
must not accept a misleading subset. To **serialize dispatches** is to make
otherwise independent launches wait for one another solely because the tool
uses a global bottleneck. **Overhead facts** quantify resources, added work,
memory, and occupancy effects introduced by the framework.

A **data-described calling convention** is a typed descriptor that tools can
inspect, rather than calling behavior implicit in one emitter. A
**fused kernel** combines work that might otherwise have been several kernels and is
often large. **Packet-visible segment growth** means private/group resource
increases appear both in code-object metadata and in the actual dispatch packet
seen by hardware.

Record/Replay can become the first production DBI client when the framework can
provide:

1. offset/predicate selection for access, barrier, atomic, fence, entry, and
   termination sites, with per-site typed rejection;
2. before and after probes, including an exact ordering relation around the
   relocated guest instruction and access to dynamic atomic outcomes;
3. a fixed, versioned catalogue of wave- and lane-granular access/synchronization probes,
   with decoded clobbers/effects and a data-described calling convention;
4. exact code-object/site, dispatch, x/y/z/cluster workgroup, wave/owner and
   active-mask context obtainable by a probe without ConSan-owned ABI guessing;
5. transparent resource handling on all five targets, including entry capture,
   zero-scratch, dynamic stack, register growth, unified/separate AccVGPR,
   wait counters, and packet-visible segment growth;
6. large-object placement for heavily fused/generated kernels and a mapping
   back to original semantic sites;
7. host-allocated, fine-grained, per-dispatch buffer binding with a lifetime
   extending through device publication and host consumption;
8. typed access and synchronization record kinds, event-atomic publication,
   per-kind drops, malformed/torn protection, and a bounded flow-control mode;
9. load-time variants plus queue-time selection/sampling/binding, without
   serializing unrelated application dispatches;
10. composable runtime interception and visible failure containment; and
11. framework coverage/overhead facts that the ConSan trust evaluator can join
    with semantic applicability.

Record/Replay continues to own which semantic sites matter, payload fields,
happens-before replay, conflict semantics, trust policy, and diagnostics. It
does not own target encoders, spill placement, stream reservation, kernarg
extension, or hook chaining.

### 11.4 Later mode requirements

#### Sampled

A **provided-probe facility** would let a reviewed client-supplied probe body
execute within framework-enforced state-preservation, resource, and coverage
rules. It is more general than selecting a fixed stateless catalogue entry and
must not become an unchecked escape hatch.

In addition to the RR foundation, Sampled needs a reviewed on-device processing
contract for one per-workgroup selector, bounded window state, atomic
publication, and synchronization attachment. Dispatch-time selection alone
cannot express selection among workgroups inside a dispatch. DBI must either
support framework-owned tool state supplied to fixed probes or define a
provided-probe facility for bounded on-device aggregation. The choice must
retain explicit statistical selection and collision/loss accounting.

#### SuperCollider

SuperCollider needs before/after operand/result access, a delayed redundant
observation or store readback, a device comparison, and a sticky device-visible
marker. It may be implementable with fixed provided probes sooner than Inline
Shadow, but it still violates a strict “device only streams, host only reduces”
interpretation. The framework must define how a provided probe safely performs
this bounded on-device decision and how the original access remains exactly
once.

#### Inline Shadow

**Low latency** here means the shadow operation must complete near the observed
access and influence the next device-side shadow decision; waiting for a host
round trip is not equivalent. An **exact shadow transaction** is the atomic or
otherwise synchronized read/check/update of one shadow cell under the supported
model.

Inline Shadow needs the largest DBI evolution: low-latency in-kernel exact
shadow transactions, bounded ordering tables, persistent entry-captured state,
workgroup/local or per-dispatch tool state, atomic device decisions, and
structured diagnostics. Streaming every access to emulate it on the host is a
different mode, not a migration. The internal `ObservationPlan` and
`EvidenceRequirements` make this requirement explicit while allowing RR to
ship without it.

#### Mutation and perturbation

Fault injection is transformational, whereas committed DBI is observational.
DBI should remain composition-friendly, but validation mutation is not a
Record/Replay delivery requirement. Until a general transformational API is
reviewed, keep mutation on the current transactional patch path and outside the
production sanitizer client interface.

### 11.5 Questions for the DBI design discussion

These questions are evidence-backed and do not block the internal Stage 2
sequence:

A dispatch **completion signal** is the runtime object the GPU marks when the
dispatch has finished; a dispatch may omit one, so lifetime logic cannot blindly
depend on it. Record **multiplicity** is the number of physical stream records
needed for one logical event. **Induced serialization** is new ordering or
waiting introduced by instrumentation. A **mixed-version transition** is a
period in which ConSan, DBI, and other tools use different revisions of the
runtime hook contract.

1. **Five-target ownership:** Which layer commits to enable and qualify
   gfx942/gfx950/gfx1100/gfx1201/gfx1250, and how does a client express that a
   missing target mechanism is a release blocker rather than an optional probe?
2. **Semantic instruction properties:** Will DBI expose normalized access
   range, address space, atomic ordering/scope/outcome, and barrier properties,
   or should those classifiers live in a shared RocJitsu layer below DBI?
3. **Probe context ABI:** Precisely how do fixed probes obtain site, dispatch,
   full workgroup including cluster coordinate, wave, lane mask, and buffer
   identity? Which are passed, preloaded, or recorded by the envelope?
4. **Entry/exit state:** How are kernel-entry capture and termination flush
   represented and composed with ordinary before/after sites? Current DBI lists
   entry/exit instrumentation as a known gap.
5. **Kernarg binding lifetime:** Who allocates the valid kernarg-region copy,
   preserves explicit and implicit arguments/alignment, observes completion,
   and reclaims it safely for dispatches without completion signals?
6. **Probe effects:** What descriptor states clobbers, memory-counter effects,
   execution-mask policy, argument locations, required before/after values, and
   event-atomic record multiplicity? How is it verified against compiled code?
7. **Loss and trust:** Can the framework reserve/commit a multi-record event
   atomically and report drops by record kind? Losing a synchronization record has a
   different correctness consequence from losing an access record.
8. **Large-kernel relocation:** Which of the overview's layout strategies will
   own generated kernels beyond relative reach, and what original-site mapping
   is promised to clients and external profilers?
9. **Resource contract:** How do clients request persistent versus transient
   state without choosing registers, and how are occupancy changes, induced
   serialization, zero-scratch and dynamic-stack failure reported?
10. **On-device supplied probes:** Is bounded stateful processing by
    RocJitsu-provided probes an intended future extension, or must Sampled,
    SuperCollider, and Inline Shadow motivate a distinct framework facility?
11. **Hook transition:** What is the cutover unit from current `HSA_TOOLS_LIB`
    ownership to the common hook, and how are ConSan, DBT, and external profiler
    hooks chained during mixed-version transition?
12. **Report scope:** Allyson confirmed per-dispatch binding as intended; does
    DBI also permit a deliberately shared code-object/workgroup state object
    when an on-device algorithm requires cross-probe persistence, and how is
    its concurrency scope represented?

The following are explicitly deferred rather than unanswered blockers:
instrumenting DBT-translated code without a demonstrated use case, the public
status of an advanced-user API, and final placement of tests between ConSan and
DBI suites.

## 12. Internal decisions still requiring implementation evidence

The architecture and migration route do not depend on these answers, but the
named slice must resolve them from evidence rather than let an adapter decide
accidentally.

Terms used in these questions are:

- An **empirical control** is a tuning option used to study behavior rather than
  an ordinary supported setting. A register **override** forces an internal
  choice for focused validation. A **deprecated compatibility** option remains
  temporarily so existing use does not break, but has a documented
  replacement/removal path.
- An analysis **facet** is one optional, explicitly named subset of inventory
  facts. **Eager** construction computes it whether or not a mode consumes it;
  requested construction computes it only when named while preserving stable
  identities for all present facts.
- A content **fingerprint** is a hash or similarly derived identity for binary
  contents. A **collision** occurs if different inputs receive the same
  identity; collision testing/handling prevents silent misattribution. An
  **opaque handle** is a runtime token whose internal value and cross-lifetime
  stability are not promised to the client.
- **Exhaustively observed** means every relevant dynamic event in the execution
  was retained, not merely that every static site was instrumented. No current
  bounded ConSan mode may imply this without evidence.
- A component's **durable owner** is the layer expected to maintain it after
  temporary migration adapters are gone.

### 12.1 Supported request surface

Current environment options include production policy, empirical controls,
fault selection, register overrides, and test-only seeds. Slice 2 must classify
each option as supported `ConSanRequest`, `TransformPolicy`, `RuntimePolicy`,
`MutationRequest`, `ConSanDebugOverrides`, deprecated compatibility, or removal.
Evidence is its documentation, hook configuration tests, validation manifests,
and actual E2E use. An undocumented field in `ConSanOptions` is not evidence of
a public contract.

### 12.2 Report ABI stability

The current ABI is consumed within the in-tree hook and tests, but some expert
users may inspect buffers directly. Slice 5 must inventory external/documented
consumers before declaring a field internal. The logical event and completeness
semantics are stable; exact offsets and table multiplicities become a supported
ABI only if evidence shows an external promise. This question changes versioning
and compatibility, not the component boundary.

### 12.3 Inventory construction cost

The current pipeline avoids synchronization analysis for ordinary
SuperCollider. A production inventory may therefore have explicitly requested
analysis facets rather than eagerly computing every fact. The invariant is one
immutable inventory instance with declared present/absent facets, not mandatory
work that no mode consumes. Slice 3B should measure representative large
objects and preserve deterministic identities across facet choices.

### 12.4 Code-object identity source

The transform can use a content fingerprint, while the runtime also has reader,
load, executable, and dispatch identities. The semantic `PhysicalSiteId` needs a
content-stable code-object identity; runtime instances need separate lifetime
IDs. Slice 3A must choose and collision-test the content identity without
embedding opaque HSA handles in persistent semantic records.

### 12.5 Complete-within-contract wording

Record/Replay's bounded first-light snapshot and Inline's bounded diagnostics
cannot prove global race freedom. The trust evaluator needs precise terms for
“all requested static sites lowered and all retained evidence valid” versus
“the execution was exhaustively observed.” Slice 4A should settle enum names and
user wording with device-test oracles; it must not upgrade the former into the
latter.

### 12.6 Shared classifier ownership

The classifier is needed during the internal week and can initially live beside
ConSan while exposing no engine policy. Its durable owner—ConSan, shared
RocJitsu analysis, or DBI—depends on the DBI discussion in question 2. The
boundary is stable enough that moving ownership later does not alter inventory
or engine interfaces.

## 13. Stage 1 review gate

Stage 1 is complete only when reviewers can answer “yes” to every item below:

- Does the current-state map cover common analysis, SuperCollider, all three
  MOI engines, mutation composition, the HSA lifecycle, and host analysis?
- Are product semantics, instrumentation transparency, evidence publication,
  completeness, runtime lifetime, and target isolation expressed as invariants?
- Does every proposed component have one responsibility, explicit input/output
  ownership, dependency direction, isolation tests, and documentation duties?
- Does every component state whether it is shared unchanged, parameterized by
  capability/profile data, or genuinely target-specific across all five
  targets?
- Are mode similarities reused without forcing SuperCollider or the three
  distinct MOI evidence strategies into a false common implementation?
- Does the test map preserve paired correct/incorrect behavior while identifying
  exact-register/patch/word assertions that may constrain only a legacy
  component?
- Is the one-week endpoint credible given the current implementation size, and
  is what remains behind the compatibility mechanism explicit?
- For every migration slice, are current responsibility, new contract,
  temporary seam, affected consumers, test gate, cutover, deletion, and
  prerequisite named?
- Can every intermediate revision build and run without a future global switch?
- Does the DBI ledger distinguish current facilities, intended facilities,
  confirmed answers, near-term RR blockers, later on-device-mode requirements,
  and deferred questions?
- Would the design still make sense if the current patch placement, report
  layout, selected registers, or HSA hook implementation were replaced?

If implementation evidence changes an answer, update this document and the
remaining sequence before proceeding. Stage 2 begins only after this gate is
reviewed and explicitly opened.
