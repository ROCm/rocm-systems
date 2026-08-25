# ConSan Production Implementation Plan

This document covers the work to replace the current multi-architecture ConSan
prototype with a principled, production-grade implementation. The preparation
phase that built and qualified the host, checked-in device, physical-device,
and end-to-end test coverage is essentially complete and is no longer tracked
as a separate phase here. That test suite is now the foundation for the
replacement.

The supported target set is CDNA3 (`gfx942`), CDNA4 (`gfx950`), CDNA5
(`gfx1250`), RDNA3 (`gfx1100`), and RDNA4 (`gfx1201`).

## Objective

The current implementation is a prototype, not production-grade code. It was
assembled incrementally by multiple AI agents that were encouraged to add code
to solve local problems, without a corresponding requirement to reuse,
generalize, simplify, or minimize. The agents did not share a coherent design,
and the result did not pass through a normal human review process.

Consequently, ConSan contains a large amount of architecture-specific and
engine-specific code, repeated policy decisions, ad hoc solutions, mixed
layers, and abstractions that do not consistently express the underlying
sanitizer and ISA semantics. This makes correctness hard to reason about,
review difficult, and cross-architecture regressions likely.

The goal is not to polish this structure in place. It is to replace it with a
smaller, coherent implementation suitable for human review and long-term
maintenance.

The overall effort must design two things together:

1. **The destination:** the principled architecture, abstractions, invariants,
   and ownership boundaries of the production implementation.
2. **The incremental path:** a sequence of small, reviewable changes that
   reaches that destination while continuously preserving verified behavior
   and deleting the prototype paths that each change supersedes.

A destination without a credible migration path is incomplete, and a sequence
of local refactorings without an agreed destination risks reproducing the
prototype's accidental structure.

All code migration in this plan is incremental. A system-wide or mode-wide
flag-day switch is not an acceptable implementation strategy. The largest
permitted cutover is one sufficiently small, well-defined component, after its
new contract and implementation have been tested and while the rest of the
system remains on its existing path.

## Phase 1: ConSan-internal design and incremental migration

**Cadence: approximately one design-only day, followed by approximately one
week of incremental implementation and migration.**

Before attempting to fit ConSan onto DBI's still-evolving design, put ConSan's
own house in order. We need a coherent account of what ConSan does, which
concepts and invariants belong to it, how its current modes relate, and which
parts of the prototype are accidental instrumentation machinery. That gives us
something concrete to bring to the DBI design discussion next week and lets us
answer precisely what ConSan needs from the framework.

This phase deliberately designs ConSan from its own behavioral and sanitizer
requirements rather than reverse-engineering the exact shape of current or
proposed DBI APIs. It is nevertheless directionally informed by DBI, as
described under the constraints below. Do not spend this phase migrating to
DBI, anticipating unfinished interfaces in detail, or papering over DBI gaps.
Record requirements and questions as they emerge, then take that evidence to
the DBI owners.

The first day is pure design iteration before rewriting production code. It
must produce not only a scoped end-state for the implementation week, but also
the complete incremental route to that state. That route is a prerequisite for
implementation, not something to improvise after a new architecture has been
built on the side.

### Provisional questions

The internal design has not been chosen yet. The first work is to answer, rather
than assume, questions such as:

- What is ConSan's smallest target-neutral semantic vocabulary for memory
  accesses, synchronization, ownership, ordering, evidence, and diagnostics?
- Which semantics genuinely belong to a common core, and which belong only to
  Record/Replay, Sampled, SuperCollider, or Inline Shadow?
- What are the distinct phases and data flows inside each mode, from inventory
  through runtime observation to analysis and reporting?
- Which current abstractions express real sanitizer concepts, and which merely
  hide duplicated target encodings, placement algorithms, buffers, or prototype
  history?
- How much ConSan code can be shared across all supported gfx architectures
  instead of remaining bifurcated into architecture-specific implementations?
- How far can we pursue well-defined, well-layered components that each do one
  thing, are testable in isolation to the greatest practical extent, and apply
  uniformly across gfx architectures?
- Where should policy, semantic analysis, evidence representation, resource
  requests, runtime state, host analysis, coverage, and reporting meet?
- Which behavior is part of the product contract, and which behavior is only a
  property of the current implementation?
- How do we make the resulting design not merely correct, but explainable in
  clear design documentation that helps both human reviewers and AI agents
  understand each component, its invariants, its boundaries, and its role in
  the whole system?

These are prompts for the week's design work, not conclusions to encode in a
framework before examining the code.

### Stage 1: approximately one day of design only

Inspect code, tests, and documentation and run read-only validation as needed,
but do not begin the production-code rewrite during this stage. Its deliverable
has two equally important parts:

The concrete Stage 1 outcome is
[PRODUCTION_DESIGN.md](PRODUCTION_DESIGN.md), in this directory. Complete that
document autonomously from the code, existing documentation, and tests. The
remaining uncertainties are subjects for Stage 1 investigation rather than
prerequisites to negotiate in advance. Where the available evidence still does
not justify a decision, record a precise unresolved question, its consequences,
and the evidence needed to answer it instead of silently guessing.

No further clarification is required before starting Stage 1. The current
code, its host and device tests, the ConSan design and validation documents,
the current and forward-looking DBI design material, Allyson's clarifications,
and this plan provide enough evidence and constraints to do the design work.
Stage 1 should use that evidence in several deliberate passes rather than
turning a first impression into the proposed architecture:

1. trace the present end-to-end implementation of each of the four engines,
   including runtime ownership, device instrumentation, evidence transport,
   host processing, diagnostics, coverage, failure, and teardown;
2. separate behavioral contracts and architectural facts from incidental
   prototype mechanisms, using host tests, paired device tests, and relevant
   end-to-end workloads as evidence;
3. identify the genuinely shared semantic core, the deliberate differences
   among engines, and the narrow ISA and ABI boundaries required across all
   five gfx targets;
4. derive and compare candidate component boundaries, interfaces, ownership,
   dependency direction, isolation-test surfaces, and documentation contracts;
5. design the scoped implementation-week destination together with its fully
   incremental component migration sequence, testing every proposed seam
   against the Stage 2 constraints before accepting it;
6. map the resulting design toward DBI, distinguishing facilities that exist,
   capabilities that are intended but not implemented, longer-term on-device
   requirements, and questions to take to the DBI owners; and
7. perform a final adversarial audit of the complete document against the code,
   all four modes, all five targets, the behavioral test contracts, the
   one-week scope, and every required intermediate cutover.

The design work may make ordinary engineering judgments without pausing for
approval. It should pause only if new evidence exposes a product decision that
materially changes ConSan's externally intended behavior and cannot be resolved
from the recorded contracts. Ordinary uncertainty about current prototype
structure is not such a blocker: investigate it, choose the best-supported
design, and document the reasoning. A DBI question that does not block the
internal component design should likewise be entered in the requirements and
gaps ledger rather than delaying Stage 1.

Although Stage 1 does not rewrite production code, its proposed end-state and
incremental route must already incorporate every requirement of
[Stage 2](#stage-2-approximately-one-week-of-incremental-implementation) and
the Phase 1 [constraints and non-goals](#constraints-and-non-goals). In
particular, the design must be executable as individually testable component
migrations, with bounded old/new seams and no eventual mode-wide or system-wide
switch. A design that looks clean on paper but cannot be reached under those
implementation constraints is not a useful Stage 1 result.

1. **The scoped end-state for the implementation week.** Map the current
   implementation and end-to-end data flow for all four modes. Define the
   terminology, semantic data model, invariants, ownership, dependency
   direction, mode boundaries, error and incompleteness semantics, and major
   component interfaces that the week can realistically establish. For every
   component, define what is shared across all gfx architectures, where a
   capability parameter is sufficient, and what genuinely requires a
   target-specific implementation.
2. **The complete incremental path to that end-state.** Slice the work into
   small components and order their migrations. For every slice, record the
   current responsibility, new boundary and contract, temporary old/new seam,
   affected consumers, test gate, component-level cutover, superseded code to
   delete, and any prerequisite supplied by an earlier slice.

`PRODUCTION_DESIGN.md` must contain:

- an inventory of the current implementation and end-to-end data flows for all
  four modes;
- behavioral and architectural invariants;
- the proposed components, ownership, interfaces, and dependency direction;
- each component's sharing strategy across all five gfx architectures, with
  genuine target-specific mechanics isolated and justified;
- the semantic relationships and deliberate differences among Record/Replay,
  Sampled, SuperCollider, and Inline Shadow, without allowing Record/Replay's
  near-term DBI fit to dictate the other modes;
- the intended isolation-test surface and documentation contract for every
  component;
- a map from host and device tests to behavioral contracts, including tests
  suspected of entrenching prototype implementation details;
- the realistically scoped implementation-week end-state;
- the complete incremental migration sequence, including for every slice its
  old/new seam, affected consumers, test gate, component-level cutover,
  superseded-code deletion, and dependencies; and
- a DBI compatibility analysis and concrete requirements/gaps ledger, plus any
  evidence-backed unresolved questions.

The design stage must also:

- map host and device tests to behavioral contracts and current mechanisms,
  identifying tests that may constrain prototype shape rather than behavior;
- identify the minimum genuinely shared core without forcing unlike modes into
  one abstraction or preserving parallel target implementations where semantics
  agree;
- give every proposed component an explicit all-five-target disposition:
  shared unchanged, shared behind capability data or a narrow target adapter,
  or target-specific for a documented architectural reason;
- show that every intermediate state builds, runs, and has a meaningful test
  gate, rather than relying on an unfinished parallel implementation;
- split any component whose cutover would affect too much of the system to be
  understood and qualified independently;
- scope the intended end-state to what the implementation timebox can actually
  complete;
- cross-check every proposed component boundary and migration slice against the
  Stage 2 implementation procedure and the constraints below, splitting or
  redesigning anything that would require a large or untestable cutover; and
- start a requirements ledger for instrumentation, target, runtime, buffer, and
  probe capabilities that should later be discussed with the DBI owners.

Implementation begins only after both the end-state and this migration sequence
have been reviewed against the forward-referenced Stage 2 procedure and
constraints. A design review that approves the destination but leaves the route
to be invented during coding, or postpones an incremental-migration problem
until implementation, does not pass the gate. Stage 1 ends after
`PRODUCTION_DESIGN.md` is complete; do not begin Stage 2 until that document has
been reviewed and the implementation gate has been explicitly opened.

### Stage 2: approximately one week of incremental implementation

Execute the reviewed component sequence. This is substantial implementation
and migration work, not a design-only week and not merely an optional pilot.

For each component:

1. establish or test its new internal contract;
2. implement it once across gfx architectures wherever its semantics agree,
   confining genuine target differences to the reviewed capability or adapter
   boundary from Stage 1;
3. introduce the smallest usable implementation behind a narrow seam;
4. migrate only that component's consumers;
5. run focused host tests and the relevant cross-architecture device contracts;
6. switch that component only after the new path is green;
7. delete its superseded implementation and temporary seam promptly; and
8. run the broader proportional test gate before beginning a dependent slice.

Every commit should leave ConSan buildable and testable. Temporary old/new
coexistence is permitted only within the component currently being migrated,
with a named deletion point. There is no final global activation flag, no
week-long parallel rewrite waiting to be connected, and no accumulation of
several unqualified component migrations behind one eventual switch.

Keep the design documentation and DBI requirements ledger current as code
reveals new facts. If a slice invalidates the design, stop and revise the
remaining route before continuing; do not compensate with an ad hoc large
cutover. End the implementation week with the scoped internal end-state landed,
the superseded code for completed slices removed, and an evidence-backed package
for the DBI discussion.

### Constraints and non-goals

- All applicable tests remain green, especially the checked-in ConSan device
  contracts. Apply the behavioral-contract correction process described in
  Phase 2 if a test is found to encode a prototype accident.
- This is internal restructuring, not permission for a broad rewrite. Establish
  a design and validate it through narrow slices before moving large bodies of
  code.
- Do not rewrite production code during the initial design-only day. The gate
  into implementation requires a reviewed weekly end-state and a fully
  incremental, component-by-component migration sequence.
- No big switch is permitted. At most, switch one sufficiently small component
  at a time after its focused and relevant device tests pass. If a proposed
  component cannot be migrated while the rest of ConSan remains working, split
  it further or redesign the seam.
- Each intermediate revision must be a valid stopping point: buildable,
  testable, and free of a dependency on a future global cutover. Temporary
  compatibility paths must be local, short-lived, and paired with an explicit
  deletion slice.
- Apply the standing type-first contract policy from
  `PRODUCTION_DESIGN.md` to every slice. Every introduced or materially changed
  named type and enum must explain its semantic role, invariants, ownership or
  lifetime, and sentinel states generously at the declaration. Every
  reasonably testable behavior owned by that type must land with a focused
  host unit test; device-only behavior must land with the narrowest applicable
  paired correct/incorrect device contract. Indirect integration coverage does
  not satisfy this requirement.
- Do not migrate to the proposed DBI record stream, common hook, probe API, or
  variant model during this phase.
- The internal design should generally move in a direction that will compose
  naturally with DBI. Wherever DBI's broad design is compatible with ConSan's
  requirements, go with that flow: prefer clean client/framework boundaries,
  target-neutral sanitizer semantics, explicit capability requests, and
  independently testable contracts over ConSan-specific instrumentation
  machinery. This is directional alignment, not an instruction to distort or
  discard features to fit DBI's near-term limits. Where there is genuine
  friction, especially DBI's present host-processing model versus Inline
  Shadow's on-device processing, preserve and describe the ConSan requirement,
  isolate the boundary, and state concretely how DBI would need to evolve. The
  intended long-term result is that every retained ConSan feature, including
  Inline Shadow, can migrate to DBI as the framework gains the necessary
  capabilities.
- Do not make Record/Replay's near-term DBI path dictate the internal semantics
  of Sampled, SuperCollider, or Inline Shadow.
- Do not resolve an uncertain DBI boundary by adding another permanent ConSan
  framework. Record the requirement for next week's discussion.
- The current `HSA_TOOLS_LIB` integration remains in place for this phase.

### Exit criteria

- `PRODUCTION_DESIGN.md` is complete and reviewable and defines both the scoped
  end-state for the implementation week and the complete incremental component
  sequence for reaching it.
- That design and sequence were reviewed against the Stage 2 implementation
  procedure and Phase 1 constraints. Every cutover is component-sized, every
  intermediate state has a test gate, and no part of the plan depends on a
  later global switch.
- The design explains each mode's semantic contract and data flow, the valid
  sharing between modes, and the dependency direction between internal layers.
- Every component has an implemented or explicitly planned all-five-target
  disposition. Shared semantics use shared code, and every remaining
  architecture-specific path has a documented architectural reason and a narrow
  boundary.
- Current code has been classified against that design, with a prioritized set
  of deletion, reuse, separation, and refactoring slices.
- The scoped implementation-week end-state has been reached through individually
  tested component migrations. No mode-wide or system-wide switch was used, and
  completed slices have removed their superseded code and temporary seams.
- Every intermediate revision was buildable and testable, and each component
  cutover was qualified before work depended on it.
- The host and device test suites remain green, and new internal contracts
  introduced by refactoring have focused regression tests.
- The design documentation describes the resulting components, invariants,
  boundaries, and migration decisions clearly enough for human reviewers and AI
  agents to continue the work without reconstructing the design from code.
- We have a concise, concrete DBI requirements and questions package with which
  to join the framework design discussion next week.

## Phase 2: Align and migrate with DBI

Begin this phase after the scoped Phase 1 implementation is complete and its
internal design and requirements package have been reviewed. The DBI material
below records the current direction and known answers, but Phase 1 findings are
expected to refine it before implementation.

### Governing DBI constraint

Allyson Cauble-Chantrenne's forward-looking
[DBI overview](https://github.com/ROCm/rocm-systems/pull/10407) is a governing
architectural input to this plan. The overview describes the intended stable
DBI framework, while the checked-in [DBI design](../dbi-design.md) describes the
implementation that exists today. Where they differ, ConSan must target the
overview without pretending that its required framework facilities already
exist. Track the overview PR until it lands; its merged form then becomes the
canonical constraint, and this plan must be reconciled with any intervening
design change.

ConSan is DBI's first client, not an independent instrumentation framework. The
production replacement must therefore become a comparatively small client of
DBI. It may supply the sanitizer semantics and requirements that shape DBI, but
it must not retain private copies of generally useful instrumentation,
relocation, resource-management, transport, or runtime-interposition machinery.
The DBI overview deliberately leaves ConSan's detection algorithm, shadow
semantics, payloads, and reports unspecified; those remain this plan's design
responsibility on the client side of the boundary.

This is an architectural ownership boundary inside RocJitsu, not a repository
or packaging boundary. Production ConSan, including its in-tree probes and host
analysis, remains RocJitsu code. For now, DBI clients and their fixed probes are
developed in-tree rather than through a separately supported external plugin or
probe-authoring API.

The DBI overview distinguishes **committed**, **must remain possible**, and
**open** decisions. ConSan must obey committed framework contracts, must not
foreclose the second category, and must not accidentally settle an open DBI
question inside client code. When ConSan needs an open question resolved, it
should provide a concrete requirement and test and coordinate the decision with
the DBI owner.

The intended ownership boundary is:

| Concern | Production owner |
| --- | --- |
| Decode, CFG and liveness analysis, target capabilities, instruction properties, relocation and target emission | Shared RocJitsu substrate and DBI |
| Code-object-to-code-object transforms, variant production, composition with translation, and original/instrumented address mapping | DBI |
| Instrumentability, site placement, probe invocation, execution-mask policy, register preservation, spilling, outstanding-memory handling, and per-site outcomes | DBI |
| Runtime interception, dispatch-time variant selection and sampling, buffer binding, hook composition, and failure containment | DBI |
| Record-stream envelope, transport, flow control, loss accounting, and coverage accounting | DBI |
| Which memory and synchronization events matter to the sanitizer, which provided probes and payloads represent them, and how the four current modes select or interpret evidence | ConSan |
| Race semantics, happens-before and ownership models, host-side analysis, diagnostic policy, and user-facing ConSan reports | ConSan |

The DBI framework's committed invariants are inherited rather than restated as
ConSan inventions. In particular, instrumentation is observational and must
preserve all guest-visible state; coverage loss, rejected sites, unsampled
dispatches, and dropped records must be visible; transforms operate on arbitrary
input code objects and compose; load-time variant production is separate from
dispatch-time selection; probes come from a fixed, verified catalogue; host-side
analysis is the normal place for complex client logic; device-visible buffers
are acquired by the host ahead of execution, with pinned fine-grained host
memory as the committed streaming placement; probes do not own scratch, LDS,
or persistent state; hooks chain rather than silently displacing other tools;
and simulator and hardware are both first-class targets.

Several existing ConSan mechanisms are therefore suspect by construction. Any
target-specific trampoline, register allocator, spiller, wait sequence,
code-object patcher, report transport, or HSA interception path in ConSan must
either migrate to DBI/shared RocJitsu code or be deleted. A genuinely
sanitizer-specific exception needs a first-principles explanation of why the
DBI contract cannot express it.

The HSA hook is a deliberate temporary exception to the migration timing. Keep
using the current `HSA_TOOLS_LIB` path while building the first production
ConSan client. DBI is intended eventually to own the common hook, at which point
ConSan will migrate to it, but that framework path is not ready and must not
block the earlier Record/Replay work. Keep the temporary hook boundary narrow
and avoid adding new generally useful instrumentation policy to it.

The four current execution modes follow two deliberately dissociated timelines.
Near-term DBI requires host-only processing and therefore strongly favors
Record/Replay. Make Record/Replay the first production ConSan client and do not
make its delivery wait for the other modes. DBI must keep the door open for
future on-device processing, and other instrumentation use cases are expected
to drive that evolution. Sampled, SuperCollider, and Inline Shadow therefore
remain later migration tracks rather than being forced prematurely into the
host-only model or declared obsolete.

Until DBI has a reviewed on-device-processing contract, preserve the other
modes and keep their tests green, but do not make production Record/Replay
patches carry their prototype abstractions. When the DBI capability exists,
assess each mode against it and preserve its useful user-visible behavior
without assuming that its current data path or internal hierarchy is the
contract. Any later retirement or material contract change still requires the
behavioral-contract test process below.

The current DBI implementation is only partway to the overview. It already has
transactional preflight, probe-call trampolines, shared decode/liveness support,
register spilling, and simulator coverage for CDNA3, CDNA4, and RDNA4. Its
checked-in design also records important gaps: instrumentation is presently
all-or-nothing, site kinds and predicate selection are limited, resource growth
and zero-scratch cases are incomplete, the full ConSan target matrix is not
covered, and the forward-looking variant, dispatch-selection, stream, and
runtime-coexistence model is not yet the production path. The ConSan migration
must therefore be capability-driven rather than a flag-day cutover. Closing a
gap means implementing the general DBI contract with DBI-focused tests, not
moving a ConSan-specific prototype mechanism wholesale into the framework.

Two previously uncertain capabilities can be treated as intended DBI design:
both before- and after-instruction instrumentation, and per-dispatch buffer
binding. Their current implementation gaps remain work, but they are not open
architectural questions for ConSan. Conversely, ownership of the complete
`gfx942`/`gfx950`/`gfx1100`/`gfx1201`/`gfx1250` support guarantee remains
unresolved. Retain the five-target ConSan requirement and track the missing DBI
enablement explicitly rather than assuming either that DBI owns it already or
that target branches must remain in ConSan.

Instrumentation of DBT-translated code and composition of attribution mappings
through DBT are deferred until a concrete use case arises. Preserve DBI's
code-object transform abstraction and avoid gratuitously foreclosing
composition, but do not make translated-code support or its mapping chain a
Record/Replay milestone.

### Behavioral constraint

The host and device tests developed during the completed preparation work are
the primary constraint on the migration. Every incremental step must keep all
applicable tests passing, with particular attention to the checked-in ConSan
device tests across every supported target.
Targeted host-unit tests should guard individual interfaces and invariants;
the device suite should guard observable correct/incorrect workload contracts;
physical-device and end-to-end validation should remain qualification gates for
behavior that the quicker tiers cannot faithfully exercise.

The tests are a behavioral oracle, but they are not automatically the product
specification. Some tests will inevitably turn out to entrench an incidental
trait of the prototype rather than an intended ConSan contract. Such a test may
be changed, but never merely because it blocks a refactoring. Before changing
it, we must:

1. identify the prototype-specific assumption that it encodes;
2. determine the intended behavior from first principles, documentation,
   architecture semantics, and relevant end-to-end workloads;
3. record the resulting contract explicitly;
4. replace or revise the test so it checks that contract, retaining meaningful
   correct/incorrect coverage and cross-architecture coverage; and
5. make the contract clarification reviewable separately from, or clearly
   visible within, the implementation migration that depends on it.

Deleting, weakening, or narrowing a test simply to recover greenness is not an
acceptable migration technique. Conversely, preserving a prototype accident
solely because a test happened to capture it is not a design requirement.

### Design principles

Human reviewers should be able to see that the replacement:

1. **Minimizes added code.** Prefer deletion, reuse, tables, and existing
   RocJitsu facilities over new parallel mechanisms. Any net growth must have a
   clear semantic justification.
2. **Expresses first-principles meaning.** Code should state architectural and
   sanitizer invariants directly. It should not encode a history of individual
   workload failures as unexplained branches, encodings, and constants.
3. **Maximizes sharing across architectures.** Equivalent behavior should use
   shared DBI analysis, planning, instrumentation, and validation paths.
   Target-specific code belongs at DBI's ISA boundaries and should be limited
   to genuine ISA, ABI, and capability differences.
4. **Honors ownership boundaries.** ConSan owns sanitizer policy, analysis, and
   diagnostics. DBI owns the mechanics that safely instrument and observe GPU
   execution. Interfaces between them should be explicit and independently
   testable.
5. **Reuses and strengthens DBI.** Use DBI's decoder, instruction-building,
   code-object, relocation, spilling, probe, stream, and dispatch facilities.
   Improve DBI when a missing capability is general instead of creating a
   ConSan-only copy.
6. **Remains explainable and reviewable.** Each abstraction should correspond
   to a real invariant and normally serve multiple consumers. Generic layers
   that merely hide duplication are not an improvement.

### Candidate destination architecture

The destination has a DBI-owned substrate and a ConSan-owned client, with
dependencies flowing from ConSan into stable DBI interfaces rather than into
target-specific implementation details.

#### DBI-owned substrate

The DBI overview, not this plan, is authoritative for this layer. ConSan relies
on it to provide:

1. composable code-object transforms and load-time production of instrumented
   variants;
2. offset- and predicate-based site selection, before- and after-instruction
   placement, and explicit instrumented or rejected outcomes;
3. verified, target-correct probe placement that preserves guest state,
   relocates displaced code, handles registers and outstanding memory, and
   updates all code-object and dispatch resources consistently;
4. a fixed probe catalogue and data-driven probe descriptors rather than
   client-specific calling-convention assumptions;
5. dispatch-time variant selection, filtering, sampling, per-dispatch buffer
   binding, and eventually composable runtime interception;
6. a self-describing record stream with explicit flow control, record loss,
   coverage, and original-site attribution; and
7. consistent operation on simulator and hardware, including differential
   validation of the shared transform.

Target capabilities, instruction builders, relocation, register/resource
planning, and runtime-specific integration terminate inside this substrate. A
target name appearing in production ConSan code is therefore a design-review
event.

#### ConSan-owned client

The production client should contain only:

1. **A target-neutral sanitizer event model** describing accesses, barriers,
   atomics, fences, owners, epochs, scope, confidence, and typed exclusions.
   It asks DBI for decoded properties; it does not decode target instructions
   itself.
2. **Selection and mode policy** mapping those semantic events and user
   configuration to DBI sites, provided probe kinds, payloads, dispatch filters,
   and required-versus-best-effort coverage.
3. **ConSan payload schemas** within DBI's stable record envelope. Device work
   should capture the minimum evidence needed by the selected client policy,
   not embed general instrumentation machinery.
4. **Host-side race analysis** implementing the supported ordering, ownership,
   sampling, and conflict semantics over the received evidence.
5. **Coverage and trust evaluation** combining DBI's rejected-site, sampling,
   saturation, and loss information with ConSan's semantic requirements so that
   an incomplete run can never masquerade as a clean one.
6. **Stable diagnostics and reports** that expose conflicts, attribution,
   coverage, incompleteness, and overhead without exposing prototype layout or
   target encoding details.

Record/Replay is the first client to move onto this architecture. The exact
treatment of the other three prototype modes remains a later design output, not
a presupposed four-way implementation hierarchy. Share the target-neutral event
and analysis model wherever retained product contracts genuinely agree, but do
not manufacture a generic engine layer merely to preserve prototype names.

This split is a hypothesis to validate through the implementation inventory and
early migration slices, not an excuse to pre-build a large framework. Revise a
boundary when concrete evidence warrants it, in coordination with DBI ownership,
while preserving one-directional dependencies and the thin-client goal.

### Design work before broad replacement

Begin with the reviewed Phase 1 internal design, component inventory, test map,
and requirements ledger. Do not repeat that exploration. Reconcile those
artifacts with DBI's ownership and readiness, then produce the DBI transition
design:

1. Turn the DBI overview's committed requirements into a responsibility and
   dependency checklist for ConSan. Track its open questions separately; do not
   bury provisional answers in implementation code.
2. Compare the checked-in current DBI design with the forward-looking overview.
   Identify which production facilities exist, which are incomplete, and which
   require coordinated DBI work before a ConSan path can migrate.
3. Reclassify each Phase 1 component at the DBI boundary as ConSan-owned,
   DBI-owned, shared RocJitsu substrate, temporary adapter, or prototype-only
   deletion. Preserve Phase 1's cross-architecture sharing unless a genuine DBI
   capability boundary gives a documented reason to change it.
4. Extend the Phase 1 test map with DBI framework invariants and integration
   gates. Generic DBI invariants should gain focused DBI tests while retaining
   ConSan device coverage of the resulting user-visible behavior; do not lose
   or duplicate behavioral intent merely because the owning suite changes.
5. Specify the production Record/Replay client against DBI's near-term
   host-processing model. Write down its event, payload, per-dispatch binding,
   coverage/failure, host-replay, and reporting contracts, including the exact
   before/after instrumentation expected from DBI.
6. Inventory Sampled, SuperCollider, and Inline Shadow separately as future
   on-device-processing clients. Preserve their behavioral contracts and record
   what DBI facilities each would eventually require, but do not put those
   facilities on the Record/Replay critical path.
7. Represent target differences as DBI capabilities rather than ConSan branches.
   Resolve the currently unanswered ownership and enablement plan for all five
   required targets before declaring production Record/Replay complete.
8. Resolve or schedule the DBI decisions that block Record/Replay. Immediate
   subjects include the concrete per-dispatch buffer/kernarg lifetime contract,
   relocation of large kernels, original/instrumented mapping for ordinary DBI,
   memory-region scope, entry/exit instrumentation, scale-out, and buffer
   sizing. The future common hook, external advanced-user API, on-device mode
   support, and DBT-translated-code attribution are not first-client blockers.
9. Define the ordered migration slices, the old/new boundary during each slice,
   the DBI and ConSan tests that gate it, the upstream dependency it assumes,
   and the prototype code that will be deleted when it lands.
10. Record quantitative baselines for total code, architecture-specific code,
    duplicated DBI machinery, and dependency shape so that the claimed
    simplification can be measured rather than inferred.

### Dissociated delivery milestones

#### Milestone A: production Record/Replay

Deliver a production-grade, host-processing Record/Replay client first. It may
continue to enter through the existing `HSA_TOOLS_LIB` hook, but its
instrumentation mechanics should move onto the DBI abstractions available or
added for site selection, before/after probes, transparent placement,
per-dispatch buffer binding, record streaming, and explicit coverage.

This milestone must support all five ConSan targets and pass the complete
Record/Replay host, device, physical, and relevant end-to-end contracts. It
does not wait for DBI-owned common-hook migration, on-device processing,
production migration of the other three modes, an external probe API, or
instrumentation of DBT-translated code.

#### Milestone B: later on-device modes

As DBI's on-device-processing model becomes concrete, reassess Sampled,
SuperCollider, and Inline Shadow independently. Migrate each useful behavior
onto reviewed DBI facilities, with its current behavioral tests as constraints,
without coupling their schedules to one another or recreating their shared
prototype framework inside ConSan.

The future common DBI hook can also replace the temporary ConSan hook when it is
ready. That migration is orthogonal to proving the first production client and
should occur as its own tested slice.

### Incremental migration method

Each migration slice should establish one real production boundary or move one
coherent behavior across an already established boundary. As in Phase 1, there
is no migration-wide switch. The maximum cutover is one sufficiently small
component whose contract and implementation have already passed their
proportional gates. If a DBI integration slice cannot coexist with components
that have not yet migrated, its boundary is too large and must be split or
redesigned.

For every slice:

1. State the invariant or duplication being addressed and show how the change
   advances the destination architecture.
2. Put the implementation and focused unit test at the owning layer. A generic
   instrumentation correction belongs in DBI with a DBI regression; retain or
   add the ConSan device contract that demonstrates its user-visible effect.
   If the work exposes a test that encodes a prototype accident, apply the
   contract-review process above before relying on changed expectations.
3. Generalize from multiple targets when possible, then validate the result on
   all five supported architectures. Target-native encoding changes should be
   confined to DBI; ConSan should see one semantic event and one client
   contract.
4. Introduce the smallest usable DBI or ConSan production path, migrate its
   consumers, and delete the corresponding prototype implementation in the same
   reviewable series. When a needed DBI facility is not yet available, add it
   through the DBI design and ownership process or use a narrow adapter with a
   named upstream dependency and removal point. Do not grow a second framework
   under the label of temporary compatibility.
5. Keep behavior-preserving restructuring separate from intentional behavioral
   changes whenever practical. Where they cannot be separated, make the
   contract decision and its evidence explicit.
6. Run the proportional fast gates while iterating, then discharge omitted
   gates before the slice is complete. Do not accumulate multiple unqualified
   slices behind an untested old/new boundary.

Prefer a sequence that moves general mechanics downward into DBI and leaves
sanitizer meaning above its client boundary. The exact order should be finalized
from the responsibility inventory and the DBI implementation roadmap, but the
likely progression is:

1. establish the DBI/ConSan responsibility ledger and isolate the Record/Replay
   path from abstractions needed only by the three later modes;
2. define the target-neutral Record/Replay event, payload, per-dispatch binding,
   coverage, and host-analysis contracts against the forward-looking DBI API;
3. close the smallest DBI gaps needed for a representative vertical slice,
   guarded by DBI unit and simulator/hardware transparency tests;
4. migrate that slice to DBI site selection, probes, transport, and coverage
   outcomes while retaining its ConSan correct/incorrect device contracts;
5. expand Record/Replay slice by slice across its semantic inventory and all
   five targets, deleting the corresponding ConSan-owned instrumentation
   mechanics as their DBI equivalents take over;
6. qualify Milestone A and remove all Record/Replay-specific temporary adapters,
   target emitters, and superseded prototype scaffolding; and
7. only as the required DBI facilities mature, begin independent Milestone B
   migrations for Sampled, SuperCollider, Inline Shadow, and the common hook.

Pilot the approach on a record-streaming vertical slice that is small enough to
review, fits the DBI overview directly, and is representative across at least
two meaningfully different targets. Record/Replay is the selected first client.
Use the pilot to validate the DBI/client boundary, the ownership and review
process, and the migration mechanics before expanding it across Record/Replay.
Do not use the pilot to force the later on-device modes through the same data
path.

### Test and qualification cadence

During inner-loop development, run the focused host tests and relevant device
contracts first, including the focused DBI tests for any framework change,
followed by the parallel RocJitsu-emulated matrix. Run the entire checked-in
ConSan device suite frequently enough that architecture- and mode-specific
regressions do not accumulate. The serialized physical `gfx950` tier may be
omitted temporarily from fast iterations, but must run for native-sensitive
changes and before a migration slice is considered complete.

At review and qualification milestones, run:

- all applicable RocJitsu host and unit tests;
- DBI transform, probe, transparency, simulator, and applicable hardware tests;
- the complete checked-in ConSan device suite under RocJitsu for `gfx942`,
  `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`;
- the physical `gfx950` ConSan device tier; and
- relevant CDNA4 and gfx1250 end-to-end validation, expanding milestone
  coverage as prototype paths are removed.

The migration must not knowingly regress unrelated RocJitsu tests. A focused
test passing is evidence for a change, not permission to ignore the broader
suite. Failures exposed through ConSan should be triaged to their owning layer:
DBI mechanics receive DBI regressions, ConSan semantics receive ConSan unit
regressions, and observable sanitizer behavior remains covered by the ConSan
device contracts. Test placement is a maintainability choice rather than an
early architectural blocker: use engineering judgment, and allow a focused test
to remain in the ConSan suite temporarily when that gives the quickest durable
coverage. Tests may be shuffled to DBI later as ownership settles, but never at
the cost of losing the behavior they guard.

### Completion criteria

#### Milestone A

- Production Record/Replay is a host-processing client of the forward-looking
  DBI architecture. It contains no private relocation, trampoline,
  register/spill, wait, target-emission, code-object mutation, or record
  transport framework.
- The existing `HSA_TOOLS_LIB` integration is the only intentional temporary
  framework boundary. Its eventual replacement by the common DBI hook is
  recorded and it does not accumulate unrelated instrumentation policy.
- Record/Replay policy and host analysis consume target-neutral DBI interfaces.
  Equivalent behavior is implemented once in ConSan across `gfx942`, `gfx950`,
  `gfx1100`, `gfx1201`, and `gfx1250`; genuine ISA variation is represented by
  shared RocJitsu/DBI capabilities and lowerers.
- Before- and after-instruction placement, per-dispatch buffer identity,
  transparency, explicit coverage/loss, and simulator/hardware consistency are
  demonstrated by focused DBI tests and retained ConSan behavioral contracts.
- Record/Replay prototype duplication, compatibility branches, and superseded
  target code are removed. Measurements show a material reduction in its total
  code, architecture-specific code, duplication, and dependency complexity.
- Sampled, SuperCollider, and Inline Shadow remain functional and their tests
  remain green, but their production migrations and DBI on-device-processing
  dependencies do not block this milestone.
- All retained and corrected Record/Replay contracts pass across every
  applicable target. Any test changed during the migration has a recorded
  explanation of the prototype accident it previously encoded and the intended
  behavior that replaced it.
- Host, emulated-device, physical-device, and relevant end-to-end qualification
  gates pass on the same reviewed revision.
- The Record/Replay patch series and resulting implementation are organized
  around explicit invariants and are small and clear enough for meaningful
  human review without knowledge of the prototype's construction history.

#### Later milestones

- DBI has a reviewed on-device-processing model sufficient for each mode being
  migrated.
- Sampled, SuperCollider, and Inline Shadow are migrated independently, with
  explicit contracts and without reintroducing target-specific instrumentation
  frameworks into ConSan.
- The common DBI hook replaces the temporary ConSan hook in its own tested
  migration slice.
- Tests may move to their final DBI or ConSan owning suites, while all observable
  behavioral coverage and the complete supported-target matrix remain intact.
