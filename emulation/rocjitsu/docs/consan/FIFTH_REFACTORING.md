# ConSan fifth refactoring: deepen the component architecture

This document is the working charter and execution record for ConSan's fifth
refactoring.
It begins with a post-fourth-refactoring deep read of the production code. That
assessment is deliberately separated from the mandate and possible directions
for the next refactoring: the current-state sections describe what the code is,
while the later sections are an open design space that will grow as further
questions, experiments, and implementation evidence refine the destination.

The fourth refactoring produced major structural gains. ConSan now has a real
semantic dataflow, explicit lowering transactions, typed runtime mappings,
separately testable host-report responsibilities, and far smaller compiler-
visible implementation closures. It did not, however, complete the stronger
goal of a deeply encapsulated core whose component and build boundaries enforce
the conceptual architecture. The fifth refactoring starts from that distinction
rather than either dismissing the gains or accepting the prior completion audit
uncritically.

No destination class diagram or fixed list of mechanical extractions is
prescribed. The destination remains discoverable, but the direction of travel,
iteration discipline, and evidence required for completion are binding.

# Part I: post-fourth-refactoring assessment of the existing code

## 1. Overall finding

The implementation is no longer a textual monolith. Its strongest layers now
communicate through meaningful semantic products, and the lowerer/runtime
boundary is materially better. Internally, though, much of the core is still
organized as many named translation units around broad shared contracts and
mutable transaction state. The present architecture is therefore best
described as **physically decomposed, semantically well directed along its main
spine, but only partially encapsulated at the component boundaries**.

The largest remaining design knot is MOI resource placement. It combines mode
demand, target ABI and register constraints, fallback selection, owner-local
liveness, descriptor growth, dynamic-stack behavior, and placement state in a
single large algorithmic region. This is also the clearest surviving
mode-by-target interaction region.

## 2. The actual semantic dataflow

The following forward flow is present in the code and is one of the fourth
refactoring's genuine successes:

```text
code object
   |
   v
program and synchronization analysis
   |
   v
normalized access and atomic classifications
   |
   v
semantic observation policy
   |
   v
observation plan and probe intents
   |
   v
resource planning and engine lowering
   |
   v
intent-bound commits or rejections
   +----> coverage ledger
   +----> runtime static mapping
   `----> private patch proof
                    |
                    v
             final validation
                    |
                    v
              public result
                    |
                    v
        host snapshot -> decode -> analyze -> trust -> render
```

`ProgramInventory`, `ConSanObservationPlan`, `ConSanCommittedLowering`,
`ConSanCoverageLedger`, and `ConSanRuntimeStaticMapping` carry semantic meaning
forward. Coverage and runtime attribution no longer have to reconstruct that
meaning from patch kinds, offsets, or trampoline geometry.

## 3. Effective components in the current implementation

The code now has the following de-facto components. This table describes their
actual cohesion and boundaries, not merely their filenames or the component
labels used in the fourth-refactoring completion audit.

| Effective component | Actual authority and product | Present boundary quality |
| --- | --- | --- |
| Request, contracts, and public result | Configuration contracts, public stage records, replacement image, public diagnostics | Semantically coherent, but exposed through a broad aggregation header |
| Program and synchronization analysis | Code-object parsing, decoded semantic inventory, normalized synchronization graph | Genuine inventory authority; also a large cross-target raw decoder |
| Exact target normalization | Access and atomic lowerability classification and normalized lowering forms | Strong centralized authorities with focused target knowledge |
| Semantic policy and observation assembly | Access, barrier, atomic/fence, and perturbation policy assembled into one observation product | One of the cleanest and most target-neutral layers |
| Executable transformation transaction | Stage execution, deferred binding, resume, publication, and public/private result projection | Real pipeline; still knows several engine-specific evidence and cleanup cases |
| Lowering composition | Analysis, mutation, engine selection, placement, patch construction, and finalization coordination | Physically separate, but depends on nearly all lowerer components and the broad artifact aggregate |
| MOI resource solver | Immutable problem input, mutable search workspace, attempted and accepted operating points | Explicit products are an improvement; internal mode/target interaction remains concentrated here |
| Shared MOI mechanics | Probe planning, addressing, relocation, native ABI, reporting, synchronization, prologues, routing, and workgroup gates | Many useful typed seams; still a broad collection rather than one small component |
| Engine lowerers | Record/Replay, Sampled, InlineShadow, and SuperCollider lowering | Named owners exist, but important mode decisions remain in common orchestration and placement |
| Target operations | Fault, relay, MOI, SuperCollider, selectable-bank, and family/member recipes | Good typed facades in several areas; raw target work is not uniformly contained here |
| Mutation and independent validation | Fault and perturbation application plus independent verification of emitted code and private proof | Independence is real; validation remains intentionally broad and cross-target |
| Runtime report path | Lifecycle registry, snapshot, decoder, analyzer, trust, renderer, and coordinator | Strongest physically enforced component split in the system |

## 4. Physical decomposition versus enforced layering

### 4.1 Translation units

The old mega-translation-unit closure is gone. The core build compiles 77
separate `.cpp` files, and every active implementation fragment has one textual
owner. No implementation fragment includes another except the deliberate
three-region SuperCollider wrapper. This materially improves source ownership,
compiler isolation, focused tests, and the ability to navigate the code.

The remaining active `.inc` bodies still account for 36,477 implementation
lines. Some are substantial local component bodies, including the MOI placement
body, independent validation, and the SuperCollider LDS body. They are no
longer a hidden include graph, but a one-owner `.inc` is not by itself a narrow
component.

### 4.2 Build graph

All 77 core translation units are registered as sources of the single
`rocjitsu_code` target. Unlike the host report path, the core has no component
libraries or object-library layers that enforce dependency direction. The
source tree expresses conceptual boundaries through filenames and headers, but
the build graph does not prevent a lower layer from acquiring a higher-layer
dependency.

The host side is stronger. Snapshot, decoder, analyzer, trust, and renderer are
separate static libraries and are directly testable apart from HSA lifecycle
integration.

### 4.3 Broad visibility and shared buses

The main residual coupling surfaces are not accidental textual includes; they
are explicit, broad types and headers:

- `ConSanTransformArtifacts` remains the internal universal transaction bus.
  It contains inventory, observation, coverage, fault state, mutation state,
  resource alternatives, the MOI operating point, commits, runtime mapping,
  staged synchronization results, patch proof, replacement bytes, outcome,
  warnings, and errors. It appears in 57 core files.
- `ConSanMoiOperatingPoint` contains decisions and fallback state spanning
  several engines, target families, owner scopes, and resource mechanisms.
- `MoiOptions` combines the full immutable caller input with the complete MOI
  operating point. Low-level emitters no longer receive it, but mode lowering,
  barrier, synchronization, prologue, composition, and coordinator paths still
  do.
- `consan.h` aggregates almost all public and internal schemas and has 35 direct
  core consumers.
- `consan_moi_internal.h` is a 1,383-line shared contract surface with 23
  consumers. It contains solver, prologue, synchronization, routing, shadow,
  ABI-preservation, and placement contracts that do not constitute one cohesive
  responsibility.

`TransformResult` does successfully prevent these lowerer-private products from
becoming runtime API. That external boundary is strong even though internal
ownership remains weak.

## 5. Mode-specific isolation

### 5.1 What is now well isolated

Each engine has recognizable compiled owners:

- Record/Replay owns access recording, atomic and fence records, record
  planning, and record emission.
- Sampled owns watchpoint access lowering, sampled synchronization, sampled
  contracts, and its native emitters.
- InlineShadow owns its access path, atomic ordering, exact-shadow operations,
  and native emitters.
- SuperCollider owns a separate large engine component and uses named
  family/member target-operation files.

Low-level native emitters generally receive typed plans rather than the entire
option bus. Exact instruction admission is shared through normalized
classifiers rather than reimplemented per engine. Semantic policy is organized
by observation domain -- access, barriers, atomic/fence -- instead of being
copied four times. Mode switches in these policy authorities are often an
appropriate expression of one semantic decision table, not a violation.

### 5.2 Mode behavior still living in common code

`consan_moi.cpp` is considerably more than a dispatcher. It decides or adjusts:

- InlineShadow owner-source defaults;
- when Sampled atomic tracking has a usable access-window consumer;
- when InlineShadow barrier or atomic tracking should be suppressed;
- Record/Replay dense-router demand;
- mode-specific persistent-state suppression and fallback behavior;
- mode-dependent prologue timing;
- post-placement state removal based on emitted patch kinds; and
- mode-specific result diagnostics.

A fifth engine would therefore require edits throughout this common
coordinator, not merely registration of a new engine owner.

The MOI placement component is the larger interaction point. Its 6,694 lines
contain extensive Record/Replay, Sampled, and InlineShadow decisions intertwined
with target-family predicates and register/ABI constraints. The common barrier
and prologue components similarly own exact multi-mode behavior. Some of these
are honest subset-shared mechanisms, but the current interfaces do not always
separate the mode's requirements from the shared mechanism that satisfies
them.

`consan_moi_model.cpp` also colocates substantial Record/Replay and Sampled
host models. That is a real two-mode/subset responsibility, but its current name
and surface do not clearly advertise that ownership.

### 5.3 Current mode-isolation verdict

- Semantic policy: strong and appropriately shared by domain.
- Exact normalization and low-level emission: strong.
- Engine-owned lowering paths: materially improved.
- Common orchestration, resource solving, barriers, and prologues: only
  partially isolated.
- Adding an engine: additive for many target mechanics, but not plug-in-like.

## 6. Architecture-specific isolation

### 6.1 Strong target boundaries

The declarative five-entry `ConSanTargetProfile` table is a good authority for
target capabilities and family facts. Central access and atomic classifiers
normalize target encodings before semantic policy consumes them. This avoids
each mode independently interpreting raw instructions.

Named target and family owners now exist for:

- gfx9 and gfx12 fault operations;
- gfx9 MOI operations;
- gfx1250 LDS and selectable-register-bank behavior; and
- SuperCollider gfx9, gfx11/gfx12, RDNA3, RDNA4, and gfx1250 recipes.

The SuperCollider target-operation facade is a particularly clear example of a
common typed operation dispatching to family/member implementations without
putting engine policy in those implementations.

### 6.2 Raw target knowledge that remains centralized or leaked

Program analysis remains a large multi-architecture raw decoder. One component
directly manipulates generated CDNA4, CDNA5, RDNA3, and RDNA4 machine structures
for scratch, flat, global, buffer, LDS, atomic, and ordinary-memory forms. This
is target-normalization work, but it is not physically decomposed into
target-owned decoding adapters.

Independent validation similarly includes generated structures for several
architecture families in one 4,694-line implementation. Validation legitimately
needs raw proof access and must remain independent from construction, but that
does not make its target-specific portions isolated.

Shared MOI native-ABI code declares raw special-register constants with RDNA4
names while describing them as common to all supported profiles, including
EXEC, VCC, workitem ID, TTMP, and scope operands. If these are universal ABI
concepts, the common interface should be target-neutral. If they differ by
target, they should be supplied by a target contract. In the current code,
RDNA4 vocabulary is propagated through otherwise common and mode-owned paths.

Most importantly, the resource solver jointly reasons about engine semantics
and architecture constraints. Adding a sixth target for an already-normalized
operation would reuse much of the existing machinery, but it could still
require changes to analysis, classifiers, synchronization analysis, placement,
validation, native ABI handling, and target operations. Target addition is
therefore not yet confined to a target package.

### 6.3 Current architecture-isolation verdict

- Declarative capabilities and exact normalization: strong.
- Several concrete target-operation families: strong.
- Common semantic policy: target-neutral as intended.
- Raw program decoding and proof validation: centralized cross-target
  components rather than member-owned adapters.
- MOI ABI and placement: significant remaining architecture knowledge in
  common code.
- Adding a target: substantially better than per-engine duplication, but not
  plug-in-like.

## 7. Runtime and host-report layering

The host report path is the most convincing component realization produced by
the fourth refactoring:

1. lifecycle code owns allocation, binding, retirement, and registry state;
2. snapshot code captures quiescent bytes without owning evidence semantics;
3. the decoder turns bytes into typed events, issues, and losses;
4. the analyzer consumes decoded evidence and static mappings;
5. trust evaluation projects evidence completeness; and
6. rendering consumes typed products without reinterpreting raw bytes.

Runtime registration and analysis consume `ConSanRuntimeStaticMapping`, not
`ConSanPatchInfo`. That is a strong and important lowerer/runtime boundary.

There are still smaller ownership issues. `AutoMoiReportSummary`, currently
declared in the trust header, is a large mutable accumulator populated by
multiple stages even though trust evaluation merely consumes it. Decoder
contracts include the pipeline header, analyzer contracts include decoder
contracts, and renderer contracts include analyzer contracts, producing a
stacked contract dependency rather than a neutral report-product layer.
Sampled decoded evidence also borrows a static-mapping pointer, which is safe
under the synchronous pipeline but prevents the decoded product from being
self-contained.

These are layering imperfections, not a return to the previous host-report
monolith.

## 8. What the boundary checker proves -- and what it does not

`ConSan.ArchitectureBoundaries` supplies useful durable checks. It prevents:

- generated ISA dependencies in semantic-policy sources;
- lowerer-private patch and resource types in hooks;
- patch-geometry reconstruction in coverage and runtime analysis;
- `MoiOptions` use in files classified as native emitters or target operations;
- unowned or nested active implementation fragments; and
- unreviewed growth of several common mode-switch regions.

The checker does not mechanically prove the full component model. Its common
mode containment is partly a set of spelling-count budgets, including a budget
of 95 engine-enum references in MOI placement. Such a budget prevents silent
growth but preserves the current interaction as accepted debt. Generated ISA
dependencies in central program analysis and validation are explicitly
allowlisted. Raw-word matching recognizes selected literal shapes, not all
ways that architecture knowledge can enter common code. Core build-target
direction is not checked because the core remains one build target.

The checker should therefore be understood as a regression fence around
specific achieved properties, not proof that all architecture, mode, and
component boundaries are complete.

## 9. Fourth-refactoring goals: post-refactoring verdict

### 9.1 Goals that are substantively achieved

- Direct coverage from intent-bound commit and rejection events.
- Validated runtime static mapping with no patch-geometry dependency.
- Central exact access and atomic normalization authorities.
- Target-neutral semantic policy at the raw-ISA level.
- One executable transaction with typed deferred binding and stage records.
- One observation-policy and coverage-ledger assembly authority.
- Removal of unnamed generic eligibility such as `supported_mvp`.
- Public `TransformResult` isolation from private patch/resource proof.
- Separately testable host-report responsibilities.
- Independent final validation.
- Elimination of the old mega-translation-unit implementation closure.

### 9.2 Goals that are only partially achieved

- A short, stable component map enforced by the production source and build
  graph.
- One owner and one narrow product for every cross-component decision.
- A resource solver whose engine demands and target constraints are separately
  owned.
- Elimination of broad mutable option and artifact buses.
- Mode-specific and exact-subset containment.
- Raw target/member containment.
- Additive engine and target extension.
- Predominantly `O(N + M)` rather than interaction-shaped growth.

The narrow F5 result for native emitters is real, but the broader statement
that remaining `MoiOptions` references are confined to top-level orchestration
is not accurate. Mode lowering, shared barriers, synchronization, prologue,
composition, and coordinator paths still consume it. Likewise, the production
component table is useful documentation, but separate translation units in one
library do not constitute a directed build graph.

# Part II: executable charter for the fifth refactoring

## 10. Goal and mandate

### 10.1 The executable `/goal`

The fifth refactoring should be run under this goal:

> Complete the ConSan fifth refactoring by repeatedly migrating the
> highest-leverage mode/architecture/component interaction through a complete
> vertical slice: establish its rightful owner, replace broad or reverse
> dependencies with narrow forward typed contracts, converge every applicable
> production consumer, enforce the resulting boundary, and delete the
> superseded implementation. Preserve and extend behavioral evidence, commit
> each reviewable convergence step locally, and continue until the independent
> deep-read completion audit in Section 14 proves every required property
> across all existing modes and architectures. Do not push.

This is a convergence goal, not a task list. It intentionally specifies how
the code must improve and what evidence ends the work without assuming the
final class hierarchy, directory tree, solver organization, or migration
sequence.

The expanded architectural objective behind that operational statement is:

**Re-architect ConSan so concrete gfx mechanics and mode-specific behavior are
separately local, common mechanisms are implemented once, and the production
pipeline composes them through narrow typed products and enforced dependency
boundaries; migrate every existing mode and target to that architecture,
delete the superseded implementation, and materially reduce production code
without losing behavior, validation strength, diagnostics, or test coverage.**

This is the single goal of the fifth refactoring. The audit, locality ideals,
exploration questions, metrics, and eventual workstreams exist to advance and
test this statement. They are not independent objectives that can be completed
while the central composition problem remains.

The fifth refactoring is authorized to pursue a **deep redesign of ConSan's
internal architecture**, using the post-fourth assessment as evidence rather
than treating the existing components as boundaries that must be preserved.
It may move, split, merge, replace, or eliminate current components; introduce
new intermediate representations; change ownership of decisions; redesign
resource solving and engine composition; change build targets and visibility;
and revise the current mode/target decomposition when a clearer model emerges.

The aim is not merely to execute a short list of immediate-looking cleanups.
Those visible cleanups are evidence of pressure, not a sufficient scope. The
work should search for a coherent architecture in which:

- the conceptual design, source ownership, build graph, and test seams agree;
- the main dataflow is expressed by narrow, immutable, typed products;
- components do not coordinate by opportunistically reading and mutating a
  common aggregate;
- mode meaning and target mechanics compose without recreating mode-by-target
  implementations;
- validation remains genuinely independent without becoming a duplicate
  implementation of construction;
- adding an engine or target exercises the intended extension axes; and
- the resulting code is easier to explain, test, and change while containing
  materially less production implementation.

Code-size reduction is a required deliverable of the fifth refactoring, not
merely an expected side effect. There is still no license to optimize an
arbitrary line-count quota by weakening validation, coverage, diagnostics,
tests, target support, or clarity. Within those constraints, however, a design
that creates more opportunities to consolidate or delete code is preferable to
one that preserves the same implementation volume behind cleaner filenames.

### 10.2 Direction of travel

The destination representation is deliberately not prescribed, but every
convergence step must move knowledge and dependencies in the same direction:

```text
concrete gfx knowledge  ---->  architecture/family owners
mode-specific meaning   ---->  mode owners
duplicated mechanisms   ---->  one named shared owner
mutable shared state    ---->  immutable forward products
implicit dependencies   ---->  declared and enforced interfaces
superseded machinery    ---->  deletion
```

Equivalently, target packages publish normalized mechanics, constraints, and
facts toward the composition boundary; mode packages publish semantic demands,
strategies, and evidence contracts toward that boundary; and the common
transaction carries accepted products forward through analysis, policy,
resource solving, emission, proof, publication, and runtime consumption.
Concrete gfx knowledge must not flow into modes, mode policy must not flow into
targets, and downstream patch shape must not flow backward to recreate an
upstream semantic decision.

This supplies local guidance even while the final design is unknown:

- when concrete gfx knowledge appears in common or mode code, move it toward an
  architecture or family owner behind a normalized operation;
- when mode meaning appears in common or target code, move it toward a mode
  owner behind a semantic demand or strategy contract;
- when equivalent behavior appears in several owners, move it inward to one
  mechanism owner and make the owners compose it;
- when components communicate through a mutable union of their states, replace
  that exchange with the smallest forward product that expresses the accepted
  fact or decision;
- when a new path supersedes an old one, converge all consumers and delete the
  old path before calling the slice complete; and
- when two candidate designs are otherwise sound, prefer the one that removes
  more cross-axis knowledge, broad state, duplicated authority, and production
  implementation.

The exact boundary may move as evidence accumulates. The direction may not:
ownership becomes more local, products become narrower and more forward-only,
dependencies become more enforceable, sharing becomes more explicit, and
legacy implementation decreases.

Refactoring activity that does not produce one or more of those durable
changes is investigation or migration overhead, not convergence. It must
either enable a named near-term convergence step or be removed.

### 10.3 Create deletion opportunities and reap them

The refactoring has a two-part code-size contract:

1. **Create opportunities.** Redesign boundaries, representations, mode/target
   composition, and ownership so duplicated authorities, parallel mechanisms,
   compatibility paths, special cases, and legacy state machines become
   unnecessary.
2. **Reap opportunities.** Once a replacement is validated, consolidate every
   applicable consumer onto it and delete the superseded production
   implementation. An opportunity recorded but left behind as future cleanup
   does not satisfy the contract.

This makes deletion part of each vertical migration rather than a final sweep.
A completed slice should normally leave one authority, remove its displaced
adapters and implementations, pass its focused tests, and record the net
production-code change. Temporary coexistence is acceptable only while a
reviewable migration crosses a real dependency boundary; it must have a named
owner and a near-term removal checkpoint.

Code sharing means one implementation behind an honest contract. Moving
equivalent code into several mode or architecture packages, wrapping every old
path in a new facade, or replacing duplication with one giant switch-driven
implementation does not realize the intended benefit. The design should seek
representations and operations that allow whole branches, fallback protocols,
patch-shape joins, option plumbing, validators, and compatibility conversions
to disappear.

The exact post-fourth production baseline and recurring accounting method will
be recorded before implementation starts. Measurements must exclude tests,
generated code, documentation, comments, and blank lines so additional
regression coverage cannot obscure the production result. The final numerical
target remains to be chosen with the rest of the convergence criteria, but the
absence of an arbitrary quota cannot be used to declare success with no
material net reduction.

The operational no-file-deletion rule does not weaken this requirement. Legacy
implementation must actually be removed. If a filename must remain as an inert
comment-only tombstone, it contains no production implementation and is
accounted as deleted code rather than retained compatibility.

### 10.4 The vertical-slice iteration

Autonomous work should repeat the following loop, choosing one semantic fact,
mechanism, or interaction rather than reorganizing a whole horizontal layer at
once:

1. **Trace it deeply.** Identify the current authority, every producer and
   consumer, mode and target sensitivity, mutation and rollback behavior,
   validation proof, tests, and legacy alternatives.
2. **Assign ownership.** Decide whether the knowledge belongs to common
   semantics, one mode, an exact mode subset, one architecture, a target
   family, or the transaction/composition boundary. Name the reason, not the
   current filename.
3. **Define the forward contract.** Introduce or refine the smallest typed
   product or operation that lets the owner publish the fact without exposing
   its private working state.
4. **Migrate a complete vertical path.** Move production, consumption,
   validation, diagnostics, and focused tests together so the new boundary is
   exercised by real behavior rather than a disconnected abstraction.
5. **Converge and delete.** Move every applicable consumer, remove the old
   authority, adapters, fields, branches, and implementations, and leave no
   unbounded dual path.
6. **Enforce the boundary.** Add the narrowest useful build, visibility,
   compile-time, or structural check that prevents the dependency from growing
   back.
7. **Validate and account.** Run focused tests, the appropriate mode/target
   matrix, and periodic full gates; record production-line change, broad-type
   consumers, cross-axis references, dependency edges, and deleted legacy
   regions.
8. **Choose the next highest-leverage interaction.** Prefer a region that
   removes a broad dependency, unlocks several later deletions, or tests the
   emerging mode/target composition model on a second consumer.

A slice is not complete at step 3 or 4. Introducing a new interface without
consumer convergence, deletion, enforcement, and evidence is migration work in
progress. The next unrelated speculative slice should not begin while the
current slice leaves an unowned parallel authority.

### 10.5 Monotonic convergence scorecard

Before implementation begins, the refactoring will record a reproducible
baseline for the following signals. At every convergence checkpoint, the
cumulative direction must be non-regressing unless the checkpoint names the
specific near-term deletion payoff:

- concrete gfx identifiers and raw target dependencies outside target/family
  owners and the narrow registry;
- mode-specific branches and dependencies outside mode, named mechanism, exact
  subset, and narrow composition owners;
- illegal mode-to-target-implementation and target-to-mode-policy edges;
- consumers and fields of broad buses such as `ConSanTransformArtifacts`,
  `MoiOptions`, and the union-shaped MOI operating point;
- common-source engine-switch concentration and target-switch concentration;
- parallel authorities, migration adapters, and dormant legacy paths;
- unenforced component dependency edges;
- comment/blank/test/generated-excluded production implementation lines; and
- focused and full test inventory and coverage.

Not every individual commit must improve every signal. A vertical slice may
temporarily add a contract or adapter. It must identify the metric it will pay
back and the convergence checkpoint that removes the temporary path. Two
successive convergence checkpoints may not both defer the same promised
payback. Test inventory and validated behavior never decrease.

### 10.6 How to choose the next slice

At each checkpoint, choose the next interaction by **convergence leverage**,
not by convenience or file proximity. Prefer work that has several of these
properties:

- it is a mode-by-target knot or a high-fanout broad-bus dependency;
- it contains duplicated authority or enough legacy machinery to delete;
- resolving it makes a later mode or target extension more additive;
- it establishes a boundary that can be mechanically enforced;
- it exercises the emerging design on an additional mode, target family, or
  pipeline stage rather than polishing one exemplar; and
- it unlocks subsequent slices while leaving the tree bisectable and tested.

Use deep tracing to rank candidates. Line counts, token searches, and file size
may locate pressure, but they do not establish leverage or ownership. When two
candidates are comparable, prefer the one that simultaneously reduces
cross-axis knowledge, broad-state fanout, duplicated implementation, and
unenforced dependency surface.

This rule prevents wandering among equally attractive cleanups. A lower-value
cleanup may be taken when it is required to complete the current vertical
slice, pay back temporary migration growth, fix a discovered bug, or restore a
test gate; otherwise the highest-leverage unresolved interaction remains the
next task.

### 10.7 Anti-circling and continuation rules

The refactoring must accumulate irreversible architectural evidence rather
than repeatedly redescribing or rearranging the same code:

- Do not revisit a settled boundary merely to try a different aesthetic. A
  revision requires new implementation evidence: a failed extension exercise,
  an unavoidable forbidden dependency, surviving duplication, an invalidated
  semantic assumption, or a measurable inability to converge consumers.
- Do not count renaming, file movement, facade insertion, or a new product as
  progress by itself. Count the consumers converged, forbidden edges removed,
  old fields and branches deleted, boundaries enforced, and matrix behavior
  retained.
- Do not leave an attractive exemplar. A pattern demonstrated on one mode or
  target must be migrated across the applicable production set, or explicitly
  remain an incomplete slice in the ledger.
- Do not open several speculative architectures at once. Finish or falsify the
  current vertical slice, remove abandoned scaffolding, and record the
  evidence before pursuing the next hypothesis.
- Do not let two consecutive convergence checkpoints defer the same promised
  deletion, broad-bus reduction, or boundary enforcement. Pay back the
  migration or revise the approach.

Autonomous iteration continues while any Section 14 property lacks evidence or
the deep read still exposes a major broad bus, cross-axis knot, parallel
authority, scattered architecture/mode implementation, or material unharvested
deletion opportunity on the main production path. Exhausting the exploration
map, passing the existing tests, reaching the old line-count baseline, or
finishing one component family is not a stopping condition.

The work ends only after a fresh independent deep read, the extension
exercises, the quantitative accounting, and the full nonphysical and physical
test gates all support Section 14. If progress is genuinely blocked, record the
exact missing authority or external fact and stop as blocked; do not reinterpret
a local optimum as completion.

## 11. Design freedom and discovery

The destination remains intentionally open. In particular, this charter does
not yet assume that:

- the present pipeline stages are the final component boundaries;
- the four current engines should all implement the same internal interface;
- `ProgramInventory`, `ConSanObservationPlan`, or the current operating point
  are the final intermediate representations;
- a single universal resource solver is preferable to composed solvers;
- every existing shared-MOI region represents valid long-term commonality;
- target profiles, target-operation virtuality, generated adapters, templates,
  variants, or registries are the right mechanism for target composition;
- independent validation must remain one component; or
- current public/private header placement is the correct library API surface.

These questions should be answered through deep reads, dependency tracing,
focused prototypes, and tested vertical slices. The refactoring may discover
that some current commonality is accidental and should be separated, or that
apparently mode-specific code expresses a more general semantic operation. It
may also discover components not named in this initial audit.

### 11.1 Architecture locality as a directional ideal

One explicit ideal for the fifth refactoring is **physical locality of gfx
architecture support**. Code whose reason for existing is support for one
concrete gfx architecture should eventually be recognizable from its path
without reading its implementation. A near-term form could be a small set of
files bearing the architecture name, such as `*_gfx1201_*`; a stronger eventual
form could be an `arch/gfx1201/` package. The exact directory and interface
design remain open, but locality itself is a goal.

This ideal serves two audiences:

1. A maintainer adding a future gfx architecture should find an obvious,
   bounded place to implement its profile, decoding/normalization, ABI and
   placement constraints, native operations, and validation support.
2. A reader concerned with ConSan semantics, one engine, or a different target
   should be able to skip the concrete architecture packages and still
   understand the common design.

The desired reading boundary is stronger than hiding raw generated-ISA
includes. Common policy, engine, solver, and transaction code should speak in
target-neutral semantic operations and constraints. Except for a narrow
registration/composition authority, those components should not need to name a
concrete gfx architecture or know which file implements it. Conversely, a
concrete architecture package should implement target mechanics and facts; it
should not decide which ConSan mode semantically wants an observation.

Genuinely shared family behavior remains valid, but should be equally local and
honest: for example, an explicitly named `arch/rdna4/` or `*_rdna4_*` owner may
hold behavior proved common across that family, while thin concrete-target
packages supply member differences. “Shared” should not become a miscellaneous
home for target branches that happen to have more than one caller.

Physical locality does not require one giant file per architecture. A target
package may expose several cohesive facets -- for example profile facts,
inventory decoding, lowering operations, resource constraints, and independent
validation -- while keeping all concrete-target dependencies behind the same
package boundary. Nor does the ideal require duplicating family-shared recipes
into every member directory. The design problem is to make concrete and family
ownership visible while preserving exact normalization and independent proof.

A useful extension test for candidate designs is the hypothetical addition of
one fully supported architecture. For operations already represented by the
common semantic contracts, the expected change should be approximately:

- add one target or target-family package implementing the required facets;
- add one narrow registry/profile entry;
- add target-specific fixtures and matrix coverage; and
- make no edits to mode-owned policy or lowering code.

This is a design test, not yet a frozen plugin API or a claim that every future
ISA feature will fit without extending common contracts. When a new target
introduces a genuinely new semantic operation, the common contract may grow
once and all interested modes may elect to consume it. What should disappear
is the need to scatter routine target enablement through central analysis,
placement, validation, and every engine.

### 11.2 Mode locality as a directional ideal

A corresponding ideal is **physical and semantic locality of ConSan modes**.
Code whose reason for existing is Record/Replay, Sampled, InlineShadow, or
SuperCollider behavior should be recognizable from its path without requiring
a reader to infer ownership from enum branches inside common files. A near-term
form could be consistently named files such as `*_record_replay_*`; a stronger
eventual form could be packages such as `mode/record_replay/`, `mode/sampled/`,
`mode/inline_shadow/`, and `mode/supercollider/`. As with architecture
locality, the exact directory and registration design remain open.

Mode locality should provide the same two practical properties:

1. A maintainer adding a future ConSan mode should find a bounded place to
   define its semantic-policy contribution, evidence model, resource demand,
   lowering strategy, runtime/static mapping, report analysis, diagnostics,
   and validation proof.
2. A reader interested in common ConSan infrastructure or in one particular
   mode should be able to omit the other mode packages without losing the
   common control and dataflow.

This is not a mandate to duplicate domain authorities. Access, barrier,
atomic/fence, inventory, transaction, resource-search, and report-lifecycle
code may remain common when they express one genuine operation. The intended
separation is that common components own the operation and its composition
rules, while a mode-owned product states what that mode requests, supplies, or
implements. Common code should not gradually accumulate a complete handwritten
branch for each mode at every stage.

Locality is therefore an ownership rule, not a self-containment rule. A mode
package should depend on shared infrastructure instead of carrying private
copies of it. Moving equivalent access classification, routing, register
search, report handling, or validation logic into several mode directories
would make navigation look local while restoring `O(N)` duplication and future
semantic drift. That outcome is a failure of the refactoring even if every
duplicated copy has an unambiguous mode-prefixed filename.

The desired factoring is compositional:

- one shared implementation owns each genuinely common semantic operation or
  mechanism;
- each mode locally owns only its distinct policy, parameters, selected
  strategies, evidence contract, and irreducibly mode-specific behavior; and
- a mode assembles common mechanisms through narrow typed contracts rather than
  copying them or asking a central coordinator to reproduce the mode's entire
  implementation in a switch arm.

When two or more modes need nearly the same code, the default response should
be to identify and name the shared operation, not to accept parallel mode-owned
copies. Exact-subset ownership is appropriate only when the commonality is real
and its contract excludes the other modes for a semantic reason.

Mode locality is inherently less one-dimensional than architecture locality.
A mode has behavior across static policy, resource solving, device emission,
runtime evidence, host analysis, and final validation. The design must decide
whether one mode package contains those facets, whether it publishes them to
stage-oriented components, or whether consistently named mode-owned files
remain in several build layers. Any of these can satisfy the goal if ownership
and dependency direction are obvious and a reader can exclude the unwanted
mode. Merely moving deep switches into a file named `common` cannot.

Behavior shared by several modes needs similarly explicit treatment. When the
shared behavior has a real semantic name -- for example exact-shadow storage,
event logging, causal-window tracking, workgroup gating, or a routing strategy
-- it should be owned by a component bearing that mechanism's name. If no more
general abstraction is honest, an explicitly named exact-subset package is
acceptable. What should disappear is anonymous subset behavior encoded as
repeated mode tests inside a nominally universal coordinator.

The architecture and mode ideals are reciprocal boundaries:

- mode packages may select target-neutral operations and declare semantic or
  resource requirements, but should not include generated ISA types, raw gfx
  encodings, or concrete-target recipes; and
- architecture packages may implement normalized operations and constraints,
  but should not know which ConSan mode wants them or decide mode policy.

That reciprocity is the structural route toward `O(N + M)` growth. New modes
compose existing target operations; new targets implement existing normalized
operations. A genuine new semantic capability may extend the interface once,
but routine support should not require one implementation for every mode and
target pair.

A useful extension test is the hypothetical addition of a fifth mode. For
operations already represented by common semantic and target contracts, the
expected change should be approximately:

- add one mode package, or one clearly bounded set of mode-owned facets;
- add one narrow registration/composition entry;
- add mode-focused host, emulation-matrix, and physical tests as applicable;
- reuse shared mechanisms without copying their implementations; and
- make no edits to concrete architecture packages.

Common stage code may need a deliberately extensible interface, but should not
need new deep branches in analysis, placement, lowering coordination, report
processing, and validation merely to recognize the mode. This too is a design
test rather than a prematurely frozen plugin ABI.

## 12. Initial exploration map

The following is a non-exhaustive map of pressure points and design questions.
It is not a stage plan, priority order, or commitment to a particular solution.
It is expected to grow substantially.

### 12.1 Component and build architecture

- What is the smallest stable set of core components that explains ConSan
  without reference to include order or historical engine structure?
- Which dependency directions should be enforced by build targets, interface
  headers, or visibility rather than convention?
- Which current translation units are true components, which are implementation
  regions, and which are merely compatibility wrappers?
- Should the production library expose separate analysis, policy, lowering,
  target, validation, and runtime products, or is a different cut more natural?

### 12.2 Semantic facts, transactions, and ownership

- Can the transformation transaction own several stage products without
  passing one mutable `ConSanTransformArtifacts` bus through every component?
- Which facts are immutable observations, which are proposals, which are
  accepted decisions, and which are proof artifacts?
- Where are decisions currently recomputed, inferred from downstream state, or
  corrected after emission rather than established by one authority?
- Can rejection and fallback be modeled as first-class products without
  embedding every possible mechanism in one operating point?
- What transaction boundaries allow rollback and retry without broad mutable
  snapshots?

### 12.3 Mode composition

- What should the cohesive facets of one mode package be, and how should they
  cross static lowering, runtime evidence, host analysis, and validation build
  layers without losing mode ownership?
- Can mode-specific policy, evidence models, resource demands, lowering,
  mappings, analysis, diagnostics, and proof be made local to clearly named
  files or one mode directory?
- Can a reader understand the common transformation and one selected mode while
  deliberately excluding every other mode package?
- What is the essential semantic contract of an engine?
- Which behavior is truly common to all modes, to exact subsets, or only to
  one mode?
- What checks or measurements will detect equivalent implementations copied
  into separate mode packages instead of factored behind one shared contract?
- Can engine-owned demand/planning products replace deep mode switches in
  coordinators and the resource solver?
- Are Record/Replay, Sampled, and InlineShadow best understood as three
  implementations of one MOI abstraction, or as compositions of a smaller set
  of orthogonal observation and storage strategies?
- Which current subset components are authentic abstractions, and which hide
  coincidental similarity?
- Can a hypothetical fifth engine be used as a design test without requiring
  implementation of a new feature?
- Can that hypothetical extension be expressed as one mode package, one
  registry change, and tests, with build and boundary checks rejecting
  mode-specific additions to concrete architecture packages or unrelated
  common stages?

### 12.4 Target composition

- What should the cohesive facets of one concrete `gfxXYZ` architecture
  package be, and which interfaces connect those facets to common ConSan code?
- Can all concrete-target identifiers, generated ISA types, raw encodings,
  special-register operands, ABI differences, placement constraints, and
  validation recipes be made local to a few clearly named files or one target
  directory?
- Can a reader understand the complete target-neutral transformation while
  deliberately excluding `arch/gfxXYZ/` and family directories from the files
  they read?
- What normalized target facts should program analysis publish before common
  semantic code runs?
- Which decoding, emission, ABI, relocation, placement, and validation
  operations belong to a target, a target family, or target-neutral code?
- Can target/member adapters prevent generated ISA structures and raw operands
  from entering common analysis and validation while keeping those authorities
  exact?
- Which apparently universal ABI constants are genuinely universal, and which
  should be supplied by a target contract?
- Can a hypothetical sixth target implement already-normalized operations
  without edits to every engine or to central switch statements?
- Can that hypothetical extension be expressed as one package, one registry
  change, and tests, with build and boundary checks rejecting target-specific
  additions elsewhere?

### 12.5 Resource solving and mode-by-target interaction

- Can mode-specific resource demand, target-specific constraints, and
  target-neutral search be represented as separate products?
- Is the current fixed-point search one solver, several nested solvers, or a
  transaction among independent owner components?
- Which fallbacks are engine semantic decisions, which are target capability
  decisions, and which are generic optimization choices?
- Can owner-local liveness, descriptor growth, dynamic-stack support, spill
  policy, ABI preservation, and routing be composed without one union-shaped
  operating point?
- What would make the mode/target interaction term structurally small rather
  than merely reviewed and counted?

### 12.6 Construction and independent validation

- What proof should each construction component publish so final validation can
  remain independent without reverse-engineering every engine's patch shape?
- Which validation rules are target-independent invariants, target-native
  decoding, engine semantic proof, placement proof, or composition proof?
- Could validation be independently compiled from a collection of validators
  with explicit proof contracts while retaining defense in depth?
- Where does validation currently duplicate construction knowledge, and where
  is that duplication necessary independence?

### 12.7 Runtime and report products

- Should lifecycle-neutral report product types be separated from the pipeline,
  decoder, analyzer, trust, and renderer headers?
- Who owns summary accumulation, and can decoding products be self-contained?
- Can engine-specific report analysis use the same composition model as static
  lowering without coupling runtime to private lowerer mechanisms?
- Are static mappings the final runtime contract, or can their ownership and
  validation become still clearer?

### 12.8 Diagnostics, testing, and enforcement

- Which diagnostics are stable typed outcomes and which are edge-only
  rendering?
- How should focused tests align with the eventual component graph?
- Can architecture checks enforce semantic dependency direction and API
  visibility rather than filenames, regexes, and occurrence budgets?
- What compile-time or build-time negative tests would demonstrate that an
  engine cannot see raw target encodings and that a target cannot decide engine
  policy?
- Which extension tests can demonstrate genuine additive engine and target
  behavior?

### 12.9 Code deletion, sharing, and implementation size

- Which code volume represents necessary domain complexity, and which exists
  because authority, state, or variability is duplicated?
- For each proposed design change, which existing implementations, adapters,
  branches, fields, or compatibility paths will become deletable?
- Does the migration plan include the consumer convergence and deletion step,
  or does it merely introduce a new abstraction beside the legacy path?
- After every major boundary change, how do implementation size, dependency
  edges, broad-type consumers, switch concentration, and test coverage change?
- Does a clearer design make formerly indispensable validation, fallback, or
  mode-specific code collapse into shared declarative forms?
- Are there larger representational changes that would remove entire classes
  of code rather than merely redistribute them?
- Which recurring patterns should be consolidated into one honest mechanism,
  and how will the old copies be proven unused and removed?
- What temporary production growth has each migration introduced, when is it
  paid back, and what is the resulting net production-code reduction?

## 13. Operating invariants

The vertical-slice loop supplies the work plan; the next slice is selected from
current implementation evidence rather than frozen in advance. The following
invariants apply throughout execution:

1. All existing ConSan tests must continue to pass at periodic convergence
   points, with focused tests between full gates and serialized physical-GPU
   testing where required.
2. A discovered bug receives a regression test and an owner-level fix before
   the refactoring proceeds past the affected boundary.
3. Changes remain reviewable and bisectable, with frequent local commits and no
   pushes.
4. Behavior, coverage, target support, validation strength, and stable
   diagnostics are not traded away to simplify structure.
5. Temporary compatibility layers must have named ownership and an exit
   condition at a near-term convergence checkpoint; the refactoring must not
   leave parallel long-term authorities.
6. Measurements are evidence, not substitutes for reading the code or
   understanding semantic ownership.
7. The architecture may be revised as implementation evidence accumulates;
   previously written hypotheses have no authority over a demonstrably clearer
   design.
8. Concrete gfx-architecture support should move toward physically local,
   skippable packages, while genuinely shared family behavior remains in
   explicitly named family owners rather than being duplicated.
9. Mode-specific behavior should move toward physically local, skippable
   packages or consistently named facets, while genuine cross-mode mechanisms
   retain one shared implementation in explicitly named mechanism or
   exact-subset owners rather than being duplicated or hidden in deep
   common-code switches.
10. Every completed migration must remove the legacy production code it makes
    unnecessary; commented-out implementations, permanently dormant paths,
    and indefinite compatibility wrappers do not count as deletion.
11. Production implementation size, excluding tests, generated code,
    documentation, comments, and blank lines, must fall materially over the
    fifth refactoring. Local temporary growth requires an identified deletion
    payoff and recurring accounting.
12. Destination-design choices should consider deletion leverage explicitly:
    when two designs are otherwise sound, prefer the one that enables more
    duplicated or legacy machinery to be removed.

## 14. Evidence required before declaring completion

Completion is defined by observable architectural properties, not by executing
a predetermined list of stages. The exact classes, directories, interfaces,
and solver design may be discovered during the work, but the fifth refactoring
must not be declared complete until an independent post-refactoring deep read
demonstrates all of the following:

1. **The architecture is explainable and enforced.** The production component
   and build graph has a short directed explanation. Major dependencies cross
   declared interfaces, and boundary checks enforce the important forbidden
   directions rather than merely budgeting their present spellings.
2. **Architecture locality applies across the matrix.** Concrete and
   family-specific gfx implementation for all five supported targets is
   physically local and skippable. Common and mode code consume normalized
   facts, operations, and constraints rather than scattered target branches.
3. **Mode locality applies across all engines.** Record/Replay, Sampled,
   InlineShadow, and SuperCollider each have evident ownership across their
   relevant static and runtime facets. A reader can omit unwanted modes, while
   common and exact-subset mechanisms retain one implementation rather than
   mode-owned copies.
4. **The two extension axes are credible.** A concrete extension exercise and
   its enforcement show how an already-represented operation is supplied by a
   new target without editing mode implementations, and how a new mode composes
   existing target operations without editing concrete target packages. This
   may use focused test providers or another non-product extension fixture; it
   need not add a production mode or architecture merely for the audit.
5. **The main pipeline is forward-only at component boundaries.** Semantic
   decisions have named authorities and narrow products. Broad mutable buses
   are eliminated from cross-component APIs or confined as private transaction
   storage that components cannot opportunistically inspect and mutate.
6. **The surviving mode/target interaction is small and semantic.** Remaining
   interactions are explicit composition points with named owners and reasons,
   not placement, validation, analysis, or coordinator regions that jointly
   rediscover both axes.
7. **Legacy implementation has been harvested.** Superseded authorities,
   adapters, branches, state, and duplicated mechanisms are removed throughout
   the migrated production surface. There is no deferred cleanup ledger large
   enough to constitute a parallel architecture.
8. **Production code is materially smaller.** The agreed production-line
   target is met using the reproducible test/generated/comment/blank-excluded
   accounting, and the reduction comes from deleted or consolidated
   implementation rather than lost behavior.
9. **Behavioral evidence remains complete.** Focused component tests, the full
   nonphysical matrix over five targets, and serialized physical `gfx1201`
   tests pass with no unexplained loss of inventory. Every bug discovered
   during the work has a regression test at its owning boundary.

Passing the existing tests, compiling many translation units, moving files, or
making the boundary checker pass is necessary evidence but is not independently
sufficient. Nor is one successful architecture package or one successful mode
slice enough: the resulting model must have been exercised across the existing
production matrix so that apparent locality is not an untested exemplar.

The completion audit must use the same deep-read method as the initial audit:
trace definitions, inputs, outputs, mutation, callers, consumers, validation,
and runtime use. Token searches and counters may measure and enforce known
properties, but they may not infer semantic ownership or justify a surviving
interaction by themselves. If the audit still finds a major broad bus,
cross-axis implementation knot, parallel authority, or unharvested deletion
opportunity on the main production path, the goal remains active even if the
provisional workstreams are exhausted.

## 15. What remains deliberately open

The goal is precise; the destination design is not. Additional audit angles,
revised component hypotheses, alternative architectures, experiments, slice
ordering, and detailed workstream thresholds remain open to implementation
evidence. They belong in the execution record as they are discovered rather
than being guessed upfront.

That freedom cannot weaken the direction or completion bar. A newly discovered
design may change component boundaries and invalidate an earlier hypothesis,
but it must still improve the monotonic scorecard, converge all applicable
consumers, delete its abandoned and superseded machinery, and satisfy the same
Section 14 audit. Open design questions are permission to discover a better
route, not permission to postpone convergence or redefine completion around
the work already performed.

# Part III: execution record

## 16. Baseline and convergence ledger

### 16.1 Starting baseline

Implementation began from commit `496fff6c619f` with the exact production
scope used by the fourth-refactoring audit:

- `lib/rocjitsu/src/rocjitsu/code/patch/consan/` and
  `lib/rocjitsu/src/rocjitsu/hooks/consan/`;
- production `*.cpp`, `*.h`, and `*.inc` files only;
- tests, generated code, documentation, blank lines, and comment-only lines
  excluded from the implementation count.

The starting measurements are:

| Signal | Starting value |
| --- | ---: |
| Production files | 229 |
| Physical production lines | 104,975 |
| Nonblank production lines | 99,083 |
| Comment/blank-excluded production implementation lines | 91,450 |
| `MoiOptions` references / files | 87 / 25 |
| `ConSanTransformArtifacts` references / files | 276 / 57 |
| `ConSanPatchInfo` references / files | 200 / 28 |
| `ConSanMoiOperatingPoint` references / files | 290 / 51 |
| ConSan tests | 5,345 |
| Nonphysical tests | 4,710, including 2,908 simulated-device tests |
| Physical gfx1201 tests | 635 |

The initial gate passed all 4,710 nonphysical tests at `-j16` and all 635
physical gfx1201 tests at `-j1`. The test inventory exactly matches the end of
the fourth refactoring.

The line counter is lexical only: it removes comments and blank lines from the
declared production scope. It does not infer mode or architecture ownership
from identifiers. Semantic ownership and interaction accounting continue to
require the deep-read method described in Section 2.1.

### 16.2 Convergence checkpoint 1: immutable target-profile ownership

The first vertical slice traced the immutable `ConSanTargetProfile` product
from its central definition through every lookup, capability projection,
resource consumer, manifest generator, validator, and exact profile test. The
five concrete profiles had one authority, but that authority was a single
mixed table in the common capability header. It repeated gfx9-CDNA and RDNA
facts and made extension of one target require editing a large common record.

The converged ownership is now:

- gfx942 and gfx950 publish concrete profiles from gfx-named owners and share
  one gfx9-CDNA defaults owner;
- gfx1100 and gfx1201 publish concrete profiles from gfx-named owners and share
  one RDNA defaults owner;
- gfx1250 publishes its distinct profile and semantic-form mask from its own
  gfx-named owner; and
- `consan_capability_contract.h` owns only the narrow five-product registry and
  target-neutral lookup, validation, and capability projections.

The five superseded central profile records and their repeated assignments were
deleted. A structural regression rejects concrete target or architecture IDs
in the common capability contract, requires every reviewed family/concrete
profile owner to be imported exactly once, and verifies that every concrete
owner declares its target/architecture pair.

Checkpoint accounting relative to the starting baseline:

| Signal | Checkpoint 1 | Cumulative change |
| --- | ---: | ---: |
| Production files | 236 | +7 locality owners |
| Physical production lines | 104,875 | -100 |
| Nonblank production lines | 98,974 | -109 |
| Production implementation lines | 91,321 | **-129** |
| Concrete product/architecture IDs in the common capability contract | 0 | -10 |
| Broad-bus reference/file counts | unchanged | 0 |
| Test inventory | 5,345 | 0 |

Validation at the checkpoint includes a full `-j16` rebuild, all 15 exhaustive
capability-contract tests, the generated capability-manifest comparison, and
the architecture-boundary test. The post-change nonphysical gate also passed
all 4,710 tests at `-j16`; the serialized physical baseline remains green at
635 tests and will be repeated at a later periodic physical checkpoint.

### 16.3 Convergence checkpoint 2: immutable fault-selection input

Fault selection and exact synchronization-sequence proof were traced through
fault planning, mutation application, pristine-inventory rederivation, final
validation, composite retry, and diagnostic retention. The selectors only read
the immutable program inventory and exact fault-site product, but their public
contract accepted the complete mutable `ConSanTransformArtifacts` transaction.
The exact-sequence-member verifier had also duplicated the same ordering and
boundary proof for immutable inventory and analysis-time index lookups.

The converged path now publishes `ConSanFaultSelectionView`, containing only a
`ProgramInventory` reference and a read-only fault-site span. Planning,
composition, and independent validation construct that view at their boundary;
the selection component cannot inspect mutation state, resources, patches,
diagnostics, replacement bytes, or other transaction fields. Both immutable
inventory and construction-time indexes use one exact-member verifier with
different lookup functions, and the duplicated proof loop is deleted.

The boundary checker now rejects `ConSanTransformArtifacts` in both fault
selection and synchronization-event indexing and requires the narrow fault
selection contract to remain present.

| Signal | Checkpoint 2 | Cumulative change |
| --- | ---: | ---: |
| Production files | 236 | +7 |
| Physical production lines | 104,899 | -76 |
| Nonblank production lines | 98,994 | -89 |
| Production implementation lines | 91,335 | **-115** |
| `ConSanTransformArtifacts` references / files | 262 / 53 | **-14 / -4** |
| Other broad-bus reference/file counts | unchanged | 0 |
| Test inventory | 5,345 | 0 |

This slice spends 14 implementation lines on the explicit view and caller-side
construction while deleting the parallel exact-member proof body. It is a
completed boundary migration rather than a compatibility layer: no selector
overload accepting the broad transaction remains. Cumulative production size
continues downward, and the broad transaction loses four component consumers.

Validation includes a full `-j16` rebuild, the architecture-boundary test, and
134 focused fault, perturbation, exact-barrier, ordinary-acquire, LDS-address,
composition, and independent-final-validation tests.

### 16.4 Convergence checkpoint 3: fault planning as a typed product

The next trace followed fault planning separately from mutation emission.
Planning needs the immutable program inventory, eligible fault sites,
barrier-move destinations, the target architecture, and fault-policy options.
It produces exact typed mutation plans plus planning diagnostics and an
unsupported classification. It does not need replacement bytes, emitted
patches, resource plans, coverage state, operating points, committed lowering,
or the rest of the mutable transformation transaction. Nevertheless, its API
previously accepted and modified that entire transaction in place.

The converged path now has an explicit `consan_fault_planning.h` contract.
`ConSanFaultPlanningInput` contains the narrow selection view, a read-only
barrier-destination span, and only the prior unsupported fact needed to
preserve exactly-one diagnostic semantics. `ConSanFaultPlanningResult` owns
the selected plans, diagnostics, and unsupported result. The planner returns
that product without naming or mutating `ConSanTransformArtifacts`.

The composition layer is now the sole owner of publishing the planning product
into its transformation transaction. Both pristine composite planning and the
ordinary dry-run path cross that same adapter. The closely related runtime-
kernel predicate now consumes `ProgramInventory`, while the selection,
ownership, lifecycle, and participant-init helpers used inside fault mutation
consume their exact immutable products. Stateful byte emitters remain in the
fault-injection component and retain the mutable transaction deliberately;
they are the subsequent mutation transaction, not part of pure planning.

The architecture-boundary checker rejects `ConSanTransformArtifacts` in the
new planning contract and requires both the explicit input and result types.
No compatibility overload accepting the broad transaction remains.

| Signal | Checkpoint 3 | Cumulative change |
| --- | ---: | ---: |
| Production files | 237 | +8 |
| Physical production lines | 104,999 | +24 |
| Nonblank production lines | 99,083 | 0 |
| Production implementation lines | 91,409 | **-41** |
| `ConSanTransformArtifacts` references / files | 250 / 53 | **-26 / -4** |
| Other broad-bus reference/file counts | unchanged | 0 |
| Test inventory | 5,345 | 0 |

This boundary costs 74 implementation lines relative to checkpoint 2, mostly
for the named input/product contract and the single composition publisher. It
still leaves cumulative production implementation below the starting
baseline, removes twelve more broad-transaction references, and completes the
migration rather than installing a parallel path. The next convergence slice
must cash in a deletion or sharing opportunity: a second consecutive local
size increase would violate the forward-progress rule in Sections 8 and 10.

Validation includes a full `-j16` rebuild; the architecture-boundary test; 269
focused fault, barrier, atomic, ordinary-access, LDS, composition, and final-
validation tests; 14 focused MOI fault-composition and retry tests; and the
complete 4,710-test nonphysical gate over all five emulated targets at `-j16`
in 242.73 seconds. The serialized 635-test physical gfx1201 baseline remains
green and will be repeated at a later periodic physical checkpoint.

### 16.5 Convergence checkpoint 4: direct typed fault application

The planning product exposed one remaining compatibility layer. Composition
translated each retained `ConSanFaultMutationPlan` back into a mostly empty
`ConSanOptions`, and the mutation emitters then reinterpreted that legacy
request to recover the exact site, pair, sequence, destination, and mutation
payload that planning had already chosen. This was parallel authority across
the planning/application boundary: the application path could accidentally
reselect a different site or reconstruct a different policy decision.

All seven fault-application families now consume typed plans directly:
barrier drop, barrier move, barrier ID/scope, barrier participants, atomic,
LDS, and ordinary-memory mutation. Each emitter resolves the exact retained
identities against the same pristine inventory and fails closed if the plan is
stale. Barrier-move application additionally validates its retained logical
pair and structured-CFG proof instead of rerunning request-level admission.
The old plan-to-options construction helper and its per-family field-copying
blocks have been deleted. A structural regression requires every fault
emitter's public contract to name `ConSanFaultMutationPlan`.

Stateful emitters still receive `ConSanTransformArtifacts` deliberately: that
object owns the byte transaction, emitted patch telemetry, diagnostics, and
modified outcome. It no longer acts as the fault-selection or planning input.
The one non-plan policy needed by barrier-move byte growth is passed as the
narrow `ConSanPatchedImageGrowthLimit` value. This completes the boundary
migration rather than leaving typed and legacy application paths in parallel.

| Signal | Checkpoint 4 | Cumulative change |
| --- | ---: | ---: |
| Production files | 237 | +8 |
| Physical production lines | 104,938 | -37 |
| Nonblank production lines | 99,022 | -61 |
| Production implementation lines | 91,353 | **-97** |
| `ConSanTransformArtifacts` references / files | 251 / 53 | **-25 / -4** |
| `ConSanOptions` references / files | 86 / 36 | **-24 / 0** |
| Other recorded broad-bus reference/file counts | unchanged | 0 |
| Test inventory | 5,345 | 0 |

This slice pays back the preceding contract investment: it removes 56
implementation lines relative to checkpoint 3 and restores a larger
cumulative reduction than checkpoint 2, while retaining the explicit planning
product. The extra broad-transaction reference relative to checkpoint 3 is
the temporary compatibility overload in the shared growth-policy helper; the
next trace of that policy boundary must either remove the overload or justify
why its callers cannot consume the narrow limit directly.

Validation includes a full `-j16` rebuild, 14 focused barrier-move and typed-
plan contract tests, the architecture-boundary test, and the complete 4,710-
test nonphysical gate over all five emulated targets at `-j16` in 244.53
seconds. The test inventory is unchanged. The serialized 635-test physical
gfx1201 baseline remains green and will be repeated at a later periodic
physical checkpoint.

### 16.6 Convergence checkpoint 5: target-normalized analysis and mode-owned MOI demand

This checkpoint combines two related boundary migrations completed through
commit `6d94968f72`. First, program analysis no longer spreads raw target-family
instruction layouts through common inventory code. Pointer provenance, atomic
inventory, FLAT and GLOBAL decoding, gfx12-family decoding, and pre-gfx12
decoding now enter analysis through target-normalized products. Shared family
decoding is implemented once; genuinely distinct target behavior remains in
the relevant target owner.

Second, the MOI coordinator no longer selects behavior by mode. Record/Replay,
Sampled, and InlineShadow register one operations product containing their
object-demand planner, lowering sequence, access-scratch demand, and
persistent-state demand. Common orchestration and placement consume the typed
results. The following policy moved out of the common coordinator or placement
monolith and into mode owners:

- automatic owner-source defaults and post-inventory admission changes;
- access/synchronization lowering order and mode-local cleanup;
- access scratch demand, Sampled spill eligibility, and InlineShadow scratch
  contracts;
- prologue publication and dynamic-stack entry-reservation policy;
- Record/Replay assignment validation and compact-barrier preference;
- InlineShadow dynamic-stack dispatch and dynamic-LDS private-state choices;
  and
- persistent owner, workgroup-key, exact-workgroup, and dispatch-capture
  demand.

The migration exposed a previously hidden two-mode rule: a helper named as
Record/Replay policy required exact entry-captured workgroup identity for both
Record/Replay and Sampled. The new boundary represents that rule once as a
shared Record/Replay-plus-Sampled demand helper. The modes then independently
add their own persistence semantics. This is the intended form of mode
locality: mode-owned policy without duplicating genuinely shared behavior.

The coordinator is now an actual common pipeline. Its physical size fell from
672 to 396 lines, and its explicit `ConSanMoiEngine::{...}` references fell
from 20 to zero. The still-large common placement body fell from 6,694 to
6,385 physical lines and from 95 to 69 explicit mode-enum references. Those
remaining 69 references are the principal evidence that mode locality is not
finished; many are architecture-sensitive representation fallbacks inside
persistent and transient placement.

Checkpoint accounting relative to the starting baseline is deliberately
mixed:

| Signal | Checkpoint 5 | Cumulative change |
| --- | ---: | ---: |
| Production files | 248 | +19 |
| Physical production lines | 105,361 | +386 |
| Nonblank production lines | 99,296 | +213 |
| Production implementation lines | 91,678 | **+228** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 258 / 58 | -18 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 308 / 56 | +18 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | 69 | -26 |
| Test inventory | 5,352 | +7 |

This is real component convergence but not yet code-size convergence. New
typed products and direct tests account for part of the growth, but the fifth
refactoring contract requires the resulting deletion opportunities to be
reaped. The larger file count and wider operating-point surface are also not
end-state virtues. The next iterations must use the new mode and target
boundaries to delete more common branching, duplicated scans, stale imports,
and superseded broad contracts than they add. A future checkpoint must restore
the cumulative implementation count below 91,450 rather than treating
locality alone as sufficient progress.

Validation for this checkpoint includes repeated 864/865-test focused MOI and
architecture-boundary gates, the complete 4,717-test nonphysical matrix over
all five simulated targets at `-j16` in 244.70 seconds, and all 635 serialized
physical gfx1201 tests at `-j1` in 109.49 seconds. All are green. The seven new
tests directly pin mode-owned object planning, prologue policy, persistent
state demand, and the shared Record/Replay-plus-Sampled identity rule.

### 16.7 Convergence checkpoint 6: persistent representation ownership and legacy harvesting

This checkpoint, through commit `60bcbe505a`, follows the mode-demand boundary
into the storage-representation decisions that common placement still made on
behalf of each engine. `MoiPersistentStateDemand` now states whether its mode
supports scalar and private persistent state, when capacity/private fallback
requires scalar state, and whether descriptor growth should prefer private
epoch storage. Record/Replay, Sampled, and InlineShadow populate those facts in
their own translation units. The common solver now chooses a legal location
for the requested representation without enumerating modes in that region.

Two scalar helpers that were already semantically and nominally mode-local
have moved from common placement into their actual owners: InlineShadow owns
visible-evidence scalar selection, and Sampled owns its stable access-return
SCC selection. No duplicate compatibility definitions remain.

The now-explicit boundaries exposed several small but concrete legacy layers,
which this checkpoint harvests rather than merely recording as future work:

- transient and persistent component assignments use their types' structural
  equality instead of duplicated hand-maintained field inventories;
- private assignment lookup functions and their public one-line forwarding
  wrappers have collapsed into one canonical lookup per assignment kind;
- host-hook diagnostics call the canonical core naming functions directly
  instead of six local forwarding aliases; and
- fault planning calls the canonical target-operation classification and
  capability functions directly instead of four compatibility aliases.

These deletions are individually modest, but they are directionally useful:
the canonical structural comparisons automatically cover future fields, and
the canonical naming/target APIs eliminate parallel authorities. Common
placement fell from 6,385 to 6,317 physical lines and from 69 to 52 explicit
mode-enum references.

| Signal | Checkpoint 6 | Cumulative change |
| --- | ---: | ---: |
| Production files | 248 | +19 |
| Physical production lines | 105,299 | +324 |
| Nonblank production lines | 99,246 | +163 |
| Production implementation lines | 91,628 | **+178** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 258 / 58 | -18 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 306 / 56 | +16 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | 52 | -43 |
| Test inventory | 5,352 | +7 |

This is the first checkpoint after the mode-boundary investment to move both
locality and size in the desired direction: 50 implementation lines and 62
physical lines have been removed since checkpoint 5. It is not completion.
The implementation remains 178 lines above the starting baseline, common
placement still contains 52 mode checks, and the mode/target extension
exercises and broader build-graph enforcement required by Section 14 remain
open. The next slices must continue harvesting superseded paths and move
cohesive remaining policy out of common placement; reaching the old baseline
is a near-term threshold, not the final material-shrinkage target.

Validation includes repeated focused MOI, assignment, scalar-layout, fault,
and host-hook gates; the complete 4,717-test nonphysical matrix over all five
simulated targets at `-j16` in 247.06 seconds; and all 635 serialized physical
gfx1201 tests at `-j1` in 109.23 seconds. All are green.

### 16.8 Convergence checkpoint 7: shared mechanisms and report composition

This checkpoint, through commit `0c12da3738`, follows the new ownership
boundaries into concrete legacy and duplication harvesting. It contains seven
production slices rather than one new abstraction campaign:

- a production atomic-address planner adapter used only by focused tests moved
  into test support, and a production fault-application test adapter moved into
  its sole test translation unit;
- Sampled and InlineShadow nested resource fallbacks now use one named trace
  adoption mechanism instead of composing equivalent diagnostic trails twice;
- SuperCollider and MOI host hooks share one bounded report-registry lifecycle
  and one fine-grained-preferred HSA region selector while retaining separate
  allocation policies, evidence formats, budgets, decoding, and diagnostics;
- three generated/direct/displaced relay-reservoir payload validators collapsed
  into one proof mechanism while their distinct geometry proofs remain
  independent;
- Record/Replay and Sampled direct-call and inline-island layouts share one
  runtime workgroup-selection predicate while retaining their distinct
  call/return geometry;
- report layout finalization now marks fields written by the selected mode
  planner and aliases every untouched region generically, deleting the common
  mode-by-region cleanup matrix; and
- Record/Replay, Sampled, and InlineShadow evidence planners share one runtime
  report-contract publication authority after their mode-owned sizing logic.

The report-layout boundary check was tightened from 18 to 15 explicit
mode-enum references so the deleted cleanup matrix cannot silently return.
The remaining 15 references are the reviewed mode planner selection,
mode-specific well-formedness, capacity fitting, and independent layout
revalidation paths; their continued presence means report mode locality is not
yet complete.

A fresh exact occurrence recount found two transcription errors in the
checkpoint 5 and 6 tables: the actual declared production-scope counts were
258 rather than 256 `ConSanTransformArtifacts` references at checkpoint 5,
and 308/306 rather than 297/296 `ConSanMoiOperatingPoint` references at
checkpoints 5/6. The tables above are corrected. No production change in this
checkpoint created the apparent ten-reference increase.

| Signal | Checkpoint 7 | Cumulative change |
| --- | ---: | ---: |
| Production files | 249 | +20 |
| Physical production lines | 105,031 | +56 |
| Nonblank production lines | 98,974 | -109 |
| Production implementation lines | 91,351 | **-99** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 256 / 58 | -20 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 306 / 56 | +16 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | 52 | -43 |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | 15 | -3 from checkpoint 6 |
| Test inventory | 5,352 | +7 |

This checkpoint removes 277 implementation lines since checkpoint 6 and puts
the cumulative implementation count 99 lines below the starting baseline. It
therefore pays back the typed mode-boundary investment, but it does not yet
satisfy the material-shrinkage or architecture criteria. Physical line count
remains above baseline, the operating-point surface remains wider than at the
start, common placement still contains 52 explicit mode checks, and the
component build graph, full architecture locality, and both extension
exercises required by Section 14 remain open. Crossing the old implementation
baseline is a convergence threshold, not a completion condition.

Validation includes repeated focused resource, relay, report lifecycle,
runtime-gate, evidence-contract, auto-report-layout, and architecture-boundary
gates; the complete 4,717-test nonphysical matrix over all five simulated
targets at `-j16` in 246.20 seconds; and all 635 serialized physical gfx1201
tests at `-j1` in 109.29 seconds. All are green and the test inventory is
unchanged from checkpoint 6.

### 16.9 Convergence checkpoint 8: mode-owned spill policy

This checkpoint, through commit `760a32a4a5`, follows the mode boundary into
the shared resource solver's spill fallbacks:

- Record/Replay, Sampled, and InlineShadow now own their dynamic-stack spill
  policy. The common solver consumes target-normalized backend availability
  and the selected mode's mixed-owner requirement without naming InlineShadow
  as an exception;
- the three distinct guest-operand-overlap rules now produce one narrow policy
  containing retry permission and any guest result range that must remain
  disjoint. Common placement performs the allocation retry without decoding
  mode semantics;
- Sampled and InlineShadow now own the eligibility and scratch shape of their
  spill-backed access fallbacks. The common access-plan loop only invokes the
  selected optional planner and adopts the returned fallback through the
  existing shared trace mechanism;
- the primary and overlap-spill allocation paths share one persistent-VGPR
  exclusion mechanism, deleting two copies of owner, epoch, dispatch, and
  workgroup-tuple traversal; and
- the dynamic-stack policy proved to be declarative after extraction, so its
  three temporary callbacks were collapsed into two traits in each mode-owned
  operations table rather than retained as abstraction ceremony.

The architecture-boundary gate now fixes the reviewed
`consan_moi_placement.inc` budget at 46 explicit mode-enum references, down
from 52 at checkpoint 7 and from 95 at the starting review. The file also fell
from 6,288 to 6,221 physical lines in this checkpoint. These are real locality
and knot-size improvements: the access spill loop no longer contains an
explicit mode branch, and a future mode declares its policies beside its other
mode operations.

| Signal | Checkpoint 8 | Cumulative change |
| --- | ---: | ---: |
| Production files | 249 | +20 |
| Physical production lines | 105,112 | +137 |
| Nonblank production lines | 99,041 | -42 |
| Production implementation lines | 91,416 | **-34** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 256 / 58 | -20 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 310 / 56 | +20 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | 46 | -49 |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | 15 | -3 |
| Test inventory | 5,354 | +9 |

The typed policy products and two new mode-contract regressions cost 65
implementation lines relative to checkpoint 7. The persistent-register
consolidation and declarative-trait cleanup reap part of that cost immediately,
but the result remains only 34 implementation lines below the starting
baseline. This is therefore a structural convergence checkpoint, not the
material shrinkage required for completion. The remaining 46 placement mode
references, architecture locality, build-graph simplification, wide operating
point, both extension exercises, and the Section 14 completion evidence all
remain open.

Validation includes repeated 95-test five-target spill/pressure gates and
173-test dynamic-stack/mode-planning gates; the complete 4,719-test
nonphysical matrix over all five simulated targets at `-j16` in 245.35
seconds; and all 635 serialized physical gfx1201 tests at `-j1` in 109.79
seconds. All are green. The two new tests directly pin per-mode dynamic-stack
and operand-overlap policy ownership.

### 16.10 Convergence checkpoint 9: mode-owned dispatch identity

This checkpoint, through commit `134bbecda5`, follows the mode boundary into
dispatch-identity demand and fallback selection, then harvests repeated
register-shape knowledge exposed by that work:

- Record/Replay, Sampled, and InlineShadow now declare whether their semantic
  consumers require dispatch identity, which lossless fallback they permit,
  and whether fallback replanning replaces the complete scalar ABI or only the
  dispatch pair. Common placement receives normalized target and site facts
  and retains one register-search and retry mechanism;
- a new mode-contract test fixes the three distinct policies without teaching
  the common solver how to recognize an engine;
- the exact Record/Replay workgroup-register tuple publishes one iterable
  representation used by nine placement, overlap, sizing, and validation
  consumers instead of nine hand-unpacked copies;
- the accepted operating point similarly publishes one fixed-width auxiliary
  SGPR traversal. Conflict detection, architectural validation, and descriptor
  sizing no longer carry three independent lists of the Inline and
  Record/Replay spill ABI fields; and
- the fallback path no longer recomputes target and site-demand facts that it
  does not consume.

The reviewed `consan_moi_placement.inc` budget is now 38 explicit mode-enum
references, down from 46 at checkpoint 8 and 95 at the starting review. The
file fell by another 78 physical lines, from 6,221 to 6,143, despite adding the
shared dispatch-policy consumer. This is the intended one-way movement: mode
meaning leaves the common solver, the shared mechanism remains singular, and
representation-owned traversal deletes copies rather than hiding them behind
mode wrappers.

| Signal | Checkpoint 9 | Cumulative change |
| --- | ---: | ---: |
| Production files | 249 | +20 |
| Physical production lines | 105,119 | +144 |
| Nonblank production lines | 99,040 | -43 |
| Production implementation lines | 91,407 | **-43** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 256 / 58 | -20 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 310 / 56 | +20 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | 38 | -57 |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | 15 | -3 |
| Test inventory | 5,355 | +10 |

The dispatch contract temporarily adds more policy structure than its deleted
branches alone repay. The two representation consolidations and removal of
the redundant demand scan nevertheless leave this checkpoint nine
implementation lines smaller than checkpoint 8 and 43 below baseline. That
is genuine local convergence, but still not the material production shrinkage
required by Section 14. The remaining placement interactions, wide operating
point, target locality, component build graph, two extension exercises, and
independent final deep audit all remain open; this checkpoint does not satisfy
the completion bar.

Validation includes 130-test five-target dispatch/fallback gates and a
421-test scalar-spill, dynamic-stack, descriptor, and architectural-validation
gate; the complete 4,720-test nonphysical matrix, including all 2,908 simulated
device tests over five targets, at `-j16` in 246.98 seconds; and all 635
serialized physical gfx1201 tests at `-j1` in 108.28 seconds. All are green.

### 16.11 Convergence checkpoint 10: scalar ABI ownership and shared register search

This checkpoint, through commit `f989245d23`, follows the dispatch-policy
boundary into scalar ABI selection and then harvests invalid state and repeated
search mechanics exposed by that migration:

- Record/Replay, Sampled, and InlineShadow now own their complete scalar ABI
  layout. The mode operations publish special-state and indirect-jump
  registers, while one common builder retains the spill-aware mechanics;
- the former placement-owned `moi_special_state_sgprs` and
  `moi_indirect_jump_sgprs` implementations moved behind that mode boundary,
  removing another five explicit mode references from common placement;
- the operating point's two independent scalar-spill booleans, which could
  represent the impossible state that both incompatible layouts were active,
  collapsed into one exclusive `None`, `Inline`, or `Compact` layout. Common,
  Inline, and Record/Replay-plus-Sampled consumers now ask the representation
  the exact question they need; and
- eighteen deterministic forward first-fit register loops now use one shared
  traversal. Each caller still owns its range width, floor, alignment, and
  admission predicate, but candidate order and termination are implemented
  once. Reverse searches, candidate-list construction, and the stateful
  multi-result Inline ABI search remain separate because they are not the same
  mechanism.

The scalar ABI migration exposed and fixed a real fallback bug before commit:
the first common builder incorrectly required a global EXEC-save base before
honoring an explicitly selected spill-backed indirect PC/SCC pair. Four
gfx1250 Inline dense-router tests failed deterministically. The builder now
honors the explicit spill ABI first, and the owner-level mode-planning test
directly covers a spill-backed layout with no global EXEC-save base.

The reviewed placement body is now 6,060 physical lines with 33 explicit
mode-enum references, down from 6,143 and 38 at checkpoint 9 and from 6,694 and
95 at the starting review. The exclusive spill representation also reduces
the broad operating-point occurrence count despite replacing many raw field
reads with typed queries.

| Signal | Checkpoint 10 | Cumulative change |
| --- | ---: | ---: |
| Production files | 249 | +20 |
| Physical production lines | 105,139 | +164 |
| Nonblank production lines | 99,050 | -33 |
| Production implementation lines | 91,416 | **-34** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 256 / 58 | -20 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 307 / 56 | +17 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | 33 | -62 |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | 15 | -3 |
| Test inventory | 5,356 | +11 |

The mode-owned ABI contract temporarily grew production more than its first
five deleted placement branches repaid. Making the spill layout exclusive and
sharing the register traversal then removed that invalid state and 34
implementation lines. The cumulative result is nine implementation lines
larger than checkpoint 9 and still only 34 below the starting baseline. This
is therefore another structural checkpoint, not the material reduction
required by Section 14. The remaining placement interactions, wide operating
point, target locality, component build graph, extension exercises, and
independent completion audit remain open.

Validation includes a 1,034-test spill, dense-routing, barrier, publication,
and indirect-state gate plus the previously overloaded gfx1250 publication
case serialized separately; the complete 4,721-test nonphysical matrix over
all five simulated targets at `-j16` in 248.12 seconds; and all 635 physical
gfx1201 tests at `-j1` in 108.06 seconds. All are green. The inventory increase
is the direct scalar-ABI fallback regression.

### 16.12 Convergence checkpoint 11: mode-free shared placement

This checkpoint, through commit `42be3760b0`, completes the migration of mode
selection out of the common placement solver. It follows transient scalar
allocation, synchronization-island reservation, persistent-state overflow,
and dispatch-identity exhaustion to their semantic owners, then consumes the
resulting products without rediscovering the selected engine:

- each mode now declares its transient scalar representation and the exact
  constraints of that representation. Common placement still owns one
  register search, liveness proof, spill-window search, and assignment path;
- synchronization-island reservation consumes admitted barrier decisions,
  dense-router state, and typed fence intents directly. The old engine tests
  duplicated facts already present in those products and were deleted;
- the three CDNA persistent overflow representations are explicit mode-owned
  choices: exact-workgroup state, owner snapshot, and resident-wave private
  state. The CDNA allocator implements those named representations without
  asking which mode selected one; and
- dispatch-identity exhaustion consumes the mode-owned fallback kind and
  transient representation. The last two explicit engine tests in placement
  and redundant Inline fallback initializers were removed.

This is a completed boundary migration, not a parallel interface. There is no
engine-aware compatibility path in the shared placement body. The structural
gate now fixes `consan_moi_placement.inc` at **zero** explicit mode-enum
references, down from 33 at checkpoint 10 and 95 at the starting review. A new
mode must publish the narrow planning products beside its implementation; it
cannot add an engine switch to the common solver without failing the build.

| Signal | Checkpoint 11 | Cumulative change |
| --- | ---: | ---: |
| Production files | 249 | +20 |
| Physical production lines | 105,170 | +195 |
| Nonblank production lines | 99,080 | -3 |
| Production implementation lines | 91,441 | **-9** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 256 / 58 | -20 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 56 | +26 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | 15 | -3 |
| Test inventory | 5,356 | +11 |

The transient and persistent representation contracts add 25 implementation
lines relative to checkpoint 10 after the synchronization and dispatch
deletions are harvested. Production remains nine implementation lines below
the starting baseline, so the checkpoint has not met the material-shrinkage
property in Section 14. The next slices must exploit the now-explicit
representations to consolidate or delete mechanics; merely adding more policy
fields would reverse the required flow. The wide operating point, target
locality beyond the existing profile packages, component build graph, two
extension exercises, and independent completion audit also remain open.

Validation includes a 250-test scalar-spill/dense-routing gate, a 181-test
CDNA persistent-state and pressure gate, and a 125-test dispatch-identity and
full-pressure gate. The complete 4,721-test nonphysical matrix over all five
simulated targets passed at `-j16` in 244.18 seconds, and all 635 serialized
physical gfx1201 tests passed at `-j1` in 109.51 seconds. The test inventory is
unchanged at this checkpoint.

### 16.13 Convergence checkpoint 12: target-normalized analysis and mode-owned report ABI

This checkpoint, through commit `c8450bdeda`, advances two independent
locality fronts and harvests duplication exposed by both.

On the architecture front, raw decoder and synchronization code no longer
select target behavior by repeating concrete architecture tests. Atomic target
selection, synchronization selection, preservation of normalized scalar
addresses, and atomic-address capabilities now flow through target-owned
operations or target profiles. Program analysis has one narrow five-entry
target-operations registry, while the redundant per-decoder target declaration
surface has been deleted. Concrete architecture constants are confined to the
five target-profile implementations, the gfx1250 selectable-bank owner, and
that registry. Structural checks prohibit their reintroduction into the common
access, atomic, synchronization, MOI address, fault, and placement paths.

This boundary also exposed two shareable mechanisms. Exact-workgroup state
commit now has one implementation rather than two mode-shaped copies, and the
general spill manager owns dynamic-stack scratch transfers that had been
duplicated in ConSan support code. Target-specific examples were moved out of
common implementation comments and into target-neutral contract tests, so a
reader can traverse the shared mechanics without encountering accidental gfx
recipes.

On the mode front, evidence planning now receives a narrow immutable view
instead of the whole mutable transformation transaction. One shared component
resolves kernel/function ownership, descriptor fallback, unique sites, and the
rocclr exclusion for barrier, fence, and atomic evidence planning. Those
planners can no longer acquire direct code-object lookup dependencies without
failing the structural gate.

Report ABI construction, capacity fitting, and inverse inventory
reconstruction are now explicit facets of each mode's existing operations
product. Record/Replay, Sampled, and InlineShadow own those facets in named
mode-local files; the common report planner composes the selected operations
and retains shared semantic evidence classification and validation. This is
not a second mode dispatch mechanism. After the split exposed three copies of
capacity conversion and byte-region layout, one typed region planner replaced
them. A shared power-of-two capacity predicate likewise replaced the repeated
four-dimensional Record/Replay ABI checks. The common report planner now has
six explicit mode references, down from 15 at checkpoint 11 and 18 at the
starting review; the six remaining references belong to evidence-requirement
construction and validation, not report layout.

| Signal | Checkpoint 12 | Cumulative change |
| --- | ---: | ---: |
| Production files | 255 | +26 |
| Physical production lines | 105,233 | +258 |
| Nonblank production lines | 99,078 | -5 |
| Production implementation lines | 91,417 | **-33** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 245 / 58 | -31 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 315 / 56 | +25 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | 6 | **-12** |
| Test inventory | 5,357 | +12 |

The mode-local report split was an intentional 62-implementation-line locality
investment. Consolidating the newly visible common region mechanics harvested
25 lines in the immediately following slice. Together with the target and
evidence work, production is now 33 implementation lines below the starting
baseline. This is real forward movement, but it is still not the material
shrinkage required by Section 14. In particular, the wide operating point,
core build graph, remaining report evidence policy, target locality of emitted
ISA and independent validation, both extension exercises, and independent
completion audit remain open.

Validation includes focused target-normalization, evidence-planning,
report-ABI, capacity, spill, and structural-boundary gates. The complete
4,722-test nonphysical matrix over all five simulated targets passed at `-j16`,
and all 635 serialized physical gfx1201 tests passed at `-j1` in 109.30
seconds. The inventory increase is the target-capability contract regression.

### 16.14 Convergence checkpoint 13: one mode-owned evidence path

This checkpoint, through commit `6566586342`, completes the report/evidence
mode-locality slice begun at checkpoint 12. The common report planner now owns
only engine-neutral intent classification, report-region composition, and
canonical layout validation. Record/Replay, Sampled, InlineShadow, and
SuperCollider each own their evidence-requirement construction and validation
beside the corresponding report implementation.

The three report-backed engines derive from one shared MOI evidence contract
for the common runtime requirements, sizing inventory, ABI plan, and typed
construction result. This deletes the repeated four-field payload without
forcing mode-specific invariants into the base. Shared intent validation,
count accumulation, publication requirements, and report-region mechanics
remain single common implementations used by the mode owners.

The transform pipeline no longer switches over MOI engines to translate
request bounds into three separate capacity-policy APIs. Evidence planning is
a facet of the existing `MoiModeOperations` registry, fed by one narrow
immutable context. Record/Replay owns the dynamic-ring ceiling exception,
Sampled owns its bank policy, and InlineShadow owns the program-inventory/LDS
aperture join. The former per-mode production planner functions and three
capacity-policy types were then deleted rather than retained as compatibility
facades. Focused tests use test-only typed adapters; production has one path.

Structural checks fix both `consan_moi_report_plan.cpp` and
`consan_pipeline.cpp` at zero explicit MOI engine references, prohibit direct
per-mode evidence-planner calls in the pipeline, and reject resurrection of
the retired production planner names anywhere in the ConSan implementation.
A new mode adds its report/evidence owner and one operations-registry entry; it
does not add another branch to shared report composition or the transform
pipeline.

| Signal | Checkpoint 13 | Cumulative change |
| --- | ---: | ---: |
| Production files | 256 | +27 |
| Physical production lines | 105,191 | +216 |
| Nonblank production lines | 99,031 | **-52** |
| Production implementation lines | 91,407 | **-43** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 245 / 58 | -31 / +1 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 315 / 56 | +25 / +5 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a |
| Test inventory | 5,357 | +12 |

The initial file localization and common contract cost five implementation
lines, and registering evidence planning cost another 35. The immediately
following convergence commit removed the parallel APIs and 50 implementation
lines, paying back all 40 temporary lines plus ten more. Production is now 43
implementation lines below the starting baseline. This is a completed
boundary migration with measurable deletion, but it still does not constitute
the material code-size reduction or overall completion evidence required by
Section 14. The wide operating point, core build graph, target locality of
emitted ISA and independent validation, both extension exercises, and final
independent audit remain open.

Validation includes the 106-test evidence-requirement, report-ABI,
mode-planning, pipeline, and structural-boundary gate, followed by a broader
931-test MOI/evidence/pipeline nonphysical gate at `-j16` in 43.70 seconds.
Both are green. The immediately preceding checkpoint supplied the complete
4,722-test nonphysical and 635-test physical gates, so the serialized physical
matrix was not repeated for this target-neutral ownership-only slice.

### 16.15 Convergence checkpoint 14: target-normalized proof and validation inventory

This checkpoint, through commit `83dbe7622b`, follows the architecture and
mode boundaries into independent encoded-mutation proof, report dispatch
identity, descriptor allocation, and pristine validation inventory.

On the target side, the common final validator no longer includes RDNA4
machine structures to prove ordinary-global, atomic-address, or atomic-scope
mutations. One typed `ConSanEncodedMutationKind` operation crosses the target
boundary, and the gfx12 owner implements the exact raw-word and generated-type
proof. Four former per-mutation facade functions, their internal declarations,
and their registry wrappers collapsed into that operation. Address, scope, and
DS-address comparisons share one set of exact proof primitives. The obsolete
internal validation adapter remains only as the required inert filename
tombstone, and structural checks prohibit the former facade names from
returning.

InlineShadow atomic admission likewise consumes target-normalized constraints
instead of testing concrete target facts in the mode implementation. Descriptor
placement consumes one `ConSanDescriptorVgprAllocation` product, so common
placement no longer decodes the gfx9 `ACCUM_OFFSET` field or reconstructs the
ordinary/accumulator boundary. These migrations retain one shared mechanism
without copying target recipes into modes.

On the mode/shared-mechanism side, a report dispatch identity is now one
inseparable authorized source: an SGPR pair, a VGPR pair, or a 64-bit literal.
Low-level emitters cannot independently reapply literal policy to its two
words. Target-owned and externally bound source planners preserve the distinct
authorities, while the shared runtime workgroup gate consumes its exact
Record/Replay-versus-Sampled flavor rather than the global engine enum.

The final slice traces pristine mutation and perturbation rederivation. The old
APIs returned or embedded the complete mutable `ConSanTransformArtifacts` bus,
although mutation proof consumed only program inventory and fault sites, and
perturbation proof consumed only program inventory, candidates, and analysis
success. A dedicated validation-inventory owner now publishes two exact
immutable products and stops both lowerer calls immediately after semantic
inventory, before unused fault or perturbation planning.

The products deliberately remain distinct. A first attempted merge was
falsified by the existing bounded-Qwen regression: mutation proof intentionally
enables extended barrier pairing, whereas perturbation proof intentionally
rederives ordinary synchronization semantics. The converged contract preserves
that independent distinction rather than sharing analyses whose semantics are
only superficially similar. The boundary gate rejects the mutable transaction
and perturbation workspace in the public inventory contract, and requires both
rederivations to stop at their reviewed semantic-inventory extent.

| Signal | Checkpoint 14 | Cumulative change |
| --- | ---: | ---: |
| Production files | 261 | +32 |
| Physical production lines | 105,360 | +385 |
| Nonblank production lines | 99,157 | +74 |
| Production implementation lines | 91,506 | **+56** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 243 / 58 | **-33 / +1** |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 305 / 53 | +15 / +2 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a |
| Test inventory | 5,358 | +13 |

The typed target-validation extraction and the subsequent sharing/deletion
paybacks removed 59 physical production lines after the initial locality
investment. Narrow pristine validation removes four more broad-transaction
references and avoids unused planning, but its dedicated owner and exact
products cost 18 implementation lines. Across the complete checkpoint,
production is 99 implementation lines larger than checkpoint 13 and 56 lines
above the fifth-refactoring baseline. This is durable locality and narrower
dataflow, not the material shrinkage required by Section 14.

The next convergence checkpoint must therefore harvest more implementation
than this checkpoint added. It may exploit the new products and target
operations, simplify the remaining broad transaction and operating point, or
remove another repeated authority, but it must not make a second consecutive
locality investment that defers the same size payoff. The core build graph,
full physical locality of all target facets, broad transaction/operating-point
surfaces, both extension exercises, and independent completion audit remain
open.

Validation includes repeated focused dispatch-identity, descriptor, target
mutation, fault, perturbation, composite, and final-validation matrices; the
124-test fault/perturbation/final-validation gate; and the structural boundary
gate. The complete 4,723-test nonphysical matrix over all five simulated
targets passed at `-j16` in 245.85 seconds. All 635 serialized physical
gfx1201 tests passed at `-j1` in 110.16 seconds. The inventory increase is the
dispatch-source planning regression.

### 16.16 Convergence checkpoint 15: shared final-validation environment

This checkpoint, through commit `c7ae9770ed`, harvests the deletion required
by checkpoint 14 from the final semantic-proof path. A deep trace of every
final validator found fourteen independent passes repeatedly reconstructing
the same one-text-section views, target profile, architecture, decoder, and
bounded word comparisons. The repetition was prerequisite plumbing rather
than pass-specific proof semantics.

`FinalValidationEnvironment` now owns those immutable universal
prerequisites exactly once. Its `ValidationText` views provide the common
bounded word and encoded-word-span operations. Every proof pass still owns
its own applicability test, semantic walk, fail-closed decisions, and exact
diagnostic text; no construction policy, mutable transaction state, or
pass-specific derived analysis entered the shared environment. The one decoder
is safe to reuse because these validation calls do not activate its instruction
pool and decoding does not retain cross-call state.

The passes also append directly to one ordered validation-error sink instead
of allocating and moving fourteen temporary vectors. Two validators use their
error count as local fail-fast state; they now compare against the count on
entry so an error from an earlier independent proof cannot suppress their own
work. Thus the consolidation preserves both error ordering and the former
independence of the passes. A boundary gate requires the immutable text and
environment owners and exactly one decoder construction and one environment
instance in final validation.

| Signal | Checkpoint 15 | Cumulative change |
| --- | ---: | ---: |
| Production files | 261 | +32 |
| Physical production lines | 105,256 | +281 |
| Nonblank production lines | 99,045 | **-38** |
| Production implementation lines | 91,390 | **-60** |
| `MoiOptions` references / files | 95 / 30 | +8 / +5 |
| `ConSanTransformArtifacts` references / files | 243 / 58 | **-33 / +1** |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 305 / 53 | +15 / +2 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a |
| Test inventory | 5,358 | +13 |

The slice removes 104 physical, 112 nonblank, and 116 implementation lines
relative to checkpoint 14. It therefore pays back all 99 implementation lines
added by that checkpoint plus 17 more, and places production 60 implementation
lines below the fifth-refactoring baseline. This satisfies the immediate
deletion contract without claiming the material overall shrinkage or the
independent completion evidence required by Section 14. The core build graph,
remaining broad transaction and operating-point surfaces, full target-facet
locality, both extension exercises, and final independent audit remain open.

Validation includes the focused 124-test fault, perturbation, composite,
final-validation, and structural-boundary gate. The complete 4,723-test
nonphysical matrix over all five simulated targets passed at `-j16` in 241.80
seconds. All 635 serialized physical gfx1201 tests passed at `-j1` in 109.30
seconds. The test inventory is unchanged.

### 16.17 Convergence checkpoint 16: directed core build graph

This checkpoint, through commit `ad34057504`, turns the de-facto core
components into an enforced production build graph. The prerequisite slice
first removed the inert lowering forwarding facade and then broke the one real
reverse edge between independent validation and top-level composition.
Pristine mutation and perturbation proof formerly called the complete composer
only to stop after program inventory, while composition called final
validation. Both now invoke the shared program-analysis authority directly.
The two validation products and their deliberately different barrier semantics
remain unchanged; no second inventory implementation was introduced.

ConSan production now builds as six object-library components with one short
direction:

`contracts -> targets -> analysis -> transform -> validation -> orchestration`

Validation also names its intentional direct analysis dependency. Contracts
own target-neutral types and policy; targets own concrete normalization and
machine operations; analysis owns program, synchronization, and fault
inventory; transform owns shared mechanics, modes, mutation, and resource
work; validation independently proves the proposed result; orchestration owns
the public pipeline and composition. This is an honest coarse graph rather
than a claim that every large component is already at its final internal cut.
In particular, mutation and mode mechanics remain together because their
current calls are not yet acyclic enough to justify a fictitious split.

Every production consumer now assembles the same component object list rather
than obtaining ConSan incidentally from `rocjitsu_code`. Configure-time checks
require every active production `.cpp` to belong to exactly one component (the
required inert `consan_lowerer.cpp` tombstone is the sole exception). The
architecture-boundary test requires all six targets, rejects a collapse back
into `rocjitsu_code`, and rejects upward includes from contracts, target
normalization, analysis, transformation, and validation. Thus the conceptual
direction, compilation ownership, final binaries, and structural enforcement
now agree.

| Signal | Checkpoint 16 | Cumulative change |
| --- | ---: | ---: |
| Production files | 261 | +32 |
| Physical production lines | 105,243 | +268 |
| Nonblank production lines | 99,041 | **-42** |
| Production implementation lines | 91,378 | **-72** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 |
| `ConSanTransformArtifacts` references / files | 239 / 58 | **-37 / +1** |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 304 / 53 | +14 / +2 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a |
| Test inventory | 5,358 | +13 |

The facade deletion and shared analysis entry point together reduce production
by 13 physical, four nonblank, and twelve implementation lines relative to
checkpoint 15. More importantly, final validation no longer reaches backward
through orchestration, four additional broad-transaction references disappear,
and two option consumers disappear. The build split itself adds no production
C++ implementation. Production is now 72 implementation lines below the
fifth-refactoring baseline, still far short of the material reduction required
for completion.

Validation includes repeated full `-j16` rebuilds after all ConSan sources were
recompiled under their new component targets, plus the 123-test program-
inventory, fault, perturbation, final-proof, and architecture-boundary gate.
All are green. The immediately preceding checkpoint supplied the complete
4,723-test nonphysical and 635-test physical matrices, so those expensive gates
were not repeated for this semantics-preserving source/build-ownership slice.

Section 14's first criterion now has a concrete core graph rather than an open
placeholder, but the completion audit must still judge whether its component
cuts survive the remaining migrations. Full target-facet locality, both
extension exercises, the broad transaction and operating-point surfaces,
material code shrinkage, and the final independent deep-read audit remain
open.

### 16.18 Convergence checkpoint 17: additive extension proof and typed MOI state

This checkpoint, through commit `3c19d7ec3c`, closes the previously open
extension exercises and continues replacing the MOI operating point's loose
scalar fields with semantic allocations. It is a substantial locality and
dataflow investment, but its accounting is also an explicit warning: the
checkpoint adds 160 implementation lines relative to checkpoint 16 and leaves
production 88 lines above the fifth-refactoring baseline. It therefore does
not satisfy the material-shrinkage criterion, and the next work must harvest
the typed model rather than continue accumulating wrappers.

On the target axis, CDNA descriptor validation now lives in
`consan_validation_gfx9_cdna_target_ops.cpp`. Common final validation asks a
target operation to validate and normalize descriptor state; it no longer
owns the gfx9 register-field recipe. A focused hypothetical target registers
an already-represented program-analysis operation through the target registry
and proves `HypotheticalTargetRegistersNormalizedAnalysisWithoutModeChanges`
without editing a mode implementation. On the orthogonal axis, a hypothetical
mode registers its semantic MOI demand through the mode registry and proves
`HypotheticalModeRegistersWithoutConcreteTargetChanges` without editing a
concrete target package. The architecture-boundary gate requires both
extension fixtures and their additive registration paths. These are now
concrete evidence for Section 14.4 rather than proposed exercises.

The larger slice follows accepted MOI register state from mode demand through
placement, prologue planning, relocation, emission, and validation. Scalar
router state, its call and jump allocations, branch-only preservation,
dispatch identity, owner-scalar provenance, persistent workgroup tuples, and
owner/epoch pairs now have typed owners. Invalid half-present tuples cannot be
constructed through their public interfaces. Accepted code-object-wide
owner/epoch allocation is distinct from effective site-local materialization;
materialized sources no longer ride on the broad `MoiOptions` transaction.
The last migration centralizes the one optional raw owner/epoch pair in
`ConSanMoiOwnerEpochRegisterState`, reuses it in both accepted vector and
persistent scalar state, and removes generic `moi_owner_vgpr` and
`moi_epoch_vgpr` projection peepholes. Structural checks prohibit those free
projections and duplicate optional-pair storage from returning.

The deep trace exposed one real authority bug while accepted and materialized
state were being separated. Required-prologue-VGPR calculation reread the
broad accepted allocation after the site-local plan had selected scalar
persistent state, allowing router state to alias the epoch source. The
calculation now consumes the materialized plan, and
`Gfx1100RecordReplayRouteKeyDoesNotAliasPersistentEpoch` keeps the failure at
its owning boundary. This is the only behavior bug found in the checkpoint;
new focused tests cover the two extension exercises and typed-state contracts,
and the test inventory grows by five after two former contract cases were
replaced by their stricter typed-state forms.

| Signal | Checkpoint 17 | Cumulative change |
| --- | ---: | ---: |
| Production files | 262 | +33 |
| Physical production lines | 105,479 | +504 |
| Nonblank production lines | 99,230 | +147 |
| Production implementation lines | 91,538 | **+88** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a |
| Test inventory | 5,363 | +18 |

Relative to checkpoint 16, the operating-point work removes eleven direct
`ConSanMoiOperatingPoint` references and seventeen implementation lines from
the final accepted-versus-materialized cleanup, after the earlier typed-state
slices paid for stronger invariants. The checkpoint as a whole nevertheless
adds one target-owned production file, 236 physical lines, 189 nonblank lines,
and 160 implementation lines. This fails checkpoint 16's requested immediate
size payback. The result is retained because it closes an explicit completion
criterion, fixes a demonstrated bug, and replaces representable invalid states
on a main production path; it is not evidence that Section 14.8 is complete.
The next convergence work must turn the new typed seams into deletion and
narrower APIs, with the broad operating point and transform transaction as
primary evidence to inspect rather than assumptions about where savings lie.

Validation includes a complete rebuild; a 582-test focused component gate; a
1,780-test affected device matrix over gfx942, gfx950, gfx1100, gfx1201, and
gfx1250; the complete 4,728-test nonphysical matrix at `-j16`; and all 635
serialized physical gfx1201 tests at `-j1`. Two sub-second simulator cases
exceeded their 60-second budgets only under full-gate contention, passed
immediately in isolation, and received 120-second budgets before the entire
nonphysical gate was repeated successfully. No test was removed or disabled.

Section 14.4 now has direct evidence, and the typed state removes several
opportunistic cross-layer reads. Completion remains unproved. In particular,
the physical locality of every target facet, reader-skippable ownership across
all four modes, the breadth of the mutable transaction and operating-point
surfaces, the smallness of surviving mode/target interactions, legacy
harvesting, material code shrinkage, and the required independent deep-read
audit remain open.

### 16.19 Convergence checkpoint 18: one lowering-commit authority

The resumed branch contains two intentionally separate measurement intervals.
Checkpoint 17 was followed by the urgent issue fixes and the merge of
`origin/develop`; those intervening changes enlarged the declared production
scope by 283 physical, 271 nonblank, and 257 implementation lines before this
refactoring slice began at merge commit `8a6b4dd2f39`. They are retained in the
cumulative totals below but are not attributed to this slice. Against that
exact merged starting tree, this checkpoint removes 14 physical, 21 nonblank,
and 19 implementation lines.

The slice deep-traced every accepted lowering from each mode-owned emitter
through staging, final validation, public result publication, runtime mapping,
and candidate rollback. `ConSanTransformArtifacts` and
`TransformResult::PrivateLoweringArtifacts` both carried parallel accepted
commit inventories, while the transform artifact also carried a separately
mutable runtime-mapping projection. The coverage ledger is now the sole owner
of accepted `ConSanCommittedLowering` transactions. It validates and publishes
single or batched transactions atomically, derives runtime mapping from its
owned commits, verifies its immutable relationship to the observation plan,
and owns the operation that retracts instrumented commits when candidate bytes
are discarded. Mode emitters explicitly publish to that authority; final
validation, result publication, runtime-binding failure, and rollback no
longer reconstruct or synchronize parallel representations.

The public `TransformResult` still publishes the runtime mapping needed by
hooks, but it is a derived output of the ledger and `well_formed()` rejects any
divergence. Tests no longer bypass the production owner with a public
outcome-only mutation API: test-only support constructs complete synthetic
commits, including their required runtime attribution. Owner-level regressions
cover malformed mapping rejection, atomic batch publication, and selective
rollback. The architecture-boundary gate prohibits a second
`committed_lowerings` inventory or transform-artifact runtime-mapping field
from returning and requires the ledger-owned projection and rollback contract.

| Signal | Checkpoint 18 | Cumulative change | Slice change from `8a6b4dd2f39` |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,748 | +773 | **-14** |
| Nonblank production lines | 99,480 | +397 | **-21** |
| Production implementation lines | 91,776 | **+326** | **-19** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 | 0 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** | 0 |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** | 0 |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a | 0 |
| Test inventory | 5,371 | +26 | 0 |

Validation includes a clean incremental rebuild; 74 focused owner, policy,
pipeline, and engine-conformance tests; 26 focused runtime-hook mapping tests;
the structural architecture-boundary test; the complete 4,736-test
nonphysical ConSan matrix at `-j16`, including all 2,908 simulated-device rows
over five architectures; and all 635 physical gfx1201 tests serialized at
`-j1`. No test was removed or disabled.

This closes one broad mutable-transaction violation and immediately harvests
the superseded synchronization code, rather than leaving the typed owner beside
the legacy representations. It is useful local size payback, not the material
whole-refactoring shrinkage required by Section 14.8. The cumulative production
scope remains 326 implementation lines above the fifth-refactoring baseline,
and the remaining Section 14 gaps still require a new deep-read after this
checkpoint.
