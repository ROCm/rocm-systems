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
| Physical production lines | 104,976 |
| Nonblank production lines | 99,084 |
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

### 16.20 Convergence checkpoint 19: no synchronization staging authority

The lowering-commit deep read continued through the one apparent exception to
checkpoint 18 ownership: atomic, fence, and barrier emitters still accumulated
accepted commits in `ConSanTransformArtifacts::staged_moi_sync_lowerings`.
That inventory existed because a capture intent may contribute to more than one
emitted synchronization operation. It nevertheless constituted a second
mutable commit authority. It crossed the public-to-private transform-result
boundary, required a separate terminal publication phase, and had to be cleared
independently on every rollback path.

The coverage ledger now owns that composition directly. Its explicit
coalescing publication operation validates every incoming instrumented
transaction, transitively merges accepted transactions that share an intent,
and republishes the merged transaction on a private ledger copy. A malformed or
stale member therefore leaves the original ledger unchanged. Ordinary strict
publication still cannot reopen an accepted intent; only the named
synchronization composition operation can do so. Common commit construction and
ledger publication now share one structural validator instead of maintaining
two nearly identical sets of outcome, geometry, intent, and runtime-mapping
checks.

The old staging vector, its inline merge and terminal-publication methods, all
`TransformResult` transport, the terminal MOI publication step, rollback
clearing, and tests that inspected the incidental staging container have been
removed. Atomic, fence, and barrier emitters publish to the ledger at their
existing common synchronization boundary. An owner-level regression proves
strict rejection, multi-location coalescing, whole-batch rollback, and final
plan/ledger consistency. The architecture-boundary gate now prohibits the old
staging inventory anywhere in production and requires the coalescing operation
to remain on the ledger contract.

| Signal | Checkpoint 19 | Cumulative change | Slice change from checkpoint 18 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,728 | +753 | **-20** |
| Nonblank production lines | 99,457 | +374 | **-23** |
| Production implementation lines | 91,758 | **+308** | **-18** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 | 0 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** | 0 |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** | 0 |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a | 0 |
| Test inventory | 5,372 | +27 | +1 |

Validation includes a clean incremental rebuild; all 15 observation-plan and
coverage-ledger owner tests; the structural architecture-boundary test; the
complete 4,737-test nonphysical ConSan matrix at `-j16`, including all 2,908
simulated-device rows over five architectures; and all 635 physical gfx1201
tests serialized at `-j1`. No test was removed or disabled.

This slice eliminates the final caller-owned lowering-commit inventory and
leaves accepted semantic lowering transactions with one authority throughout
their lifetime. It continues the required forward-only and deletion-bearing
direction, but it does not prove the remaining Section 14 criteria. Production
is still 308 implementation lines above the starting baseline; architecture
and mode locality across every facet, the remaining broad operating-point and
transform-artifact surfaces, smallness of the surviving cross-axis composition,
and an independent whole-codebase audit remain open.

### 16.21 Convergence checkpoint 20: one immutable observation-plan authority

The post-checkpoint-19 dataflow audit found a parallel immutable authority next
to the newly centralized lowering authority. `ConSanObservationPlan` was stored
separately on the observation product, lowering input, private transform
artifacts, and public pipeline result. `ConSanCoverageLedger` then copied every
decision inventory and every complete probe intent into its own representation.
The two representations had to be transported and rolled forward together and
validated by `matches_plan()`. Immutability prevented arbitrary mutation, but
the duplicate ownership still admitted divergence at every construction and
publication boundary.

The coverage ledger now solely owns the composed observation plan by value.
Observation assembly builds the policy fragments in a local plan and moves the
completed value once into the initial ledger. Observation products, lowering
inputs, private artifacts, and public results expose narrow const accessors to
that owner rather than retaining another plan field. Ledger lowering entries
retain only their stable plan-local intent IDs; validation, coverage queries,
dispatch-requirement construction, runtime hooks, lowerers, and tests resolve
the immutable intent through the ledger. The four copied decision vectors, the
copied intent values, all duplicate plan transport, and `matches_plan()` have
been removed.

The architecture-boundary gate now requires the ledger-owned plan, rejects a
plan field on either result bus, rejects the four duplicate decision
inventories, and rejects restoration of a plan-comparison operation. The owner
regression proves that lowering state is joined to plan-local IDs while both
the immutable plan and accepted commits remain available from the one ledger.
During the first full gate, the conversion also exposed an incorrect attempted
replacement for `matches_plan()`: final validation required `plan.valid()` even
for fault-only transforms whose observation stage is intentionally not
applicable. That new check was removed rather than preserving obsolete
validation for a representation that no longer exists; all 81 affected fault,
composition, and simulator regressions then passed.

| Signal | Checkpoint 20 | Cumulative change | Slice change from checkpoint 19 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,716 | +741 | **-12** |
| Nonblank production lines | 99,441 | +358 | **-16** |
| Production implementation lines | 91,742 | **+292** | **-16** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Explicit mode-enum references in `consan_moi.cpp` | 0 | -20 | 0 |
| Explicit mode-enum references in `consan_moi_placement.inc` | **0** | **-95** | 0 |
| Explicit mode-enum references in `consan_moi_report_plan.cpp` | **0** | **-18** | 0 |
| Explicit mode-enum references in `consan_pipeline.cpp` | **0** | n/a | 0 |
| Test inventory | 5,372 | +27 | 0 |

Validation includes a clean 160-step incremental rebuild; all 18 observation-
plan and observation-policy tests; all 94 ConSan runtime-hook unit tests; the
structural architecture-boundary test; a focused rerun of all 81 cases affected
by the fault-only validation error; a fresh complete 4,737-test nonphysical
ConSan matrix at `-j16`, including all 2,908 simulated-device rows over five
architectures; and all 635 physical gfx1201 tests serialized at `-j1`. No test
was removed or disabled.

This slice removes another whole parallel representation from the main semantic
spine and immediately harvests its validation and transport machinery. It is
forward-only and deletion-bearing, but it is not completion evidence for the
whole refactoring. Production remains 292 implementation lines above the
starting baseline, and the remaining broad operating-point and transform-
artifact surfaces, full architecture and mode locality, smallness of surviving
cross-axis composition, and an independent Section 14 audit remain open. The
two extension exercises have remained closed and enforced since checkpoint 17;
they are not part of this remaining-work list.

### 16.22 Convergence checkpoint 21: derive the runtime mapping projection

The next adjacent ownership trace followed `ConSanRuntimeStaticMapping` from
accepted lowering commits to HSA registration. `TransformResult` retained a
stored copy even though its only assignment derived it from the coverage
ledger, its only production consumer passed it to the runtime hook, rollback
cleared it beside the ledger, and `well_formed()` compared it back to a fresh
ledger projection. The field was therefore a cache without independent
semantics and another representation capable of diverging from its authority.

`TransformResult` now exposes a const semantic accessor that derives the
runtime mapping directly from the ledger's authoritative commits. Publication,
rollback, deferred-binding failure, result validation, the hook consumer, and
the pipeline equivalence regression no longer transport, clear, or compare a
stored copy. The architecture-boundary gate now rejects a runtime-mapping field
on both the private and public result buses while retaining the ledger's named
projection operation.

| Signal | Checkpoint 21 | Cumulative change | Slice change from checkpoint 20 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,714 | +739 | **-2** |
| Nonblank production lines | 99,438 | +355 | **-3** |
| Production implementation lines | 91,739 | **+289** | **-3** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Test inventory | 5,372 | +27 | 0 |

Validation includes a clean CMake regeneration and 68-step incremental build,
all 31 pipeline tests, all 94 ConSan runtime-hook unit tests, and the structural
architecture-boundary test. The complete nonphysical and serialized physical
inventories passed immediately before this narrow slice at checkpoint 20; this
checkpoint does not misattribute those earlier full runs to the later change.
No test was removed or disabled.

This closes the last independently stored static runtime projection found on
the main observation-to-runtime spine, but it is only a small deletion payoff.
The fifth refactoring remains incomplete: production is 289 implementation
lines above baseline, and the wider Section 14 architecture, locality,
cross-axis-composition, size, and independent-audit evidence remains open.
Section 14.4's two extension exercises remain closed and enforced by the
checkpoint-17 fixtures.

### 16.23 Convergence checkpoint 22: target-owned dispatch-identity placement

The next cross-axis trace followed report dispatch identity from each mode's
semantic demand through scalar search, persistent-vector capture, literal
fallback, private entry capture, and final resource validation. The mode
registry already owned whether an engine needed the identity and which
lossless fallback it allowed. Target profiles already owned the source of that
identity. Nevertheless, the shared placement routine rediscovered gfx9, CDNA,
RDNA, gfx11, and gfx12 families at each overflow step and performed a second
mode-registry lookup to recognize InlineShadow's private-capture permission.
Those branches made one common register-search mechanism jointly interpret
both extension axes.

Every target profile now selects one normalized
`ConSanMoiDispatchIdentityPlacement` strategy:
`PreloadedScalar`, `PersistentVectorPreferred`, `ScalarThenLiteral`, or
`ScalarThenPersistentVector`. The default is deliberately `Unsupported`, and
profile validation rejects both that fail-closed default and inconsistent
source/strategy combinations from the five-entry production registry. The
profiles also state whether access reports need explicit dispatch identity;
mode planning consumes that semantic fact without learning the target family.
Conversely, the InlineShadow plan alone grants private entry capture, while
Record/Replay and Sampled do not. Shared placement now composes those two
products and continues to own the one scalar/VGPR search implementation.

The boundary gate isolates the complete automatic-dispatch placement
subroutine and rejects target-family predicates or a second mode-registry
lookup inside it. Exact target-row and malformed-profile tests cover all five
target rows, all four supported strategies, and the rejected default;
mode-planning tests cover the three dispatch consumers and the InlineShadow-
only private permission. Across the larger placement component, target-family
predicate references fall from 36 to 28 and mode-registry lookups fall from 11
to 10. The subroutine itself now has neither.

| Signal | Checkpoint 22 | Cumulative change | Slice change from checkpoint 21 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,756 | +781 | **+42** |
| Nonblank production lines | 99,479 | +396 | **+41** |
| Production implementation lines | 91,777 | **+327** | **+38** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Target-family predicates in `consan_moi_placement.inc` | 28 | n/a | **-8** |
| Mode-registry lookups in `consan_moi_placement.inc` | 10 | n/a | **-1** |
| Test inventory | 5,372 | +27 | 0 |

This is a locality investment, not a deletion-bearing or material-shrinkage
checkpoint. It adds 38 implementation lines and moves cumulative production
to 327 lines above baseline. Its concrete deletion leverage is that all four
dispatch-placement variants can now be simplified or separated by semantic
strategy without repeating family inference, and no future target needs to
add another branch to this common solver. The next checkpoint must harvest
that leverage or another already-established typed seam; it must not make a
second consecutive schema investment that postpones the same size payoff.

Validation includes a 167-step incremental rebuild and an exact-final-tree
74-test capability, mode-planning, dispatch-placement, and structural-boundary
gate. The complete exact-final-tree 4,737-test nonphysical matrix, including all
2,908 simulator rows over five targets, passed at `-j16` in 195.69 seconds. All
635 physical gfx1201 tests passed on the exact final tree serialized at `-j1`
in 109.23 seconds. No test was removed or disabled.

The slice improves Section 14.2 and 14.6 but does not prove either across the
whole codebase. Section 14.4 remains complete through the checkpoint-17
extension fixtures. Broad operating-point and transform-artifact surfaces,
the other target and mode facets, material shrinkage, and the independent
whole-codebase audit remain open.

### 16.24 Convergence checkpoint 23: one synchronization publication transaction

The required deletion pass traced the terminal half of every MOI
synchronization mutation. Record/Replay atomic and fence records, Sampled
barrier and atomic metadata, InlineShadow atomic ordering, and the common
barrier paths each independently performed the same sequence: apply patched-
image growth policy, replace executable text, publish the coalescing semantic
commits, emit the replacement image, append patch proof, and mark the result
modified. Planning, target emission, and diagnostics were mode-specific, but
that six-step transaction was not. Eight copies made it possible for one mode
to publish bytes or patch telemetry in a different order from another.

`publish_moi_sync_patch` now owns that inseparable terminal transaction once.
Mode and shared-barrier owners still build their exact bytes, typed patch
products, and intent-bound commits; they move those completed products across
the common boundary and retain no terminal publication policy. The former
commit-only `publish_moi_sync_lowering_commits` API and all seven additional
transaction implementations are gone. Patch products are moved rather than
copied into the lasting artifact. The architecture-boundary gate prohibits
the commit-only API from returning and permits direct coalescing publication
only in the coverage-ledger implementation and this one transaction owner.

| Signal | Checkpoint 23 | Cumulative change | Slice change from checkpoint 22 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,721 | +746 | **-35** |
| Nonblank production lines | 99,444 | +361 | **-35** |
| Production implementation lines | 91,739 | **+289** | **-38** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 200 / 28 | 0 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Independent synchronization terminal implementations | 1 | n/a | **-7** |
| Test inventory | 5,372 | +27 | 0 |

This checkpoint exactly repays the 38 implementation lines added by
checkpoint 22 while retaining its target/mode locality, and returns cumulative
production to checkpoint 21's 289-line increase over baseline. It is therefore
the promised immediate deletion payoff, not material whole-refactoring
shrinkage. The next work must continue deleting from that cumulative total;
restoring eight mode-local transaction copies would be both a size regression
and a false form of mode locality.

Validation includes a regenerated 69-step build and a 256-test focused atomic,
fence, barrier, ledger, growth-policy, and structural-boundary gate. Two full
`-j16` runs each exposed a different 60-second gfx1250 simulator timeout:
InlineShadow scaled WMMA and SuperCollider reduction. Each passed immediately
in serial isolation in 0.55 and 0.47 seconds respectively, so their individual
budgets were raised to 120 seconds. The repeated complete 4,737-test
nonphysical matrix, including all 2,908 simulator rows over five targets, then
passed at `-j16` in 195.47 seconds. All 635 physical gfx1201 tests passed
serialized at `-j1` in 108.85 seconds. No test was removed or disabled.

The checkpoint strengthens Sections 14.3, 14.6, and 14.7 at one exact subset-
shared boundary. It does not close the remaining broad operating-point and
transform-artifact surfaces, prove locality for every mode and target facet,
produce material net shrinkage, or replace the required independent deep-read
audit. Section 14.4 remains complete through the checkpoint-17 extension
fixtures.

### 16.25 Convergence checkpoint 24: one access publication transaction

A complete lifecycle read of MOI access application found four copies of the
same terminal ownership policy. Record/Replay, Sampled, and InlineShadow each
had an appended-probe path that independently emitted the replacement image,
published ordinary intent-bound lowering commits, appended patch proof, and
marked the transform modified. The shared inline byte-application template
repeated the same publication sequence after its different in-place byte and
descriptor strategy. The apparent failure-policy difference was not a mode
contract: access application is the first mutation in all three engine
orchestrators, and every later synchronization or prologue mutation consumes
its published result.

`publish_moi_access_patch` now owns the inseparable completed-access
transaction once. Inline application builds its candidate replacement in a
local value rather than using `ConSanTransformArtifacts::replacement` as
scratch storage, and transfers that image together with its intent-bound
commits and patch proof only after all construction succeeds. The three mode-
local appended paths retain their distinct growth-policy label, byte
construction, descriptor requirements, relocation geometry, and diagnostics,
then move the same three completed products across the shared boundary.
Record/Replay also retains its mode-private reserved barrier-island and relay-
range continuation state after successful publication.

The former four direct terminal implementations and their compensating
replacement clears or whole-candidate discards are gone. The architecture-
boundary gate now rejects direct ledger publication, direct replacement-bus
mutation, direct modification marking, or candidate discard in each of the
three mode-owned access bodies. That makes access publication a mechanically
enforced shared mechanism without moving probe semantics out of their engine
owners or duplicating them in a nominally local facade.

| Signal | Checkpoint 24 | Cumulative change | Slice change from checkpoint 23 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,718 | +743 | **-3** |
| Nonblank production lines | 99,439 | +356 | **-5** |
| Production implementation lines | 91,731 | **+281** | **-8** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 243 / 58 | **-33 / +1** | +2 / 0 |
| `ConSanPatchInfo` references / files | 202 / 28 | +2 / 0 | +2 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Independent access terminal implementations | 1 | n/a | **-3** |
| Test inventory | 5,372 | +27 | 0 |

The new transaction signature costs two occurrences each of the broad
transform-artifact and patch-proof types. That is explicit shared-boundary
coupling rather than a hidden mode-to-mode peephole, but it remains part of the
Section 14 broad-surface debt. In return, the slice deletes eight implementation
lines and prevents three mode-local copies from regrowing. Cumulative
production remains 281 implementation lines above baseline, so this is a
directional deletion checkpoint, not the material whole-refactoring shrinkage
or completion evidence required by Section 14.

Validation includes a regenerated build and a 484-test focused Record/Replay,
Sampled, InlineShadow, access, and structural-boundary gate. The complete
4,737-test nonphysical matrix, including all 2,908 simulator rows over five
targets, passed at `-j16` in 197.84 seconds. All 635 physical gfx1201 tests
passed serialized at `-j1` in 109.05 seconds. No test was removed, disabled, or
renamed. No behavior bug was discovered in this ownership slice, so the new
coverage is the structural regression gate that makes the publication boundary
durable rather than a behavior-specific regression fixture.

This checkpoint strengthens Sections 14.2, 14.3, 14.6, and 14.7 at the common
access-publication seam. It does not close the remaining wide operating-point
and transform-artifact surfaces, prove locality for every target and mode
facet, produce material net shrinkage, or replace the required independent
whole-codebase audit. Section 14.4 remains complete through the checkpoint-17
extension fixtures.

### 16.26 Convergence checkpoint 25: all-engine access transactions

The post-checkpoint-24 audit followed the same access-publication lifecycle
through the fourth engine. SuperCollider's flat and LDS redundant-access
lowerers still published ordinary lowering commits directly. More seriously,
the LDS lowerer used `ConSanTransformArtifacts::replacement` and `patches` as
construction scratch while it emitted descriptor growth, dense dispatchers,
entry islands, relay reservoirs, branch-only routes, and final access proof.
Twenty-four failure sites then cleared the lasting patch inventory to simulate
rollback. That was a forward-only pipeline violation and left the supposedly
common access transaction as an MOI-only mechanism.

The completed-access transaction now belongs to its actual state owner:
`ConSanTransformArtifacts::publish_access_lowering`. It accepts a locally
completed replacement image, intent-bound commits, patch proof, and a caller-
owned diagnostic subject. The coverage ledger is published first; success
makes the bytes and proof visible together, while failure retracts the complete
instrumented candidate. Record/Replay, Sampled, InlineShadow, and both
SuperCollider redundant-access paths now use this one transaction. The
checkpoint-24 MOI-specific publication function is deleted rather than kept as
an adapter.

SuperCollider LDS construction now owns local replacement and proof values from
the first emitted byte through descriptor-image verification. Its appended and
inline/local strategies converge before publication, removing both direct
terminal branches, all 24 compensating patch clears, and all 43 direct
`result.replacement`/`result.patches` construction references. SuperCollider
flat construction still intentionally starts from the already published LDS
image when the two access families compose, but its new flat products remain
local until the same transaction accepts them. Engine-specific planning,
target recipes, relay geometry, diagnostics, and composition order remain in
the SuperCollider owner.

The boundary gate forbids the retired MOI-specific publication API everywhere,
confines ordinary batch publication to the coverage owner, pipeline rejection
assembly, and the one access transaction, and prevents the SuperCollider LDS
body from reacquiring the lasting replacement/patch buses or rollback method.
An owner-level test publishes a complete access transaction, then submits a
mixed stale batch and proves that replacement bytes, patch proof, and earlier
instrumented ledger state are retracted together.

| Signal | Checkpoint 25 | Cumulative change | Slice change from checkpoint 24 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,673 | +698 | **-45** |
| Nonblank production lines | 99,397 | +314 | **-42** |
| Production implementation lines | 91,689 | **+239** | **-42** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 241 / 58 | **-35 / +1** | **-2 / 0** |
| `ConSanPatchInfo` references / files | 202 / 28 | +2 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| SuperCollider LDS direct replacement/patch-bus construction references | 0 | n/a | **-43** |
| Independent access publication transactions | 1 | n/a | 0 |
| Test inventory | 5,373 | +28 | **+1** |

This is the first access-publication boundary that covers all four engines
rather than treating SuperCollider as an exception. It deletes 42 production
implementation lines and reduces the broad artifact type by two occurrences,
but cumulative production remains 239 implementation lines above the starting
baseline. It therefore advances the deletion contract and all-engine mode
locality without satisfying the material whole-refactoring shrinkage bar.

Validation includes a regenerated 188-step build; a 1,168-test focused gate
covering the structural boundary, every MOI access family, SuperCollider,
LDS/flat probes, and five-target simulator rows; and the dedicated two-test
transaction/boundary gate. The complete 4,738-test nonphysical matrix,
including all 2,908 simulator rows over five targets, passed at `-j16` in
195.98 seconds. All 635 physical gfx1201 tests passed serialized at `-j1` in
109.68 seconds. No test was removed, disabled, or renamed; the one new test is
the transaction-owner regression described above.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, and 14.7. It does
not eliminate the remaining broad artifact/operating-point surfaces, prove
architecture and mode locality across every production facet, produce material
net shrinkage, or replace the required independent whole-codebase audit.
Section 14.4 remains complete through the checkpoint-17 extension fixtures.

### 16.27 Convergence checkpoint 26: owner-resolved descriptor mutation

A lifecycle read of descriptor requirements found parallel mutation authorities
below otherwise-converged access and prologue planning. MOI had separate VGPR,
SGPR, private-segment, and LDS applicators for patcher and byte images plus a
mode-private active-kernel resolver. Its owner/epoch and dispatch prologues had
additional direct register-growth helpers. SuperCollider separately aggregated
register-growth vectors, applied four register/image variants, repeated private-
segment loops, and reconstructed moved descriptor offsets in its flat lowerer.
These were not distinct mode semantics: all translated immutable
`ProgramInventory` descriptor identities to the current image by stable kernel
name, took maximum resource extents, and grew fields monotonically.

`ConSanDescriptorMutationBatch` now carries owner-keyed VGPR, SGPR, private-
segment, and group-segment requirements through one descriptor owner. That
owner indexes the immutable and active images once, resolves a descriptor once
even after text growth moved it, reads it once, applies every requested field,
and commits it once. The patcher and byte entry points share that implementation.
Zero extents remain intentional no-ops. The genuinely different policies stay
outside the mechanism: MOI supplies its ordinary-VGPR limit, inventory-backed
empty-accumulator proof, and optional target capability for maximum LDS;
SuperCollider supplies its own VGPR limit and does not claim the accumulator
proof. Low-level target-normalized VGPR/SGPR encoding remains in the descriptor
primitive rather than moving into either engine.

All MOI access engines, synchronization paths, barriers, atomics, fences, and
private and owner/epoch prologues now submit complete batches through one narrow
MOI policy adapter. The inline access template receives private and LDS
requirements as ordinary inputs instead of an open-ended extra-mutation
callback. SuperCollider flat and LDS paths submit the same common batch for
both appended and byte-image strategies. The old MOI resolver and eight
field/image applicators, the prologue register-growth helpers, SuperCollider's
register-growth merge representation, both duplicated private loops, and the
flat-specific moved-descriptor resolver are deleted. No behavior defect was
established during this slice; the change makes all already-required resource
fields participate in the same owner transaction and removes divergent future
growth paths.

The structural gate confines direct use of the transaction to its descriptor
owner, the MOI policy adapter, and the two SuperCollider consumers. It rejects
all retired APIs and types, and confines low-level descriptor register/spill
mutation to the descriptor owner and its validation probe. An owner-level
regression grows `.text` so the descriptor moves, submits all four resource
requirements using the pristine owner identity, and proves that every field is
published at the moved descriptor.

| Signal | Checkpoint 26 | Cumulative change | Slice change from checkpoint 25 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,388 | +413 | **-284** |
| Nonblank production lines | 99,124 | +41 | **-273** |
| Production implementation lines | 91,420 | **-30** | **-269** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 227 / 57 | **-49 / 0** | **-14 / -1** |
| `ConSanPatchInfo` references / files | 202 / 28 | +2 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Independent descriptor resource-mutation owners | 1 | n/a | converged |
| Test inventory | 5,374 | +29 | **+1** |

The physical-line row uses an exact-tree recount and corrects checkpoint 25's
printed value from 105,673 to 105,672. The slice delta is therefore the exact
284-line change.

This is the first checkpoint in the latest convergence sequence to return the
implementation count below the fifth-refactoring baseline. It repays all 239
lines still above baseline after checkpoint 25 and deletes another 30. That is
real harvested implementation, but 30 lines is not yet a material whole-
refactoring reduction, so it is directional evidence for Section 14.8 rather
than completion of that requirement.

Validation includes a regenerated build, the dedicated descriptor-owner and
architecture-boundary regressions, and an 882-test descriptor, MOI, and
SuperCollider focused gate. The complete exact-tree 4,739-test nonphysical
matrix, including all 2,908 simulator rows over five targets, passed at `-j16`
in 201.53 seconds. All 635 physical gfx1201 tests passed on the same tree,
serialized at `-j1`, in 108.69 seconds. No test was removed, renamed, or
disabled; the one added test is the moved-owner transaction regression.

The checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, and the
direction required by 14.8. It does not prove architecture and mode locality
across every remaining production facet, materially shrink the whole codebase,
eliminate the other broad artifact and operating-point surfaces, or replace the
required independent whole-codebase deep-read audit. Section 14.4 remains
complete through the checkpoint-17 extension fixtures.

### 16.28 Convergence checkpoint 27: transactional fault application

A post-checkpoint-26 read followed the typed fault plans from planning through
composition and every mutation mechanism. Planning and selection already had
narrow products, but composition still validated plans, rediscovered seven
mutation kinds, dispatched seven separately exported mechanism functions, and
let each function mutate the shared transform artifact directly. The image-
growth policy and uncovered-local-cave search also accepted that broad artifact
despite consuming only the pristine image identity, an error sink, or immutable
program inventory.

The fault component now exports one operation,
`apply_consan_fault_mutations`, for a complete typed-plan set. Validation,
kind grouping, compatibility checks, dispatch order, cardinality accounting,
and all seven concrete mechanisms are private to that owner. Composition no
longer knows the mechanism names or reconstructs their ordering. The one
public operation consumes only the patched-image growth limit and cardinality
policy rather than the broad request object. The seven public mechanism
declarations and the parallel composition dispatch are deleted rather than
retained as compatibility adapters.

All mechanisms build through one private `FaultApplicationState`. It contains
only immutable selection inputs and the candidate image, patch proof, mutation
tally, outcome, and diagnostics needed by this component; it is not another
public transform bus. A complete successful plan set publishes those products
together. A mechanism error publishes its diagnostics but rolls back candidate
bytes, patch proof, applied tally, outcome, and success diagnostics. A distinct
post-application cardinality rejection retains the completed application tally
for audit compatibility while still withholding the rejected image. Two new
owner-level regressions prove both boundaries: independent atomic and LDS
mechanisms commit together, while a valid atomic candidate followed by a stale
LDS plan leaks neither mutation products nor a false success diagnostic.

That failure regression exposed a real preexisting bug. Before the private
transaction, the first mechanism could publish replacement bytes, proof, and
an applied tally before a later mechanism rejected a stale semantic owner.
Finalization normally converted the result to invalid, but the component
boundary itself exposed a partial candidate and a misleading success warning.
The regression calls the production application boundary directly, after
planning, so later coordinator cleanup cannot mask the ownership defect.

Two adjacent mechanical contracts are narrower as part of the same slice.
`replace_consan_text` consumes an explicit pristine code-object identity and
error sink instead of the whole transform artifact. Uncovered NOP-cave
discovery consumes immutable `ProgramInventory` rather than reaching through
the mutable result. Every caller across all four modes now supplies those
products explicitly. The structural gate forbids broad options or transform
artifacts in the growth-policy contract, proves every private fault mechanism
accepts the candidate state rather than the public bus, prevents mechanism
names from escaping through the public fault contract, and prevents
composition from regaining private dispatch.

| Signal | Checkpoint 27 | Cumulative change | Slice change from checkpoint 26 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,434 | +459 | **+46** |
| Nonblank production lines | 99,164 | +81 | **+40** |
| Production implementation lines | 91,455 | **+5** | **+35** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | **-20 / -1** |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | +2 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Public concrete fault-application mechanisms | 0 | n/a | **-7** |
| Complete typed-plan application owners | 1 | n/a | converged |
| Test inventory | 5,376 | +31 | **+2** |

The exact-tree recount also corrects checkpoint 26's printed
`ConSanPatchInfo` value from 202 to 204; the baseline and current values above
use exact-token counts. The local dispatch-convergence commit initially
deleted 21 production implementation lines. Making rollback explicit and
narrowing every fault, growth, and cave contract then added 56, for a net
35-line slice increase. The cumulative implementation count is consequently
five lines above the starting baseline. This is substantial broad-bus and
public-surface reduction, but it is not acceptable evidence for Section 14.8:
later slices must harvest enough superseded machinery to produce material
whole-refactoring shrinkage.

Validation includes a full `-j16` rebuild; all 104 fault/application and
structural-boundary focused tests; the complete 4,741-test nonphysical matrix,
including all 2,908 simulator rows over five targets, at `-j16` in 197.71
seconds; and all 635 physical gfx1201 tests serialized at `-j1` in 108.79
seconds. No test was removed, renamed, or disabled. The inventory grew by the
two successful-commit and rollback regressions described above.

This checkpoint strengthens Sections 14.1, 14.5, 14.6, 14.7, and 14.9. It
does not complete architecture or mode locality across the remaining
production surface, eliminate the broad operating-point and transform buses,
achieve material code shrinkage, or replace the required independent deep-read
completion audit. Section 14.4 remains complete through the checkpoint-17
extension fixtures, and the goal remains active.

### 16.29 Convergence checkpoint 28: pipeline-owned lowering summaries

The MOI coordinator formerly ended lowering with fourteen patch-kind searches
and mode-aware diagnostic branches. That block was neither orchestration nor
mode policy: it reconstructed a presentation summary from the already
committed lowering products. Its location made the coordinator know the
diagnostic vocabulary of Record/Replay, Sampled, and InlineShadow, repeated
the same whole-patch scan for every message, and obscured the pipeline's
forward-only terminal path.

The common MOI pipeline now owns one declarative lowering-summary table and
renderer. This is intentionally shared infrastructure, not one copy per mode:
common first-light, prologue, barrier, atomic, synchronization, and fence
products retain one rendering mechanism, while each mode-specific product is
represented by one table row. Adding a new mode does not require copying the
renderer or adding branches to the coordinator. The coordinator projects the
joined private patch proof to a `ConSanPatchKind` inventory, passes the active
engine, modification state, and that inventory across the boundary, and only
appends the returned diagnostics.

The contract is narrower than the initial extraction suggested by the source
shape. A contiguous vector of the private joined `ConSanPatchInfo` cannot be
viewed as a span of its `ConSanPatchLoweringProduct` base because the element
stride differs. Rather than expose the joined proof, the final API consumes
only `span<const ConSanPatchKind>`. Compile-time negative checks reject both a
patch-info span and the broad transform artifact as summary inputs. The direct
owner test supplies every summarized patch kind, duplicates one counted kind,
and proves exact message ordering, active-mode naming, count rendering,
modified-without-summary behavior, and the inventory-only case. Existing
end-to-end InlineShadow and Sampled diagnostic tests continue to exercise the
same production messages.

The superseded coordinator block is deleted in the same slice. The coordinator
contains none of its fourteen presentation cases, no summary diagnostic text,
and no explicit concrete mode-enum reference. The pipeline summary contract
contains no `MoiOptions`, `ConSanOptions`, `ConSanTransformArtifacts`,
`ConSanPatchInfo`, or `ConSanPatchLoweringProduct`. This keeps a genuinely
shared mechanism shared while making its one extension point explicit and
preventing presentation from reopening the broad mutable bus.

| Signal | Checkpoint 28 | Cumulative change | Slice change from checkpoint 27 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,428 | +453 | **-6** |
| Nonblank production lines | 99,150 | +67 | **-14** |
| Production implementation lines | 91,440 | **-10** | **-15** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Concrete summary cases in the MOI coordinator | 0 | n/a | **-14** |
| Broad types in the lowering-summary contract | 0 | n/a | converged |
| Test inventory | 5,377 | +32 | **+1** |

Validation includes a full final-tree `-j16` rebuild; 34 focused pipeline and
existing end-to-end summary tests; all 4,742 nonphysical tests at `-j16` in
197.06 seconds, including the unchanged 2,908 simulator rows over five
targets; and all 635 physical gfx1201 tests serialized at `-j1` in 108.60
seconds. No test was removed, renamed, or disabled. The inventory grew by the
one direct summary-owner regression.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, and 14.9:
it removes a mode-aware coordinator peephole, enforces a narrow forward-only
product, deletes the replaced implementation, and reverses checkpoint 27's
temporary size growth. A cumulative ten-line implementation reduction is not
the material whole-refactoring shrinkage required by Section 14.8, however.
The broad operating point and transform transaction, remaining placement and
validation concentrations, architecture and mode locality across the complete
surface, and the independent deep-read completion audit remain open. The goal
therefore remains active.

### 16.30 Convergence checkpoint 29: one private InlineShadow atomic-table address mechanism

The post-checkpoint deep read followed InlineShadow atomic-table addressing
from its release-slot and causal-snapshot definitions through every native
emission call, the report-buffer layout, host lookup models, exact instruction
tests, and the physical atomic handoff matrix. The release table has 32-byte
entries and the causal-snapshot table has 40-byte entries, but their device
address builders independently emitted the same ten-instruction hash of the
64-bit atomic address. The only semantic difference after that hash was the
entry stride. Despite having no caller outside `consan_moi_sync_emission.cpp`,
both builders were also declared in the shared synchronization-emission
header. A separate shared-lowering translation unit retained redundant forward
declarations for those two functions and two exact-shadow helpers that its
implementation did not call.

The two copies are now one private ABI-typed
`append_inline_atomic_table_address<Entry>` mechanism. Its accepted entry
types are exactly `ConSanMoiInlineAtomicReleaseSlot` and
`ConSanMoiInlineCausalSnapshot`; compile-time size assertions tie the emitted
32- and 40-byte strides to those runtime ABI types. Two small private adapters
retain the distinct register-lifetime plans at their existing call sites. The
common hash, capacity proof, base materialization, and offset addition exist
once. Each specialization still constructs its complete instruction sequence
before one `InstructionSequence::emit_all` transaction, preserving the former
failure atomicity and exact instruction order.

The two public declarations and all four unrelated shared-lowering forward
declarations are deleted. The architecture-boundary gate now rejects any
atomic-table address helper in the public synchronization header, rejects the
four emission-owner helper names in shared lowering, limits the hash constant
to one occurrence, and requires both ABI entry types to instantiate the one
private mechanism. A direct emitted-code regression lowers an InlineShadow
atomic acquire on gfx1100, gfx1201, gfx942, gfx950, and gfx1250 and proves that
the release and causal-snapshot address paths retain their exact 32- and
40-byte stride sequences on every target.

This is a genuine shared-mechanism consolidation within one mode, not a
cross-mode abstraction. It removes duplication and leaked surface without
copying common work into mode packages. The surrounding synchronization
translation unit still contains several engines and remains physically broad;
therefore this slice is deletion and boundary evidence for InlineShadow, not a
claim that Section 14.3 mode locality is complete.

| Signal | Checkpoint 29 | Cumulative change | Slice change from checkpoint 28 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,383 | +408 | **-45** |
| Nonblank production lines | 99,110 | +27 | **-40** |
| Production implementation lines | 91,400 | **-50** | **-40** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Atomic-table address-hash implementations | 1 | n/a | **-1** |
| Public atomic-table address declarations | 0 | n/a | **-2** |
| Redundant emission-helper declarations in shared lowering | 0 | n/a | **-4** |
| Test inventory | 5,378 | +33 | **+1** |

Validation includes a full final-tree `-j16` rebuild; all 34 focused
InlineAtomic and architecture-boundary tests; all 4,743 nonphysical tests at
`-j16` in 197.61 seconds, including the unchanged 2,908 simulator rows over
five targets; and all 635 physical gfx1201 tests serialized at `-j1` in 108.62
seconds. No test was removed, renamed, or disabled. The inventory grew by the
one five-target ABI-stride regression.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, and 14.9:
one duplicate mode-local mechanism and six leaked or redundant declarations
are gone, the surviving operation is private and type-bound, and production
implementation shrinks by forty lines. A cumulative fifty-line reduction is
still not the material whole-refactoring shrinkage required by Section 14.8.
The broad operating point and transform transaction, remaining placement and
validation concentrations, physical locality of the complete architecture and
mode surfaces, surviving mode/target interactions, and the independent
deep-read completion audit remain open. The goal therefore remains active.

### 16.31 Convergence checkpoint 30: one target-normalized indexed report-table operation

The checkpoint-29 follow-through deep-read traced every device-side
`table_base + runtime_index * ABI_stride` calculation rather than stopping at
InlineShadow's atomic tables. RecordReplay access records and dispatch-token
slots, Sampled access banks, atomic windows, watchpoints, barrier records, and
epoch tables, dynamic diagnostic records, and InlineShadow release,
causal-snapshot, and acquired-token tables all require the same target-sensitive
64-bit address formation. Their semantic owners choose different keys, hashes,
banks, indices, tables, and ABI entry types, but several of those owners still
open-coded the identical multiply-and-add lowering. Sampled also exposed a
mode helper from its access-emission header solely so its atomic and
synchronization components could forward through that implementation.

Those paths now converge on one internal `MoiIndexedAddressRequest` and
`append_moi_indexed_address` operation. Its contract is deliberately narrower
than any mode: materialize a 64-bit table base, multiply one preserved VGPR
index by an ABI byte stride, and form the target-normalized 64-bit address. It
constructs the complete sequence before publishing it, clobbers only the
declared address pair and target-required condition state, and consumes a
typed `ConSanTargetProfile` rather than rediscovering architecture families.
Mode owners still own all semantic index construction. RecordReplay still
selects automatic record banks and dispatch-token slots; Sampled still owns
its hashes, bank selection, and single-bank shortcut; InlineShadow still owns
its atomic-key hashes; and the dynamic-record adapter still binds concrete
record layouts. The common operation therefore shares mechanism without
creating a mode policy authority.

The Sampled cross-component forwarding declaration and implementation are
deleted. Its access, atomic, and synchronization owners consume the common
operation directly, as do the two RecordReplay paths and all three
InlineShadow table ABIs. The architecture-boundary gate rejects the retired
dynamic-record-shaped contract and Sampled forwarding name, forbids the
low-level 64-bit VGPR-offset builder in all migrated semantic owners, and pins
the expected common-operation consumers. This makes future reintroduction of
a mode-local address implementation a checked structural failure rather than
a convention.

The direct simulator regression now exercises the mode-neutral operation over
gfx1100, gfx1201, gfx942, gfx950, and gfx1250 with eight real ABI strides,
including the 32-, 40-, and 56-byte InlineShadow entries, four runtime indices,
and three 64-bit table bases. It proves the exact computed address and
preservation of the index and unrelated registers. Existing RecordReplay and
Sampled exact-emission tests were strengthened to require their complete
shared address sequences at the integration boundary; this discovered no
behavioral change, but prevents those paths from silently falling back to
local arithmetic.

| Signal | Checkpoint 30 | Cumulative change | Slice change from checkpoint 29 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,400 | +425 | **+17** |
| Nonblank production lines | 99,128 | +45 | **+18** |
| Production implementation lines | 91,419 | **-31** | **+19** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Target-normalized indexed-address implementations | 1 | n/a | consolidated |
| Sampled cross-component indexed-address forwarding APIs | 0 | n/a | **-2** |
| Migrated owner files with manual VGPR-offset address addition | 0 | n/a | consolidated |
| Test inventory | 5,378 | +33 | 0 |

Validation includes a full final-tree `-j16` rebuild; the focused common,
InlineShadow, RecordReplay, Sampled, and architecture-boundary regressions; all
4,743 nonphysical tests at `-j16` in 194.84 seconds, including the unchanged
2,908 simulator rows over five targets; and all 635 physical gfx1201 tests
serialized at `-j1` in 110.01 seconds. No test was removed, renamed, disabled,
or replaced. The inventory is unchanged because the broader matrix and exact
integration assertions strengthen existing owning tests.

This checkpoint strengthens Sections 14.1, 14.2, 14.3, 14.6, 14.7, and 14.9:
one target-normalized operation now composes with three mode-owned policy
surfaces, and the obsolete Sampled forwarding seam and repeated low-level
lowering are gone. The nineteen-line slice cost is the explicit common request,
broader transactional call sites, and structural enforcement; cumulative
production remains thirty-one implementation lines below baseline. That is
still not the material whole-refactoring shrinkage required by Section 14.8,
nor does one cross-axis operation prove architecture or mode locality across
the whole codebase. The broad operating point and transform transaction,
remaining placement and validation concentrations, surviving mode/target
interactions, physical locality of the complete architecture and mode
surfaces, and the independent deep-read completion audit remain open. The goal
therefore remains active.

### 16.32 Convergence checkpoint 31: gfx12-owned atomic-fault instruction rewriting

The next architecture-locality deep read followed every raw instruction-field
mutation in the fault-injection engine. That engine already classified atomic
sites through the target-operation facade, but then reached back through the
facade: common fault policy decoded and rewrote gfx12 DS `offset0`, flat and
buffer signed 24-bit `ioffset`, and atomic-scope bits itself. Thus a reader of
the common engine still needed gfx12 encoding knowledge, and adding a target
with a different atomic layout would have required editing both its target
owner and common policy.

Raw atomic address and scope rewriting now belongs to the existing gfx12 fault
target owner. The common engine supplies a normalized atomic encoding, access
width, and semantic address delta, and receives a typed result distinguishing
success, invalid encoding, overflow, misalignment, and an already-wave scope.
It continues to own fault selection, semantic delta validation, diagnostics,
patch proof, and the outer all-or-nothing transaction. The gfx12 owner alone
validates exact instruction sizes and owns the word, mask, signed-offset, and
scope-bit recipes. Rejected rewrites leave the instruction unchanged.

This deliberately does not move fault policy into architecture code. Nor is it
a claim that every fault mutation is now target-local: the common engine still
contains an LDS address-operand mutation path whose architectural ownership
must be assessed separately. It does close the concrete atomic peephole without
inventing a second target hierarchy or duplicating mode policy.

The architecture-boundary gate now rejects the raw gfx12 atomic word recipes
in the common fault engine, requires both normalized target operations there,
and pins the concrete recipes to the gfx12 owner. A direct regression covers
flat, buffer, and DS address rewrites; wave-scope rewriting and the
already-wave result; overflow, alignment, and invalid-encoding rejection; and
failure atomicity. The test inventory grows by that one owner-level regression.

| Signal | Checkpoint 31 | Cumulative change | Slice change from checkpoint 30 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,510 | +535 | **+110** |
| Nonblank production lines | 99,229 | +146 | **+101** |
| Production implementation lines | 91,517 | **+67** | **+98** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Common-engine raw gfx12 atomic-field rewrites | 0 | n/a | consolidated |
| Direct gfx12 atomic-rewrite owner regressions | 1 | n/a | **+1** |
| Test inventory | 5,379 | +34 | **+1** |

Validation includes a full final-tree `-j16` rebuild; all 38 focused fault,
target-operation, and architecture-boundary tests; and all 4,744 nonphysical
tests at `-j16` in 196.27 seconds, including the unchanged 2,908 simulator rows
over five targets. In the first serialized physical run, 634 tests passed and
`RecordReplayRepeatedDispatchIdentity.Correct` failed transiently; that test
immediately passed alone, and the complete rerun passed all 635 physical
gfx1201 tests at `-j1` in 109.62 seconds. No test was removed, renamed,
disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.2, 14.5, 14.6, 14.7, and 14.9:
common policy no longer edits gfx12 atomic bitfields and the target owner has a
checked, transactional contract. Its 98-line implementation cost, however,
makes the cumulative production implementation 67 lines larger than baseline
and is negative evidence for Section 14.8. The cost cannot be declared paid
merely by improved locality; subsequent slices must exploit the clarified
boundary to consolidate or delete implementation. Remaining raw target
peepholes, mode locality across the complete surface, the broad operating
point and transform transaction, placement and validation concentrations, and
the independent deep-read completion audit all remain open. The goal therefore
remains active.

### 16.33 Convergence checkpoint 32: one Sampled dense synchronization dispatcher

The post-checkpoint mode-locality read traced Sampled barrier and atomic
synchronization relays through host selection, entry-island construction,
call-key recovery, dispatcher emission, routed probe bodies, and patch-proof
publication. The two semantic owners select different synchronization events
and build different probe bodies, but each independently implemented the same
dense dispatcher: recover or decode the call key, compare every route, restore
SCC when the key carries it, jump through the selected target, terminate the
unmatched path, and publish the same host/dispatcher proof pair.

Those paths now share one private Sampled-owned dispatcher transaction. Each
owner supplies only typed route identities, caller returns, and targets. The
common mechanism owns call-key arithmetic, clone-local versus return-derived
keys, SCC restoration, indirect jumps, and termination. A second private
helper publishes the identical host and dispatcher proof records. Barrier
policy, atomic semantics, target-specific direct-call selection, relay-host
qualification, body construction, and anchor rewriting remain with their
existing owners; no cross-mode abstraction or compatibility wrapper was
introduced.

The architecture-boundary gate requires exactly one raw call-key arithmetic
and SCC-restoration recipe in the Sampled synchronization owner and exactly two
consumers of the one dispatcher. The focused evidence is the existing 23-test
matrix: exact dense barrier and atomic routes on RDNA4, gfx1250, CDNA4, gfx942,
and gfx950; spill-backed and clone-local keys; far targets; mixed direct/dense
routes; host fallback; and the structural gate. No replacement test was needed
because both formerly independent implementations already had direct
behavioral coverage.

| Signal | Checkpoint 32 | Cumulative change | Slice change from checkpoint 31 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,492 | +517 | **-18** |
| Nonblank production lines | 99,208 | +125 | **-21** |
| Production implementation lines | 91,491 | **+41** | **-26** |
| `MoiOptions` references / files | 93 / 28 | +6 / +3 | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | **-1 / 0** |
| `ConSanMoiOperatingPoint` references / files | 293 / 53 | +3 / +2 | 0 / 0 |
| Sampled dense synchronization dispatcher implementations | 1 | n/a | **-1** |
| Sampled dense relay proof publishers | 1 | n/a | **-1** |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a full final-tree `-j16` rebuild; all 23 focused dense
Sampled and architecture-boundary tests; all 4,744 nonphysical tests at `-j16`
in 195.36 seconds, including the unchanged 2,908 simulator rows over five
targets; and all 635 physical gfx1201 tests serialized at `-j1` in 109.86
seconds. No test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.3, 14.5, 14.7, 14.8, and 14.9. It
keeps shared mechanics within the Sampled package, preserves the distinct
barrier and atomic policies, deletes both duplicate recipes and proof
publication, and pays back 26 of checkpoint 31's added implementation lines.
Production implementation remains 41 lines above baseline, so this is useful
deletion evidence rather than the material whole-refactoring shrinkage needed
for completion. The remaining broad attempt and operating-point surfaces,
other mode-local duplication, architecture peepholes, placement and validation
concentrations, and the independent deep-read completion audit remain open.
The goal therefore remains active.

### 16.34 Convergence checkpoint 33: typed Sampled synchronization attempt boundary

Checkpoint 32's dispatcher consolidation exposed the larger dependency around
it. The Sampled synchronization component accepted inherited `MoiOptions`,
copied that complete immutable-input-plus-operating-point aggregate for every
owner, probe, dense group, and ordinary route, and even assigned the immutable
debug `scratch_vgpr` field to communicate a scratch base that every downstream
operation already received explicitly. Inheritance made those copies appear
convenient, but obscured whether each use read caller input or changed resolved
placement state.

The Sampled barrier and atomic synchronization boundary now receives immutable
`ConSanOptions` and `ConSanMoiOperatingPoint` separately. Owner-local binding
copies only the operating point. Immutable request, runtime-resource, policy,
and delay inputs continue to come from the options aggregate; resolved scalar,
persistent, private, dispatch, and router state comes from the point. Scratch
bases remain explicit probe-resource values. The component contains no
`MoiOptions`, no base-class casts, and no mutation of immutable configuration.
Its mode entry remains responsible for decomposing the current attempt into
those two existing products, so no replacement view or compatibility overload
was introduced.

The boundary gate rejects `MoiOptions` and inherited input/point casts in the
Sampled synchronization owner. Together with checkpoint 32's one-dispatcher
rule, this makes the physical mode package express both the semantic split and
its shared internal mechanism. The increased explicit operating-point count is
intentional evidence of the dependency that inheritance previously hid; later
work may narrow that product further, but must not restore the broad attempt
bus to conceal it.

| Signal | Checkpoint 33 | Cumulative change | Slice change from checkpoint 32 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,479 | +504 | **-13** |
| Nonblank production lines | 99,195 | +112 | **-13** |
| Production implementation lines | 91,478 | **+28** | **-13** |
| `MoiOptions` references / files | 79 / 27 | **-8 / +2** | **-14 / -1** |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 303 / 54 | +13 / +3 | **+10 / +1** |
| `MoiOptions` references in Sampled synchronization | 0 | n/a | **-12** |
| Immutable scratch-field assignments in Sampled synchronization | 0 | n/a | deleted |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a full final-tree `-j16` rebuild; all 23 focused dense
Sampled and architecture-boundary tests; all 4,744 nonphysical tests at `-j16`
in 200.64 seconds, including the unchanged 2,908 simulator rows over five
targets; and all 635 physical gfx1201 tests serialized at `-j1` in 108.33
seconds. No test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, and 14.9:
one complete mode component now exposes immutable input and mutable resolved
state as distinct existing products, the broad inherited bus and scratch-field
side channel are gone from it, and the conversion deletes rather than wraps
legacy adaptation. Production implementation is still 28 lines above
baseline, and explicit use of the still-wide operating point is not the final
narrow component contract. Other Sampled regions, the remaining mode owners,
common barriers and prologues, architecture peepholes, placement and validation
concentrations, and the independent completion audit remain open. The goal
therefore remains active.

### 16.35 Convergence checkpoint 34: typed Sampled access attempt boundary

The synchronization boundary in checkpoint 33 left the adjacent Sampled
access owner as the last compiled Sampled component that accepted the inherited
`MoiOptions` attempt. That owner copied the complete aggregate four times while
planning, accounting, finalizing, and emitting candidate-local patches. It then
assigned the immutable debug `scratch_vgpr` field in three of those copies even
though scratch bases were already explicit resource-plan values. As in the
synchronization owner, inheritance obscured which reads were immutable mode
input and which were resolved operating-point state.

The Sampled access boundary now receives `ConSanOptions` and
`ConSanMoiOperatingPoint` separately. Candidate-local binding copies only the
operating point; request, runtime-resource, policy, and delay inputs remain
immutable; and candidate scratch bases remain explicit values. The public
Sampled internal header, access owner, and synchronization owner contain no
`MoiOptions`, inherited input/point casts, or scratch-field assignments. The
mode entry is now the sole compiled Sampled location that accepts the broad
attempt and decomposes it into the two existing products. No replacement view,
compatibility overload, or parallel path was introduced.

The architecture-boundary gate applies that prohibition to all three compiled
Sampled component surfaces and budgets at most one `MoiOptions` reference in
the mode entry. This is a containment rule rather than only an occurrence
reduction: future access or synchronization work cannot regain the inherited
attempt as a convenience. Existing Sampled tests already exercise direct,
spill-backed, dense, workgroup-gated, private-epoch, barrier, and atomic paths
across the supported emulation targets, so no replacement behavioral test was
needed.

| Signal | Checkpoint 34 | Cumulative change | Slice change from checkpoint 33 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,475 | +500 | **-4** |
| Nonblank production lines | 99,191 | +108 | **-4** |
| Production implementation lines | 91,474 | **+24** | **-4** |
| `MoiOptions` references / files | 73 / 25 | **-14 / 0** | **-6 / -2** |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 308 / 54 | +18 / +3 | **+5 / 0** |
| `MoiOptions` references in compiled Sampled components | 1 | n/a | **-6** |
| Sampled whole-attempt candidate copies | 0 | n/a | **-4** |
| Sampled immutable scratch-field assignments | 0 | n/a | **-3** |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a full final-tree `-j16` rebuild; all 159 Sampled and
architecture-boundary focused tests in 1.98 seconds; all 4,744 nonphysical
tests at `-j16` in 196.69 seconds, including the unchanged 2,908 simulator rows
over five targets; and all 635 physical gfx1201 tests serialized at `-j1` in
108.33 seconds. No test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, and 14.9:
all compiled implementation regions of one mode now expose immutable input and
resolved state separately, broad attempt copies and their scratch side channel
are deleted, and the physical boundary enforces the result. Production
implementation is still 24 lines above baseline, and the operating point
remains wider than the access and synchronization owners ultimately need.
Other modes, common barriers and prologues, architecture peepholes, placement
and validation concentrations, code-size convergence, the extension exercises,
and the independent deep-read completion audit remain open. The goal therefore
remains active.

### 16.36 Convergence checkpoint 35: typed Record/Replay construction boundary

The Sampled boundary work made the adjacent Record/Replay contrast explicit.
Record/Replay access, atomic-event, and fence-event construction still accepted
the inherited `MoiOptions` attempt. Access construction copied the complete
aggregate four times for candidate planning, descriptor accounting, appended
emission, and inline emission, and used three `scratch_vgpr` assignments to
carry bases already present in the selected resource plans. Atomic and fence
construction did not copy the aggregate, but silently obtained semantic input,
bound resources, and accepted placement state through the same inheritance.

Those three Record/Replay-owned construction facets now receive immutable
`ConSanOptions` and `ConSanMoiOperatingPoint` separately. Candidate-local access
binding copies only the operating point; scratch bases remain explicit;
resource and record-event planning name the accepted point; and request,
runtime-resource, policy, debug, and growth-limit input remains immutable. The
mode entry retains the broad mutable attempt because it owns the existing
post-placement decision to discard unconsumed automatic state, then passes its
two base products into access and event construction. No compatibility
overload, replacement view, or parallel implementation was added.

The architecture-boundary gate rejects `MoiOptions` and inherited input/point
casts in the Record/Replay internal header and its access, atomic, and fence
owners. It budgets at most one broad-attempt reference in the mode entry. This
exercises the typed mode-component boundary on a second engine and prevents
future Record/Replay event work from restoring inheritance as a shortcut. The
shared barrier owner remains a separate exact-subset/common-mechanism boundary
and is deliberately not misrepresented as Record/Replay-local by this gate.

| Signal | Checkpoint 35 | Cumulative change | Slice change from checkpoint 34 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,472 | +497 | **-3** |
| Nonblank production lines | 99,188 | +105 | **-3** |
| Production implementation lines | 91,471 | **+21** | **-3** |
| `MoiOptions` references / files | 63 / 21 | **-24 / -4** | **-10 / -4** |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 57 | +26 / +6 | **+8 / +3** |
| `MoiOptions` references in Record/Replay-owned construction | 1 | n/a | **-10** |
| Record/Replay whole-attempt candidate copies | 0 | n/a | **-4** |
| Record/Replay immutable scratch-field assignments | 0 | n/a | **-3** |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a full final-tree `-j16` rebuild; all 825 `ConSanMoi` and
architecture-boundary focused tests in 4.65 seconds; all 4,744 nonphysical
tests at `-j16` in 199.85 seconds, including the unchanged 2,908 simulator rows
over five targets; and all 635 physical gfx1201 tests serialized at `-j1` in
108.88 seconds. No test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, and 14.9:
two mode packages now use the same explicit immutable-input/resolved-state
boundary without duplicating an adapter, and the legacy broad copies and side
channel are deleted from the migrated Record/Replay surface. Production
implementation remains 21 lines above baseline, so material whole-refactoring
shrinkage is still open. InlineShadow, the exact-subset barrier owner, common
prologues, architecture peepholes, placement and validation concentrations,
the extension exercises, and the independent deep-read completion audit also
remain open. The goal therefore remains active.

### 16.37 Convergence checkpoint 36: typed InlineShadow construction boundary

After checkpoints 34 and 35, InlineShadow was the remaining MOI engine whose
mode-owned construction inherited the complete `MoiOptions` attempt. The
access owner cached an optional whole attempt per directly owned descriptor,
then copied or assigned the aggregate across planning, descriptor accounting,
appended emission, and inline emission. Atomic construction repeated the same
pattern for site planning, requirements, dense groups, dense bodies, and
ordinary bodies. Across the two owners there were eleven whole-attempt
copy/assignment sites and six assignments to the immutable `scratch_vgpr`
debug field despite every consumer already receiving the selected scratch base.

InlineShadow access and atomic construction now receive immutable
`ConSanOptions` and `ConSanMoiOperatingPoint` separately. The per-descriptor
cache stores only resolved operating points. Owner-local binding copies only
that state; resource selection, spill construction, private owner/dispatch
layout, dense routing, exact-shadow emission, and atomic-order emission name
their immutable and resolved inputs independently; and scratch remains an
explicit resource-plan value. The mode entry is the sole remaining
InlineShadow location that accepts the broad attempt before invoking these
owners and the separately owned barrier mechanism. No InlineShadow-specific
view, compatibility overload, or duplicate shared mechanism was introduced.

The architecture-boundary gate rejects `MoiOptions` and inherited input/point
casts in the InlineShadow internal header, access owner, and atomic owner, and
budgets at most one broad-attempt reference at mode entry. Together with the
Sampled and Record/Replay gates, all three MOI engine packages now obey the same
physical input/state boundary while retaining their distinct policies and
evidence models. SuperCollider does not use the MOI attempt type and is
unaffected by this particular invariant.

| Signal | Checkpoint 36 | Cumulative change | Slice change from checkpoint 35 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,453 | +478 | **-19** |
| Nonblank production lines | 99,169 | +86 | **-19** |
| Production implementation lines | 91,452 | **+2** | **-19** |
| `MoiOptions` references / files | 48 / 18 | **-39 / -7** | **-15 / -3** |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 58 | +36 / +7 | **+10 / +1** |
| `MoiOptions` references in InlineShadow-owned construction | 1 | n/a | **-15** |
| InlineShadow whole-attempt copy/assignment sites | 0 | n/a | **-11** |
| InlineShadow immutable scratch-field assignments | 0 | n/a | **-6** |
| Descriptor caches containing whole attempts | 0 | n/a | **-1** |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a full final-tree `-j16` rebuild; all 825 `ConSanMoi` and
architecture-boundary focused tests; all 4,744 nonphysical tests at `-j16` in
197.39 seconds, including the unchanged 2,908 simulator rows over five targets;
and all 635 physical gfx1201 tests serialized at `-j1` in 108.41 seconds. No
test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, and 14.9:
every MOI engine package now uses one enforced immutable-input/resolved-state
boundary, the last engine-local whole-attempt cache and copies are deleted, and
the same existing shared contracts serve all three engines. Production
implementation is only two lines above baseline, fully paying back 19 of the
previous checkpoint's remaining 21-line temporary growth, but Section 14.8
requires material shrinkage rather than merely reaching parity. The
exact-subset barrier owner, common prologues, the remaining orchestration and
placement attempts, architecture peepholes, extension exercises, and the
independent deep-read completion audit remain open. The goal therefore remains
active.

### 16.38 Convergence checkpoint 37: typed shared barrier boundary

Once the three MOI engine packages stopped inheriting the broad attempt, their
shared barrier owner became the largest remaining mode-adjacent exception. It
implements one exact-subset mechanism for Record/Replay barrier records and
InlineShadow epoch advancement: common site planning, owner-local resource
binding, private-state handling, spill preservation, direct and dense routing,
dispatcher construction, and proof publication surround two distinct evidence
bodies. The owner nevertheless accepted `MoiOptions` through its public and
private entries, copied it ten times for owners, candidates, groups, lambdas,
and the Inline barrier wrapper, and assigned `scratch_vgpr` twice even though
both bases were explicit resource products.

The barrier contract now receives immutable `ConSanOptions` and
`ConSanMoiOperatingPoint` separately. Owner-, candidate-, and group-local
binding copies only the operating point. Record/Replay emission keeps its
typed record-event plan; InlineShadow emission keeps its epoch policy; shared
routing and preservation mechanics consume the same request/state contracts.
The Inline spill-backed wrapper still suppresses its barrier-local EXEC-save
window, but does so in a point-only copy. Scratch bases remain explicit, and
the internal body-builder lambda carries only candidate state. No mechanism
was copied into either engine package and no compatibility path remains.

The architecture-boundary gate rejects `MoiOptions` and inherited input/point
casts in both the compiled barrier header and implementation. This makes the
exact-subset ownership mechanically visible: the mode entries choose the
shared barrier operation, while the barrier owner can inspect immutable engine
policy and resolved allocation state without receiving either complete mode
implementation or the mutable attempt bus.

| Signal | Checkpoint 37 | Cumulative change | Slice change from checkpoint 36 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,449 | +474 | **-4** |
| Nonblank production lines | 99,165 | +82 | **-4** |
| Production implementation lines | 91,448 | **-2** | **-4** |
| `MoiOptions` references / files | 31 / 16 | **-56 / -9** | **-17 / -2** |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 338 / 58 | +48 / +7 | **+12 / 0** |
| `MoiOptions` references in shared barrier construction | 0 | n/a | **-17** |
| Shared-barrier whole-attempt copy sites | 0 | n/a | **-10** |
| Shared-barrier immutable scratch-field assignments | 0 | n/a | **-2** |
| Record/Replay + InlineShadow barrier implementations | 1 | n/a | unchanged |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a full final-tree `-j16` rebuild; all 825 `ConSanMoi` and
architecture-boundary focused tests; all 4,744 nonphysical tests at `-j16` in
193.89 seconds, including the unchanged 2,908 simulator rows over five targets;
and all 635 physical gfx1201 tests serialized at `-j1` in 108.86 seconds. No
test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. The three MOI modes and their principal exact-subset synchronization
mechanism now share one explicit input/state architecture, and the migrated
surface contains neither a legacy broad path nor mode-local duplication.
Production implementation is two lines below baseline for the first time, but
that is parity rather than the material reduction required by Section 14.8.
Common prologues, orchestration and placement attempts, remaining architecture
peepholes, extension exercises, and the independent deep-read completion audit
remain open. The goal therefore remains active.

### 16.39 Convergence checkpoint 38: typed shared prologue boundary

The shared owner/epoch prologue was the final construction component outside
MOI orchestration that inherited the complete `MoiOptions` attempt. It owns two
related entry mechanisms: private-state initialization after private-epoch
access emission and owner/epoch initialization for persistent vector or scalar
state. Both mechanisms inspect immutable engine and runtime policy, bind the
accepted allocation to one kernel owner, derive descriptor requirements, and
emit entry transactions. Their two kernel loops nevertheless copied the whole
request-plus-operating-point aggregate before applying owner-local persistent
or transient assignments.

The public and private prologue contracts now receive `ConSanOptions` and
`ConSanMoiOperatingPoint` separately. Each kernel-local transaction copies only
the operating point before binding its owner assignment. Engine choice, owner
source, runtime sampling, dispatch reporting, and bound runtime resources stay
on the immutable input; resolved EXEC-save, owner/epoch, persistent scalar,
workgroup, spill, and dispatch-identity choices stay on the kernel-local point.
Calls to shared requirement and InlineShadow evidence helpers now name those
two authorities explicitly. No compatibility overload or replacement view was
introduced, and both whole-attempt prologue copies were deleted.

The architecture-boundary gate rejects `MoiOptions` and inherited input/point
casts in both the prologue header and implementation. Together with the prior
mode and barrier gates, every native MOI construction owner below orchestration
now exposes the same immutable-input/resolved-state split. The remaining broad
attempt references are confined to attempt construction, placement and mode
orchestration, the three mode-entry adapters, and the aggregate type itself;
they are no longer available to prologue mechanics as a convenience bus.

| Signal | Checkpoint 38 | Cumulative change | Slice change from checkpoint 37 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,449 | +474 | 0 |
| Nonblank production lines | 99,165 | +82 | 0 |
| Production implementation lines | 91,448 | **-2** | 0 |
| `MoiOptions` references / files | 26 / 14 | **-61 / -11** | **-5 / -2** |
| `ConSanTransformArtifacts` references / files | 207 / 56 | **-69 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 342 / 58 | +52 / +7 | **+4 / 0** |
| `MoiOptions` references in shared prologue construction | 0 | n/a | **-5** |
| Shared-prologue whole-attempt copy sites | 0 | n/a | **-2** |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a full final-tree `-j16` rebuild; all 825 `ConSanMoi` and
architecture-boundary focused tests in 4.6 seconds, plus the broader 874-test
`ConSanMoi*` superset in 4.8 seconds; all 4,744 nonphysical tests at `-j16` in
about 178 seconds, including the unchanged 2,908 simulator rows over five
targets; and all 635 physical gfx1201 tests serialized at `-j1` in about 102
seconds. No test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, and 14.9:
the exact-subset barrier and common entry mechanisms now obey the same enforced
construction boundary as every mode package, and no downstream owner retains a
mixed mutable attempt. It does not improve the implementation-line count, and
the still-wide operating point remains explicit rather than narrow. Broad
attempt ownership in placement/composition/orchestration, material code-size
reduction, the remaining architecture peepholes, extension exercises, and the
independent deep-read completion audit remain open. The goal therefore remains
active.

### 16.40 Convergence checkpoint 39: eliminate the production attempt aggregate

Checkpoint 38 confined `MoiOptions` to what appeared to be an orchestration
lifecycle, but a full trace showed that the lifecycle did not require an
aggregate at all. Its constructor seeded three explicit register overrides;
placement then mutated the operating-point base; composition copied the whole
object three times while changing only immutable fault or instrumentation
input; the MOI coordinator copied it once to carry effective request and
placement state together; mode dispatch passed it to three entries that had
already split it immediately; and one fallback reconstructed it solely to
mutate the operating-point base. This was residual inheritance plumbing, not a
cohesive production component.

Production now carries `ConSanOptions` and `ConSanMoiOperatingPoint` as separate
products from point construction through composition, retry, resource solving,
mode dispatch, mode-local cleanup, common barriers, prologues, and final
publication. The one conversion from caller overrides to an unresolved point
is `initial_consan_moi_operating_point`; every later change is visibly solver-
or mode-owned state. Record/Replay's legitimate post-placement cleanup mutates
only the point. Composition's fault-planning, fault-application, and
instrumentation copies contain only immutable input. Dispatch fallback copies
only its base point. The production type, all six production whole-attempt
copies/constructions, all 26 remaining references, and every inherited cast
needed by that aggregate are gone.

The trace also exposed a parallel test-seeding channel that passed an
owner-local transient assignment beside the initial point through completion,
composition, recursion, and result construction. Focused test support now puts
that assignment into its test point before entering production, so the complete
state follows the same one-product route as ordinary lowering. The inherited
`MoiOptions` spelling survives only as a test-source convenience in
`consan_test_support.h`; it delegates initial point construction to the
production authority and is absent from the declared production scope. This
preserves hundreds of readable focused fixtures without retaining a production
compatibility path.

The architecture-boundary gate now rejects `MoiOptions` in every production
`.cpp`, `.h`, and `.inc` file. That production-wide rule replaces the old three
mode-entry occurrence budgets and the emitter-only prohibition; those
superseded checks were deleted. Existing component-specific rules still reject
inherited input/point casts because they enforce a broader and still useful
boundary.

| Signal | Checkpoint 39 | Cumulative change | Slice change from checkpoint 38 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,437 | +462 | **-12** |
| Nonblank production lines | 99,155 | +72 | **-10** |
| Production implementation lines | 91,445 | **-5** | **-3** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | **-26 / -14** |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | **-1 / 0** |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 350 / 63 | +60 / +12 | **+8 / +5** |
| Production whole-attempt copies/constructions | **0** | n/a | **-6** |
| Parallel initial transient-assignment channels | **0** | n/a | **-1** |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a final-tree `-j16` rebuild; all 874 `ConSanMoi*` and
architecture-boundary focused tests in 4.9 seconds, including the 825-test core
subset; all 4,744 nonphysical tests at `-j16` in about 180 seconds, including
the unchanged 2,908 simulator rows over five targets; and all 635 physical
gfx1201 tests serialized at `-j1` in about 99 seconds. The production-wide
boundary test was rerun after removing its superseded rules and passed in 1.62
seconds. No test was removed, renamed, disabled, added, or replaced.

This checkpoint materially strengthens Sections 14.1, 14.3, 14.5, 14.6,
14.7, 14.8, and 14.9. The principal mixed request/state bus identified in the
post-fourth-refactoring audit no longer exists in production; modes and shared
mechanisms consume the same explicit products without duplicating code; a
parallel state channel and its redundant gates were harvested; and the slice
shrinks rather than wrapping the legacy design. The explicit operating point
is still union-shaped and now appears in more named boundaries, so its
remaining field/component fanout needs a fresh semantic audit rather than being
declared solved from the zero aggregate count. Material whole-refactoring
shrinkage, remaining target locality, the extension exercises, and the
independent Section 14 completion audit also remain open. The goal therefore
remains active.

### 16.41 Convergence checkpoint 40: harvest dead convergence state

The post-attempt audit traced the remaining mode-planning fields from each
producer through dispatch, coordination, emission, diagnostics, tests, and
runtime use. `MoiObjectModePlan::atomic_or_fence_relevant` was not a contract:
Record/Replay read its own value only inside the planner that computed it,
Sampled wrote it solely for tests, InlineShadow used it only to derive another
mode-local result, and no coordinator or emitter consumed it. The field is now
gone. Record/Replay keeps the predicate as a local planning fact, InlineShadow
derives its result from admitted object facts, and the focused tests assert the
actual plan outputs and warnings rather than dead observability. No mode
contract, mechanism, or behavior was duplicated.

The same definition-to-runtime trace found four further remnants. The public
`consan_atomic_classifier_reason_name` formatter had no caller; the owning
policy and inventory boundaries already render their different semantic
diagnostics. Sampled access planning assigned `runtime_sample_index` but never
read it. SuperCollider's candidate carried an `appended_cave_text_offset` that
was never read. The same audit classified `entry_island_is_appended` as dead
because direct entry-island candidates always constructed it false. That
classification was incomplete: generated-island seeds constructed it true and
later acquired an island pointer through copying. The field itself remains
unnecessary, but removing its semantic distinction made a reserved appended
island look like unreserved existing text. A subsequent clean rebuild exposed
the error, and checkpoint 42 derives the distinction from the island's actual
location instead of restoring parallel state. Finally, borrowed-entry VGPR
selection no longer constructs a private duck-typed adapter: both call paths
consume the already-declared `MoiPersistentVgprStateView`, and the operating-
point entry explicitly projects that canonical view.

This slice deliberately adds no replacement abstraction. It removes an unused
API, cross-mode telemetry, never-consumed candidate state, redundant writes,
and a parallel structural adapter from their complete production surfaces.
Final-tree searches find none of the deleted API, fields, or adapter. The
compiler also checked every aggregate construction after the candidate layout
shrank.

| Signal | Checkpoint 40 | Cumulative change | Slice change from checkpoint 39 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,367 | +392 | **-70** |
| Nonblank production lines | 99,087 | +4 | **-68** |
| Production implementation lines | 91,377 | **-73** | **-68** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 350 / 63 | +60 / +12 | 0 / 0 |
| Dead mode-plan telemetry fields | **0** | n/a | **-1** |
| Dead atomic-classifier public formatters | **0** | n/a | **-1** |
| Dead Sampled/SuperCollider candidate fields | **0** | n/a | **-3** |
| Borrowed-entry private state adapters | **0** | n/a | **-1** |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a clean final-tree `-j16` rebuild; all 1,375 affected-path
mode-planning, atomic-classifier, Sampled, and SuperCollider tests; all 4,744
nonphysical tests at `-j16`, including the unchanged 2,908 simulator rows over
five targets; and all 635 physical gfx1201 tests serialized at `-j1`. No test
was removed, renamed, disabled, added, or replaced.

The checkpoint-42 clean rebuild invalidated the SuperCollider portion of that
claim: seven far-relay tests failed deterministically until the generated-
island reservation distinction was restored without restoring the field. The
current-tree validation in checkpoint 42 supersedes this historical result.

This checkpoint strengthens Sections 14.5, 14.7, 14.8, and 14.9: named products
carry fewer facts that have no downstream semantic consumer; one canonical
state view replaces an opportunistic adapter; and the migrated production
surface is 68 implementation lines smaller without reducing behavior or test
inventory. The cumulative 73-line reduction is real but not yet material for a
91-thousand-line implementation. The operating-point field audit, transform
transaction audit, remaining target/mode interactions, and independent
Section 14 completion audit therefore remain open, and the goal remains active.

### 16.42 Convergence checkpoint 41: mode-owned operational evidence

The mode-locality audit found one remaining `N`-way decision embedded in common
MOI resource-plan orchestration. `consan_moi_pipeline.inc` selected the barrier
intent, atomic intent, fence inclusion, and scratch-sizing path with eleven
direct `ConSanMoiEngine` references. Those choices repeated facts already owned
by the three mode packages and meant that a hypothetical mode composing an
existing evidence operation would still require edits in the common solver.

Each `MoiModeOperations` registration now publishes a narrow
`MoiOperationalEvidenceKinds` value. Record/Replay selects barrier records,
atomic records, and fence records; Sampled selects sampled barrier epochs and
sampled atomic ordering; InlineShadow selects exact barrier epochs and exact
atomic ordering. The common solver consumes those semantic operation kinds.
Its only switches are now over the selected evidence operation in order to
apply that operation's shared resource contract; it contains no mode
enumerator. This distinction is important for `O(N+M)` growth: a new mode can
compose the existing operations by registration, while a genuinely new
operation adds one shared semantic resource rule rather than branches for
every mode using it.

The three production registrations were also converted from positional
aggregates to designated fields. This makes the mode extension surface
self-documenting and prevents a later contract-field insertion from silently
rebinding unrelated callbacks. The existing hypothetical fifth-mode fixture
now registers and observes its operational evidence selection without touching
any target package or common resource-planning branch. The architecture gate
replaces the pipeline's reviewed 11-occurrence mode budget with an exact zero
rule.

| Signal | Checkpoint 41 | Cumulative change | Slice change from checkpoint 40 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,395 | +420 | +28 |
| Nonblank production lines | 99,112 | +29 | +25 |
| Production implementation lines | 91,399 | **-51** | +22 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 351 / 63 | +61 / +12 | +1 / 0 |
| Direct mode references in common resource-plan orchestration | **0** | n/a | **-11** |
| Mode-local operational-evidence registrations | 3 | n/a | +3 |
| Resource-plan mode-switch gate | exact zero | n/a | strengthened from budget 11 |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a clean final-tree `-j16` rebuild; all 875 `ConSanMoi*` and
architecture-boundary focused tests, with the exact boundary gate separately
confirmed in 1.64 seconds; all 4,744 nonphysical tests at `-j16`, including the
unchanged 2,908 simulator rows over five targets; and all 635 physical gfx1201
tests serialized at `-j1`. No test was removed, renamed, disabled, added, or
replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.6, 14.7, and 14.9.
It deletes the superseded common mode switches and makes their absence an
invariant, but the explicit operation contract costs 22 implementation lines.
That is recorded temporary growth, not code-size progress; it reduces the
cumulative shrink from 73 to 51 lines and must be repaid by subsequent
convergence and harvesting. Other reviewed common mode budgets, the broad
transform transaction, material whole-refactoring shrinkage, and the
independent Section 14 completion audit remain open. The goal therefore
remains active.

### 16.43 Convergence checkpoint 42: mode-owned transient scalar ABI

The next common-mode audit found ten direct engine-enumerator references in
`consan_moi_support.cpp`. They jointly implemented three unrelated transient
scalar ABIs: each mode's complete EXEC-save window size, its dynamic-stack
frame-save offset, and Record/Replay's automatic banked-capture classification.
The common helper therefore needed editing for every new mode and mixed target
variation into the same mode switch.

Each `MoiModeOperations` registration now owns its exact EXEC-save sizing
function and dynamic-stack frame-save offset. Record/Replay owns automatic
banked capture, runtime workgroup-gate, compact spill, dense barrier-router,
and dynamic-stack widths. Sampled owns publication, atomic, compact-spill, and
dynamic-stack widths. InlineShadow owns access-present, atomic-only, and
dynamic-stack widths. Common support constructs the already-narrow
`MoiExecSaveRequirement`, resolves one `MoiExecSaveTargetFacts`, and invokes
the selected operation. The old three-way switch, repeated engine guards,
dynamic-stack count adapter, and duplicated banked-capture predicate are gone.

The target variation is deliberately not a raw architecture parameter on the
mode callback. Sampled needs to know whether its dense route has the target's
`SCallI64` return-pair form; common support projects only that normalized
`ConSanDirectCallForm` from the target profile. This preserves the exact five-
target behavior while making the extension axes additive: a new target
publishes its call form without editing Sampled, and a new mode supplies one
scalar ABI operation without adding an architecture branch or editing common
support. Focused tests cover every dynamic-stack offset and width as well as
the existing RDNA3, RDNA4, and CDNA5 ordinary widths. The architecture gate
replaces support's reviewed ten-occurrence mode budget with an exact zero rule.

The full gate also found and repaired a regression introduced by checkpoint
40's incomplete generated-island audit. Local SuperCollider NOP islands need
an exclusive existing-text reservation, whereas generated islands already
belong to a reserved appended prefix. Candidate selection now derives that
fact from the island's actual text offset; it neither restores the removed
boolean nor adds another provenance channel. Seven existing focused tests at
the SuperCollider placement boundary directly reproduced the failure and now
pass, including wide composite donors, shared gfx1250 owners, variable relay
reservoirs, CDNA4 wave64 reservoirs, relocated second-word anchors, skipped-
kernel donors, and generated-bank preplanning. The repair is checkpointed
separately as `c1949b018c7`.

| Signal | Checkpoint 42 | Cumulative change | Slice change from checkpoint 41 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,408 | +433 | +13 |
| Nonblank production lines | 99,121 | +38 | +9 |
| Production implementation lines | 91,400 | **-50** | +1 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 361 / 63 | +71 / +12 | 0 / 0[^checkpoint-42-point-count] |
| Direct mode references in common scalar-ABI support | **0** | n/a | **-10** |
| Mode-local scalar-ABI registrations | 3 | n/a | +3 |
| Common-support mode-switch gate | exact zero | n/a | strengthened from budget 10 |
| Test inventory | 5,379 | +34 | 0 |

[^checkpoint-42-point-count]: Recounting the checkpoint-41 tree finds 361
    occurrences, not the 351 recorded in its table. This row corrects that
    stale ledger value; the checkpoint-42 production slice adds none.

Validation includes a clean final-tree `-j16` rebuild; all 877 `ConSanMoi*`,
scalar-ABI, and architecture-boundary focused tests; the seven directly
affected SuperCollider far-relay regressions; all 4,744 nonphysical tests at
`-j16`, including the unchanged 2,908 simulator rows over five targets; and all
635 physical gfx1201 tests serialized at `-j1`. The two existing scalar-ABI
tests and the hypothetical-mode extension fixture gained assertions, but no
test was removed, renamed, disabled, added, or replaced, so the inventory
remains unchanged.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.5, 14.6, 14.7,
14.8, and 14.9. Common scalar support is now mode-neutral, modes consume a
narrow target fact rather than architecture identity, all superseded dispatch
logic is deleted, and a discovered legacy-harvest regression is fixed at its
owning placement boundary. The slice costs one net implementation line after
the regression repair, so it does not repay checkpoint 41's temporary growth
or satisfy material shrinkage. Other reviewed common mode budgets, the broad
operating point and transform transaction, remaining target locality, and the
independent Section 14 completion audit remain open. The goal therefore
remains active.

### 16.44 Convergence checkpoint 43: mode-owned shared-prologue policy

A definition-to-runtime audit of the shared owner/epoch prologue found four
remaining mode-enumerator decisions. They did not select different prologue
implementations: they selected whether an otherwise common prologue skips an
unobserved barrier-only owner, protects a compact spill before runtime sampling,
uses one-based owner identifiers, and requires an in-place entry when persistent
state is present. Leaving those semantics in `consan_moi_prologue.cpp` made the
common mechanism an implicit mode registry and required that file to change for
every mode with a different combination.

`MoiModeOperations` now carries one narrow `MoiPrologueModePolicy`. Record/Replay
owns the early compact-spill backup requirement, Sampled explicitly selects the
all-default policy, and InlineShadow owns the barrier-only elision, one-based
owner, and persistent-entry requirements. The common prologue reads that policy
once and still exclusively owns descriptor inspection, resource preservation,
placement, entry rewriting, and emission. No mechanism was copied into a mode
package, and the four old mode guards and their mode-specific explanation in
common code were deleted.

The hypothetical fifth-mode fixture now composes a non-default prologue policy
without naming a production mode or changing a target package. The existing
all-engine planning fixture asserts every production policy field, while the
focused behavioral set exercises the actual entry, owner, persistent-state,
runtime-gate, and scalar-backup consequences across the supported targets. The
architecture gate replaces the prologue's reviewed four-reference budget with
an exact zero rule.

| Signal | Checkpoint 43 | Cumulative change | Slice change from checkpoint 42 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,428 | +453 | +20 |
| Nonblank production lines | 99,140 | +57 | +19 |
| Production implementation lines | 91,417 | **-33** | +17 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 205 / 28 | +5 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 361 / 63 | +71 / +12 | 0 / 0 |
| Direct mode references in shared prologue construction | **0** | n/a | **-4** |
| Mode-local shared-prologue policy registrations | 3 | n/a | +3 |
| Shared-prologue mode-switch gate | exact zero | n/a | strengthened from budget 4 |
| Test inventory | 5,379 | +34 | 0 |

Validation includes a final-tree `-j16` build; all 227 focused mode-planning,
prologue, entry, owner, persistent-state, runtime-gate, scalar, and architecture-
boundary tests; all 4,744 nonphysical tests at `-j16`, including the unchanged
2,908 simulator rows over five targets; and all 635 physical gfx1201 tests
serialized at `-j1`. Existing mode-planning tests gained policy assertions, but
no test was removed, renamed, disabled, added, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.5, 14.6, 14.7, and
14.9. The shared prologue is now mode-neutral and a new mode can compose its
semantics through its own package, while the exact common mechanism remains one
implementation. The explicit contract costs 17 implementation lines, reducing
the cumulative shrink to 33 lines; this is locality progress but emphatically
not the material code-size evidence required by Section 14.8. Other reviewed
common mode budgets, broad mutable products, remaining target locality, legacy
harvesting, and the independent completion audit remain open. Section 14.4's
two extension exercises remain complete and enforced. The goal therefore
remains active.

### 16.45 Convergence checkpoint 44: operation-composed shared barriers

The next deep read distinguished mode identity from evidence-operation identity
inside the named shared barrier component. Its fifteen concrete mode references
did not describe fifteen mode-owned algorithms. Record/Replay selects
`BarrierRecord`, InlineShadow selects `ExactBarrierEpoch`, and Sampled selects
`SampledBarrierEpoch`; the first two deliberately share the same difficult
placement, relay, private-state, and relocation machinery, while Sampled owns a
different synchronization implementation. The shared component was repeatedly
rediscovering Record/Replay versus InlineShadow even though checkpoint 41 had
already made the selected operation an explicit mode-owned contract.

Shared exact-subset lowering now resolves that registered barrier operation
once. Record emission, exact-epoch update, patch classification, dense-router
eligibility, body construction, and relocated-guest bookkeeping compose from
the operation rather than a concrete engine enumerator. An incompatible
operation fails at this boundary instead of silently becoming InlineShadow.
This does not move shared mechanics into mode packages: both production modes,
and any future mode selecting either existing operation, still use the same
placement and relay implementation.

The migration also harvested two parallel mechanisms. Resource solving and
barrier application previously maintained separate barrier scratch-width
authorities, including a duplicated Record/Replay width. One
`operational_barrier_scratch_count` now defines the Record, Sampled, and exact-
shadow resource contracts for both consumers. Three copies of record-body
guest-instruction lookup versus exact-epoch offset publication are now one
typed helper. The old functions, branches, and duplicate relocation blocks are
gone rather than retained as adapters.

A new focused test exercises all three operation widths, Sampled's persistent-
state width, InlineShadow's visible-evidence width, and invalid-operation
rejection. The pre-existing hypothetical fifth-mode fixture already composes a
`BarrierRecord` operation without editing a target or shared resource switch;
the full barrier set exercises the resulting Record/Replay, Sampled, and
InlineShadow behavior over the target matrix. The architecture gate replaces
the barrier component's reviewed 17-reference budget with exact zero. It also
replaces the already-stale 20-reference top-level coordinator budget with exact
zero, preventing that forward-only coordinator from silently regaining mode
dispatch.

| Signal | Checkpoint 44 | Cumulative change | Slice change from checkpoint 43 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,409 | +434 | **-19** |
| Nonblank production lines | 99,121 | +38 | **-19** |
| Production implementation lines | 91,398 | **-52** | **-19** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | +1 / 0 |
| `ConSanMoiOperatingPoint` references / files | 360 / 63 | +70 / +12 | **-1 / 0** |
| Direct mode references in shared barrier lowering | **0** | n/a | **-15** |
| Shared barrier scratch-width authorities | **1** | n/a | **-1** |
| Repeated barrier relocation decision blocks | **1** | n/a | **-2** |
| Shared-barrier mode-switch gate | exact zero | n/a | strengthened from budget 17 |
| Top-level coordinator mode-switch gate | exact zero | n/a | strengthened from stale budget 20 |
| Test inventory | **5,380** | **+35** | **+1** |

Validation includes a final-tree `-j16` build; all 164 focused barrier, dense-
relay, operation-contract, hypothetical-mode, and architecture-boundary tests;
all 4,745 nonphysical tests at `-j16`, including the unchanged 2,908 simulator
rows over five targets; and all 635 physical gfx1201 tests serialized at `-j1`.
No test was removed, renamed, disabled, or replaced; the direct operation-
contract regression increases the inventory by one.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.5, 14.6, 14.7,
14.8, and 14.9. One shared exact-subset mechanism now composes mode-owned
semantic operations, a new mode can reuse those operations without shared or
target edits, two parallel authorities are harvested, enforcement is tighter,
coverage grows, and production shrinks. The cumulative 52-line reduction is
still not material for a 91-thousand-line implementation, however. Remaining
target locality, broad mutable products, larger deletion opportunities, and the
independent Section 14 completion audit remain open. Section 14.4's two
extension exercises remain complete and enforced. The goal therefore remains
active.

### 16.46 Convergence checkpoint 45: immutable object-mode semantics

A definition-to-runtime trace of the broad MOI operating point found two facts
that were not resource-solver choices at all. Record/Replay planning selected
whether a normalized object and target require the dense barrier router, while
InlineShadow planning selected whether the admitted object contains an ordinary
access probe. The top-level coordinator then copied both facts into
`ConSanMoiOperatingPoint`, after which common resource sizing, placement,
prologue construction, and emission read them as though they were mutable
solver results. This created a second authority-shaped representation for mode
semantics and made retries carry facts that never participate in retry
selection.

`MoiObjectModePlan` now publishes one narrow `MoiObjectModeSemantics` value.
Record/Replay is the sole production writer of `dense_barrier_router`, deriving
it from normalized target support and barrier inventory; InlineShadow is the
sole writer of `inline_access_present`, deriving it from admitted object
inventory. The coordinator selects that value once, `MoiResourceProblem`
captures it by value with the other immutable solver inputs, and common
resource and emission components receive it through const interfaces. The
operating point retains only selections that the resource/placement pipeline
can actually refine. No common component rediscovers either fact from a mode
enumerator or concrete architecture identity.

The two operating-point fields and all fourteen production references to their
old names are deleted. The architecture gate requires those names to remain
absent from every production file and rejects either semantic assignment
outside its one mode-owned planner (apart from the product's field definition).
Existing focused tests now prove that the resource problem binds the immutable
value, scalar-ABI projection consumes it separately from the operating point,
all normalized dense-router cases remain mode-owned, InlineShadow access
presence remains inventory-driven, and shared barrier sizing consumes the
selected semantics. The hypothetical fifth-mode fixture continues to publish
object semantics without target-package changes.

| Signal | Checkpoint 45 | Cumulative change | Slice change from checkpoint 44 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,475 | +500 | +66 |
| Nonblank production lines | 99,183 | +100 | +62 |
| Production implementation lines | 91,464 | **+14** | +66 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 358 / 63 | +68 / +12 | **-2 / 0** |
| Mode-semantic facts stored in the mutable operating point | **0** | n/a | **-2** |
| Production assignments of each object-mode semantic | **1 mode owner** | n/a | exact owner gate added |
| Retired operating-point semantic-name references | **0** | n/a | **-14** |
| Test inventory | **5,380** | **+35** | 0 |

Validation includes a final-tree `-j16` build; sixteen direct mode-planning,
immutable-binding, scalar-projection, and barrier-sizing regressions; the exact
architecture-boundary test; all 4,745 nonphysical tests at `-j16`; and all 635
physical gfx1201 tests serialized at `-j1`. No test was removed, renamed,
disabled, added, or replaced; existing tests were strengthened at the changed
ownership boundary.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.5, 14.6, 14.7, and
14.9. It removes duplicate mutable authority and leaves a small, explicit
mode-to-common semantic product. The explicit const threading across the
existing wide call graph costs 66 implementation lines, however, erasing the
previous cumulative 52-line reduction and leaving production fourteen lines
above the starting baseline. That is an honestly recorded migration cost, not
Section 14.8 progress, and it increases the urgency of converging or deleting
the wide plumbing that made a two-field product expensive to carry. Remaining
target locality, broad mutable products and transactions, larger legacy
harvesting, material whole-refactoring shrinkage, and the independent Section
14 completion audit remain open. Section 14.4's two extension exercises remain
complete and enforced. The goal therefore remains active.

### 16.47 Convergence checkpoint 46: mode-owned report ABI selection

Checkpoint 45 made object-mode semantics immutable but exposed why the new
product had initially cost so much: eleven lowering sites still called a
central `resolve_moi_report_layout` mode switch to rediscover the same report
ABI from the request and runtime binding. Sampled access and synchronization
lowering additionally recomputed the same barrier- and atomic-island reservation
counts at separate phases.

Each mode planner now selects its legacy raw-buffer geometry beside its other
object semantics. One small shared operation only chooses between that
mode-owned legacy geometry and revalidation of an already-bound automatic
layout; it contains no mode switch. The immutable product carries the selected
layout through the resource problem to every existing consumer, so lowering no
longer consults the request to infer an ABI. Sampled planning also selects its
two bounded island counts once from admitted semantic counts, patch policy, and
the selected layout. The two late reservation helpers and every repeated call
are deleted.

The architecture gate permits report-layout assignment only in the three mode
planners, permits Sampled reservation assignment only in the Sampled planner,
rejects the retired reservation helpers everywhere, and rejects report-layout
resolution from all lowering components. A new direct regression checks the
exact Record/Replay, Sampled, and InlineShadow legacy layouts and Sampled's
barrier/atomic reservation bounds. The hypothetical fifth-mode registration
still needs no target-package change.

| Signal | Checkpoint 46 | Cumulative change | Slice change from checkpoint 45 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,453 | +478 | **-22** |
| Nonblank production lines | 99,166 | +83 | **-17** |
| Production implementation lines | 91,446 | **-4** | **-18** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 206 / 56 | **-70 / -1** | 0 / 0 |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 358 / 63 | +68 / +12 | 0 / 0 |
| Report-layout resolution calls in lowering components | **0** | n/a | **-11** |
| Central report-layout mode switches | **0** | n/a | **-1** |
| Late Sampled island-reservation helpers | **0** | n/a | **-2** |
| Test inventory | **5,381** | **+36** | **+1** |

Validation includes a final-tree `-j16` build; fifteen direct mode-planning,
immutable-binding, and scalar-projection tests; the exact architecture-boundary
test; all 4,746 nonphysical tests at `-j16`; and all 635 physical gfx1201 tests
serialized at `-j1`. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.5, 14.6, 14.7,
14.8, and 14.9. It is the first payback harvested from checkpoint 45's explicit
immutable path: mode ABI selection is local, consumers cannot rediscover it,
legacy helpers are gone, coverage grows, and production shrinks by eighteen
implementation lines. The whole refactoring is again net-negative, but a
four-line cumulative reduction is not material. Remaining target locality,
broad mutable products and transactions, larger legacy harvesting, material
whole-refactoring shrinkage, and the independent Section 14 completion audit
remain open. Section 14.4's two extension exercises remain complete and
enforced. The goal therefore remains active.

### 16.48 Convergence checkpoint 47: immutable resource-planning input

A deep read of the remaining operating-point provenance flags found that they
encode real lifetime proofs and cannot safely be inferred from register
presence. The same trace exposed a more immediate parallel input authority in
the shared MOI resource pipeline. `MoiResourceProblem` already owned the exact
immutable request, runtime binding, target, inventory, observation plan,
candidate span, and object-mode semantics for one solving run. Resource
planning nevertheless also accepted the complete mutable
`ConSanTransformArtifacts` transaction to retrieve its inventory and
observation plan, and separately accepted the same candidate span. Its public
rebuild helper then mutated the broad transaction merely to publish a typed
planning result.

All access and operational resource planning now consume the one immutable
problem. There is no alternate transaction path and no duplicate candidate
input. Operational barrier, atomic, and fence planning read the problem's
inventory and observation plan directly. Automatic transient-scalar retries
call the same `plan_moi_resources` operation and receive a complete
`ConSanMoiResourcePlanningResult`; intermediate attempts cannot publish plans,
errors, or operating points. The coordinator is the sole publisher of the
selected typed result into its transaction. The old
`rebuild_moi_resource_plans` mutation wrapper is deleted rather than retained
as a compatibility API.

The architecture gate rejects a const transform-artifact input in both the
resource-pipeline contract and implementation and rejects the retired rebuild
entry point. No behavior defect was established in this slice, so the existing
cross-mode resource tests remain the behavioral regression. The structural
test is extended to make the newly closed input and publication boundaries
durable.

| Signal | Checkpoint 47 | Cumulative change | Slice change from checkpoint 46 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,434 | +459 | **-19** |
| Nonblank production lines | 99,148 | +65 | **-18** |
| Production implementation lines | 91,428 | **-22** | **-18** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 200 / 56 | **-76 / -1** | **-6 / 0** |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 357 / 63 | +67 / +12 | **-1 / 0** |
| Resource-planning broad-transaction references | **0** | n/a | **-6** |
| Duplicate resource-planning candidate inputs | **0** | n/a | **-1** |
| Mutable resource-plan rebuild wrappers | **0** | n/a | **-1** |
| Test inventory | **5,381** | **+36** | 0 |

Validation includes a final-tree `-j16` build; the exact architecture-boundary
test; 907 focused MOI and pipeline tests, of which 905 passed and the two
external benchmark-object cases skipped as designed; all 4,746 nonphysical
tests over the five emulated targets at `-j16`; and all 635 physical gfx1201
tests serialized at `-j1`. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.5, 14.6, 14.7, 14.8, and 14.9.
The main resource solver now has one immutable input authority and a typed
forward result, while production shrinks by another eighteen implementation
lines. The cumulative twenty-two-line reduction is still not material.
Remaining target locality, operating-point and transaction breadth, larger
legacy harvesting, material whole-refactoring shrinkage, and the independent
Section 14 completion audit remain open. Section 14.4's two extension exercises
remain complete and enforced. The goal therefore remains active.

### 16.49 Convergence checkpoint 48: plan-owned synchronization commits

The resource-planning trace continued through synchronization publication.
Atomic, fence, and barrier evidence already crossed planning and emission as
typed products, but three exported commit adapters independently rebuilt the
intent-ID set belonging to each product. Atomic and fence adapters duplicated
the same address-capture-plus-evidence pair; the barrier adapter rebuilt its
single evidence intent. Every Record/Replay, Sampled, and InlineShadow caller
therefore selected a mechanism-specific adapter even though all three
ultimately called the same private semantic-commit implementation and mutated
the same broad transaction.

Each typed evidence plan now publishes its own complete intent set. One shared
compile-time adapter accepts any such plan and forwards that set to one
intent-bound commit authority. All twelve call regions across the three modes
use this operation directly. The atomic, fence, and barrier adapter
declarations and definitions are deleted; there are no compatibility names or
mode-local copies. This is mode locality without duplication: policy owns the
intent membership, each mode owns its byte mechanism, and common
infrastructure owns exactly one commit construction path.

The architecture gate rejects all three retired adapter names and requires the
shared contract to obtain IDs from `plan.intent_ids()`. No behavior defect was
established during the convergence. Existing synchronization coverage tests
exercise exact intent closure, coalescing, rejection, and publication for all
three evidence-plan shapes; the structural extension prevents the duplicated
authorities from returning.

| Signal | Checkpoint 48 | Cumulative change | Slice change from checkpoint 47 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,413 | +438 | **-21** |
| Nonblank production lines | 99,131 | +48 | **-17** |
| Production implementation lines | 91,411 | **-39** | **-17** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 196 / 56 | **-80 / -1** | **-4 / 0** |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 357 / 63 | +67 / +12 | 0 / 0 |
| Mechanism-specific synchronization commit adapters | **0** | n/a | **-3** |
| Shared plan-owned synchronization commit authorities | **1** | n/a | converged |
| Test inventory | **5,381** | **+36** | 0 |

Validation includes a final-tree `-j16` build; the exact architecture-boundary
test; 926 focused MOI, evidence, and observation tests, of which 924 passed and
the two external benchmark-object cases skipped as designed; and all 4,746
nonphysical tests over the five emulated targets at `-j16`. No test was
removed, renamed, disabled, or replaced. The physical byte-emission path did
not change in this host-side semantic-ownership slice, so the immediately
preceding checkpoint's complete 635-test serialized gfx1201 result remains the
periodic physical baseline rather than being repeated here.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. It deletes another duplicated cross-mode mechanism and brings the
cumulative implementation reduction to thirty-nine lines. That is still not
material. Remaining target locality, operating-point and transaction breadth,
larger legacy harvesting, material whole-refactoring shrinkage, and the
independent Section 14 completion audit remain open. Section 14.4's two
extension exercises remain complete and enforced. The goal therefore remains
active.

### 16.50 Convergence checkpoint 49: shared instrumented-patch geometry

Following the synchronization commit path into its neighboring access paths
found three independent owners of the same transformation fact. MOI
synchronization, MOI access, and SuperCollider access each reconstructed
semantic anchor and trampoline locations from a committed patch geometry.
Their implementations differed only in where intent identity came from and in
the extra runtime attribution carried by MOI accesses. The copies crossed mode
boundaries, and the SuperCollider copy accepted the complete mutable transform
transaction even though it needed only the observation plan.

Transformation placement now owns one
`make_consan_instrumented_patch_lowering` operation. It resolves plan-owned
intent identity, deduplicates original physical sites, converts one committed
patch geometry into the corresponding semantic anchor/trampoline locations,
and delegates validation to the existing semantic-commit authority. MOI
synchronization, all three MOI access engines, and SuperCollider flat and LDS
lowering use that operation. MOI access adds its mode-owned runtime attribution
without rebuilding geometry; SuperCollider passes the narrow immutable
observation plan and retains its single-physical-site precondition. The local
MOI synchronization builder and the other two location reconstruction loops
are deleted rather than wrapped.

The architecture gate rejects local committed-location vectors and the retired
MOI synchronization builder in all three former clients, and requires the
shared construction to remain in the transformation-placement owner. A direct
regression proves that two intents coalesced at one physical site produce
exactly one anchor/trampoline pair and that a stale plan-local intent ID is
rejected. The existing malformed-patch behavior remains fail-closed: MOI and
SuperCollider access commits reject a zero-sized original patch, and
SuperCollider rejects an empty or cross-site intent set.

| Signal | Checkpoint 49 | Cumulative change | Slice change from checkpoint 48 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,395 | +420 | **-18** |
| Nonblank production lines | 99,112 | +29 | **-19** |
| Production implementation lines | 91,392 | **-58** | **-19** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 195 / 56 | **-81 / -1** | **-1 / 0** |
| `ConSanPatchInfo` references / files | 206 / 28 | +6 / 0 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 357 / 63 | +67 / +12 | 0 / 0 |
| Client-local patch-geometry location builders | **0** | n/a | **-3** |
| Shared instrumented-patch geometry authorities | **1** | n/a | converged |
| Test inventory | **5,382** | **+37** | **+1** |

Validation includes a final-tree `-j16` build; the exact architecture-boundary
test; all 1,492 focused MOI, SuperCollider, observation, and five-target
emulation tests; all 4,747 nonphysical tests over the five emulated targets at
`-j16`; and all 635 physical gfx1201 tests serialized at `-j1`. No test was
removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. One transformation-level fact now has one owner, mode clients retain only
their distinct attribution and applicability rules, a broad transaction input
is narrowed, coverage grows, and production shrinks. The cumulative
fifty-eight-line implementation reduction is still not material. Remaining
target locality, operating-point and transaction breadth, larger legacy
harvesting, material whole-refactoring shrinkage, and the independent Section
14 completion audit remain open. Section 14.4's two extension exercises remain
complete and enforced. The goal therefore remains active.

### 16.51 Convergence checkpoint 50: narrow read-only transformation boundaries

A deep read of the remaining broad transformation-transaction inputs separated
mutation ownership from read-only proof consumption. Synchronization analysis,
descriptor mutation, existing-patch reservation, SuperCollider selection, and
fault-candidate queries accepted the complete `ConSanTransformArtifacts`
transaction even though they read only one immutable product. MOI candidate
image selection was a free helper that also inspected the transaction, while
three rejection paths independently constructed empty-location semantic
commits before publishing them to the coverage ledger.

Read-only component boundaries now consume the exact product they require:
`ProgramInventory`, `ConSanObservationPlan`, `ConSanCoverageLedger`, or a span
of committed `ConSanPatchInfo` proof. The explicit patch-proof spans account
for the small increase in `ConSanPatchInfo` references; they replace a much
broader and less informative dependency. The transformation transaction itself
owns selection of its current candidate image, and remains visible only where
mutation and rollback are actually coordinated. The coverage ledger now owns
construction and publication of non-instrumented outcomes through
`publish_lowering_rejection`; MOI terminal rejection, SuperCollider rejection,
and runtime-binding rollback no longer manufacture semantic commits locally.
Raw semantic-commit construction is confined to the access-policy authority
and the shared placement adapter that converts committed patch geometry into
instrumented proof.

The architecture gate rejects const broad-transaction inputs from the migrated
read-only clients, rejects the retired free candidate-image helper everywhere,
requires descriptor mutation to consume program inventory and patch
reservation to consume patch proof, and rejects raw semantic-commit
construction outside its two declared owners. The existing direct coverage
test now exercises the ledger-owned rejection operation, including rejection
of an invalid reason/outcome pairing. No behavior defect was established in
this slice.

| Signal | Checkpoint 50 | Cumulative change | Slice change from checkpoint 49 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,395 | +420 | 0 |
| Nonblank production lines | 99,112 | +29 | 0 |
| Production implementation lines | 91,391 | **-59** | **-1** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | **-20 / -4** |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | +4 / +2 |
| `ConSanMoiOperatingPoint` references / files | 357 / 63 | +67 / +12 | 0 / 0 |
| Raw semantic-commit construction owners | **2** | n/a | converged |
| Retired free candidate-image helpers | **0** | n/a | **-1** |
| Test inventory | **5,382** | **+37** | 0 |

Validation includes a final-tree `-j16` build; 36 focused boundary, coverage,
and pipeline tests; the exact architecture-boundary test; and all 4,747
nonphysical tests over the five emulated targets at `-j16`. No test was
removed, renamed, disabled, or replaced. This host-side ownership slice does
not change emitted bytes; checkpoint 49's immediately preceding complete 635-
test serialized gfx1201 result remains the periodic physical baseline.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, and 14.9.
The broad transaction surface contracted materially, but the one-line slice
reduction and cumulative fifty-nine-line implementation reduction are not yet
material code-size convergence. Section 14.4's mode and target extension
fixtures remain complete and enforced since checkpoint 17. Remaining target
locality, operating-point and mutable-transaction breadth, larger legacy
harvesting, material whole-refactoring shrinkage, and the independent Section
14 completion audit remain open. The goal therefore remains active.

### 16.52 Convergence checkpoint 51: one persistent-VGPR state projection

The operating-point deep read followed persistent vector state through
resource exclusion, scratch-extent calculation, dispatch-layout validation,
and borrowed-entry planning. The same semantic tuple was represented in two
allocation scopes: one code-object-wide operating point and per-owner
`ConSanMoiPersistentVgprAssignment` records. Four consumers independently
enumerated owner/epoch, InlineShadow's compact workgroup key, the hardware
dispatch pair, and Record/Replay/Sampled's exact workgroup coordinates. Two
borrowed-entry adapters then distinguished the source representation even
though the algorithm consumed the same state.

`MoiPersistentVgprStateView` is now the one target-neutral projection for both
allocation scopes. It retains the semantic distinction between the partial
site-local owner/epoch sources, the one-register workgroup key, the two-register
dispatch identity, and the exact workgroup tuple, while exposing one width-
aware traversal. Register exclusion, owner-assignment exclusion, required-VGPR
extent, dispatch-overlap validation, and both Record/Replay and shared-barrier
borrowed-entry planning consume that projection. The two source-specific
borrowed-entry adapters and all four hand-written field traversals are deleted.
Mode-local state is not duplicated: InlineShadow and Record/Replay/Sampled
contribute their distinct fields to one common persistent-register mechanism.

The architecture gate rejects the retired source-specific adapters and
requires both allocation-scope projections and their common traversal. A new
direct regression proves that code-object-wide and per-owner allocations
produce exactly the same register/width sequence and that self-validation can
exclude the dispatch pair without losing any other persistent range. No
behavior defect was established in this slice.

| Signal | Checkpoint 51 | Cumulative change | Slice change from checkpoint 50 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,354 | +379 | **-41** |
| Nonblank production lines | 99,069 | **-14** | **-43** |
| Production implementation lines | 91,348 | **-102** | **-43** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | **-1 / 0** |
| Hand-written persistent-VGPR field traversals in placement | **0** | n/a | **-4** |
| Source-specific borrowed-entry adapters | **0** | n/a | **-2** |
| Test inventory | **5,383** | **+38** | **+1** |

Validation includes a final-tree `-j16` build; five direct projection,
allocation, dispatch, and architecture-boundary tests; all 909 focused MOI and
pipeline tests; all 4,748 nonphysical tests over the five emulated targets at
`-j16`; and all 635 physical gfx1201 tests serialized at `-j1`. No test was
removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. One resource fact now has one representation-independent consumer
contract, the broad operating-point type loses a consumer, and the slice
harvests 43 implementation lines. The cumulative 102-line reduction is real
but is only about one tenth of one percent of the starting implementation, so
it is not yet material Section 14.8 evidence. Section 14.4's extension fixtures
remain complete and enforced. Remaining target locality, other operating-point
and mutable-transaction breadth, larger legacy harvesting, material whole-
refactoring shrinkage, and the independent Section 14 completion audit remain
open. The goal therefore remains active.

### 16.53 Convergence checkpoint 52: persistent-SGPR state-owned ranges

The persistent-scalar-state deep read followed the accepted ordinary-SGPR ABI
through placement exclusion, architectural validation, dynamic-stack
bootstrap validation, descriptor sizing, entry-prologue backup validation,
prologue descriptor growth, and Sampled synchronization routing. Those
consumers independently enumerated owner, epoch, InlineShadow's compact
workgroup key, and Record/Replay/Sampled's exact workgroup tuple. The repeated
enumerations were already drifting: `note_moi_sgpr_requirements` stopped after
the compact key and omitted every exact-workgroup register. The later prologue
pass happened to grow the descriptor again, masking the incomplete earlier
contract in end-to-end transformations and making correctness depend on a
second authority repairing it.

`ConSanMoiPersistentSgprState` now owns one width-aware `for_each_range`
projection over its complete ABI. Placement-window exclusion, ordinary-state
validation, dynamic-stack bootstrap exclusion, descriptor sizing, prologue
backup validation, prologue descriptor growth, and Sampled synchronization
route exclusion all consume it. The seven hand-written traversals are deleted
rather than retained behind adapters. This common mechanism does not merge
mode semantics: InlineShadow still selects only its compact key, while
Record/Replay and Sampled still select their exact tuple; the shared state
owner merely makes every selected persistent range visible to generic register
mechanisms.

A direct regression constructs a descriptor whose exact tuple extends beyond
every other persistent scalar and proves that the descriptor-requirement
authority includes the final register. It failed against the pre-checkpoint
helper, which reported only the compact-key extent. The architecture gate
requires the state-owned traversal and rejects direct exact-tuple enumeration
from the three migrated production consumers, preventing a partial local
enumeration from returning.

| Signal | Checkpoint 52 | Cumulative change | Slice change from checkpoint 51 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,297 | +322 | **-57** |
| Nonblank production lines | 99,011 | **-72** | **-58** |
| Production implementation lines | 91,290 | **-160** | **-58** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Hand-written complete persistent-SGPR traversals in consumers | **0** | n/a | **-7** |
| Persistent-SGPR range authorities | **1** | n/a | converged |
| Test inventory | **5,384** | **+39** | **+1** |

Validation includes a final-tree `-j16` build; seven direct descriptor,
workgroup, Sampled, and architecture-boundary tests; all 909 focused MOI and
pipeline tests; all 4,749 nonphysical tests over the five emulated targets at
`-j16`; and all 635 physical gfx1201 tests serialized at `-j1`. No test was
removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8,
14.9, and 14.10. One complete persistent-register fact now has one owner, the
partial descriptor contract is fixed with a regression at that boundary, and
the superseded traversals are harvested for a 58-line implementation
reduction. The cumulative 160-line reduction is measurable but remains too
small to establish material whole-refactoring shrinkage. Section 14.4's
extension fixtures remain complete and enforced. Remaining target locality,
other operating-point and mutable-transaction breadth, larger legacy
harvesting, material Section 14.8 evidence, and the independent Section 14
completion audit remain open. The goal therefore remains active.

### 16.54 Convergence checkpoint 53: target-normalized acquire ordering

The synchronization-analysis deep read followed compiler-generated
workgroup-acquire loads from raw target decoding through ordinary-memory
inventory and sequence association. Program analysis already normalized most
VFLAT fields, but common synchronization analysis still interpreted the raw
ordering representation itself: gfx942/gfx950 used `SC0`, gfx1100 used `GLC`,
gfx1201 used explicit workgroup `SCOPE`, and gfx1250 used its implicit
group-aperture scope. This was one four-family target switch in a supposedly
target-neutral semantic stage. Adding a sixth target with a different spelling
would have required editing that common stage even if its decoder already
understood the instruction.

The program-analysis target contract now publishes
`workgroup_acquire_ordering`. The shared pre-gfx12 family decoder supplies the
common exact `TH == 1` rule for gfx942, gfx950, and gfx1100; the gfx1201 and
gfx1250 packages supply their distinct explicit-scope values to the shared
gfx12 raw decoder. Ordinary-memory inventory carries that normalized fact.
Synchronization analysis still owns the semantic requirements that the event
is one exact group-FLAT load in the correct container, but it no longer knows
how any target encodes the acquire qualifier. The raw `TH` and `SCOPE` fields
remain available for diagnostics and mutation proof rather than serving as a
second semantic authority.

A direct regression constructs the exact acquire spelling for all five
targets, proves that each target decoder publishes the normalized fact, and
proves that gfx1201's and gfx1250's superficially opposite scope values are not
interchangeable. Existing end-to-end ordinary-acquire tests then prove that the
fact survives inventory and drives Record/Replay, fault, cache-association,
and dense-router behavior. The architecture gate rejects raw site-level
`TH`/`SCOPE` interpretation and the retired gfx9/gfx12 family predicates from
common synchronization analysis, and requires the normalized target contract.

| Signal | Checkpoint 53 | Cumulative change | Slice change from checkpoint 52 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,300 | +325 | +3 |
| Nonblank production lines | 99,014 | **-69** | +3 |
| Production implementation lines | 91,289 | **-161** | **-1** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Target-family acquire-encoding alternatives in synchronization analysis | **0** | n/a | **-4** |
| Target-normalized acquire-ordering authorities | **1 contract** | n/a | converged |
| Test inventory | **5,385** | **+40** | **+1** |

Validation includes a final-tree `-j16` build; 19 focused five-target decode,
ordinary-acquire, fault, Record/Replay, policy, and architecture-boundary
tests; and all 4,750 nonphysical tests over the five emulated targets at
`-j16`. No test was removed, renamed, disabled, or replaced. The transformation
for gfx1201 is byte-for-byte governed by the same scope predicate now supplied
by its target decoder; checkpoint 52's immediately preceding complete 635-test
serialized gfx1201 run remains the periodic physical baseline.

This checkpoint strengthens Sections 14.1, 14.2, 14.4, 14.5, 14.6, 14.7,
14.8, and 14.9. One concrete target distinction is now physically owned by
target/family decoder files and common semantic code consumes one forward
fact. The slice deletes the legacy common interpretations but is only a
one-line implementation reduction after paying for the explicit contract, so
material shrinkage remains unproved. Remaining synchronization target
peepholes, other target locality, operating-point and mutable-transaction
breadth, larger legacy harvesting, material Section 14.8 evidence, and the
independent completion audit remain open. The goal therefore remains active.

### 16.55 Convergence checkpoint 54: target-normalized cache operations

The synchronization deep read continued from acquire-ordering fields through
cache-operation discovery, atomic and ordinary-memory association, fence
qualification, and exact fault selection. Common semantic code repeatedly
recognized `global_wb`/`global_inv`, CDNA's buffer operations, gfx1100's
ordered two-invalidate sequence, and scalar-cache invalidation by mnemonic.
Atomic association additionally asked whether the target used gfx11 encoding
before admitting the pair. Ordinary acquire mutation selected the identically
spelled gfx1201 operation in common code, even though gfx1250 decodes the same
mnemonic but deliberately exposes synchronization-only ordinary memory. Thus
one target addition or cache-vocabulary change crossed program analysis,
association, metadata policy, fence selection, and fault selection.

The program-analysis target facet now owns a `ConSanCacheOperation` vocabulary:
release, complete acquire, acquire-pair prefix, acquire-pair completion, and
unsupported. The gfx942/gfx950, gfx1100, gfx1201, and gfx1250 target/family
packages translate only their native mnemonics into that vocabulary. Program
analysis records the result on each fence site, synchronization inventory
transports it, and every semantic consumer reads the normalized operation.
The exact pair matcher consequently proves prefix, completion, ordering, and
wait-only adjacency without naming gfx1100 or consulting an architecture
predicate. The old architecture-independent acceptance of the nonexistent
`buffer_wb` spelling is deleted.

Ordinary-acquire mutation support remains deliberately narrower than semantic
acquire support. The target result carries an independent mutation-supported
fact: gfx1201's removable acquire publishes it, while gfx1250's same-spelled
operation does not. Fault selection consumes that fact rather than treating a
mnemonic as lowering authorization. Physical-alias equivalence also includes
both normalized fields, so aliases cannot silently disagree about cache
semantics or mutation capability.

A direct five-target regression proves every supported native vocabulary,
the gfx1100 pair roles, the gfx1201/gfx1250 mutation distinction, and negative
cross-target spellings. Existing exact-pair tests prove that a missing,
reversed, or intervened pair remains unassociated; the ordinary pair inventory
test now also proves that both normalized roles survive into synchronization
events. Metadata regressions construct typed operations and prove that changing
a diagnostic mnemonic cannot change semantics. The architecture gate rejects
all native cache spellings and the gfx11 encoding predicate from shared
synchronization analysis, metadata, and fault selection, and requires the
target normalization operation.

| Signal | Checkpoint 54 | Cumulative change | Slice change from checkpoint 53 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,407 | +432 | +107 |
| Nonblank production lines | 99,112 | +29 | +98 |
| Production implementation lines | 91,375 | **-75** | +86 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Native cache-mnemonic predicates in semantic consumers | **0** | n/a | converged |
| Architecture predicates in cache-pair association | **0** | n/a | **-2** |
| Target-normalized cache-operation authorities | **1 contract** | n/a | converged |
| Test inventory | **5,386** | **+41** | **+1** |

Validation includes a final-tree `-j16` build; five direct vocabulary, exact-
pair, metadata, and architecture-boundary tests; 481 broad program-analysis
and atomic/fence-policy unit tests; all 4,751 nonphysical tests over the five
emulated targets at `-j16`; and all 635 physical gfx1201 tests serialized at
`-j1`. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.2, 14.4, 14.5, 14.6, 14.7,
14.9, and 14.10. A future cache vocabulary now extends through one target
facet rather than edits to four common semantic regions, and the legacy
parallel predicates are deleted. The explicit contract, transport fields,
alias proof, and mutation distinction cost 86 implementation lines, however;
the cumulative reduction retreats from 161 to 75 lines. This is bounded
target-locality investment, not Section 14.8 progress, and the next convergence
work must harvest enough superseded structure to repay it rather than adding a
second layer. Remaining target locality, operating-point and mutable-
transaction breadth, larger legacy harvesting, material shrinkage, and the
independent Section 14 completion audit remain open. The goal therefore
remains active.

### 16.56 Convergence checkpoint 55: target-normalized wait effects

The synchronization deep read continued through compiler wait suffixes. Common
synchronization analysis independently reconstructed exact target words for
workgroup release, workgroup acquire, and atomic/ordinary release. Those
reconstructions encoded materially different contracts: gfx942/gfx950 have
exact combined VM/LDS drains but no standalone compiler-release wait, gfx1100
uses `s_waitcnt_vscnt 0` as its standalone release boundary, and gfx1201 and
gfx1250 use exact store-count or store/DS-count waits. The release scanner also
had to distinguish a nonzero native counter spelling, which invalidates the
bounded suffix, from unrelated scalar bookkeeping, which may be crossed.
Fault application then reconstructed the gfx12 release words three more times
while discovering, validating, naming, and removing the selected boundary.

The program-analysis target facet now publishes one
`ConSanWaitInstructionEncoding` result. Its target-neutral effects say whether
an exact instruction drains loads, stores, or LDS, whether it is a standalone
release boundary, and whether its native counter form bounds the release scan.
The gfx942/gfx950 family package supplies its combined wait behavior, the
gfx1100 package owns the `vscnt` form, and the shared gfx12 package owns the
four split-counter spellings and exact store-release behavior. Common
synchronization analysis consequently expresses only the semantic
requirements: a workgroup release must include an LDS drain, a workgroup
acquire must obtain both load and LDS drains, and a compiler release suffix
must contain an exact release boundary without crossing a nonzero bounded
counter. Neither synchronization analysis nor fault application names a
native wait mnemonic or builds a target-native wait word.

Atomic fault dry-run planning and application previously repeated the same
sequence-confidence, edge, fence, and release-wait selection. One
`AtomicOrderBoundary` resolver now owns that decision for both phases. The
application path has one validation/removal flow for a wait-only boundary and
a cache/wait boundary instead of two copies, while retaining separate patch
proof for each removed instruction. No behavior defect was established, but
the convergence removes the possibility that planning and application drift
on whether a CDNA wait is an independently removable release boundary.

The forward contract also made related legacy wait structure unnecessary. The
broad `ConSanWaitCounterFamily` profile fact had only one production consumer:
a SuperCollider flat-completion branch. It and all twelve production
references are deleted; an exact SuperCollider target operation now returns
the required guest-flat completion wait. Four load/store and global/flat wait
builders share one two-axis implementation. The two byte-for-byte duplicate
`flat_*_lds0` aliases and the unused generic `s_wait_alu_va_sdst0` alias are
deleted, and their callers use the surviving semantic operations. This is the
required harvesting half of the slice rather than leaving a new normalization
layer beside the old profile fact and aliases.

A direct five-target regression proves combined drains on gfx942/gfx950,
gfx1100's exact and nonzero `vscnt` forms, both gfx12 release forms, the
gfx1201/gfx1250 agreement, and the absence of a standalone CDNA release
boundary. Existing end-to-end regressions prove workgroup acquire/release,
ordinary and atomic compiler-release association, exact fault removal across
an intervening wait-ALU instruction, CDNA fault behavior, and SuperCollider
flat completion on every target. The architecture gate requires the normalized
wait operation, rejects native wait recognition and builder reconstruction
from common synchronization/fault consumers, rejects restoration of the broad
profile fact, keeps the SuperCollider branch behind its exact target
operation, and prohibits the three retired builder aliases.

| Signal | Checkpoint 55 | Cumulative change | Slice change from checkpoint 54 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,318 | +343 | **-89** |
| Nonblank production lines | 99,017 | **-66** | **-95** |
| Production implementation lines | 91,289 | **-161** | **-86** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Native wait recipes in shared synchronization/fault consumers | **0** | n/a | converged |
| Broad wait-counter-family references | **0** | n/a | **-12** |
| Superseded duplicate wait-builder references | **0** | n/a | **-5** |
| Target-normalized wait-effect authorities | **1 contract** | n/a | converged |
| Test inventory | **5,387** | **+42** | **+1** |

Validation includes a final-tree `-j16` build; 72 focused five-target wait,
synchronization, atomic-fault, capability, builder, SuperCollider, and
architecture-boundary tests; all 4,752 nonphysical tests over the five
emulated targets at `-j16`; and all 635 physical gfx1201 tests serialized at
`-j1`. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.2, 14.4, 14.5, 14.6, 14.7,
14.8, 14.9, and 14.10. Wait meaning now crosses one target-owned typed
boundary, a future target does not require edits to the common association or
fault algorithms, and the slice deletes exactly the 86 implementation lines
added by checkpoint 54. The cumulative implementation reduction is restored
to 161 lines, but that remains far short of material whole-refactoring
shrinkage. Remaining target locality, broad operating-point and mutable-
transaction surfaces, larger legacy harvesting, material Section 14.8
evidence, and the independent Section 14 completion audit remain open. The
goal therefore remains active.

### 16.57 Convergence checkpoint 56: site-local private workgroup binding

The operating-point deep read followed persistent workgroup identity from
entry-state allocation through Record/Replay access and barrier planning,
Sampled access, barrier and atomic planning, and final emission. Scalar and
VGPR workgroup tuples are code-object-wide representation choices selected by
the resource solver. The private tuple is different: its offsets belong to an
owner-local private layout and can vary by site. Nevertheless, ten planning
and application paths copied that site-local tuple into
`ConSanMoiOperatingPoint`, making a broad accepted-allocation product also act
as a temporary site-emission bus. Capture detection and source resolution then
could not state whether private offsets were an accepted global choice or a
binding for the current patch.

The private tuple is now absent from `ConSanMoiOperatingPoint`. Private-layout
and planned-patch products remain its owners, and the narrow Record/Replay and
Sampled planning/emission boundaries accept an explicit optional binding.
Code-object-wide capture checks therefore inspect only scalar and VGPR
choices; a site that selected private state supplies its exact offsets when it
validates or resolves emission sources. Ambiguity checking still rejects a
private binding combined with a scalar or VGPR representation, but that proof
no longer requires mutating or cloning the accepted operating point.

The same trace exposed three independent assemblies of effective Sampled
synchronization state for direct barriers, dense barriers, and atomics. One
owner-local `SampledSyncProbeState` resolver now applies the selected transient
assignment, selects scratch-materialized owner/epoch sources for scalar or
private entry state, or applies the owner-local persistent VGPR assignment.
All three paths consume that complete result. Direct Sampled access emission
also resolves the exact workgroup-source product once per patched access and
passes it to every address-range emitter instead of repeating descriptor or
persistent-source resolution per range.

A direct regression proves that private capture is invisible without an
explicit site binding, becomes complete with the binding, and is rejected as
ambiguous beside a code-object-wide VGPR tuple. Existing private-state,
Record/Replay access/barrier, Sampled access/barrier/atomic, dense-route, and
five-target tests exercise every migrated caller. The architecture gate
rejects restoration of the deleted operating-point field across all
production files.

| Signal | Checkpoint 56 | Cumulative change | Slice change from checkpoint 55 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,323 | +348 | +5 |
| Nonblank production lines | 99,021 | **-62** | +4 |
| Production implementation lines | 91,290 | **-160** | +1 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Site-local private workgroup fields in the operating point | **0** | n/a | **-1** |
| Independent Sampled synchronization state-resolution paths | **1** | n/a | **-2** |
| Per-range Sampled workgroup-source resolution | **0** | n/a | converged |
| Test inventory | **5,388** | **+43** | **+1** |

Validation includes a final-tree `-j16` build; 143 focused private-state,
Record/Replay, Sampled, dense-route, and architecture-boundary tests; all
4,753 nonphysical tests over the five emulated targets at `-j16`; and all 635
physical gfx1201 tests serialized at `-j1`. No test was removed, renamed,
disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.5, 14.6, 14.7,
14.9, and 14.10. A site-local representation no longer travels through the
code-object-wide solver result, and three mode-local state assemblies converge
on one forward product. The complete migration and sharing are effectively
size-neutral at one added implementation line, however. They do not advance
the material shrinkage requirement in Section 14.8, and the cumulative
reduction remains only 160 lines. Remaining target locality, broader
operating-point and mutable-transaction surfaces, larger legacy harvesting,
material Section 14.8 evidence, and the independent Section 14 completion
audit remain open. The goal therefore remains active.

### 16.58 Convergence checkpoint 57: single sampling-configuration authority

The forward-pipeline deep read followed static and runtime sampling parameters
from request construction through configuration validation and into
Record/Replay and Sampled access lowering. The typed configuration contract
already rejects a zero static or runtime stride, a runtime stride that is not
a power of two or exceeds `2^24`, and either offset outside its stride. The
pipeline records the typed issue and blocks inventory, resource solving, and
lowering. Despite that boundary, both access-mode implementations repeated
parts of the same validation, with different mode-local warning strings and
slightly different predicates.

Those unreachable post-contract checks are deleted. Mode implementations now
consume an admitted sampling configuration and contain only mode behavior:
candidate selection, bank sizing, runtime gating, placement, and emission.
The request contract remains the single validity authority. The architecture
gate rejects zero-stride or out-of-range-offset validation in both mode
implementation owners so a later change cannot silently recreate parallel
configuration semantics.

| Signal | Checkpoint 57 | Cumulative change | Slice change from checkpoint 56 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,299 | +324 | **-24** |
| Nonblank production lines | 98,997 | **-86** | **-24** |
| Production implementation lines | 91,266 | **-184** | **-24** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Sampling-validity authorities | **1 typed configuration contract** | n/a | converged |
| Post-contract sampling-validity branches in mode implementation | **0** | n/a | **-5** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 24 focused request-contract,
pipeline-stop, five-target strided Record/Replay and Sampled, and architecture-
boundary tests; and a clean final run of all 4,753 nonphysical tests over the
five emulated targets at `-j16`. One unrelated gfx1250 SuperCollider simulator
case failed in the first parallel run and passed immediately in 0.48 seconds
when rerun serially; the complete repeated `-j16` gate passed. Checkpoint 56's
immediately preceding complete 635-test serialized gfx1201 run remains the
periodic physical baseline. No test was removed, renamed, disabled, or
replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, 14.9, and
14.10. It restores a forward-only configuration boundary and harvests 24
implementation lines without adding a replacement abstraction. The cumulative
reduction reaches 184 lines, which is still not material whole-refactoring
shrinkage. Remaining target locality, broader operating-point and mutable-
transaction surfaces, larger legacy harvesting, material Section 14.8
evidence, and the independent Section 14 completion audit remain open. The
goal therefore remains active.

### 16.59 Convergence checkpoint 58: typed private-state demand and shared layout cache

The private-state deep read continued from site-local binding into layout
construction. Record/Replay, Sampled, and Inline all use the same address-free
private layout mechanism, but its contract took four positional booleans.
Their meaning was visible only through comments at each call. Shared lowering
also contained a Sampled-named wrapper that selected one particular boolean
combination, while the actual Sampled component remained only an indirect
consumer of that mode choice.

The three access paths also independently implemented the same cache:
single-owner layouts, including failed resolutions, were memoized by kernel
descriptor; multi-owner layouts were resolved independently. Record/Replay
and Sampled spelled that rule against the resource-owner list, while Inline
spelled it against its already resolved direct descriptor. These were three
copies of mechanism-neutral infrastructure embedded in mode implementation.

`MoiPrivateStateDemand` now names the four independently meaningful values:
owner, workgroup key, exact Record/Replay workgroup tuple, and dispatch
identity. Every mode declares its exact demand at its own call site. The
Sampled-specific wrapper is deleted from shared lowering. One
`MoiPrivateEpochLayoutCache` owns descriptor-local success/failure reuse and
uncached multi-owner resolution. Its key includes both descriptor and demand,
so the shared mechanism remains correct if a future mode or site requests two
different layouts for one kernel. Record/Replay, Sampled, and Inline retain
their distinct cache-key decisions but no longer own map mutation or layout
construction policy.

The architecture gate requires the typed demand and shared cache contracts and
rejects restoration of the Sampled-named shared wrapper. Existing tests cover
private owner/epoch, workgroup tuple, workgroup key, private dispatch identity,
single- and multi-owner layouts, dynamic-stack rejection, mixed private/VGPR
components, and every affected access, barrier, and atomic path.

| Signal | Checkpoint 58 | Cumulative change | Slice change from checkpoint 57 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,285 | +310 | **-14** |
| Nonblank production lines | 98,980 | **-103** | **-17** |
| Production implementation lines | 91,251 | **-199** | **-15** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Positional private-layout demand flags | **0** | n/a | **-4** |
| Independent descriptor-layout caches | **1 shared mechanism** | n/a | **-2** |
| Sampled-named helpers in shared private-layout infrastructure | **0** | n/a | **-1** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 206 focused private-state,
Record/Replay, Sampled, Inline, dense-route, and architecture-boundary tests;
all 4,753 nonphysical tests over the five emulated targets at `-j16`; and all
635 physical gfx1201 tests serialized at `-j1`. No test was removed, renamed,
disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.4, 14.5, 14.6, 14.7,
14.8, 14.9, and 14.10. Mode-specific demand is explicit and local, while the
mechanism-neutral cache has one owner and the superseded copies and wrapper are
deleted. The slice reduces production, but the cumulative 199-line reduction
is still not material whole-refactoring shrinkage. Remaining target locality,
broader operating-point and mutable-transaction surfaces, larger legacy
harvesting, material Section 14.8 evidence, and the independent Section 14
completion audit remain open. The goal therefore remains active.

### 16.60 Convergence checkpoint 59: target-neutral universal operand ABI

The architecture-locality deep read returned to an inconsistency identified in
the initial audit. Common MOI construction, all three MOI mode emitters,
SuperCollider construction, resource placement, and independent validation
used constants named for RDNA4 to refer to EXEC, VCC, workitem-x, and device
scope encodings. These values are deliberately shared by every supported
target profile; their old names made common mechanisms appear target-specific
and propagated concrete-architecture vocabulary into otherwise normalized
mode code. The relay component and the Inline dynamic-record component also
declared private copies of subsets of the same constants.

The six universal operands now have target-neutral `kAmdGpu*` names and one
definition in `consan_capability_contract.h`, beside the normalized target
facts that establish their applicability. MOI native ABI and relay contracts
consume that authority, and the dynamic-record emitter no longer redeclares
raw numeric values. This is not a claim that all future targets must share
these encodings: a target introducing a real exception must extend the target
contract or publish a target-owned operation. It does ensure that common and
mode code no longer pretends one currently universal ABI fact belongs to
RDNA4.

All production and affected test consumers were converged in the same slice.
The architecture gate rejects the retired RDNA4 spellings throughout
production and requires all six target-neutral operands in the capability
contract. There is therefore one definition authority, 17 production consumer
files, no compatibility aliases, and no surviving duplicate declaration.

| Signal | Checkpoint 59 | Cumulative change | Slice change from checkpoint 58 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,258 | +283 | **-27** |
| Nonblank production lines | 98,949 | **-134** | **-31** |
| Production implementation lines | 91,218 | **-232** | **-33** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **175 / 52** | **-101 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 210 / 30 | +10 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Universal ABI definition authorities | **1 target-neutral contract** | n/a | **-2 duplicates** |
| RDNA4-named universal ABI operands in production | **0** | n/a | converged |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 18 focused target-capability,
five-architecture ABI, and architecture-boundary tests; all 4,753 nonphysical
tests over the five emulated targets at `-j16`; and all 635 physical gfx1201
tests serialized at `-j1`. The physical HIP fixtures were rebuilt after their
consumer migration. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.2, 14.4, 14.6, 14.7, 14.8, and
14.9. It removes a concrete-architecture peephole from all common and mode
consumers and deletes the duplicate authorities rather than retaining aliases.
The cumulative 232-line reduction is nevertheless still not material
whole-refactoring shrinkage. Remaining architecture locality, broader
operating-point and mutable-transaction surfaces, larger legacy harvesting,
material Section 14.8 evidence, and the independent Section 14 completion
audit remain open. The goal therefore remains active.

### 16.61 Convergence checkpoint 60: one independent relay-proof inventory

The final-validation deep read traced branch-relay provenance from the patch
inventory into the branch-only continuation, wide donor, and instruction-
reservoir proofs. Each validator independently scanned every patch and rebuilt
a relay graph. The scans overlapped substantially, but were not identical:
continuation ownership includes local FLAT tails, exact-shadow tails, and
validated original NOPs; donor payloads accept only active indirect-reservoir
slots; reservoir payloads accept every indirect-reservoir slot; and both
payload validators separately recognize indirect islands. Reservoir validation
also reconstructed branch-only route targets a second time.

`BranchRelayValidationInventory` now derives those facts once from the
published patch inventory. It deliberately exposes separate continuation,
donor-target, reservoir-target, indirect-island, branch-only-target, NOP,
donor, and reservoir facets. The validators share the traversal without
sharing an over-broad accepted-vertex set. Original-byte NOP qualification,
donor geometry, pristine-instruction decoding, payload decoding, route
integrity, and reservoir geometry remain independently checked by their owning
proofs; only the repeated inventory reconstruction is gone.

The inventory is built once at the final-validation composition boundary and
passed read-only to all three proofs. The structural gate requires that typed
product and rejects restoration of the old parallel relay/island graph locals.
Existing negative tests corrupt routes, displaced bodies, payload use,
provenance, NOP ownership, direct reservoirs, indirect reservoirs, donor
paths, and recursively routed Record/Replay reservoirs.

| Signal | Checkpoint 60 | Cumulative change | Slice change from checkpoint 59 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,208 | +233 | **-50** |
| Nonblank production lines | 98,898 | **-185** | **-51** |
| Production implementation lines | 91,170 | **-280** | **-48** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | **-1 / 0** |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | **-1 / 0** |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Relay patch-inventory traversals in final proof | **1 typed traversal** | n/a | **-2** |
| Per-validator relay/island graph reconstructions | **0** | n/a | **-3** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 10 focused NOP, continuation,
donor, direct-reservoir, indirect-reservoir, recursive-route, corruption, and
architecture-boundary tests; all 4,753 nonphysical tests over the five
emulated targets at `-j16`; and all 635 physical gfx1201 tests serialized at
`-j1`. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.5, 14.6, 14.7, 14.8, and 14.9.
Independent validation now has one narrow forward product for relay topology
while retaining distinct proof authorities and accepted graphs. The slice
harvests 48 implementation lines, bringing cumulative reduction to 280 lines;
that remains insufficient material whole-refactoring shrinkage. Remaining
architecture locality, broader operating-point and mutable-transaction
surfaces, larger legacy harvesting, material Section 14.8 evidence, and the
independent Section 14 completion audit remain open. The goal therefore remains
active.

### 16.62 Convergence checkpoint 61: derived placement transitions

The operating-point deep read traced the coordinator's resource-replanning
decision back through both placement products. Dispatch placement and
persistent placement each returned the complete attempted operating point, but
also carried a mutable `changed` bit. Seventeen accepted exits manually set
that bit after mutating some subset of the point. The coordinator therefore
trusted a second, weaker authority for a fact already represented exactly by
the product it accepted.

Both placement products now publish only their attempted point, typed
rejection, diagnostics, and—in the persistent case—the scratch assignments
that belong to that attempt. Before accepting either point, the coordinator
compares it with the effective point and derives whether resource plans must be
rebuilt. `ConSanMoiOperatingPoint` equality covers the complete point, so a new
field or a newly accepted placement path cannot silently omit maintenance of a
parallel transition flag. Rejected attempts retain their attempted point and
diagnostics exactly as before and never reach the replanning decision.

The 17 assignments, two product fields, and their initializer/test ceremony
are deleted rather than deprecated. The architecture gate rejects restoration
of a placement-product `changed` field and requires both coordinator
transitions to be derived from the complete attempted and effective points.
Existing automatic dispatch-SGPR, persistent private/VGPR/scalar-state,
descriptor-growth, dynamic-stack, accepted-fallback, and typed-rejection tests
exercise the affected exits.

| Signal | Checkpoint 61 | Cumulative change | Slice change from checkpoint 60 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,190 | +215 | **-18** |
| Nonblank production lines | 98,880 | **-203** | **-18** |
| Production implementation lines | 91,152 | **-298** | **-18** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Mutable placement-transition authorities | **0** | n/a | **-2 fields / -17 writes** |
| Derived accepted-point transition checks | **2** | n/a | **+2** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; the focused placement-product,
automatic-placement, persistent-state, dispatch-identity, and
architecture-boundary tests; and all 4,753 nonphysical tests over the five
emulated targets at `-j16`. The immediately preceding checkpoint's final tree
also passed all 635 physical gfx1201 tests serialized at `-j1`; this host-only
product simplification does not change generated GPU code. No test was
removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.5, 14.6, 14.7, 14.8, and 14.9.
The coordinator now derives resource replanning from the narrow product it
actually accepts, and the superseded mutable authority is gone. The slice
brings cumulative production reduction to 298 implementation lines, which is
still not material whole-refactoring shrinkage. Remaining architecture
locality, broader operating-point and mutable-transaction surfaces, larger
legacy harvesting, material Section 14.8 evidence, and the independent Section
14 completion audit remain open. The goal therefore remains active.

### 16.63 Convergence checkpoint 62: shared guest-relocation publication

The cross-mode emission deep read compared displaced guest-access handling in
Record/Replay, Sampled, and InlineShadow rather than classifying it by mode
tokens. Four paths—Record/Replay dynamic access, Sampled dense access, Sampled
spill-backed recovery, and InlineShadow access—made the same relocation
request, checked the same optional word product, copied its exact width, and
appended it to an emission buffer. Their semantic differences occur before and
after that sequence: each mode still chooses its replay address, optional
split-address scratch, guest offset, bank transition, and required wait.

One `append_moi_relocated_guest_access` operation now owns construction and
publication of that mechanism-neutral product. All four paths pass their
mode-selected inputs to it and retain their distinct ordering. The deep read
also found that immediate indirect jumps and the Sampled deferred-guest return
each independently built the same SCC snapshot, PC materialization, target
delta, and SCC restoration. One private relocation-owner operation now emits
that prefix; the public operations append either `s_setpc` for an immediate
transfer or the target wait needed before a guest instruction and later
`s_setpc`.

The four mode-local append sequences, four stale builder imports, and the
second indirect-target recipe are deleted. The architecture gate forbids mode
emitters from calling the word builder directly and requires exactly one
PC-delta recipe in the relocation owner. It does not forbid mode-local address
or ordering decisions, which are the genuine semantic differences.

| Signal | Checkpoint 62 | Cumulative change | Slice change from checkpoint 61 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,172 | +197 | **-18** |
| Nonblank production lines | 98,857 | **-226** | **-23** |
| Production implementation lines | 91,126 | **-324** | **-26** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Mode-local relocated-word append sequences | **0** | n/a | **-4** |
| SCC-preserving indirect-target recipes | **1** | n/a | **-1** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 33 focused relocation,
split-address, far-return, mode-emission, and architecture-boundary tests; all
4,753 nonphysical tests over the five emulated targets at `-j16`; and all 635
physical gfx1201 tests serialized at `-j1`. No test was removed, renamed,
disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. Mode emitters now own only the semantic choices around guest relocation,
while one common mechanism owns word publication and indirect target
preparation. The cumulative reduction reaches 324 implementation lines, still
short of material whole-refactoring shrinkage. Remaining architecture
locality, broader operating-point and mutable-transaction surfaces, larger
legacy harvesting, material Section 14.8 evidence, and the independent Section
14 completion audit remain open. The goal therefore remains active.

### 16.64 Convergence checkpoint 63: one InlineShadow EXEC-mask emitter

The InlineShadow synchronization deep read compared complete transaction
bodies rather than treating their mode-local file as inherently cohesive. The
acquired-token transaction, causal-snapshot capture, and versioned-release
transaction each independently declared the same instruction mini-language:
save EXEC, restore EXEC, intersect EXEC with VCC, and optionally compare a
VGPR with a literal before intersecting. Their mask registers and predicates
are intentionally different, but the target-normalized instruction recipes
and optional-instruction failure handling were exact copies.

One translation-unit-private `InlineExecMaskEmission` now owns those four
mechanical operations. Each transaction constructs it with its own narrow-save
pair, retains its own named mask lifetimes, and retains every semantic
predicate, load, table operation, retry rule, and failure path. This boundary
therefore shares implementation inside InlineShadow without promoting
Inline-specific synchronization policy into common MOI infrastructure.

The three local instruction recipes are deleted. The architecture gate
requires all three transactions to instantiate the private emitter and rejects
restoration of local EXEC-mask builder lambdas. Existing acquired-token,
causal-frontier, versioned-release, compare-exchange, malformed-publication,
and architecture-matrix tests exercise the shared operations under their
different policies.

| Signal | Checkpoint 63 | Cumulative change | Slice change from checkpoint 62 |
| --- | ---: | ---: | ---: |
| Production files | 262 | +33 | 0 |
| Physical production lines | 105,148 | +173 | **-24** |
| Nonblank production lines | 98,825 | **-258** | **-32** |
| Production implementation lines | 91,091 | **-359** | **-35** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| InlineShadow transaction-local EXEC-mask recipes | **0** | n/a | **-3** |
| Shared InlineShadow EXEC-mask emitters | **1 private mechanism** | n/a | **+1** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 223 focused InlineShadow,
acquired-token, causal, release, acquire, and architecture-boundary tests; and
all 4,753 nonphysical tests over the five emulated targets at `-j16`. The
immediately preceding checkpoint's final tree passed all 635 physical gfx1201
tests serialized at `-j1`; this slice changes no transaction policy or encoded
instruction sequence. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. InlineShadow has a clearer internal mechanism/policy boundary and three
copies have been harvested rather than hidden behind mode dispatch. The
cumulative reduction reaches 359 implementation lines, still short of material
whole-refactoring shrinkage. Remaining architecture locality, broader
operating-point and mutable-transaction surfaces, larger legacy harvesting,
material Section 14.8 evidence, and the independent Section 14 completion audit
remain open. The goal therefore remains active.

### 16.65 Convergence checkpoint 64: one Sampled causal-window validator

The Sampled synchronization deep read compared the complete causal-window
validation protocols used by barriers and atomics. Both routes loaded and
compared the same generation, dispatch identity, workgroup tuple, epoch,
selected entry, entry count, ready-publication state, and cluster identity.
They also used the same field-load, compare, EXEC-intersection, and no-lanes
branch mechanisms. The only intended route difference is the mismatch target:
a barrier advances to its next candidate window, while an atomic leaves the
selected-window path.

The duplicated implementations had drifted semantically. Atomic validation
consumed the typed `ConSanMoiWorkgroupSource` contract and therefore handled
scalar, vector, or entry-captured private sources. Barrier validation still
tested `scalar_src` directly and compared zero for every other representation.
At an accumulator-register boundary, placement deliberately captures the
workgroup tuple in private state, so the barrier could select a window without
checking the real workgroup identity.

One Sampled-local `consan_moi_sampled_window_emission` component now owns the
complete causal-window identity policy and its emission protocol. Barrier and
atomic lowering pass their separately chosen mismatch labels and register
assignments through one typed request. All twelve identity comparisons are
declared once, while iteration, watchpoint validation, atomic outcomes,
metadata publication, and restore paths remain with their genuine route
owners. The two old validator bodies are deleted.

The existing accumulator-boundary barrier test now requires private loads for
all three captured workgroup coordinates. This assertion failed against the
pre-fix implementation on both gfx942/CDNA3 and gfx950/CDNA4 configurations and
passes with the shared typed-source policy. The architecture gate requires
exactly one shared-validator call from each synchronization route and rejects
restoration of their local field-comparison mini-languages.

| Signal | Checkpoint 64 | Cumulative change | Slice change from checkpoint 63 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | +2 Sampled-local owner files |
| Physical production lines | 105,142 | +167 | **-6** |
| Nonblank production lines | 98,806 | **-277** | **-19** |
| Production implementation lines | 91,063 | **-387** | **-28** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Sampled causal-window validation policy owners | **1** | n/a | **-1 duplicate owner** |
| Synchronization consumers of the shared validator | **2** | n/a | barrier + atomic |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 80 focused Sampled barrier,
atomic, causal-window, private-state, and architecture tests; all 4,753
nonphysical tests over the five emulated targets at `-j16`; and all 635
physical gfx1201 tests serialized at `-j1`. No test was removed, renamed,
disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. Sampled window identity now has one mode-local authority, the two
synchronization routes contain only their real control-flow differences, and a
cross-component bug has permanent owning-boundary coverage. The cumulative
reduction reaches 387 implementation lines, still short of material
whole-refactoring shrinkage. Remaining architecture locality, broader
operating-point and mutable-transaction surfaces, larger legacy harvesting,
material Section 14.8 evidence, and the independent Section 14 completion audit
remain open. The goal therefore remains active.

### 16.66 Convergence checkpoint 65: target-owned workgroup TTMP ABI

The workgroup-source deep read found an architecture-locality contradiction in
the target capability contract. The contract distinguished descriptor system
SGPRs from command-processor TTMPs, but it exposed only that coarse choice.
Common MOI placement therefore still declared the concrete gfx1201 grid-x and
packed grid-yz TTMP indices and the gfx1250 cluster-workgroup TTMP index. A new
target could select the TTMP facility only by silently inheriting those old
targets' register numbers, and readers could not understand either target's
complete workgroup ABI from its gfx-named profile.

The coarse source enum is deleted. `ConSanTargetProfile` now carries an
optional exact command-processor workgroup-identity tuple. Absence selects the
existing descriptor system-SGPR path; gfx1201 declares its two grid TTMPs in
`consan_gfx1201_target_profile.h.inc`, and gfx1250 declares those plus its
cluster-workgroup TTMP in `consan_gfx1250_target_profile.h.inc`. Common
placement consumes the tuple generically and retains only the shared packed-yz
decoding policy. The three target-named constants have been deleted from the
common native-ABI header.

The target-profile validator rejects out-of-range or overlapping TTMPs and
requires cluster-facility support and a cluster-workgroup source to agree. Its
independent five-target fixture asserts the exact values and all malformed
invariants. The architecture gate prevents target-named TTMP constants from
returning to the common native-ABI owner, requires both affected gfx profiles
to declare their ABI, and requires the common workgroup-source region to
consume the exact profile rather than rebuilding the retired source enum.

| Signal | Checkpoint 65 | Cumulative change | Slice change from checkpoint 64 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,157 | +182 | +15 |
| Nonblank production lines | 98,820 | **-263** | +14 |
| Production implementation lines | 91,079 | **-371** | +16 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Common target-named workgroup TTMP constants | **0** | n/a | **-3** |
| Exact command-processor workgroup ABI owners | **2 gfx-named profiles** | n/a | **+2** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; 124 focused target-profile,
workgroup-identity, all-engine independent-workgroup, and
architecture-boundary tests; all 4,753 nonphysical tests over the five
emulated targets at `-j16`; and all 635 physical gfx1201 tests serialized at
`-j1`. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.2, 14.4, 14.5, 14.7, and 14.9.
Adding or reviewing a command-processor workgroup ABI now begins and ends in a
gfx-named profile, while common placement sees only the normalized product.
The exact-contract validation costs sixteen implementation lines, so this is a
target-locality and extension-safety investment rather than Section 14.8
progress; cumulative shrinkage moves from 387 to 371 implementation lines.
Remaining architecture locality, broader operating-point and mutable-
transaction surfaces, larger legacy harvesting, material Section 14.8
evidence, and the independent Section 14 completion audit remain open. The goal
therefore remains active.

### 16.67 Convergence checkpoint 66: one diagnostic presentation facet

The transform-publication deep read found three parallel data-model owners.
Fault sites, selected fault mutations, and barrier-move destinations each had
one complete analysis product and a second diagnostic struct repeating every
externally meaningful field. The pipeline then copied those duplicate schemas
field by field. The diagnostic variants deliberately omitted code-object
bindings, decoded operand locations, selection provenance, and exact
application proofs, but that useful authority boundary was expressed only by
three manually synchronized field lists and three projection maps.

Each analysis contract now owns one presentation value facet. The completed
analysis product derives from that facet and adds only its private identity,
operand, selection, and application-proof fields. The diagnostic API aliases
the facet itself, so presentation consumers cannot acquire the added mutation
authority and there is no second schema to drift. Pipeline publication copies
the facet directly; all three hand-written field maps and all three duplicate
diagnostic definitions are deleted.

The pipeline regression populates every kind of presentation fact, gives the
analysis products non-presentation proof data, and compares the published
facets as complete values. This proves both that future presentation fields
flow automatically and that the private proof fields remain outside the
diagnostic type. The architecture gate requires the three product/facet
relationships, the three diagnostic aliases, and direct facet publication; it
rejects reintroduction of parallel diagnostic schemas or manual projection.

| Signal | Checkpoint 66 | Cumulative change | Slice change from checkpoint 65 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,063 | +88 | **-94** |
| Nonblank production lines | 98,723 | **-360** | **-97** |
| Production implementation lines | 90,983 | **-467** | **-96** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Parallel diagnostic field schemas | **0** | n/a | **-3** |
| Hand-written diagnostic projection maps | **0** | n/a | **-3** |
| Test inventory | **5,388** | **+43** | 0 |

Validation includes a final-tree `-j16` build; all 33 pipeline tests plus the
architecture-boundary test; all 4,753 nonphysical tests over the five emulated
targets at `-j16`; and all 635 physical gfx1201 tests serialized at `-j1`. No
test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.5, 14.7, 14.8, and 14.9. A
single analysis-owned value now defines what each product may disclose, while
the derived product alone retains authority needed to apply a transformation.
The change removes 96 implementation lines and raises cumulative shrinkage to
467 lines. Remaining architecture locality, mode locality, broader operating-
point and mutable-transaction surfaces, larger legacy harvesting, material
whole-refactoring shrinkage, and the independent Section 14 completion audit
remain open. The goal therefore remains active.

### 16.68 Convergence checkpoint 67: one normalized access scratch contract

The access-placement deep read found that common LDS placement and
SuperCollider FLAT lowering independently interpreted the same normalized
`ConSanAccessLoweringForm`. Both computed the access width, excluded address,
destination, and data operands, enforced the classifier-owned tuple alignment,
found the first register after those operands, honored an explicit scratch
override, searched liveness, and fell back to a spill-backed allocation. The
FLAT copy was mode-local implementation of common placement policy and did not
include the normalized second data tuple in its range calculations.

Common placement now owns one access scratch contract and its liveness and
spill selection mechanisms. Both native LDS and SuperCollider FLAT lowering
consume those operations; each retains only its genuine differences, such as
multi-owner liveness intersection, descriptor-growth policy, spill-frame
construction, and emitted probe semantics. The complete FLAT scratch helper
family and its private operand-exclusion implementation are deleted from the
SuperCollider support component.

A direct contract regression constructs a normalized two-range access and
proves that width, search floor, tuple alignment, explicit selection, and spill
selection all account for every operand tuple. The architecture gate requires
SuperCollider FLAT to consume all five common operations and forbids the
mode-local scratch helper family from returning.

| Signal | Checkpoint 67 | Cumulative change | Slice change from checkpoint 66 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 104,945 | **-30** | **-118** |
| Nonblank production lines | 98,615 | **-468** | **-108** |
| Production implementation lines | 90,875 | **-575** | **-108** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Access scratch policy owners | **1 common placement owner** | n/a | **-1** |
| SuperCollider-local FLAT scratch helpers | **0** | n/a | **-5** |
| Test inventory | **5,389** | **+44** | **+1** |

Validation includes a final-tree `-j16` build; 182 focused normalized-scratch,
FLAT, LDS, and architecture-boundary tests; all 4,754 nonphysical tests over
the five emulated targets at `-j16`; and all 635 physical gfx1201 tests
serialized at `-j1`. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.7, 14.8, and 14.9.
Mode locality no longer means that SuperCollider owns a copy of
architecture-normalized placement policy, and 108 more implementation lines
are harvested. Cumulative shrinkage reaches 575 lines. Remaining architecture
locality, mode locality, broader operating-point and mutable-transaction
surfaces, larger legacy harvesting, material whole-refactoring shrinkage, and
the independent Section 14 completion audit remain open. The goal therefore
remains active.

### 16.69 Convergence checkpoint 68: one target-normalized causal scope

The synchronization deep read found that `raw_scope` was doing three different
jobs. It retained an instruction operand for diagnostics and stale-byte
mutation proof, acted as the common causal visibility scope in policy and
association, and was serialized directly into two mode report ABIs. Most
supported instructions happened to use values in the range zero through
three, but that numeric agreement was not one semantic contract. In
particular, gfx12-family CU and SE cache domains are not literally HSA
wavefront and workgroup scopes, and gfx1201 and gfx1250 use different raw
values for the separately normalized compiler workgroup-acquire qualifier.
Common analysis and modes were therefore interpreting a target representation
even when the numeric result looked architecture-independent.

Program-analysis target operations now publish a typed `ConSanMemoryScope`
alongside the retained raw operand. The shared gfx12-family target owner has an
explicit exhaustive raw-to-semantic mapping with the conservative CU/SE
rationale; pre-gfx12 target decoding publishes its implicit agent scope or
declares that scope follows the resolved address space. Common program
analysis only copies or derives these typed target products. Events and
sequences carry raw and semantic values independently: raw values may flow
unchanged to diagnostics and exact mutation proof, while association may
strengthen only the semantic sequence scope. Policy, fault selection,
perturbation, and mode planning consume the semantic value and never compare
raw numeric codes.

The mode boundary is explicit in the other direction. Sampled owns one typed
conversion from the common scope to its sampled synchronization ABI, replacing
two numeric conversion implementations. Record/Replay owns its own conversion
to the compact-trace ABI and rejects wavefront-local communication rather than
depending on the common enum's declaration order. The atomic address-lowering
form's unused duplicate scope field was deleted instead of being converted
into a third authority. Where an encoded scope exists, diagnostics retain its
exact numeric output; older implicit forms keep any compatibility presentation
separate from the typed semantic authority. Ordinary scope mutation compares
current encoded bits with the retained raw operand rather than casting the
semantic enum back into a target encoding.

A direct target-boundary matrix drives raw values zero through three through
gfx1201 and gfx1250 vector-memory and atomic decoders and the gfx1250 buffer
decoder, checking the raw operand and normalized product independently. Direct
mode tests pin the Record/Replay and Sampled ABI mappings. The architecture
gate requires the target-owned mapping and both mode-owned mappings, rejects
numeric scope reinterpretation in common semantic consumers, rejects raw
scope use by semantic-only consumers, and prevents the dead lowering-form
copy and the retired Sampled conversion from returning.

The exact lexical recount also found that the historical absolute ledger had
become stale after checkpoint 54. The three latest checkpoint commits were
each underreported by 68 physical, 70 nonblank, and 68 implementation lines;
checkpoint 67's exact implementation count was 90,943, not 90,875. The
starting physical and nonblank values were each low by one while the starting
implementation count of 91,450 was exact. The baseline table above is
corrected and this checkpoint resets the live ledger from the exact tree.
Between checkpoint 67 and the resumed refactoring, urgent issue-fix commit
`e50b89339fb` added 99 implementation lines and two tests. This scope slice
then adds 70 implementation lines and four tests; its explicit raw/semantic
separation is a locality and correctness investment, not a size payoff.

| Signal | Checkpoint 68 | Cumulative change | Change since checkpoint 67 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,223 | +247 | +210 |
| Nonblank production lines | 98,886 | **-198** | +201 |
| Production implementation lines | 91,112 | **-338** | +169 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 356 / 63 | +66 / +12 | 0 / 0 |
| Numeric raw-scope interpretations in semantic consumers | **0** | n/a | converged |
| Explicit mode scope-to-report ABI owners | **2** | n/a | Record/Replay + Sampled |
| Atomic lowering-form scope copies | **0** | n/a | **-1 dead field** |
| Test inventory | **5,395** | **+50** | **+6** |

Validation includes a final-tree `-j16` build; 63 focused target-scope,
atomic/fence-policy, fault, perturbation, mode-ABI, and architecture-boundary
tests; all 4,760 nonphysical tests over the five emulated targets at `-j16` in
196.37 seconds; and all 635 physical gfx1201 tests serialized at `-j1` in
107.09 seconds. The first full nonphysical gate exposed one gfx1250 simulator
case that is sub-second alone but exhausted its 60-second allowance under the
full `-j16` load; its baseline workload allowance is now 120 seconds, and the
complete repeated gate passed without a retry. No test was removed, renamed,
disabled, or replaced.

This checkpoint strengthens Sections 14.1 through 14.7 and 14.9. Target scope
encoding, common causal meaning, and mode report representation now meet at
explicit typed boundaries, and a future target cannot silently inherit an
ordinal coincidence. It does not strengthen Section 14.8: the urgent fixes
and this correctness boundary reduce cumulative implementation shrinkage from
the corrected 507 lines at checkpoint 67 to 338 lines. Remaining architecture
and mode locality, broader operating-point and mutable-transaction surfaces,
larger legacy harvesting, material whole-refactoring shrinkage, and the
independent Section 14 completion audit remain open. The goal therefore
remains active.

### 16.70 Convergence checkpoint 69: one resolved owner/epoch-initialization authority

The operating-point deep read found that owner/epoch initialization still had
two authorities. `ConSanMoiOperatingPoint::moi_initialize_owner_epoch` was an
optional refinement, while downstream placement, prologue, and mode code used
`moi_initializes_owner_epoch` to fall back to the immutable request whenever
the point had no value. Consequently, the supposedly accepted operating point
did not completely describe the transformation that consumers would emit.

Initial operating-point construction now resolves the request exactly once
into a non-optional Boolean. Planning and placement may refine that value, and
all later lowering consumes only the accepted point. The fallback helper is
deleted, as is the optional state in the mode plan. The architecture gate
forbids both the retired helper and optional-value fallback in production and
requires the request-to-point assignment at the construction boundary.

The existing direct operating-point test now proves snapshot semantics: it
changes the request after constructing a point and verifies that mode planning
continues to obey the resolved point. The first broad MOI gate also exposed 18
test-fixture failures. `MoiOptions`, the tests' convenience type, intentionally
lets fixtures default-construct and then mutate request fields; its lowering
adapter now performs the same explicit request-to-point resolution at that
test-only boundary while preserving all independently seeded allocation
state. This keeps fixture ergonomics without reintroducing a production
fallback authority. All 18 affected reproducers and the complete MOI gate pass.

Between checkpoint 68 and this slice, urgent gfx950 private-dispatch fallback
commit `f9428fa1ba6` added eight production implementation lines and eleven
tests. This slice removes five implementation lines, leaving the exact current
tree three implementation lines larger than checkpoint 68.

| Signal | Checkpoint 69 | Cumulative change | Slice change from `f9428fa1ba6` |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,224 | +248 | **-7** |
| Nonblank production lines | 98,889 | **-195** | **-5** |
| Production implementation lines | 91,115 | **-335** | **-5** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 354 / 63 | +64 / +12 | **-2 / 0** |
| Production request-fallback authorities | **0** | n/a | helper deleted |
| Test inventory | **5,406** | **+61** | 0 |

Validation includes a final-tree `-j16` build; 15 direct mode-planning,
ownership, and architecture-boundary tests; all 18 initially affected fixture
reproducers; all 897 MOI tests; all 4,771 nonphysical tests over the five
emulated targets at `-j16` in 211.36 seconds, including 2,918 simulator tests;
and all 635 physical gfx1201 tests serialized at `-j1` in 109.49 seconds. No
test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5, 14.6, 14.7, 14.8, and
14.9. Owner/epoch initialization is now fully resolved before consumers see an
operating point, and the obsolete parallel request path is gone. Cumulative
implementation shrinkage is 335 lines, which is still not material completion.
Remaining architecture and mode locality, broader operating-point and mutable-
transaction surfaces, larger legacy harvesting, material whole-refactoring
shrinkage, and the independent Section 14 completion audit remain open. The
goal therefore remains active.

### 16.71 Convergence checkpoint 70: retire physical-qualification report mode

The architecture-locality and legacy-harvest deep read found a temporary
gfx1201 qualification facility still installed as a production-visible hidden
mode. `RJ_CONSAN_MOI_PARTITION_MASK_DEBUG` flowed through hook configuration,
the broad debug request, every InlineShadow access-application path, generated
wave-partition code, report-header fields borrowed from unrelated engines, a
decoder exception that suppressed ordinary counts, and a dedicated renderer
branch. The renderer itself labeled the output `acceptance=false`: the path
was diagnostic scaffolding used to investigate the now-accepted production
wave-coalescing implementation, not supported ConSan behavior.

The complete side channel is deleted. InlineShadow now has one generated
wave-partition path; the report header has one meaning; the runtime decoder has
one accounting interpretation; and hook configuration no longer exposes an
architecture-bring-up control. The existing focused emitter test is retained
at the same inventory position but now names and proves the durable production
invariant that the qualification work established: both diagnostic and address-
group publishers rank lanes within the active subset. A structural gate scans
both the lowerer and the HSA hook and rejects reintroduction of either the
field or environment variable.

| Signal | Checkpoint 70 | Cumulative change | Slice change from checkpoint 69 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,131 | +155 | **-93** |
| Nonblank production lines | 98,798 | **-286** | **-91** |
| Production implementation lines | 91,044 | **-406** | **-71** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 354 / 63 | +64 / +12 | 0 / 0 |
| Partition-mask debug production authorities | **0** | n/a | option, emitter, decoder, renderer deleted |
| Test inventory | **5,406** | **+61** | 0 |

Validation includes a clean `-j16` rebuild; five direct configuration,
emission, decoder, renderer, and architecture-boundary tests; 45 broader
InlineShadow, request-construction, and typed report-pipeline tests; and all
4,771 nonphysical tests over the five emulated targets at `-j16` in 206.92
seconds, including all 2,918 simulator tests. The physical gfx1201 gate was
stopped at the requested lower physical-test cadence after 593 of 635 tests
had passed with no failure; this partial run is not counted as a validation
gate. The test inventory is unchanged; one temporary-debug assertion was
converted and renamed to state the corresponding permanent production
invariant.

This checkpoint strengthens Sections 14.1, 14.2, 14.3, 14.5, 14.7, 14.8, and
14.9. Architecture-specific bring-up machinery no longer cuts across a mode,
the common request, and the runtime report pipeline, and the deletion harvests
71 implementation lines. Cumulative shrinkage is 406 lines, which is still not
material completion. Remaining architecture and mode locality, broader
operating-point and mutable-transaction surfaces, larger legacy harvesting,
material whole-refactoring shrinkage, and the independent Section 14
completion audit remain open. The goal therefore remains active.

### 16.72 Convergence checkpoint 71: mode-published dense-router mechanics

The mode-locality deep read found one remaining three-engine cross-axis knot
in dense access routing. Shared placement accepted a
`MoiDenseAccessRouteAbi` enum and used it to distinguish Record/Replay replay-
ordering constraints, InlineShadow island geometry, and gfx11 eligibility.
Shared emission independently inspected `request.moi_engine`, reconstructed
two scalar ABIs, and threaded a Record/Replay-versus-Sampled collapse Boolean
through the common access-application template. An InlineShadow-only scalar-
ABI helper lived in common placement and was also a peephole used by atomic
and barrier routing. The mechanism was mechanically shared, but its common
owners still knew every participating mode's representation.

Each mode now publishes a complete `MoiDenseRouterPlan` through the mode
registry. The plan resolves the indirect-jump state, dispatch key, optional
call-return pair, entry and relocated-host geometry, spill-router collapse,
SCC restoration, dependency wait, and patch-metadata publication. Record/
Replay and Sampled compose their common recording representation through one
shared factory; Sampled alone owns the explicit-key alias collapse that differs
from Record/Replay. InlineShadow owns its fixed and spill-backed scalar
representations in `consan_moi_inline_shadow.cpp`. A narrow route-traits
product separately publishes the replay-ordering constraint needed before a
group-specific scalar assignment exists.

Common placement now asks the selected mode for the resolved group plan and
uses only its island geometry plus the replay-ordering trait. Common emission
accepts the resolved plan instead of a request, operating point, engine test,
and mode-flavor Boolean. The atomic and barrier consumers use the same
registry operation, so the former InlineShadow placement helper and its
parallel eligibility authority are deleted. The three mode call sites no
longer name an ABI enum or pass collapse policy through common templates.

The final review also removed an incipient architecture peephole from the new
registry boundary. Modes declare whether they require the target's general
dense-call facility; target support is derived from the existing normalized
`direct_call_form` and `supports_moi_dense_s_call_b64` profile facts. Common
mode planning contains no encoding-family or architecture enumerator, and the
five gfx-owned profiles remain the authority for that capability. Thus adding
a compatible target extends the matrix through its target profile, while
adding a mode extends it through one mode registration rather than a new
common mode-by-architecture switch.

Two direct tests pin the complete mode/target support matrix across gfx1100,
gfx1201, gfx942, gfx950, and gfx1250 and the resolved call mechanics that
intentionally differ among Record/Replay, Sampled, and InlineShadow. The
architecture gate rejects the retired ABI enum, InlineShadow-only helper, and
common emission branch; requires every participating mode to register its
dense-router operation; and forbids common mode planning from regaining raw
architecture or encoding-family selection.

| Signal | Checkpoint 71 | Cumulative change | Slice change from checkpoint 70 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,177 | +201 | +46 |
| Nonblank production lines | 98,837 | **-247** | +39 |
| Production implementation lines | 91,082 | **-368** | +38 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 358 / 61 | +68 / +10 | +4 / **-2** |
| Common dense-route engine branches | **0** | n/a | enum and InlineShadow branch deleted |
| Common dense-route architecture identities | **0** | n/a | normalized target capability only |
| Test inventory | **5,408** | **+63** | **+2** |

Validation includes a final-tree `-j16` build; all 16 mode-planning and
architecture-boundary tests; and all 4,773 nonphysical tests over the five
emulated targets at `-j16` in 205.34 seconds, including all 2,918 simulator
tests. In accordance with the reduced physical-test cadence, no physical
gfx1201 test was run for this slice. No test was removed, renamed, disabled,
or replaced.

This checkpoint strengthens Sections 14.1 through 14.7 and 14.9. Dense access
routing now has one common mechanical consumer and mode-local policy
publishers, while target eligibility flows from gfx-owned normalized facts.
The explicit contract costs 38 implementation lines despite deleting the
superseded enum, helper, common ABI reconstruction, and Boolean plumbing, so it
does not strengthen Section 14.8; cumulative implementation shrinkage is 368
lines and remains non-material. Remaining architecture and mode locality,
broader operating-point and mutable-transaction surfaces, larger legacy
harvesting, material whole-refactoring shrinkage, and the independent Section
14 completion audit remain open. The goal therefore remains active.

### 16.73 Convergence checkpoint 72: one runtime report-layout authority

The runtime-report deep read found that the allocation registry retained a
complete `ConSanMoiReportBufferLayout` and then copied ten of its capacities
plus two mode Booleans into parallel fields. The report-pipeline input copied
the fence capacity and the two Booleans again. The decoder reconstructed the
engine from those Booleans, used them to gate mode-local capacities, and used
the separate fence-capacity copy. This made one validated layout coexist with
several partial authorities whose consistency depended on positional aggregate
initialization and manual forwarding.

The registry entry and report-pipeline input now carry only the complete typed
layout. Registry construction uses named initialization, snapshot validation
gets its expected engine from `layout.engine`, and the decoder consumes the
validated layout's engine, fence capacity, and mode-local capacities directly.
The renderer uses the same layout for fence-capacity diagnostics. Twelve
redundant registry fields and three redundant pipeline-input fields are
deleted; there is no mode reconstruction or capacity projection between report
allocation and decoding.

A decoder regression constructs valid report layouts for Record/Replay,
Sampled, and InlineShadow and proves that the decoded mode comes solely from
the layout. Structural gates forbid the retired mode and capacity fields in
the public pipeline input, the registry entry's consumers, and decoder input
accesses, preventing a second authority from being threaded back into the
pipeline.

| Signal | Checkpoint 72 | Cumulative change | Slice change from checkpoint 71 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,151 | +175 | **-26** |
| Nonblank production lines | 98,811 | **-273** | **-26** |
| Production implementation lines | 91,056 | **-394** | **-26** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **174 / 52** | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 358 / 61 | +68 / +10 | 0 / 0 |
| Parallel runtime report-layout fields | **0** | n/a | **-15 fields** |
| Test inventory | **5,408** | **+63** | 0; one in-target regression added |

Validation includes a final-tree `-j16` build; all 20 focused report-planning,
snapshot, decoder, renderer, registry, and architecture-boundary tests; and all
4,773 nonphysical tests over the five emulated targets at `-j16` in 208.77
seconds, including all 2,918 simulator tests. In accordance with the reduced
physical-test cadence, no physical gfx1201 test was run for this slice. No test
was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, 14.5 through 14.9. Runtime
mode and capacity semantics cross the allocation/decoding boundary as one
typed product, the parallel union-shaped projections are gone, and the slice
harvests 26 implementation lines. Cumulative implementation shrinkage reaches
394 lines, which is still not material completion. Remaining architecture and
mode locality, broader operating-point and mutable-transaction surfaces,
larger legacy harvesting, material whole-refactoring shrinkage, and the
independent Section 14 completion audit remain open. The goal therefore
remains active.

### 16.74 Convergence checkpoint 73: one discriminated runtime metadata product

The runtime-attribution deep read found that one transform retained three
independently populated static-mapping vectors even though the observation
plan selects exactly one evidence engine. The host registry repeated that
union-shaped representation as two mapping vectors, one compact-mapping
count, and three malformed flags. The report-pipeline input repeated those six
fields again. Valid transforms populated only one family, but the public
products represented mixed Record/Replay, Sampled, and InlineShadow metadata,
and every aggregator and runtime consumer had to preserve that convention.

Static lowering now publishes one `ConSanRuntimeStaticMapping` variant whose
alternatives own the Record/Replay, Sampled, or InlineCompact mapping sequence.
Same-mode commits append transactionally; a cross-mode append is rejected
without mutating the accepted product. Commit validation and coverage-ledger
aggregation consume the selected typed alternative rather than scanning three
containers. This makes the observation plan's one-engine invariant structural
at the transform/runtime boundary.

The host registry and report pipeline share one owning
`AutoMoiRuntimeStaticMetadata` variant. The registry stores that product and
the synchronous pipeline borrows it directly; an earlier draft's second
borrowed variant and manual registry-to-pipeline adapter were deleted before
the checkpoint. Decoder and analyzer components use `std::get_if` for only
their mode-local alternative. Sampled malformed attribution, InlineCompact
token attribution, and Record/Replay provenance can no longer coexist as
parallel input fields.

One transform regression proves same-mode aggregation and cross-mode rejection
without partial mutation. One host regression proves that the pipeline
contract carries only one selected metadata alternative. Existing end-to-end
registry tests continue to exercise Record/Replay provenance, Sampled mapping
and malformed-state reporting, and InlineShadow compact-token registration.
Structural gates reject the three retired transform containers and all twelve
retired registry/pipeline fields, require both discriminated products, and
require the registry to use the pipeline's shared owning type rather than a
second representation.

| Signal | Checkpoint 73 | Cumulative change | Slice change from checkpoint 72 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,256 | +280 | **+105** |
| Nonblank production lines | 98,906 | **-178** | **+95** |
| Production implementation lines | 91,147 | **-303** | **+91** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **173 / 52** | **-103 / -5** | **-1 / 0** |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 355 / 61 | +65 / +10 | **-3 / 0** |
| Parallel runtime static-metadata fields | **0** | n/a | **-15 fields** |
| Test inventory | **5,409** | **+64** | **+1** |

Validation includes a final-tree `-j16` build; all 12 focused transform,
registry, decoder, provenance, malformed-mapping, and architecture-boundary
tests; and all 4,774 nonphysical tests over the five emulated targets at
`-j16` in 203.10 seconds, including all 2,918 simulator tests. In accordance
with the reduced physical-test cadence, no physical gfx1201 test was run for
this slice. No test was removed, renamed, disabled, or replaced.

This checkpoint strengthens Sections 14.1, 14.3, and 14.5 through 14.7 and
14.9. Runtime attribution is now one typed mode product from lowering through
host decoding, and the duplicate host representation discovered during the
slice was harvested rather than retained as an adapter. The explicit variant
factories, accessors, transactional aggregation, and owning runtime metadata
cost 91 implementation lines relative to checkpoint 72, so this slice weakens
the provisional Section 14.8 count and cumulative implementation shrinkage
falls from 394 to 303 lines. That remains non-material. Remaining architecture
and mode locality, broader operating-point and mutable-transaction surfaces,
larger legacy harvesting, material whole-refactoring shrinkage, and the
independent Section 14 completion audit remain open. The goal therefore
remains active.

### 16.75 Convergence checkpoint 74: one normalized access-resource fact product

The access-resource deep read found one coherent mode/target interaction still
crossing the common placement boundary in unnormalized form. Access scratch
sizing, operand-overlap spill admission, and spill-backed retry accepted the
complete `ConSanMoiOperatingPoint` and raw architecture identity. Record/Replay,
Sampled, and InlineShadow then independently reconstructed address-capture
scratch, gfx1250 two-address replay scratch, dynamic-stack reservoir demand,
target admission, and architecture-sensitive spill-recovery predicates. Two
Sampled helpers were parallel adapters: one repeated the target predicate and
one merely forwarded to the ordinary scratch counter.

Access targeting now publishes one `MoiAccessResourceFacts` value for each
site. Common normalization composes the candidate, target capabilities, and
only the relevant operating-point projections into mechanism-named facts:
address and replay scratch counts, dynamic-stack reservoir demand, persistent
state availability, target availability, native-LDS recovery,
clobbered-address recovery, and the disjoint-address constraint. The three
mode owners consume those facts without receiving a raw architecture or the
broad operating point through their scratch or spill-policy contracts.
Placement recomputes the same value after owner-local persistent assignment,
so the provisional and committed resource transactions retain one authority.

The migration is complete rather than additive. The old
`sampled_access_supports_spill_backed_operand_recovery` target adapter and
`sampled_spill_backed_scratch_count` forwarding counter are deleted. Repeated
address-scratch and two-address formulas are deleted from the three mode
owners, and the no-longer-observed spill-recovery parameter is removed from
Sampled emission. One direct contract test exercises CDNA5, RDNA4, and CDNA4
normalization as well as the persistent-to-private epoch transition. The
architecture gate requires the normalized product, rejects broad-point or
raw-architecture fields in both spill-policy contexts, and prevents the two
retired Sampled adapters from returning.

| Signal | Checkpoint 74 | Cumulative change | Slice change from checkpoint 73 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,248 | +272 | **-8** |
| Nonblank production lines | 98,898 | **-186** | **-8** |
| Production implementation lines | 91,136 | **-314** | **-11** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 174 / 52 | **-102 / -5** | +1 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 332 / 63 | +42 / +12 | **-13 / +2** |
| Broad-point / raw-architecture fields in migrated spill contexts | **0 / 0** | n/a | completed |
| Retired Sampled access-resource adapters | **0** | n/a | **-2** |
| Test inventory | **5,410** | **+65** | **+1** |

Validation includes a final-tree `-j16` build; all 17 mode-planning and
architecture-boundary tests; and all 4,775 nonphysical tests over the five
emulated targets at `-j16` in 204.77 seconds, including all 2,918 simulator
tests. The final source-only cleanup removed an obsolete capability include
and was followed by the final build and focused contract gate. No test was
removed, renamed, disabled, or replaced. In accordance with the reduced
physical-test cadence, no physical gfx1201 test was run for this slice.

This checkpoint strengthens Sections 14.2, 14.3, and 14.5 through 14.7. One
common composition product now separates target normalization and mutable
placement state from mode policy, while retaining distinct mode-owned choices.
It also reverses checkpoint 73's local size growth, but cumulative production
is only 314 implementation lines smaller than the baseline and therefore still
does not satisfy Section 14.8. The broad operating point remains spread across
63 production files despite the lower reference count, and the mutable
transaction, remaining placement knots, material legacy harvesting, and
independent completion audit remain open. The goal therefore remains active.

### 16.76 Convergence checkpoint 75: one projected scalar-routing transaction

The scalar-routing deep read found that the mode registry still exposed the
complete operating point through both its scalar-ABI and dense-router
callbacks. Record/Replay, Sampled, and InlineShadow each derived a scalar ABI,
then each dense-router planner independently derived the same ABI again from
the broad point. The dense-router callbacks also received a complete
`ConSanTargetProfile` although they used only its direct-call form. Sampled's
publication-state helper additionally accepted a full request solely to prove
that its already mode-local callers were Sampled.

The boundary now projects one immutable `MoiScalarRoutingState` from the
accepted operating point. It contains only the selected EXEC-save base,
scalar-spill representation, shared router jump and call allocations, and the
presence of branch-only preservation. Scalar-ABI mode callbacks consume that
projection without a request or operating point. Common dense-router planning
selects the scalar ABI exactly once and passes the resolved ABI to the selected
mode's router callback; no mode router calls its scalar planner again.

The pre-existing direct-call projection used for save-window sizing is now the
shared `MoiScalarTargetFacts` product. Dense-router callbacks consume it rather
than the full target profile. Common planning retains the target capability
admission check, then passes only the direct-call semantic needed by the mode.
Sampled publication-state planning now consumes only the optional EXEC-save
base, and its access-return SCC resolver is translation-unit-local rather than
declared from the common placement contract.

One direct regression proves that scalar routing projects all five relevant
fields while ignoring owner/epoch, private-state, and persistent-state
placement. Existing mode-planning tests prove all three fixed and spill-backed
scalar ABIs and their target-specific dense call mechanics. The architecture
gate rejects a broad operating point or complete target profile in either mode
callback, requires dense routing to consume the already-selected scalar ABI,
and prevents Sampled publication layout from regaining request or operating-
point buses.

| Signal | Checkpoint 75 | Cumulative change | Slice change from checkpoint 74 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,280 | +304 | +32 |
| Nonblank production lines | 98,927 | **-157** | +29 |
| Production implementation lines | 91,160 | **-290** | +24 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 174 / 52 | **-102 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 317 / 61 | +27 / +10 | **-15 / -2** |
| Broad-point / complete-target fields in scalar mode callbacks | **0 / 0** | n/a | completed |
| Mode-local scalar-ABI recomputation inside dense routers | **0** | n/a | **-3** |
| Test inventory | **5,411** | **+66** | **+1** |

Validation includes a final-tree `-j16` build; all 18 mode-planning and
architecture-boundary tests; and 442 focused dense-router, scalar-state,
SGPR-spill, EXEC-save, and branch-only tests at `-j16`. The focused gate
includes 146 simulator cases spanning gfx942, gfx950, gfx1100, gfx1201, and
gfx1250 and all four evidence engines. Checkpoint 74 immediately before this
slice passed the full 4,775-test nonphysical inventory; the additional unit
test raises the current nonphysical inventory to 4,776. No test was removed,
renamed, disabled, or replaced. No physical gfx1201 test was run.

This checkpoint strengthens Sections 14.2 through 14.7: scalar representation,
mode choice, and target call semantics now meet at one explicit composition
point instead of through a mode/target/broad-state knot. The new projection and
its invariant test cost 24 production implementation lines even after deleting
the duplicate derivations, so cumulative shrinkage retreats from 314 to 290
lines and Section 14.8 remains materially unsatisfied. The next convergence
slice must cash in a deletion or sharing opportunity rather than adding another
contract layer. Remaining broad placement and mutation transactions, larger
legacy harvesting, and the independent completion audit remain open. The goal
therefore remains active.

### 16.77 Convergence checkpoint 76: immutable automatic-resume inventory

The automatic-resume deep read found a reverse transaction adapter at the
pipeline/lowerer boundary. Initial lowering published a complete
`ConSanTransformArtifacts` into split public and private pipeline storage.
`TransformResult::take_lowering_artifacts` later reconstructed that broad
mutable transaction solely so MOI could repeat planning after runtime report
resources were bound. The pipeline also retained the initial
`ConSanMoiOperatingPoint`, but resume never consumed it: lowering replanned from
the newly bound request and immutable program facts.

Automatic MOI resume now consumes one deliberately narrow
`ConSanMoiRetryInventory`. It carries only the immutable semantic facts that
survive resource binding: `ProgramInventory`, `ConSanCoverageLedger`, fault-site
inventory, and barrier-move destinations. The lowerer creates fresh private
mutation, resource, patch, candidate-image, outcome, diagnostic, and operating-
point state at its own boundary. Those mutable products cannot be supplied by a
pipeline caller through the new type.

The first narrow implementation retained only the program and coverage
inventories. The existing direct-versus-resume equivalence test immediately
exposed that late fault diagnostics also require the fault-site and barrier-move
inventories. Adding exactly those two semantic products restored equivalence;
no mutation plan, resource attempt, prior patch, candidate bytes, diagnostic,
outcome, or operating point was added back. Direct lowerer mechanism tests use
a test-only adapter from their intentionally complete fixture result, while the
production pipeline constructs the narrow product directly.

The old `take_lowering_artifacts` reverse adapter, its complete-transaction
reconstruction, the dead retained operating point, mutable-inventory rejection,
and stale-warning cleanup are deleted. The architecture gate requires the
narrow resume product, rejects a broad transaction parameter, and prevents both
the reverse adapter and retained private operating point from returning.

An authoritative revision-by-revision recount also found that checkpoints 74
and 75 had each transcribed the operating-point reference count ten too high.
Their rows are corrected from 342 to 332 and from 327 to 317; their file counts
and slice deltas were already correct. The current count below is taken directly
from this tree.

| Signal | Checkpoint 76 | Cumulative change | Slice change from checkpoint 75 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,261 | +285 | **-19** |
| Nonblank production lines | 98,909 | **-175** | **-18** |
| Production implementation lines | 91,145 | **-305** | **-15** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | **-6 / 0** |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 60 | +26 / +9 | **-1 / -1** |
| Reverse lowerer-transaction adapters | **0** | n/a | **-1** |
| Retained pipeline operating points | **0** | n/a | **-1** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a final-tree `-j16` build; all 19 automatic-resume,
direct-versus-resume, fault-composition, and architecture-boundary tests; and
all 4,776 nonphysical tests over the five emulated targets at `-j16`, including
all 2,918 simulator tests. No test was removed, disabled, renamed, or replaced.
In accordance with the reduced physical-test cadence, no physical gfx1201 test
was run for this slice.

This checkpoint strengthens Sections 14.1, 14.5, and 14.7 through 14.9. A
pipeline component can no longer reverse its result into a lowerer transaction,
and automatic resume carries only the semantic inventory required across the
runtime-resource boundary. It also repays 15 of checkpoint 75's 24 added
implementation lines. Cumulative production remains only 305 implementation
lines smaller than baseline, however, so Section 14.8 is still materially
unsatisfied. Other broad placement and mutation transactions, both extension
exercises, larger legacy harvesting, and the independent completion audit
remain open. The goal therefore remains active.

### 16.78 Convergence checkpoint 77: harvest the abandoned kernel-target facade

The target-contract deep read found that `ConSanKernelTargetProfile` was not a
production component. Its type and three wave-selection/allocation functions
had no production consumer: the only caller was a unit test that exercised the
projection itself. Descriptor-selected wave and allocation facts already flow
through `ProgramInventory`, resource plans, descriptor helpers, and independent
target validation. The advertised kernel-target view therefore duplicated an
authority without participating in the transformation.

Four other capability helpers had the same shape. The target-admission alias,
GFX12-plus-RDNA predicate, S_CALL_B64 predicate, and descriptor-partitioned-
accumulator predicate were referenced only by tests of those helpers. Their
underlying facts already have one direct representation in
`ConSanTargetProfile`, and no production component consumed the aliases.

The unused kernel-target type, its constructor/projections, and all four test-
only predicates are deleted. The existing capability-contract test inventory
now checks the authoritative target-profile fields directly; its wave/allocation
case is renamed to describe that actual owner. `DESIGN.md` and the completed
reimplementation map no longer claim a nonexistent descriptor-profile layer.
The architecture gate rejects all eight abandoned symbols so a second target-
fact authority cannot return merely to satisfy a self-test.

| Signal | Checkpoint 77 | Cumulative change | Slice change from checkpoint 76 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,190 | +214 | **-71** |
| Nonblank production lines | 98,846 | **-238** | **-63** |
| Production implementation lines | 91,098 | **-352** | **-47** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 60 | +26 / +9 | 0 / 0 |
| Abandoned kernel-target/test-only profile symbols | **0** | n/a | **-8** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild after changing the widely
included capability header; all 16 capability-contract and architecture-
boundary tests; and all 4,776 nonphysical tests over the five emulated targets
at `-j16` in 202.48 seconds, including all 2,918 simulator tests. One test was
renamed to match its unchanged authoritative subject; no test was removed,
disabled, or replaced, and the inventory is unchanged. In accordance with the
reduced physical-test cadence, no physical gfx1201 test was run for this slice.

This checkpoint strengthens Sections 14.1, 14.2, 14.7, and 14.8. The target
contract now describes only facts that production actually consumes, while
descriptor-selected state stays with its real program/resource owners. The
slice is purely deletion-driven and increases cumulative production shrinkage
from 305 to 352 implementation lines. That is still not a material reduction
for a 91,450-line baseline, so Section 14.8 remains open. The remaining broad
placement and mutation transactions, the extension-exercise sufficiency audit,
larger legacy harvesting, and the independent completion audit also remain
open. The goal therefore remains active.

### 16.79 Convergence checkpoint 78: test-owned InlineShadow transaction oracle

The InlineShadow model deep read distinguished executable report contracts from
host reference models. The versioned release-claim planner and release-
transaction ordering checker were compiled into the production report-contract
header, but no production component called them. GPU emission implements the
actual transaction; only three focused host tests consumed these C++ functions
to check claim-state and event-order examples. Their claim/result/event types
were likewise test-only.

The complete release-claim and transaction oracle now lives in
`consan_inline_model_test_support.h`. The two tests that consume this part of
the oracle include it explicitly. Production retains the real report ABI
records, version encoding helpers that report processing consumes, and the
stable release identity shared with later report contracts. It no longer
exports a parallel host implementation of the GPU transaction as though it
were a runtime component. The architecture gate prevents the six root oracle
symbols and their dependent types from returning to the production InlineShadow
report model.

| Signal | Checkpoint 78 | Cumulative change | Slice change from checkpoint 77 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 105,034 | +58 | **-156** |
| Nonblank production lines | 98,701 | **-383** | **-145** |
| Production implementation lines | 90,958 | **-492** | **-140** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 60 | +26 / +9 | 0 / 0 |
| Host release-transaction oracle roots in production | **0** | n/a | **-6** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild across the report-contract header
fan-out and all four release-claim, release-transaction, adversarial, and
architecture-boundary tests. Checkpoint 77 immediately before this source-only
ownership change passed all 4,776 nonphysical tests, including 2,918 simulator
tests; no test was removed, disabled, renamed, or replaced in this slice. No
physical gfx1201 test was run.

This checkpoint strengthens Sections 14.1, 14.3, 14.7, and 14.8. A mode-local
production contract no longer contains a host-only duplicate of emitted GPU
semantics, while the independent oracle and all of its coverage remain clearly
test-owned. Cumulative production shrinkage rises from 352 to 492
implementation lines. The same deep read found additional host-only causal and
qualification models still embedded in the InlineShadow production contract;
those are concrete follow-on harvesting candidates. Section 14.8 is not yet
materially satisfied, and the other completion-audit gaps remain open. The goal
therefore remains active.

### 16.80 Convergence checkpoint 79: test-owned InlineShadow causal and evidence oracles

The follow-on InlineShadow deep read traced the causal-snapshot constructor and
the end-to-end evidence qualifier through every caller. Neither participates in
GPU construction, report decoding, runtime analysis, or validation. The
snapshot constructor is a host reference implementation exercised by causal
model tests; the qualifier combines synthetic releases, tokens, accesses, and
counters solely to judge focused test fixtures. Their input, expectation,
failure, and result types likewise had no production consumer.

Both complete oracle regions now live beside the release-transaction oracle in
`consan_inline_model_test_support.h`. The atomic-model tests include that
test-owned contract directly. Production retains the real report ABI, evidence
kind, stable-release identity, snapshot validation/import operations consumed
by production shadow models, and the encoding helpers used by emission and
report processing. This is therefore an ownership correction and deletion from
the production surface, not removal of executable evidence semantics or test
coverage. The architecture gate rejects the causal-construction and evidence-
qualification roots from the production InlineShadow report contract.

| Signal | Checkpoint 79 | Cumulative change | Slice change from checkpoint 78 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 104,719 | **-257** | **-315** |
| Nonblank production lines | 98,405 | **-679** | **-296** |
| Production implementation lines | 90,669 | **-781** | **-289** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 60 | +26 / +9 | 0 / 0 |
| Host causal/evidence oracle roots in production | **0** | n/a | **-6** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild and all 11 causal-snapshot,
causal-import, evidence-qualification, release-transaction, and architecture-
boundary tests. Checkpoint 77's immediately preceding periodic gate passed all
4,776 nonphysical tests over five emulated targets, including all 2,918
simulator tests. No test was removed, disabled, renamed, or replaced. In
accordance with the reduced physical-test cadence, no physical gfx1201 test was
run.

This checkpoint strengthens Sections 14.1, 14.3, 14.7, and 14.8. Three related
host reference models now form one explicitly test-owned InlineShadow support
surface rather than enlarging a production report contract. Cumulative
production shrinkage increases from 492 to 781 implementation lines. The
remaining snapshot validation/import helpers are production dependencies and
must be assessed together with their shadow-model consumers rather than moved
by association. Material shrinkage, the larger placement and mutation buses,
and the independent completion audit remain open; the goal therefore remains
active.

### 16.81 Convergence checkpoint 80: test-owned InlineShadow import and token oracles

The next InlineShadow deep read followed causal import and acquired-token
ordering beyond their adjacent types to every caller. Release-snapshot
classification and deferred-diagnostic filtering are production report-decoder
mechanisms and remain in production. In contrast, the causal-import planner,
token publication model, immediate acquired-order predicates, and stable-token
ordering model had no production caller. They are host references used by
focused tests to check examples of GPU transactions emitted separately.

Those six oracle roots and their test-only plan/result types now join the other
InlineShadow reference models in `consan_inline_model_test_support.h`. The
first rebuild exposed that the epoch-to-token conversion is also consumed by
the production deferred-diagnostic filter; it remains in the production shadow
model and the test oracle consumes that one shared definition. Production also
retains release and token snapshot classification, acquired-token lookup, and
the actual deferred-diagnostic filter. The architecture gate rejects the moved
causal-import roots from the report model and the moved publication/order roots
from the production shadow model.

| Signal | Checkpoint 80 | Cumulative change | Slice change from checkpoint 79 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 104,507 | **-469** | **-212** |
| Nonblank production lines | 98,203 | **-881** | **-202** |
| Production implementation lines | 90,494 | **-956** | **-175** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 60 | +26 / +9 | 0 / 0 |
| Host causal-import/token-order oracle roots in production | **0** | n/a | **-6** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild of the broad shadow-model header
fan-out and all nine causal-import, acquired-token lookup/publication/order,
stable-token, and architecture-boundary tests. Checkpoint 77's preceding
periodic gate passed all 4,776 nonphysical tests over five emulated targets,
including all 2,918 simulator tests. No test was removed, disabled, renamed, or
replaced. No physical gfx1201 test was run.

This checkpoint strengthens Sections 14.1, 14.3, 14.7, and 14.8. The production
InlineShadow contracts retain only behavior used by construction or report
processing, while independent executable specifications remain available to
tests under explicit test ownership. Cumulative production shrinkage reaches
956 implementation lines. Material shrinkage is improving but remains
unproven at the whole-refactoring scale; broad placement and mutation buses and
the independent completion audit remain open. The goal therefore remains
active.

### 16.82 Convergence checkpoint 81: Sampled publication/replay models are test components

The mixed `consan_moi_model.cpp` deep read separated production Sampled
authorities from host specification models. Sync metadata encoding is consumed
by GPU emission; metadata and snapshot decoding, pending-acquire
classification, and atomic attachment matching are consumed by the runtime
report decoder. Those remain production. Nine other functions -- bounded sync
publication, access/window publication, entry/snapshot/window replay, and
causal claim/commit/abort -- had callers only in two unit-test files. They model
device publication and replay examples but do not implement either production
lowering or runtime report processing.

The complete closed subsystem now compiles as the Sampled-owned test
translation unit `consan_sampled_model_test_support.cpp`, behind the explicit
`consan_sampled_model_test_support.h` contract. The two consuming tests include
that contract directly. Its definitions and declarations are deleted from the
production model and broad shadow-model contract; production ABI records remain
because the real emitter and decoder use them. The architecture gate rejects
all nine roots from both former production owners.

| Signal | Checkpoint 81 | Cumulative change | Slice change from checkpoint 80 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 103,988 | **-988** | **-519** |
| Nonblank production lines | 97,718 | **-1,366** | **-485** |
| Production implementation lines | 90,016 | **-1,434** | **-478** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 316 / 60 | +26 / +9 | 0 / 0 |
| Sampled publication/replay oracle roots in production | **0** | n/a | **-9** |
| Test inventory | **5,411** | **+66** | 0[^checkpoint-81-inventory] |

No test definition was added by this slice. The complete `-j16` rebuild passed,
as did all 15 focused Sampled publication, replay, claim, selection, and
architecture-boundary tests. The periodic gate then passed every registered
nonphysical row in 203.45 seconds, including all 2,918 simulator tests over
gfx942, gfx950, gfx1100, gfx1250, and gfx1201. No test was removed, disabled,
renamed, or replaced, and no physical gfx1201 test was run.

[^checkpoint-81-inventory]: A fresh source-definition audit and clean
    CMake/GTest rediscovery at checkpoint 84 establish that checkpoint 81 did
    not add a test: there are 5,411 actual tests, comprising 4,776 nonphysical
    and 635 physical rows. The original 5,412/4,777 count included one stale
    generated registration and is corrected here. No `TEST`, `TEST_P`, test
    instantiation, or `add_test` definition changed from checkpoint 80 through
    checkpoint 84.

This checkpoint strengthens Sections 14.1, 14.3, 14.7 through 14.9. Sampled
reference behavior is physically skippable from production and has one
mode-named test owner, while its genuinely shared ABI and production encode/
decode authorities remain single implementations. Cumulative shrinkage reaches
1,434 production implementation lines. The neighboring Record/Replay capture
model has the same apparent ownership problem but shares internal replay
helpers with the runtime analyzer and requires a separate dependency-closure
migration. Broad placement and mutation buses and the independent completion
audit remain open. The goal therefore remains active.

### 16.83 Convergence checkpoint 82: Record/Replay capture is a test component

The Record/Replay capture dependency-closure read separated the actual runtime
analyzer from a second host specification model. Production report processing
uses access/barrier/atomic/fence replay, untouched fixed-capacity-slot
classification, and wave-wide atomic-outcome normalization. It does not use
the compact trace builder, bounded complete-epoch selector, selected-window
replay wrapper, or any of their compact-trace and capture types. The latter
three roots were called only by focused tests; even their five ABI-size
assertions lived in the common test convenience header rather than proving a
production consumer.

The three oracle roots, nine private types, and five size assertions now form
the explicit test contract `consan_record_replay_model_test_support.h` and its
single implementation translation unit. The two consuming test files include
that contract directly. Production retains the real report replay analyzer
and gives its three shared normalization predicates narrow Record/Replay names;
the test oracle calls those definitions rather than copying their logic. The
superseded declarations and types are deleted from the production report
contract, and the architecture gate rejects their return.

| Signal | Checkpoint 82 | Cumulative change | Slice change from checkpoint 81 |
| --- | ---: | ---: | ---: |
| Production files | 264 | +35 | 0 |
| Physical production lines | 103,067 | **-1,909** | **-921** |
| Nonblank production lines | 96,839 | **-2,245** | **-879** |
| Production implementation lines | 89,174 | **-2,276** | **-842** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0[^checkpoint-82-point-count] |
| Record/Replay capture oracle roots/types in production | **0** | n/a | **-12** |
| Test inventory | **5,411** | **+66** | 0 |

[^checkpoint-82-point-count]: Recounting the checkpoint-81 tree finds 326
    `ConSanMoiOperatingPoint` occurrences, not the 316 copied into its table.
    This row corrects that stale ledger value; checkpoint 82 adds none.

Validation includes a complete `-j16` rebuild of the production report
contract and every ConSan test translation unit, all 97 Record/Replay capture,
construction, runtime-replay, and hook tests, and the architecture-boundary
test. Checkpoint 81's immediately preceding periodic gate passed every
registered nonphysical row, including all 2,918 simulator tests over five
targets. The corrected authoritative inventory is 5,411 tests: 4,776
nonphysical and 635 physical. No test was removed, disabled, renamed, or
replaced, and no physical gfx1201 test was run.

This checkpoint strengthens Sections 14.1, 14.3, and 14.7 through 14.9.
Record/Replay's independent capture specification is physically skippable from
production, while the real mode-specific runtime analyzer retains one shared
normalization authority. Cumulative production shrinkage reaches 2,276
implementation lines. Material whole-refactoring shrinkage, broad placement
and mutation buses, remaining target/mode locality, and the independent
completion audit remain open. The goal therefore remains active.

### 16.84 Convergence checkpoint 83: mode-owned host report models

After removing the test-only models, the remaining generic
`consan_moi_model.cpp` had exactly two independent production responsibilities.
Seven Sampled definitions encode, decode, snapshot-classify, and qualify
Sampled metadata. The rest normalizes and replays Record/Replay reports,
including exact-byte shadow state and causal atomic/fence ordering. A complete
definition-and-caller trace found no mechanism or state shared between those
regions; their only common dependency is the public report contract.

The production build graph now compiles `consan_moi_sampled_model.cpp` and
`consan_moi_record_replay_model.cpp` as the two explicit owners. The former has
only the Sampled metadata and qualification definitions. The latter has only
Record/Replay normalization, causal replay, exact shadow, diagnostics, and
runtime analysis. The old generic filename remains compiled as a comment-only
tombstone because of the operational no-delete rule. Narrowing each translation
unit's includes also deleted the monolith's unrelated analysis, patching,
builder, ISA, and standard-library include closure. No implementation was
copied, wrapped, or left behind.

The architecture gate requires the generic file to remain inert, rejects
Sampled identifiers from the Record/Replay owner and Record/Replay identifiers
from the Sampled owner, applies the reviewed mode-switch budgets to the named
owners, and continues to reject the test-only model roots from their relevant
production owner.

| Signal | Checkpoint 83 | Cumulative change | Slice change from checkpoint 82 |
| --- | ---: | ---: | ---: |
| Production files | 266 | +37 | +2 named owners |
| Physical production lines | 103,058 | **-1,918** | **-9** |
| Nonblank production lines | 96,825 | **-2,259** | **-14** |
| Production implementation lines | 89,149 | **-2,301** | **-25** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Mode implementations in generic report-model file | **0** | n/a | **-2 regions** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild and all 161 focused Sampled and
Record/Replay construction, report-model, runtime-replay, and hook tests. The
architecture-boundary test then passed with the new cross-mode and tombstone
rules. Checkpoint 81's preceding periodic gate passed every then-registered
nonphysical row, including all 2,918 simulator tests over five targets. The
corrected actual inventory is 4,776 nonphysical and 635 physical tests. No test
was removed, disabled, renamed, or replaced, and no physical gfx1201 test was
run.

This checkpoint strengthens Sections 14.1, 14.3, 14.6 through 14.9. A reader
of either runtime report model can now skip the other mode completely, and the
build and boundary graphs state the same ownership as the implementation.
Cumulative production shrinkage reaches 2,301 implementation lines. The broad
placement and mutation buses, further architecture/mode locality, material
whole-refactoring shrinkage, and the independent completion audit remain open.
The goal therefore remains active.

### 16.85 Convergence checkpoint 84: physically local report-model contracts

The report-contract deep read found four independent ownership regions inside
the 1,104-line `consan_moi_shadow_models.h.inc`. Bit-field construction, access
kind conflict rules, four-byte report-cell geometry, and range projection are
shared mechanisms. Exact packed entries, byte provenance, sparse exact replay,
and InlineShadow's exact evidence form an exact-subset region. Sampled owns its
watchpoint ABI, stable-snapshot classification, causal selection, and metadata
codec declarations. Record/Replay owns causal-token ordering, atomic
publication, unpublished-slot classification, and its replay entry points.
There is no state or implementation shared directly between the Sampled and
Record/Replay regions.

The public report contract now includes four explicit fragments in dependency
order: `consan_moi_shadow_common.h.inc`,
`consan_moi_exact_shadow_model.h.inc`,
`consan_moi_record_replay_model.h.inc`, and
`consan_moi_sampled_model.h.inc`. The former generic filename is a
comment-only tombstone. The shared owner now names the report-wide cell
geometry as `consan_moi_shadow_cell`; Sampled, Record/Replay, shared lowering,
runtime provenance, and exact emission no longer reach through an
InlineShadow-looking `consan_moi_exact_shadow::granule_*` namespace for this
common ABI fact.

The architecture gate requires the old mixed contract to remain inert,
prevents the common fragment from acquiring any mode, rejects Sampled and
Record/Replay policy from the exact-subset owner, and rejects cross-mode
identifiers from both named mode contracts. Existing bans on retired test
oracles now point at their actual mode/exact owners. No definition is copied or
wrapped; every definition from the old file has exactly one surviving owner.

| Signal | Checkpoint 84 | Cumulative change | Slice change from checkpoint 83 |
| --- | ---: | ---: | ---: |
| Production files | 270 | +41 | +4 explicit owners, old filename retained inert |
| Physical production lines | 103,082 | **-1,894** | +24 |
| Nonblank production lines | 96,842 | **-2,242** | +17 |
| Production implementation lines | 89,154 | **-2,296** | +5 contract/include structure |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Mode regions in the retired mixed contract | **0** | n/a | **-3 regions** |
| Exact-namespace references to shared cell geometry | **0** | n/a | **-62** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild of the full report-contract fan
out and every ConSan test translation unit. The architecture-boundary gate
passed, followed by all **4,776/4,776 nonphysical tests** in 203.75 seconds,
including all 2,918 simulator rows over gfx942, gfx950, gfx1100, gfx1250, and
gfx1201. No test definition or inventory changed, and no physical gfx1201 test
was run.

This checkpoint strengthens Sections 14.1, 14.3, and 14.6 through 14.9. A
reader can now omit the Sampled or Record/Replay report contract physically,
and both consume one shared cell/range mechanism rather than depending on one
another or duplicating it. The exact-subset fragment remains the next internal
locality seam: its deep read distinguishes shared exact replay, generic
dispatch-preload planning, and InlineShadow-only snapshot/token behavior.
Those must be assigned honest owners in a later slice rather than treating the
new filename as final. Broad placement and mutation buses, further target and
mode locality, material whole-refactoring shrinkage, and the independent
completion audit remain open. The goal therefore remains active.

### 16.86 Convergence checkpoint 85: exact-subset, InlineShadow, and dispatch-preload ownership

The follow-up deep read of the exact-subset fragment traced every declaration
through production construction, runtime decoding, host replay, and tests. The
735-line fragment still combined three independent responsibilities. Packed
exact entries, byte provenance, sparse range replay, and conflict mechanics are
shared by the exact mode subset. Stable exact snapshots and acquired-token
filtering are InlineShadow runtime behavior. AMDHSA dispatch-ID preload
planning is lowerer infrastructure used by placement and prologue emission; it
does not consume report state or mode policy.

Those responsibilities now have three physical owners. The shared
`consan_moi_exact_shadow_model.h.inc` contains only exact-subset mechanics.
`consan_moi_inline_exact_model.h.inc` contains only the production
InlineShadow snapshot and deferred-token contract and is imported immediately
after its shared exact dependency. `consan_moi_dispatch_preload.h` is a
standalone target-neutral planning contract included by lowerer internals, not
by the public MOI entry point or report contract. Record/Replay continues to
consume the shared exact owner without acquiring InlineShadow declarations,
and runtime InlineShadow decoding no longer makes generic preload planning
visible to every report-contract consumer.

The ownership trace also found three host reference oracles with no production
caller: bounded Inline workgroup-key packing, packed-entry conflict, and
packed-byte-cell conflict. They now live in
`consan_inline_model_test_support.h`, beside the other InlineShadow GPU
transaction oracles. Their existing focused tests continue to compare emitted
semantics against those independent host models, but the models no longer
inflate or broaden the production contract.

The architecture gate rejects every mode and dispatch-preload identifier from
the shared exact owner, rejects other-mode and preload planning from the
InlineShadow exact owner, prevents the dispatch planner from acquiring report
or mode knowledge, and prevents that lowerer-private planner from leaking
through the public MOI entry point. It also prevents all three harvested
oracles from returning to production.

| Signal | Checkpoint 85 | Cumulative change | Slice change from checkpoint 84 |
| --- | ---: | ---: | ---: |
| Production files | 272 | +43 | +2 explicit owners |
| Physical production lines | 103,024 | **-1,952** | **-58** |
| Nonblank production lines | 96,783 | **-2,301** | **-59** |
| Production implementation lines | 89,101 | **-2,349** | **-53** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | 168 / 52 | **-108 / -5** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Mode/preload identifiers in the shared exact owner | **0** | n/a | converged |
| Host-only exact/Inline oracles in production | **0** | n/a | **-3** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild and 126 focused dispatch-preload,
Inline workgroup/snapshot/token, shared exact replay, Record/Replay, adversarial,
and architecture-boundary tests. The immediately preceding checkpoint passed
all 4,776 nonphysical tests, including all 2,918 simulator rows over the five
targets; this slice changes ownership and test-oracle visibility without
changing production definitions or generated GPU behavior. No test was
removed, renamed, disabled, or replaced, and no physical gfx1201 test was run.

This checkpoint strengthens Sections 14.1, 14.3, and 14.5 through 14.9. The
report contract now exposes one shared exact mechanism followed by honest
mode-owned contracts, while generic lowerer planning is outside that graph and
three superseded production oracles are harvested. Broad placement and
mutation buses, further target and mode locality, material whole-refactoring
shrinkage, and the independent completion audit remain open. The goal
therefore remains active.

### 16.87 Convergence checkpoint 86: one program-analysis product

A deep read of code-object and synchronization analysis traced its complete
forward output through composition, fault planning, perturbation planning, and
independent final validation. The analysis boundary accepted the entire mutable
`ConSanTransformArtifacts` transaction even though this region touched only six
outputs: the immutable program inventory, fault sites, barrier-move
destinations, outcome, warnings, and errors. Independent validation consequently
constructed a dummy whole transformation transaction merely to obtain pristine
proof inventories.

Those six outputs now form the explicit `ConSanProgramAnalysisResult`. Program
analysis and synchronization analysis publish only that product, while the
composition layer is the single adapter that moves it into the subsequent
transformation transaction. Independent mutation and perturbation validation
project their proof inventories directly from the analysis product and no
longer manufacture a dummy transaction. The architecture gate prevents the
broad transformation transaction from returning to either analysis component
or the independent inventory rederiver, and pins the complete named output
contract.

The first focused gate caught an important failure-path detail in the boundary
migration. Even unsuccessful parsing must publish the identity-bearing empty
inventory view so pipeline accounting can distinguish an attempted invalid
analysis from a stage that was never entered. The analysis owner now publishes
that initial view before its first fallible operation. The three existing exact
regressions for invalid code objects and malformed metadata notes caught the
mistake and pass after the owner-local correction; no compatibility path was
added.

| Signal | Checkpoint 86 | Cumulative change | Slice change from checkpoint 85 |
| --- | ---: | ---: | ---: |
| Production files | 272 | +43 | 0 |
| Physical production lines | 103,051 | **-1,925** | +27 |
| Nonblank production lines | 96,809 | **-2,275** | +26 |
| Production implementation lines | 89,121 | **-2,329** | +20 |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **154 / 47** | **-122 / -10** | **-14 / -5** |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Analysis/validation owners accepting the broad transaction | **0 / 5** | n/a | **-5 owners** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild; the 524-test program-analysis,
inventory, pipeline, and architecture-boundary focused gate; and the complete
4,776-test nonphysical matrix at `-j16` in 198.48 seconds, including all 2,918
simulator rows over gfx942, gfx950, gfx1250, gfx1100, and gfx1201. No test was
added, removed, renamed, disabled, or replaced. In accordance with the reduced
physical-test cadence, no physical gfx1201 test was run for this slice.

This checkpoint strengthens Sections 14.1, 14.5, 14.7, and 14.9. Its 20-line
contract cost removes five whole component consumers from the broad mutable bus
and eliminates the dummy validation transaction. The immediately preceding
slice was deletion-bearing, so this one permitted contract investment; the
next convergence slice must cash in deletion or sharing rather than compound
local growth. Final validation, composition, and mutation still contain broad
transaction surfaces, and material whole-refactoring shrinkage plus the
independent completion audit remain open. The goal therefore remains active.

### 16.88 Convergence checkpoint 87: test-owned Sampled selection and encoding oracles

The caller trace of the remaining Sampled host model separated report/runtime
mechanics from three independent device-reference operations. Production host
code decodes watchpoint entries, classifies stable snapshots and conflicts, and
checks causal attachment. It never selects a causal window with the host hash
or constructs a complete packed watchpoint entry: those operations are used by
tests to calculate expected device behavior and to seed synthetic hook reports.
The shared byte-count field encoder is different; the production GPU emitter
uses it directly and it therefore remains in the Sampled production contract.

The causal hash, whole causal-window selector, and complete watchpoint encoder
now live in `consan_sampled_model_test_support.h`. Both the core Sampled tests
and the separately linked HSA-hook tests consume that explicit test component.
No production replacement or forwarding wrapper remains. The architecture gate
prevents all three complete host oracles from returning to the production
Sampled report model while deliberately permitting the byte-count primitive
shared with device emission.

| Signal | Checkpoint 87 | Cumulative change | Slice change from checkpoint 86 |
| --- | ---: | ---: | ---: |
| Production files | 272 | +43 | 0 |
| Physical production lines | 103,002 | **-1,974** | **-49** |
| Nonblank production lines | 96,763 | **-2,321** | **-46** |
| Production implementation lines | 89,075 | **-2,375** | **-46** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **154 / 47** | **-122 / -10** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Host-only Sampled selector/encoder operations in production | **0** | n/a | **-3** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild, including both the core test
binary and the separately linked hook-test binary, and 171 focused Sampled,
hook, and architecture-boundary tests. The exact inventory remains 5,411 tests:
4,776 nonphysical tests, including 2,918 simulator rows, plus 635 physical
gfx1201 tests. The immediately preceding checkpoint passed the complete 4,776-
test nonphysical matrix on the same tree ancestry; this slice deletes unused
production inline definitions without changing the production device emitter,
decoder, analyzer, or generated code. No physical gfx1201 test was run.

This deletion-bearing checkpoint strengthens Sections 14.3, 14.7, and 14.8.
Sampled production now contains only behavior used by its emitter or report
runtime, while reference construction stays independently test-owned. Broader
pipeline and validation transactions, further target and mode locality,
material whole-refactoring shrinkage, and the independent completion audit
remain open. The goal therefore remains active.

### 16.89 Convergence checkpoint 88: test-owned exact byte-cell emission oracles

The follow-up caller trace of the shared exact-byte model distinguished host
runtime mechanics from complete reference encoders. Production needs the
byte-provenance decoder, exact conflict semantics, maximum cell geometry, and
two compile-time lookup tables consumed by the InlineShadow GPU emitter. It
does not call the host function that constructs a complete provenance word or
the host function that decomposes an unaligned access into per-cell masks.
Those two functions are used only as independent expected-value calculations
for emitted InlineShadow behavior.

The provenance packer and byte-cell mask decomposition now live beside the
existing InlineShadow transaction and packed-conflict test oracles in
`consan_inline_model_test_support.h`. The shared exact production contract
retains the primitives actually consumed by Record/Replay, InlineShadow
decoding, validation, and GPU emission. No production forwarding definition
remains, and the architecture gate prevents both complete emission oracles
from returning to the shared exact owner.

| Signal | Checkpoint 88 | Cumulative change | Slice change from checkpoint 87 |
| --- | ---: | ---: | ---: |
| Production files | 272 | +43 | 0 |
| Physical production lines | 102,969 | **-2,007** | **-33** |
| Nonblank production lines | 96,732 | **-2,352** | **-31** |
| Production implementation lines | 89,044 | **-2,406** | **-31** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **154 / 47** | **-122 / -10** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Host-only exact byte-cell emission operations in production | **0** | n/a | **-2** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild and all 23 focused exact-byte,
exact-shadow, sparse-replay, InlineShadow snapshot, and architecture-boundary
tests. The immediately preceding checkpoint exercised the full Sampled/hook
consumer set, and checkpoint 86 passed the complete 4,776-test nonphysical
matrix on the same tree ancestry. This slice removes unused production inline
definitions without changing the production lookup tables, decoder, validator,
or emitted GPU program. No test inventory or physical gfx1201 run changed.

This deletion-bearing checkpoint strengthens Sections 14.3, 14.7, and 14.8.
The shared exact contract is smaller and more honest about the runtime
mechanisms it owns, while device-reference construction stays test-owned.
Broad pipeline and validation transactions, further target and mode locality,
material whole-refactoring shrinkage, and the independent completion audit
remain open. The goal therefore remains active.

### 16.90 Convergence checkpoint 89: complete production replay and snapshot operations

The mode-model caller trace next followed every Record/Replay replay overload
and InlineShadow release-snapshot predicate. Production report analysis always
supplies the complete Record/Replay product: access, barrier, atomic, and fence
events plus diagnostic and exact-shadow storage. Three shorter overloads merely
inserted empty spans for tests exercising a prefix of that product. Similarly,
production InlineShadow readers classify the complete version-bracketed release
payload; the predicate that checks only the two version words is a partial test
convenience and is never accepted as production ordering authority.

Production Record/Replay now declares and implements exactly one complete
replay boundary. Its three prefix conveniences live in
`consan_record_replay_model_test_support.h`, and all core and separately linked
hook tests consume that mode-local test component. The partial InlineShadow
version-envelope predicate now lives in `consan_inline_model_test_support.h`.
No production forwarding overload or predicate remains. Boundary checks cap
the production Record/Replay declaration and implementation at one replay
operation and prevent the partial InlineShadow predicate from returning.

| Signal | Checkpoint 89 | Cumulative change | Slice change from checkpoint 88 |
| --- | ---: | ---: | ---: |
| Production files | 272 | +43 | 0 |
| Physical production lines | 102,909 | **-2,067** | **-60** |
| Nonblank production lines | 96,679 | **-2,405** | **-53** |
| Production implementation lines | 88,993 | **-2,457** | **-51** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **154 / 47** | **-122 / -10** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Production Record/Replay replay declarations / implementations | **1 / 1** | n/a | **-3 / -3** |
| Partial InlineShadow snapshot predicates in production | **0** | n/a | **-1** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild, including the separate HSA-hook
binary, and all 190 focused Record/Replay, InlineShadow release-snapshot, hook,
and architecture-boundary tests. Checkpoint 88 immediately before this slice
also passed the complete exact/sparse shared-model gate, while checkpoint 86
passed all 4,776 nonphysical tests on the same ancestry. This slice changes no
production replay algorithm or report format; it deletes only incomplete
convenience entry points. No test inventory or physical gfx1201 run changed.

This deletion-bearing checkpoint strengthens Sections 14.3, 14.5, 14.7, and
14.8. Each affected mode now exposes complete production operations while its
test-only prefix/reference API remains physically mode-local. Broad pipeline
and validation transactions, further target and mode locality, material
whole-refactoring shrinkage, and the independent completion audit remain open.
The goal therefore remains active.

### 16.91 Convergence checkpoint 90: causal-state-aware Record/Replay release

The final shortened Record/Replay synchronization overload followed the same
production-versus-test split exposed by checkpoint 89. Production replay
always has the current acquired-epoch-token product and publishes it together
with the releasing owner's epoch. The overload that silently substituted an
empty token span had only fixture callers representing a release with no
imported causal state; it was not a distinct production operation.

Production Record/Replay now declares and implements only the complete atomic
release boundary. The empty-token convenience lives in
`consan_record_replay_model_test_support.h`, beside the other test-owned prefix
operations, and forwards explicitly to that complete boundary. A structural
check caps the production header at one atomic-release operation so that an
incomplete forwarding facade cannot quietly return.

| Signal | Checkpoint 90 | Cumulative change | Slice change from checkpoint 89 |
| --- | ---: | ---: | ---: |
| Production files | 272 | +43 | 0 |
| Physical production lines | 102,893 | **-2,083** | **-16** |
| Nonblank production lines | 96,665 | **-2,419** | **-14** |
| Production implementation lines | 88,979 | **-2,471** | **-14** |
| `MoiOptions` references / files | **0 / 0** | **-87 / -25** | 0 / 0 |
| `ConSanTransformArtifacts` references / files | **154 / 47** | **-122 / -10** | 0 / 0 |
| `ConSanPatchInfo` references / files | 209 / 30 | +9 / +2 | 0 / 0 |
| `ConSanMoiOperatingPoint` references / files | 326 / 60 | +36 / +9 | 0 / 0 |
| Production Record/Replay atomic-release operations | **1** | n/a | **-1** |
| Test inventory | **5,411** | **+66** | 0 |

Validation includes a complete `-j16` rebuild and all 27 focused
Record/Replay atomic, causal, acquire/release, and architecture-boundary tests.
Checkpoint 89 immediately before this slice passed the broader 190-test
Record/Replay, InlineShadow, hook, and boundary gate, while checkpoint 86
passed the complete 4,776-test nonphysical matrix. The slice changes no replay
algorithm or wire format and removes no behavior or inventory. No physical
gfx1201 test was run.

This deletion-bearing checkpoint strengthens Sections 14.3, 14.5, 14.7, and
14.8 by requiring the production mode boundary to receive its complete causal
input instead of manufacturing an implicit empty product. Broad pipeline and
validation transactions, further target and mode locality, material
whole-refactoring shrinkage, and the independent completion audit remain open.
The goal therefore remains active.
