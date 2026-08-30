# ConSan fifth refactoring: deepen the component architecture

This document is the initial working charter for ConSan's fifth refactoring.
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

No destination class diagram, stage sequence, or small fixed list of mechanical
extractions is established here yet. This document is intended to absorb
additional investigative angles and a substantially more ambitious mandate
before implementation stages are chosen.

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

# Part II: open charter for the fifth refactoring

## 10. Mandate

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

### 10.1 Create deletion opportunities and reap them

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

The detailed work plan and convergence gates will be added only after the
charter has absorbed the next investigative requests. The following invariants
already carry forward:

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

## 14. Sections intentionally left open

The following sections will be developed after further investigation and user
direction. Their omission is deliberate; prematurely filling them would turn
the initial pressure-point list into an unjustified destination design.

- Additional audit angles and revised component hypotheses.
- Alternative destination architectures.
- Experiments and vertical slices used to choose among them.
- Migration workstreams and stage ordering.
- Component-level and full-matrix test strategy.
- Exact quantitative baseline, recurring measurements, and the final
  production-code reduction target.
- Definition of success and final convergence criteria.
- Completion and handoff records.
