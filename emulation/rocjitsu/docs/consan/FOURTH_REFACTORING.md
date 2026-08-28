# ConSan fourth refactoring: enforce the component architecture

This document is the working charter for ConSan's fourth refactoring. It begins
with the de-facto component structure found in the current implementation, then
identifies where the code does not respect that structure, and finally proposes
an ordered route for turning the effective architecture into enforced
boundaries.

The measurements below use commit
`01e6a6f1a23997866ff7a318966f6d809f9de6b8`. They cover production code under:

- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/patch/consan/`
- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/hooks/consan/`

Tests, documentation, build files, blank lines, and comments are excluded. The
result is **84,041 implementation lines in 83 files**. The categories are exact
whole-file sums by dominant responsibility; they do not imply that every line
in a file belongs exclusively to that responsibility.

## 1. De-facto components

| Dominant responsibility | Implementation lines | Share | Files |
| --- | ---: | ---: | ---: |
| Shared MOI planning and lowering | 23,615 | 28.1% | 17 |
| HSA runtime adapter and report lifecycle | 9,059 | 10.8% | 8 |
| Mutation, composition, and final validation | 7,871 | 9.4% | 3 |
| Inventory and semantic policy | 7,253 | 8.6% | 5 |
| SuperCollider lowering | 6,839 | 8.1% | 4 |
| Sampled engine | 6,050 | 7.2% | 4 |
| Inline Shadow engine | 5,775 | 6.9% | 4 |
| Shared mechanisms and pipeline | 5,414 | 6.4% | 13 |
| Evidence/report model and planning | 5,288 | 6.3% | 7 |
| Core contracts | 4,145 | 4.9% | 14 |
| Record/Replay-specific lowering | 2,732 | 3.3% | 4 |
| **Total** | **84,041** | **100%** | **83** |

This is not an 84,000-line undifferentiated implementation. The code already
has a recognizable architecture, and the main data flow is substantially the
one intended by the design documents:

```text
configuration and runtime facts
              |
              v
target profile -> immutable program inventory
              |
              v
semantic policy -> observation plan -> coverage ledger
              |              \
              v               -> evidence requirements
resource planning
              |
              v
SuperCollider or MOI lowering
              |
              v
placement and patch metadata -> final validation
              |
              v
runtime binding -> snapshot/replay/analysis -> trust and diagnostics
```

The upper half of this pipeline is already comparatively well factored. The
lowering and runtime halves have identifiable responsibilities too, but their
interfaces are frequently implicit, over-broad, or reconstructed from another
component's internal representation.

### 1.1 Components that are already strong

The fourth refactoring should preserve and build on these parts rather than
redesigning them gratuitously.

#### Immutable inventory

`ProgramInventory` is a genuine shared semantic boundary. Its immutable
`shared_ptr<const Storage>` representation is distinct from its builder, and
consumers receive read-only views. It is a good model for other cross-stage
products.

#### Central target profile

`ConSanTargetProfile` centralizes the supported architecture matrix in a typed
five-row table. This is the right authority for broad capability facts. The
remaining problem is not the existence of target-specific facts; it is exact
instruction admission leaking above and below the classifier boundary.

#### Semantic policy and observation planning

Access, barrier, and atomic/fence policy are separate compiled functions that
produce `ConSanObservationPlan` entries and typed reasons. The
`ConSanCoverageLedger` is also a separate concept rather than an incidental
counter in a lowerer. These are sound component boundaries even though their
inputs and the later publication of their results still need correction.

#### Evidence and resource planning

Evidence sizing and report planning are focused in
`consan_moi_report_plan.cpp`, while resource planning has a small, focused
implementation in `consan_resource.cpp`. Both express useful intermediate
products rather than asking emitters to rediscover global decisions.

#### Narrow immutable emission plans

Several newer lowerer paths already use narrow immutable plans, including
`MoiPrivateEpochPrologueEmissionPlan`,
`MoiOwnerEpochPrologueEmissionPlan`, and
`MoiRecordEventEmissionPlan`. These plans are the appropriate pattern for
continuing the decomposition: values grouped by an invariant and computed at
one authority, not arbitrary subsets of a large options object.

#### Trust evaluation

The runtime side contains a pure trust-evaluation core. Separating the trust
decision from report allocation and HSA lifecycle management was an important
step; the surrounding snapshot, decode, analysis, and rendering path should now
be brought to the same standard.

#### Focused tests

There are direct tests for requests, target profiles, inventory, observation
plans, policy, evidence requirements, resources, and pipeline behavior. The
architecture is therefore not only visible in names: significant portions of
it can already be exercised without treating the complete transform as a black
box.

The principal implementation anchors for these boundaries are:

| Existing boundary | Current authority |
| --- | --- |
| Immutable inventory and builder | `consan_program_inventory.h.inc` |
| Five-target capability table | `consan_capability_contract.h` |
| Observation intent and coverage ledger contracts | `consan_observation_plan.h.inc` |
| Evidence/report sizing | `consan_moi_report_plan.cpp` |
| Resource planning | `consan_resource.cpp` |
| Narrow prologue emission plans | `consan_moi_internal.h` |
| Narrow record-event emission plan | `consan_moi_record_planning.inc` |
| Pure runtime trust evaluation | `rj_hsa_dbi_hook_internal.h` |

### 1.2 Physical composition does not match the conceptual composition

The conceptual components above are mostly assembled through two textual
translation units:

| Textual translation unit | Implementation lines | Physical files |
| --- | ---: | ---: |
| `consan.cpp` and its `.inc` closure | 21,834 | 11 |
| `consan_moi.cpp` and its `.inc` closure | 35,295 | 23 |
| **Combined** | **57,129** | **34** |

Those two translation units contain **68.0% of all production implementation
lines** in scope. In contrast, the core CMake target lists only 11 compiled
sources and the hooks target lists 3 `.cpp` sources. Most `.inc` files share a
namespace, depend on textual inclusion order, and expose no compiler-enforced
interface to their neighbors.

This matters because the source tree visually suggests more isolation than the
compiler enforces. The `.inc` files are useful clues to the intended
components, but they are not yet component boundaries.

## 2. Where the implementation violates the component model

The present code is not spaghetti and it does not contain a catastrophic
circular semantic dependency. Its responsibility decomposition is good, its
data ownership is moderate, and its enforced dependency boundaries are poor.
Extensive tests currently compensate for boundaries that the type system and
build graph do not enforce.

The violations below are ordered by architectural and correctness risk, not by
how easy they are to edit.

### 2.1 Coverage is reconstructed from patch geometry

This is the highest-priority violation.

On the SuperCollider path,
`finalize_sc_access_coverage_ledger` scans `ConSanPatchInfo.kind` and anchor
offsets to infer which semantic intents were covered. On the MOI path,
`finalize_moi_site_lowering_outcomes` performs a larger reconstruction: it maps
patch kinds back to site kinds, maps offsets back to resource plans,
special-cases sampled synchronization, re-joins synchronization sequences, and
then publishes the coverage ledger.

That creates a reverse dependency:

```text
patch mechanism vocabulary -> semantic coverage truth
```

The semantic layer should know whether a planned intent was committed. It
should not infer that fact by recognizing the lowerer's current patch shapes.
A new patch kind, coalescing rule, anchor convention, or multi-site patch can
therefore be semantically successful yet appear missing, or can be attributed
to the wrong intent.

Coverage must instead be published from the same transaction that commits or
rejects an intent-bound lowering operation.

### 2.2 Exact instruction admission is duplicated across policy and lowering

`consan_access_policy.cpp` includes generated CDNA5 and RDNA4 machine headers
and knows exact constants and operand representations such as `SREG_NULL`, raw
`saddr`/`ioffset` arrangements, instruction sizes, and mnemonics.
`consan_atomic_fence_policy.cpp` similarly performs exact FLAT/VGLOBAL encoding
checks.

The MOI lowerer then repeats a near-equivalent classifier in
`consan_moi_core_types.h.inc` and filters the admitted candidates again in
`consan_moi_inline_atomic.inc`.

This fuses three different questions:

1. Is the operation semantically relevant to ConSan?
2. Can this target's instruction form be normalized into a supported lowering
   recipe?
3. Can this particular engine place and emit that recipe with the available
   resources?

The current duplication allows policy and lowering to disagree. When they do,
the disagreement can surface as a generic `PlacementRejected` result rather
than an explicit classifier inconsistency. Exact target normalization belongs
in one target-classifier authority. Semantic policy should consume the
normalized fact, and the lowerer should consume the resulting recipe.

### 2.3 Broad structures act as cross-component buses

#### `MoiOptions`

`MoiOptions` inherits `ConSanOptions` and contains the roughly 40-field
`ConSanMoiOperatingPoint`. It has 225 type references across 27 files, including
146 `const MoiOptions &` parameter occurrences; 46 of those occur in placement
code alone.

Consequently, emitters can see report binding, engine policy, target input,
debug controls, and selected operating state at the same time. Even when an
emitter uses only three facts, its signature does not state that dependency.

#### `ConSanTransformArtifacts`

`ConSanTransformArtifacts` has 214 references across 32 files. It is a mutable
bundle containing inventory, observation plan, coverage ledger, mutation,
resources, operating point, patches, replacement bytes, outcome, and
diagnostics. `TransformResult` publicly inherits from it, exposing private
lowerer state to the hook and any other result consumer.

This obscures which values are inputs, which are stage products, which are
validated public results, and which exist only to prove a lowering internally.

#### `ConSanPatchInfo`

`ConSanPatchInfo` is approximately 112 fields and 215 source lines, with 283
references across 32 files. It serves simultaneously as routing metadata,
mutation geometry, persistent-report ABI, spill description, descriptor state,
sampled-engine state, owner state, segment state, and validation proof.

Some detailed patch telemetry is legitimately needed by final validation. The
problem is that the same representation is also used as the semantic and
runtime communication channel.

### 2.4 Runtime analysis consumes lowerer patch geometry

`AutoMoiReportBufferRegistry::summarize` in
`rj_hsa_dbi_hook_moi_report.cpp` combines HSA snapshotting, ABI validation,
decoding, replay, loss accounting, conflict suppression, trust evaluation, and
rendering in one long operation.

The registry's metadata registration accepts `ConSanPatchInfo` and derives
runtime mappings from patch kinds and geometry. Replay and conflict attribution
then consume those patch-derived mappings. This directly violates the intended
direction: the host analyzer should receive a typed semantic/static mapping,
not reverse-engineer meaning from a lowerer proof object.

The pure trust evaluator inside this path demonstrates that decomposition is
practical. Lifecycle, decoding, analysis, trust, and rendering now need their
own contracts.

### 2.5 One source file crosses the host/target boundary

The first roughly 2,500 lines of `consan_moi_model.cpp` implement host-side
report and replay models. Its final roughly 500 lines implement exact
target-atomic address planning and emission. The interfaces in that tail are
reasonably narrow, but its physical placement is a clear component error and
makes dependencies harder to state and enforce.

### 2.6 Diagnostics are not yet an edge-only concern

There are 1,858 `warnings`/`errors` `emplace_back` or `push_back` sites across
36 production files. Current design permits string diagnostics from lowerers
and final validation, so this is not an immediate correctness defect. It does,
however, mean that stable failure concepts are frequently rendered before
reaching a presentation edge.

Typed reasons should replace strings where a reason is part of a stable
component contract. Diagnostic wording should remain stable and should be
rendered at the appropriate edge. This is a later cleanup, not a reason to
destabilize the early cuts.

### 2.7 Raw architecture branches are not uniformly violations

The implementation contains 105 `ROCJITSU_CODE_ARCH_*` references across 20
files and 120 `consan_uses_gfx*` references across 19 files. Most references in
target lowering, decoding, and independent validation are legitimate. The
architectural smell is specifically raw target encoding knowledge in semantic
policy or duplicated target classifiers, not every architecture branch.

The fourth refactoring must avoid replacing explicit and reviewable target
code with an abstraction that merely hides necessary ISA differences.

### 2.8 Detailed patch access in final validation is appropriate

Final validation contains 631 `patch.*` accesses and uses much of
`ConSanPatchInfo`. This is largely correct: an independent validator must inspect
the emitted bytes, geometry, ABI effects, and claimed proof. Validation should
not be weakened merely to make dependency counts look cleaner.

The desired boundary is that detailed patch proof remains private to lowering
and validation, while semantic coverage and runtime analysis use smaller typed
products.

### 2.9 Severity summary

| Violation | Layering severity | Correctness risk |
| --- | --- | --- |
| Coverage reconstructed from patch kinds and offsets | High | Medium-high |
| Exact admission duplicated in policy and lowerer | High | Medium |
| Runtime analysis consumes patch geometry | High | Medium |
| Two textual mega-translation-units | High maintenance cost | Low immediate risk |
| `MoiOptions` and artifacts used as broad buses | Medium-high | Medium |
| `TransformResult` exposes private lowerer artifacts | Medium-high | Low-medium |
| Report registry combines lifecycle, decode, analysis, and rendering | Medium-high | Medium |
| Raw architecture branching in target lowerers | Mostly legitimate | Low |
| Detailed patch telemetry in independent validation | Appropriate | Low |

## 3. Governing rules for the fourth refactoring

The refactoring should optimize for enforceable ownership and dependency
direction. A smaller implementation is desirable, especially where duplicate
classifiers and reverse mappings can be deleted, but line count is not the
primary stage gate.

1. **Preserve semantic authorities already established by the completed
   reimplementation stages.** Inventory, semantic policy, observation intent,
   resource planning, evidence requirements, lowering, final validation, and
   runtime trust must not silently acquire competing authorities.
2. **Move truth forward; do not reconstruct it downstream.** When a stage knows
   an intent ID, normalized operation, resource decision, or static mapping, it
   must publish that typed fact for its consumers.
3. **Establish typed seams before mechanically splitting `.inc` files.** A
   physical split performed first would merely expose hundreds of undeclared
   dependencies and encourage broad internal headers.
4. **Keep one authority during migration.** Short-lived adapters are acceptable,
   but old and new coverage, classification, or runtime-mapping pipelines must
   not remain parallel sources of truth.
5. **Do not weaken final validation.** Its deliberate independence and detailed
   patch inspection are assets.
6. **Preserve behavior, ABI, and user-visible diagnostics by default.** Any
   intended change must be called out and reviewed separately from a boundary
   move.
7. **Make new components directly testable.** A classifier, transaction,
   decoder, analyzer, or trust evaluator should not require a complete ELF and
   HSA execution when its immediate inputs can be constructed directly.
8. **Prefer value types grouped by invariant over long argument lists.** Narrow
   contracts should not become twenty scalar parameters.
9. **Use frequent, reviewable commits.** Each completed cut must leave one
   authoritative path and a passing relevant test set.
10. **Apply the normal ConSan gates.** Stage-specific tests come first. Each
    completed fourth-refactoring stage is then checked on all five emulated
    targets with `-j16`, followed by the serialized physical `gfx1201` gate with
    `-j1`. Physical GPU jobs must never run concurrently.

## 4. Ordered implementation route

The order is intentional. The first cuts repair information flow; the later
cuts use those typed products to reduce visibility and split physical
translation units. These stages are labeled `F0` through `F9` to distinguish
them from the completed reimplementation stages.

### F0. Freeze the baseline and dependency checks

Record the exact code-size, reference-count, test-count, and test-result
baseline used at the start of the work. Preserve the current all-target and
physical-GPU results. Add lightweight checks for the most important forbidden
dependencies as soon as the corresponding boundary exists.

At minimum, retain the following baseline metrics for comparison:

- 84,041 nonblank, comment-excluded production implementation lines;
- 57,129 lines, or 68.0%, textually compiled through the two main `.inc`
  closures;
- 146 `const MoiOptions &` parameters;
- 214 references to `ConSanTransformArtifacts`;
- 283 references to `ConSanPatchInfo`;
- runtime metadata registration and coverage finalization both dependent on
  patch representation;
- 5,302 registered ConSan tests at the Stage 10 exit: 4,667 nonphysical tests
  and 635 serialized physical `gfx1201` tests.

The purpose of these numbers is to detect whether dependencies actually move,
not to encourage mechanical changes that game a metric.

### F1. Bind semantic intent to the lowering transaction

Introduce a typed committed-lowering product, conceptually something such as
`ConSanCommittedPatch` or `ConSanValidatedPatchMapping`, that binds one or more
`ConSanProbeIntentId` values to:

- the committed patch or replacement location;
- the original semantic site or sites;
- the explicit lowering outcome;
- any static mapping needed by later stages.

Site-local plans must carry intent identity. Placement must commit the byte
mutation and intent result as one transaction. Rejection must publish the typed
reason against the same intent. Coalesced patches and multi-range sequences
must explicitly name every intent they satisfy.

Once the forward path is authoritative, delete
`finalize_sc_access_coverage_ledger` and
`finalize_moi_site_lowering_outcomes` along with their patch-kind, anchor, and
resource reverse joins.

Required focused tests include:

- one committed patch satisfying multiple intents;
- a multi-patch sequence satisfying one intent;
- sampled synchronization and multi-range cases;
- rejection attributed to the correct intent;
- an unknown or newly introduced patch kind having no effect on coverage
  semantics.

**F1 completion criterion:** no coverage result is inferred from patch kind,
anchor offset, or resource-plan reverse lookup.

### F2. Publish a typed static mapping to the runtime

Define the semantic/static mapping required for report attribution and replay.
It should contain only the facts the runtime actually needs, for example:

- original program site and semantic owners;
- relocated guest location when applicable;
- sampled slot/bank identity;
- persistent-record attribution;
- any explicit suppression or replay relation.

This mapping should be a validated output of transformation, derived from the
intent-bound committed-lowering product. Runtime metadata registration,
snapshot attribution, replay, and conflict suppression must consume it instead
of `ConSanPatchInfo`.

Raw patch telemetry may remain available to final validation and an explicitly
separate debug projection. It must not remain the runtime semantic interface.

**F2 completion criterion:** the hook's report path has no dependency on
`ConSanPatchInfo`, patch kinds, or patch geometry for semantic attribution.

### F3. Separate the public transform result from private artifacts

Replace `TransformResult`'s public inheritance from
`ConSanTransformArtifacts` with composition and explicit projections.

Private lowering artifacts should remain visible only to the transformer and
final validator. The public result should expose a reviewed set of typed
products, likely including:

- inventory and observation outcome where externally required;
- coverage and evidence requirements;
- validated static/runtime mapping;
- dispatch requirements;
- replacement image or mutation outcome;
- transform outcome and diagnostics.

Resource plans, raw patch proof, intermediate operating state, and mutable
workspaces should not escape by default. If diagnostics or development tooling
needs them, define a separate typed debug report rather than reopening the
production result.

**F3 completion criterion:** a runtime consumer cannot access lowerer-private
artifacts through `TransformResult`.

### F4. Make the target classifier the sole exact-admission authority

Introduce normalized target lowering forms or recipes, such as
`ConSanAccessLoweringForm` and `ConSanAtomicLoweringForm`. These values should
capture the exact supported operand/address form and an explicit typed
classifier rejection when normalization is impossible.

The dependency direction should become:

```text
decoded target instruction
          |
          v
target classifier -> normalized lowering form
          |                    |
          v                    v
semantic policy          engine lowerer/emitter
```

Semantic policy decides relevance and required observation from semantic facts
and the classifier result. It must not include generated instruction builders,
raw target register constants, or reproduce exact encoding predicates. The
lowerer must consume the normalized form rather than independently deciding
whether the original instruction is supported.

The lowerer may still reject resource allocation, placement, or emission. It
must not report a generic placement failure for an instruction form that the
classifier should have rejected.

Required tests include target-specific classifier goldens and common semantic
policy fixtures fed by normalized forms on all supported architectures.

**F4 completion criterion:** there is one exact lowerability classifier per
operation class, semantic policy contains no generated ISA headers or raw
architecture encoding constants, and emitters do not reclassify candidates.

### F5. Replace broad option buses with stage contracts

Use the existing immutable prologue and record-event plans as the model for all
native emitters. Replace `MoiOptions` parameters with the narrow stage products
actually required:

- immutable request/configuration input where it is genuinely global;
- accepted `ConSanMoiOperatingPoint` facts;
- normalized target lowering form;
- per-site semantic and resource plan;
- target-operation interface;
- explicit debug controls only where debug output is produced.

Do not replace one broad object with arbitrary anonymous subsets. Each value
type should name an invariant and have a single construction authority.

**F5 completion criterion:** native emission and placement components have zero
`const MoiOptions &` parameters. Any surviving aggregate is confined to
top-level orchestration and is not a cross-component bus.

### F6. Turn conceptual lowerer components into compiled components

Only after F1-F5 establish their interfaces should the textual `.inc`
composition be split. Candidate compiled components are:

- inventory construction and target classification;
- shared placement transaction;
- SuperCollider lowering;
- MOI observation/resource planning;
- per-engine semantic planning and emission;
- target operations and exact emission;
- independent final validation.

Use private headers for deliberately shared implementation contracts and public
headers only for true external contracts. The goal is not one translation unit
per current `.inc` file; it is a small set of coherent translation units whose
include graph enforces the intended dependency direction.

Move the target-address planning/emission tail out of
`consan_moi_model.cpp` as part of this stage.

**F6 completion criterion:** the build no longer relies on 20,000-35,000-line
textual translation units or on include-order visibility between conceptual
components.

### F7. Split the host report path

Decompose `AutoMoiReportBufferRegistry::summarize` into directly testable
components:

1. **Lifecycle registry:** allocate, bind, retire, and snapshot report buffers.
2. **Evidence decoder:** convert a raw snapshot into typed events and typed
   malformed/loss reasons.
3. **Engine analyzer:** convert typed events and static mapping into replay or
   conflict analysis.
4. **Trust evaluator:** combine coverage, evidence quality, and loss with the
   analysis result.
5. **Renderer:** convert the typed result into stable user diagnostics.

The registry should become a coordinator or lose `summarize` entirely. HSA
lifecycle state must not be needed to unit-test decoding, analysis, trust, or
rendering.

**F7 completion criterion:** each of the five responsibilities has a narrow
contract and direct tests, and report lifecycle code contains no engine
analysis policy.

### F8. Shrink patch proof and type stable diagnostics

With coverage and runtime mapping removed from `ConSanPatchInfo`, split or
shrink it into the remaining private concepts. Likely products include
committed patch geometry, mutation effects, ABI effects, and validation proof.
Only final validation should receive the full proof set.

Replace string diagnostics with typed reasons where those reasons cross a
component boundary or participate in control flow. Preserve established
wording in the renderer and avoid converting one-off internal validation detail
into an unnecessary public enum.

**F8 completion criterion:** no broad patch structure serves lowering,
validation, runtime semantics, and user reporting simultaneously.

### F9. Enforce boundaries and delete migration scaffolding

Finish by deleting adapters, reverse maps, duplicated classifiers, obsolete
fields, and compatibility paths introduced during the migration. Add durable
checks for the architectural rules that can be stated mechanically:

- semantic policy cannot include generated target instruction headers or raw
  target constants;
- hooks cannot include lowerer-private patch or resource headers;
- coverage code cannot inspect patch kinds or geometry;
- native emitters cannot accept `MoiOptions`;
- runtime analysis accepts typed static mapping rather than patch telemetry;
- conceptual components compile through declared interfaces rather than a
  textual include closure.

Recompute implementation lines and all dependency metrics. A reduction is
expected from deleting reverse coverage reconstruction, duplicate exact
classifiers, broad adapter fields, and redundant runtime mappings. Falling
below 80,000 lines would be welcome, but it is not sufficient evidence of a
successful refactoring and is not worth preserving a second authority or
weakening validation.

## 5. Test and commit discipline

Each stage should be a sequence of small local commits. A useful commit changes
one authority or one consumer set, adds or updates its focused tests, and leaves
the tree in a bisectable state. Never leave both old and new authorities active
across a stage boundary.

For each stage:

1. run the focused host/unit tests for the component being changed;
2. run the relevant engine and architecture matrix while iterating;
3. at stage completion, build with `-j16` and run the complete nonphysical
   ConSan suite with `ctest -j16`;
4. then run the physical `gfx1201` tests alone with `ctest -j1`;
5. compare the test inventory with the baseline so that a green result caused
   by lost test discovery cannot pass the gate;
6. update this document or the handoff with the stage result and exact commit.

The five emulated targets remain the routine cross-architecture gate. Physical
GPU tests are serialized and less frequent during iteration, but mandatory at
each completed stage. Existing paired device tests must not be weakened to make
a boundary change pass.

## 6. Definition of success

The fourth refactoring is complete when all of the following are true:

- coverage is produced directly from intent-bound commit/rejection events and
  contains no patch-shape reconstruction;
- runtime registration, replay, and analysis consume a validated semantic
  mapping and do not consume `ConSanPatchInfo`;
- exact instruction lowerability has one target-classifier authority;
- semantic policy has no generated ISA builders or raw architecture encoding
  constants;
- `TransformResult` does not publicly expose private lowerer artifacts;
- native emitters and placement helpers no longer accept `MoiOptions` as a
  general-purpose bus;
- host lifecycle, decoding, analysis, trust, and rendering are separately
  testable components;
- final validation retains independent access to all proof needed to verify the
  emitted program;
- the two mega-translation-units have been replaced by compiled components with
  declared interfaces and enforced dependency direction;
- all five emulated architectures and serialized physical `gfx1201` tests pass,
  with no loss of test inventory;
- implementation size and coupling metrics are recomputed and documented.

The key outcome is not merely a rearranged source tree. It is that a future
change to a patch kind cannot alter semantic coverage, a new target encoding
cannot be admitted differently by policy and lowering, and the host analyzer
cannot learn semantic meaning by peering into target-patching internals. Once
those properties are enforced by types and compilation boundaries, the
component architecture described in the design will also be the architecture
the code is capable of expressing.
