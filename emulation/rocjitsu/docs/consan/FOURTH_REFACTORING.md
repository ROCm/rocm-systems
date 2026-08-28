# ConSan fourth refactoring: enforce components and factor variability

This document is the working charter for ConSan's fourth refactoring. It starts
from two independent investigations of the current implementation:

1. what the de-facto components are and whether dependencies respect them; and
2. how implementation size varies across engines and targets, especially
   whether the source is scaling as `O(N * M)` rather than `O(N + M)`.

The two views are related but not interchangeable. A well-layered program can
still duplicate every engine/target combination, while a compact shared helper
can still violate every intended dependency boundary. The refactoring must
improve both properties.

The destination design in this document is deliberately a set of hypotheses
and invariants, not a frozen class diagram. Each extraction must be tested as a
vertical slice, measured, and either generalized, revised, or removed before
the next slice. The freedom to discover a better interface during the work is
part of the plan; parallel authorities and unmeasured rewrites are not.

The measurements below use commit
`01e6a6f1a23997866ff7a318966f6d809f9de6b8`. They cover production code under:

- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/patch/consan/`
- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/hooks/consan/`

Tests, documentation, build files, blank lines, and comments are excluded. The
result is **84,041 implementation lines in 83 files**. The component table in
Section 1 uses exact whole-file sums by dominant responsibility; it does not
imply that every line in a file belongs exclusively to that responsibility.
The variability survey in Section 2 instead uses reviewed source regions so
that mixed files do not inherit one label wholesale.

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

## 2. Variability across modes and targets

ConSan currently supports four engines and five target products:

| Dimension | Members used in this survey |
| --- | --- |
| Modes | Record/Replay, Sampled, InlineShadow, SuperCollider |
| Targets | `gfx942`/CDNA3, `gfx950`/CDNA4, `gfx1100`/RDNA3, `gfx1201`/RDNA4, `gfx1250`/CDNA5 |

The desired source-growth model is approximately `O(1) + O(N) + O(M)`, with a
small, justified interaction term. This does not mean erasing real ISA
differences. It means expressing an engine policy once, expressing a target
mechanism once, and composing the two through a narrow contract.

### 2.1 Measurement method

This survey does **not** classify a line by whether the spelling of a mode or
architecture occurs on that line. That approach is badly wrong here. Most of
an engine implementation has no engine name on each line, and most target
variation is reached through a capability field or a target-aware builder.
Conversely, identifiers such as `kRdna4ExecLo`, `kRdna4VccLo`, and
`kRdna4ScopeDevice` occur as normalized operands in algorithms used on all five
targets. Those uses are not RDNA4-only code.

The survey instead used the following semantic procedure:

1. Assign every complete file, function, generated fragment, and coherent
   branch region to the set of modes that can use the implementation. Callers
   and engine selection, not names, determine the set. Mixed HSA and report
   coordinators remain shared only for their common orchestration; separable
   Record/Replay, Sampled, InlineShadow, and SuperCollider analysis or
   allocation bodies receive their actual engine ownership.
2. Within those regions, identify exact decoders/classifiers, target-selection
   branches, capability-guarded bodies, target profile data, and emission paths
   whose implementation shape changes with the target matrix.
3. Do not mark an unchanged engine algorithm as target-sensitive merely because
   it calls `build_*(_, arch)`; the target-aware builder is the target seam.
4. Count nonblank, comment-excluded code lines in the reviewed regions and
   check that the mode partition covers all 84,041 lines exactly once.
5. Keep subset ownership. Code shared by all three MOI engines, by
   Record/Replay and InlineShadow, or by a target family must not be replicated
   into several exclusive buckets merely to make a rectangular table.

The result is a source-change-sensitivity measurement. A target classifier that
handles all five products is target-specific because it is an authority that
must be extended or deliberately reject a sixth target. An ordinary algorithm
which merely passes `arch` through a stable target interface is target-neutral.

Semantic attribution is not an intrinsic lexical property. Moving a wrapper
line across a reviewed region boundary can change a few lines without changing
the design. The numbers below are therefore appropriate as a refactoring
baseline and trend metric, not as a claim that every brace has metaphysical
ownership. The large regions and conclusions are stable under reasonable
boundary choices. Future measurements must preserve the reviewed-region
method, record boundary changes, and never substitute filename or token
classification.

### 2.2 The four requested top-level buckets

| Source sensitivity | Implementation lines | Share |
| --- | ---: | ---: |
| Neither mode- nor target-specific | 27,336 | 32.5% |
| Mode-specific, target-neutral | 49,623 | 59.0% |
| Target-specific, mode-neutral | 3,512 | 4.2% |
| Both mode- and target-specific | 3,570 | 4.2% |
| **Total** | **84,041** | **100%** |

This is not a codebase dominated by five copies of four engines. The strict
interaction term is 4.2%, not the majority of the implementation. The largest
category is engine policy and machinery that is already shared across all five
targets. That is reassuring about asymptotic scaling, but it does not make the
3,570-line interaction term harmless: those lines contain several of the
largest and most fragile lowering paths, and their current placement makes both
new-engine and new-target work harder.

### 2.3 Mode ownership before and after target sensitivity

The exclusive partition under this attribution is clearer when subset sharing
is retained:

| Mode ownership set | Target-neutral | Target-sensitive | Total |
| --- | ---: | ---: | ---: |
| All four modes | 27,336 | 3,512 | 30,848 |
| Record/Replay + Sampled + InlineShadow (shared MOI) | 16,729 | 1,708 | 18,437 |
| Record/Replay + InlineShadow | 2,170 | 52 | 2,222 |
| Record/Replay only | 8,073 | 65 | 8,138 |
| Sampled only | 7,116 | 520 | 7,636 |
| InlineShadow only | 9,080 | 664 | 9,744 |
| SuperCollider only | 6,455 | 561 | 7,016 |
| **Total** | **76,959** | **7,082** | **84,041** |

The 16,729 target-neutral lines shared by the three MOI engines are important.
They show that substantial `O(N * M)` duplication has already been avoided,
but they also expose a hidden component: “shared MOI” is larger than any one
exclusive engine implementation and is mostly textually composed rather than
enforced as a component.

For the requested per-mode view, shared-subset lines are incidence counts: a
line used by the three MOI engines appears in all three rows, but exists only
once in the exclusive table above.

| Mode | Target-neutral mode-sensitive incidence | Target-sensitive mode incidence |
| --- | ---: | ---: |
| Record/Replay | 26,972 | 1,825 |
| Sampled | 23,845 | 2,228 |
| InlineShadow | 27,979 | 2,424 |
| SuperCollider | 6,455 | 561 |

The incidence columns must not be summed. They answer “how much specialized
source can affect this mode?”, not “how many physical lines exist?”.

### 2.4 Architecture and mode-by-architecture incidence

Target-sensitive but mode-neutral code is highly shared. The following table
counts, for each product, the reviewed target-authority source that applies to
that product. Most exact classifiers and validators cover several or all
targets, so these are deliberately overlapping incidence counts over 3,512
physical lines.

| Target | Mode-neutral target-sensitive incidence |
| --- | ---: |
| `gfx942` / CDNA3 | 3,259 |
| `gfx950` / CDNA4 | 3,258 |
| `gfx1100` / RDNA3 | 3,256 |
| `gfx1201` / RDNA4 | 3,255 |
| `gfx1250` / CDNA5 | 3,293 |

The near-equality is a good result: target normalization, target profile
validation, inventory construction, exact sequence recognition, and independent
validation are mostly common target components rather than five product
copies. The small `gfx1250` increase reflects genuinely unique selectable-bank,
split-LDS, `S_CALL_I64`, and transport behavior.

The requested mode/target-pair breakdown is likewise an incidence matrix over
the 3,570 physical lines in the interaction bucket:

| Mode \ target | `gfx942` | `gfx950` | `gfx1100` | `gfx1201` | `gfx1250` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Record/Replay | 1,587 | 1,587 | 1,399 | 1,531 | 1,731 |
| Sampled | 2,009 | 2,009 | 1,398 | 1,530 | 1,711 |
| InlineShadow | 2,022 | 2,022 | 1,460 | 1,740 | 1,956 |
| SuperCollider | 193 | 193 | 155 | 499 | 484 |

These cells also must not be summed. For example, 1,708 physical lines of
target-sensitive shared-MOI code contribute to three mode rows, and a single
gfx9-family far-route body contributes to both `gfx942` and `gfx950`. Counting
it as four copies would manufacture the very `N * M` duplication being
measured.

### 2.5 Where the actual interaction term lives

The important both-sensitive regions are concentrated rather than diffuse:

- InlineShadow's far dense atomic route in
  `consan_moi_inline_atomic.inc` is a roughly 374-code-line coherent region for
  the `gfx942`/`gfx950` capability set.
- Sampled's corresponding far dense atomic route in
  `consan_moi_sampled_sync.inc` is roughly 423 code lines for the same pair.
- SuperCollider's branch-only dense route in
  `consan_supercollider_lds.inc` is roughly 275 code lines used by
  `gfx1201`/`gfx1250`.
- The SuperCollider inline flat rewrite is roughly 63 code lines and is
  currently a `gfx1201`-only capability.
- InlineShadow's lazy workgroup-local shadow initialization is roughly 148 code
  lines shared by `gfx1201` and `gfx1250`.
- `gfx1250` contributes smaller but repeated per-engine branches for
  `S_CALL_I64`, per-kernel owner translation, selectable VGPR banks,
  split-two-address LDS relocation, and return-PC route identity.
- Workgroup identity, dense routing, address materialization, VCC preservation,
  wave width, and register-bank transitions recur in both shared-MOI and
  engine-specific code.

The first two bullets are especially revealing. They implement analogous
target routing capabilities separately inside two modes. This is genuine local
`N * M` growth and a good extraction candidate. In contrast, the five rows of
`kConSanTargetProfiles` are desirable `O(M)` data, and common target-aware
builders are desirable shared target mechanisms.

### 2.6 Scaling diagnosis

The codebase is not fundamentally `O(N * M)`. Its present shape is closer to:

```text
large common core
+ large O(N) engine and shared-engine-subset implementations
+ modest O(M) target authorities
+ small but costly engine/target interaction regions
```

Three qualifications matter:

1. The interaction term is disproportionately complex. Dense routing and
   register-state preservation account for more engineering risk than their
   line share suggests.
2. Subset sharing is often physical rather than architectural. The shared-MOI
   implementation exists once, but textual inclusion and broad buses let every
   engine reach too much of it.
3. Similar target capabilities are sometimes re-expressed per engine rather
   than lowered through one target operation. A sixth target with a new call or
   transport model could therefore grow several separate branches even though
   the current aggregate is only 4.2%.

The fourth refactoring should reduce the interaction term and make its
remaining members explicit. It should not chase zero: some engine policies
legitimately select a target capability, and independent validation must retain
target- and mode-aware proof.

### 2.7 How the variability and layering work coexist

The layering investigation asks **who owns a fact and who may depend on it**.
The variability investigation asks **which dimension owns a difference and how
often that difference is expressed**. Every proposed cut must answer both.

The leading design hypothesis is a composition of three kinds of product:

```text
semantic site + evidence requirement
                 |
                 v
          engine operation plan
                 |
                 v
       target operation lowering
                 |
                 v
 placement transaction + validated static mapping
```

This is intentionally not yet a commitment to one universal operation IR,
virtual interface, variant, or table-driven backend. The refactoring should
discover the smallest useful seam through repeated vertical slices:

1. choose one repeated interaction, such as far routing, report atomics,
   workgroup identity, or address materialization;
2. describe the engine's required operation without exposing exact encoding;
3. lower it for at least two materially different target families;
4. consume it from a second engine and delete the duplicate path;
5. measure the four source-sensitivity buckets and adapter size;
6. keep, revise, or discard the abstraction based on the result.

An extraction that merely moves 400 lines behind a broader options object has
not improved either axis. An extraction that makes the target operation reusable
but gives it authority over semantic coverage has improved scaling while
breaking layering. Both tests are required at every checkpoint.

## 3. Where the implementation violates the component model

The present code is not spaghetti and it does not contain a catastrophic
circular semantic dependency. Its responsibility decomposition is good, its
data ownership is moderate, and its enforced dependency boundaries are poor.
Extensive tests currently compensate for boundaries that the type system and
build graph do not enforce.

The violations below are ordered by architectural and correctness risk, not by
how easy they are to edit.

### 3.1 Coverage is reconstructed from patch geometry

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

### 3.2 Exact instruction admission is duplicated across policy and lowering

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

### 3.3 Broad structures act as cross-component buses

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

### 3.4 Runtime analysis consumes lowerer patch geometry

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

### 3.5 One source file crosses the host/target boundary

The first roughly 2,500 lines of `consan_moi_model.cpp` implement host-side
report and replay models. Its final roughly 500 lines implement exact
target-atomic address planning and emission. The interfaces in that tail are
reasonably narrow, but its physical placement is a clear component error and
makes dependencies harder to state and enforce.

### 3.6 Diagnostics are not yet an edge-only concern

There are 1,858 `warnings`/`errors` `emplace_back` or `push_back` sites across
36 production files. Current design permits string diagnostics from lowerers
and final validation, so this is not an immediate correctness defect. It does,
however, mean that stable failure concepts are frequently rendered before
reaching a presentation edge.

Typed reasons should replace strings where a reason is part of a stable
component contract. Diagnostic wording should remain stable and should be
rendered at the appropriate edge. This is a later cleanup, not a reason to
destabilize the early cuts.

### 3.7 Raw architecture branches are not uniformly violations

The implementation contains 105 `ROCJITSU_CODE_ARCH_*` references across 20
files and 120 `consan_uses_gfx*` references across 19 files. Most references in
target lowering, decoding, and independent validation are legitimate. The
architectural smell is specifically raw target encoding knowledge in semantic
policy or duplicated target classifiers, not every architecture branch.

The fourth refactoring must avoid replacing explicit and reviewable target
code with an abstraction that merely hides necessary ISA differences.

### 3.8 Detailed patch access in final validation is appropriate

Final validation contains 631 `patch.*` accesses and uses much of
`ConSanPatchInfo`. This is largely correct: an independent validator must inspect
the emitted bytes, geometry, ABI effects, and claimed proof. Validation should
not be weakened merely to make dependency counts look cleaner.

The desired boundary is that detailed patch proof remains private to lowering
and validation, while semantic coverage and runtime analysis use smaller typed
products.

### 3.9 Severity summary

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

## 4. Governing rules for the fourth refactoring

The refactoring should optimize for enforceable ownership, dependency
direction, and additive growth across the mode and target dimensions. A
smaller implementation is desirable, especially where duplicate classifiers,
target routes, and reverse mappings can be deleted. Line count alone is not a
stage gate, but a boundary that makes the interaction term larger without a
clear semantic gain is suspect.

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
11. **Assign variation to one dimension.** Engine policy may select a semantic
    operation; target lowering may realize it. Do not let each engine recreate
    the same target call, routing, identity, or register-state mechanism.
12. **Retain subset sharing explicitly.** Shared MOI, exact-shadow, gfx9,
    gfx12, CDNA, and other real subsets are preferable to duplication. They must
    have named contracts rather than being inferred from textual inclusion.
13. **Measure both exclusive lines and incidence.** The four exclusive buckets
    detect physical growth. Per-mode and per-target incidence detects blast
    radius. Neither should be replaced by lexical token counts.

## 5. Provisional convergence route

The dependency order is intentional, but the concrete destination interfaces
are provisional. The first cuts repair information flow. Target-operation
seams are then discovered through vertical slices, not designed all at once.
Later cuts use the proven typed products to reduce visibility and split
physical translation units. These stages are labeled `F0` through `F9` to
distinguish them from the completed reimplementation stages.

There are two workstreams inside this order:

- the **boundary workstream** moves coverage, static mapping, options, runtime
  analysis, and proof to their rightful owners; and
- the **variability workstream** extracts repeated engine/target interactions
  into reusable target operations and verifies that a second engine can consume
  them without importing target internals.

The workstreams converge at the classifier, stage-contract, and compiled-
component cuts. A stage may revise an earlier interface when a second vertical
slice disproves it, provided the revision is committed separately, tested, and
does not leave two authorities.

### F0. Freeze the baseline and dependency checks

Record the exact code-size, reference-count, test-count, and test-result
baseline used at the start of the work. Preserve the current all-target and
physical-GPU results. Add lightweight checks for the most important forbidden
dependencies as soon as the corresponding boundary exists.

At minimum, retain the following baseline metrics for comparison:

- 84,041 nonblank, comment-excluded production implementation lines;
- 27,336 mode- and target-neutral lines, 49,623 mode-only lines, 3,512
  target-only lines, and 3,570 lines in the interaction bucket;
- 16,729 target-neutral shared-MOI lines and 1,708 target-sensitive shared-MOI
  lines;
- 57,129 lines, or 68.0%, textually compiled through the two main `.inc`
  closures;
- 146 `const MoiOptions &` parameters;
- 214 references to `ConSanTransformArtifacts`;
- 283 references to `ConSanPatchInfo`;
- runtime metadata registration and coverage finalization both dependent on
  patch representation;
- 5,302 registered ConSan tests at the Stage 10 exit: 4,667 nonphysical tests
  and 635 serialized physical `gfx1201` tests.

The purpose of these numbers is to detect whether dependencies and variation
actually move, not to encourage mechanical changes that game a metric. Update
the semantic region ledger when a source boundary moves; do not silently switch
to filename or token attribution.

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

Use this stage for the first variability vertical slice. Select one operation
that is currently expressed in more than one engine—far routing is the leading
candidate—and separate:

- the engine's semantic request and required preserved state;
- the target's call, branch, and identity recipe;
- placement feasibility and the committed result.

Implement at least two materially different target capability sets, then use
the operation from a second engine. Do not generalize the interface beyond the
evidence from those consumers. If far routing proves to combine too many
invariants, split the experiment into a smaller operation such as call/return
state, route-key construction, or SCC preservation.

Required tests include target-specific classifier goldens and common semantic
policy fixtures fed by normalized forms on all supported architectures.

**F4 completion criterion:** there is one exact lowerability classifier per
operation class, semantic policy contains no generated ISA headers or raw
architecture encoding constants, emitters do not reclassify candidates, and
one target-operation slice has replaced duplicated mode/target lowering in at
least two engines without broadening semantic authority.

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

Continue the variability work one operation at a time. Candidate slices after
far routing are workgroup identity, atomic address materialization, report
atomics/cache completion, VCC/SCC preservation, and selectable-bank
transitions. After each slice, recompute the four buckets, per-mode/target
incidence, adapter lines, and number of target branches remaining in engines.
Retain the seam only when a second consumer makes it narrower or deletes more
code than its adapters introduce.

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

The physical split should enforce both axes: an engine component must not see
generated target encodings, and a target-operation component must not decide
coverage, evidence policy, or engine semantics. Shared-MOI and exact-shadow
subsets should become named internal components rather than accidental include
closures.

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

Also recompute the semantic variability ledger. Any remaining interaction
region must be reviewed as one of:

- a legitimate mode selection of a target capability;
- independent mode-and-target-aware validation proof; or
- debt with a named next extraction.

Recompute implementation lines and all dependency metrics. A reduction is
expected from deleting reverse coverage reconstruction, duplicate exact
classifiers, broad adapter fields, and redundant runtime mappings. Falling
below 80,000 lines would be welcome, but it is not sufficient evidence of a
successful refactoring and is not worth preserving a second authority or
weakening validation.

## 6. Test and commit discipline

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

## 7. Definition of success

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
- implementation size and coupling metrics are recomputed and documented;
- the mode/target interaction bucket is smaller, or every surviving region has
  an explicit justification and owner;
- adding an engine does not require copying target routing, identity,
  addressing, or register-state mechanisms, and adding a target does not
  require editing each engine for an already-normalized operation;
- shared-MOI and other subset-shared implementations have declared contracts
  and enforced visibility rather than relying on textual inclusion.

The key outcome is not merely a rearranged source tree. It is that a future
change to a patch kind cannot alter semantic coverage, a new target encoding
cannot be admitted differently by policy and lowering, and the host analyzer
cannot learn semantic meaning by peering into target-patching internals. Once
those properties are enforced by types and compilation boundaries, the
component architecture described in the design will also be the architecture
the code is capable of expressing. Once repeated target operations are owned by
target components, the fifth engine or sixth target should extend one dimension
rather than reopening a matrix of engine-specific target lowerers.
