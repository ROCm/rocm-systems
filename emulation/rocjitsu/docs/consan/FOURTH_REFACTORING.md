# ConSan fourth refactoring: enforce components and factor variability

This document is the working charter for ConSan's fourth refactoring. It starts
from two independent investigations of the current implementation:

1. what the de-facto components are and whether dependencies respect them; and
2. how implementation size varies across engines and targets, especially
   whether the source is scaling as `O(N * M)` rather than `O(N + M)`.

The two views are related but not interchangeable. A well-layered program can
still duplicate every engine/target combination, while a compact shared helper
can still violate every intended dependency boundary. The refactoring must
put the design in order on both axes.

Implementation size is a diagnostic, not the governing objective. There is no
line-count quota and no requirement that every intermediate or final checkpoint
be smaller. The working hypothesis is that the current size and resistance to
deletion reflect muddied ownership, duplicated authority, hidden state
machines, and mode/target interactions. Correcting those causes should make
later simplification and deletion natural. Shrinking code by weakening tests,
validation, diagnostics, or explicit contracts would be a failure; temporary
growth that establishes a real boundary or regression test can be justified.

The destination design in this document is deliberately a set of hypotheses
and invariants, not a frozen class diagram. Each extraction must be tested as a
vertical slice, measured, and either generalized, revised, or removed before
the next slice. The freedom to discover a better interface during the work is
part of the plan; parallel authorities and unmeasured rewrites are not.

## Mandate and scope

This is a mandate to **deep-fix ConSan's design**, not merely to tidy files
around the current implementation. The de-facto components identified below
are a diagnosis of today's ownership, not boundaries that the refactoring must
preserve. The work may move, split, merge, or replace components; redesign
internal contracts and intermediate representations; replace coordinators and
state machines; change file and target structure; and delete obsolete paths.
Existing names and include closures have no architectural authority of their
own.

The destination should be easy to explain and easy to navigate. When the
design documents are updated after the implementation stabilizes, their
component map should correspond directly to source ownership and build
boundaries. A reader should be able to determine where a fact is created, who
may transform it, who consumes it, and where mode- or target-specific behavior
lives without tracing textual include order or reverse-engineering a broad
artifact bundle.

The primary scope is all ConSan production, host/runtime, test, build, and
documentation code in Rocjitsu. Immediate shared Rocjitsu infrastructure may
change when a clean ConSan boundary genuinely requires it; unrelated Rocjitsu
or ROCm redesign is outside scope. Adding features, engines, or targets is not
an objective, except where a regression fix or a second consumer is needed to
prove an abstraction.

The non-negotiable operating constraints are:

1. **Preserve tested behavior throughout.** The complete ConSan test inventory
   must continue to pass at periodic convergence checkpoints, with focused and
   relevant matrix tests between them. Testing must be frequent enough to
   localize regressions but batched enough that the full matrix does not
   dominate every small edit.
2. **Treat discovered bugs as immediate work.** Characterize each bug, add a
   regression test that would have caught it, fix it, and validate the affected
   modes and targets before continuing past that boundary. Do not preserve a
   known bug merely to call a change behavior-neutral.
3. **Move toward one legible component architecture.** Each fact and decision
   must have one authority, dependencies must flow through explicit typed
   products, and source/build boundaries must make the intended visibility
   enforceable.
4. **Contain target variation.** Raw architecture encodings, generated ISA
   types, register constants, and target recipes should live in files owned by
   one target or a named target family. Target-neutral semantic and engine code
   should consume normalized target operations rather than branch on gfx
   architecture.
5. **Contain mode variation.** Mode-specific policy, planning, emission, and
   analysis should live in mode-owned regions or files. Behavior genuinely
   shared by a subset of modes should have a named subset component and
   contract; it should not be duplicated or left as unexplained mode switches
   in nominally common code.

These constraints govern the work more strongly than the provisional stage
sequence later in this document. Deep changes remain incremental in the sense
that they must be reviewable, tested, bisectable, and leave one authority at a
time; they need not be shallow or preserve the present component graph.

The measurements below use commit
`01e6a6f1a23997866ff7a318966f6d809f9de6b8`. They cover production code under:

- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/patch/consan/`
- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/hooks/consan/`

Tests, documentation, build files, blank lines, and comments are excluded. The
result is **84,041 implementation lines in 83 files**. The component table in
Section 1 and the variability survey in Section 2 both use reviewed semantic
source regions. A region was assigned only after reading its definitions,
inputs, outputs, callers, mutations, and downstream consumers. Scripts were
used to strip comments and blank lines and to total that explicit ledger, not
to infer ownership from filenames or tokens. The component ledger is exclusive
by primary responsibility; Section 2 independently attributes the same lines
by mode and target sensitivity.

## 1. De-facto components

The deep read changes the first-pass component map substantially. The following
are the implementation's actual primary responsibilities, whether or not the
source and build graph enforce them as components:

| Reviewed semantic responsibility | Implementation lines | Share |
| --- | ---: | ---: |
| InlineShadow-specific lowering | 9,747 | 11.6% |
| HSA integration and resource lifecycle | 7,101 | 8.4% |
| Program semantic inventory and analysis | 7,038 | 8.4% |
| Shared MOI lowering | 6,987 | 8.3% |
| SuperCollider-specific lowering | 6,676 | 7.9% |
| Evidence ABI, pure models, and sizing | 6,658 | 7.9% |
| Sampled-specific lowering | 6,198 | 7.4% |
| Target profile, normalization, and target operations | 5,959 | 7.1% |
| Record/Replay-specific lowering | 5,825 | 6.9% |
| MOI operating-point and resource solver | 5,487 | 6.5% |
| Independent final validation | 4,468 | 5.3% |
| Fault and perturbation planning/lowering | 2,583 | 3.1% |
| Semantic policy, intent, and coverage | 2,426 | 2.9% |
| Host evidence decode, analysis, trust, and rendering | 1,958 | 2.3% |
| Transformation and retry coordination | 1,901 | 2.3% |
| Request, configuration, and result facade | 1,889 | 2.2% |
| Generic placement and image mutation | 1,140 | 1.4% |
| **Total** | **84,041** | **100%** |

These numbers do not say that InlineShadow, for example, is cleanly isolated.
They say that 9,747 lines have InlineShadow lowering as their primary semantic
responsibility. Section 2 may independently classify such a line as
target-neutral or target-sensitive.

Mixed files were split at semantic boundaries rather than assigned wholesale.
For example, the host model portion and target-emission tail of
`consan_moi_model.cpp` have different owners; report-buffer lifecycle and the
later host analyzer in `rj_hsa_dbi_hook_moi_report.cpp` have different owners;
the policy-publication prefix of `consan_supercollider_common.inc` is not SC
emission; and target-fact extraction, resource solving, and patch-plan
construction inside `consan_moi_placement.inc` were followed separately.

The first-pass labels concealed several important facts:

- **There is no `core contracts` component.** The colocated schemas belong to
  configuration, inventory, policy, evidence, resource solving, mutation,
  patch proof, and result publication. Their common header visibility is a
  physical convenience, not one semantic authority.
- **Evidence planning and resource planning are not one component.** Evidence
  sizing is focused and mostly functional. The small generic register-window
  planner in `consan_resource.cpp` is also focused, but the real MOI resource
  authority is a roughly 5,500-line operating-point solver dominated by
  `consan_moi_placement.inc`.
- **Mutation, composition, and validation are three components.** Fault and
  perturbation planning/lowering, staged transformation coordination, and the
  independent validator have different inputs, authorities, and failure
  semantics.
- **The HSA side is not one adapter.** Configuration and HSA lifecycle,
  transformation/report allocation coordination, report-buffer lifecycle, and
  host decode/analysis/trust/rendering are distinct responsibilities.
- **Shared MOI is not one planning/lowering block.** Its coordinator, mutable
  resource fixed-point, shared emission mechanisms, and engine lowerings are
  separate de-facto components despite sharing a textual translation unit.
- **Target work is larger and more dispersed than the target-profile table.**
  It includes normalization, register/descriptor rules, relay routing,
  address materialization, exact emission, and target-aware placement.

### 1.1 Actual control and data flow

The design documents suggest a forward stage pipeline. The implementation does
not currently execute that pipeline. Its actual control topology is:

```text
HSA load hook
  |
  +--> TransformResult::publish_optional --------------------------+
  |         | validates configuration                              |
  |         v                                                      |
  |      lower_consan                                              |
  |         v                                                      |
  |      try_patch_consan_impl                                     |
  |         +--> decode + build ProgramInventory                   |
  |         +--> build sync graph and semantic associations        |
  |         +--> fault/staged-composition coordinator              |
  |         +--> SuperCollider lowerer, or                         |
  |         `--> try_patch_consan_moi                              |
  |                 +--> policy/ledger publication                 |
  |                 +--> mutable MOI resource fixed-point          |
  |                 `--> engine and target emission                |
  |         v                                                      |
  |      independent final validation                              |
  |         v                                                      |
  |      publish stage statuses and evidence requirements <--------+
  |
  `-- automatic MOI only: read those requirements, allocate report
      storage, then retry lowering against the retained inventory
```

`TransformResult::publish_optional` does validate configuration and runtime
capability contracts before lowering. However, its inventory, observation,
evidence, and binding stage records mostly inspect products after
`lower_consan` returns. Evidence intent and sizing are also computed after that
first lower. The HSA hook compensates for this order by owning the automatic
MOI two-pass protocol: pristine unbound transform, evidence sizing, allocation,
then retry from retained inventory.

The published `ConSanPipelineStage` sequence is therefore a useful result
summary, but not yet an orchestration boundary. There are three effective
controllers: staged composition in `try_patch_consan_impl`, the MOI
policy/resource/emission state machine in `try_patch_consan_moi`, and the HSA
hook's binding/retry state machine. `TransformResult::publish_optional` sits
over them as a fourth, partly retrospective facade. They communicate through
the broad `ConSanTransformArtifacts` aggregate.

### 1.2 Boundaries that are already strong

The fourth refactoring should preserve and build on the properties of these
parts rather than redesigning them gratuitously. Their types or physical homes
may still change when a deeper boundary requires it.

#### Immutable inventory

`ProgramInventory` is a genuine shared semantic boundary. Its immutable
`shared_ptr<const Storage>` representation is distinct from its builder, and
consumers receive read-only views. The inventory's scope is broader than the
first pass stated: it includes decoded sites, ownership, a synchronization
graph, fence candidates, and other semantic associations. That coherent
"program semantic inventory" is a useful boundary. The `supported_mvp` field
is one exception: it embeds a legacy target/lowerability decision in otherwise
normalized inventory and is consumed by synchronization and fault selection.

#### Central target profile

`ConSanTargetProfile` centralizes the supported architecture matrix in a typed
five-row table. This is the right authority for broad capability facts. The
remaining problem is not the existence of target-specific facts; it is exact
instruction normalization and target operations being dispersed above and
below this table.

#### Semantic policy and observation planning

Access, barrier, and atomic/fence policy are separate compiled functions that
produce `ConSanObservationPlan` entries and typed reasons. The
`ConSanCoverageLedger` is also a separate concept rather than an incidental
counter in a lowerer. The pure decision functions and value types are sound.
Their assembly and publication are not yet a component: SuperCollider and MOI
each initialize plans, ledgers, alias diagnostics, and downstream projections
inside their lowerers.

#### Evidence sizing and pure models

Evidence sizing and report planning are focused in
`consan_moi_report_plan.cpp`. Report ABI/layout values and several replay or
shadow models are similarly usable without HSA lifecycle. These are real
seams, although their placement after the first lowering attempt is not the
forward pipeline implied by their contracts.

#### Generic register-window planner

`consan_resource.cpp` is a small, focused generic register-window planner. It
is a useful primitive, not the complete MOI resource-planning component. The
larger solver repeatedly changes `MoiOptions`, rebuilds site plans, freezes and
restores operating points, and coordinates fallback choices with diagnostics.

#### Independent final validation

Final validation is deliberately independent and rechecks the replacement
image, patch geometry, ABI effects, routing, spills, descriptor changes, and
engine invariants. Its detailed access to private patch proof is appropriate.
Its physical size is large, but its authority and dependency direction are
coherent.

#### Trust evaluation

The runtime side contains a pure trust-evaluation core. Separating the trust
decision from report allocation and HSA lifecycle management was an important
step. It is currently a strong function-level seam in a mixed internal header,
not yet an independently compiled host-analysis component.

#### Narrow emission plans are useful local seams

`MoiPrivateEpochPrologueEmissionPlan`,
`MoiOwnerEpochPrologueEmissionPlan`, and `MoiRecordEventEmissionPlan` group
values by an invariant and improve local reasoning. The deep read downgrades
the earlier claim that they are component boundaries: they are constructed
near emitters while surrounding code can still see `MoiOptions`, mutable
artifacts, resource state, and target internals. They are the right extraction
pattern, but not evidence that the larger boundary is enforced.

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
| Generic register-window planning | `consan_resource.cpp` |
| Independent final validation | `consan_validation.inc` |
| Pure runtime trust evaluation | `rj_hsa_dbi_hook_internal.h` |

### 1.3 Physical composition does not match semantic ownership

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

### 3.2 The stage pipeline is retrospective and orchestration is split

`TransformResult::publish_optional` invokes `lower_consan` before it marks the
inventory and observation stages or constructs evidence intent and evidence
requirements. Its stage records therefore mostly classify the artifacts that
the monolithic lowerer happened to return; they do not drive stage execution.

This is more than a naming issue because evidence requirements determine the
runtime allocation that lowering needs. The HSA hook must implement the
missing forward edge itself:

```text
unbound transform -> publish evidence size -> allocate -> retry lowerer
```

Meanwhile, `try_patch_consan_impl` owns parse/inventory, staged fault
composition, dispatch to SC or MOI, and rollback; `try_patch_consan_moi` owns
policy publication, candidate projection, the resource fixed-point, engine
dispatch, and coverage finalization. State passes among those coordinators,
the public facade, and the hook through mutable artifacts rather than explicit
stage products.

Consequences include:

- the documented pipeline order is not the executable dependency order;
- library callers do not own the complete automatic-binding protocol;
- stage status can be internally well-formed while still describing a
  different control structure from the one that ran;
- retry correctness depends on knowing which portions of a previous artifact
  bundle are pristine and reusable; and
- changes to evidence planning, mutation composition, or binding can require
  coordinated edits in three nominal layers.

The refactoring needs one authoritative transform transaction with explicit
pre-binding and post-binding products. A two-pass implementation may remain
necessary, but it must be a library protocol expressed by types and tested as
such, not an HSA-hook reconstruction of a nominal stage sequence.

### 3.3 Exact instruction admission is duplicated across policy and lowering

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

### 3.4 The real MOI resource planner is a hidden mutable fixed-point

The first pass mistook `consan_resource.cpp` for the resource-planning
component. That file plans a generic register window. The effective MOI
resource authority is spread through `consan_moi_placement.inc` and
`try_patch_consan_moi`.

The coordinator copies `MoiOptions` into mutable `effective_options`, builds
resource plans, selects automatic owner and dispatch state, rebuilds plans,
selects EXEC/VCC/SCC preservation, may discover a dynamic-stack spill only
after that selection, restores a captured operating point, truncates warnings,
and reruns planning. Later persistent-VGPR and engine-specific choices can
freeze new operating points and trigger further plan changes.

This is a legitimate constraint-solving problem, but it is represented as
mutation and rollback of the same broad options and artifact buses used by
emitters. The attempted state, accepted state, diagnostics, and site resource
plans are not distinct products. That makes fallback order semantically
significant, permits stale derived plans after an option change, and makes the
solver impossible to test without much of the lowerer in scope.

The component should eventually accept an immutable resource problem and
return an explicit attempt or accepted operating point with its site plans and
typed rejections. The exact solving algorithm can remain iterative. The
boundary, not the absence of iteration, is what matters.

### 3.5 Broad structures act as cross-component buses

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

### 3.6 Policy is pure at its core but not at its publication boundary

The compiled access, barrier, and atomic/fence policy functions are genuine
semantic cores. The rest of the supposed policy component is duplicated in
engine lowerers. `initialize_sc_access_coverage` and
`initialize_moi_access_observation_plan` each assemble policy calls, append
plans, initialize the ledger, render physical-alias failures, and project
admitted intents into a lowerer-specific work list. They mutate
`ConSanTransformArtifacts` directly.

This duplication is why policy appears well layered when only its compiled
functions are inspected but remains coupled to engine control flow in
practice. A shared observation-planning transaction should own assembly,
validation, ledger initialization, and typed publication. Engines should
provide a request and consume the resulting plan, not each implement the
publication protocol.

There is also a smaller upstream leak. `ConSanAccessInventorySite::supported_mvp`
is populated by target/form-specific decode checks and then affects
synchronization and fault candidate selection. The field is not merely a raw
decoded fact, and its legacy name does not state which engine or lowering
contract it supports. Exact normalization should become a typed target product;
inventory should retain raw/normalized semantic facts without embedding an
unnamed downstream admission decision.

### 3.7 Runtime analysis consumes lowerer patch geometry

`AutoMoiReportBufferRegistry::summarize` in
`rj_hsa_dbi_hook_moi_report.cpp` combines HSA snapshotting, ABI validation,
decoding, replay, loss accounting, conflict suppression, trust evaluation, and
rendering in one long operation.

The registry's metadata registration accepts `ConSanPatchInfo` and derives
runtime mappings from patch kinds and geometry. Replay and conflict attribution
then consume those patch-derived mappings. This directly violates the intended
direction: the host analyzer should receive a typed semantic/static mapping,
not reverse-engineer meaning from a lowerer proof object.

The deep read also reveals that the first roughly 700 lines of that source own
allocation, binding, retirement, snapshot bookkeeping, and process budgets,
while `summarize` combines snapshotting, ABI checks, engine-specific decoding,
replay/conflict analysis, loss accounting, trust, and rendering. Related host
models and presentation helpers are mixed into
`rj_hsa_dbi_hook_internal.h`. The pure trust evaluator inside this path
demonstrates that decomposition is practical. Lifecycle, decoding, analysis,
trust, and rendering need their own contracts.

### 3.8 One source file crosses the host/target boundary

The first roughly 2,500 lines of `consan_moi_model.cpp` implement host-side
report and replay models. Its final roughly 500 lines implement exact
target-atomic address planning and emission. The interfaces in that tail are
reasonably narrow, but its physical placement is a clear component error and
makes dependencies harder to state and enforce.

### 3.9 Diagnostics are not yet an edge-only concern

There are 1,858 `warnings`/`errors` `emplace_back` or `push_back` sites across
36 production files. Current design permits string diagnostics from lowerers
and final validation, so this is not an immediate correctness defect. It does,
however, mean that stable failure concepts are frequently rendered before
reaching a presentation edge.

Typed reasons should replace strings where a reason is part of a stable
component contract. Diagnostic wording should remain stable and should be
rendered at the appropriate edge. This is a later cleanup, not a reason to
destabilize the early cuts.

### 3.10 Raw architecture branches are not uniformly violations

The implementation contains 105 `ROCJITSU_CODE_ARCH_*` references across 20
files and 120 `consan_uses_gfx*` references across 19 files. Most references in
target lowering, decoding, and independent validation are legitimate. The
architectural smell is specifically raw target encoding knowledge in semantic
policy or duplicated target classifiers, not every architecture branch.

The fourth refactoring must avoid replacing explicit and reviewable target
code with an abstraction that merely hides necessary ISA differences.

### 3.11 Detailed patch access in final validation is appropriate

Final validation contains 631 `patch.*` accesses and uses much of
`ConSanPatchInfo`. This is largely correct: an independent validator must inspect
the emitted bytes, geometry, ABI effects, and claimed proof. Validation should
not be weakened merely to make dependency counts look cleaner.

The desired boundary is that detailed patch proof remains private to lowering
and validation, while semantic coverage and runtime analysis use smaller typed
products.

### 3.12 Severity summary

| Violation | Layering severity | Correctness risk |
| --- | --- | --- |
| Coverage reconstructed from patch kinds and offsets | High | Medium-high |
| Retrospective stage facade and hook-owned binding/retry protocol | High | Medium-high |
| Exact admission duplicated in policy and lowerer | High | Medium |
| MOI resource solving expressed as mutable options/artifact rollback | High | Medium-high |
| Runtime analysis consumes patch geometry | High | Medium |
| Two textual mega-translation-units | High maintenance cost | Low immediate risk |
| `MoiOptions` and artifacts used as broad buses | Medium-high | Medium |
| `TransformResult` exposes private lowerer artifacts | Medium-high | Low-medium |
| Policy assembly/publication duplicated in engine lowerers | Medium | Medium |
| Inventory embeds unnamed downstream `supported_mvp` eligibility | Medium | Low-medium |
| Report registry combines lifecycle, decode, analysis, and rendering | Medium-high | Medium |
| Raw architecture branching in target lowerers | Mostly legitimate | Low |
| Detailed patch telemetry in independent validation | Appropriate | Low |

## 4. Governing rules for the fourth refactoring

The refactoring should optimize for enforceable ownership, dependency
direction, a source tree that matches the conceptual map, and additive growth
across the mode and target dimensions. A smaller implementation is an expected
long-term consequence of deleting duplicate authorities and accidental
adapters, not a numeric acceptance criterion. Added structure must still earn
its keep: a boundary that grows the interaction term or leaves its predecessor
alive without a clear semantic gain is suspect.

1. **Preserve verified semantics and useful properties, not present shapes.**
   Retain immutable ownership, pure decisions, typed intent, independent final
   validation, and pure trust evaluation as architectural properties. Their
   current types, files, and component assignments may move when a cleaner
   design preserves or strengthens those properties. The current resource
   fixed-point and retrospective pipeline have no presumption of survival.
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
   move. A discovered bug is the exception: first capture it in a regression
   test, then fix it and validate the affected matrix rather than codifying the
   bad behavior as a refactoring invariant.
7. **Make new components directly testable.** A classifier, transaction,
   decoder, analyzer, or trust evaluator should not require a complete ELF and
   HSA execution when its immediate inputs can be constructed directly.
8. **Prefer value types grouped by invariant over long argument lists.** Narrow
   contracts should not become twenty scalar parameters.
9. **Use frequent, reviewable commits.** Each completed cut must leave one
   authoritative path and a passing relevant test set.
10. **Apply the normal ConSan gates at useful convergence points.** Focused
    tests come first, followed by the relevant engine/target matrix. Run the
    complete nonphysical suite periodically during a long stage and after each
    coherent cross-component migration, not after every mechanical edit. Each
    completed fourth-refactoring stage is checked on all five emulated targets
    with `-j16`, followed by the serialized physical `gfx1201` gate with `-j1`.
    Physical GPU jobs must never run concurrently.
11. **Assign variation to one dimension and contain it physically.** Engine
    policy may select a semantic operation; target lowering may realize it. Do
    not let each engine recreate the same target call, routing, identity, or
    register-state mechanism. Generated ISA headers, raw gfx constants, and
    target recipes must migrate out of common semantic and mode code into
    target- or target-family-owned files.
12. **Retain subset sharing explicitly and visibly.** Shared MOI, exact-shadow,
    Record/Replay-plus-InlineShadow, gfx9, gfx12, CDNA, and other real subsets
    are preferable to duplication. They must have named contracts and owned
    files or regions rather than being inferred from textual inclusion or
    scattered switches. Common code must not become a dumping ground for
    behavior that merely has two consumers.
13. **Measure both exclusive lines and incidence.** The four exclusive buckets
    detect physical growth. Per-mode and per-target incidence detects blast
    radius. Neither should be replaced by lexical token counts.
14. **Stage records must describe stages that actually executed.** If a two-pass
    bind/retry protocol is required, model both passes explicitly. Do not infer
    a forward pipeline by inspecting a monolithic lowerer's final artifact
    bundle.
15. **Shift component boundaries when the evidence calls for it.** The region
    ledger in Section 1 describes current responsibility, not the desired
    package graph. Merge concepts that share one invariant, split concepts with
    different authorities, and move responsibilities across host, pipeline,
    policy, solver, mode, and target boundaries when that produces a clearer
    dependency direction.
16. **Optimize for a stable mental map.** Every resulting component should have
    a short purpose statement, explicit inputs and outputs, one reason to
    change, owned tests, and an enforceable visibility boundary. A reader
    should not need knowledge of include order, patch-kind conventions, or
    option-bus folklore to locate behavior.
17. **Do not optimize the ledger directly.** Temporary implementation growth is
    acceptable for regression tests, typed products, or migration adapters.
    Adapters must have a deletion condition, and the final accounting must
    explain durable growth or remaining duplication, but no stage succeeds or
    fails merely by crossing 80,000 lines.

## 5. Provisional convergence route

The dependency order is intentional, but the concrete destination interfaces
are provisional. The first cuts repair information flow. Target-operation
seams are then discovered through vertical slices, not designed all at once.
Later cuts use the proven typed products to reduce visibility and split
physical translation units. These stages are labeled `F0` through `F9` to
distinguish them from the completed reimplementation stages.

This route is a dependency hypothesis, not a limit on the mandate. A deep read
during implementation may show that a proposed product belongs to another
component, that two stages must be split, or that an earlier boundary must be
replaced before the next vertical slice can work. Reorder or revise the route
when evidence requires it, document the reason, and preserve tested checkpoints
and one authority throughout the migration.

There are three workstreams inside this order:

- the **boundary workstream** moves coverage, static mapping, options, runtime
  analysis, and proof to their rightful owners; and
- the **variability workstream** extracts repeated engine/target interactions
  into reusable target operations and verifies that a second engine can consume
  them without importing target internals; and
- the **component-realization workstream** moves target, mode, subset-shared,
  solver, host, and validation responsibilities into owned files and compiled
  boundaries whose dependency graph matches the conceptual design.

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

#### F1 checkpoint — complete (2026-08-28)

F1 was implemented by commits `ad978bd7c2`, `fdfb9d2d26`, `8751dbaa7a`,
and `32f1321dfa`. Lowering now publishes typed intent-bound commit or rejection
transactions for SuperCollider and every MOI access and synchronization path.
The legacy SuperCollider and MOI reverse-coverage finalizers and their
patch-kind, anchor, and resource-plan joins are gone. The final F1 gate passed
4,672/4,672 nonphysical tests at `-j16` and 635/635 physical `gfx1201` tests at
`-j1`.

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

#### F2 checkpoint — complete (2026-08-28)

F2 was implemented by commits `9fa8edd427`, `7228ad449c`, `8d5e1604ac`, and
`4ccbf6b3b0`. Intent-bound access commits now publish a validated
`ConSanRuntimeStaticMapping` containing original semantic attribution,
execution-owner provenance, Record/Replay identity, Sampled slot/bank and
emitted/relocated locations, and InlineShadow compact-token attribution where
applicable. The aggregate mapping is required to equal the ordered projection
of committed lowerings and is independently checked by final validation.

Runtime registration, Sampled snapshot attribution, Record/Replay replay
suppression, Sampled owner-scope suppression, and compact-token registration
consume only that typed product. Raw `ConSanPatchInfo` remains in the separate
debug proof log and independent final validator, not in
`rj_hsa_dbi_hook_moi_report.cpp`. A negative hook regression supplies complete
legacy-looking Sampled patch telemetry without a typed mapping and proves that
runtime attribution remains absent.

The final F2 gate passed 4,673/4,673 nonphysical tests at `-j16` in 245.56
seconds and 635/635 serialized physical `gfx1201` tests at `-j1` in 106.21
seconds. The resulting 5,308-test inventory is the 5,302-test baseline plus six
new F1/F2 regressions, with no lost discovery.

### F3. Make the pipeline executable and separate its products

Replace the retrospective `publish_optional` model with an authoritative
library transform transaction. It must make the dependency order explicit,
including the automatic-binding case. A likely shape is:

```text
validated request
  -> semantic inventory
  -> assembled observation plan and initial ledger
  -> evidence requirements and runtime capability requirements
  -> bound resources or a typed deferred-binding result
  -> resource solving and lowering
  -> independent validation
  -> public result
```

If retaining and resuming the inventory is necessary for performance, expose a
typed immutable pre-binding product or resume token. It must state which input
image, mutation provenance, observation plan, and evidence requirements it
represents. The HSA adapter may allocate and bind resources, but it should call
the library protocol rather than own the rules for which lowering artifacts
can be retried. Pipeline stage status must be recorded as each stage executes,
not synthesized from the final aggregate.

At this boundary, replace `TransformResult`'s public inheritance from
`ConSanTransformArtifacts` with composition and explicit projections. Private
lowering artifacts should remain visible only to the transaction and final
validator. The public result should expose a reviewed set of typed products,
likely including:

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

Move the duplicated SC/MOI policy assembly into the transaction during this
stage. It should accept an engine policy request and publish one validated
observation product; engines consume the product rather than initialize their
own ledgers and render policy failures.

**F3 completion criterion:** the automatic binding/retry path is a typed
library protocol; stage records correspond to executed stages; policy
assembly has one authority; and a runtime consumer cannot access
lowerer-private artifacts through `TransformResult`.

#### F3 checkpoint — complete (2026-08-28)

F3 was implemented by commits `86d8e3a32c`, `9b04663415`, `a0cb39ed56`,
`a722c3470f`, `c71985c9d5`, `ece21d3dc4`, `987a59c56d`, `cb56ff98f3`,
`cdf4f81b7c`, `25e1b20335`, `0eb47b750c`, `4b92473a75`, `5ee6ae9243`,
and `8d318cb616`, followed by the separate lowerer-debug projection in
`66b9d50e4d` and `dc81afd32f`.

The library now owns one authoritative transform transaction and one
`assemble_consan_observation_product` policy-assembly authority. Automatic
binding stops at a typed `ConSanDeferredBinding` value which binds the input
image and mutation provenance to its inventory, observation product, evidence
requirements, and executed-stage records. Resumption validates that identity
and binding before it can lower. Stage execution counts are recorded at the
point of execution, including repeated inventory or binding work, rather than
projected from the eventual aggregate.

`TransformResult` now exposes reviewed products by composition; lowerer-private
patch, resource, and placement artifacts are available only to the transaction
and final validator. Tests and development diagnostics use the separate owned
`consan_transform_diagnostic_report` presentation product instead of reopening
the runtime result surface or aliasing lowerer-private storage. The combined
F3/F4 gate recorded below exercises both the direct and deferred transaction
paths.

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

Replace `ConSanAccessInventorySite::supported_mvp` in this stage. Preserve any
raw decoded fact it was standing in for, but attach exact lowerability to a
named normalized form and an explicit consumer contract. Synchronization and
fault planning must request the fact they need rather than inherit a generic
legacy eligibility bit.

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
no unnamed `supported_mvp` eligibility remains in semantic inventory. One
target-operation slice has replaced duplicated mode/target lowering in at
least two engines without broadening semantic authority.

#### F4 checkpoint — complete (2026-08-28)

F4 was implemented by commits `ad2f0dcdf7`, `c06bc7db1d`, `1d666146fc`,
`e37031af25`, `b75d3aa85c`, `6db6d8cc34`, `62807d7fc5`, `897fb6212c`,
`a4f46fe2cd`, `ff7f442eec`, `093540159e`, `8fe9cf83e6`, and `bc1727375c`.
The access and atomic classifier authorities now publish named normalized
forms through semantic inventory and policy into placement and emission.
These forms own exact address shape, operand-register spans, subword value
placement, partial destinations, tuple alignment, and descriptor headroom.
MOI and SuperCollider consumers no longer recover those facts from mnemonics,
raw decoded operands, origin tags, or architecture constants.

The shared normalized access and atomic-address materialization slice is
consumed by Record/Replay, Sampled, and InlineShadow across the five supported
target profiles. It therefore replaces mode/target lowering decisions in more
than two engines without moving semantic relevance into the target layer.
Semantic policy has no generated ISA-builder include or raw target encoding
constant, and semantic inventory has no `supported_mvp` field or equivalent
unnamed eligibility bit. Classifier goldens and common policy/lowering fixtures
cover `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`.

During this gate, physical regression `DbiOverflowIsVisible` exposed that a
runtime ring capacity is not a static semantic-evidence ceiling. Commit
`becd961599` fixes that distinct bug and adds the regression required by the
refactoring mandate. The final focused normalized-form slice passed 194/194
tests. The complete nonphysical gate passed 4,694/4,694 tests at `-j16` in
245.56 seconds, and the serialized physical `gfx1201` gate passed 635/635 tests
at `-j1` in 105.45 seconds.

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

Make the MOI fixed-point an explicit component in the same stage. Separate an
immutable `resource problem`, an attempted operating point, an accepted
operating point with site plans, and typed rejection/fallback information.
The solver may iterate and backtrack internally, but callers and emitters must
not participate by mutating `MoiOptions`, resizing a shared warning vector, or
assuming which derived plans survived a rollback.

Continue the variability work one operation at a time. Candidate slices after
far routing are workgroup identity, atomic address materialization, report
atomics/cache completion, VCC/SCC preservation, and selectable-bank
transitions. After each slice, recompute the four buckets, per-mode/target
incidence, adapter lines, and number of target branches remaining in engines.
Retain the seam only when a second consumer makes it narrower or deletes more
code than its adapters introduce.

**F5 completion criterion:** native emission and placement components have zero
`const MoiOptions &` parameters; the operating-point solver has direct tests
over immutable problems and explicit results; and no fallback restores state
through a shared artifact or diagnostic bus. Any surviving aggregate is
confined to top-level orchestration and is not a cross-component bus.

#### F5 checkpoint — complete (2026-08-29)

F5 was implemented by commits `e202685042` through `0b9c143560` (with the
preceding scalar-preservation contract and dummy-dependency removal in
`2f65592f71` and `1272cb11d1`). Placement and native emission now consume
named request, bound-resource, accepted-operating-point, per-site plan,
scratch, and debug contracts rather than `MoiOptions`. The 19 surviving
`const MoiOptions &` parameters are confined to top-level transformation,
composition, and engine-application entry points; none remains in placement or
native emission.

`MoiResourceProblem` binds the immutable image, target, request, resources,
inventory, observation plan, and candidate set. Solver attempts publish
`ConSanMoiOperatingPointAttempt` or `ConSanMoiResourcePlanningResult`, with
typed `ConSanMoiFallbackKind` acceptance and owned plans and diagnostics.
Direct contract tests cover immutable problem binding, structural failure
versus unsupported sites, and publication of accepted typed fallbacks. The
solver no longer snapshots or resizes a shared artifact, warning, error, or
resource-plan bus to roll back a rejected attempt.

The complete nonphysical gate passed 4,702/4,702 tests at `-j16` in 242.11
seconds across `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`. This is
eight tests more than the F4 inventory. The serialized physical `gfx1201` gate
passed 635/635 tests at `-j1` in 107.16 seconds, preserving the full physical
inventory.

### F6. Turn conceptual lowerer components into compiled components

Only after F1-F5 establish their interfaces should the textual `.inc`
composition be split. Candidate compiled components are:

- inventory construction and semantic association;
- observation-policy assembly and coverage contracts;
- target profile, exact normalization, and target operations;
- shared placement transaction;
- SuperCollider lowering;
- MOI operating-point/resource solving;
- shared MOI lowering;
- per-engine semantic planning and emission;
- independent final validation.

Use private headers for deliberately shared implementation contracts and public
headers only for true external contracts. The goal is not one translation unit
per current `.inc` file; it is a small set of coherent translation units whose
include graph enforces the intended dependency direction.

Do not realize the target boundary as one new mega-backend full of architecture
switches. Use a target-neutral operation contract, named family-shared
implementations where the ISA fact is genuinely shared, and gfx-member-owned
sources for member-specific recipes. Apply the analogous rule to modes: each
engine owns its policy/planning/emission and host analysis, while exact
multi-mode subsets receive a named internal component only after their shared
invariant is stated.

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

#### F6 checkpoint — complete (2026-08-29)

F6 was implemented by commits `4f56ace115` through `2152c3da24`. Inventory
analysis, semantic classification, policy assembly, placement, perturbation,
composition, fault selection and mutation, SuperCollider, final validation,
and the MOI subsystems are now registered compiled sources. MOI has compiled
components for target-address operations, native ABI and relocation,
candidate/probe planning, resource solving, placement, shared lowering,
prologue and synchronization support, and the Record/Replay, Sampled, and
InlineShadow engines. The former target-address tail of
`consan_moi_model.cpp` is owned by `consan_moi_target_address.cpp`.

Implementation fragments now have one owning translation unit rather than
depending on visibility established by a preceding include. `consan_moi.cpp`
is a 671-line top-level coordinator with no implementation `.inc` include.
The largest compiled implementation body is the approximately 6,750-line MOI
placement component; the SuperCollider family component is approximately
7,100 lines including its named common, LDS, and flat regions. Neither is
textually combined with the coordinator or another conceptual component.

The MOI placement workspace is opaque outside its component and is created and
owned through `MoiResourcePlanningStatePtr`. Engines and the resource pipeline
consume semantic placement queries instead of inspecting CFG blocks, liveness
objects, owner contexts, decoder caches, or reservation storage. Record/Replay
fence lowering and Sampled synchronization were moved out of the coordinator;
Sampled scratch sizing and operand-recovery rules form a pure mode-specific
resource contract rather than leaking from an emission header into placement.
The placement source no longer includes engine or target-emission headers.

Target-operation sources do not decide observation coverage or engine evidence
policy, and engine sources do not include generated target encodings. Exact
multi-mode behavior and family-shared mechanics are named compiled components,
including exact-shadow emission, shared MOI lowering, synchronization emission,
relay operations, and SuperCollider target operations, rather than accidental
include-order closures.

The complete nonphysical gate passed 4,702/4,702 tests at `-j16` in 242.80
seconds across `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`. The
serialized physical `gfx1201` gate passed 635/635 tests at `-j1` in 106.93
seconds.

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

#### F7 checkpoint — complete (2026-08-29)

F7 was implemented by commits `f9ec3ac47a`, `db941e4b40`, `62aac0a223`,
`5f99af0bd6`, `2ef23c5c93`, and `d4b9cd120d`. The report path now has compiled
components for snapshot capture, raw-evidence decoding, engine analysis, trust
evaluation, and diagnostic rendering. The lifecycle registry allocates, binds,
retires, snapshots, and aggregates report buffers; its source contains no
replay or conflict-analysis policy. The remaining pipeline is a 68-line
coordinator which passes immutable typed products between the components and
is the only reporting layer that invokes the hook logger.

Decoding, analysis, trust, and rendering have no HSA lifecycle dependency and
each has a direct host test. The renderer accepts no raw report bytes and
returns typed diagnostics whose established text is pinned independently of
the logger. Registry churn and snapshot tests cover allocation, lifetime,
visibility, copy failure, and reclamation. During extraction, the decoder
exposed and fixed an existing accumulation bug in which dynamic decoding could
overwrite malformed evidence already supplied by static lifecycle metadata;
the decoder regression now pins preservation of both sources.

The complete nonphysical gate passed 4,703/4,703 tests at `-j16` in 250.43
seconds across all five emulated targets. The serialized physical `gfx1201`
gate passed 635/635 tests at `-j1` in 108.14 seconds.

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

#### F8 checkpoint — complete (2026-08-29)

F8 was implemented by commits `026bbe2bc2`, `e0e8717496`, `f964ec569c`,
`717ded711e`, `fc6dbcfaff`, and `8abdf58025`. The former broad patch record is
now assembled from narrow proof contracts for geometry, routing, evidence,
mutation, ABI, and fault effects. `ConSanPatchPlacementEffects`,
`ConSanPatchLoweringProduct`, and `ConSanPatchMutationProduct` expose only the
products needed at each transition. Their joined `ConSanPatchInfo` remains a
private final-validation proof: validation is the only consumer that inspects
the complete product, while transform debugging and runtime resource projection
receive purpose-specific views.

Stable reasons which cross component boundaries or select control flow are now
typed for perturbation decisions, Sampled atomic semantics, exact barrier-pair
and barrier-group selection, barrier move destinations, and barrier lifecycle
groups. Rendering remains at presentation edges, and exhaustive tests pin the
established text for every typed alternative, including nested barrier-group
failures. Remaining strings are dynamic explanatory detail or one-off error
sinks paired with typed decisions; production code neither compares nor parses
them to recover semantics.

The complete nonphysical gate passed 4,707/4,707 tests at `-j16` in 243.55
seconds across all five emulated targets. The serialized physical `gfx1201`
gate passed 635/635 tests at `-j1` in 109.22 seconds.

### F9. Enforce boundaries and delete migration scaffolding

Finish by deleting adapters, reverse maps, duplicated classifiers, obsolete
fields, and compatibility paths introduced during the migration. Add durable
checks for the architectural rules that can be stated mechanically:

- semantic policy cannot include generated target instruction headers or raw
  target constants;
- target-neutral and mode-owned sources cannot contain raw gfx-member recipes;
- raw gfx-member recipes occur only in that member's files, with named
  family-shared files for real family-wide behavior;
- common sources cannot contain deep mode switches for behavior owned by one
  mode or an exact subset of modes;
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
plausible after deleting reverse coverage reconstruction, duplicate exact
classifiers, broad adapter fields, redundant runtime mappings, and migration
scaffolding. It is not required. The central hypothesis is that a tidy design
will unlock simplification that cannot be forced safely while ownership is
muddied. Falling below 80,000 lines would therefore be an outcome to explain,
not a target to optimize; remaining above it is acceptable when every durable
component and line category has a clear owner and purpose.

## 6. Test and commit discipline

Each stage should be a sequence of small local commits. A useful commit changes
one authority or one consumer set, adds or updates its focused tests, and leaves
the tree in a bisectable state. Never leave both old and new authorities active
across a stage boundary. Do not knowingly commit a regression in the tests
relevant to the cut; the full matrix is a periodic checkpoint rather than a
tax on every edit.

For each stage:

1. run the focused host/unit tests for the component being changed;
2. run the relevant engine and architecture matrix while iterating;
3. during a long stage, run the complete nonphysical ConSan suite at coherent
   convergence points, particularly after replacing an authority or changing
   a cross-component contract;
4. at stage completion, build with `-j16` and run the complete nonphysical
   ConSan suite with `ctest -j16`;
5. then run the physical `gfx1201` tests alone with `ctest -j1`;
6. compare the test inventory with the baseline so that a green result caused
   by lost test discovery cannot pass the gate;
7. update this document or the handoff with the stage result and exact commit.

The five emulated targets remain the routine cross-architecture gate. Physical
GPU tests are serialized and less frequent during iteration, but mandatory at
each completed stage. Existing paired device tests must not be weakened to make
a boundary change pass.

When investigation exposes an existing bug, treat it as part of the current
work rather than deferring it behind the refactoring:

1. reduce the bug to the smallest useful reproducer and identify the affected
   modes and targets;
2. add a regression test that fails for the bug for the right reason;
3. fix the bug at the component that owns the violated invariant, even when
   that requires adjusting the provisional boundary;
4. run the focused test, affected mode/target matrix, and any neighboring
   contract tests; and
5. commit the bug fix and regression test as a distinct reviewable change when
   practical, then resume the boundary migration.

A test that only encodes the current implementation shape is not a substitute
for a semantic regression test. Conversely, an architecture- or mode-specific
bug should have a specific fixture when that specificity is real; it should not
force target or mode knowledge back into common code.

## 7. Definition of success

The fourth refactoring is complete when all of the following are true:

- the production source and build graph present a short, stable component map
  whose names, purposes, inputs, outputs, and dependency direction can be
  explained without reference to textual include order;
- every cross-component semantic fact and decision has one named authority and
  a typed forward product;
- coverage is produced directly from intent-bound commit/rejection events and
  contains no patch-shape reconstruction;
- runtime registration, replay, and analysis consume a validated semantic
  mapping and do not consume `ConSanPatchInfo`;
- exact instruction lowerability has one target-classifier authority;
- semantic policy has no generated ISA builders or raw architecture encoding
  constants;
- stage records are emitted by one executable library pipeline, including a
  typed deferred-binding/resume protocol where needed;
- observation-policy assembly and ledger initialization have one authority;
- semantic inventory contains no unnamed downstream eligibility flag such as
  `supported_mvp`;
- `TransformResult` does not publicly expose private lowerer artifacts;
- MOI resource solving accepts immutable problems and returns explicit
  attempted or accepted operating points without shared-bus rollback;
- native emitters and placement helpers no longer accept `MoiOptions` as a
  general-purpose bus;
- host lifecycle, decoding, analysis, trust, and rendering are separately
  testable components;
- final validation retains independent access to all proof needed to verify the
  emitted program;
- the two mega-translation-units have been replaced by compiled components with
  declared interfaces and enforced dependency direction;
- raw gfx-member-specific implementation lives in that target's owned files,
  and genuinely family-shared implementation lives in named family-owned
  files; common semantic and engine components do not contain scattered raw
  architecture recipes;
- mode-specific implementation lives in mode-owned files or regions, and code
  shared by an exact subset of modes lives in a named subset component rather
  than being duplicated or hidden behind deep common-code switches;
- all five emulated architectures and serialized physical `gfx1201` tests pass,
  with no loss of test inventory;
- every bug discovered during the work has a regression test and an owner-level
  fix validated on its affected modes and targets;
- implementation size and coupling metrics are recomputed and documented;
- the mode/target interaction bucket is smaller, or every surviving region has
  an explicit justification and owner;
- adding an engine does not require copying target routing, identity,
  addressing, or register-state mechanisms, and adding a target does not
  require editing each engine for an already-normalized operation;
- shared-MOI and other subset-shared implementations have declared contracts
  and enforced visibility rather than relying on textual inclusion.

There is deliberately no success condition tied to an 80,000-line threshold.
The final accounting should reveal whether clarified ownership unlocked
deletion, but legibility, correctness, containment, and enforceable dependency
direction decide completion. The final component inventory should be direct
source material for the later design-document update, not another forensic
survey of what the code actually does.

The key outcome is not merely a rearranged source tree. It is that a future
change to a patch kind cannot alter semantic coverage, a new target encoding
cannot be admitted differently by policy and lowering, and the host analyzer
cannot learn semantic meaning by peering into target-patching internals. Once
those properties are enforced by types and compilation boundaries, the
component architecture described in the design will also be the architecture
the code is capable of expressing. Once repeated target operations are owned by
target components, the fifth engine or sixth target should extend one dimension
rather than reopening a matrix of engine-specific target lowerers.
