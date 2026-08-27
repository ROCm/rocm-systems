# ConSan implementation-size study

This document breaks down the current ConSan implementation so that future
redesign and deletion work can be prioritized using more than the total line
count. It is a size map, not a claim that every line has equal complexity or
that the largest component is automatically the best deletion target.

## Snapshot and method

The snapshot is commit `f727f422dc`. The measured implementation consists of
the production files under:

- `lib/rocjitsu/src/rocjitsu/code/patch/consan/`; and
- `lib/rocjitsu/src/rocjitsu/hooks/consan/`.

Tests, documentation, and `CMakeLists.txt` files are excluded. The counter
removes blank lines and comments, retaining a line when any implementation
text remains on it. It is the same counter used for the quantitative baseline
in `PLAN.md`.

The resulting total is **83,402 code lines in 79 files**. The deletion-driven
phase started from 89,549 lines at `c6dbe1da20^`, so the current snapshot is
6,147 lines, or 6.9%, smaller.

The component classifications below are exact sums of whole files. A large
`.inc` file often performs several jobs, so a file-level component name states
primary ownership rather than attributing every line semantically. The later
architecture-awareness breakdown is explicitly lexical and should be read as
an entanglement signal, not as a count of code that must be target-specific.

## Breakdown by feature ownership

This first division follows the engine or subsystem that owns each file.
Files beginning with `consan_moi_` are classified as shared MOI unless their
name identifies Record/Replay, Sampled, or InlineShadow. Files beginning with
`consan_supercollider` belong to SuperCollider. The remaining patcher files
form the shared transform core.

| Component | Current lines | Share | Phase start | Net change | Component change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared MOI | 28,248 | 33.9% | 29,431 | -1,183 | -4.0% |
| Shared transform core | 23,935 | 28.7% | 25,811 | -1,876 | -7.3% |
| Runtime hook and reporting | 9,044 | 10.8% | 9,868 | -824 | -8.4% |
| MOI InlineShadow | 8,856 | 10.6% | 9,509 | -653 | -6.9% |
| SuperCollider | 6,879 | 8.2% | 7,144 | -265 | -3.7% |
| MOI Sampled | 4,427 | 5.3% | 5,121 | -694 | -13.6% |
| MOI Record/Replay | 2,013 | 2.4% | 2,665 | -652 | -24.5% |
| **Total** | **83,402** | **100.0%** | **89,549** | **-6,147** | **-6.9%** |

At this granularity, 52,183 lines, or 62.6%, are in shared transform/MOI
files; 22,175 lines, or 26.6%, are flavor-specific; and 9,044 lines, or 10.8%,
are runtime integration. “Shared” here means not owned by one instrumentation
flavor. It does not mean architecture-neutral: most large shared files still
contain target decisions.

Record/Replay has already become the smallest flavor-specific implementation
and saw the largest proportional reduction. Shared MOI remains by far the
largest component and shrank less than the overall implementation. That is the
main quantitative reason to focus the next internal redesign on common MOI
planning and emission rather than on another isolated Record/Replay cleanup.

## Shared MOI breakdown

The 28,248 shared-MOI lines can be divided by the primary responsibility of
their files:

| Shared-MOI responsibility | Lines | Share of total | Included implementation |
| --- | ---: | ---: | --- |
| Placement and admission | 7,219 | 8.7% | Placement, candidate construction, probe planning |
| Emission and prologue mechanics | 6,185 | 7.4% | Common emission, prologues, access application |
| Synchronization and barriers | 6,108 | 7.3% | Shared synchronization lowering and barrier handling |
| Evidence models and report planning | 5,864 | 7.0% | MOI model, shadow models, report plans/layout/helpers |
| Orchestration, ABI, and internal contracts | 2,872 | 3.4% | Pipeline, top-level lowering, result/core/ABI types |
| **Shared MOI total** | **28,248** | **33.9%** | |

The three mechanical groups—placement, emission/prologue, and
synchronization/barriers—are each about 6,000–7,000 lines. No single small
helper deletion can materially change the total. A larger reduction requires
giving each group a typed input/output contract and then deleting policy,
resource, and target decisions that are currently repeated across those
groups.

In particular, `consan_moi_placement.inc` still combines candidate admission,
register and private-memory allocation, retry/fallback policy, target
capabilities, descriptor growth, and publication into mutable lowering state.
Its size is therefore evidence of mixed ownership, not evidence that
“placement” is intrinsically a 6,212-line operation.

## Shared transform-core breakdown

The 23,935 non-MOI-core lines divide as follows:

| Transform-core responsibility | Lines | Share of total | Included implementation |
| --- | ---: | ---: | --- |
| Semantic analysis, admission, policy, and inventory | 9,595 | 11.5% | Access/synchronization analysis and typed semantic policy |
| Validation | 4,411 | 5.3% | Final structural and emitted-code validation |
| Mutation, placement, and composition | 4,094 | 4.9% | Fault injection and transform composition |
| Orchestration, contracts, and resources | 3,319 | 4.0% | Requests, results, pipeline, options, descriptor/resource types |
| Branch-only relay search | 2,516 | 3.0% | Relay-router algorithm and interface |
| **Shared transform-core total** | **23,935** | **28.7%** | |

The 9,595-line semantic group deserves conceptual review, but it should not be
assumed to be mostly redundant: some of its size represents the actual ConSan
contract. Conversely, the 4,411 validation lines should not be deleted merely
because they are large. Validation is most likely to shrink safely after typed
plans make invalid intermediate combinations unrepresentable and eliminate
the need to rediscover planning facts from emitted bytes.

The branch-only relay router is a focused algorithm despite its 2,516 lines.
Its size alone is weaker evidence for redesign than the mixed responsibilities
in placement or emission.

## Flavor-specific and runtime breakdown

| Component | Lines | Share of total | Largest internal pieces |
| --- | ---: | ---: | --- |
| MOI InlineShadow | 8,856 | 10.6% | Main planning 2,965; emission 2,963; atomics 2,190 |
| SuperCollider | 6,879 | 8.2% | LDS 4,048; FLAT 1,865; common 963 |
| MOI Sampled | 4,427 | 5.3% | Synchronization 3,182; accesses 1,245 |
| MOI Record/Replay | 2,013 | 2.4% | Main implementation 1,544; atomics 299 |
| Runtime hook and reporting | 9,044 | 10.8% | Hook coordinator 4,654; report processing 1,982; configuration 1,158 |

InlineShadow is the largest flavor-specific implementation. Its unique shadow
semantics should remain distinct, but the size distribution suggests that its
planning, preservation, ABI, and instruction-emission mechanics should be
compared against shared MOI before attempting an InlineShadow-only rewrite.

SuperCollider has shrunk by only 265 lines during the deletion-driven phase.
Its 4,048-line LDS implementation is therefore a relatively untouched
candidate for a separate design study after the common MOI seams are clearer.

The hook is large, but DBI is expected eventually to own more of that
integration boundary. Heavy refactoring of `rj_hsa_dbi_hooks.cpp` solely for
local aesthetics risks polishing code that DBI migration will delete. The
near-term goal should be to make the ConSan-to-hook contract narrow and make
the eventual deletion boundary explicit.

## File concentration

The largest 5 files contain 27.2% of the implementation, the largest 10
contain 45.7%, the largest 18 contain 69.3%, and the largest 30 contain 85.2%.
The top 20 files are:

| Rank | File | Lines | Share | Cumulative share |
| ---: | --- | ---: | ---: | ---: |
| 1 | `consan_moi_placement.inc` | 6,212 | 7.4% | 7.4% |
| 2 | `rj_hsa_dbi_hooks.cpp` | 4,654 | 5.6% | 13.0% |
| 3 | `consan_validation.inc` | 4,411 | 5.3% | 18.3% |
| 4 | `consan_supercollider_lds.inc` | 4,048 | 4.9% | 23.2% |
| 5 | `consan_moi_sync_common.inc` | 3,370 | 4.0% | 27.2% |
| 6 | `consan_sync_analysis.inc` | 3,309 | 4.0% | 31.2% |
| 7 | `consan_moi_sampled_sync.inc` | 3,182 | 3.8% | 35.0% |
| 8 | `consan_moi_prologue.inc` | 2,996 | 3.6% | 38.6% |
| 9 | `consan_moi_inline_shadow.inc` | 2,965 | 3.6% | 42.1% |
| 10 | `consan_moi_inline_shadow_emission.inc` | 2,963 | 3.6% | 45.7% |
| 11 | `consan_moi_common_emission.inc` | 2,879 | 3.5% | 49.1% |
| 12 | `consan_moi_model.cpp` | 2,828 | 3.4% | 52.5% |
| 13 | `consan_moi_barrier.inc` | 2,738 | 3.3% | 55.8% |
| 14 | `consan_analysis.inc` | 2,470 | 3.0% | 58.8% |
| 15 | `consan_fault_injection.inc` | 2,390 | 2.9% | 61.6% |
| 16 | `consan_branch_only_relay_router.cpp` | 2,252 | 2.7% | 64.3% |
| 17 | `consan_moi_inline_atomic.inc` | 2,190 | 2.6% | 67.0% |
| 18 | `rj_hsa_dbi_hook_moi_report.cpp` | 1,982 | 2.4% | 69.3% |
| 19 | `consan_supercollider_flat.inc` | 1,865 | 2.2% | 71.6% |
| 20 | `consan_moi_record_replay.inc` | 1,544 | 1.9% | 73.4% |

This concentration makes the next design review tractable: understanding the
responsibilities and repeated mechanisms in roughly 20 files covers nearly
three quarters of the implementation.

## Architecture-awareness breakdown

To measure architectural entanglement without pretending to understand every
statement semantically, comment-stripped code was classified lexically:

- **explicit product/family knowledge**: the file contains `gfx9`, `gfx11`,
  `gfx12`, `CDNA`, or `RDNA` vocabulary;
- **abstract target/profile consumer**: it has architecture, target-profile,
  encoding-family, or capability vocabulary but no explicit product/family
  token; and
- **no target vocabulary**: neither of the above appears.

| File-level class | Files | Lines | Share |
| --- | ---: | ---: | ---: |
| Explicit product/family knowledge | 39 | 63,878 | 76.6% |
| Abstract target/profile consumer | 18 | 13,901 | 16.7% |
| No target vocabulary | 22 | 5,623 | 6.7% |
| **Total** | **79** | **83,402** | **100.0%** |

This does **not** mean that 76.6% of ConSan is architecture-specific. Only
1,358 code lines directly contain one of the explicit target/family tokens,
or 1.6% of the implementation. Rather, those 1,358 lines are interleaved with
shared logic in files containing 63,878 lines. The size of the contaminated
surface is the useful result: architecture decisions have not yet been
confined to narrow adapters.

The 1,358 direct-reference lines are distributed as follows:

| Component | Direct target-reference lines | Share of direct references |
| --- | ---: | ---: |
| Shared MOI | 577 | 42.5% |
| Shared transform core | 401 | 29.5% |
| MOI InlineShadow | 190 | 14.0% |
| SuperCollider | 108 | 8.0% |
| MOI Sampled | 64 | 4.7% |
| MOI Record/Replay | 18 | 1.3% |
| Runtime hook and reporting | 0 | 0.0% |

Moving target selection behind profiles and narrow ISA/ABI operations would
not delete all 1,358 lines, much less all 63,878 mixed-file lines. It would,
however, make common algorithms readable without five-target reasoning and
make subsequent cross-architecture deduplication much safer.

## Where to turn attention: incremental refactoring plan

The measurements imply a dependency-ordered refactoring plan, not merely a
ranking by file size. Architecture separation is the cross-cutting design axis
for every stage: it begins before the first component cutover and is measured
throughout rather than being deferred to an eventual target-adapter rewrite.
The execution gate in `PLAN.md` still applies; this section defines what the
next work should accomplish and how to know whether it did so.

### Mandate: deep redesign, incremental landing

This is not a mandate for conservative cleanup or a sequence of small-scale
refactorings that preserve the prototype's shape. The production destination
may—and where useful, should—substantially replace ConSan's data structures,
control and data flow, internal interfaces, algorithms, file boundaries,
engine decomposition, and target decomposition. No internal type, mutable
state object, helper hierarchy, compatibility seam, or implementation pattern
is protected merely because the prototype currently uses it.

Optimize the final implementation in this order:

1. remove the most code while preserving the ConSan behavioral contract;
2. make the remaining concepts, ownership, dependency direction, and component
   responsibilities substantially clearer;
3. maximize genuinely shared semantics and mechanics across engines and gfx
   architectures; and
4. confine unavoidable target differences to narrow, explainable boundaries.

The constraints are behavioral, not structural: all checked-in tests remain
green, and observable behavior must not change significantly in a direction
misaligned with ConSan's goals. Internal source compatibility with the
prototype is not a goal. A behavior change that appears desirable must be
identified and justified rather than smuggled through as refactoring.

“Incremental” describes how this deep redesign lands, not how ambitious it may
be. Design the substantially better end state and its dependency order first;
then replace one sufficiently small component or vertical slice at a time so
that every intermediate commit remains buildable, testable, and bisectable.
Do not prefer a low-impact local edit when it entrenches a prototype structure
that the end-state design should delete.

### Governing rules and tranche accounting

- Make no global switch. Introduce the smallest durable contract that can
  replace one responsibility, migrate one consumer or operation family, and
  delete the superseded fields, branches, and helpers in the same short
  tranche.
- Treat architecture separation as a deletion constraint, not an additive
  facade. A new target abstraction is successful only when its old product or
  family branches disappear from their former consumers.
- Share semantic policy, inventory, resource intent, and orchestration. Keep
  genuinely different ABI facts and instruction sequences behind narrow
  target operations. Do not replace scattered branches with one generic
  mega-switch called an adapter.
- Use Record/Replay as the first consumer when a production interface needs a
  proving ground because it is the near-term DBI priority. The interface must
  remain engine-neutral and must subsequently delete code from Sampled and
  InlineShadow; a Record/Replay-only abstraction is not the objective.
- Preserve flavor-specific evidence semantics. Similar-looking lowering code
  is not shareable merely because its emitted instructions resemble another
  engine's instructions.
- Keep commits small enough to bisect. Compile each coherent batch, run focused
  tests for changed behavior or fragile contracts, and run the broad checked-in
  host and emulated-device gates after several related commits or at a tranche
  boundary. Run physical-gfx950 tests periodically and at major gates rather
  than after every internal move.
- End-to-end workload revalidation remains outside this internal refactoring
  phase. Use the checked-in host and device contracts to iterate; resume E2E
  qualification after the implementation converges, as specified by `PLAN.md`.
- At every tranche boundary, record total implementation lines, lines in the
  affected component, implementation lines added and deleted, remaining
  `ConSanOptions` planning fields, direct target-reference lines and files, and
  large mixed files. A temporarily net-positive tranche must name the immediate
  deletion that pays it back; do not accumulate infrastructure for a distant
  cutover.
- New or materially changed contracts receive generous type documentation and
  focused unit tests where their behavior is testable. Existing correct/incorrect
  device-test pairs remain the behavioral gate across every supported target.

The plan covers every measured component as follows; a large component is not
left implicit merely because several stages will change it.

| Measured component | Lines | Primary stages | Intended disposition |
| --- | ---: | --- | --- |
| Shared MOI placement/admission | 7,219 | 0–2 | Replace mutable planning state with one typed operating point |
| Shared MOI emission/prologues | 6,185 | 1, 3 | Separate semantic requests from narrow target operations |
| Shared MOI synchronization/barriers | 6,108 | 1, 4 | Separate causal semantics, flavor evidence, placement, and emission |
| Shared MOI evidence/report planning | 5,864 | 5 | Give evidence and report intent one typed authority |
| Shared MOI orchestration/contracts | 2,872 | 2–6, 10 | Retain only contracts that own state or enable immediate deletion |
| Core semantic analysis/policy/inventory | 9,595 | 0, 2, 4, 5 | Consolidate inventory and policy without erasing behavioral semantics |
| Core validation | 4,411 | 10 | Simplify after authoritative plans make mirror checks redundant |
| Core mutation/placement/composition | 4,094 | 6 | Compose typed plans and mutations without parallel state or reanalysis |
| Core orchestration/contracts/resources | 3,319 | 2–6, 10 | Collapse transitional boundaries as their consumers migrate |
| Branch-only relay search | 2,516 | 2, 10 | Retain as a focused leaf unless Stage 0 proves duplicated responsibility |
| InlineShadow | 8,856 | 7 | Retain shadow semantics; delete private copies of common mechanics |
| SuperCollider | 6,879 | 8 | Reuse production boundaries without merging sanitizer semantics |
| Sampled | 4,427 | 2–5, 7 | Retain sampling/evidence policy; delete common mechanics |
| Record/Replay | 2,013 | 2–5, 7 | Act as first consumer, then delete proving-ground adapters |
| Runtime hook/reporting | 9,044 | 0, 9 | Narrow and map to DBI ownership; avoid a cosmetic rewrite |

### Stage 0: map responsibilities and design the cut sequence

Before changing another large component, turn the file-level data into a
function- and state-level map. This is the design work needed to avoid creating
a second implementation beside the first.

The concrete Stage 0 map, target-reference ledger, `ConSanOptions` ownership
inventory, end-state interfaces, first vertical cuts, and code-size opportunity
range live in [`REIMPLEMENTATION_MAP.md`](REIMPLEMENTATION_MAP.md).

1. Classify the functions and mutable state in the top 20 files, which cover
   73.4% of the implementation, as semantic analysis/policy, evidence policy,
   candidate admission, resource planning, allocation, retry/fallback,
   mutation, target ABI, native emission, reporting, or validation.
2. Classify all 1,358 direct target-reference lines as semantic capability,
   resource limit, ABI fact, instruction encoding/emission, product workaround,
   diagnostic-only naming, or an unjustified product check. Record which
   existing target-profile facility can own each fact and which missing narrow
   operation is actually required.
3. Inventory every remaining `ConSanOptions` field by producer, consumers,
   lifetime, semantic owner, and whether it is caller input or a planning
   output. Planning outputs and attempt-local facts must have a named typed
   destination and deletion slice.
4. Map repeated operations across Record/Replay, Sampled, InlineShadow, and
   SuperCollider. Distinguish shared meaning from textual coincidence before
   proposing a common component.
5. Define the end-state interfaces and the complete incremental cut sequence.
   Every proposed type or component must identify the old representation and
   approximate code body it will delete during the same stage.

Stage 0 is complete when placement, emission, synchronization, flavor cleanup,
and target separation each have component boundaries, dependency direction,
first consumers, deletion points, and focused test obligations. It should also
produce a reasoned code-size opportunity range; this lexical study alone cannot
provide one.

### Stage 1: establish the target boundary and keep it as a cross-cutting gate

The architecture result is one of the study's strongest signals: only 1,358
lines directly name a product/family, but they make 63,878 lines of otherwise
shared files architecture-aware. The first implementation stage establishes
where those decisions belong and then applies that boundary throughout later
component work.

1. Keep high-level semantic capabilities and supported-feature policy in the
   typed target profile.
2. Represent wave, register-bank, descriptor, kernarg, LDS, and instruction-
   availability facts as typed ABI/resource facts rather than product lists in
   algorithms.
3. Define narrow target operations for native encodings that are genuinely
   different. Their interfaces describe the semantic operation—such as saving
   an execution mask or emitting an atomic claim—not a gfx product switch.
4. Confine unavoidable product workarounds to named, documented target-policy
   entries with focused tests.
5. Migrate target decisions one domain at a time as Stages 2–8 cut over their
   consumers. Delete the former branches immediately; do not wait for all
   1,358 references to move at once.

The foundation gate before Stage 2 is a reviewed taxonomy, typed profile/ABI
facts, narrow target-operation interfaces for the first Record/Replay vertical
slice, and a ledger assigning every direct target reference to a later cutover
or a justified final owner. Stage 1 then remains open as a cross-cutting gate
through Stage 8.

Its final exit criterion is not zero target vocabulary everywhere: target
profiles, adapters, diagnostics, and validation fixtures may legitimately name
products. It is that shared semantic, placement, resource, and orchestration
algorithms no longer select behavior from product names, and every remaining
direct reference outside a target-owned component has an explicit justification.

#### Stage 1 foundation exit evidence

The first Record/Replay vertical slice and the target-boundary foundation are
complete. All 18 Record/Replay-owned direct target references from the Stage 0
ledger have moved to existing typed profile facts, generic program/allocation
state, or target-native instruction builders; the Record/Replay-owned source
now has no direct product/family vocabulary. The detailed migration ledger is
in [`REIMPLEMENTATION_MAP.md`](REIMPLEMENTATION_MAP.md#stage-1-target-boundary-foundation).

Focused coverage passed 160/160 host checks and 28/28 paired Record/Replay
device checks across all five emulated targets. The required full gate was:

```sh
ctest -j64 --output-on-failure -R 'ConSan|consan'
```

All **5,245/5,245** tests passed, including 3,501 checked-in device tests and
593 physical-gfx950 tests, in 568.71 seconds of wall time. Stage 1 remains an
active cross-cutting gate through Stage 8; its broader final exit is audited
after the domain migrations rather than claimed by this foundation slice.

### Stage 2: replace mutable MOI placement state with one operating point

Placement/admission is 7,219 lines, and `consan_moi_placement.inc` alone has 191
direct target-reference lines. It is the first high-payoff component cut because
it currently mixes admission, register/private allocation, descriptor growth,
retry policy, and publication through mutable lowerer state.

Cut it over in this order:

1. make candidate admission return a typed admitted-site set and rejection
   evidence instead of mutating engine-global containers;
2. make resource requirement collection a pure derivation from admitted sites,
   inventory, engine policy, and target facts;
3. make allocation produce one typed operating point owning register
   assignments, private/LDS storage, descriptor requirements, persistence
   proofs, and caller-versus-automatic provenance;
4. make retry and fallback consume and replace complete attempted operating
   points rather than partially restoring mutable options; and
5. publish emitted-patch and telemetry results from the accepted operating
   point, deleting the old `ConSanOptions` fields after each consumer moves.

Use Record/Replay for the first full path, then move Sampled and InlineShadow
before declaring the contract shared. Stage 2 exits when planning outputs are
no longer stored in `ConSanOptions`, each accepted patch refers to one
authoritative allocation/resource result, placement no longer makes direct
product selections, and the old parallel state and retry protocols are absent.

#### Stage 2 exit evidence

Stage 2 is complete. `ConSanOptions` now contains only caller-owned contracts;
explicit register inputs have `requested_*` names and seed, but never alias,
the selected registers in a lowering attempt. Both an in-progress attempt and
the published result use the same complete `ConSanMoiOperatingPoint`. That type
owns code-object-wide choices and owner-local transient and persistent
assignments. Fallback checkpoints copy and restore the complete value, and
resource analysis holds a reference to that same attempt instead of a parallel
result-side allocation.

Admission is the typed `ConSanObservationPlan` produced before lowering.
Resource derivation returns a private `ConSanMoiResourcePlanningResult`
containing the complete plan set and any structural diagnostics; the caller
publishes both together. A typed unsupported site is therefore distinct from a
failed planning transaction. Sampled, InlineShadow, and Record/Replay all use
these contracts. The removed compatibility state includes
`ConSanMoiResolvedState`, `ConSanMoiTransientSgprState`, the duplicated default
transient bundle, partial owner-vector rollback, and caller/selection register
name aliasing.

Placement has no runtime `ROCJITSU_CODE_ARCH_*` comparison or product switch.
Its remaining family vocabulary is expressed through target-profile facts,
named encoding/ABI predicates, native mechanics, comments, and compile-time
profile assertions. Its comment-excluded direct target-reference lines fell
from 191 to 95. No planning-output field remains in `ConSanOptions`.
`MoiOptions` remains a 243-reference transitional aggregate of immutable input
plus the operating point; Stage 3 deletes that broad emitter interface one
semantic operation at a time.

At this exit, the measured implementation has 94,055 physical lines, 89,801
nonblank lines, and 83,441 comment-excluded code lines in the same 79 files.
That is 39 code lines above the 83,402-line Stage 0 snapshot, while
`consan_moi_placement.inc` fell from 6,212 to 6,184 code lines. Since the Stage
1 foundation, production changes added 701 and deleted 637 physical lines.
The temporary net-positive total is the cost of the now-enforced typed
boundaries; its immediate payback is Stage 3's removal of broad `MoiOptions`
emitter plumbing and repeated target/mechanism decisions. Under the lexical
architecture classifier, 35 files/61,579 lines retain explicit product or
family vocabulary, 24 files/19,594 lines consume abstract target vocabulary,
and 20 files/2,268 lines contain neither.

The fast shared-MOI/input-contract gate passed 846/846. The complete
non-physical gate passed 4,654/4,654 in 209.55 seconds wall time. The full run
exercised all 593 physical-gfx950 tests: 592 passed, while
`SampledModuleLifecycle.Correct` had one output-free 60-second timeout and then
passed alone in 0.16 seconds. Thus every one of the 5,247 checked-in ConSan
tests passed at this Stage 2 revision; the isolated timeout is recorded rather
than misclassified as a semantic failure.

### Stage 3: separate semantic emission requests from target emission

Common emission and prologue mechanics account for 6,185 lines. They are the
next target because shared meaning and target encodings repeatedly meet there.

1. Define typed semantic requests for each reusable operation family: entry
   initialization, register/execution-mask preservation, address formation,
   report reservation/claim, workgroup identity, atomics/fences/barriers, and
   guest-instruction relocation.
2. Reuse existing RocJitsu builders when they express the required operation.
   Add a ConSan target operation only for a missing semantic primitive.
3. Move one operation family at a time behind the target boundary, starting
   with Record/Replay, then transport it to Sampled and InlineShadow.
4. Delete engine-local instruction sequences, address calculations, ABI
   inventories, and target branches once all consumers use the shared request.
5. Keep evidence selection and report meaning outside the target emitter.

Stage 3 exits when shared emitters consume typed operating points and semantic
requests, target-specific instruction choices are confined to narrow target
operations, and prologue/emission code no longer recomputes placement or
descriptor facts.

#### Stage 3 exit evidence

Stage 3 is complete. Reusable emission mechanics now cross the planning-to-
emission boundary as immutable typed requests or plans. The covered families
include private and register-backed entry initialization, scalar/execution-mask
preservation, dynamic record-address formation, report-counter reservation,
resident-wave and work-item owner identity, workgroup-key formation and shadow
clearing, guest-access relocation, cache refresh, and global atomic completion.
These contracts are defined in `consan_moi_internal.h`, carry documented
invariants, and have direct host-unit coverage. Record/Replay established the
boundary first; shared owner derivation, preservation, relocation, and target
operations are also used by Sampled and InlineShadow where their semantics
match.

The final prologue emitters consume `MoiPrivateEpochPrologueEmissionPlan` or
`MoiOwnerEpochPrologueEmissionPlan`. Resource counts, spill choices, ABI
sources, runtime-selection facts, workgroup identity sources, and owner
derivation are resolved before those plans are published. The emitters no
longer read the descriptor or placement options to rediscover those choices.
Access and synchronization paths similarly delegate their reusable target
mechanics to narrow operations such as `MoiDynamicRecordAddressRequest`,
`MoiAtomicCounterIncrementRequest`, `MoiResidentWaveOwnerRequest`, and
`MoiGuestAccessRelocationRequest`; evidence selection and report meaning remain
outside those operations.

This tranche deliberately pays for explicit contracts before later stages
delete all of the surrounding policy duplication. Relative to the Stage 2
revision, production changes added 1,800 and deleted 907 physical lines. The
measured implementation is now 94,948 physical lines, 90,594 nonblank lines,
and 83,894 comment-excluded code lines in the same 79 files: a temporary
453-code-line increase. Immediate deletion included 177 net physical lines
from common emission, 50 from candidate construction, 20 from placement, and
7 from pipeline orchestration, along with duplicated per-engine owner and
special-state sequences. The remaining size debt is explicit input to Stages
4, 5, and 7, which can now reuse these contracts rather than growing another
parallel lowering path.

The complete checked-in ConSan gate passed 5,271/5,271 tests at the Stage 3
revision. It included 3,501 device tests and all 593 serialized physical-gfx950
tests, and completed in 574.58 seconds wall time with `ctest -j64`. The slowest
test, `ConSanGfx1250HipMoiSim.JakubProducerSkewBarrierDrop`, completed in 65.01
seconds; there were no failures or timeouts.

### Stage 4: split synchronization semantics, flavor evidence, and emission

Shared MOI synchronization/barrier code is 6,108 lines; core synchronization
analysis plus Sampled synchronization contributes another 6,491. This is a
large opportunity but also the highest-risk boundary because similar event
shapes can have different evidence meaning.

1. Make one immutable synchronization graph own normalized events, ordering,
   participant identity, and causal relationships.
2. Give each engine a small evidence-policy step that selects what must be
   observed or perturbed from that graph.
3. Express the result as engine-neutral resource and placement requests where
   the meaning really is shared.
4. Lower the selected semantic operations through the Stage 3 target boundary.
5. Migrate by event family—ordinary accesses, atomics, fences, then barriers—
   so every intermediate commit retains a complete path and its tests.

Record/Replay again proves the host-processing path first, followed by Sampled
and InlineShadow. Stage 4 exits when discovery/causality is not repeated by
engines, evidence policies remain explicit, and synchronization emission no
longer performs semantic rediscovery or private target selection.

#### Stage 4 exit evidence

Stage 4 is complete. `SynchronizationInventoryView` now owns the normalized
event, sequence, communication, and stable-identity joins used by all MOI
engines. In particular, implicit gfx1250 ordered-LDS scope is normalized in
the shared graph rather than repaired by one policy or lowerer. Atomic, fence,
and barrier evidence start from admitted typed policy intents and become
`MoiAtomicEvidenceSitePlan`, `MoiFenceEvidenceSitePlan`, or
`MoiBarrierEvidenceSitePlan` values before resource planning or emission.
Those values carry the unique graph association, placement event, decoded
operation, communication scope, patch interval, and other family-specific
facts needed downstream. Their invariants and the graph's unique-query
semantics have direct host-unit coverage.

Record/Replay, Sampled, InlineShadow, and shared resource planning now consume
those plans. The migration deleted engine-private synchronization indexes,
raw event/sequence rescans, candidate canonicalizers, post-policy admission
filters, completing-barrier rediscovery, and lowerer-side scope repair. Flavor
policy still explicitly selects the evidence it needs, while graph discovery
and causality and target emission each have one owner.

Relative to the Stage 3 revision, production changes added 834 and deleted
1,113 physical lines. The measured implementation is now 94,669 physical
lines, 90,298 nonblank lines, and 83,535 comment-excluded code lines in the
same 79 files: reductions of 279 physical, 296 nonblank, and 359 code lines.

The full checked-in ConSan gate exercised all 5,276 tests, including 3,501
device tests and all 593 serialized physical-gfx950 tests, in 618.61 seconds
wall time with `ctest -j64`. One physical SuperCollider clean test timed out
after its captured test body reported success; its immediate isolated rerun
passed in 0.13 seconds. Thus every checked-in test passed at this revision,
with the one transient teardown timeout recorded rather than hidden.

### Stage 5: give evidence and report intent one authority

The shared MOI evidence-model and report-planning group is 5,864 lines, and the
9,595-line core semantic group contains related inventory, admission, and
policy. These responsibilities are easy to overlook because their code is
spread across models, reports, candidates, analysis, and typed headers rather
than one obviously duplicated engine body.

1. Make `ProgramInventory` and stable physical/logical site identity the sole
   source of decoded semantic facts; remove engine-local reconstruction and
   compatibility projections.
2. Let each flavor's explicit evidence policy select observations from that
   inventory and from the synchronization graph produced by Stage 4.
3. Represent selected evidence as typed report intents independent of buffer
   addresses, target encoding, and hook-owned storage.
4. Make one report-layout component lower intents to record kinds, fields,
   capacity, and offsets; keep host interpretation on the same ABI authority.
5. Delete copied coverage calculations, report-shape inventories, string-based
   role projections, and validation-time rediscovery after their consumers
   move.

Migrate Record/Replay intents first because host processing needs a stable
contract, followed by Sampled and InlineShadow. Stage 5 exits when semantic
inventory, flavor evidence, report intent, report layout, and host report
interpretation have distinct owners and no engine privately reconstructs a
shared semantic or ABI fact.

#### Stage 5 exit evidence

Stage 5 is complete. `ProgramInventory` and `SynchronizationInventoryView`
remain the sole decoded program and synchronization authorities, while the
flavor policies publish stable selected observations in
`ConSanObservationPlan`. The new `ConSanEvidenceIntentPlan` is the explicit
boundary between those policies and report sizing. It maps every admitted
probe exactly once to a smaller address-free vocabulary—access, barrier,
atomic, fence, address capture, or sticky marker—and retains stable source and
semantic-site identities plus the evidence element count. `TransformResult`
publishes that canonical plan before the ABI-bearing evidence requirements, so
later stages do not recover policy meaning from report capacities, emitted
patches, or mechanism telemetry.

Record/Replay, Sampled, InlineShadow, and SuperCollider evidence planners now
consume the typed intent plan. A shared accumulator owns access admission and
the common synchronization counts; only InlineShadow joins retained access
identities back to immutable `ProgramInventory` facts to derive its LDS
aperture. The flavor planners add only their protocol-specific capacity rules.
One automatic report planner owns record regions, fields, capacities, offsets,
and canonical revalidation. Lowering resolves that geometry through one
boundary, and host interpretation checks the resulting header against the
same `ConSanMoiReportBufferLayout` ABI authority.

The migration deleted the four engine-private probe-kind switch traversals,
their copied domain and count validation, three engine-specific lowering-side
layout resolvers, a duplicate report-capacity validator, and redundant test
projections. Direct unit tests cover every intent kind and engine, malformed
and cross-engine plans, typed-plan-to-capacity equivalence, deterministic
pipeline publication, and the cross-type result invariants.

Relative to the Stage 4 revision, production changes added 506 and deleted
360 physical lines. The measured implementation is now 94,815 physical lines,
90,419 nonblank lines, and 83,595 comment-excluded code lines in the same 79
files: temporary increases of 146 physical, 121 nonblank, and 60 code lines to
establish and document the retained typed boundary before later deletion.

The full checked-in ConSan gate passed all 5,279 tests at revision
`2dd3e217e9`, including 3,501 device tests and all 593 serialized
physical-gfx950 tests. `ctest -j64` completed in 560.13 seconds wall time with
no failures.

### Stage 6: simplify mutation and composition around typed plans

Mutation, placement, and composition account for 4,094 core lines. Earlier
deletion work removed recursive fault reanalysis, but this area still needs an
explicit destination so it does not remain a parallel pipeline beside the
production stages.

1. Keep mutation selection as typed semantic intent over immutable inventory,
   separate from byte rewriting and from instrumentation-engine evidence.
2. Make composed transforms consume and produce the same authoritative plans,
   operating points, and replacement-image contract as ordinary lowering.
3. Translate identities and offsets across a mutation exactly once; do not
   rebuild semantic state from partially mutated bytes when an existing plan
   remains authoritative.
4. Keep fault kinds and perturbation semantics explicit, but delete mutation-
   specific copies of inventory, placement, target facts, and result state.
5. Migrate one mutation family at a time with focused composition tests and
   the applicable cross-architecture device tests.

Stage 6 exits when mutation is a composable producer/consumer of the production
contracts rather than a second analysis/lowering path, and every remaining
mutation-specific structure represents an actual fault or perturbation
semantic.

### Stage 7: delete residual flavor-private mechanics

Once the shared placement, target-emission, synchronization, evidence/report,
and mutation-composition contracts are real, revisit flavor-owned files in
payoff order.

1. **InlineShadow, 8,856 lines.** Move preservation, placement, reporting, ABI,
   and native emission to the shared contracts. Retain a focused component for
   shadow-state representation, epoch transitions, and InlineShadow evidence.
   Document any requirement the current DBI design cannot yet satisfy rather
   than treating the entire engine as exceptional.
2. **Sampled, 4,427 lines.** Remove private synchronization and access mechanics
   now supplied by shared stages. Retain sampling policy, window selection, and
   sampled evidence semantics.
3. **Record/Replay, 2,013 lines.** Delete proving-ground adapters and transitional
   projections left from Stages 2–5. Its surviving code should describe replay
   evidence and host-processing behavior, not common lowering mechanics.

Each flavor exits this stage with a short statement of its unique responsibility
and no private copy of shared inventory, allocation, target ABI, or emission
mechanics. The goal is not equal flavor sizes; it is that size differences
correspond to real semantic differences.

### Stage 8: transport the production boundaries to SuperCollider

SuperCollider is 6,879 lines and shrank only 3.7% during the deletion-driven
phase. Its 4,048-line LDS body merits a separate responsibility audit, but the
audit should occur after the common target and resource boundaries exist.

1. Reuse shared access classification, inventory, descriptor growth, resource
   planning, and target operations where their contracts match.
2. Keep SuperCollider's race-detection semantics distinct from MOI evidence
   semantics.
3. Compare FLAT and LDS lowering by semantic operation and target requirement,
   not by instruction text alone.
4. Delete SuperCollider-private helpers and product branches only when the
   shared production component expresses the same invariant.

Stage 8 exits when the remaining SuperCollider-specific code can be explained
as sanitizer semantics or unavoidable native lowering, and not historical
duplication of the common transform pipeline.

### Stage 9: narrow the hook boundary without preempting DBI

The hook and reporting layer is 9,044 lines, but DBI is expected eventually to
own more of this integration. Define its deletion boundary early during Stage
0, but avoid a cosmetic internal rewrite while DBI's design is still moving.

1. Make the hook consume the production request/result and report-layout
   contracts without lowerer-private state.
2. Separate ConSan report interpretation from generic HSA tool lifecycle and
   code-object coordination.
3. Document the exact lifecycle, ownership, before/after instrumentation, and
   host-processing operations that DBI must replace.
4. Delete hook code only when an established production or DBI contract owns
   the responsibility; do not create a local imitation of the future DBI hook.

This stage exits when ConSan's hook-facing surface is narrow and the 9,044-line
layer has an explicit keep/delete map tied to the DBI migration timeline.

### Stage 10: simplify validation and perform the final deletion audit

Validation is a beneficiary of the preceding stages, not an isolated size
target. Typed plans should make some combinations unrepresentable and remove
the need to reconstruct planning facts from emitted bytes.

1. Change validation to consume authoritative semantic plans, operating points,
   and target facts.
2. Retain independent checks of emitted code, ABI conformance, bounds, and
   behavioral invariants; delete only mirror checks made redundant by
   construction.
3. Search production code and tests for legacy projections, compatibility
   aliases, duplicate target facts, temporary adapters, and write-only state.
4. Update `DESIGN.md`, `PRODUCTION_DESIGN.md`, this study, and the quantitative
   ledger to describe the implementation that actually remains.
5. Run all checked-in host and device tests, including the physical-gfx950
   gate, and resolve every regression before calling the reimplementation
   complete.

The complete plan exits when no temporary old/new seam remains reachable; each
shared component has one typed authority; architecture-specific code is confined
to justified profile, ABI, and native-emission boundaries; flavor-private code
describes genuine flavor semantics; the hook boundary is ready for DBI; the
documentation matches the code; and the implementation is materially smaller
without weakening the behavioral test contract.

## What this study does not establish

- It does not estimate an achievable final line count. That requires tracing
  duplicate algorithms and state ownership inside the large mixed files.
- It does not equate a target-name reference with target-specific semantics.
- It does not judge correctness value per line; validation, diagnostics, and
  uncommon ABI handling can be both large and necessary.
- It does not prove that two similarly named engine paths should be shared.
  Evidence semantics must remain distinct where the behavioral contracts
  differ.
- It does not count tests or documentation. Those are constraints and assets,
  not implementation shrinkage targets.

Stage 0 resolves these limits by producing the responsibility map, target-
reference classification, state-ownership inventory, opportunity range, and
incremental component cuts that this lexical study cannot infer on its own.
