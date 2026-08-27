# ConSan reimplementation responsibility map

This document is the Stage 0 deliverable for the incremental refactoring plan
in `IMPLEMENTATION_SIZE_STUDY.md`. It turns the file-level size study into a
responsibility, state-ownership, target-reference, and deletion map. The
snapshot is commit `f727f422dc`, with 83,402 nonblank, non-comment
implementation lines.

The destination is deliberately allowed to differ substantially from the
prototype. The tables below do not preserve current files or types as design
constraints. They identify what the implementation currently does so that
each responsibility can move once, every old representation can be deleted,
and every intermediate revision can remain under the checked-in test gate.

## Decisions established by Stage 0

1. **Do not build another target facade.** `ConSanTargetProfile` already owns
   many immutable target facts. Make it and a per-kernel ABI view authoritative,
   add only facts that are genuinely missing, and route consumers through those
   contracts. Native classification and emission use existing RocJitsu
   decoders/builders plus narrow semantic operations rather than an engine-by-
   target virtual hierarchy.
2. **Replace `ConSanOptions`, do not reorganize it.** Its inherited immutable
   request subobjects remain useful, but its 27 declared lowering fields are
   planning outputs with different owners and lifetimes. Stage 2 moves them to
   an immutable accepted operating point and per-patch plans, then deletes the
   mutable derived object from production lowering.
3. **Separate meaning from mechanics.** Inventory and observation policy decide
   what an event means. Resource planning decides what it needs. Allocation
   selects concrete state. Target emission encodes an already-decided semantic
   operation. Validation checks an authoritative plan and the emitted result;
   none of these stages may rediscover the preceding stage's decision.
4. **Use Record/Replay as the first vertical consumer, not as the design.** Its
   dedicated implementation is only 2,013 lines and DBI needs its host path
   first. New contracts must be engine-neutral and must pay for themselves by
   deleting common, Sampled, and InlineShadow code after the first cutover.
5. **Keep real semantic differences.** InlineShadow shadow/epoch semantics,
   Sampled selection and statistical completeness, Record/Replay host replay,
   and SuperCollider value-instability observation remain separate policies.
   Preservation, placement, ABI, report layout, and native emission do not
   remain private merely because the prototype implemented them there.
6. **Retain focused algorithms as leaves.** The branch-only relay router has a
   coherent input/output and no target vocabulary. Keep it behind a narrower
   placement interface unless a call-by-call audit proves duplicated policy;
   file size alone is not a reason to rewrite it.

## Top-20 responsibility map

These files contain 73.4% of the implementation. “Options touches” counts
source lines that name `ConSanOptions` or an `options.` field; it is a coupling
signal, not a semantic line count. “Target references” uses the exhaustive
lexical classification defined below.

| File | Code lines | Options touches | Target references | Current responsibilities | Destination and deletion point |
| --- | ---: | ---: | ---: | --- | --- |
| `consan_moi_placement.inc` | 6,212 | 705 | 191 | Candidate sizing; descriptor/ABI queries; liveness; scalar/vector/private allocation; dense and branch-only routes; retry/fallback; publication | Stage 2 splits admission, requirements, allocation, route selection, and publication around one `MoiOperatingPoint`; Stage 1 moves target facts; delete mutable-option planning and duplicate descriptor selectors |
| `rj_hsa_dbi_hooks.cpp` | 4,654 | 1 | 0 | HSA lifecycle; reader/executable registry; transform admission; automatic report allocation; retry; dispatch edits; installation and failure policy | Stage 9 narrows request/result/report and backend-lifetime contracts; keep/delete map aligns remaining code with DBI ownership |
| `consan_validation.inc` | 4,411 | 16 | 31 | Structural ELF validation; patch geometry; descriptor/resource consistency; engine-specific emitted-form and coverage checks | Stage 10 consumes authoritative plans and target facts; retain independent byte/ABI checks, delete reconstructed planning and duplicated target rules |
| `consan_supercollider_lds.inc` | 4,048 | 19 | 48 | Native LDS probe construction; liveness/spills; inline/cave/relay placement; delay/reload/compare/report; descriptor growth | Stage 8 retains instability semantics but uses shared inventory, resources, placement transaction, and target operations |
| `consan_moi_sync_common.inc` | 3,370 | 99 | 95 | Shared synchronization candidate preparation; identity/state acquisition; record/shadow mechanics; placement and coverage handoffs | Stage 4 separates synchronization graph, flavor evidence, resource requests, and emission; Stage 1/3 remove product/encoding choices |
| `consan_sync_analysis.inc` | 3,309 | 33 | 46 | Event normalization; sequence/owner construction; barrier/atomic/fence/ordinary ordering; causal association | Stage 4 makes one immutable synchronization graph; Stage 1 moves raw target classification and capability selection |
| `consan_moi_sampled_sync.inc` | 3,182 | 164 | 47 | Sampled barrier/atomic window selection; bank compatibility; pending acquire; resources; emission and coverage | Stages 4/5 retain sampling/evidence policy, move common synchronization, report intent, allocation, and target emission |
| `consan_moi_prologue.inc` | 2,996 | 164 | 62 | Entry capture of owner/epoch/dispatch/workgroup state; descriptor growth; persistent initialization; spills | Stage 3 consumes the accepted operating point and emits typed initialization requests through target ABI operations; delete planning recomputation |
| `consan_moi_inline_shadow.inc` | 2,965 | 185 | 44 | InlineShadow access planning; local/external shadow selection; routing; resources; coverage | Stage 7 retains shadow/evidence policy; Stages 2–5 own admission, allocation, placement, reporting, and target mechanics |
| `consan_moi_inline_shadow_emission.inc` | 2,963 | 145 | 107 | Exact-shadow address/version/token operations; guest preservation; diagnostics; target encodings | Stage 3 moves reusable preservation/ABI/emission; Stage 7 retains focused shadow-state semantic operations |
| `consan_moi_common_emission.inc` | 2,879 | 151 | 43 | Shared identity, spill, record, LDS/private, dispatch, and helper emission; telemetry projection | Stage 3 replaces broad option input with semantic requests plus operating point and target operations; delete engine/target branches and recomputation |
| `consan_moi_model.cpp` | 2,828 | 0 | 17 | Host Record/Replay and Sampled evidence validation, ordering/conflict model, completeness | Stage 5 consumes normalized typed report events; product names disappear from host semantics except versioned ABI decoding |
| `consan_moi_barrier.inc` | 2,738 | 139 | 50 | Barrier evidence planning and lowering for MOI engines; dense/shared routing; epoch updates; report publication | Stage 4 owns semantic barrier intent; Stages 2/3 own resources, placement, and native emission; delete engine-private route/ABI decisions |
| `consan_analysis.inc` | 2,470 | 0 | 147 | Raw instruction-family decoding; access/atomic/barrier/fence classification; normalized inventory construction | Stage 1 splits target classifier implementations from target-neutral inventory assembly; no engine policy remains in raw decoding |
| `consan_fault_injection.inc` | 2,390 | 161 | 53 | Fault-site selection; native instruction mutation; mutation application and telemetry | Stage 6 consumes immutable inventory and typed mutation plans; target-native mutation moves behind classifier/lowerer operations |
| `consan_branch_only_relay_router.cpp` | 2,252 | 0 | 0 | Bounded liveness-safe relay-host feasibility search and route selection | Retain as focused target-neutral leaf; Stage 2 narrows callers to immutable placement inputs; Stage 10 deletes only proven compatibility telemetry |
| `consan_moi_inline_atomic.inc` | 2,190 | 130 | 38 | InlineShadow atomic evidence/token semantics, preservation, address handling, target emission | Stage 4 retains atomic evidence policy; Stage 3 shares preservation/address/native operations; Stage 7 retains shadow-specific token semantics |
| `rj_hsa_dbi_hook_moi_report.cpp` | 1,982 | 0 | 0 | Report lifetime registry; snapshot/validation; replay/sample analysis; completeness and diagnostic rendering | Stages 5/9 separate ABI decoder, engine analyzer, trust evaluator, rendering, and runtime lifetime ownership |
| `consan_supercollider_flat.inc` | 1,865 | 17 | 8 | Group-FLAT eligibility and address reconstruction; probe resources; placement; repeat/compare/report | Stage 8 reuses common access/resource/target boundaries while retaining instability evidence semantics |
| `consan_moi_record_replay.inc` | 1,544 | 68 | 18 | Record/Replay access intent, record reservation/emission, borrowed entry handling, coverage | First consumer in Stages 1–5; Stage 7 deletes proving-ground adapters and leaves only replay evidence policy |

The table makes the main structural problem quantitative: the largest file is
not one component. Its 705 mutable-options touches and 191 target references
cross almost every planned boundary. Splitting that responsibility is more
important than merely splitting the file.

## Exhaustive target-reference ledger

The size study found 1,358 comment-stripped source lines containing `gfx9`,
`gfx11`, `gfx12`, `CDNA`, or `RDNA` vocabulary. Every such line was assigned by
a deterministic priority rule; the categories sum exactly to 1,358. This is a
planning ledger, not a claim that each lexical match is an independent branch.

| Current category | Lines | Required final owner |
| --- | ---: | --- |
| Native ISA classification or emission | 775 | Target classifier, existing RocJitsu instruction builder, or a narrow semantic target operation |
| ABI or resource fact | 327 | `ConSanTargetProfile`, per-kernel ABI facts, or accepted resource/allocation plan |
| Direct product selection to migrate or justify | 162 | One of the named owners above; any survivor requires an explicit justification |
| Canonical target-profile data | 56 | Remains in the audited `kConSanTargetProfiles` table |
| Semantic capability or policy | 22 | Normalized capability/form data; shared policy must not switch on product names |
| Diagnostic or target naming | 16 | Edge diagnostics or stable target display only |
| **Total** | **1,358** | |

The classifier gives target-profile data precedence, then recognizes native
instruction/encoding vocabulary, ABI/resource vocabulary, semantic-policy
vocabulary, and diagnostic naming. Any remaining product reference enters the
162-line migration/justification set. The component distribution is:

| Component | Profile | ISA | ABI/resource | Policy | Diagnostic | Migrate/justify | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared MOI | 0 | 302 | 180 | 13 | 5 | 77 | 577 |
| Shared transform core | 56 | 256 | 23 | 1 | 10 | 55 | 401 |
| InlineShadow | 0 | 142 | 36 | 5 | 1 | 6 | 190 |
| SuperCollider | 0 | 42 | 53 | 1 | 0 | 12 | 108 |
| Sampled | 0 | 29 | 25 | 2 | 0 | 8 | 64 |
| Record/Replay | 0 | 4 | 10 | 0 | 0 | 4 | 18 |
| Runtime hook | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **56** | **775** | **327** | **22** | **16** | **162** | **1,358** |

The 162 unresolved selections are fully accounted for by file and assigned to
a destination boundary:

| Current file | Lines | Destination |
| --- | ---: | --- |
| `consan_sync_analysis.inc` | 23 | Target classifier or normalized capability/ABI fact |
| `consan_moi_placement.inc` | 19 | Target profile, kernel ABI facts, or resource planner |
| `consan_analysis.inc` | 16 | Target classifier |
| `consan_moi_prologue.inc` | 15 | Kernel ABI facts or target emitter |
| `consan_moi_sync_common.inc` | 10 | Semantic capability, resource request, or target emitter |
| `consan_moi_barrier.inc` | 8 | Barrier capability/ABI or target emitter |
| `consan_moi_model.cpp` | 7 | Versioned report ABI; none may remain as host semantic policy |
| `consan_moi_candidates.inc` | 6 | Target classifier or kernel ABI facts |
| `consan_validation.inc` | 5 | Independent target validator using the same target facts |
| `consan_supercollider_common.inc` | 5 | Target profile or emitter |
| `consan_moi_sampled_sync.inc` | 5 | Shared synchronization plan or target emitter |
| `consan_moi_common_emission.inc` | 5 | Target emitter |
| `consan_fault_injection.inc` | 5 | Target-native mutation lowerer |
| `consan_supercollider_lds.inc` | 4 | Target classifier/emitter |
| `consan_moi_record_replay.inc` | 4 | Target profile or emitter; first Stage 1 vertical slice |
| `consan_moi_inline_shadow.inc` | 4 | Target profile/resource plan |
| `consan_moi.cpp` | 4 | Target profile or orchestration input |
| `consan_supercollider_flat.inc` | 3 | Target classifier/emitter |
| `consan_moi_sampled_access.inc` | 3 | Target resource/emit operation |
| `consan_moi_pipeline.inc` | 3 | Target profile or accepted operating point |
| `consan_moi_inline_shadow_emission.inc` | 2 | Target emitter |
| `consan_atomic_fence_policy.cpp` | 2 | Normalized semantic capability/form |
| `consan.cpp` | 2 | Target-neutral policy over normalized facts |
| `consan_access_shape.h` | 1 | Target classifier/profile fact |
| `consan_access_policy.cpp` | 1 | Normalized operand/target-classifier fact |
| **Total** | **162** | |

Stage 1 begins with the 18 Record/Replay-owned references and the shared
profile/ABI/emission operations they require. Later stages reduce both the
1,358 total and the number of non-target-owned files containing it. A decrease
in lexical matches is not sufficient if branches merely move into an
unstructured switch.

### Stage 1 target-boundary foundation

The table above is the Stage 0 snapshot. The first Stage 1 vertical slice has
now migrated every one of the 18 Record/Replay-owned references: four native
ISA references, ten ABI/resource references, and four previously unresolved
product selections. `consan_moi_record_replay.inc` now contains no direct
`gfx9`, `gfx11`, `gfx12`, `CDNA`, or `RDNA` vocabulary.

The migrated selections use existing typed facts rather than a second target
facade:

- target admission resolves one `ConSanTargetProfile` before Record/Replay
  planning begins;
- direct-call and compact-return layout use `direct_call_form`;
- VGPR-bank admission, routing, capture, and restoration use
  `has_selectable_vgpr_bank` plus generic incoming-bank state;
- cluster-use inventory is a generic program fact whose target-specific
  decoder remains in the native classifier; and
- target-native instruction builders still own the actual encodings.

The former local product branches were deleted rather than wrapped. The exact
gfx1250 VGPR-bank scanner remains deliberately target-owned, and product names
in its tests remain validation fixtures. No new target fact or catch-all target
interface was required: the existing profile and instruction builders were
sufficient for this slice.

Focused host coverage passed **160/160** target-profile, inventory,
Record/Replay, cluster, high-bank, Sampled, and InlineShadow checks. The paired
Record/Replay device slice passed **28/28** checks across gfx942, gfx950,
gfx1100, gfx1201, and gfx1250, including gfx1250 cluster, wide-cluster, and
high-bank-address cases. The complete Stage gate passed **5,245/5,245** tests,
including 3,501 checked-in device tests and 593 serialized physical-gfx950
tests, in 568.71 seconds of wall time (3,544.48 seconds aggregate user-plus-
system CPU time).

This completes the foundation required before Stage 2, not the cross-cutting
Stage 1 final exit. The same rule remains active through Stage 8: as each
domain moves, its shared algorithms must consume typed facts or semantic target
operations and its superseded product branches must be deleted immediately.

## `ConSanOptions` state-ownership inventory

`ConSanOptions` inherits six immutable input contracts, then adds 27 mutable
lowering fields. The inherited values should be passed as their actual const
types; the derived object itself has no production destination. All declared
fields are listed below.

| Current field or fields | Current producer and consumers | Lifetime and semantic owner | Destination and deletion stage |
| --- | --- | --- | --- |
| `automatic_moi_persistent_vgprs` | Placement; barrier, InlineShadow, and prologue paths | Attempt-local proof of selected persistent representation | Derive from `MoiPersistentStatePlan`/allocation; delete in Stage 2 |
| `automatic_moi_private_epoch` | Placement; thirteen planning/emission files | Accepted epoch-storage alternative | Typed `MoiEpochStorage` variant in operating point; delete broad flag in Stage 2 |
| `automatic_moi_partial_exec_save_sgprs` | Placement and pipeline retry | Attempt-local partial-owner fallback | Scalar-preservation plan with admitted/excluded owners; delete in Stage 2 |
| `automatic_moi_inline_sgpr_spill` | Placement; Inline/barrier/prologue emitters | Inline scalar-preservation alternative | `MoiInlineScalarPreservationPlan`; delete in Stage 2 |
| `automatic_moi_record_replay_sgpr_spill` | Placement; twelve common and engine emitters | Replay scalar-preservation alternative | Shared replay preservation plan; delete in Stage 2 |
| `moi_record_replay_dense_barrier_router` | MOI orchestration, placement, barrier | Record/Replay barrier-route demand/result | `MoiBarrierRoutePlan`; delete in Stages 2/4 |
| `moi_exec_save_sgprs_persistent` | Placement and prologue | Proof attached to selected scalar allocation | Scalar allocation proof, not a free flag; delete in Stage 2 |
| `moi_dynamic_stack_spill` | MOI orchestration, placement, common emission, prologue | Per-patch/per-owner spill-frame mechanics | `MoiSpillFramePlan`; delete code-object-wide flag in Stages 2/3 |
| `moi_inline_access_present` | Inventory orchestration, placement, barrier, sync | Semantic fact that ordinary Inline accesses were admitted | Derive from observation plan/evidence intents; delete in Stages 2/5 |
| `automatic_moi_owner_sgpr` | Placement; common emission, prologue, sync | Provenance of selected owner scalar | `ConSanMoiTransientSgprAssignment` source; delete flag in Stage 2 |
| `automatic_moi_dispatch_id_sgprs` | Placement and pipeline | Provenance of selected persistent dispatch pair | `MoiDispatchIdentityAllocation` source; delete flag in Stage 2 |
| `automatic_moi_private_dispatch_id` | Placement; Inline access/atomic paths | Selected dispatch-identity storage alternative | Dispatch-identity allocation variant; delete in Stage 2 |
| `moi_inline_indirect_pc_sgpr` | Placement; common/atomic/Sampled emission | One member of Inline/replay scalar ABI | Typed scalar ABI subobject in operating point; delete loose optional in Stage 2 |
| `moi_inline_call_return_sgpr` | Placement and Inline planning | Inline route call-return ABI | Inline route plan; delete loose optional in Stage 2 |
| `moi_inline_dispatch_key_sgpr` | Placement | Inline route key ABI | Inline route plan; delete loose optional in Stage 2 |
| `moi_inline_indirect_scc_sgpr` | Placement and Sampled synchronization | Saved SCC ABI member | Scalar-preservation plan; delete loose optional in Stage 2 |
| `moi_inline_visible_evidence_sgpr` | Placement and prologue | Inline evidence-state ABI member | Inline evidence allocation; delete loose optional in Stage 2 |
| `moi_inline_branch_only_scalar_spill` | Placement; Inline/barrier/prologue | Selected branch-only preservation alternative | `MoiInlineScalarPreservationPlan::BranchOnlySpill`; delete flag in Stage 2 |
| `moi_inline_dynamic_stack_borrowed_sgpr` | Placement; Inline/prologue | Borrowed pair valid only for one preservation variant | Required field of that typed variant; delete loose optional in Stage 2 |
| `moi_record_replay_dispatch_key_sgpr` | Placement; pipeline, barrier, common/Sampled emission | Replay route ABI | `MoiReplayRouterAbi`; delete loose optional in Stage 2 |
| `moi_record_replay_call_return_sgpr` | Placement; pipeline, barrier, common/Sampled emission | Replay route ABI | `MoiReplayRouterAbi`; delete loose optional in Stage 2 |
| `moi_dispatch_id_sgpr` | Placement; candidate, pipeline, prologue and five emitters | Concrete scalar dispatch-identity allocation | `MoiDispatchIdentityAllocation` variant; delete in Stage 2 |
| `moi_dispatch_id_vgpr` | Placement; candidate, prologue, Inline emission | Concrete vector dispatch-identity allocation, partly mirrored in result allocation | Sole field of accepted allocation; delete options copy in Stage 2 |
| `moi_persistent_sgprs` | Placement; thirteen planning/emission files | Persistent scalar owner/epoch/workgroup ABI | Move into authoritative register/persistent-state allocation; delete options copy in Stage 2 |
| `moi_record_replay_workgroup_vgprs` | Placement; candidate, prologue and access emission | Persistent replay workgroup vector tuple | Persistent-state allocation; delete options copy in Stage 2 |
| `moi_record_replay_workgroup_private_offsets` | Placement; barrier, record, Sampled and candidate paths | Persistent replay workgroup private tuple | Persistent-state allocation; delete options copy in Stage 2 |
| `moi_workgroup_key_vgpr` | Placement; candidate, prologue, common/Inline/sync emission | Persistent Inline packed workgroup identity | Persistent-state allocation; delete options copy in Stage 2 |

The dominant coupling is representation choice expressed as independent booleans
plus optional registers. Variants in the operating point make impossible
combinations unrepresentable: private epoch cannot accidentally coexist with
an unrelated persistent tuple, and a branch-only spill cannot omit its borrowed
bootstrap pair.

## Cross-engine responsibility and duplication map

| Responsibility | Current state | Production owner | Migration consequence |
| --- | --- | --- | --- |
| Decoded access and synchronization facts | Mostly shared inventory, with raw target decoding and some engine reconstruction | `ProgramInventory`, target classifiers, synchronization graph | Delete engine-local reconstruction in Stages 1, 4, and 5 |
| Access admission and evidence selection | Typed policy exists, but lowering still filters and sizes candidates while mutating options | Engine policy producing admitted evidence intents | Stage 2 consumes admitted sites; Stage 5 owns evidence intent |
| Register/private/LDS requirements | Rebuilt during placement and in several event/access paths | Pure resource-request derivation | Stage 2 unifies requirements before allocation |
| Concrete allocation and preservation | Partly typed, partly 27 `ConSanOptions` fields, partly per-patch | `MoiOperatingPoint` plus per-patch preservation plans | Stage 2 deletes options state; Stage 3 consumes the result |
| Dense/direct/branch-only placement | Shared access planners exist; synchronization and flavor paths still carry route policy and telemetry | Placement transaction over engine-selected demand | Stages 2/4 share mechanics; preserve engine route demand |
| Prologue and native emission | Common helpers exist but accept broad mutable options and contain target branches | Semantic emission requests plus narrow target operations | Stage 3 deletes broad option input and private sequences |
| Synchronization causality | Core analysis exists, but engine paths re-associate roles/windows during lowering | Immutable synchronization graph | Stage 4 separates causality from evidence policy |
| Report meaning and layout | Typed layouts exist, but engine emitters and host code still reconstruct parts of schema/capacity | Evidence intent, one report-layout authority, typed decoder | Stage 5 removes reconstructions and target product policy from host models |
| Mutation composition | Typed outcomes exist, but mutation lowerer still consumes broad options and native target branches | Typed mutation intent/plan using ordinary inventory and placement | Stage 6 removes the parallel lowering path |
| Validation | Independent checks mixed with reconstruction of planning facts | Validator over authoritative plans, target facts, and final bytes | Stage 10 retains independent safety checks and deletes mirrors |
| Runtime lifecycle | Large HSA coordinator mixes product policy, transform retry, memory lifetime, and evidence rendering | Runtime coordinator, backend adapter, evidence analyzer/renderer | Stage 9 creates a narrow DBI migration boundary |

Deliberate non-unifications remain: engine evidence/trust policy, InlineShadow
state transitions, Sampled selection/statistics, Record/Replay host replay,
SuperCollider instability semantics, and target-native instruction encodings.

## End-state contracts and dependency direction

The destination uses plain documented value types and pure functions where
possible. Names below describe responsibilities; an existing type should be
extended or renamed rather than duplicated when it already has the right
owner.

1. `ConSanRequest`, `TransformPolicy`, `ConSanDebugOverrides`, and
   `MutationRequest` are immutable caller inputs. Production lowering accepts
   them separately; there is no inherited mutable options object.
2. `ConSanTargetProfile` owns immutable code-object facts.
   `ConSanKernelTargetProfile` grows into the validated descriptor/ABI view.
   Target classifiers normalize raw instructions into inventory facts. Narrow
   target operations lower semantic requests using RocJitsu builders.
3. `ProgramInventory` and one immutable synchronization graph own decoded
   program meaning and identity. Engine policy maps those facts to typed
   observation/evidence intents without liveness, registers, or encoders.
4. Resource-request derivation maps admitted intents to abstract requirements.
   `MoiOperatingPoint` is the accepted result of resource planning: concrete
   register allocations, persistent-state representation, private/LDS layout,
   route ABI, preservation variants, descriptor extents, and provenance.
5. Per-patch plans reference the operating point and carry only site-local
   placement and evidence state. They do not copy the code-object allocation.
6. Semantic emission requests combine one intent, its site-local plan, and the
   accepted operating point. Target operations return instruction fragments,
   fixups, and explicit failure; they do not select evidence policy.
7. The placement transaction owns caves, relays, branches, descriptor edits,
   final bytes, and original-to-replacement mapping. Coverage is published only
   for committed patches.
8. Evidence intents lower through one report-layout authority. The host decoder
   returns typed events; engine analyzers and trust evaluation do not inspect
   patch geometry or product names.
9. Mutation is a client of inventory and placement, not another transform
   pipeline. Runtime coordination owns late binding and lifetimes, not semantic
   planning.

The dependency order is therefore:

```text
target metadata -> target/kernel profile -> target classifiers
original image + classifiers -> program inventory + synchronization graph
request + inventory -> engine evidence intents
intents + target/kernel facts -> resource requests -> operating point
intent + operating point -> semantic emission request -> target operations
target fragments -> placement transaction -> validated transform + mapping
report intent -> report layout -> runtime binding -> typed evidence
typed evidence + coverage -> engine analysis/trust -> diagnostics
```

No arrow points back to mutate an earlier stage.

## Incremental cut sequence refined by Stage 0

The stages in `IMPLEMENTATION_SIZE_STUDY.md` remain the governing sequence.
Stage 0 adds the following first cuts and deletion obligations:

1. **Stage 1 foundation:** reuse `ConSanTargetProfile` as the authority for
   already-recorded facts such as ordinary SGPR limits, architecture/encoding
   family, call form, identity source, selectable banks, private granularity,
   and cluster capability. Delete local selectors instead of wrapping them.
2. **Stage 1 Record/Replay vertical slice:** migrate all 18 direct references
   in Record/Replay-owned files and the shared ABI/emission operations they call.
   Add facts or target operations only when the existing profile/builders cannot
   express the semantic request. This is the first proof that the boundary
   works across all five targets.
3. **Stage 2 first operating-point slice:** move dispatch identity, persistent
   scalar/vector/workgroup state, and their provenance from `ConSanOptions` into
   the authoritative allocation. Convert the first Record/Replay access path,
   then Sampled and InlineShadow, deleting each old field after its final
   consumer moves.
4. **Stage 3 first emission slice:** replace broad options input for report
   reservation/claim and workgroup identity with a semantic request and target
   operations. Transport the same operation to Sampled and InlineShadow before
   adding another operation family.
5. **Stage 4 first synchronization slice:** normalize ordinary access/atomic
   ordering in the shared graph, retain explicit Record/Replay evidence policy,
   and lower it through Stages 2/3. Move fences and barriers only after this
   vertical path is complete.
6. Subsequent stages follow their documented dependency order. Every temporary
   adapter names the old callers and is deleted within its stage; no global
   compatibility switch is permitted.

## Code-size opportunity range

The ranges below are engineering estimates derived from the responsibility and
duplication map, not quotas. They overlap: moving a target branch while
consolidating emission cannot be counted twice.

| Area | Gross deletion opportunity | Basis |
| --- | ---: | --- |
| Target selection/classification/emission consolidation | 1,000–2,500 | 1,358 direct-reference lines plus surrounding duplicate selectors/builders |
| MOI placement and mutable operating state | 2,000–4,000 | 7,219-line group, 705 options touches in the largest file, 27 fields |
| Common emission and prologue | 1,500–3,000 | 6,185-line group with broad option input and repeated ABI/preservation mechanics |
| Synchronization analysis/planning/emission | 2,500–5,000 | More than 12,000 related core/shared/Sampled lines with repeated association and lowering |
| Evidence models and report planning | 1,500–3,500 | 5,864 shared-MOI lines plus host/schema reconstruction |
| Mutation and composition | 500–1,500 | 4,094 core lines, with some prior recursive duplication already removed |
| Residual flavor-private mechanics | 3,000–6,000 | 15,296 dedicated MOI flavor lines after semantics are separated from mechanics |
| SuperCollider | 1,000–3,000 | 6,879 lines, especially an independently evolved 4,048-line LDS lowerer |
| Hook/DBI boundary | 2,000–5,000 | 9,044 lines mixing lifecycle, policy, analysis, rendering, and future DBI ownership |
| Validation after typed plans | 500–1,500 | 4,411 lines; only mirror/reconstruction checks are deletable |

The gross sum is 15,500–35,000 lines before overlap. A defensible current net
opportunity is approximately **14,000–28,000 implementation lines**, leaving
roughly 55,000–69,000 lines from the 83,402-line snapshot. This is a floor for
ambition, not a cap: Stage 0 cannot yet quantify simplifications exposed only
after the first operating-point and target-emission cuts. Recompute the range
after every major stage, and pursue a deeper reduction whenever the new
structure reveals that behavior can be expressed with less code.

A twofold reduction to about 41,700 lines is not yet supported by concrete
deletion inventory, but neither is it ruled out. It becomes credible only if
the Stage 2–5 contracts eliminate substantially more engine/target duplication
than file-level analysis can currently prove. Do not preserve code merely to
stay within this estimate.

## Stage 0 exit evidence

The Stage 0 gate was run from `/home/ossci/xx/rocjitsu-build` with:

```sh
ctest -j64 --output-on-failure -R 'ConSan|consan'
```

All **5,245/5,245** tests passed, including 3,501 checked-in device tests and
593 serialized physical-gfx950 tests. CTest reported 9 minutes 38 seconds of
wall time and 57 minutes 8 seconds of aggregate user-plus-system CPU time.

The first exhaustive pass exposed a preexisting disagreement at the new typed
runtime-binding boundary: a small caller-owned fixed MOI report buffer was
being validated as though it were an automatically allocated executable-wide
multi-bank layout. The repair distinguishes the two contracts, accepts both
code-object and executable lifetime for caller-owned fixed buffers, retains
executable lifetime for automatic typed layouts, and adds focused pipeline and
hook tests. Device checks that depended on removed per-site success logging now
assert the bounded aggregate resource proof or typed emitted-patch proof; the
forced-spill InlineShadow case uses the 64 MiB ceiling already required by its
full-aperture exact-shadow ABI. The complete gate above passed after those
repairs.

Stage 0 is complete because:

- every top-20 file has a responsibility, input/output direction, destination,
  and deletion point;
- all 1,358 direct target-reference lines are classified and the 162 unresolved
  selections have a destination owner;
- all 27 `ConSanOptions` fields have producer/consumer, lifetime, destination,
  and deletion-stage ownership;
- repeated and deliberately distinct engine responsibilities are named;
- end-state contracts and the first incremental vertical cuts are explicit;
- the opportunity range is recorded without treating it as a cap; and
- every checked-in ConSan host and device test, including physical gfx950, has
  passed on the Stage 0 tree.
