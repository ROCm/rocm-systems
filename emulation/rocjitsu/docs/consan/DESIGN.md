# ConSan design

ConSan is a final-ISA concurrency instrumentation system for AMD GPUs. It
intercepts native AMDGPU code objects at the HSA loader boundary, derives a
typed semantic inventory, lowers one selected analysis mode, validates the
complete replacement independently, and binds its runtime evidence to the
loaded executable. It never translates between GPU architectures.

This document describes the code as it exists now. It is an ownership map and
an invariant reference, not a development history. Everyday commands are in
[USAGE.md](USAGE.md), detailed controls are in [EXPERT_CONTROLS.md](EXPERT_CONTROLS.md),
mode tradeoffs are in [MODES.md](MODES.md), the
supported semantic forms are in [CAPABILITIES.md](CAPABILITIES.md), and
register/private-memory mechanics are expanded in [SPILLING.md](SPILLING.md).

## C++ namespaces

Core ConSan types and functions live in `rocjitsu::consan`. Runtime hooks live
in `rocjitsu::consan::hook`, so they can refer directly to enclosing core names.
Implementation helpers use nested `detail` namespaces. Within these namespaces,
identifiers omit redundant ConSan prefixes; SuperCollider-specific identifiers
retain their SuperCollider qualifier. Callers outside these namespaces qualify
references explicitly, without `using namespace` directives. Unqualified C++
type names in this document refer to `rocjitsu::consan` unless otherwise stated.

## Scope and guarantees

ConSan supports native code objects for these exact targets:

| Architecture | Admitted target |
| --- | --- |
| CDNA3 | `gfx942` |
| CDNA4 | `gfx950` |
| CDNA5 | `gfx1250` |
| RDNA3 | `gfx1100` |
| RDNA4 | `gfx1201` |

Its access model is focused on LDS/shared memory. Native LDS operations and
admitted group-FLAT operations enter the access model. Selected global and LDS
atomics, cache operations, and fences enter the ordering model when their
address, scope, order, and dynamic-result requirements are known. Ordinary
global-memory accesses are not generally race-checked.

The static transformation guarantee is transactional:

- `ModifiedValid` owns nonempty replacement bytes, at least one real patch,
  coherent semantic/coverage products, and successful independent final
  validation;
- `Unchanged`, `Unsupported`, and `Invalid` own no replacement bytes;
- semantic admission cannot be upgraded by a machine-code lowerer;
- failed resource, placement, binding, or validation cannot publish a
  partially patched image; and
- a transform result contains no runtime conflict evidence and makes no
  race-freedom claim.

Runtime evidence is bounded. A positive diagnostic is evidence under the
selected mode's stated model. A clean report is meaningful only for what was
admitted, instrumented, executed, retained, decoded, and judged trustworthy.
Static coverage, dynamic completeness, overflow or saturation, and diagnostic
presence are independent facts.

## System overview

One code-object load follows this logical path:

```text
HSA reader interception
  -> bounded kernel-allowlist precheck
  -> waitcheck on the original code object
  -> parse typed request and runtime capabilities
  -> analyze immutable program inventory
  -> assemble target-neutral observation plan
  -> plan address-free evidence requirements
  -> allocate and bind runtime report resources when needed
  -> plan resources, placement, descriptors, and native encodings
  -> apply changes to a private image
  -> independently validate the complete replacement
  -> load original or replacement according to typed outcome and policy
  -> bind per-kernel dispatch requirements to loaded symbols
  -> execute and retain bounded evidence
  -> snapshot, decode, analyze, assess trust, and render at retirement
```

Waitcheck and ConSan share the hook but not a result model. Waitcheck examines
the original object first and reports missing waits or analysis failures;
ConSan then continues so a suspect kernel can still be instrumented.

The automatic-report path is two-phase. The library first publishes an
address-free, mode-specific evidence requirement. The HSA adapter allocates
that exact layout, producing `BoundRuntimeResources`, then returns the opaque
`DeferredBinding` to the library. Only the library chooses whether resume
relowers from the input or retries retained ConSan inventory. A caller cannot
substitute a different inventory, request, or input between phases.

## Component direction and authority

The compiled source graph in `code/patch/consan/CMakeLists.txt` is exhaustive
and exclusive: every production `.cpp` belongs to exactly one component.
Dependencies flow in one direction:

```text
contracts -> targets -> analysis -> transform -> validation -> orchestration
```

| Component | Owns | Must not own |
| --- | --- | --- |
| `rocjitsu_consan_contracts` | Value types, identities, semantic classifiers, pure policy, inventory metadata, normalized request/result vocabulary. | Target decoding/emission, mutation, report allocation, loader state. |
| `rocjitsu_consan_targets` | Exact-target profiles and target-native decode, validation, fault, LDS, ABI, and SuperCollider operations. | Mode selection, per-kernel mutable state, report policy, orchestration. |
| `rocjitsu_consan_analysis` | ELF/program inventory, CFG ownership, synchronization analysis, and fault-site selection. | Native patch emission or runtime binding. |
| `rocjitsu_consan_transform` | Mode planning, evidence sizing, resources, placement, descriptor growth, relocation, mutation, and native emission. | HSA object lifetimes or loader policy. |
| `rocjitsu_consan_validation` | Reconstruction and independent proof of the final encoded artifact. | Deciding what the selected mode ought to observe. |
| `rocjitsu_consan_orchestration` | Public transform entry points, composition, publication, and typed transaction outcome. | Reimplementing analysis, targets, modes, or validation. |

The production CMake graph assigns implementation sources to components.
Component APIs, physical mode and target ownership, and behavioral tests carry
the boundary contract.

The HSA adapter is outside this static graph. Files under `hooks/consan/` own
environment parsing, runtime capability queries, HSA reader/executable/symbol
lifetimes, report allocation and retirement, dispatch adjustment, and
human-readable reporting. They consume the typed result; they do not infer
semantic success from byte vectors or patch names.

## Identity and immutable inventory

ConSan uses distinct identities because one instruction may have several
semantic roles:

- `CodeObjectId` identifies the exact pristine image;
- `ProgramContainerId` names a kernel or executable container;
- `ProgramSiteId` names the authoritative decoded physical record;
- `PhysicalSiteId` is its stable pristine instruction location; and
- `SemanticSiteId` names one access range or synchronization facet.

A dual-range LDS instruction is one physical program site with two semantic
access ranges. An instruction that is both an access and an ordering operation
also remains one program-site record. Parallel decoder products therefore
cannot drift or force later offset-based joins.

`ProgramInventory` is the immutable analysis product. It owns the exact target
and architecture, containers and descriptors, ordered program-site arena,
normalized access ranges, execution-owner sets, CFG and liveness-derived
facts, synchronization events and sequences, and typed inventory exclusions.

`ProgramInventoryBuilder` is the only structural mutation interface during
analysis. Derived synchronization products are invalidated when source facts
change and rebuilt before publication. Later policy and lowering retain typed
handles into the inventory rather than copying partial site schemas.

Analysis is target-aware but mode-neutral. Target providers decode exact ISA
forms into normalized access, barrier, atomic, fence, ownership, and resource
facts. Analysis does not decide whether ConSan or SuperCollider should instrument
an access, or whether a probe can afford its registers.

## Semantic policy, intents, and coverage

Policy converts inventory into a `ObservationPlan` through three domain
functions:

- `plan_access_observation` decides LDS/group-FLAT applicability;
- `plan_barrier_observation` qualifies synchronization sequences; and
- `plan_atomic_fence_observation` qualifies atomic/fence associations.

Each semantic site receives one stable decision: `NotApplicable`,
`Unsupported`, or `Admitted`. Decisions carry typed reasons and public
capability dispositions. Unsupported is not silently converted into
not-applicable, and lowering cannot revise policy to make coverage complete.

Admitted decisions point forward to `ProbeIntent` values. An intent says
what must be observed, at which pristine instruction, before or after the guest
operation, under which active-lane mask, and with which dynamic-result or
atomic-address requirement. It can coalesce multiple semantic ranges or events
at one insertion point. It contains no registers, report address, native words,
branch route, or patch kind.

The `CoverageLedger` owns the observation plan and the final lowering
outcome of every admitted intent: `Instrumented`, `ResourceRejected`, or
`PlacementRejected`. It also owns the forward runtime mapping from original
intent/site identity to validated emitted locations and report slots. Runtime
attribution is never reconstructed by matching patch offsets after the fact.

The aggregate identities are:

```text
discovered = supported + unsupported
supported  = selected + expert_limit_omitted
selected   = instrumented + resource_rejected + placement_rejected
```

Static completeness means the selected semantic contract reached validated
machine code. It says nothing by itself about dispatch, report retention, or a
race.

## Target boundary

`TargetProfile` is the one exact-target row for immutable target-wide
facts: architecture, accumulator and scalar placement, dispatch/workgroup
identity, direct-call and cache-refresh forms, transport and translation,
native LDS dialect, resident-wave identity, atomic
address support, and normalized memory operations.

Profiles contain no mode choice, per-kernel allocation, report address, or
mutable lowering result. Exact-target providers own target-specific decoding,
validation, fault mutation, LDS operations, SuperCollider mechanics, and
special state such as CDNA5 selectable VGPR banks. Mode emitters consume a
normalized `TargetProfile` and common architecture-aware instruction
builders; they may account for a target capability but do not select behavior
by concrete product ID.

Exact-target and target-family providers live under `targets/rdna3`,
`targets/rdna4`, `targets/cdna3`, `targets/cdna4`, or `targets/cdna5`. A
provider shared by a real subset lives under `targets/shared` and names that
subset. Common code can consume normalized architectural facts or generic
instruction builders, but concrete target dispatch remains under `targets`.
Exact product IDs remain in profile filenames—such as
`consan_gfx1250_target_profile.h.inc`—because a profile describes one admitted
target, not all CDNA5 products.

The boundary follows two rules:

1. A mode consumes normalized target facts and common native builders and
   never switches on a concrete target ID.
2. A target provider supplies architecture mechanics and never consumes
   `Options`, selects a mode, or defines report policy.

Adding a target means adding an exact profile and missing target-operation
providers, registering them with target dispatch, extending the generated
capability contract, and qualifying the same modes. A hypothetical-target test
registers normalized facts without editing a concrete mode and guards this
extension shape.

Target parity is semantic, not mnemonic parity. CDNA `ds_read`/`ds_write`,
RDNA `ds_load`/`ds_store`, and CDNA5 VDS spellings can implement the same
normalized LDS form. A target-only native form can remain a typed extension
without forcing other architectures to imitate it.

## Mode boundary

Default detection and shared infrastructure live in `code/patch/consan/`.
The `supercollider/` subdirectory contains SuperCollider's exclusive lowering.
The default detector collects access, owner, epoch, and synchronization
information for bounded causal evidence and host conflict analysis.
SuperCollider perturbs accesses and checks value stability.

The pipeline calls planning and lowering directly. Semantic admission remains
in the access, barrier, and atomic/fence policy functions and precedes resource
decisions. Resource planning, register allocation, report planning, and probe
lowering have separate modules named for those responsibilities.

The hook owns evidence decoding, conflict analysis, and rendering in
`hooks/consan/`. Snapshot and trust code manage report lifetime and evidence
integrity separately from conflict analysis. SuperCollider uses its own
mismatch marker rather than a causal-evidence report.

Static access mappings are a vector of intent-bound access records, not a
mode-tagged container. Report decoding borrows optional access metadata through
a nullable pointer, and the pipeline directly invokes conflict analysis.
Object planning holds the report geometry and atomic patch reservation needed
by emission; resource solving does not carry a separate mode-semantics object.
The report ABI, sizing, and host-model contracts are exposed through
`consan_report.h`. One renderer produces evidence, the summary, and bounded
details in that order; it has no separate mode-rendering adapter.

## Execution models

### SuperCollider

SuperCollider preserves an LDS/group-FLAT access, delays, repeats a load or
reads back a store, compares values, and records a mismatch. Ordinary evidence
is a sticky non-trapping marker; expert trap mode is process-disrupting.

This is value-instability detection, not happens-before analysis. A changed
observation does not identify a racing peer; a legal interleaving can change a
value, and same-value races can be invisible. Barrier/atomic perturbation may
compose for validation but does not change those semantics.

### ConSan

ConSan retains causal windows rather than a general event history. Selected
access instances publish immutable watchpoint banks; qualified barrier and
atomic metadata shares their causal identity. The host distinguishes
conflicts, statistical misses, saturation, and true evidence loss.
Barrier-only execution owners still retain a barrier probe and advance their
persistent sampled epoch. They publish no causal-window metadata because they
have no selected LDS window; missing mappings for selected accesses remain
lowering failures.

The default runtime stride is 256. When entry-captured identity and scalar
resources allow it, ConSan places a uniform dispatch/workgroup gate before the
shared access body. Owners requiring private workgroup identity or compact
scalar spilling use the correct in-body fallback. Selected workgroups still
apply a per-cell selector, and all eligible static sites stay represented.
Sampling reduces dynamic evidence work; it does not reduce analysis, patching,
or report planning in proportion to stride, and no proportional-overhead
guarantee is made.

ConSan owns a literal dispatch-identity fallback for scalar-pressure operating
points. That fallback is intentionally weaker than the hardware launch
fingerprint and cannot establish exact separation across concurrent launches;
common placement and runtime trust preserve rather than conceal that mode
choice.

## Evidence planning and runtime binding

Semantic planning produces one `EvidenceRequirements` variant:
SuperCollider marker or ConSan report. Requirements are address-free and include exact capacity, regions,
alignment, initialization, and runtime capabilities. Overflow and ceiling
checks precede allocation.

For automatic evidence, `prepare_automatic_transform` returns either a
complete `TransformResult` or `DeferredBinding`. The hook allocates the
requested bytes under mode and process ceilings, initializes them, and calls
`resume_automatic_transform`. Allocation failure goes through
`cancel_automatic_transform`; it never silently shrinks the layout.

Caller-owned buffers use the same binding validation. Report addresses are
absent from inventory and policy and appear only at runtime binding.

Each successful replacement publishes `DispatchRequirements` by kernel
name: absolute private and group minima, a dynamic private-frame addend, and
whether an instrumented probe can execute through that kernel. The hook binds
names to loaded symbols. Before AQL dispatch it combines descriptor minima
with launch-selected stack and site-local frame demand. It does not rescan
patch kinds to rediscover these facts.

## Resources, placement, and mutation

Resource planning is owner-aware. A direct site uses its kernel's liveness and
descriptor. Shared helper text uses union live-before state and one assignment
valid for every reachable owner. Unknown ownership is a typed rejection.

The planner chooses a safe explicit override, dead registers, fresh
descriptor-backed registers, a supported spill-backed window, or rejection.
Persistent owner/epoch/workgroup state and transient EXEC/VCC/SCC/router state
are reconciled before emission. Private state and spill leases share one
owner-compatible layout; dynamic-stack recipes remain relative to the launch
frame.

Placement resolves a complete native transaction: inline rewrite, existing
cave, appended text, dense router, relay/island, or branch-only route. Branch
reach, relocated guest prefixes, instruction mapping, descriptor growth, and
resource effects remain proof inputs. An emitter consumes the resolved plan; it
does not rediscover policy or choose new registers.

Descriptor mutation has one owner. Kernel requirements are joined before
bytes change, and shared helpers grow every execution owner consistently. Text
and descriptors commit to a private image; the pristine input remains intact.

Fault injection and SuperCollider perturbation are validation-only and enter
through `transform_with_mutation`, not ordinary `transform`. A
live fault is selected against pristine inventory, applied and independently
validated, then the mutated image is reinventoried and instrumented. Provenance
and descriptor identities translate across revisions. If the composition
cannot be proved, the entire transaction rolls back.

## Final validation and installation

Final validation reparses the candidate and checks pristine identity and ELF
structure; instruction boundaries and relocated guest code; branches, caves,
relays, islands, and continuations; ownership and descriptor growth; private,
group, and dynamic-frame requirements; target-native ABI effects; and the
relationships among intents, lowering outcomes, patches, and runtime mappings.

Only a coherent image becomes `ModifiedValid`. `TransformResult::well_formed`
checks result-wide invariants, and `install_action(fail_closed)` derives loader
policy. A runtime failure can demote a result through `discard_replacement` but
cannot repair fields individually.

Fail-open loads the original on unsupported or invalid transformation. Strict
or explicit fail-closed policy rejects those outcomes. `Unchanged` is a valid
load-original result; separate require-patch policy decides whether an
applicable no-patch result is acceptable. Failed final validation is never
installed under either policy.

## Runtime report lifecycle

The adapter intercepts reader creation, executable loading/destruction,
symbol lookup, queue creation, and AQL writes. It registers automatic evidence
by reader and generation, binds it to a loaded executable, adjusts matching
dispatches, then retires evidence at executable destruction or final teardown.

Retirement runs:

```text
generation-consistent snapshot
  -> validated header and evidence decoding
  -> conflict analysis
  -> independent trust assessment
  -> bounded renderer
```

The adapter automatically checkpoints when intercepted waits show that every
tracked instrumented dispatch is complete. A dispatch's own completion signal
is preferred; an ordered barrier completion signal safely proxies for
preceding signal-less dispatches on the same queue. Submission and recycling
share a gate, so a new report writer cannot appear between the quiescence check
and reset.

The default `every` epoch-analysis policy runs the full snapshot, decode,
analysis, and render pipeline at each checkpoint. `nth`, `periodic`, and
explicit manual-window policies can select fewer epochs. A selected checkpoint
first captures and validates every live report, analyzes every captured epoch,
and only then reinitializes all allocations. An unselected checkpoint validates
every live header and layout, then resets the allocations without copying or
interpreting their evidence. Both paths are transactional: failure leaves all
reports and the logical epoch number unchanged and retryable. Accumulated host
summaries retain verdict evidence only from selected epochs, while the device
stores only the current epoch. The default therefore preserves complete
per-iteration analysis; selective policies deliberately trade dynamic coverage
for bounded host-analysis cost.

An explicit checkpoint API is retained for custom runtimes whose completion or
synchronization cannot be observed safely. A separate begin/end window API
lets a harness select semantic operations under the manual policy without
embedding iteration knowledge in the hook.

Work separated by a device-wide synchronization cannot race across the
checkpoint boundary. Reset clears ConSan windows and publication/ordering
state for the next epoch while keeping allocation identity and embedded
generation stable. SuperCollider's lifetime-sticky mismatch marker is not
reset by ConSan checkpoints.

Coarse-grained report memory is copied through the appropriate snapshot path.
A decoder can recover records while also observing overflow, changed
generation, malformed publication, missing dispatch, or saturation. Trust
decides whether absence of conflict is meaningful; renderers preserve it.

Automatic reports and retained replacement images have process byte budgets.
Executable destruction releases ownership. Hook unload reports current and
peak state but does not pretend live executable ownership vanished; final
process teardown handles remaining quiescent reports.

## Source map

| Concern | Primary location under `lib/rocjitsu/src/rocjitsu/` |
| --- | --- |
| Public static pipeline | `code/patch/consan/consan_pipeline.{h,cpp}` |
| Request/result contracts | `code/patch/consan/consan_request_contract.h.inc`, `consan_result.h.inc`, `consan_options.h.inc` |
| Inventory and identities | `code/patch/consan/consan_program_inventory.h.inc`, `consan_site_identity.h.inc`, `consan_program_analysis.cpp` |
| Semantic policy and coverage | `code/patch/consan/consan_*_policy.cpp`, `consan_observation_plan.h.inc` |
| Target profiles/providers | `code/patch/consan/targets/` |
| ConSan planning/common lowering | `consan_lowering_plan.h`, `consan_resource_planning.cpp`, `consan_shared_lowering.cpp` |
| Probe and SuperCollider lowering | `code/patch/consan/consan_probe_lowering.cpp`, `code/patch/consan/supercollider/` |
| Resources and placement | `consan_resource.*`, `consan_probe_planning.cpp`, `consan_register_allocation.cpp`, `consan_placement.cpp` |
| Descriptor/text transaction | `consan_descriptor_growth.cpp`, `consan_text_relocation.cpp` |
| Final validation | `consan_final_validation.cpp`, `consan_validation_inventory.cpp`, `targets/consan_validation_*` |
| HSA integration | `hooks/consan/rj_hsa_dbi_hooks.cpp`, `rj_hsa_dbi_hook_config.cpp` |
| Runtime reports | `hooks/consan/rj_hsa_dbi_report_*`, `rj_hsa_dbi_evidence_decoder.*`, `rj_hsa_dbi_conflict_*` |

The compiled component, physical mode/target locality, typed inputs, and
behavioral tests—not file size or an `.inc` suffix—define authority.

## Non-negotiable invariants

1. One authoritative program-site arena and stable typed handles.
2. Inventory is target-normalized and mode-neutral.
3. Semantic policy precedes resources, placement, and emission.
4. Probe intents contain semantics, never native mechanism.
5. Coverage flows forward from intents; there is no patch-to-policy join.
6. Target providers contain architecture mechanics and no mode policy.
7. Mode directories contain mode semantics and name no concrete target.
8. Shared code operates on normalized contracts, not every mode-target pair.
9. Descriptor, text, private-state, and dispatch effects form one transaction.
10. Automatic planning publishes exact address-free requirements before
    runtime allocation.
11. Only an independently validated full replacement is installable.
12. Runtime report trust is independent of static coverage and diagnostics.
13. Unsupported, incomplete, saturated, overflowed, malformed, and clean are
    distinct observable states.
14. A timeout, signal, wrong program result, or GPU reset is not a ConSan race
    diagnostic.

These invariants are more stable than any encoding or placement strategy. They
decide whether target support, a new mode, or shared optimization belongs at
the intended layer.

## Transform memory accounting

The [process memory controls](EXPERT_CONTROLS.md#shared-controls) bound
concurrent transformation storage, retained replacement-image bytes, and
retained replacement-image growth independently. The concurrent-transform control is
acquired before semantic inventory. It is a conservative admission unit for
major ELF and parser storage, not a strict RSS limit: allocator bookkeeping,
other non-image analysis state, and unrelated process memory are outside the
model.

Let `I` be the original input size and `M` be `I` plus the configured per-object
maximum growth. The modeled phases are `I + 12*M` for an ordinary incremental
patch, `I + 13*M` while independently validated composite mutation storage
remains live, and `9*I + 10*M` during final validation.

Each parser has eight major-image units: its image, up to two units for section
objects and their vector slots, bounded payload, bounded section names, and up
to three units for compact section classification plus symbol- and
kernel-metadata-derived state.

That final budget charges each newly retained role for its copied name plus
conservative aggregate kernel/function record and container state; roles sharing
one logical symbol name share overlapping transient-state charges. It also
charges retained kernel metadata map entries before insertion and vector
capacity beyond the requested entry count. Section classification is one bit per
section and keeps symbol lookup linear in the ELF section and symbol counts.
Classification, symbol, and metadata state share that three-unit budget rather
than receiving separate allowances, so metadata can reject an object already at
the symbol boundary. This tightens parser admission without changing the
eight-unit parser coefficient or the `12*M`, `13*M`, and `9*I + 10*M` phase
coefficients.

Section headers, transient symbol-name characters, and metadata names are views
into the image rather than duplicate owning collections. Repeated kernel names
within one AMDGPU metadata note retain the final record, while an earlier note
keeps precedence over later notes, matching the parser's existing merge rules.
Metadata note walking separately charges both planned payload passes against
four image-sized units of work, so overlapping or repeated program-header
references cannot multiply parser time without limit. Reserved-range ELF symbol
section indices do not resolve against oversized directly encoded section
tables.

Admission uses the largest phase value and reports the governing phase and
coefficients. The parser rejects aggregate copied section payload or
section-name bytes larger than its backing image, and conservatively charged
symbol- and metadata-derived state larger than its three-unit budget.

Deployments with a tuned concurrent-transform ceiling should derive it from the
reported phase and current coefficients. The patcher preallocates every file
insertion, commits same-size rewrites directly, moves every emitted image, and
avoids a separate padding buffer so vector growth cannot add an unmodelled
geometric full-image allocation.

The two retained-image controls are charged together after transformation, when
the exact replacement size is known: one counts the full image and the other
counts only its growth delta. Admission and ownership are one transaction, so
failure of either retained budget commits neither charge. Failed
replacement-reader creation or loading releases every local storage owner before
refunding the retained charge or invoking a fallback loader.

Unload starts a new peak-reporting interval without releasing live transform
charges or retained replacement ownership; the latter remains until an
executable destruction observed while the hook is active, or process exit. New
HSA API calls made after the tool's `OnUnload` callback and before a synthetic
reinstall are outside the hook lifetime and cannot be reconciled on reload.
Teardown reports the live and peak values for all three controls, including
baseline runs where the ceilings are unlimited.

The absolute and percentage growth variables are mutually exclusive. A
growth-policy rejection reports the exact alignment-inclusive bytes required,
the effective total limit, and the selected policy. Successive ConSan stages
share that one original-image budget rather than receiving independent limits.
