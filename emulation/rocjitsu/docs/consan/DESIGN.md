# ConSan design

ConSan is a final-ISA concurrency instrumentation system for AMD GPUs. It
intercepts native AMDGPU code objects at the HSA loader boundary, derives a
typed semantic inventory, lowers one selected analysis mode, validates the
complete replacement independently, and binds its runtime evidence to the
loaded executable. It never translates between GPU architectures.

This document describes the code as it exists now. It explains the detection
decisions, their assumptions and failure directions, and the invariants that constrain
implementation changes. Everyday commands are in
[USAGE.md](USAGE.md), detailed controls are in [EXPERT_CONTROLS.md](EXPERT_CONTROLS.md),
mode tradeoffs are in [MODES.md](MODES.md), the
supported semantic forms are in [CAPABILITIES.md](CAPABILITIES.md), and
register/private-memory mechanics are expanded in [SPILLING.md](SPILLING.md).

Read the [detection strategy](#detection-strategy-and-design-tradeoffs) first for
the algorithm and its limits. The [system overview](#system-overview) and
[component boundaries](#component-direction-and-authority) describe how it is
implemented; [reviewing a design change](#reviewing-a-design-change) states the
questions a proposed change should answer. The final section records the
[transform memory model](#transform-memory-accounting).

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

The static transformation contract is transactional:

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

## Detection strategy and design tradeoffs

Final ISA exposes operations and machine state, but has lost much of the
source-level address-space and synchronization intent. ConSan reconstructs
what it can, selects a bounded subset of dynamic observations, and compares
those observations on the host. These are separate decisions. A safe native
rewrite does not imply a complete reconstruction of the program's concurrency.

The default detector is designed around three cost constraints:

- Instrument already allocated machine code without compiler cooperation.
  Liveness, ownership, descriptor growth, and private spilling constrain where
  a probe can be installed.
- Bound device storage independently of dispatch count and loop trip count.
  Static access ranges receive fixed banks instead of an unbounded event log.
- Avoid recording every dynamic access. Workgroup and address selectors reduce
  publication work, while checkpoints bound the lifetime of retained evidence.

Those choices make detection incomplete by construction. They also introduce
model assumptions that can produce a conflict for a program whose actual
synchronization is outside the recognized model. `RJ_CONSAN_POLICY=strict` is an operational failure policy, not a different
detection algorithm or a proof mode.

### Heuristics and their failure directions

| Decision | Implemented rule and purpose | Limitation / failure direction | Observable control or evidence |
| --- | --- | --- | --- |
| Recover group-FLAT provenance | Track address components through supported register, spill, and call forms; default policy admits exact and likely-group forms. | Incorrect likely-group inference can admit a non-LDS access; unresolved or strict-excluded accesses can hide races. | FLAT provenance policy and typed site exclusions. |
| Identify execution | Retain dispatch fingerprint, workgroup coordinates, cluster-workgroup ID, and compact wave owner. | Fingerprint collisions can merge launches; owner aliasing can hide cross-wave conflicts. | Identity operating point and ownership restrictions; no universal collision alarm. |
| Choose dynamic instances | Hash workgroup identity and select a residue of each access's starting four-byte LDS cell. | A conflicting pair may never be selected; overlapping ranges can start in different cells. | Presets, strides, offsets, selected evidence. |
| Retain instances | One owner-aware hash bucket per static range; first publication wins. | Later addresses, lanes, phases, and owners can lose their representative, even with empty banks elsewhere. | Achieved bank count and saturation counters; not every omission is counted. |
| Recognize ordering | Qualified barriers advance a bounded epoch; associated atomic release/acquire metadata can suppress a pair. | Missing ordering can create conservative diagnostics; matching an atomic object does not establish a dynamic reads-from edge. | Sync coverage, metadata validity, and incomplete-evidence counters. |
| Fit a probe | Choose an owner-compatible register/private-state plan and a reachable placement. | Rejection reduces static coverage; accepted growth/spilling can change occupancy and timing. | Resource/placement outcomes, descriptor growth, runtime overhead. |
| Interpret a run | Analyze retained evidence from selected report epochs and assess integrity separately. | A complete selected report can miss races and does not validate every model assumption. | Static/dynamic completeness, diagnostics, saturation, selected epoch policy. |

### Recovering access semantics

Native LDS instructions carry their address-space meaning directly. FLAT
instructions require reconstructed provenance. Analysis follows supported
SGPR/VGPR components, accumulator transfers, static private spills, and call
relays. Its classifications distinguish exact group provenance from
`MaybeGroup`; they are not probabilities calibrated against a workload corpus.

There is also a deliberately narrow naming heuristic: a non-kernel helper whose
name contains `lds_load_at` or `lds_store_at` can have its last matching FLAT
load/store promoted from unknown or maybe-private to likely-group. This encodes
a hip-moi helper convention, not a property guaranteed by arbitrary symbol
names. The default `likely` policy admits it; `strict` provenance excludes it.
Target-specific runtime guards, where implemented, do not make every static
likely-group inference exact.

Semantic admission precedes register and placement decisions. A lowering
failure must remain a coverage failure; the emitter cannot reclassify a site
as irrelevant merely because instrumentation is expensive.

### Selection and retention

A logical range is one statically identified access range. A dual-range
instruction contributes two ranges, each with its own bank table. Selection
and retention proceed as follows:

1. Hash the dispatch fingerprint and workgroup coordinates (including the
   cluster-workgroup ID), without the owner, and compare against the selected
   workgroup residue. A uniform entry gate is used where resources permit;
   otherwise the access body performs this check.
2. For an active lane, shift the effective starting LDS byte address right by
   two and select its residue modulo the cell stride. This selects starting
   cells, not all four-byte cells overlapped by an access.
3. Within the static range's table, hash dispatch/workgroup identity **with the
   wave owner** to choose one bank. Neither address, epoch, nor a dynamic
   sequence number participates in that hash.
4. Claim that bank through the publication protocol and retain a representative
   access. The first successful publication wins. A locally proven uniform
   address can additionally carry the exact participating lane mask; general
   accesses do not acquire exact lane provenance.
5. An occupied bank is never searched around or replaced. A matching window
   identity leaves its original representative unchanged. A different window
   identity increments saturation. The repeated-window comparison checks
   generation, dispatch, workgroup, epoch, record index, and publication state,
   but does not compare the packed access address or owner. Consequently some
   discarded observations do not increase saturation.

For example, two owners can hash to the same bank while another bank is empty.
The second owner's access is not retained independently. Likewise, a loop can
first touch address A and later touch a racing address B at the same static
site: the representative for A can survive while B is invisible. Increasing
the number of banks can reduce collisions between identities, but cannot turn
one owner's repeated accesses into a complete history.

Automatic planning requests eight banks per range by default. An explicit
power-of-two request may select up to 1024. The planner can halve the bank count
to fit the per-buffer ceiling, preserving static ranges and reserved sync
slots. The allocator then allocates the exact resulting layout. Always use the
**achieved** geometry when interpreting a run.

`max` removes workgroup and cell filtering; it does not remove bank collisions,
representative-lane loss, or the first-publication policy. Sampling is
deterministic for a given identity and configuration, not independent random
sampling of accesses. Neither the stride product nor the bank count yields a
justified race-detection probability. Offset sweeps vary selectors but cannot
join evidence from different executions into one conflict.

### Identity and barrier epochs

These identities serve different purposes and must not be conflated:

- The report generation rejects evidence from a different allocation lifetime.
- The normal dispatch value is a 64-bit fingerprint incorporating queue pointer
  and absolute queue-local dispatch ID. It is not an injective global launch
  identifier; collisions and queue-address reuse remain assumptions. The
  literal fallback under scalar pressure provides weaker launch separation.
- Workgroup x/y/z and cluster-workgroup coordinates partition LDS evidence.
- The default wave owner is derived from entry work-item x and wave size.
  Workgroups whose distinct waves differ only in y/z can alias. The expert
  hardware-owner path has its own target restrictions and supports access-only
  checking; it is not a general replacement for barrier/atomic ownership.
- A per-wave barrier epoch partitions accesses across recognized synchronization
  phases **inside a dispatch**.
- A host report epoch ends at a proven quiescent checkpoint and controls report
  recycling across completed dispatches. It is not the barrier epoch.

**Current limitation:** the packed barrier epoch is ten bits. The device
barrier path advances through 1023 and then saturates; later phases share 1023.
It does not currently publish an exhaustion counter or invalidate the trust
verdict. Retained accesses from actually ordered later phases can therefore be
compared as if they belonged to one phase. `analysis_complete=true` does not
exclude this case. Host recycling after a dispatch cannot repair saturation
inside that dispatch. This is an implementation limitation, not an intended
synchronization guarantee.

Barrier recognition is itself bounded: only qualified normalized sequences
with established execution ownership advance the model. Recognizing supported
single or split signal/wait templates does not establish the semantics of an
arbitrary synchronization protocol or make divergent barriers safe.

### Host conflict and ordering model

After decoding stable evidence, the host performs two checks:

1. An exact-lane group for a conflicting ordinary write can diagnose
   same-instruction lane collisions. The explicit same-value-write opt-in
   suppresses only locally proven uniform native LDS stores in this check;
   cross-wave comparisons remain enabled.
2. A pairwise scan compares retained accesses with matching report generation,
   dispatch fingerprint, workgroup/cluster coordinates, and barrier epoch.
   Disjoint complete static execution-owner sets exclude a pair. Otherwise,
   different compact owners, overlapping byte ranges, and conflicting access
   kinds form a candidate. Two reads and two atomic accesses do not conflict
   under the access-kind model.

When synchronization evidence is complete, associated atomic metadata can
suppress a candidate. Both records must be valid atomics at exactly the same
address and width, in the same unchanged epoch, with scopes covering the
workgroup. One must have release and the other acquire role. Failed
compare-exchange cannot supply release, but may supply acquire. The device
publisher is responsible for attaching release metadata to accesses before the
edge and acquire metadata to accesses after it; hashed slot order is not event
order. Incomplete synchronization evidence disables this suppression.

This is a bounded ordering approximation. The records do not retain a general
happens-before graph, vector clocks, or an atomic reads-from relation. Matching
roles and address do not alone show that the acquire observed that release.
Repeated use of one atomic object can therefore suppress a pair without the
full dynamic ordering evidence needed for a proof. Conversely, unrecognized
synchronization can leave an actually ordered pair as a reported conflict.
These are model limitations, separate from malformed evidence and sampling.

The scan is quadratic in retained access count, with a separate linear
exact-lane pass. Diagnostic limits bound stored examples and output, not the
pair scan or conflict count. Sampling, report geometry, and checkpoint policy
therefore affect device work, retained memory, and host work differently.

### What completeness establishes

Static completeness asks whether the selected semantic contract reached
validated code. Dynamic completeness checks usable report evidence: allocation
or cleanup failures, dropped windows, unusable snapshots, unsupported or
malformed synchronization, and missing required records can make it false.
The require-records check counts visible watchpoints, not standalone sync
metadata.

Expected selection misses and bank saturation do not themselves make the
verdict incomplete. Neither do unvalidated assumptions such as fingerprint
uniqueness or the current epoch-saturation limitation. Selective host-epoch
policies intentionally discard unselected epochs; a complete verdict applies
to the selected analysis. A successful application oracle and a complete,
conflict-free report are useful observations, not a race-freedom certificate.

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

## C++ namespaces

Core ConSan types and functions live in `rocjitsu::consan`. Runtime hooks live
in `rocjitsu::consan::hook`, so they can refer directly to enclosing core names.
Implementation helpers use nested `detail` namespaces. Within these namespaces,
identifiers omit redundant ConSan prefixes; SuperCollider-specific identifiers
retain their SuperCollider qualifier. Callers outside these namespaces qualify
references explicitly, without `using namespace` directives. Unqualified C++
type names in this document refer to `rocjitsu::consan` unless otherwise stated.

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
| `rocjitsu_consan_validation` | Reconstruction and independent checks of the final encoded artifact. | Deciding what the selected mode ought to observe. |
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

Policy converts inventory into an `ObservationPlan` through three domain
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

ConSan retains causal windows rather than a general event history; the
[selection and retention algorithm](#selection-and-retention) defines what
those windows omit. Selected
access instances publish immutable watchpoint banks; qualified barrier and
atomic metadata shares their causal identity. The host distinguishes
conflicts, reported saturation, and true evidence loss. It cannot identify an
unobserved race or count all selection and retention misses.
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

The planner considers a checked explicit override, dead registers, fresh
descriptor-backed registers, a supported spill-backed window, or rejection.
Persistent owner/epoch/workgroup state and transient EXEC/VCC/SCC/router state
are reconciled before emission. This is a feasibility search, not a guarantee
of minimum overhead or optimal occupancy. Descriptor growth can reduce resident
waves and private spills add memory traffic; either can change which races
manifest. Private state and spill leases share one
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

These are checks of the encoded replacement and declared effects, not a formal
proof of arbitrary guest semantics or of the detector's concurrency model.
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

## Reviewing a design change

Changes to a heuristic should state the information available at the decision,
the rule being changed, why the cost is acceptable, and which misses or
conservative diagnostics it can add or remove. Preserve exclusions and
achieved resource/retention geometry in the observable result. Do not promote a
heuristic to an invariant merely because existing workloads pass.

Validation should exercise both sides of the decision: an admitted and excluded
FLAT provenance case; selected and unselected addresses; colliding banks with
free capacity elsewhere; repeated accesses at one site; ordered and unordered
atomic uses, including object reuse; owner aliases and dispatch lifetimes; and
barrier counter boundaries. Tests of today's behavior are not evidence that a
known approximation is sound for all programs. Keep physical qualification and
performance results tied to exact binary and configuration provenance as
required by the maintained [validation](validation/VALIDATION.md) and
[benchmark](benchmark/BENCHMARK.md) procedures. The authoritative end-to-end
signals are the most recently rerun campaigns recorded there, interpreted with
their artifacts and provenance.

The key implementation and test entry points for this review are:

| Decision | Implementation under `lib/rocjitsu/src/rocjitsu/` | Tests under `tests/` |
| --- | --- | --- |
| FLAT reconstruction and policy | `code/patch/consan/consan_analysis.inc`, `code/patch/consan/consan_access_policy.cpp` | `patch/consan/analysis_test.cpp`, `patch/consan/access_classifier_test.cpp` |
| Selectors and publication | `code/patch/consan/consan_access_emission.cpp` | `patch/consan/probe_lowering_test.cpp`, device selection fixtures in `dbi/consan/` |
| Bank sizing | `code/patch/consan/consan_report_plan.cpp` | `patch/consan/evidence_requirements_test.cpp`, `patch/consan_report_plan_test.cpp` |
| Barrier epochs | `code/patch/consan/consan_sync.inc`, `code/patch/consan/consan_model.h.inc` | `patch/consan/barrier_policy_test.cpp`, `patch/consan/probe_lowering_test.cpp` |
| Conflict and ordering predicates | `hooks/consan/rj_hsa_dbi_conflict_analysis.cpp`, `hooks/consan/rj_hsa_dbi_sync.h` | `dbt/hsa_hooks_unit_test.cpp`, `dbt/consan_report_test.cpp` |
| Report trust | `hooks/consan/rj_hsa_dbi_report_trust.cpp` | `dbt/hsa_hooks_unit_test.cpp`, `dbt/consan_report_test.cpp` |

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
