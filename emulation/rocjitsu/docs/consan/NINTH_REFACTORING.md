# ConSan ninth refactoring: unify program identity and delete representation boundaries

## 1. Status and baseline

The eighth refactoring and its follow-on whole-text relocation work consumed
the clearest remaining duplicated backends. In particular, ConSan retired
branch-only routing, incremental MOI routing, a local island allocator, and
substantial mode-owned application, prologue, barrier, placement, and
publication orchestration.

The ninth-refactoring baseline is commit `fd99cd218fe` at **67,505 governing
production implementation lines** in 316 `*.cpp`, `*.h`, and `*.inc` files.
The governing scope is:

- `lib/rocjitsu/src/rocjitsu/code/patch/consan/` and
  `lib/rocjitsu/src/rocjitsu/hooks/consan/`;
- production implementation files only; and
- tests, generated code, build files, documentation, comments, and blank
  lines excluded.

For reference, those files contain 79,693 physical lines and 73,988 nonblank
lines. Since the end-of-eighth baseline `668674e6874`, the production tree
has changed by 2,128 additions and 14,323 deletions: a physical net reduction
of 12,195 lines and a governing reduction of approximately **10,844 lines**.
The whole-MOI relocation backend identified as future work in the eighth
refactoring has therefore already been implemented; it is not a ninth-
refactoring candidate.

The current governing implementation is distributed approximately as follows:

| Area | Lines | Share |
| --- | ---: | ---: |
| Common patch implementation | 38,314 | 56.8% |
| Mode patch implementation | 17,473 | 25.9% |
| Target implementation | 2,197 | 3.3% |
| Host hooks, including mode-local hooks | 9,521 | 14.1% |
| **Total** | **67,505** | **100%** |

The mode patch portion comprises 6,536 InlineShadow lines, 4,932 Sampled
lines, 3,835 Record/Replay lines, and 2,170 SuperCollider lines. Target-local
code is now too small for architecture-only consolidation to produce the next
multi-thousand-line reduction. The remaining leverage lies mainly in common
program representation and in the boundary between shared machinery and mode
emitters.

## 2. Assessment: large opportunities remain, but they are deeper

ConSan is **not out of credible large-scale refactoring opportunities**. It is
out of obvious parallel backends that can be consolidated without changing
the structures on which they operate.

The strongest remaining opportunity is to replace the parallel
representations of decoded program sites, program containers,
synchronization events, and synchronization sequences with one stable-ID-
based program-site arena and phase-specific side tables. The present code
copies the same facts between several representations and later joins them
back together by offsets and owner metadata. This is an evidenced
representation boundary, not a resemblance inferred from type names.

Two higher-risk opportunities may follow:

1. A single typed, source-coordinate edit transaction spanning validation
   mutations, fault injection, perturbation, and instrumentation.
2. A compact typed probe-program IR that lifts the low-level instruction-
   building mechanism still repeated across otherwise distinct mode
   algorithms.

For prioritization only, the program-site and synchronization lift appears
capable of contributing to another **3,000–6,000 net governing lines** of
overall shrinkage. An aggressive upside case of **6,000–10,000 lines** remains
possible if the edit transaction and probe-program IR also prove themselves.
These ranges overlap, are not additive commitments, and are expressly **not
completion targets**. If every macro prototype fails its deletion test, the
present tree still appears to contain an aggregate **1,500–3,000 lines** of
smaller cleanup, but that route alone would return the project to a shallow
slope. Work ends because the opportunity space has been exhausted by deep
investigation, not because any line count has been reached.

## 3. Ninth-refactoring mandate

The ninth refactoring will seek shrinkage by deleting representation
boundaries and the joins between them:

> Establish stable identity and shared storage for the program facts that
> genuinely have one lifetime. Make later phases store only the facts they
> derive. Delete the copied representations, projections, reverse joins, and
> parallel traversals displaced by that model.

This continues the eighth refactoring's permission to replace locally exact,
bottom-up structures with deliberately smaller heuristic structures. A
shared structure need not be maximally tailored to every current consumer.
Some bounded representational slack is acceptable when it replaces several
independently evolved forms and makes identity, ownership, and phase
boundaries explicit.

The controlling rule is nevertheless stricter than “unify similar structs”:

> Change a data structure only when doing so deletes a representation
> boundary and its joins.

Common field names alone do not justify a merger. Adding a new canonical
model while retaining adapters to all old representations is not progress.
The currency of this refactoring is deleted storage, deleted projection
passes, deleted reconciliation logic, and deleted end-to-end implementation.

The plan deliberately does not prescribe the final class hierarchy or exact
type names. The destination design remains part of the variable being solved
for. It does prescribe one-way movement toward stable identity, fewer
representations, narrower phase-owned facts, and deletion-backed results.

## 4. Primary campaign: one program-site arena and synchronization graph

### 4.1 Current representation chain

The same decoded program facts currently appear in a chain of partially
overlapping representations:

1. Container-owned arrays of `ConSanOrdinaryMemorySite`, `ConSanBarrierSite`,
   `ConSanFenceSite`, and `ConSanAtomicSite`.
2. Structurally repeated arrays in both `ConSanKernelInfo` and
   `ConSanFunctionInfo`.
3. A normalized `ConSanAccessInventorySite` projection for access-oriented
   consumers.
4. `ConSanSyncEvent`, which copies a broad union of instruction, ownership,
   location, scope, barrier, atomic, and confidence fields.
5. `ConSanSyncSequence`, which repeats much of the event payload while adding
   sequence-level meaning.
6. Specialized projections such as barrier lifecycle groups and MOI fence
   candidates.
7. Observation intent, candidate, resource plan, staged text fragment, patch,
   and committed-lowering records, each with further identity translations.

`append_sync_events_for_container` visibly performs field-by-field copying
from all four raw site vectors into `ConSanSyncEvent`. Event semantic equality
then compares a large copied tuple. Later phases search back through raw
container vectors by text offset to recover facts that existed before the
projection. Kernel and function containers cause paired traversals and lookup
paths even when consumers need the same container identity, extent,
ownership, and site access.

Intent IDs, owner-descriptor offsets, anchor offsets, semantic-site IDs,
physical IDs, candidate indices, and runtime mappings are propagated across
dozens of files. Some identities are legitimately distinct. The architectural
problem is that foundational storage supplies no stable identity that later
phases can retain, so phases repeatedly manufacture and reconcile substitutes.

### 4.2 Directional model

The implementation should converge on an immutable decoded-program arena
with stable identities. One plausible shape is:

- a `ProgramContainer` record holding common kernel/function identity,
  extents, ownership, and a tagged or optional kernel-only descriptor
  payload;
- a `ProgramSite` common header holding stable `SiteId`, container ID,
  physical identity, file/text location, instruction size, decode identity,
  and owners;
- a tagged payload for irreducible Access, Barrier, Fence, or Atomic operands
  and facts;
- synchronization events that reference `SiteId` and store only facts
  derived by synchronization analysis;
- synchronization sequences that reference event IDs and store only
  sequence-derived facts and edges; and
- phase side tables keyed by stable IDs for policy decisions, evidence,
  resource assignment, placement, and committed state.

This is a storage and identity model, not one universal semantic object.
Phase-local decisions remain phase-local. The final validator must continue
to rederive claims independently from original and final bytes; stable IDs do
not authorize it to trust a producer-side verdict.

### 4.3 Expected deletion

The payoff comes from removing categories of code rather than shortening a
few declarations:

- duplicate kernel/function container storage and traversal;
- raw-to-access and raw-to-sync projections that derive no new fact;
- field-by-field event and sequence copying;
- large equality tuples over copied payload;
- text-offset reverse joins into original site vectors;
- repeated owner, container, and location propagation; and
- ad hoc identity conversion and lookup helpers.

The affected envelope includes program inventory, general analysis,
synchronization analysis, policy/evidence planning, and portions of pipeline
and validation lookup plumbing. The credible net opportunity is **2,000–
4,000 governing lines**, with medium-high implementation risk. Unlike the
similar-looking structures rejected during the eighth refactoring, this
candidate has direct evidence of copy-and-rejoin behavior.

### 4.4 Hard prototype and acceptance test

The first prototype must cover difficult ownership and semantic cases:

- a helper function reachable or aliased by multiple kernels;
- an atomic site that also participates in ordinary-memory reasoning;
- barrier lifecycle construction;
- preapplied code ranges; and
- all five supported architecture families/targets through emulation.

The prototype must demonstrate a path to removing the old container site
vectors, copied synchronization payload, offset-based reverse joins, and
semantically duplicated kernel/function loops. If it merely adds an arena
beside those representations, it fails.

The full lift succeeds when it actually removes the old representations and
their joins while leaving a clearer, smaller system. Intermediate commits may
temporarily grow when they name the old representation they enable a near-
term commit to delete. A growing compatibility layer without a scheduled
deletion point must be reverted rather than normalized. No numerical line
threshold determines whether this campaign is finished.

## 5. Secondary campaign: one typed edit transaction

Whole-text relocation now provides MOI and SuperCollider instrumentation with
a unified staging and publication backend. Validation mutation and fault
injection still compose through a separate recursive two-pass flow. That flow
plans and applies mutation, validates it, rebuilds inventory over an
intermediate image, instruments the image, carries perturbation plans across
the mutation, and merges patch artifacts. Anchors, overlaps, and locations
must be translated between coordinate systems.

Some of this is semantically necessary: mutation can change the program that
instrumentation must analyze. The refactoring must preserve re-analysis of
that intermediate semantic image. It should eliminate the independently
committed transform and manual carrying and merging, not pretend that every
edit was planned independently against the original bytes.

A credible destination is a typed source-coordinate transaction graph that
can represent:

- fault replacements and validation mutations;
- barrier moves;
- perturbation;
- instrumentation fragments and branches; and
- relocation and publication metadata.

The intermediate mutated image can then be a derived overlay or view of the
transaction, against which the required second analysis runs. One final
relocation, placement, and publication pass realizes the coordinates.

The plausible net opportunity is **1,500–3,000 lines**, with high risk; this
estimate ranks the candidate rather than defining success. Its hard prototype
must combine a barrier move or fault replacement,
perturbation, and MOI instrumentation. It succeeds only if semantic
re-analysis remains intact and the old recursive composition and carrying
machinery can be deleted. Routing one additional producer through
`ConSanTextFragment` without changing the transaction model is not this
refactoring and is unlikely to shrink the tree.

Stable program-site IDs should make this campaign easier. The edit transaction
should therefore follow the primary campaign unless investigation uncovers a
clearer independent deletion path.

## 6. High-risk reserve: a compact typed probe-program IR

The preceding refactorings centralized patch lifecycle, resource planning,
version claims, routing, and publication. More than ten thousand lines remain
in and around mode emitters. These lines implement genuinely different
evidence algorithms, but express their programs through a low-level
`InstructionSequence`: concrete registers, individual instruction-builder
calls, manual labels and branches, and repeated failure propagation.

A deliberately small probe-program representation might provide:

- virtual operands and named resource roles;
- structured control flow and labels;
- typed memory, observation, report, and guest-boundary operations;
- transactional or sticky emission failure; and
- target lowering after resources have been resolved.

Modes must retain their evidence algorithms and mode-local leaf operations.
A universal callback table or a variant that contains all of the old mode
state would preserve the old complexity under a new facade and must be
rejected.

A safe builder-only cleanup may yield **400–900 lines**. A successful probe
IR may yield **2,000–4,000 lines**, but confidence is lower than for the
program-site arena. These figures guide prioritization only. This work starts
only after the new site representation has clarified emitter inputs, and only
as a hard-pair prototype chosen to combine shared mechanism with meaningfully
different policy. It is abandoned if it translates existing imperative code
one-for-one into a new vocabulary instead of deleting a real mechanism.

## 7. Cross-subsystem candidate: shared HSA interposition

The main ConSan HSA hook contains about 4.7k governing lines. RocJitsu's
generic DBT hook has another independently evolved HSA wrapper. Both own
code-object readers, HSA load/reader interposition, API table and layer setup,
executable tracking, and lifecycle state. Their low textual similarity means
this is not mechanical deduplication, but a shared HSA interposition substrate
with client-owned policy may eliminate one wrapper implementation.

The likely opportunity is **800–2,000 lines across RocJitsu**, with high
integration risk. It must be measured globally: moving ConSan code outside
the ConSan counting scope is not shrinkage. Because this crosses subsystem
ownership and does not directly clarify device-side representation, it ranks
behind the site arena and edit transaction.

## 8. Areas not presently justified as macro replacements

### 8.1 Architecture support

Target-local code is only about 2.2k lines, or 3.3% of ConSan. Architecture
consolidation alone cannot produce another several-thousand-line reduction.
Common placement and policy code still contains family predicates even though
`ConSanTargetProfile` carries capabilities. Replacing scattered predicates
with coherent strategy records—such as scalar placement model, persistent
overflow policy, workgroup-identity source, and accumulator model—may remove
**250–700 lines** and improve architecture locality. Merely moving each branch
to a target file improves layout but not total complexity.

### 8.2 The placement solver

`consan_moi_placement.inc` is about 3.5k governing lines, but it is now one
shared object-wide solver rather than parallel solvers. Size alone does not
justify another framework. Stable site IDs, container lifting, target
strategies, and narrower result types should simplify it from the outside
in. A wholesale replacement currently lacks a deletion-backed thesis.

### 8.3 Independent validation

`consan_validation.inc` is about 3.0k governing lines. Repetitive decoding,
bounds, and diagnostic scaffolding may admit a declarative cleanup, but
independent derivation is a trust property. Validation must not consume
encoder products or producer verdicts to reduce duplication. An independent
instruction-pattern matcher may save **300–800 lines**; no safe multi-
thousand-line deletion is presently evidenced.

### 8.4 Direct mode-emitter merging

Mode pairs continue to share concepts while differing materially in resource
selection and probe construction. Previous work already extracted common
lifecycle. Direct merging now appears worth only **100–300 lines** unless the
probe-program experiment proves a deeper common representation. Similar
names or control-flow silhouettes are not sufficient evidence.

## 9. Smaller shrinkage harvest

Smaller work should normally follow a macro slice so that it does not polish
a representation scheduled for deletion.

| Opportunity | Credible net | Direction |
| --- | ---: | --- |
| Lift `ConSanKernelInfo` and `ConSanFunctionInfo` into a common container plus kernel payload | 400–1,000 | First slice of the site arena, not a parallel abstraction. |
| Factor synchronization event/sequence payload through stable IDs | 400–1,000 alone | Part of the 2k–4k arena payoff when performed end to end. |
| Make patch proof payload a tagged variant instead of a universal optional bag | 300–800 | Common geometry plus family-specific proof payload; synergizes with the edit transaction. |
| Add sticky failure and transactional checkpoints to `InstructionSequence` | 400–900 | Preserve rollback and label-resolution semantics. |
| Replace target-family predicates with target strategy records | 250–700 | Improves shrinkage and architecture locality. |
| Introduce a tagged resolved-state/storage handle | 250–700 | Replaces mutually exclusive register/private/owner-local fields. |
| Introduce a narrow generic placement attempt/result type | 100–300 | Removes repeated rejection and diagnostic plumbing. |
| Consolidate host report-registry wrappers | 150–400 | Shared lifecycle exists; retain mode-specific identity and retirement rules. |
| Use declarative, independently decoded validation patterns | 300–800 | Never weaken the producer/validator trust boundary. |
| Retire legacy configuration spellings | 100–250 | Requires an explicit compatibility decision. |
| Move test-only model contracts out of production headers | 40–80 | Do not mistake the production report model for test-only code. |
| Remove empty implementation tombstones | Single digits | Already exhausted; not a scheduled shrinkage project. |

These estimates overlap. Container lifting and synchronization factoring are
components of the primary arena estimate, not additions to it.

## 10. Execution discipline

The work proceeds through a deletion-backed decision funnel:

1. Prototype the program-site arena on hard ownership and synchronization
   cases.
2. Compare new arena and adapter code with the old storage, projection, copy,
   join, and traversal mechanisms that can actually be deleted.
3. If the representation works, extend it through synchronization and policy
   side tables and remove the displaced forms.
4. Reassess and prototype the edit transaction using stable semantic IDs.
5. Harvest local simplifications exposed by the accepted foundational model.
6. Re-measure the emitter envelope and attempt the probe-program hard pair
   only if it exposes a genuinely shared semantic vocabulary.
7. Recalculate smaller opportunities against the resulting tree rather than
   harvesting structures scheduled for replacement.

Every completed vertical slice must:

- preserve all observable behavior and supported modes and architectures;
- add a focused regression test immediately for any bug discovered;
- preserve independent validation and explicit phase ownership;
- pass focused tests during development and the established complete
  nonphysical RocJitsu/ConSan matrix at meaningful checkpoints using `-j16`;
- avoid frequent physical-gfx1201 testing, reserving it for risk-appropriate
  milestones;
- record governing production-line movement using the baseline scope above;
- use frequent local commits and never push; and
- remove the displaced legacy implementation rather than leaving permanent
  compatibility layers.

Temporary growth is permissible only within a named vertical replacement
whose old implementation and deletion point are explicit. A slice that
cannot produce structural simplification and real deletion after its hard
case is understood should be documented as negative evidence and reverted or
cleanly retired. Iteration must not continue indefinitely around a non-
shrinking abstraction. Line counts remain useful telemetry for detecting
growth and measuring realized gains, never a gate for ending the refactoring.

## 11. Exhaustion criterion

The ninth refactoring is not complete merely because one prototype works, a
round-number line target is reached, the named candidates above have been
visited, or the remaining files look tidier. The candidates in this document
seed the search; they do not bound it. The work continues while a deep read
can identify any credible structural or local change that should reduce the
implementation while preserving behavior and improving or maintaining its
architecture.

At a minimum, the three presently identified macro questions need evidence-
backed answers:

1. The program-site/container/synchronization arena has either produced a
   deletion-backed implementation or failed a documented hard prototype.
2. The typed edit transaction has either eliminated the old composition and
   carrying machinery or failed the combined mutation, perturbation, and
   instrumentation case.
3. After those results, a probe-program hard pair has either demonstrated a
   useful shared conceptual boundary and deleted shared mechanism, or shown
   that the mode algorithms do not share enough mechanism.

Accepted work must leave no duplicate old representation, shadow lifecycle,
or transitional adapter debt. Architecture-specific behavior remains behind
target capabilities or strategies; mode-specific algorithms remain locally
owned rather than leaking through common orchestration.

After these macro candidates have been proved or falsified, the code must be
deep-read again rather than declared done. The remaining smaller cleanup
inventory is then harvested and remeasured, and the simplified structures are
examined for opportunities that were previously hidden. That cycle repeats:
investigate, implement or falsify, delete displaced code, consolidate, and
deep-read the new actual shape.

Completion means that this repeated audit no longer finds a credible large,
medium, or small shrinkage opportunity—not that the tree has reached a chosen
size. “No opportunity” requires positive evidence from the current code: no
parallel representation or lifecycle whose replacement would delete its
joins; no internal data structure whose redesign would unlock meaningful
sharing; no remaining mode or architecture leakage that duplicates common
mechanism; no legacy path or compatibility layer that can responsibly be
retired; and no collection of local cleanups worth harvesting. At the present
baseline, that condition has not been met.
