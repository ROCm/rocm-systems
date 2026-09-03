# ConSan seventh refactoring: discover the next structural collapse

## 1. Status and mandate

The sixth refactoring succeeded. It replaced five subsystems, reverted three
economically unsuccessful experiments, and reduced the governing production
implementation count from 88,023 to 83,516 lines without reducing supported
modes, targets, or test coverage. Its final audit found no already-proven next
replacement of comparable value.

That conclusion is a starting condition for this plan, not a claim that ConSan
is irreducibly an 83,000-line program. The seventh refactoring must discover
the next large deletion opportunity by exerting pressure on the architecture
that now exists. It may move component boundaries, replace representations,
or replace whole subsystems when the evidence supports a large payoff. It may
not resume an indefinite sequence of locally tidy, low-yield edits.

The direction of travel is:

> Make extension and change pressure reveal duplicated authority, then replace
> the widest duplicated representation or lifecycle with one smaller semantic
> path and delete every displaced implementation.

Code shrinkage is a required outcome of the campaign, but line count is not a
license to erase necessary distinctions, generate hidden code, weaken
validation, or move implementation into tests. Better ownership and a smaller
representation should cause deletion; deletion must then be reaped in the
same macro-slice.

The exact subsystem to replace remains a variable being solved for. The
candidate examples in this document guide investigation but do not bound it.

## 2. Consolidated starting tree

Before writing this plan, the post-sixth tree was consolidated around its two
extension axes.

Architecture-owned code now lives under `targets/`:

```text
targets/
  rdna3/
  rdna4/
  cdna3/
  cdna4/
  cdna5/
  shared/
    cdna3_cdna4/
    rdna3_rdna4/
    consan_*_rdna4_cdna5_*  # mechanisms shared by those architectures
    consan_*_common_*       # mechanisms with a wider consumer set
  consan_*_target_ops.*       # normalized contracts and dispatch
```

Architecture owners use the canonical names RDNA3, RDNA4, CDNA3, CDNA4, and
CDNA5. In ConSan, `gfx11` means RDNA3 and `gfx12` means RDNA4; they do not name
distinct layers or broader families. CDNA5 is likewise its own architecture,
not a flavor of GFX12. Exact target names such as gfx1100, gfx1201, gfx942,
gfx950, and gfx1250 remain only where a file, fixture, target ID, or diagnostic
is about that exact target.

Cross-architecture implementations are named for their actual consumers or
their semantic responsibility. CDNA3/CDNA4 sharing remains explicit, as does
RDNA3/RDNA4 sharing. Individual mechanisms shared by otherwise distinct
architectures, such as RDNA4 and CDNA5 memory decoding, live directly under
`targets/shared/`; that sharing does not create an artificial architecture
family.

Mode-owned transformer code now lives under:

```text
modes/
  record_replay/
  sampled/
  inline_shadow/
  supercollider/
```

The host report stack mirrors this organization for Record/Replay, Sampled,
and InlineShadow decoders, analyzers, renderers, and mode-only replay helpers.
Mechanisms with real multiple-mode consumers remain in the shared directory.
In particular, dynamic record emission, exact-shadow emission, common
placement, report transport, patch publication, and shared lowering were not
duplicated merely to make directory ownership look pure.

The empty `consan_validation_gfx11_target_ops.cpp` tombstone was deleted. The
later canonical-name correction also removed `ConSanEncodingFamily`: it had
incorrectly grouped CDNA5 with RDNA4 merely because selected encodings are
shared. Exact architecture predicates now say which architectures consume a
mechanism, while target capabilities describe the semantic operation required
by common code.

The production CMake ownership check is recursive, and the
architecture-boundary test now rejects:

- a concrete architecture-named production file outside `targets/`;
- a mode-named production or host-report file outside `modes/`, except for
  architecture providers under `targets/`;
- target-operation files inside a mode directory;
- generated ISA or concrete architecture knowledge in mode providers; and
- mode policy in target providers;
- architecture-owner filenames using `gfx11` or `gfx12`, or using `gfx1250`
  outside the exact-target profile; and
- restoration of a synthetic GFX encoding-family contract.

The hypothetical-target exercise now registers two independent normalized
operation packages and proves exact selection plus fail-closed lookup of an
unregistered target. The hypothetical-mode exercise does the analogous work
with two independent policy packages, including target-neutral scratch,
report, prologue, and call-form behavior.

This consolidation and the subsequent nomenclature correction are deliberately
the baseline preparation, not a seventh-refactoring macro-slice. Relative to
the sixth-refactoring final tree, they remove one comment-only file and 35
implementation lines. Of those, 33 came from deleting the false encoding
family and its projections rather than from the file renames themselves:

| Signal | Sixth final | Initial consolidation | Canonical baseline | Total change |
| --- | ---: | ---: | ---: | ---: |
| Production files | 312 | 311 | 311 | -1 |
| Physical production lines | 97,525 | 97,513 | 97,460 | -65 |
| Nonblank production lines | 91,118 | 91,107 | 91,055 | -63 |
| Production implementation lines | 83,516 | 83,514 | **83,481** | -35 |

The consolidated implementation is distributed as follows:

| Physical owner | Files | Implementation lines |
| --- | ---: | ---: |
| Shared static transformer | 174 | 47,594 |
| Target packages and dispatch | 42 | 2,297 |
| Record/Replay transformer mode | 12 | 5,887 |
| Sampled transformer mode | 18 | 6,545 |
| InlineShadow transformer mode | 14 | 5,941 |
| SuperCollider transformer mode | 12 | 5,704 |
| Shared host report/hook stack | 20 | 7,839 |
| Record/Replay host mode | 7 | 559 |
| Sampled host mode | 7 | 743 |
| InlineShadow host mode | 5 | 372 |

The canonical baseline passed a full local-toolchain rebuild. The complete
nonphysical ConSan gate passed 4,787/4,787 tests. The complete nonphysical
RocJitsu gate then passed all 9,571 executed tests, with one disabled test and
nine expected environment skips. These results cover all four modes and the
gfx942, gfx950, gfx1100, gfx1201, and gfx1250 emulated targets.

## 3. What the physical layout reveals

The new tree makes the two intended growth axes visible, but physical locality
does not by itself prove semantic independence.

Target locality is already strong. Only 2,297 implementation lines are in the
target component, and exact target packages are small: 7 lines for the CDNA3
profile, 9 for CDNA4, 86 for RDNA3, 129 for RDNA4, and 291 for CDNA5. Most
target implementation is correctly shared through normalized dispatch and
explicit cross-architecture mechanisms. Architecture code is therefore
unlikely to be the largest direct deletion source. Its more valuable role in
the seventh refactoring is as an extension-pressure probe: a hypothetical new
architecture can reveal common code that still assumes the current closed set.

The canonical-name correction has reaped only its immediate consequences so
far. Deleting `ConSanEncodingFamily`, its profile field, its projections, and
their plumbing removed 33 implementation lines. It did not merge two large
implementations: the former `targets/shared/gfx12/` code was already one copy
consumed by RDNA4 and CDNA5. Renaming that ownership made the relationship
truthful, but did not by itself create a duplicate body to delete.

The larger downstream opportunity remains open. There are currently 56 calls
outside `targets/` to predicates such as `consan_arch_is_cdna3_or_cdna4`,
`consan_arch_is_rdna4_or_cdna5`, and `consan_arch_is_cdna5`. Some name genuine
shared ISA mechanisms. Others may be common or mode code reconstructing a
semantic target capability that should already have been normalized by a
target provider. The seventh refactoring must classify these call sites by the
individual fact they need and look for repeated branches, duplicated
admission, or parallel lowering paths that can disappear. It must not simply
replace them with another broad family enum or a bag of booleans: a normalized
capability is worthwhile only when it removes downstream authority and code.

This opportunity is primarily indirect. With only 2,297 implementation lines
in the complete target component, target-local cleanup alone cannot supply the
campaign's multi-thousand-line objective. The higher-payoff hypothesis is that
truthful target normalization can collapse architecture-dependent lifecycles
inside the much larger shared and mode implementations. That hypothesis has
not yet been pursued or credited as a seventh-refactoring macro-slice.

Mode locality is physically strong but semantically less complete. Several
shared aggregation or coordination files import mode-owned products:

- `consan_moi_report_contract.h` aggregates every mode's report ABI;
- `consan_moi_barrier.cpp` assembles Record/Replay and InlineShadow barrier
  bodies into a shared placement mechanism;
- `consan_moi_pipeline.cpp`, `consan_moi_placement.cpp`, and
  `consan_moi_prologue.cpp` consume selected mode products;
- `consan.h` aggregates SuperCollider route and spill types; and
- the generic host decoder, analyzer, renderer, and hook coordinator consume
  mode-owned report stages.

Some of these are legitimate one-way registries or public umbrella contracts.
Others may indicate that a common mechanism and a mode policy still meet in
more than one place, or that an intermediate representation is too broad.
Directory movement intentionally does not answer which is which. The seventh
refactoring must trace the values crossing each seam and distinguish a narrow
composition edge from a repeated lifecycle.

The largest remaining files are also not automatic refactoring targets. The
4,653-line HSA hook coordinator, 4,502-line MOI placement body, 3,788-line
validation body, 3,053-line SuperCollider LDS body, 2,716-line MOI prologue,
2,631-line synchronization emitter, 2,475-line synchronization analysis, and
2,419-line Sampled synchronization body contain substantial distinct behavior.
Sixth-refactoring experiments demonstrated that similarity of control-flow
shape or file size alone can yield a larger abstraction and almost no net
deletion.

The useful post-consolidation question is therefore not "which file is
largest?" It is:

> Which fact, decision, or lifecycle is independently reconstructed across
> these now-visible owners, and what smaller authority could make the
> reconstructions disappear?

## 4. Proposed executable goal

Carry out a discovery-driven structural refactoring of ConSan from the
83,481-line canonical baseline. Use extension pressure, change-impact
tracing, comparative mode reads, and representation-lifetime analysis to find
large duplicated authorities. For every accepted macro-slice, state a
falsifiable deletion thesis, replace the complete subsystem boundary, delete
the displaced implementations and compatibility paths, preserve independent
validation and all behavior, and measure the net result. Continue until at
least 3,000 net production implementation lines have been removed and a fresh
whole-tree audit finds no comparably strong remaining candidate, or until
multiple serious experiments produce concrete evidence that the remaining
distinctions cannot be collapsed economically. Do not substitute
micro-refactoring for this outcome.

The 3,000-line value is a review floor, not an automatic completion condition.
A 5,000-line reduction is the stretch objective. If a credible high-payoff
candidate remains after either number, the campaign continues. If a candidate
fails, record the negative result, revert its production scaffold, and change
the hypothesis rather than weakening the economics.

## 5. Discovery phase: pressure the extension seams

### 5.1 Full hypothetical-target exercise

The current target exercise proves the generic registry, which is necessary
but intentionally small. The next exercise should temporarily model a sixth
target deeply enough to touch the complete target-normalization surface:

- immutable target profile and exact target identity;
- program-analysis decoding;
- fault mutation operations;
- validation operations;
- target LDS/VGPR-bank behavior where applicable; and
- the target operations consumed by SuperCollider and MOI.

The exercise need not invent a real ISA or remain as a production target. A
test-owned synthetic encoding package is sufficient if it compiles and drives
the real normalized consumers. Maintain an edit ledger containing every
production file outside the new target package that had to change. Each such
edit is either a legitimate single registry insertion or evidence that target
knowledge remains distributed. Multiple edits expressing the same new target
fact identify a candidate authority to collapse.

The exercise is successful as discovery even if its temporary target is later
removed. Any generic seam strengthened by it should remain in executable
tests. Do not retain a fake production architecture merely to demonstrate the
point.

### 5.2 Full hypothetical-mode exercise

Similarly, temporarily model a fifth mode through a vertical but deliberately
small behavior:

- request and mode selection;
- object planning and resource demand;
- at least one access and one synchronization evidence path;
- report layout and static attribution;
- host snapshot, decode, analysis, and rendering; and
- independent final-validation obligations for its patches.

The hypothetical mode should use the existing normalized target interface and
must not add per-target implementations. Record every production edit outside
its new mode directories. A single explicit registry or umbrella import is
acceptable. Repeated switches, parallel field projections, mode-sized arrays,
or shared structs that must grow in several stages are evidence of an axis
leak.

This is not a request to ship a toy mode. Its purpose is to measure the real
cost of adding one and expose the representations responsible for edit fanout.
Keep the strongest generic extension tests and delete the temporary product
once the experiment has selected its macro candidate.

### 5.3 Cross the axes deliberately

Run the hypothetical target and hypothetical mode together. The new mode must
work over the synthetic target by composition, without a mode-by-target file,
switch, or adapter. Any required pair-specific edit is high-priority evidence
of `O(N*M)` structure.

The exercise should distinguish three costs:

1. unavoidable registration of a new public identity;
2. one-axis implementation inside the new target or mode package; and
3. edits to old common, old target, or old mode owners.

The third category is the deletion frontier. Do not hide it by adding a broad
callback table or generated registry before understanding what every edit
means.

## 6. Discovery phase: trace representations, not tokens

### 6.1 Representation-lifetime ledger

For one semantic fact at a time, trace its complete lifetime from request or
decoded instruction to host verdict. Record:

- the authority that first knows the fact;
- every struct, optional field, enum, string, index, or pointer that carries
  it;
- every place that re-derives or revalidates it;
- every consumer and the exact subset it needs;
- the point at which it becomes dead; and
- the code required only to translate between two adjacent representations.

Prefer facts that cross many of the newly visible boundaries: owner/epoch
identity, access geometry, synchronization edges, report identity, patch
geometry, dispatch/workgroup identity, scratch and persistent state, and
failure diagnostics. A candidate becomes strong when deleting one middle
representation removes several producer projections, consumer reconstruction
passes, well-formedness switches, and compatibility tests at once.

### 6.2 Change-impact archaeology

Use recent behavioral fixes and accepted sixth-refactoring slices as probes.
For each change, map the files and representations that had to move together.
Repeatedly co-edited fields across analysis, planning, emission, validation,
and host reporting reveal an implicit component even when the code is not
textually duplicated. Conversely, files that merely changed together because
an include path moved are not evidence.

Issue fixes are especially useful because they start from observable behavior
rather than a desired abstraction. A bug that required parallel corrections
to multiple authorities is a candidate for structural collapse. A bug fixed
in one semantic owner with broad tests is evidence that the current boundary
is already sound.

### 6.3 Comparative reads inside the new directories

Read corresponding responsibilities side by side across all four mode
directories and all target packages. Classify regions as:

- identical mechanism with different typed inputs;
- same semantic state machine encoded with different storage;
- shared prefix or suffix around a genuinely different core;
- independently reconstructed common fact; or
- legitimately different behavior that only looks structurally similar.

The output is a responsibility matrix with approximate line ownership and a
proposed smaller representation, not a token-hit or clone count. Similar
syntax without a common semantic lifecycle does not qualify.

### 6.4 Delete-the-middle prototypes

When the ledger identifies adjacent representations of the same decision,
prototype bypassing the middle representation for one difficult path. Measure
the producer code, consumer code, validators, adapters, and tests that would
disappear after complete convergence. This is more informative than first
building a general framework.

The prototype must include the hardest mode and target variants early. A
design tested only on the simplest path often grows into a union-shaped
abstraction during convergence and loses its expected deletion.

## 7. Current candidate families -- examples, not scope

These are the strongest places to begin the discovery instruments. They are
not a promise to refactor all or any of them.

### 7.1 One report identity from device layout to host verdict

The report path spans mode-owned device layouts, common report aggregation,
static runtime mapping, host snapshotting, mode decoders, analyzers, renderers,
and the hook coordinator. Investigate whether the same identity, bounds,
completeness, and provenance facts are represented and checked in several
forms. A smaller immutable validated report view could potentially delete
field-by-field projections and repeated bounds/lifecycle code while leaving
mode interpretation local.

This candidate is rejected if it becomes a generic field-schema DSL, generated
implementation, or variant containing every mode's semantics. The payoff must
come from deleting real repeated ownership, not from moving decoder code into
tables.

### 7.2 Evidence intent through planning, emission, and proof

Accesses, atomics, barriers, and fences travel through intent, site planning,
resource plans, mode-local emission plans, patch commits, runtime mappings,
and final proof. Investigate whether one or more middle products merely
serialize a decision that the next stage reconstructs. A direct immutable
semantic transaction or a smaller proof-carrying emission plan could delete
translation and validation scaffolding across several site kinds.

This is not permission to retry the failed sixth-refactoring monolithic access
transaction. That experiment proved that callbacks and a union of mode
policies do not pay. A new attempt requires a different representation insight
that removes a lifecycle rather than wrapping the existing ones.

### 7.3 Persistent state from prologue through placement

The sixth refactoring removed repeated scalar-range reconstruction, but MOI
placement and prologue remain large and communicate through broad operating,
resource, private-layout, and owner-local state. Use the hypothetical-mode
exercise to determine which of those fields every new mode must understand.
Look for state that is selected, projected, requalified, and restored through
parallel scalar, vector, and private-media paths.

A credible replacement would unify the semantic lifetime of persistent state
while keeping storage-specific emission explicit. It is rejected if it
becomes a general constraint solver, changes fallback priority, or merely
moves the existing branches behind virtual callbacks.

### 7.4 Synchronization knowledge across analysis and modes

Synchronization analysis, shared synchronization emission, Sampled
synchronization, Record/Replay barriers/fences, and InlineShadow causal state
all consume related ordering events. Trace whether they share one normalized
edge or repeatedly rebuild association, scope, owner, epoch, and attachment
facts. A smaller immutable synchronization event graph might remove multiple
projections and mode-local searches.

This candidate must preserve the genuine differences between barrier
lifecycle, atomic ordering, ordinary acquire/release, execution ownership, and
move-destination proof. The earlier estimate that simply merging fault and
synchronization inventories saves only 250--350 lines is not sufficient. It
becomes a macro candidate only if a deeper representation replaces at least
one substantial mode-side lifecycle as well.

### 7.5 Shared-to-mode reverse dependencies

Audit every shared file that imports a mode-owned header or fragment. For each
edge, decide whether it is:

- the one legitimate registry/umbrella edge;
- a common mechanism parameterized by a narrow mode product;
- a mode implementation physically assembled into a common owner; or
- evidence of a broad product that forces common code to know mode details.

The desired result is not zero imports at any cost. The desired result is one
obvious composition boundary per responsibility. If several reverse edges
exist because a mode decision is split across planning, placement, prologue,
and emission, replacing that split authority may delete more code than
mechanically introducing interfaces.

### 7.6 Host hook lifetime and registry state

The shared host stack is 7,839 implementation lines, including a 4,653-line
hook coordinator. The sixth audit found largely distinct HSA interception,
process lifetime, allocation, report collection, and verdict policy, so size
alone does not justify replacement. Reconsider it only through lifetime
tracing: if the same executable/report/allocation identity is maintained in
parallel registries or reconstructed at several callbacks, one ownership
object could remove cleanup, recovery, and cross-map synchronization code.

Do not split the file merely for readability and count that as progress. The
candidate qualifies only when a registry or lifecycle disappears.

### 7.7 SuperCollider access protocols

The LDS and FLAT bodies are now clearly isolated inside the SuperCollider mode
and share target operations outside it. The sixth audit found their routing
and address-space protocols genuinely different. Reopen this region only if a
new comparison identifies a representation below those protocols—for example,
one already-resolved access action whose construction, publication, and proof
are still repeated. A common body that encodes both protocols as options is a
rejected result.

## 8. Negative knowledge that must be preserved

Do not repeat a failed sixth-refactoring experiment without a materially new
deletion thesis:

- the monolithic MOI access transaction moved mode differences into callbacks
  and did not produce the forecast deletion;
- decoded validation restated independent proof obligations and added 617
  physical lines;
- the fully converged dense-relay transaction removed only 147 net lines
  because the apparent repetition was mostly distinct site admission,
  placement, body construction, routing identity, and fallback policy.

Likewise, preserve the accepted sixth-refactoring authorities. Do not
reintroduce parallel instruction builders, enum spelling tables, a private
SuperCollider direct reservoir, split fault planning and re-resolution,
ad-hoc scalar range reconstruction, or the fixed all-expected-CAS validation
bug.

Negative results are architectural evidence. Record why an experiment failed,
revert all of its production scaffold, and require a different underlying
representation before revisiting the region.

## 9. Macro-slice protocol

### 9.1 Bound and deep-read

Trace one complete semantic lifecycle and all difficult consumers. Quantify
the existing implementation that the replacement would actually delete.
Include the hardest mode and architecture cases in the initial read.

### 9.2 State a falsifiable deletion thesis

Before substantial implementation, record:

- the duplicated authority or representation;
- the smaller replacement and its owner;
- every old producer, consumer, adapter, field, and test scaffold that will be
  deleted;
- the semantic distinctions that remain explicit;
- estimated gross deletion, new implementation cost, and net reduction;
- focused differential and corruption tests; and
- abort conditions.

An accepted macro candidate should forecast at least 1,000 net implementation
lines of deletion. A 750-line net result is the normal abort floor. A smaller
enabling change is allowed only inside the same macro-slice when complete
convergence still has a credible four-digit payoff; it is not an independent
checkpoint.

### 9.3 Attack the hard variant first

Prototype the representation on the mode/target combination that puts the
most pressure on it. If the design immediately needs mode enums, target
switches, opaque callbacks, a union of every old product, or parallel old/new
state, stop and reassess before migrating easy cases.

### 9.4 Converge completely and delete

Move all consumers required by the thesis. Delete old entry points, fields,
reconstruction passes, compatibility adapters, well-formedness switches, and
tests that exist only for the removed representation. Preserve their behavior
through tests of the new semantic boundary. No accepted slice ends with two
authorities pending a later cleanup.

### 9.5 Validate, account, and reassess globally

Run focused tests during convergence, the complete nonphysical ConSan matrix
at meaningful cutovers, and the complete nonphysical RocJitsu matrix before
closing a macro-slice. Record exact implementation accounting and any new
regression tests. Then return to the whole tree and choose the strongest
remaining candidate; do not continue polishing the just-completed subsystem
by inertia.

## 10. Anti-circling and code-economics rules

1. File movement, renaming, wrapper extraction, and interface introduction do
   not count as seventh-refactoring shrinkage.
2. Do not start a candidate because a file is large or two loops look alike.
   Identify the duplicated semantic authority first.
3. Do not build a framework speculatively. Prove the smaller value or
   lifecycle on one hard vertical path, then converge or revert.
4. Count all new common code, adapters, tables, and external enabling changes
   against the deletion result.
5. Do not move implementation into tests, generators, build scripts, data
   tables, templates, or comments to improve the metric.
6. Do not keep compatibility paths for private ConSan internals after their
   callers migrate.
7. A new abstraction with one consumer must justify itself by immediate gross
   deletion or disappear with the slice.
8. If two serious candidates fail their economics consecutively, stop
   implementation and repeat the cross-axis and representation discovery
   passes before choosing a third.
9. Do not fill a numeric gap with micro-deletions. Small dead-code harvesting
   is required inside a macro-slice but is not a substitute for structural
   collapse.
10. A bug found during any experiment is fixed immediately and receives a
    regression test, even if the experiment itself is later reverted.

## 11. Fixed invariants

- Preserve Record/Replay, Sampled, InlineShadow, and SuperCollider behavior.
- Preserve gfx942/CDNA3, gfx950/CDNA4, gfx1100/RDNA3, gfx1201/RDNA4, and
  gfx1250/CDNA5 support.
- Keep concrete architecture implementation under `targets/`, using canonical
  RDNA/CDNA architecture names for architecture owners and exact gfx names only
  for exact-target files, fixtures, IDs, and diagnostics. Shared mechanisms
  must name their actual consumers without inventing a broader architecture
  family.
- Keep mode implementation under its mode directory; shared mechanisms remain
  single implementations outside those directories.
- Do not introduce a mode-by-target implementation matrix.
- Keep target providers free of mode policy and mode providers free of raw ISA
  architecture decoding.
- Preserve independent final validation. A producer's plan, pointers, or
  success verdict cannot become the validator's proof.
- Preserve fail-closed behavior, transactional mutation, exact resource and
  liveness semantics, runtime report trust, and public diagnostics.
- Preserve or strengthen tests. When a representation-specific test is
  retired, replace it with the strongest behavioral or corruption contract
  that pins the same obligation.
- Build and test with the local TheRock toolchain at `-j16`.
- Run nonphysical tests as the normal gate. Physical gfx1201 testing is an
  infrequent, deliberate milestone rather than a routine iteration cost.
- Make frequent coherent local commits and never push.

## 12. Accounting and completion

The seventh-refactoring baseline is the canonical 311-file tree at 83,481
production implementation lines. The governing scope remains:

- `lib/rocjitsu/src/rocjitsu/code/patch/consan/` and
  `lib/rocjitsu/src/rocjitsu/hooks/consan/`;
- production `*.cpp`, `*.h`, and `*.inc` files only; and
- tests, generated code, documentation, comments, and blank lines excluded.

For every macro-slice, record physical, nonblank, and implementation lines;
gross deletions; new implementation; net reduction; deleted representations;
retained distinctions; tests added or strengthened; modes and targets run;
and any production cost outside the governing directories.

The seventh refactoring is complete only when all of the following hold:

1. At least 3,000 net governing implementation lines have been removed, with
   no metric displacement or product-scope reduction.
2. At least one accepted subsystem replacement has removed 750 net lines, and
   the campaign is not merely an accumulation of small cleanups.
3. The full hypothetical target and mode pressure exercises have been
   performed, including their cross-axis composition, and their production
   edit fanout has either been eliminated or justified as a narrow registry
   boundary.
4. Every accepted slice has one surviving authority and no compatibility
   adapter, parallel representation, abandoned scaffold, or deferred deletion.
5. The target and mode directory boundaries remain truthful, and common code
   does not regain raw axis-specific policy.
6. Independent validation and runtime fail-closed behavior remain intact.
7. Every discovered bug has a regression test.
8. The complete nonphysical ConSan and RocJitsu gates pass; a deliberately
   selected physical milestone is used only when the behavioral risk warrants
   it.
9. A fresh whole-tree deep read, not just the initial candidate list, finds no
   comparably strong unattempted macro replacement that belongs in this
   campaign, or records concrete negative evidence for the remaining options.

Reaching 3,000 lines does not end the work while a stronger candidate remains.
Conversely, if repeated hard prototypes show that the remaining apparent
duplication is semantic rather than representational, stop with the measured
evidence instead of manufacturing an abstraction or silently changing the
goal.

## 13. Deliberately deferred and deliberately open

Broader design-document rewrites and CI integration are deferred to a later
phase. The executable boundary test remains the local enforcement mechanism
during this campaign. Documentation of macro-slice theses, measurements, and
negative results belongs in this file because it directly controls execution;
general user or maintainer documentation does not.

The destination design is deliberately not prescribed. The seventh
refactoring may discover that the correct collapse lies in report identity,
evidence planning, persistent-state lifetime, synchronization representation,
host registries, or a component not named here. Component boundaries may move
substantially. The stable direction is that a fact is decided once, represented
for only as long as necessary, consumed through one obvious path, independently
validated where trust requires it, and never multiplied by both mode and
architecture.

## 14. Execution ledger

This section records pressure-test evidence as the campaign proceeds. A probe
is not an accepted macro-slice, and its temporary production scaffolding does
not survive unless it establishes an economically useful replacement.

### 14.1 Initial whole-tree candidate audit

The first pass classified all 56 architecture-predicate calls outside
`targets/`. They ask for separate facts: VCC and accumulator layout, tuple
alignment, scalar preload and call form, branch-only spill support, persistent
bank behavior, CDNA5 replay behavior, and encodings shared by RDNA4 and CDNA5.
Replacing those questions with a profile boolean for each spelling would add
plumbing while leaving every downstream branch intact. The complete target
component is only 2,297 implementation lines, so target dispatch cleanup alone
cannot provide the required macro deletion. A target-normalization slice must
instead prove that one of these facts causes a substantial lifecycle in common
or mode code to disappear.

Deep reads also rejected four tempting size-based candidates:

- the report decoder, analyzer, renderer, trust, and lifecycle stages already
  have distinct owners, and converting their counters to arrays would remove
  only a few hundred lines while obscuring typed evidence;
- Record/Replay, Sampled, and InlineShadow access lowering share orchestration
  shape but not the probe body, entry gate, return ABI, evidence policy, or
  fallback contract; a prior complete transaction prototype would have added
  more common and adapter code than it removed;
- synchronization event, sequence, fence-candidate, and barrier-lifecycle
  records overlap in fields but represent different mutation and association
  lifetimes; field/index consolidation alone is again only a few hundred
  lines; and
- replacing direct diagnostic publication with a generic failure sink removed
  only about 58 lines from the hardest 1,482-line emitter before charging the
  shared abstraction, and did not delete a semantic lifecycle.

The configuration path was also rechecked rather than assumed to be broad
legacy state. `HookConfig` already inherits the normalized request, transform,
runtime, debug, mutation, and bound-resource contracts; `ConSanOptions` is a
thin aggregate of the same contracts. There is no remaining parallel option
representation large enough to replace.

### 14.2 Hypothetical-mode identity pressure

A temporary fifth `ConSanMoiEngine` enumerator was compiled through the full
tree with exhaustive-switch warnings as errors, then removed. Before any
mode-local implementation was added, the production edit ledger contained six
choke points:

1. the report ABI enum and its canonical spelling vocabulary;
2. the public capability-engine projection;
3. request validation;
4. raw report-header engine validation;
5. the host decoded-report variant selection; and
6. the host analysis-variant/engine consistency check.

All transformer sources beyond those choke points compiled unchanged. The
many test compilation failures all originated in two exhaustive switches in
the shared test-support header, not in repeated production policy. This was
strong evidence that the existing `MoiModeOperations` boundary had already
removed most transformer-side identity fanout. Section 14.6 records the later
full vertical exercise, including access, synchronization, report, host,
validation, and synthetic-target composition.

### 14.3 Full synthetic-target pressure

The unsupported-by-ConSan gfx1151/RDNA3.5 product was temporarily admitted as
a sixth target, using the existing RocJitsu decoder and an RDNA3-equivalent
ConSan profile solely as an extension probe. It was driven through native-LDS
lowering in Record/Replay, Sampled, InlineShadow, and SuperCollider, then all
temporary target and test code was removed.

Inside ConSan, the edit ledger was narrow:

1. one target-owned profile;
2. the central profile registry plus its two profile-invariant predicates;
3. one program-analysis operations registration; and
4. one SuperCollider operation dispatch case.

No mode needed target-specific policy or a mode-by-target adapter. Program
inventory and target lookup worked immediately. Initially, however, all three
MOI engines degraded to their fail-closed inventory-only result: common
special-state emission could not obtain instructions for the new architecture.
SuperCollider would have failed for the same underlying reason. The dependency
was below ConSan: RocJitsu's architecture-neutral `instrumentation_builder.h`
contains 70 direct `RDNA3` backend selections, and its RDNA3 backend admits only
the exact RDNA3 architecture identity. Temporarily routing RDNA3.5 through
that backend made all four ConSan modes pass without another ConSan edit.

This is strong negative evidence for a target-normalization macro-slice inside
the governing ConSan scope. The ConSan target boundary already composes modes
with a newly admitted architecture, and its remaining fanout is small explicit
registration or dispatch. The wider instruction-builder facade is a real
RocJitsu architecture-extension boundary, but it is outside this campaign's
governing implementation scope and cannot provide ConSan shrinkage. Any later
work there should be a separately scoped RocJitsu refactoring based on actual
RDNA3.5 encoding compatibility, not an alias retained from this synthetic
exercise.

The target half of the full pressure exercise is therefore complete. Its
cross-axis result is favorable: every existing mode composed with the
synthetic target once the lower builder dependency admitted it, and Section
14.6 subsequently drove the hypothetical fifth mode over that same target
without pair-specific code.

### 14.4 Representation and host-lifetime rejection evidence

A complete trace from `ConSanObservationPlan` through planned patches,
`ConSanCommittedLowering`, the coverage ledger, final validation, and runtime
mapping found some repeated projections: original physical and semantic sites
are derived from intent IDs, and per-intent outcome state mirrors accepted
commits. Removing them would save only a few hundred lines while forcing the
independent validator to trust producer-owned state or repeatedly join mutable
offsets. The current plan/commit split is a deliberate trust boundary, not a
macro-sized duplicated authority.

The host hook registries were likewise traced through reader creation,
transform admission, executable load, report allocation, dispatch, unload,
reader destruction, and process shutdown. The reader registry, transformed
replacement storage, SuperCollider reports, MOI reports, and private dispatch
state have different keys and retirement events. Combining them into one
executable-owner object would retain the maps internally and add optional
lifecycle state rather than delete a registry. Shared report cleanup and
summary mechanics are already factored below the two mode-specific policies.

These regions should not be reopened on field similarity alone. A later
candidate must show that an entire trust or lifecycle transition disappears,
not merely that two products contain comparable IDs or counters.

### 14.5 Macro-slice 1: one fail-fast MOI emission transaction

The post-pressure whole-tree read rejected two more apparent opportunities.
Successive mutation stages must continue to reparse and publish the current
replacement image: every stage grows `.text`, and the next stage's cave and
branch placement is defined relative to that new end. A mutable image-session
wrapper would remove boilerplate but not a planning or emission lifecycle.
Automatic scalar placement likewise cannot be collapsed into component-local
fallback. The sixth-refactoring differential tests proved that owner-wide
partial placement and RDNA4 object-wide spill coordination are observable
solver semantics.

Failure propagation, however, exposes a wider duplicated lifecycle. There are
1,065 production sites where a transformer helper appends exactly one fatal
diagnostic and immediately returns failure. Hundreds of their callers then do
nothing except return the same failure, and 175 declarations or definitions
carry an error-vector output parameter. This population is distinct from
planning and final validation, which intentionally aggregate independent
diagnostics, and from warnings that select a supported fallback.

The bounded replacement is one typed, fail-fast abort for the MOI mutation
transaction. An emitter raises the exact existing diagnostic; the owner that
still holds the partially mutated `ConSanTransformArtifacts` catches it once,
publishes it to `errors`, and follows the existing invalid-result finalization
path. RAII continues to retire patchers and temporary images. Unsupported
resource decisions, best-effort inventory rejection publication, warnings,
and independent final validation retain their current non-throwing contracts.

The hard prototype is the owner/private prologue builder because it has the
deepest nested propagation and mixes SGPR, VGPR, private-memory, dynamic-stack,
paired-entry, and branch-only paths. It must prove that exact diagnostics and
the surviving artifact are preserved before access and synchronization
emitters migrate. Complete convergence is expected to delete at least 700
direct append-and-return statements, several hundred propagation branches,
and the corresponding error-parameter plumbing, while adding only the typed
failure and two narrow catches. The conservative target is **at least 1,000
net implementation lines removed**. Abort and revert if the catch must carry a
union of stage state, if ordinary unsupported fallbacks must become
exceptions, if exact diagnostic behavior changes, or if the converged MOI
domain cannot clear the 750-line macro floor.

The hard prototype rejected the thesis. It introduced the typed failure and
narrow catches, converted the prologue's 128 direct fatal publications,
removed error parameters from the complete internal builder chain, changed
success-or-abort helpers to `void`, and deleted their caller-side propagation
branches. The result built successfully across the whole RocJitsu tree, but
reduced the governing count from 83,481 to only 83,433 lines: **48 net lines**.
Most of the apparent volume was the diagnostic text itself, which the new path
must retain, rather than duplicated control or representation. Extrapolating
the remaining MOI emitters no longer credibly clears the 1,000-line thesis and
would spread exception semantics through dozens of interfaces. All production
scaffold was therefore removed. The exact prototype source was preserved
outside the repository only as temporary forensic material; no exception
contract or parallel failure path survives in ConSan.

### 14.6 Full hypothetical-mode and cross-axis pressure

The fifth-mode probe was extended from an identity-only compile check to a
vertical executable path. The temporary `Hypothetical` MOI identity reused
Record/Replay mechanics deliberately: the experiment was measuring how a new
mode implementation plugs into common lifecycles, not pretending that copied
toy semantics would reveal useful sharing. The mode was registered as an
independent identity and driven through:

- object and resource planning;
- many-access lowering, relocation, and final validation;
- native-LDS access and workgroup-barrier evidence;
- scalar/vector group-flat address materialization;
- report layout and current-header validation;
- host snapshot representation, typed decode, static owner attribution,
  conflict analysis, and rendering; and
- RDNA3, RDNA4, and CDNA5 targets.

The executable pressure suite passed 28 engine-conformance cases after adding
the fifth engine. A separate host pipeline case proved that the fifth identity
retained its own report-layout identity while using the selected
Record/Replay decoded and analysis alternatives; it also exercised disjoint
static-owner attribution and the renderer's typed-analysis trust check.

The production edit ledger was only seven common files and eight substantive
lines outside a hypothetical mode package:

1. one public report-ABI enum value and one canonical spelling;
2. one capability-engine projection;
3. one `MoiModeOperations` registry entry;
4. one raw report-header admitted-identity bound;
5. one host decoded-variant selection case; and
6. one host analysis-variant consistency case.

Request validation, observation planning, evidence planning, access and
synchronization orchestration, resource placement, mutation, report-region
planning, and final patch validation required no new mode switch. The report
renderer already dispatches on the decoded typed alternative and needed no
identity edit. The repeated host cases are not a mode-sized lifecycle: they
are the two explicit trust-boundary projections from an untrusted ABI identity
to a closed decoded variant and from that variant to its analysis type.

The cross-axis case then temporarily restored the gfx1151/RDNA3.5 target from
Section 14.3 and lowered both an LDS access and a workgroup barrier in the
fifth mode. It passed with no mode-by-target source, switch, callback, or
adapter. The required target edits were exactly the already-measured target
profile/analysis registrations plus the lower RocJitsu instruction-builder
admission; there was no additional edit attributable to the pair.

All hypothetical identities, target admissions, test fixtures, and production
aliases were removed after measurement. This completes the mandatory extension
pressure exercises and is strong negative evidence against mode/target axis
fanout as the next macro deletion source. The remaining common switches are
narrow identity-to-type trust boundaries, while the transformer composes its
axes through normalized operations. A later macro-slice must therefore be
justified by a duplicated semantic representation or lifecycle, not by adding
another registry or callback layer around these seams.

### 14.7 Rejected branch-only deferred-owner lifecycle slice

A whole-production-call-site audit found that the branch-only relay router
still implemented deferred storage-owner materialization even though the sixth
refactoring had removed its last production producer. No production caller
used the deferred-owner `offer` overload or constructed an LDS-reservoir owner.
Direct reservoirs were already selected and paid for before their relay words
were offered, and `BranchOnlyDirectRelayReservoirSet::reservoir_by_relay` was
the authoritative relay-to-reservoir ownership map.

The hard prototype removed the complete stale lifecycle rather than merely its
unused overload:

- owner kinds, identities, and paid/deferred materialization state;
- owner grouping and activation-cost search;
- the owner-aware exact-solver objective and lower-bound proof;
- owner carry through exact-pair and greedy fallbacks;
- owner identity in offers, claims, endpoint retirement, and commit checks;
- duplicate direct-reservoir ownership tags, using `reservoir_by_relay`
  directly for selected and transitive dependencies;
- exact-batch optimization budgets and telemetry that existed only for the
  removed owner objective; and
- eighteen representation-specific tests, while retaining and adapting the
  routing, capacity, fallback, work-bound, provenance, direct-reservoir, and
  randomized oracle suites.

The prototype built successfully. All 72 retained branch-only router tests and
a strengthened architecture-boundary check passed. The complete nonphysical
ConSan matrix then passed 4,769/4,769 tests across all four engines and all five
emulated targets. An accidentally broad first selector started a few physical
gfx1201 rows; it was stopped immediately, the completed rows passed, and the
actual gate was rerun with `-LE physical` as required.

The result nevertheless missed the macro economics contract. It removed 701
net physical production lines and 670 nonblank production lines, but only
**575 net production implementation lines**. The difference was mostly the
large obsolete public contract narrative. The surviving independent
relay-count optimizer was then traced separately. It is not deferred-owner
residue: on exact-pair fallback it preserves a feasibility baseline for later
pairs while retaining a shorter committed route, and focused plus randomized
tests demonstrate that behavior. Removing it would weaken live routing and
capacity behavior solely to cross the 750-line floor.

The full prototype was therefore reverted. This is negative knowledge for the
next search: deferred relay-owner materialization is genuinely dead and a
worthwhile later cleanup, but its complete production deletion is not a
seventh-refactoring macro-slice by itself. Do not retry it unless another
representation collapse in the same routing/reservoir lifecycle gives the
combined slice a credible four-digit payoff. In particular, do not relabel
the live relay-count optimizer as owner machinery or count deletion of its
tests as production shrinkage.

### 14.8 Macro-slice 2: reset the branch-only routing transaction

The two consecutive economics failures above trigger the whole-tree reset in
Section 10.8. That reset first examined the broad patch record, synchronization
evidence projections, validation, and the two SuperCollider access lowerers.
The patch record and validator retain independent proof facts. Synchronization
does reconstruct typed evidence plans during both resource planning and
emission, but the complete repeated builders contain only about 490 lines;
removing the joins would not clear the macro floor. SuperCollider LDS and FLAT
share owner/resource and placement vocabulary, but LDS additionally owns dense
call grouping, generated island banks, and branch-only continuations, while
FLAT owns a runtime group gate and multiword displacement. A common body would
therefore be a union-shaped protocol switch. The genuinely common mechanical
portion is presently too small to justify replacing both working lowerers.

The reset did, however, expose the missing companion to Section 14.7 in the
same branch-only routing lifecycle. Once deferred owners are removed, the
router's exact solver still carries the shape of the deleted optimization:
multiple operating modes, separate feasibility and bounded-improvement
passes, lower-bound bookkeeping, and detailed meters for a choice that no
production caller can make. The production contract is smaller: qualify a
bounded immutable relay inventory; find deterministic, vertex-disjoint
monotonic routes for a batch when possible; otherwise recover pair-atomically
without destroying capacity needed by later pairs; and commit only proven
claims. Paid direct-reservoir provenance is already authoritative outside the
solver and need not participate in route choice.

This macro-slice will replace that complete transaction rather than polish its
individual peepholes:

- delete deferred owner identities, materialization state, activation costs,
  owner grouping, and owner-affinity optimization as already proved safe by
  Section 14.7;
- collapse the residual exact solver to one feasibility backtracker with one
  deterministic work budget and no optimization-mode representation;
- retain the live per-pair relay-count minimization only where it protects
  later-pair feasibility, unless the replacement search makes that policy an
  inherent consequence of its ordering;
- retain qualification, monotonic reachability, disjointness, transactional
  rollback, provenance, reservation, and transitive direct-reservoir
  materialization as observable behavior; and
- strengthen tests around those behaviors while deleting tests that specify
  only the removed owner/optimizer representation.

The hard variants are a feasible batch on which naive request-order greedy
fails, bounded-search fallback, a nonfinal pair whose shortest route preserves
the next pair, and a transitive direct-reservoir route. The slice must preserve
those variants across RDNA4, CDNA4, and CDNA5 users and pass the full
nonphysical ConSan matrix. It forecasts roughly 575 already measured
implementation lines from the dead owner lifecycle plus at least 250 lines
from replacing the residual multi-mode exact-search transaction, with one
small feasibility solver in return. The acceptance floor remains **750 net
production implementation lines removed**. Abort and revert if preserving
complete-batch feasibility requires recreating owner or optimization modes, if
work remains unbounded, if live routing capacity regresses, or if the combined
result does not clear the floor.

The hard prototype accepted the thesis. The router now owns only relay
capacity, provenance, deterministic bounded route search, reservation, and
commit. Direct-reservoir ownership remains solely in `reservoir_by_relay`.
Deferred owner kinds, identities, grouping, materialization, activation-cost
search, and duplicate ownership tags are gone. The exact-pair fallback retains
the feasibility baseline for every nonfinal pair and still refines to a shorter
subset when its bounded work permits, but feasibility and refinement now share
one exact-search transaction, configuration, meter, and failure channel. The
separate optimizer mode, invariant state, limits, telemetry, and tests no
longer exist.

Fresh accounting is:

| Signal | Baseline | Accepted slice | Change |
| --- | ---: | ---: | ---: |
| Production files | 311 | 311 | 0 |
| Physical production lines | 97,460 | 96,544 | **-916** |
| Nonblank production lines | 91,055 | 90,187 | **-868** |
| Production implementation lines | 83,481 | **82,719** | **-762** |

The production diff has 173 added and 1,089 deleted physical lines. Nineteen
tests tied only to deferred-owner or independent-optimizer representation were
removed. The 71 retained router tests cover global backtracking, entry/return
contention, shortest-route capacity preservation, bounded exact and greedy
fallback, transactional rollback, deterministic randomized comparison against
a brute-force oracle, pristine reservation, provenance, and transitive direct
reservoirs. The architecture-boundary test now rejects reintroduction of both
parallel owner state and a second optimizer lifecycle.

The focused 71-test router suite and the architecture-boundary test pass. The
complete nonphysical ConSan gate passes **4,768/4,768** tests at `-j16`, across
Record/Replay, Sampled, InlineShadow, and SuperCollider on RDNA3, RDNA4, CDNA3,
CDNA4, and CDNA5. No physical gfx1201 test was run. The slice therefore clears
the macro floor without reducing product scope or the live routing-capacity
behavior that caused Section 14.7's narrower prototype to abort.

### 14.9 Macro-slice 3: delete transient pipeline observability

A fresh consumer trace found that the public nine-entry pipeline-stage array
had no production consumer. The lowerer maintained four pass counters solely
to synthesize that array; automatic prepare and resume then mutated the array
to describe work whose durable products already expressed the actual state.
Likewise, `TransformResult` retained the address-free
`ConSanEvidenceIntentPlan` even though it is only the transient input used to
derive the mode-owned evidence requirements. Runtime code consumes the
requirements, never the intermediate plan.

The accepted replacement makes the durable products authoritative:

- `TransformResult` owns the code-object identity, semantic inventory and
  coverage ledger, mode-owned evidence requirements, dispatch requirements,
  replacement, outcome, and diagnostics;
- one compact `ConSanContractIssue` retains the machine-readable request,
  capability, or binding rejection cause without manufacturing a stage
  record around it;
- automatic deferral is recognized from complete evidence requirements that
  actually require binding, and resume validates those requirements before
  invoking the mode lowering strategy;
- the evidence-intent plan remains a local planning value and is not copied
  into the lasting result; and
- the unused `ThroughObservationPlan` partial-lowering state is gone. The one
  live partial boundary is the semantic program inventory used before runtime
  evidence binding.

Tests that specified enum ordering, stage statuses, or internal execution
counts were removed. The retained pipeline tests assert outcomes, evidence
contracts, observation ownership, binding rejection causes, deterministic
direct/resumed equivalence, replacement bytes, coverage, and dispatch
requirements. `TransformResult::well_formed()` now explicitly rejects a valid
observation without its derived evidence requirements, preserving the
cross-product invariant previously checked through the persisted intermediate
plan. The architecture-boundary test rejects reintroduction of the deleted
stage or lowerer-execution representations and rejects a persisted evidence
intent plan in the result contract.

Fresh accounting is:

| Signal | Before slice | Accepted slice | Slice change | Campaign change |
| --- | ---: | ---: | ---: | ---: |
| Production files | 311 | 311 | 0 | 0 |
| Physical production lines | 96,544 | 96,114 | **-430** | **-1,346** |
| Nonblank production lines | 90,187 | 89,786 | **-401** | **-1,269** |
| Production implementation lines | 82,719 | **82,381** | **-338** | **-1,100** |

The production diff for this slice has 153 added and 583 deleted physical
lines. The focused 32-test pipeline suite passed, followed by the complete
nonphysical ConSan gate: **4,766/4,766** tests at `-j16` across all four modes
and all five emulated targets. No physical gfx1201 test was run. This is not a
new macro-floor result by itself, but it is accepted as consolidation directly
enabled by the prior subsystem replacement: it deletes an entire parallel
lifecycle and leaves one result authority rather than polishing individual
stage peepholes.

### 14.10 InlineShadow atomic-emission ownership checkpoint

The post-pipeline physical-boundary audit found one material discrepancy
between the documented mode locality and the actual tree. The shared
`consan_moi_sync_emission.cpp` still owned roughly 2,350 lines of
InlineShadow-only atomic release, acquire-token, causal-snapshot, and nested
EXEC-mask emission. Its public header consequently imported an InlineShadow
emission plan and exposed the InlineShadow atomic body builder.

That protocol now lives under `modes/inline_shadow/` behind its own narrow
header. The shared synchronization emitter retains only multi-mode intent
commit publication, target-neutral resource sizing and semantic projection,
and scalar-clause mutation. The architecture-boundary test now rejects a
return of the causal transaction or its emission plan to the shared owner and
continues to enforce the single address-hash and EXEC-mask mechanisms inside
the mode owner.

This is an architectural checkpoint, not shrinkage credit: splitting the
translation unit adds one small header and does not delete implementation.
Its value is to put InlineShadow access, atomic, and causal protocols beside
one another so the next comparative read can distinguish a removable shared
state machine from superficially similar instruction emission. The complete
641-test nonphysical InlineShadow-focused selection passed across all five
emulated targets; the updated architecture-boundary test also passes.

### 14.11 Rejected decoded-program inventory collapse

The post-locality read first compared the decoded CFG, owner-scope, liveness,
descriptor, and maximum-register-reference setup in the SuperCollider LDS and
FLAT lowerers. Those two paths do reconstruct the same pristine code object,
but a common SuperCollider inventory would remove only roughly 200--300
implementation lines. It would also have to reconcile two intentionally
different scalability contracts: LDS restricts CFG construction to preflight
candidates in very large generated objects, while FLAT needs the complete
donor and owner view.

Widening the comparison to all ConSan CFG construction did not improve the
thesis. Program analysis can reuse its graph internally and already does so;
MOI resource placement decodes the staged image including preapplied code
ranges; final validation must independently decode pristine and replacement
bytes; and SuperCollider has its own bounded large-object view. Persisting one
graph across those epochs would make it stale or weaken the independent proof
boundary. A small SuperCollider-only cache is still possible, but it does not
meet the macro floor and is not accepted as a seventh-refactoring slice.

### 14.12 Macro-slice 4: make SuperCollider an access-placement consumer

The more important SuperCollider duplication is below its replay semantics,
not above them. The native-LDS and FLAT implementations currently own two
complete access-lowering transactions totaling 5,100 physical lines. Each
rediscovers admitted sites and NOP caves, constructs scratch and owner plans,
enumerates inline/local/appended/indirect/branch-only candidates, runs shared
placement and relay machinery, grows descriptors, mutates the code object,
and publishes patch and coverage products. The three MOI modes already use
the common access-placement mechanisms for both native-LDS and FLAT sites;
SuperCollider is the remaining mode that treats those mechanics as its own
protocol.

The replacement thesis is one SuperCollider access action consumed by that
common placement boundary. SuperCollider continues to own replay, compare,
runtime group gating, delay, mismatch evidence, and the exact guest-state
preservation needed by those bodies. Common placement owns site admission,
owner-complete resource qualification, cave and entry-island discovery,
direct and indirect routing, transactional descriptor growth and mutation,
patch publication, and rejection coverage. Native-LDS and FLAT remain typed
body variants inside the mode rather than options in common code. Processing
both origins in one transaction also removes the current sequential
LDS-then-FLAT reservation and patch-budget reconciliation path.

The hard prototype is a shared-function FLAT site requiring a far appended
body and scalar/VGPR preservation on CDNA5, followed by a native-LDS site in
the same object on RDNA4. Those cases exercise multi-owner liveness, runtime
group gating, entry relocation, branch-only or indirect reachability,
descriptor growth, and the combined-origin budget. Dense LDS generated-island
routing and the large-object bounded-CFG case are attacked before simple
inline placement is migrated.

The complete old frontier is the 3,226-line LDS and 1,874-line FLAT bodies plus
their special composition branch. The replacement is expected to retain
roughly 1,800--2,500 lines of mode-owned body construction and special dense
LDS routing, while deleting the duplicated planning, selection, commit, and
publication lifecycles. The acceptance target is **at least 1,500 net
production implementation lines removed**, with the normal 750-line abort
floor. Abort and revert if the common boundary needs a SuperCollider switch,
an origin-policy callback table, or a union of LDS/FLAT candidate state; if
the existing MOI placement cannot express the dense and bounded variants
without recreating a second transaction; if exact diagnostics or patch-budget
selection change; or if independent final validation would need to trust the
new producer.

The deep prototype read rejected this thesis before adding a second placement
path. Every apparently duplicate SuperCollider state has a live producer and
consumer. Native LDS owns generated dense-call groups, local and generated
island banks, direct-relay reservoirs, max-flow selection, and branch-only
spill continuations. FLAT owns a runtime group-address gate, a different
direct/local/appended placement order, scalar VCC preservation, and a
different indirect-entry contract. They already share the low-level placement,
relocation, reservoir, descriptor-growth, and patch-publication mechanisms
that have the same semantics.

Making either path consume the MOI access transaction would therefore require
the forbidden union of SuperCollider-only candidate and routing state. A
smaller common decoded owner/CFG context would remove only about 200--300
implementation lines and would reproduce Section 14.11's incompatible
large-object epochs. There is also no combined-origin budget to delete:
SuperCollider deliberately applies LDS and FLAT as successive mutations, and
the second pass must inventory the first pass's current image. The proposed
1,500-line deletion was consequently based on similar nouns rather than a
shared lifecycle. Macro-slice 4 is rejected; no production scaffold was added.

### 14.13 Rejected InlineShadow versioned-publication collapse

The physical-locality checkpoint made the two largest InlineShadow emitters
directly comparable. Exact-access shadow publication, atomic release
publication, causal-snapshot capture, and acquired-token publication all use
versioned global slots, but they do not implement one transaction with
different payload callbacks.

- exact-access publication claims one slot per serialized address group,
  validates byte provenance and a complete prior dispatch, retries bounded
  contention, and retains only successfully committed cross-owner priors for
  conflict diagnosis;
- release publication claims one communication-address slot, journals guest
  address and EXEC state across the guest atomic, imports acquire evidence,
  and publishes a bounded inherited causal frontier;
- acquired-token publication reserves up to five destination slots as one
  all-or-nothing operation, rolls every journal back if any member loses, and
  then commits direct and inherited edges in a second phase; and
- causal-snapshot capture is a stable multi-entry read and canonicalization
  protocol rather than a writer transaction.

Only the mechanical compare-swap claim, wait, and even-version commit fragments
are alike, totaling a few hundred lines across more than 2,400 lines of live
protocol. Abstracting those fragments would leave the validation, cardinality,
rollback, retry, and evidence state machines in place while adding callbacks
for register aliases and EXEC-mask ownership. This fails both the 750-line
economics floor and the rule against a union-shaped protocol. The colocated
files are now truthful ownership boundaries, but locality alone is not evidence
that their distinct correctness protocols should be merged.

### 14.14 Rejected caller-owned/automatic report-layout collapse

The next report-lifetime trace tested whether caller-owned raw report buffers
and automatically allocated reports were parallel implementations of one
layout policy. They are not. A caller-owned raw buffer is a public
code-object- or executable-lifetime contract. It has no persisted layout
object, admits buffers as small as the common header plus the selected direct
records, and preserves the historical fixed-record behavior. An automatic
allocation carries its exact validated layout for the executable lifetime and,
for Record/Replay, provides a report-wide dispatch directory plus a bounded
owner/site/address-group identity table.

The distinction is observable all the way through lowering. Direct
Record/Replay elects one representative lane, assigns a fixed record per
static range, coalesces waves within a dispatch/workgroup, and fails closed
when its sole publication slot is incomplete. Automatic Record/Replay
serializes distinct address groups, retains exact wave identity and lane
masks, probes collision chains, retries incomplete or mismatched identities,
and records independent dispatch-directory and owner-bank saturation. These
paths consequently require different persistent state and different scratch
and EXEC-save resources. Sampled and InlineShadow likewise construct
mode-specific direct layouts for raw buffers while automatic sizing is driven
by durable evidence requirements.

Synthesizing an automatic layout for every raw buffer would reject currently
valid small buffers, change their lifetime requirement, and change which
executions are represented. Treating the direct layout as a degenerate bank
would retain nearly every direct-versus-banked branch in the 1,420-line
Record/Replay access emitter: identity construction, lane selection,
collision qualification, retry, saturation, and restoration remain genuinely
different. The common report header, indexed-address, atomic, wait, store, and
layout-validation mechanics are already shared. This candidate therefore
offers only a few hundred lines of possible mechanical extraction and cannot
meet the macro floor without reducing the public evidence contract. No
production scaffold was added.
