# ConSan eighth refactoring: lift specialized domains into shared protocols

## 1. Status and starting point

The sixth refactoring proved that controlled subsystem replacement can shrink
ConSan substantially: it removed 4,507 governing implementation lines while
making target and mode ownership explicit. The seventh refactoring then used
those boundaries to search for the next duplicated representation or
lifecycle. It accepted two material deletions and one consolidation:

- the branch-only relay router lost its deferred-owner and independent
  optimization lifecycles, removing 762 implementation lines;
- the pipeline result lost its unused persistent stage-observability and
  intermediate evidence-intent representations, removing 338 lines; and
- test-only atomic-classifier and resource-plan projections were removed,
  saving another 79 lines.

The eighth-refactoring baseline is commit `2de6c685dfe`, at **82,302 governing
production implementation lines**. This is 1,179 lines below the seventh
refactoring's canonical 83,481-line baseline. The governing scope is:

- `lib/rocjitsu/src/rocjitsu/code/patch/consan/` and
  `lib/rocjitsu/src/rocjitsu/hooks/consan/`;
- production `*.cpp`, `*.h`, and `*.inc` files only; and
- tests, generated code, build files, documentation, comments, and blank
  lines excluded.

At this baseline the complete nonphysical ConSan gate passes **4,765/4,765**
tests at `-j16`, covering Record/Replay, Sampled, InlineShadow, and
SuperCollider on RDNA3/gfx1100, RDNA4/gfx1201, CDNA3/gfx942, CDNA4/gfx950,
and CDNA5/gfx1250. No physical test was run for the final consolidation.

The seventh refactoring was paused at this clean checkpoint. The eighth
refactoring supersedes its search criterion; it does not discard its accepted
architecture, tests, or negative evidence.

## 2. What the seventh refactoring discovered

The seventh campaign repeatedly found structures with similar nouns and
control-flow shapes. A deep read then showed that each had accumulated enough
domain-specific state and behavior that merging the existing structures
directly would produce a union of both or a callback wrapper around them.
Under the seventh-refactoring contract, that was a valid reason to reject the
candidate.

The most important results were:

| Investigated area | Apparent commonality | Difference that stopped direct sharing |
| --- | --- | --- |
| MOI access lowering | site admission, resource selection, preservation, placement, emission, and publication | each mode retained a different probe body, gate, guest boundary, return ABI, evidence policy, and fallback contract |
| SuperCollider LDS and FLAT placement | access ownership, liveness, scratch, caves, routing, mutation, and patch publication | LDS had generated dense-call groups, island banks, max-flow selection, and branch-only continuations; FLAT had a runtime address gate, displacement, VCC preservation, and different indirect entry |
| InlineShadow versioned publication | claim, inspect, wait, publish, retry, and commit of versioned global slots | exact access, atomic release, acquired-token sets, and causal snapshots differed in cardinality, journaling, validation, rollback, retry, and evidence payload |
| caller-owned and automatic report layouts | headers, bounds, mode records, report lifetime, and host decoding | direct and banked layouts had different identity, minimum-capacity, collision, saturation, retry, and lifetime contracts |
| observation decisions and probe intents | admitted sites and the semantic operation to instrument | decisions retained exclusions, reasons, aliases, and policy coverage; intents retained only coalesced executable placement and evidence demand |
| register-backed and private-backed entry prologues | dispatch/workgroup capture, runtime selection, initialization, restoration, descriptor growth, and patch publication | the storage media required different persistence, spill, offset, snapshot, routing, and descriptor transactions |
| decoded inventory and final proof | instruction, owner, address, patch, and evidence facts | the validator intentionally reconstructs facts independently rather than trusting the producer's mutable plans |
| host registries | executable, replacement, allocation, report, reader, and dispatch identities | the registries had different keys and retirement events, so one aggregate object would initially retain the maps as optional internal state |

The seventh ledger contains the detailed evidence:

- Section 14.5: a fail-fast MOI emission transaction removed only 48 lines in
  its hard prototype because diagnostic text, not propagation structure,
  dominated the apparent repetition;
- Sections 14.7 and 14.8: a narrow branch-router deletion initially missed
  its economics floor, but a wider replacement of the whole routing
  transaction succeeded;
- Section 14.11: decoded program inventory and mutable current-image epochs
  are distinct lifetimes rather than duplicate containers;
- Section 14.12: the proposed SuperCollider access-placement consumer failed
  because it tried to preserve both specialized placement protocols inside a
  common product;
- Section 14.13: versioned publication protocols shared mechanical fragments
  but had incompatible specialized state machines;
- Section 14.14: direct and automatic reports exposed different observable
  storage and evidence semantics;
- Section 14.15: policy decisions and executable intents served different
  consumers and lifetimes; and
- Section 14.16: register and private prologues had already shared their
  identical prefix and suffix, leaving storage-specific transactions.

This is genuine negative evidence against shallow merging. It is **not**
evidence that the current domains deserve perfectly tailored end-to-end data
structures. ConSan was developed largely bottom-up. Its local structures may
describe each current case precisely because they grew alongside that one
case, not because their boundaries are the best system architecture.

The diminishing returns came from continuing to ask whether the existing
structures were already the same. The eighth refactoring asks whether the
domains can be deliberately re-expressed through a smaller common algebra.

## 3. New mandate: semantic compression by lifting

The direction of travel is:

> Replace locally exact, domain-shaped orchestration with a deliberately
> smaller shared protocol. Make existing domains conform to that protocol,
> tolerate bounded representational slack, retain exact observable behavior,
> and delete the displaced end-to-end implementations.

This is a stronger mandate than field deduplication or helper extraction. It
permits changing component boundaries and replacing multiple locally natural
data structures with one heuristic system structure even when that shared
structure is not the ideal bespoke representation for any single consumer.

The analogy is a universal file descriptor. A file descriptor does not model
every property of a disk file, socket, pipe, terminal, or device in one exact
record. It gives all of them a deliberately constrained identity, lifetime,
operation, and readiness protocol. Specialized implementation remains below
that protocol. The loss of local tailoring buys composition and removes the
need for every subsystem to own a complete parallel lifecycle.

For ConSan, “heuristic” describes the fit of the **representation**, not the
correctness of the sanitizer. It is acceptable for a shared request to carry a
field unused by one implementation, for a common state machine to admit more
states than one mode needs, or for a specialized fast path to be expressed as
a generic one-element transaction. It is not acceptable to approximate
addresses, ordering, ownership, liveness, evidence, validation, or fail-closed
behavior.

The default question is no longer “are all present fields and transitions
identical?” It is:

1. What is the smallest useful operation algebra shared by these domains?
2. Which differences can become data, capabilities, cardinalities, or
   implementation-local primitives rather than separate outer lifecycles?
3. Which differences are observable semantics that must remain explicit?
4. Once both domains use the shared protocol, which complete specialized
   orchestrators, plans, adapters, and tests disappear?

A candidate may not be rejected merely because its present structures differ.
The hard prototype must first attempt to express those differences through the
proposed common algebra. Rejection requires measured evidence that the lifted
contract either cannot preserve observable behavior or cannot delete enough
specialized machinery to justify its cost.

## 4. A protocol, not a universal bag of fields

The desired result is not one giant `variant`, a structure containing every
mode's optional fields, or a forest of callbacks that merely hides the old
pipelines. Those preserve the product of all existing designs while adding a
framework.

A successful lifted protocol has these properties:

- a small stable vocabulary of operations and lifecycle states;
- one authority for ordering those operations and enforcing their common
  invariants;
- opaque or typed domain payloads only at narrow leaf operations;
- domain implementations that provide irreducible mechanics rather than
  recreating orchestration;
- capabilities that describe supported operations without embedding mode
  policy in targets or target decoding in modes;
- a resolved handle or result that common downstream code can consume without
  rediscovering its domain; and
- fewer total states, transitions, translations, and ownership sites than the
  specialized implementations it replaces.

Some representational slack is intentional. A one-slot transaction and a
five-slot transaction may use the same bounded transaction object. A
register-backed location and a private-memory location may use the same
storage handle even though their concrete emitters consume different fields.
That is acceptable when common code owns the shared lifecycle and the unused
portion stays small and well-defined.

Slack becomes failure when the common layer switches on every original domain
at each transition, when most fields are meaningful to only one consumer, or
when adapters translate immediately back into the old complete plans. Such a
result has not lifted the domains; it has embedded them.

## 5. Candidate lifted protocols

These are promising forms of semantic compression, not fixed scope. The
correct protocols and their boundaries remain variables to be discovered.

### 5.1 Probe program

A normalized probe program could describe:

- operations before the guest instruction;
- the guest execution boundary;
- operations after the guest instruction;
- runtime predicates and execution-mask regions;
- persistent and temporary resource demands;
- evidence reads, publications, and conflict actions; and
- commit, failure, and proof obligations.

Record/Replay, Sampled, InlineShadow, and possibly SuperCollider would produce
or lower compact semantic recipes instead of owning separate end-to-end
admission, resource, placement, preservation, guest-ordering, publication, and
commit pipelines. Mode code would retain its irreducible evidence operations;
common code would own the execution lifecycle.

This revisits the failed monolithic MOI access transaction with a materially
different thesis. The old attempt wrapped existing mode plans in callbacks.
The lifted version must replace those plans with a smaller instruction or
operation algebra and delete their orchestration. If it cannot do that, the
old negative result still controls.

### 5.2 State-storage handle

A shared state request and resolved storage handle could express:

- execution owner and lifetime;
- scalar/vector shape, width, alignment, and alias restrictions;
- persistence and initialization requirements;
- register, private-memory, LDS, or report-backed storage class;
- resolved location and descriptor effect; and
- save, restore, or relocation obligations.

Register and private prologues would become storage implementations beneath a
common state-placement and initialization lifecycle. Mode code would request
semantic state rather than understanding every physical carrier. The shared
handle may be less locally elegant than the current specialized plan, but it
must prevent callers from reconstructing storage-specific orchestration.

### 5.3 Publication transaction

A bounded publication protocol could lift the common algebra of:

- address or identity hashing;
- slot-set selection and cardinality;
- claim and ownership validation;
- stable inspection;
- journaling around the guest boundary;
- conditional update;
- atomic commit, rollback, retry, and saturation; and
- evidence publication.

Exact shadow access, atomic release, acquired-token publication, sampled
records, and causal snapshots need not use identical payloads or cardinality.
The experiment is whether those become parameters and leaf operations of one
transaction, allowing their separate control state machines to disappear.

### 5.4 Report-region handle

A report-region contract could combine identity, bounds, indexing strategy,
capacity, lifetime, completeness, and typed payload access. Caller-owned and
automatic reports could become two allocation/indexing policies beneath one
device-to-host region lifecycle. It is permissible for a caller-owned report
to use a generic one-bank or fixed-index policy if that preserves its public
behavior and enables deletion of a separate path.

Mode decoding and analysis remain typed. The universal part should end where
evidence meaning begins; otherwise this becomes the previously rejected
generic field-schema DSL.

### 5.5 Instrumentation-site lifecycle

A site handle could carry stable semantic identity through discovery,
classification, admission, resource binding, placement, emission, commit, and
proof. Policy exclusions and executable intents may remain distinct views of
one site lifecycle rather than separate retained collections. The goal is not
to concatenate their fields, but to stop each phase from building and joining
its own identity and outcome authority.

Independent validation must still reconstruct untrusted facts from the final
image. A shared site identity may connect producer intent to proof obligation;
it may not turn the producer's verdict into proof.

### 5.6 Other protocols remain in scope

The physical tree may reveal a better lifting point in patch transactions,
synchronization edges, owner identity, runtime registry lifetime, fault
mutation, or another component. The candidate list must not become premature
fixed scoping. Prefer the abstraction that deletes the widest repeated outer
lifecycle, even if it moves boundaries not anticipated here.

## 6. Discovery method: search for a common algebra

The first audit should read corresponding end-to-end responsibilities across
modes and site kinds. For each candidate family, construct an operation matrix
rather than a field-similarity table:

- lifecycle phases presently owned by each domain;
- operations performed in each phase;
- ordering constraints among those operations;
- resources and identities consumed or produced;
- failure, retry, rollback, and saturation transitions;
- observable semantic differences; and
- code that exists only because each domain owns the outer lifecycle.

Then propose the smallest algebra that can express at least two hard domains.
The algebra is evaluated by compiling the existing domains into it, not by how
perfectly its types describe either original implementation.

The audit must distinguish:

- **common orchestration**, which moves into the protocol;
- **policy data**, which is selected once and consumed by the protocol;
- **leaf mechanics**, which remain mode- or architecture-owned;
- **observable distinctions**, which remain explicit; and
- **accidental local precision**, which may be discarded.

Raw token, clone, or field counts may navigate the code but cannot establish
the algebra. The decision requires reading producers, consumers, mutations,
failure paths, and final proof.

## 7. Macro-slice execution protocol

### 7.1 State a lifting thesis

Before substantial implementation, record:

- the specialized outer lifecycles to be removed;
- the proposed common operation vocabulary and owner;
- which differences become data, capabilities, cardinalities, or leaf
  operations;
- which observable distinctions remain outside the protocol;
- every old plan, orchestrator, adapter, switch, field, and
  representation-specific test expected to disappear;
- estimated old implementation, new protocol and adapter cost, and net
  deletion;
- the two hardest vertical paths that will prove the abstraction; and
- behavioral, corruption, and differential tests plus abort conditions.

An initial candidate should forecast at least **1,500 net governing
implementation lines removed**. A result below **1,000 net lines** normally
fails as an eighth-refactoring macro-slice unless it demonstrably enables a
second already-bounded deletion in the same protocol.

### 7.2 Attack two hard domains first

Prototype the new protocol on two cases whose present structures differ most,
not on two nearly identical easy cases. Suitable pressure pairs include:

- InlineShadow exact publication and acquired-token-set publication;
- CDNA5 register/private dynamic-stack state and RDNA4 branch-only register
  state;
- automatic banked Record/Replay and a minimal caller-owned report;
- Sampled synchronization and InlineShadow atomic ordering; or
- SuperCollider dense LDS and displaced FLAT access placement.

These examples do not prescribe the first slice. The chosen pair must force
cardinality, lifetime, placement, failure, and architecture differences into
the proposed algebra early.

### 7.3 Permit temporary imperfection, forbid permanent duality

During a hard prototype the new protocol may initially be larger, less
specialized, and coexist with an old path. Safe intermediate commits are
encouraged. Before the macro-slice is accepted, however:

- every in-scope producer and consumer has migrated;
- the old outer lifecycles and translation adapters are gone;
- no mode immediately reconstructs its retired complete plan;
- tests assert behavior and the new boundary rather than preserving the old
  representation; and
- fresh accounting includes every new common, mode, target, host, and support
  line.

An additive framework is not a checkpoint. It either converges to deletion in
the same macro-slice or is reverted.

### 7.4 Test at semantic cutovers

Use focused unit and differential tests while migrating. Run the complete
nonphysical ConSan matrix when the second hard domain is live on the protocol
and again after the old path is removed. Run the complete nonphysical RocJitsu
matrix before accepting a macro-slice. A physical gfx1201 run is reserved for
an infrequent milestone whose behavior cannot be adequately covered by the
emulated matrix.

Any bug discovered during the refactoring is fixed immediately and receives a
regression test, even if the enclosing prototype is later reverted.

### 7.5 Reassess globally after deletion

After each accepted replacement, repeat the operation-matrix audit over the
whole tree. Do not polish the new subsystem by inertia. Select the next
protocol based on the widest remaining specialized orchestration and measured
deletion frontier.

## 8. Rules against false unification

1. Difference between current structures is not, by itself, an abort reason.
   First test whether the difference can be represented by the shared algebra.
2. Do not build a union containing every retired structure or a mode switch at
   every common transition.
3. A domain adapter may implement an irreducible leaf operation. It may not
   retain the old end-to-end orchestration behind a generic interface.
4. Do not introduce a generic framework whose only durable result is cleaner
   syntax. It must cause whole plans, passes, state machines, or registries to
   disappear.
5. Count all common scaffolding, adapters, payload definitions, tests moved
   into production, and changes outside the immediate files against the net
   result.
6. Do not shift implementation into templates, generated code, build scripts,
   tables, tests, or documentation to improve the governing count.
7. Representation slack must be bounded and named. A field or state with no
   coherent meaning across the protocol's consumers is evidence of an
   embedded domain, not acceptable slack.
8. Preserve direct typed leaf APIs where a universal protocol would otherwise
   require unsafe casts, string keys, untyped byte payloads, or unchecked
   callbacks.
9. One shared protocol must have at least two substantive consumers before it
   is accepted. Prefer three when the third migration tests whether the
   abstraction generalizes beyond the prototype pair.
10. Delete compatibility paths for private ConSan internals as soon as their
    last consumer migrates.
11. Do not fill a campaign target with unrelated micro-deletions. Harvest dead
    code exposed by each replacement, but account for it with that replacement.
12. If a candidate fails, record whether its algebra was too small, too broad,
    semantically unsound, or economically weak. Revert its production
    scaffold and change the abstraction, not merely its spelling.

## 9. Fixed invariants

The mandate permits representational compromise, not behavioral compromise:

- preserve Record/Replay, Sampled, InlineShadow, and SuperCollider behavior;
- preserve gfx942/CDNA3, gfx950/CDNA4, gfx1100/RDNA3, gfx1201/RDNA4, and
  gfx1250/CDNA5 support;
- do not introduce a mode-by-target implementation matrix;
- keep raw ISA decoding and architecture mechanics in target owners, and mode
  policy and evidence semantics in mode owners;
- keep shared mechanisms singular rather than copying them into physical
  mode or target directories;
- preserve exact resource, liveness, alias, guest-state, ordering, and
  execution-mask behavior;
- preserve fail-closed behavior, bounded work, transactional mutation,
  rollback, report trust, and public diagnostics;
- preserve independent final validation of the transformed image and runtime
  inputs;
- preserve or strengthen behavioral and corruption tests when
  representation-specific tests are retired;
- build and test with the local TheRock toolchain at `-j16`;
- use nonphysical tests as the normal gate and physical gfx1201 only at a
  deliberate infrequent milestone;
- make frequent coherent local commits and never push; and
- never use `rm` or delete a file. Superseded implementation may be removed
  from a file, and an obsolete filename may remain as an implementation-free
  tombstone.

## 10. Accounting and completion

The campaign starts at **82,302 production implementation lines**. Every
accepted macro-slice records:

- physical, nonblank, and governing implementation lines before and after;
- gross specialized implementation deleted;
- new protocol, adapter, target, mode, host, and support implementation;
- net reduction;
- specialized lifecycles and representations eliminated;
- representational slack deliberately accepted;
- observable distinctions retained;
- tests added, strengthened, migrated, or retired; and
- modes and targets exercised.

The eighth refactoring is complete only when all of the following hold:

1. At least **4,000 net governing implementation lines** have been removed,
   leaving at most 78,302 lines, with no metric displacement or product-scope
   reduction.
2. At least one lifted protocol has replaced two or more materially different
   specialized lifecycles and removed at least **1,500 net implementation
   lines**.
3. Every accepted protocol has one surviving authority, at least two real
   consumers, and no compatibility adapter or hidden copy of an old complete
   plan.
4. The resulting shared algebra is smaller than the domains it replaced and
   can be explained without enumerating every current mode and architecture.
5. Mode and architecture locality remains truthful, with no new `O(N*M)`
   source structure.
6. Independent validation, fail-closed behavior, transactional mutation, and
   exact runtime evidence semantics remain intact.
7. Every discovered bug has a regression test.
8. The complete nonphysical ConSan and RocJitsu gates pass.
9. A fresh whole-tree operation-matrix audit finds no comparably strong
   unattempted lifting candidate appropriate to this campaign.

The 4,000-line target is a review floor, not an automatic stopping point. A
**7,500-line reduction** is the stretch objective. Continue beyond either
number while a stronger bounded candidate remains.

Failure of a candidate because present structures differ does not satisfy the
alternative exit. Stopping below the floor requires at least two serious
end-to-end hard prototypes that actually attempt to impose a common algebra,
plus concrete measurements showing that preserving observable behavior leaves
the shared protocol economically larger than the specialized orchestration.
Do not manufacture a universal abstraction when that evidence is reached, but
do not substitute another similarity audit for the required prototypes.

## 11. Executable goal

Carry out a lifting-driven structural refactoring of ConSan from the
82,302-line baseline. Discover deliberately constrained common operation
algebras beneath its locally tailored mode, storage, publication, placement,
report, and site lifecycles. Prototype each algebra on two maximally different
hard domains, make those domains conform to the shared protocol without
weakening observable semantics or independent proof, migrate the complete
bounded scope, and delete every displaced specialized orchestrator,
representation, and compatibility path. Measure all new framework and adapter
cost. Continue until at least 4,000 net governing implementation lines have
been removed, including one multi-domain lifted replacement worth at least
1,500 lines, and a fresh whole-tree audit finds no comparably strong remaining
candidate; or until the stricter measured alternative-exit evidence in
Section 10 is satisfied. Preserve all four modes, all five target
architectures, all safety invariants, and the full test gates. Make frequent
local commits and never push.

## 12. Execution ledger

Record lifting theses, hard prototypes, accepted replacements, rejected
algebras, exact accounting, and validation results here as the campaign
proceeds. A prototype is evidence, not progress, until its specialized old
paths have been removed.
