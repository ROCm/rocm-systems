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

### 12.1 Thesis A: one executable MOI access program

The first hard prototype targets the three access lowerers in Record/Replay,
Sampled, and InlineShadow. They currently own separate outer lifecycles of
1,545, 1,053, and 1,348 physical source lines respectively. Their evidence
bodies are genuinely different, but a phase/operation read found the same
lowering transaction around those bodies:

| Phase | Record/Replay | Sampled | InlineShadow | Proposed owner |
| --- | --- | --- | --- | --- |
| admit and order sites | replay order and capacity | stride, range and bank capacity | exact-shadow access order | mode policy produces admitted site requests |
| bind resources | owner SGPRs, scratch, spills, private owner | owner SGPR/VGPRs, scratch, spills, private epoch | owner SGPR/VGPRs, scratch, spills, private/workgroup shadow | shared planning session resolves common resources; mode leaf requests its widths and storage |
| place | inline or appended, dense, indirect, branch-only | same | same, plus explicit relay-bank prefix | shared site lifecycle and route handle |
| construct probe | access record | sampled window | exact shadow | typed mode leaf returning native probe words and guest boundary |
| construct entry | direct anchor, island, dense call or borrowed entry | direct anchor, island/runtime gate or dense call | direct anchor, dependency-wait island or dense call | executable entry program with typed gate words as a leaf result |
| construct body | preservation, bank transition, probe, relocated guest | same | same plus deferred guest/continuation ordering | shared body program and assembler |
| continue | direct branch, indirect jump, dense return, borrowed return | direct branch or long return with empty-wave guard | direct/long/call return with dependency wait and deferred guest | small continuation instruction algebra |
| commit | descriptor transaction, byte replacement, patch geometry, intent attribution, relay publication | same | same | shared access-program executor |

The proposed stable vocabulary is `site`, `entry`, `body segment`, `guest
boundary`, `continuation`, `resource effect`, and `commit`. Cardinality,
padding, selectable-bank transitions, empty-wave guards, and continuation kind
become data. Runtime gates and evidence construction remain narrow typed leaf
mechanics. Evidence meaning, report indexing, exact-shadow policy, and
candidate admission remain mode-owned observable distinctions.

The first hard pair is Record/Replay's borrowed/branch-only dynamic-record
path and InlineShadow's deferred-guest branch-only/dense paths. They force the
program to express different guest boundaries, relay ownership, entry forms,
return ABIs, spill guards, and dependency waits. Sampled is the third consumer
and tests that the vocabulary handles windowed evidence and runtime gates
without enumerating modes in the executor.

The bounded deletion target is the three mode-owned descriptor/apply/commit
tails and their complete inline/appended split, followed by the repeated
routing-session setup and common per-site resource/placement transitions.
Expected retired code includes `apply_inline_moi_access_patches`, the three
mode-local byte mutation loops, three descriptor-commit loops, three
anchor/island/relay commit loops, mode-local continuation switches, and the
old complete local patch plans once planning also emits the normalized
program. The preliminary physical estimate is 2,350--2,750 old orchestration
lines, 650--900 new protocol/executor lines, and 1,500--2,000 net deletion.

Abort conditions are: a mode must retain its complete old apply lifecycle
behind a callback; the common program needs a mode discriminator; typed leaf
code must perform byte publication or descriptor mutation; the two hard paths
cannot preserve exact guest ordering, EXEC/SCC state, branch reachability, or
transactional failure; or measured net deletion after all three migrations is
below 1,000 lines with no already-bounded second deletion. Focused tests must
cover inline and appended placement, branch-only and dense routing, scalar and
vector spill, high selectable-VGPR banks, deferred loads/stores, runtime
workgroup gates, stale-size/reachability corruption, and committed runtime
mapping. The complete nonphysical ConSan matrix is required at the second hard
cutover and after removal of the old paths.

### 12.2 Probe-program cutover checkpoint

The first hard cutover now exists in commits `e1a500ac151` through the current
checkpoint. Record/Replay borrowed and branch-only bodies, Sampled runtime-gate
and spill bodies, and InlineShadow deferred-guest and dense bodies all use one
continuation algebra and one compiled `MoiAccessPatchProgram` executor. The
previous inline-only callback transaction and the separate inline/appended
application lifecycles are gone; even a wholly inline transformation uses the
same tentative `.text`, descriptor, intent-commit, relay, and publication
transaction.

Focused hard-path tests pass. The complete 4,765-test nonphysical ConSan gate
ran in 210.99 seconds with 4,764 passes and only the architecture-boundary test
failing because an intermediate convenience helper accepted the aggregate
`ConSanTransformArtifacts`. The helper was removed immediately and that test
then passed alone. This is still a prototype checkpoint, not an accepted
macro-slice: the new protocol and the deleted inline/appended duplication are
approximately break-even in governing lines. The next deletion frontier is
the three mode-local compilers that still independently translate their
planned access patches into the common executable program.

### 12.3 Thesis B: bounded versioned-slot publication

The post-cutover audit found a wider deletion frontier below the access
program: Record/Replay automatic records, Sampled causal windows, InlineShadow
exact slots, atomic release slots, acquired-token sets, and causal snapshots
each privately emit a versioned publication state machine. Together their
containing emitters account for more than 11,000 physical source lines. The
payloads differ, but each lifecycle performs a bounded subset of the same
operations: derive a slot set, take a stable even snapshot, distinguish empty
from replaceable state, claim an odd reservation, retry contention, qualify
prior identity, journal state across a guest boundary, write a typed payload,
commit an even version, account for losers or saturation, and restore EXEC.

The proposed protocol owns those transitions and names slot cardinality,
replacement policy, retry budget, guest boundary, and payload schema as data.
Typed leaf emitters retain address hashing, identity qualification, payload
field production, and evidence meaning. They may not retain their own
claim/retry/commit loop. The hard prototype pair is the one-slot exact-shadow
replacement transaction and the multi-entry acquired-token-set transaction:
they differ in cardinality, canonicalization, predecessor qualification,
rollback, and evidence payload. Atomic release is the third consumer because
it forces a guest RMW and causal import inside an outstanding reservation.

The bounded scope includes the complete publication lifecycles in
`consan_moi_inline_shadow_emission.cpp` and
`consan_moi_inline_atomic_emission.cpp`, then the Record/Replay and Sampled
publishers if the hard pair validates the algebra. The preliminary target is
2,300--3,000 retired implementation lines for 700--1,000 common protocol and
typed schema lines, or 1,500--2,000 net deletion. Abort if payload callbacks
still contain claim/retry/commit control flow, if the protocol needs a mode
switch, if exact dispatch and version semantics cannot be retained, or if the
complete hard-pair migration forecasts less than 1,000 net lines after actual
prototype measurements.

### 12.4 Bounded-claim hard-pair checkpoint

Exact-shadow replacement and atomic-release publication now enter one shared
`append_moi_bounded_version_claim` lifecycle. The common authority owns the
eligible-lane restore, retry counter, stable-candidate invocation, odd CAS,
winner mask, finite sleep/retry loop, exhaustion, and final claimed EXEC.
Each consumer retains only its stable-snapshot and replacement predicates.

The exact-shadow consumer deliberately gave up its bespoke cheap first read
and cache-refresh re-entry path. It now takes the same coherent atomic first
version read on every attempt as atomic release. This is the intended bounded
representational slack: exact shadow no longer has the locally optimal retry
shape, but both domains have one claim protocol and unchanged evidence
semantics. The architecture-boundary test now pins that shared authority.

The focused exact-publication and versioned-release tests pass, as does the
architecture-boundary test. This checkpoint is not an accepted macro-slice:
the common protocol plus its two first adapters is still approximately
line-neutral. Commit/rollback, multi-slot cardinality, typed payload stores,
and loser accounting remain mode-local and are the deletion frontier for the
rest of Thesis B.

### 12.5 Access-program recipe checkpoint

The three materially different access compilers now hand one
`MoiAccessProgramRecipe` to `compile_moi_access_program`. Record/Replay keeps
its borrowed-entry and relay facts, Sampled keeps its runtime-gate evidence,
and InlineShadow keeps its exact-evidence leaf, but none independently
reconstructs appended anchors, entry islands, body placement, displaced or
deferred guest continuation geometry, relocated-guest attribution, or inline
patch geometry. The architecture-boundary test requires all three consumers
and rejects any return of the old assembly/finalization sequence to mode code.

The focused InlineShadow, Sampled, Record/Replay, FirstLight, dynamic-access,
automatic-record, branch-only, and many-access tests pass. This cutover removes
48 physical production lines net, and the complete 4,765-test nonphysical
ConSan matrix passes. It is useful protocol groundwork but is far below the
macro-slice threshold. It must therefore feed a larger replacement of the
parallel mode-local access planners rather than be counted as an
Eighth-refactoring result in its own right.

### 12.6 SuperCollider mutation-transaction checkpoint

The SuperCollider LDS and FLAT lowerers no longer choose between a raw ELF-byte
mutation path and a `CodeObjectPatcher` path according to whether a selected
site needs appended text. Inline, local-cave, and appended-cave patches now all
submit descriptor effects and tentative `.text` through the same transaction.
The FLAT transaction also consumes a replacement already published by the LDS
pass, so composed access lowering has one active-image lifetime instead of
silently restarting from the pristine object. A focused composition test
caught that requirement during the cutover, and the architecture-boundary
test now prevents either SuperCollider lowerer from restoring its raw-byte
descriptor path.

The focused LDS, FLAT, combined-access, dense-route, target-conformance, and
architecture-boundary tests pass. This deletes 72 governing production lines
net. As with Section 12.5, the result is accepted groundwork rather than the
required macro-slice: it collapses one duplicated application lifecycle but
does not yet remove the large specialized placement planners that feed it.

### 12.7 Accepted macro-slice: whole-text SuperCollider access programs

Commits `56604905910` through `ec4e1e6c8ac` turned the Section 12.6
transaction groundwork into the campaign's required lifted replacement. The
old LDS and FLAT lowerers did not merely have different probe bodies. Each
owned a complete parallel lifecycle for local-cave discovery, appended cave
placement, dense-call grouping, entry-host displacement, indirect return,
branch-only relay allocation, descriptor mutation, byte publication, patch
geometry, and owner-local CFG/liveness reconstruction. Those lifecycles are
gone.

The deliberately smaller protocol is now:

1. a mode leaf selects a source-coordinate site, owners, exact resource
   effects, and a complete site-local instruction program;
2. `stage_consan_text_rewrites` applies descriptor effects transactionally and
   retains typed rewrite intent without publishing partial executable bytes;
3. perturbation and the other SuperCollider access leaf may compose while all
   facts still use pristine source coordinates;
4. `relocate_consan_text` performs one structure-preserving identity
   translation, replacing each selected instruction inline in the relocated
   program; and
5. `finalize_consan_text_rewrites` validates every translated placement and
   atomically publishes the image, committed lowerings, patch geometry, and
   relocation proof.

LDS and FLAT are materially different consumers. LDS includes two-address
expansion, D16/subword placement, selectable-bank transitions, accumulator
boundaries, and bracket-local spill allocation. FLAT includes runtime group
address gating, VCC preservation, address/data tuple rules, and readback of a
global operation. Those remain typed mode-leaf mechanics. Neither consumer
owns cave topology, routes, continuation ABI, byte mutation, final placement,
or publication anymore. `ScOwnerAnalysis` also gives them one lazy owner,
CFG, liveness, register-extent, descriptor, dynamic-stack, and scratch
analysis instead of reconstructing that graph twice.

The intentional representational slack is explicit. Every transformed object
retains its pristine source text as an unreachable coordinate-stable prefix,
and a one-site rewrite pays for the same object-wide relocation protocol as a
many-site rewrite. The protocol can transport additional already-patched code
ranges even though a pure access-only object has none. This is larger at
runtime than a fortunate local NOP cave, but it removes all reachability- and
workload-dependent cave orchestration while preserving exact executable
behavior. The target profile, rather than common relocation code, now owns
any same-revision selection required by the structural translation.

Retired representations and authorities include the SuperCollider dense-route
identity and target operations, dense and indirect route fragments, local and
appended patch-kind distinctions, per-site placement planners, dispatcher and
entry-host maps, relay-bank offers and retirement, route-specific validation,
and the LDS/FLAT copies of owner CFG/liveness analysis. Comment-only filename
tombstones remain where the no-file-deletion operating rule requires them;
they contain no compatibility implementation.

Fresh accounting from `4ac57f3a827` (immediately before the Section 12.6
groundwork) through `ec4e1e6c8ac` is:

| Signal | Before | After | Change |
| --- | ---: | ---: | ---: |
| Production files | 314 | 316 | +2 shared relocation owners |
| Physical production lines | 95,966 | **91,888** | **-4,078** |
| Nonblank production lines | 89,600 | **85,652** | **-3,948** |
| Governing implementation lines | 82,090 | **78,287** | **-3,803** |
| Git physical insertions/deletions | - | 979 / 5,056 | **-4,077** by diff accounting |

The one-line difference between snapshot and Git physical accounting is an
end-of-file newline. Governing counts use the campaign's lexical counter and
are obtained by applying its exact differential to the canonical baseline;
the counter's absolute parser convention is unchanged on both snapshots.
There are no host additions and no new target implementation. Target code
loses 104 physical route/profile lines. The two new shared relocation files
cost 318 physical lines; all other insertion cost is the smaller staged
transaction, shared owner analysis, migrated leaf assembly, proof, and tests.

The focused regression set covered LDS-only, FLAT-only, combined LDS+FLAT,
perturbation composition, inline and spill-backed sites, dense and long-range
routes, all five targets, target conformance, invalid relocation geometry,
growth limits, and architecture boundaries. The old route-specific tests were
removed only where their represented mechanism ceased to exist; behavioral
and corruption tests remain.

### 12.8 Fresh whole-tree operation-matrix audit

After the macro deletion, the production tree was reread by lifecycle rather
than searched for matching names. The table records the largest remaining
operation families, including the source size around each apparent frontier.
Size is a navigation signal, not an estimate of removable code.

| Family and present physical envelope | Shared outer operations now | Remaining specialized operations | Eighth-refactoring verdict |
| --- | --- | --- | --- |
| MOI access planning and application: 3,465 mode-planner lines plus 1,187 common access-program lines | candidate/resource products, dense-route selection, body/continuation algebra, compiled program geometry, descriptor/text transaction, attribution and commit | mode evidence construction, spill/guest-operand interaction, runtime gates, report identity/cardinality, deferred-guest ordering | The hard three-mode prototype was attempted and removed the former apply lifecycles. Further field or loop lifting still leaves each evidence state machine and measured near break-even. No second bounded 1,500-line deletion remains in this protocol. |
| Versioned InlineShadow publication: 4,815 mode-emitter lines plus 214 common protocol lines | EXEC partition, coherent version loads, odd/even transition, bounded claim/retry/exhaustion | exact-cell provenance and conflict diagnosis versus release/acquire causal import, token-set cardinality, journaling, rollback, payload stores and evidence | The maximally different exact-slot/release hard pair was attempted. Common claim mechanics survived, but widening the algebra moves the two complete semantic state machines behind callbacks or a union. Measured checkpoint remained approximately line-neutral. |
| Synchronization lowering: 3,661 mode lines plus 2,011 common barrier/sync lines | decoded sequence association, owner/resource facts, patch occupancy, generic barrier body/route mechanics, typed commit/publication | Record/Replay event logs, Sampled causal windows and metadata, InlineShadow exact epochs and atomic ordering | Similar control-flow verbs conceal different evidence and failure transitions. The large 2,563-line Sampled owner is one integrated barrier/atomic causal protocol, not a duplicate of the smaller record or exact-epoch transactions. Common placement and native emission are already singular. |
| Persistent-state prologues: 2,919 lines | dispatch/workgroup capture, runtime selection, initialization suffixes, descriptor effects and publication | register persistence/borrowed-entry routing versus private offsets, dynamic-frame bootstrap, save/restore and private extent | The hard register/private comparison remains controlled by Section 14.16 of the seventh ledger. The common prefix and suffix already survive; lifting the storage-specific middle produces a union plan and is forecast below 350 lines before framework cost. |
| Shared resource and placement solver: 5,227-line implementation fragment plus contracts | one object-wide CFG/liveness/resource authority already serves all three MOI engines and every supported target | target capabilities enter as data; mode demand enters through the mode operation registry | This is a large singular common implementation, not parallel domain orchestration. Splitting or wrapping it cannot satisfy semantic compression. |
| Program/synchronization analysis and independent validation: 1,953 + 2,691 + 3,712 lines | normalized decoded inventory and synchronization graph; final proof separately reconstructs executable facts | validator independence, corruption diagnostics and fail-closed proof | Producer and validator intentionally have separate trust domains. Making validation consume mutable producer plans would delete code by weakening an invariant, so it is out of bounds. |
| Host runtime and registries: 5,099-line main hook plus separated report/config owners | one interception layer coordinates reader, executable, replacement, report, dispatch and fault lifetimes | distinct keys and retirement events for each HSA object type | The registry hard comparison still finds no duplicated outer lifecycle: unifying records retains the maps and adds optional states. File size reflects one host coordinator, not an `N*M` family. |
| Report regions and policy/intents | common ABI bounds, typed layouts, coverage ledger and render composition | caller-owned versus automatic identity/capacity/lifetime; policy reasons versus executable intent and independent proof | The hard comparisons in seventh Sections 14.14--14.15 remain valid. A generic schema or aggregate view preserves both representations and adds translation. |
| SuperCollider LDS/FLAT: 2,099 lines including their common region | one owner analysis and one whole-text transaction | the two irreducible resource selectors and site-local probe builders | Accepted macro-slice. There is no surviving route, placement, or publication compatibility path to harvest. |

One potentially large future direction did surface: make whole-text relocation
the mutation backend for all MOI access, synchronization, and prologue
programs. It is not a bounded missed hard-pair lift in this campaign. MOI
currently applies access, barrier, atomic, fence, and entry-prologue mutations
incrementally, and later phases consume active-image coordinates and prior
patch geometry. Relocating only access sites would leave later commits and
runtime mappings referring to the unreachable source prefix. A correct cutover
must migrate every mutation family together, define composition for coincident
and multi-instruction sites, and remap every committed lowering and proof.
That is a replacement of the complete MOI mutation backend, not another
adapter around the present access protocol. Its deletion potential is real,
but it needs its own measured transaction and rollback contract before it is
bounded; beginning it here would turn a completed deletion campaign into an
open-ended dual-backend state.

No other audited family has a credible unattempted 1,500-line bounded
replacement. The large remaining files are either singular shared
authorities, independent proof, or typed semantic leaves below mechanisms
that are already common. The audit therefore finds no comparably strong
unattempted lifting candidate appropriate to this campaign. It also records
the whole-MOI relocation backend explicitly so a later campaign does not
mistake this conclusion for evidence that the current mutation strategy is
optimal.

### 12.9 Campaign accounting and completion audit

After moving structural-translation revision selection into the CDNA5 target
profile, the final production inventory is:

| Signal | Eighth baseline | Final | Campaign change |
| --- | ---: | ---: | ---: |
| Production files | 313 | **316** | +3 |
| Physical production lines | 96,100 | **91,892** | **-4,208** |
| Nonblank production lines | 89,764 | **85,656** | **-4,108** |
| Governing implementation lines | 82,302 | **78,293** | **-4,009** |

The file increase is the shared access-program implementation and the two
shared text-relocation owners; each displaced private implementation was
removed in place because operational rules prohibit deleting its filename.
Tests, build files, generated code and documentation remain outside the
governing count. No implementation moved into an excluded category.

The completion requirements now have direct evidence:

1. the governing reduction is 4,009 lines, leaving 78,293, nine lines below
   the required maximum;
2. the whole-text SuperCollider protocol replaces the materially different
   LDS and FLAT placement/publication lifecycles and removes 3,803 governing
   lines by itself;
3. `stage_consan_text_rewrites`, `relocate_consan_text`, and
   `finalize_consan_text_rewrites` are the sole outer authority, with LDS and
   FLAT as real consumers and no route compatibility implementation;
4. the algebra is site, owners, descriptor effects, typed program, relocation,
   placement proof, and atomic commit--none of which enumerates a mode or
   architecture;
5. mode leaves remain under `modes/supercollider/`, architecture revision data
   is target-owned, and the boundary test rejects either fact escaping;
6. final validation remains independent, relocation and descriptor mutation
   remain fail-closed and transactional, and runtime evidence stays typed;
7. the malformed-ELF memory defect and HIP-fixture public-include defect found
   during the campaign have bounded regression coverage;
8. final nonphysical gate results are recorded below; and
9. Section 12.8 is the required fresh operation-matrix audit.

Validation checkpoint: a clean `ninja -j16` rebuild and
`ConSan.ArchitectureBoundaries` pass. The first full nonphysical ConSan gate
attempted 16 memory-heavy InlineShadow transforms concurrently and produced a
cluster of timeouts and resource-admission failures; that overloaded run is not
completion evidence. A resource-aware full ConSan gate and the full
nonphysical RocJitsu gate remain pending.
