# ConSan sixth refactoring: controlled subsystem replacement

This document is the working charter and execution record for ConSan's sixth
refactoring. It deliberately changes the unit of progress from small boundary
repairs to subsystem-scale replacement and deletion.

The [fifth-refactoring document](FIFTH_REFACTORING.md) remains the historical
record of the component audit, 185 convergence checkpoints, operating rules,
and tests that made this change in risk posture possible. The sixth refactoring
does not repeat that ledger. It treats the boundaries and behavioral evidence
created there as infrastructure for larger surgery.

The destination design remains open. The direction, economics, safeguards, and
minimum evidence of progress are binding.

## 1. Why the operating strategy is changing

The fifth refactoring materially improved ownership, layering, mode locality,
architecture locality, and test coverage. Its ledger reported a reduction from
91,450 production implementation lines to 88,015 lines. The fresh commit-based
recount in Section 11 corrects that boundary to 88,023 lines. That small
bookkeeping discrepancy does not change the conclusion that the recent slope
was insufficient:

- checkpoint 174 removed 346 implementation lines in one deletion;
- the next eleven checkpoints moved from 88,035 to 88,015 lines, only 20 net
  lines of reduction; and
- the last four checkpoints through checkpoint 185 added 17 net implementation
  lines while improving locality.

At that velocity, substantial code shrinkage would take weeks of micro-slices.
Moving a mode body to a better file, adding a typed product, or closing one
peephole can still be correct work, but it is no longer sufficient evidence of
convergence.

The fifth refactoring bought safer seams and stronger tests. The sixth must use
them as permission for controlled subsystem replacement, not continue treating
each remaining peephole as its own project.

## 2. Executable goal

> Substantially shrink ConSan's production implementation by discovering and
> carrying out high-leverage structural replacements. Replace whole redundant
> or overgeneralized subsystem regions with a smaller, clearer design; migrate
> every affected consumer; and delete the displaced implementation in the same
> macro-slice. Preserve all supported behavior, all four modes, all five target
> architectures, validation strength, diagnostics, and test coverage. Use the
> fifth refactoring's component boundaries and test matrix as risk-control
> infrastructure, while allowing deeper evidence to move, merge, split, or
> discard the apparent component boundaries themselves. Continue until serious
> measured shrinkage has been achieved and a fresh whole-codebase deep read no
> longer identifies a higher-leverage structural replacement that should be
> undertaken before completion.

The immediate campaign has a hard **3,000-line net implementation-reduction
review floor** and a **5,000-line stretch milestone**, measured from the 88,023
line baseline. Reaching 3,000 lines is the earliest point at which completion
may be discussed; it is not by itself a completion condition. The broader
below-80,000 aspiration remains a direction of travel rather than permission to
trade away behavior or clarity.

## 3. Fixed constraints and variable being solved for

### 3.1 Fixed constraints

The following are not variables:

1. Production implementation must shrink substantially in net terms.
2. Existing supported behavior across Record/Replay, Sampled, InlineShadow,
   SuperCollider, gfx942, gfx950, gfx1100, gfx1201, and gfx1250 is preserved.
3. Existing tests continue to pass. Bugs discovered during replacement receive
   regression tests and owner-level fixes immediately.
4. Shared behavior remains shared; mode locality must not reintroduce copies.
5. Architecture-specific behavior remains physically identifiable and
   skippable; sharing must not scatter raw gfx knowledge back through modes.
6. Replacement includes consumer convergence and deletion. A second permanent
   authority is not an acceptable result.
7. Validation remains meaningfully independent of the implementation it
   checks. Shrinkage must not make validation circular or ceremonial.
8. Changes remain reviewable through frequent local commits. Nothing is ever
   pushed by the agent.

### 3.2 Structural scope is a variable

The refactoring is explicitly solving for its own best structural scope. The
following remain open to evidence:

- which subsystem boundaries should move;
- which apparent components should merge, split, or disappear;
- where duplicated control flow, state, validation, or target/mode composition
  actually resides;
- whether a smaller representation makes whole classes of logic unnecessary;
- which macro-slice has the best deletion leverage at the current point; and
- whether an initially promising candidate should be abandoned, combined with
  another region, or superseded by a newly discovered opportunity.

Direction is fixed; destination structure and work ordering are not. File size,
branch counts, similarity tools, and token searches may locate candidates, but
only a deep semantic read of definitions, callers, state, outputs, validation,
and runtime use may justify replacement or deletion.

## 4. Initial high-leverage hypotheses — not scope

The following candidates illustrate the scale and character of opportunity
sought. They are neither an exhaustive list nor a prescribed roadmap, and
their order here conveys no implementation priority.

### 4.1 SuperCollider address-space implementations

The LDS, flat, and common bodies currently account for roughly 6,400
implementation lines. A comparative read may find parallel control, resource,
patching, and reporting machinery that can be one algorithm with smaller
address-space policies. It may instead show that the two protocols are
fundamentally distinct. Either outcome is acceptable; file size alone does not
decide.

### 4.2 MOI placement

The placement body contains roughly 4,980 implementation lines and more than
500 conditional branches. It remains a dense intersection of lifetime,
liveness, routing, descriptor, spill, mode-demand, and target constraints. A
smaller normalized placement problem and result might replace substantial
procedural state. It also has a very large blast radius, so its potential must
be demonstrated rather than assumed.

### 4.3 Independent validation

The validation body contains roughly 4,075 implementation lines and more than
450 conditional branches. Shared traversal, declarative invariants, or a
smaller proof representation may eliminate repetition. Any such design must
preserve independent checking rather than reuse the producer's conclusions.

### 4.4 MOI emission

Access, synchronization, prologue, routing, and mode-specific emitters occupy
well over 10,000 implementation lines. A typed common executor might remove
several thousand lines, or it might become a universal abstraction that merely
hides the same complexity. This is a hypothesis to investigate, not a mandate
to build an emission framework.

New candidates discovered anywhere in production have equal standing. A new
opportunity should supersede these examples whenever it has stronger evidence
of structural clarity, deletable volume, and manageable behavioral risk.

## 5. Unit of work: the macro-slice

The unit of convergence is now a **macro-slice**: one coherent subsystem
replacement large enough to remove a material region of production code.

A macro-slice proceeds as follows.

### 5.1 Bound the investigation

Deep-read a small number of competing candidates with one purpose: identify
the old semantic authorities, the proposed smaller authority, every consumer,
the exact code made deletable, and a conservative net line estimate. This is a
bounded feasibility pass, not another general architecture audit.

Reject a candidate promptly when:

- fewer than roughly 500 genuinely redundant or superseded implementation
  lines can be identified;
- its apparent similarity masks materially different semantics;
- the replacement would need more general machinery than it removes;
- preservation cannot be tested convincingly; or
- a different candidate has clearly superior deletion leverage.

### 5.2 State the deletion thesis

Before significant new production scaffolding is added, record:

- what representation or algorithm replaces the old region;
- why the replacement is smaller rather than merely differently factored;
- the files, functions, fields, branches, adapters, and compatibility paths
  expected to disappear;
- the conservative gross deletion and new-code estimates;
- the focused and matrix tests that protect the cutover; and
- the condition under which the attempt will be abandoned.

This is a falsifiable working thesis, not a frozen destination design.

### 5.3 Replace, converge, and delete

Implement at subsystem scale. Temporary old/new coexistence is allowed only
inside the active macro-slice when it enables differential testing or a safe
cutover. Migrate all applicable consumers, remove the old path, remove the
temporary comparison machinery from production, and prune dead contracts and
includes before calling the slice converged.

Prefer a direct replacement over a long adapter staircase. The local commit
history supplies recovery points; production does not need to retain every
intermediate architecture.

### 5.4 Validate and account

Use focused tests throughout implementation and the full nonphysical matrix at
the cutover. Measure the complete production delta, including new contracts,
files, and dispatch machinery—not only the gross lines deleted. Record which
behavioral, mode, target, and independent-validation surfaces were exercised.

### 5.5 Reassess globally

After each macro-slice, briefly rescan the whole production codebase. Choose the
next candidate from current evidence rather than continuing mechanically in
the same directory. The structural opportunity being solved for remains open
throughout the refactoring.

## 6. Anti-plateau and code-economics rules

1. A locality-only move is not a convergence checkpoint unless the same
   macro-slice realizes its named deletion or consolidation payoff.
2. A new production abstraction must enable more same-slice deletion than it
   costs. Small net growth is not bankable as indefinite future potential.
3. A completed macro-slice should normally remove hundreds or thousands of net
   implementation lines. Single-digit deletion is cleanup within a slice, not
   a slice outcome.
4. If two consecutive buildable checkpoints on one candidate have realized
   less than 500 net lines and cannot enumerate a near-term larger deletion,
   stop and re-evaluate the candidate.
5. Do not optimize the metric by moving implementation into tests, generated
   files, data blobs, scripts, comments, or opaque metaprogramming. The
   surviving design must be easier to explain and maintain.
6. Do not achieve shrinkage by dropping supported modes, architectures,
   diagnostics, validation, or behavior unless the user explicitly changes
   product scope.
7. Repeated wrappers, projections, fallback paths, and duplicated validation
   discovered during a macro replacement are harvested before moving on.
8. The execution ledger is updated per macro-checkpoint, not per tiny commit.
   Frequent code commits continue without turning each helper edit into a
   separately documented project.

These thresholds are forcing functions against circling, not excuses to make
poor abstractions or delete necessary domain logic. If a numeric target and a
clear design conflict, expose the conflict and reassess the candidate rather
than gaming either constraint.

## 7. Testing and repository discipline

The operating rules inherited from `NEXT_HOST_HANDOFF.md` and the fifth
refactoring remain in force:

- use `-j16` for builds and nonphysical tests on this host;
- run focused owner-level tests during a replacement;
- periodically run all relevant ConSan and RocJitsu nonphysical tests, with a
  full nonphysical gate at every macro cutover;
- use `-LE physical` by default and run physical gfx1201 tests only at
  deliberately chosen, infrequent milestones;
- exercise all five emulated gfx targets for affected mode/architecture paths;
- add a regression test immediately for every existing bug uncovered;
- keep frequent local git commits, including safe intermediate recovery
  points;
- never push; and
- never delete a file or use `rm`. Remove superseded implementation from files,
  or leave an implementation-free tombstone when an obsolete filename cannot
  safely be removed.

The stronger testing discipline is what makes higher-risk replacement
responsible. It is not a reason to avoid replacement.

## 8. Baseline and measurement

The sixth-refactoring baseline is the tree at fifth-refactoring checkpoint 185,
commit `362683743db`:

| Signal | Baseline |
| --- | ---: |
| Production files | 310 |
| Physical production lines | 102,170 |
| Nonblank production lines | 95,756 |
| Production implementation lines | **88,023** |
| Test inventory | **5,424** |

Production implementation lines exclude tests, generated code, documentation,
comments, and blank lines, using the same accounting as the fifth-refactoring
ledger. Every macro-checkpoint records physical, nonblank, and implementation
line counts, but implementation lines are the governing size signal.

The ledger must distinguish:

- gross old implementation removed;
- new production implementation added;
- net implementation reduction;
- tests added or strengthened;
- behaviors, modes, and targets validated; and
- remaining or newly discovered high-leverage candidates.

## 9. Completion and continuation criteria

The sixth refactoring may not be declared complete merely because one candidate
was attempted, one subsystem was moved, all tests pass, or the 3,000-line review
floor was reached.

Completion requires a fresh post-refactoring whole-codebase deep read showing:

1. At least 3,000 net production implementation lines have been removed from
   the 88,023-line baseline, without scope loss or metric displacement.
2. One or more substantial subsystem regions were actually replaced and their
   old implementations deleted; the result is not an accumulation of local
   wrappers and file moves.
3. The resulting components and dataflow are clearer and easier to explain
   than the baseline, with no new mode-by-architecture multiplication.
4. Mode-specific and architecture-specific code remain physically local and
   skippable, while shared mechanisms have one implementation.
5. No material temporary adapter, duplicate authority, or abandoned rewrite
   scaffold remains in production.
6. All focused tests and the complete nonphysical RocJitsu/ConSan matrix pass;
   deliberately selected physical validation passes at an appropriate final
   milestone.
7. Every bug found during the work has a regression test.
8. The final deep read has reconsidered the whole codebase—not merely the
   initial four hypotheses—and either finds no clearly superior macro
   replacement that belongs in this campaign or explains concretely why the
   remaining candidates should be deferred.

If the 3,000-line floor is reached while credible high-leverage opportunities
remain, the refactoring continues. If a candidate fails, the refactoring
changes candidates rather than weakening the goal. If preservation of behavior
and substantial shrinkage appear genuinely incompatible, stop and present the
evidence; do not silently narrow product scope or declare the plateau complete.

## 10. Execution record

Macro-checkpoints will be appended here. Each entry records the deletion
thesis, structural result, gross and net size change, validation evidence,
unexpected bugs and regression tests, and the whole-codebase evidence used to
select the next macro-slice.

### 10.1 Macro-slice 1 (abandoned): one MOI access transaction

The bounded candidate comparison rejected two tempting file-size arguments.
SuperCollider LDS and FLAT lowering share mismatch reporting and low-level
scratch mechanics, but their bulk is different: LDS owns dense routing,
relay-reservoir convergence, multi-owner placement, and two-address replay,
whereas FLAT owns a smaller direct/local/appended placement problem. Merging
the two large functions would create a union-shaped framework without yet
identifying 500 deletable lines. The 2,542-line branch-only relay router also
does not currently admit the hoped-for replacement by one ordinary polynomial
flow solver. Its fixed source/destination pairs form capacity-sharing routing
commodities, and its deferred-owner objective is a group-activation cost, not
an additive edge cost. Its exact-batch, exact-pair, and greedy tiers preserve
different feasibility and bounded-work contracts covered by 90 focused tests.
It remains a candidate for a smaller representation, but no deletion thesis
currently justifies rewriting it.

The selected opportunity is the complete MOI access transaction shared by
Record/Replay, Sampled, and InlineShadow. A deep read of
`consan_moi_record_replay.inc`, `consan_moi_sampled_access.inc`, and
`consan_moi_inline_shadow.inc` found three parallel orchestration bodies around
the already-common `MoiPlannedAccessPatch`, dense-route, relay-reservoir,
descriptor-growth, appended-body, lowering-commit, and direct-patch contracts.
The modes genuinely differ in evidence selection, report allocation, probe
words, entry-gate details, return ABI, and runtime mapping. They do not differ
in ownership of the placement transaction that applies those products.

The deletion thesis is:

- replace the three mode-owned route/application skeletons with one shared MOI
  access transaction;
- make each mode supply typed planning and emission facets rather than a mode
  enum, a shared switch, or a copied end-to-end loop;
- move code-object/text setup, local-island and dense-route setup, direct-relay
  reservoir ownership, descriptor-requirement application, appended image
  initialization, anchor/island publication, inline byte replacement,
  lowering commit publication, relay commit, dense-host emission, and final
  text replacement behind that transaction;
- retain Record/Replay's banked records and borrowed entry, Sampled's static
  filter/windows and runtime gate, and InlineShadow's shadow layouts and
  deferred-guest ordering in their mode files; and
- delete the displaced per-mode orchestration immediately, including wrappers
  or duplicated state that become unnecessary after all three consumers
  converge.

The conservative target is at least 900 gross deleted production
implementation lines, no more than 300 new shared and mode-adapter lines, and
therefore at least 600 net deleted lines. The attempt is abandoned rather than
recorded as a macro-checkpoint if the common transaction needs an engine
switch, exposes one mode's evidence vocabulary to another, leaves old and new
application paths live, or cannot realize at least 500 net deleted lines after
the complete legacy harvest.

Focused validation covers common access-application contracts, each of the
three MOI access engines, dense and branch-only routing, spill and selectable-
VGPR-bank paths, lowering/runtime mapping publication, and the architecture
boundary. The cutover gate covers the complete nonphysical ConSan/RocJitsu
matrix, including all five emulated targets. Any behavioral discrepancy found
during convergence receives a regression test before the macro-slice closes.

The attempt was stopped at its first buildable comparative checkpoint. Moving
the descriptor, dense-route, relay, commit, and publication lifecycle behind
one typed transaction removed only 77 lines from the three mode bodies while
requiring 101 lines of shared template and 126 lines of mode adapters. The
remaining large regions were the intentionally different probe-body, entry-
gate, return-ABI, and evidence policies. Extending the transaction into those
regions would have required a union-shaped callback protocol rather than
making implementation disappear. The experiment was reverted in full; no
production wrapper or alternate path remains. This is the deletion thesis's
abort condition working as intended, and evidence against revisiting access
orchestration without a materially different representation.

### 10.2 Macro-slice 2 (complete): transactional instruction construction

The next bounded comparison found a wider replacement opportunity inside MOI
emission. The largest emitters still manually stage optional encoded
instructions, test each collection of optionals, and then append each word or
word range one at a time. They also manually calculate and patch many local
branch placeholders. This is not target or mode policy: it is a repeated,
fallible implementation of one instruction-stream transaction. The existing
`InstructionSequence` contract already expresses atomic `emit_all`, typed
labels, branch fixups, and rollback, but adoption is partial.

The surveyed emission files contain roughly 1,000 local `const auto`
temporaries, 490 checked optional-instruction appends, and 485 repeated failure
checks. Not every temporary is redundant, but the repeated construction
protocol spans InlineShadow, synchronization, Record/Replay access, Sampled
access/synchronization, record events, prologues, and shared helpers. That is a
single common mechanism repeated across mode-local algorithms, rather than a
request to merge those algorithms.

The deletion thesis is:

- make `InstructionSequence` the one transaction for composing instruction
  words and local branches from existing target builders;
- replace manual optional staging/check/append regions and hand-maintained
  local branch fixups with direct `emit_all` and typed labels;
- retain named intermediate instructions when their values are inspected,
  reused, or carry useful semantic meaning, and retain all mode algorithms and
  target builders in their current owners;
- avoid inventing a bytecode, table-driven mini-language, macro layer, or
  generated representation merely to reduce the line count; and
- harvest obsolete helper lambdas, temporary vectors, offset bookkeeping,
  checks, and includes as consumers converge.

The initial cut is the complete InlineShadow instruction emitter, followed by
the other large emitters only while the same representation continues to
remove real protocol. The conservative macro-slice target is at least 800 net
production implementation lines, with a 500-line abort floor. New common code
must remain small: the intended replacement is adoption of the existing typed
transaction, not a second emission framework. The attempt stops if direct
sequences obscure instruction semantics, change builder evaluation or
rollback behavior, require mode/architecture switches, or fail to clear the
floor after the high-density consumers are converted.

Focused validation compares exact emitted bytes and error behavior for the
converted owners, then exercises InlineShadow, Record/Replay, Sampled,
synchronization, spill, branch-fixup, and architecture-boundary tests across
all five emulated targets. The cutover gate is the complete nonphysical
ConSan/RocJitsu matrix. A behavioral discrepancy receives a regression test;
mere representational convergence with identical covered behavior does not
require duplicating existing exact-byte tests.

The cutover converged every high-density consumer for which the repeated
protocol was real. InlineShadow access, atomic, synchronization, and exact
shadow emission; Record/Replay access and event emission; Sampled access,
atomic, and synchronization emission; common prologue construction; runtime
workgroup gates; and SuperCollider FLAT check/trap bodies now use the same
transaction for optional builders and local control flow. `InstructionSequence`
gained only the two missing typed EXEC branch kinds. Existing target builders,
mode algorithms, resource plans, and externally placed text branches remain
in their owners.

The final whole-directory audit distinguishes the remaining branch arithmetic
from the deleted local fixup protocol. The surviving offsets connect already
placed text regions, returns, relay reservoirs, anchors, and externally shared
entries; validation independently reconstructs and checks those encodings.
Straight-line helpers in `consan_moi_support.cpp` append deliberately named
semantic fragments but neither stage fallible optional builders redundantly nor
patch local branches. Converting either class would hide placement semantics or
add a wrapper without deleting implementation, so it is outside this
replacement.

The production accounting from the `362683743db` baseline is:

| Signal | Baseline | Macro-slice 2 result | Change |
| --- | ---: | ---: | ---: |
| ConSan production files | 310 | 310 | 0 |
| Physical ConSan production lines | 102,170 | 100,517 | **-1,653** |
| Nonblank ConSan production lines | 95,756 | 94,112 | **-1,644** |
| ConSan implementation lines | 88,023 | 86,379 | **-1,644** |

The physical source diff contains 2,326 additions and 3,979 deletions in the
declared ConSan scope. The generic `InstructionSequence` owner outside that
scope adds 22 net implementation lines, so the conservative campaign result
including its full enabling cost is **1,622 net production implementation
lines deleted**. One 17-line direct unit-test extension pins the two new EXEC
branch encodings. No production mini-language, alternate emitter, adapter, or
old branch-fixup authority remains.

Focused gates passed after each owner conversion, including 222 prologue and
owner/epoch/dispatch tests, 192 InlineShadow/exact-shadow tests, 181 inline
atomic tests, 149 SuperCollider FLAT/check-trap tests, and the corresponding
Record/Replay, Sampled, synchronization, and instruction-sequence tests. At
the final cutover, the complete `-j16 -LE physical` gate executed 9,573 tests
with zero failures in 242.69 seconds; one registered test was disabled and the
expected environment-dependent skips were unchanged. The gate includes all
four modes and gfx942, gfx950, gfx1100, gfx1201, and gfx1250 emulation. No
physical gfx1201 test was run at this intermediate milestone.

No behavior defect was found during the replacement. Exact-byte and behavioral
coverage already exercised the changed paths, so only the genuinely new common
branch kinds needed new tests. This slice clears its 800-line target and makes
substantial progress toward the campaign floor, but does not satisfy the
3,000-line review floor or the fresh-audit completion criteria. Candidate
selection therefore returns to the whole production codebase rather than
continuing mechanically through low-density emission helpers.

### 10.3 Macro-slice 3 (completed): one authority for closed vocabularies

The post-cutover candidate comparison rejected four more apparent large-file
opportunities after semantic reads. The host report pipeline is already split
into snapshot, mode decoder, mode analyzer, and mode renderer; most of its
remaining volume is genuinely mode-specific evidence interpretation. Access
and synchronization emitters share routing and placement mechanisms but not
their entry, guest-replay, call-anchor, or evidence protocols; a common callback
transaction would be union-shaped. The branch-only relay router already has one
exact constraint solver reused by its batch and pair fallback paths, while its
greedy path is a separately bounded escape hatch with observable work-budget
semantics. Finally, `ConSanOptions` is a thin aggregate of six typed input
contracts: splitting its 53-file read surface would add projections while
deleting little. None supplies a credible 500-line replacement thesis.

The selected opportunity is the closed-vocabulary representation repeated
through ConSan's public contracts and host diagnostics. Many enums currently
have three separately maintained authorities: the enum declaration, a complete
iterable array, and a spelling switch. Other diagnostic enums omit the iterable
array but repeat the same switch shape. The observation contract alone has 92
such switch cases; request, capability, evidence, synchronization, result,
pipeline, and hook vocabularies add hundreds more. Adding a value currently
requires coordinated edits that the type system cannot connect, and every
spelling switch spends two or more implementation lines restating the value
inventory.

The deletion thesis is:

- replace each qualifying value-array/spelling-switch pair with one explicit
  typed vocabulary whose entries contain the enum value and stable spelling;
- make that vocabulary itself iterable as the enum values, so existing
  exhaustive contract tests and loops retain their direct shape;
- make name lookup and, where already applicable, parsing consume the same
  entries rather than reconstructing another switch or string table;
- keep semantic switches that compute policy, encoding, state transitions, or
  validation decisions; this replacement applies only to closed vocabulary,
  not arbitrary enum control flow; and
- converge all qualifying production consumers in the same macro-slice and
  delete every displaced spelling switch and parallel value array.

The conservative target is at least 650 gross deleted implementation lines,
no more than 150 lines of generic vocabulary support and migrated declarations,
and therefore at least 500 net deleted lines. The attempt is abandoned if the
typed vocabulary needs preprocessor generation, makes an enum harder to read,
weakens invalid-value handling, changes a stable spelling, or falls below 500
net lines after the complete qualifying family has converged. Focused tests
exercise exhaustive iteration, every stable spelling, invalid sentinels, and
existing parsers; the cutover gate remains the complete nonphysical RocJitsu
matrix across all four modes and five emulated targets.

The completed cutover introduces the 48-implementation-line
`ConSanEnumVocabulary` as the only generic mechanism. Twenty-one production
files now declare each qualifying value/spelling pair once. The same object is
the iterable value range and the spelling lookup, and the existing
case-insensitive flavor and MOI-engine parsers obtain their canonical spellings
from those objects while retaining their explicit aliases. No preprocessor
generation is involved. Switches that compute semantics remain switches;
mixed-format renderers whose result depends on detail beyond the enum value
(barrier-move decoder detail, grouped barrier-member rejection, and growth
policy values) are deliberately outside this vocabulary family.

The production accounting from the Macro-slice 2 result is:

| Signal | Macro-slice 2 | Macro-slice 3 result | Change |
| --- | ---: | ---: | ---: |
| ConSan production files | 310 | 311 | +1 helper |
| Physical ConSan production lines | 100,517 | 99,966 | **-551** |
| Nonblank ConSan production lines | 94,112 | 93,515 | **-597** |
| ConSan implementation lines | 86,379 | 85,778 | **-601** |

The production source diff contains 772 additions and 1,323 deletions. Most
additions are the retained value/spelling facts moved into their single typed
authority; the reusable mechanism itself is 48 implementation lines, well
below the machinery cap. The 601-line net implementation reduction clears the
500-line abort floor. Five focused test files preserve exhaustive iteration,
stable names, invalid sentinels, parser aliases, and all 83 hook-only
diagnostic spellings.

After the final checkpoint, the complete nonphysical RocJitsu gate executed
9,574 tests with zero failures; one test remained disabled and the normal
environment-dependent tests were skipped. The gate covered SuperCollider,
Record/Replay, Sampled, and InlineShadow across gfx942, gfx950, gfx1100,
gfx1201, and gfx1250 emulation. No physical-GPU tests were run. Macro-slice 3
therefore closes as a qualified subsystem replacement rather than a retained
cleanup experiment.

### 10.4 Macro-slice 4 (abandoned): decoded proofs instead of mirrored emitters

The post-vocabulary candidate comparison rejected three attractive but
low-leverage representation changes. `ConSanTransformArtifacts` is the
lowerer's mutable, proof-rich construction aggregate, while `TransformResult`
is the pipeline's reviewed public product; their only direct duplication is a
short move-publication boundary, and merging them would expose private lowering
state rather than delete a subsystem. Kernels and non-dispatchable functions
share decoded container facts, but their distinct descriptor, dispatch, and
ownership semantics leave only a small number of paired traversal lines to
remove. Finally, ConSan's target-operation packages already have thin common
dispatch and physically local concrete implementations. Combining their
registries would mostly move declarations and would not remove 500 lines.

The stronger finding is in final validation. Byte accounting, original-image
semantic reconstruction, patch association, graph ownership, and descriptor
comparison are independent proofs and must remain. In contrast, nine route,
entry, dispatch, exact-shadow, and release validators reconstruct complete
emitted instruction sequences using the same instruction-builder functions as
the producer and compare those reconstructed words to the replacement image.
The affected regions span roughly 1,600 source lines and contain 101 direct
builder invocations. This is both large and less independent than it appears:
an encoding defect shared by a builder and its mirrored validator can satisfy
the comparison.

The deletion thesis is:

- make the existing decoder and its normalized instruction/operand semantics
  the authority for proving emitted machine behavior;
- replace validation-side construction of expected instruction vectors with
  direct checks of decoded opcode meaning, register roles, constants, branch
  destinations, memory effects, and instruction boundaries;
- retain the existing independent patch-byte accounting, original-program
  reconstruction, route graph ownership, typed patch association, resource
  bounds, and whole-descriptor comparisons;
- use small direct semantic predicates, not a general instruction-pattern
  language, generated validation program, or opaque table DSL; and
- migrate every qualifying final validator and remove its mirrored builder
  imports and reconstruction helpers in the same macro-slice.

The conservative estimate is at least 1,050 gross implementation lines of
mirrored construction removed, no more than 450 lines of decoded-view and
direct semantic proof code added, and therefore at least 600 net implementation
lines deleted. The first bounded cut covers the complete indirect-route and
entry/prologue family. The attempt is abandoned and reverted if that cut
cannot preserve exact opcode/operand/branch corruption detection while still
projecting at least 500 net lines for the completed family, if target-specific
raw encodings leak back into common validation, or if a generic matcher starts
to obscure the machine behavior being proved.

Existing adversarial tests already corrupt indirect-island SCC preservation,
dense-call keys, branch-only routes, relay-reservoir geometry and displaced
payload, entry scalar backup bytes and resources, dispatch prologue state,
exact-shadow bodies, release transactions, and mutation composition. Focused
validation will retain those tests and add decoder-proof cases for any
previously unpinned opcode, operand, modifier, or branch-target distinction.
The cutover gate remains the complete nonphysical RocJitsu matrix across all
four modes and five emulated targets.

The completed experimental cutover migrated the entire named family: indirect
islands, entry scalar backups, dispatch prologues, branch-only continuations,
dense routes, relay donors and reservoirs, exact-shadow publication, release
transactions, mutation-generated control flow, perturbation delays, and
appended padding. The final-validation translation unit no longer imported an
instruction builder, and all 1,667 ConSan tests passed with the two expected
benchmark-object skips. The semantic result was stronger: producer and proof
no longer shared emitted instruction construction.

The deletion thesis nevertheless failed decisively. The complete production
source diff contained 1,324 additions and 707 deletions, a **617-line net
physical increase** instead of the projected 600-line decrease. Direct decoded
operand predicates and target normalization did not make the proof obligations
disappear; they restated those obligations at comparable or greater volume.
Meeting the 500-line abort floor would therefore have required finding more
than 1,100 further deletable lines after every qualifying mirrored builder had
already been removed. A generic pattern language could compress the spelling,
but would violate this slice's explicit anti-DSL condition and obscure the
machine behavior being proved.

The experiment was reverted rather than retained as an independence-only
improvement. No decoded-proof helper, target operation, builder-import change,
or alternate validation path remains. One existing fail-closed bug discovered
during the experiment is retained independently: corrupting every expected
release-version compare-exchange previously dereferenced an empty position
list. Final validation now reports the semantic failure, and the existing
release-transaction adversarial test exercises that case. The architecture-
local source filename created by the attempt remains only as an implementation-
free tombstone because this work session may not delete files; it stays in the
component manifest solely so the source-ownership gate remains exact.

The post-abort production accounting is:

| Signal | Macro-slice 3 | Post-abort result | Change |
| --- | ---: | ---: | ---: |
| ConSan production files | 311 | 312 | +1 empty tombstone |
| Physical ConSan production lines | 99,966 | 99,978 | +12 |
| Nonblank ConSan production lines | 93,515 | 93,526 | +11 |
| ConSan implementation lines | 85,778 | 85,780 | **+2 bug fix** |

The campaign therefore retains **2,243 net implementation lines of reduction**
from the 88,023-line baseline. Macro-slice 4 contributes no claimed shrinkage,
does not clear the 3,000-line review floor, and supplies evidence against
revisiting mirrored final-validation construction unless a materially smaller
proof representation is found. Candidate selection returns to the whole
production codebase.

### 10.5 Macro-slice 5: one direct relay-reservoir authority

The post-abort scan rejected the shared MOI access-application layer as a
deletion candidate. Its roughly 800 implementation lines are not an abandoned
adapter from Macro-slice 1: Record/Replay, Sampled, and InlineShadow actively
share its attribution, dense-host, anchor/island, preservation, descriptor,
and appended-body mechanics. Removing it would recreate those mechanics in
three mode owners. Deep reads of placement, synchronization analysis, and
fault injection also found large but predominantly distinct policy, rather
than a replacement with 500 identified deletable lines.

The stronger finding is that ConSan still has two implementations of a direct
relay reservoir. SuperCollider FLAT and the three MOI access engines use
`BranchOnlyRelayRouter::plan_direct_reservoirs`,
`BranchOnlyDirectRelayReservoirSet`, the common direct emitter, and the common
independent direct-execution-path validator. SuperCollider LDS instead owns an
older private subsystem inside its 4,000-line lowerer: it discovers overlapping
reservoir candidates, proves endpoint scratch registers, constructs an
indirect entry/return normal-execution path, iteratively ranks and replays
reservoir combinations against a max-flow cut, converges a second optimistic
branch-only routing loop, emits a different reservoir body, publishes a
SuperCollider-only route effect, and validates that separate representation.

This is not an essential LDS protocol distinction. A reservoir's semantic job
is to relocate a proven straight-line pristine instruction sequence, preserve
its ordinary execution, and expose its vacated words as capacity-one branch
vertices. The common direct reservoir already provides exactly that contract
with a simpler SOPP entry/return path; it needs no borrowed scalar state and is
already exercised by SuperCollider FLAT as well as every MOI access mode. LDS
still needs its own selection of protected ranges, relay demand, generated
islands, and probe-body placement, but it does not need another relocation
representation or execution mechanism.

The deletion thesis is:

- make `BranchOnlyDirectRelayReservoirSet` the sole direct-instruction
  reservoir representation for SuperCollider LDS, SuperCollider FLAT, and
  every MOI mode;
- let LDS supply its selected-container, synchronization/probe, preapplied,
  dense-host, and already-selected patch ranges to the common discovery
  transaction, while retaining LDS policy for when extra relay capacity is
  requested;
- reuse the common router ownership, transitive used-reservoir marking,
  transactional placement, recursive frontier extension, emitter, patch
  metadata, and direct-path validator;
- remove LDS endpoint scratch-register discovery, special indirect reservoir
  geometry, candidate/ranking/replay machinery, optimistic materialization
  loop, bespoke emission, descriptor-growth consequences, route-effect type,
  target helpers, and validation branch; and
- retain the genuinely different wide composite donor only if focused
  evidence shows that the common reservoir cannot replace its capacity. It is
  otherwise deleted in this slice as a second relocation mechanism.

The conservative estimate is at least 1,300 gross production implementation
lines removed and no more than 350 lines of common planner generalization and
LDS policy/adaptation, for at least 950 net deleted lines. This includes the
large LDS selection loops, not merely its small route-effect type. The attempt
is abandoned and reverted if the common planner must learn LDS probe or mode
vocabulary, if preserving coverage requires rebuilding the old indirect
reservoir as callbacks, if the cutover loses a supported extreme-pressure
case, or if complete convergence cannot realize at least 500 net deleted
implementation lines.

Focused validation covers the existing maximum-cardinality, wave32, wave64,
skipped-kernel, degraded-partial-route, owner-minimization, full-register-
pressure, and corrupted-reservoir SuperCollider tests, plus the common direct
reservoir transaction and all five emulated targets. Representation-specific
assertions may move from the removed borrowed-scratch route to the common
direct execution contract, but corruption detection and admitted-site
coverage may not weaken. The cutover gate is the complete nonphysical
RocJitsu matrix.

The cutover converged on that thesis without teaching the common router any
SuperCollider or LDS vocabulary. LDS now contributes only its protected-range
policy and its larger two-to-512-word donor bounds. The common direct-
reservoir transaction owns discovery, recursive placement, relay ownership,
transitive used-reservoir marking, emission, patch metadata, and validation.
The generated-island max-flow planner consumes those same reservoir words and
retires its raw relay claims before the later branch-only batch reuses the
remaining common inventory.

The displaced subsystem was deleted in the same slice. This removes LDS's
endpoint liveness and scratch-SGPR search, wave32-VCC and wave64 ordinary-SGPR
route variants, overlapping candidate windows, min-cut ranking and replay,
wide composite donor, optimistic promotion loop, indirect reservoir emitter,
special patch kind and vocabulary, descriptor-growth side effect, target slot
helpers, and independent special validation branch. There is no compatibility
adapter or second reservoir authority left. SuperCollider LDS, SuperCollider
FLAT, and all three MOI access modes now use the same direct relocation
mechanism; LDS-specific probe, dense-island, and max-flow policy remains local
to its mode owner.

The production accounting from the Macro-slice 4 post-abort result is:

| Signal | Macro-slice 4 | Macro-slice 5 result | Change |
| --- | ---: | ---: | ---: |
| ConSan production files | 312 | 312 | 0 |
| Physical ConSan production lines | 99,978 | 98,645 | **-1,333** |
| Nonblank ConSan production lines | 93,526 | 92,222 | **-1,304** |
| ConSan implementation lines | 85,780 | 84,536 | **-1,244** |

The production implementation diff contains 1,480 deleted and 236 added
lines, for the 1,244-line net reduction. This exceeds the 950-line thesis and
500-line abort floor. From the 88,023-line sixth-refactoring baseline, the
campaign has now removed **3,487 net production implementation lines** and
crossed the 3,000-line review floor. It has not reached the 5,000-line stretch
milestone, and crossing the review floor is not a completion condition.

The focused cutover gate passed 135 SuperCollider, LDS, relay-reservoir, and
common direct-reservoir tests. The complete ConSan host suite passed 1,666
tests with two optional benchmark-object skips. The full `-j16 -LE physical`
RocJitsu gate then ran 9,575 tests with zero failures; its one disabled test and
normal environment-dependent skips were unchanged. That gate exercises all
four modes on gfx942, gfx950, gfx1100, gfx1201, and gfx1250 emulation. No
physical gfx1201 test was run at this intermediate milestone.

Existing corruption tests now attack the common displaced sequence, direct
entry/return path, geometry, and unused-reservoir proof rather than deleted
scratch-route metadata. A new owner-level test pins caller-selectable donor
word bounds and their transactional rejection. No pre-existing behavior bug
was found during this replacement. Macro-slice 5 therefore closes as a
successful subsystem replacement, but the campaign remains active pending the
required whole-codebase reassessment and further high-leverage work.

### 10.6 Macro-slice 6: one dense MOI relay transaction

The review-floor reassessment returned to the whole production tree rather
than extending the direct-reservoir work mechanically. Exact-clone scans found
little meaningful copy-and-paste, and a semantic read of the largest remaining
files rejected several size-only leads. The host hook body is predominantly
distinct process-lifetime ownership, HSA interception, report collection, and
verdict policy. Final validation's independent byte, inventory, ownership, and
descriptor proofs remain necessary; the attempted decoded-proof replacement
already demonstrated that its large route validators cannot be collapsed into
the decoder economically. MOI access orchestration still has the genuinely
different mode evidence, gate, replay, and return protocols exposed by the
Macro-slice 1 experiment. Automatic MOI placement is large and procedural, but
its current volume is mostly different lifetime and capacity fallbacks; a
normalized allocator has not yet identified 500 lines that would disappear
rather than move behind a more general solver.

The stronger current finding is a repeated *mechanism*, not textual cloning.
Four synchronization paths independently implement the same dense-relay
transaction: Sampled barriers, Sampled atomics, InlineShadow atomics, and the
shared Record/Replay/InlineShadow barrier lowerer. Each path partitions sites
by container, resolves one owner-local scalar router, finds a liveness-safe
relocated host, reserves appended bodies and a dispatcher, preserves the
host's displaced instructions, emits an entry island, emits a keyed or
call-return dispatcher, rewrites route anchors, and publishes the same two
relay patch records. The paths already share `MoiDenseRouteSite`,
`MoiDenseRouterPlan`, target-normalized call forms, scalar-range proofs,
relocated-host search, indirect-jump builders, and patch kinds. Their repeated
control flow exists because those common products stop immediately before the
transaction that consumes them.

The semantic differences are bounded inputs to that transaction. A mode owner
chooses which sites are stranded, builds each evidence body, supplies its
anchor geometry and lowering commit, and decides whether a missing host skips
the group or falls back to direct lowering. Record/Replay barriers may first
extend an existing S_CALL_I64 access dispatcher. Sampled gfx1250 barriers use
clone-local route keys, and spill-backed Sampled atomics requalify their scalar
tuple at the late-discovered call anchors. InlineShadow atomics use the
explicit-key form instead of a scalar call. Those policies stay in their mode
owners; none justifies four implementations of host relocation, island and
dispatcher construction, transactional byte publication, or relay metadata.

The deletion thesis is:

- introduce one normalized dense-relay transaction whose input is a resolved
  router, owner and host proof, a host-return route, and a sequence of already
  planned semantic body routes;
- make that transaction own dispatcher reservation and encoding, displaced
  host return, entry-island encoding, anchor encoding, bounds checks,
  transactional byte publication, and the common host/dispatcher patch
  records;
- express the existing keyed, tagged-SCC, call-return, and clone-local forms as
  small route facts derived from `MoiDenseRouterPlan` and the target profile,
  rather than mode switches or callbacks that reconstruct each old path;
- leave site selection, scalar-tuple requalification, access-dispatcher reuse,
  evidence-body construction, mode diagnostics, lowering commits, and direct
  fallback in their present semantic owners;
- migrate all four synchronization consumers, and then migrate the existing
  dense-access emitter wherever the same normalized transaction removes more
  code without weakening its already-common placement boundary; and
- delete the Sampled-only dispatcher/patch helpers and every displaced local
  dense host, island, dispatcher, anchor, and publication loop in the same
  macro-slice.

The four synchronization implementations contain roughly 1,250 production
lines of repeated dense-relay mechanics, in addition to the roughly 250-line
common dense-access emitter. The conservative target is at least 1,000 gross
implementation lines removed and no more than 450 lines of normalized
transaction, route facts, and retained mode adaptation, for at least 550 net
lines deleted. Access migration is part of the slice only if it improves that
net result and leaves the access planning boundary clearer. The attempt is
abandoned and reverted if the transaction must accept mode enums, evidence
objects, or a union-shaped callback protocol; if mode owners cease to control
site admission and body semantics; if route validation becomes less
independent; or if complete synchronization convergence cannot clear the
500-line net abort floor.

Focused validation covers direct and far Sampled barriers and atomics,
InlineShadow atomics, Record/Replay and InlineShadow barriers, S_CALL_B64,
S_CALL_I64 clone translation, explicit and SCC-tagged keys, scalar-spill tuple
requalification, relocated host execution, access-dispatcher reuse, and exact
patch/lowering publication across the five emulated targets. Existing
exact-byte, corruption, resource-pressure, and architecture-boundary tests are
the differential oracle; any uncovered behavior defect receives a regression
test. The cutover gate remains the complete nonphysical RocJitsu matrix.

The attempt was stopped and reverted after complete synchronization convergence
and an additional access-emitter convergence both failed the stated economics.
The experiment did establish one normalized route record and one compiled
dispatcher/host/island/anchor transaction, and migrated Sampled barriers,
Sampled atomics, InlineShadow atomics, the shared Record/Replay/InlineShadow
barrier path, and the already-common dense-access emitter. The focused 79-test
dense/far routing matrix passed at the maximal checkpoint across S_CALL_B64,
S_CALL_I64, explicit-key, tagged-SCC, spill-backed, host-relocation, and all
five emulated-target cases.

That complete experiment changed six production files by 515 physical
additions and 665 deletions. The governing counter moved only from 84,536 to
84,389 implementation lines: **147 net lines removed**, less than one third of
the 500-line abort floor and far below the 550-line thesis. The unexpectedly
small result is substantive evidence. Most of the apparent 1,250-line region
is not repeated relay mechanics: it is different site admission, late scalar
tuple requalification, evidence-body construction, access-dispatcher reuse,
placement-planner interaction, lowering publication, and fallback policy. The
normalized transaction itself needed to represent call-return, derived-key,
clone-local, explicit-key, tagged-SCC, dependency-wait, direct-target, and
relocated-host variants. Pulling the remaining policy behind that boundary
would therefore add the mode-aware callbacks or union-shaped protocol that the
thesis explicitly forbids.

All three experimental code commits were reverted; no route record, dispatcher
helper, alternate emitter, adapter, or duplicated authority remains. The
post-revert build and the same 79 focused tests pass. Production returns to the
Macro-slice 5 total of **84,536 implementation lines**, so the campaign remains
at **3,487 net implementation lines removed** from baseline. This negative
result rejects dense synchronization routing as a current macro-replacement
candidate and returns selection to the whole production tree.

### 10.7 Macro-slice 7: one fault resolution and mutation transaction

The next whole-tree comparison again rejected large-file size as sufficient
evidence. Automatic MOI placement still consists predominantly of genuinely
different lifetime, capacity, spill, and owner-partition fallbacks. The
remaining SuperCollider LDS and FLAT bodies share low-level action and route
helpers, but their large regions are still different protocols: LDS owns
dense generated-island and multi-owner policy, while FLAT owns direct/local/
appended placement and runtime address-space gating. Synchronization analysis
constructs several related graphs, but barrier lifecycle, atomic ordering,
ordinary release/acquire, execution ownership, and move-destination proofs do
not currently expose 500 lines that one smaller graph operation would delete.

Fault injection has a stronger duplicate-authority boundary. Its roughly 470
implementation-line dry-run planner selects and semantically qualifies exact
barrier, atomic, LDS, and ordinary-memory objects, then serializes those
decisions through one wide optional-field `ConSanFaultMutationPlan`. A separate
roughly 1,087-line family of mutation functions immediately resolves the same
identities back into the same immutable inventory, repeats much of the
qualification, and emits bytes. A 124-line kind dispatcher recomposes related
atomic and ordinary plans, while a roughly 137-line `well_formed()` switch
proves that the wide optional payload happens to describe one of the admitted
shapes. The stale-plan boundary is locally robust and tested, but it protects
an internal handoff between two adjacent passes over the same pristine image;
it is not an independent final validation proof.

The deletion thesis is:

- replace planning followed by identity re-resolution with one private fault
  transaction that resolves each requested semantic target exactly once;
- let the same resolved transaction either publish only the existing dry-run
  diagnostic presentation or continue directly into mutation against the
  pristine image;
- keep kind-specific selection and encoding explicit, but pass resolved sites,
  sequences, lifecycle groups, destinations, and ordering boundaries directly
  from qualification to emission instead of flattening and reconstructing
  them through optional strings;
- preserve atomic and ordinary multi-effect composition as one transaction on
  one selected semantic target, rather than rediscovering that equality in a
  second kind scan;
- retain the public diagnostic facts currently projected from fault plans,
  while deleting private source identities, reconstruction-only fields, the
  wide-shape validator, stale-plan application API, and displaced second-pass
  checks; and
- leave final mutated-image validation fully independent: it continues to
  reconstruct fault semantics from pristine and replacement executable bytes,
  patch geometry, inventory, and mutation proof rather than trusting the
  resolver's pointers or conclusions.

The current planner, application family, kind dispatcher, and shape validator
total about 1,818 production implementation lines before composition staging.
The conservative target is at least 750 gross lines removed and no more than
200 lines of transaction control and retained diagnostic projection, for at
least 550 net deleted lines. The attempt is abandoned and reverted if explicit
dry runs cease to preserve their selection diagnostics and non-mutating
behavior, if live mutation retains a full replacement image in a public or
long-lived dry-run product, if kind-specific emission becomes a variant DSL or
callback framework, if final validation starts trusting planning state, or if
complete convergence cannot clear the 500-line net abort floor.

Focused validation covers plan diagnostics and exact selected identities,
malformed and foreign request rejection at the public request boundary,
transactional multi-effect atomic and ordinary mutations, every barrier drop/
move/retarget/participant form, LDS address mutation, fault-plus-perturbation
composition, exact-byte results, rollback, cardinality, and adversarial final
validation across all five emulated targets. Existing tests that manufacture
or corrupt the deleted internal plan bridge will be replaced by tests of the
new resolver boundary and unchanged externally visible behavior, not silently
dropped. The cutover gate remains the complete nonphysical RocJitsu matrix.

The replacement converged as one private resolution-and-application
transaction. `resolve_consan_fault_mutations` now selects and qualifies each
semantic target once, retains direct immutable references to the selected
sites and lifecycle facts, and either projects the existing diagnostic
presentation or continues directly into byte emission. Live application no
longer consumes the diagnostic plan. Atomic and ordinary multi-effect faults
therefore remain one transaction without a second identity scan, while the
diagnostic plan is now presentation-only.

The cutover deleted the public planning input/result pair, the split planning
entry point, the public stale-plan application API, the public single-mutation
composition helper, reconstruction-only source identities and payload fields,
the kind redispatch bridge, and the 137-line wide-plan shape validator. The
kind-specific qualification and encoding remain explicit. Final validation
also remains independent: it still derives its verdict from pristine and
replacement bytes, patch geometry, inventory, and mutation proof, and receives
none of the resolver's target references.

The production accounting from the committed thesis through the completed
cutover is:

| Signal | Before | After | Change |
|---|---:|---:|---:|
| tracked production files | 312 | 312 | 0 |
| physical production lines | 98,645 | 98,060 | -585 |
| nonblank production lines | 92,222 | 91,656 | -566 |
| implementation lines | 84,536 | 84,017 | **-519** |

The nine affected production files contain 259 added and 844 deleted physical
lines. The governing 519-line net implementation reduction clears the
500-line abort floor, although it is 31 lines short of the deliberately
conservative 550-line target. The campaign is now **4,006 net implementation
lines below the 88,023-line baseline**, leaving 994 lines to the stretch
milestone.

The behavioral fault matrix passed 132/132 tests. The architecture-boundary
test initially rejected stale expectations for the removed planning bridge; it
was updated to require the single private transaction and forbid resurrection
of the deleted APIs and types, then passed. The complete nonphysical ConSan
matrix passed 4,787/4,787 tests across all four modes and all five target
architectures. The complete nonphysical RocJitsu matrix then completed 9,572
CTest entries with 9,557 passes, 15 explicit environment-dependent skips, and
zero failures. No physical-GPU test was run for this slice.

### 10.8 Macro-slice 8: one scalar-placement domain

The post-fault whole-tree comparison rejected three apparent opportunities.
The host report stack's generic decoder, mode decoders, analyzers, and
renderers are already separated; its remaining volume is mode-specific
interpretation rather than parallel implementations. Owner/epoch and private
epoch prologues share entry capture helpers but intentionally persist state in
different media, and their large bodies do not expose 500 lines that one
storage-neutral prologue would eliminate. A declaration/reference audit also
found no material orphaned public-contract family left by the preceding
cutovers: the apparent single-include `.h.inc` contracts are composed once by
their public owner and remain broadly consumed through it.

MOI scalar placement has a stronger duplicate-authority boundary. The
roughly 2,700-line automatic scalar-placement region repeatedly reconstructs
the same owner components, tail floors, complete-scope references, site-local
liveness sets, initialized entry windows, architectural reservations,
dynamic-stack reservations, physical-VCC conflicts, and already-selected
instrumentation ranges. Thirty-one local range searches then restate those
constraints in bespoke predicates while placing the EXEC-save window,
dispatch identity, owner source, spill window, indirect-PC/SCC/key/return
tuple, visible-evidence latch, dynamic-stack bootstrap, and persistent scalar
tuple. The fallback order is substantive; the repeated representation and
range-accounting mechanics are not.

The deletion thesis is:

- derive one immutable scalar-placement domain per connected owner component,
  with explicit whole-owner and site-local availability views;
- let one small transactional window allocator own architectural bounds,
  alignment, fixed reservations, selected-range overlap, physical-VCC
  compatibility, and commit/rollback of a candidate tuple;
- describe each existing fallback as an ordered set of typed scalar roles and
  lifetime requirements, retaining the current mode and target policy that
  chooses which fallback is attempted and in what order;
- obtain code-object-wide placement by intersecting those same component
  domains, rather than rebuilding a second global liveness/reference model;
- retain owner-local output contracts, diagnostics, explicit overrides,
  private-state choices, and descriptor/VGPR placement behavior unchanged;
  and
- delete the displaced union-find/context scans, ad hoc unavailable sets,
  nested first-range searches, repeated overlap predicates, and fallback-only
  requalification loops as each consumer moves to the common domain.

The affected scalar-placement and validation region contains about 2,700
production implementation lines and 31 separately spelled window searches.
The conservative target is at least 900 gross implementation lines removed
and no more than 300 lines of domain, allocator, and typed role descriptions,
for at least 600 net lines deleted. This is not permission to introduce a
general constraint language: the attempt is abandoned and reverted if the
allocator needs mode enums, target switches, callbacks that encode fallback
policy, or an opaque table/DSL; if fallback priority or supported pressure
cases change; or if complete convergence cannot clear the 500-line net abort
floor.

Focused validation covers automatic global and owner-local EXEC/dispatch
placement, Inline and compact scalar spill, fixed and dynamic stacks,
full-pressure and high-reference kernels, persistent scalar/VGPR/private
fallbacks, wave32/wave64 VCC conflicts, explicit overrides, disconnected and
shared owners, and all five emulated targets. Exact operating-point,
diagnostic, descriptor, spill, and emitted-byte assertions remain the
differential oracle. The cutover gate remains the complete nonphysical
RocJitsu matrix.

The cutover converged on one range-accounting mechanism without turning it
into a mode-aware solver. `MoiScalarPlacementDomain` now owns ordinary-file
bounds, target-owned architectural reservations, alignment, fixed and newly
selected overlap, optional site availability, and the combined CDNA physical-
VCC proof. Fallback order and semantic eligibility remain visible in the
calling placement code. Inline, compact, fixed-stack, dynamic-stack, global,
owner-component, dispatch, owner, and persistent-state searches all consume
that domain rather than carrying private overlap predicates.

Owner topology and immutable owner facts are also derived once. One connected-
component builder serves transient and persistent placement. Resolved owner
contexts carry their scalar tail, initialized entry extent, and conservative
fixed/dynamic-stack classification. One availability snapshot carries both
the complete site union and per-owner unions; ownerless shared contexts remain
in the former rather than being accidentally projected through an owner map.
One explicit transient-state inventory reserves accepted global and owner-
local state for later persistent searches.

The spill fallbacks now share the same transaction boundary. Global and
component-local Inline spill placement use the domain for the saved window and
router tuple. RDNA Inline and CDNA compact branch-only spill use one borrowed-
window allocator, while their target/mode eligibility and the compact entry-
PC preservation fact remain outside it. The owner-wide partial fallback was
reduced to a domain transaction with explicit rollback, and scalar layout is
validated through one orchestration entry point. The duplicate public range
record and unused validation-reason taxonomy were deleted. The architecture-
boundary test now requires the unified validator and forbids restoring the
split validator APIs or a second scalar-range type.

Two tempting deletions were rejected by differential evidence rather than
papered over. Four heterogeneous CDNA tests proved that the owner-wide partial
fallback has semantics not supplied by component-only placement, including
rollback when dispatch relocation does not recover a component. Two RDNA4
full-pressure Inline tests proved that object-wide spill/router coordination
is distinct from owner-component spill placement. Both behaviors remain, but
their range mechanics now use the common domain. A zero-width explicit EXEC
window used only by an owner-entry prologue likewise remains a legal declared
range while being ineligible for allocation.

The production accounting from the committed Macro-slice 7 result through the
completed scalar cutover is:

| Signal | Macro-slice 7 | Macro-slice 8 result | Change |
| --- | ---: | ---: | ---: |
| ConSan production files | 312 | 312 | 0 |
| Physical ConSan production lines | 98,060 | 97,525 | **-535** |
| Nonblank ConSan production lines | 91,656 | 91,118 | **-538** |
| ConSan implementation lines | 84,017 | 83,516 | **-501** |

The production diff contains 692 added and 1,227 deleted physical lines. The
501-line governing reduction clears the 500-line abort floor, although it
falls 99 lines short of the 600-line conservative target. From the 88,023-line
sixth-refactoring baseline, the campaign has now removed **4,507 net
production implementation lines**, leaving 493 lines to the 5,000-line
stretch milestone.

The focused MOI matrix passed 827/827 tests. The complete nonphysical ConSan
matrix passed 4,787/4,787 tests across all four modes and all five target
architectures. The complete `-j16 -LE physical` RocJitsu gate then processed
9,572 registered entries with zero failures, one disabled test, and nine
expected environment-dependent skips. No physical gfx1201 test was run at
this intermediate milestone. Macro-slice 8 therefore closes as an accepted
subsystem convergence, but the sixth-refactoring campaign remains active and
returns to a fresh whole-codebase search rather than extending scalar
placement mechanically.

## 11. Post-refactoring completion audit

This audit is based on a fresh read of the current production tree, its build
manifests, the definitions and consumers around the remaining large regions,
and the final test inventory. It does not infer completion merely from the
execution record or from reaching the numeric review floor.

### 11.1 Quantitative result and scope preservation

The governing counter now reports 83,516 production implementation lines:

| Signal | Baseline | Final tree | Change |
| --- | ---: | ---: | ---: |
| Production files | 310 | 312 | +2 |
| Physical production lines | 102,170 | 97,525 | **-4,645** |
| Nonblank production lines | 95,756 | 91,118 | **-4,638** |
| Production implementation lines | 88,023 | 83,516 | **-4,507** |
| Registered ConSan tests | 5,424 | 5,422 | -2 |

The implementation reduction is 5.1% of the sixth-refactoring baseline. It
clears the 3,000-line review floor by 1,507 lines and finishes 493 lines short
of the explicitly nonbinding 5,000-line stretch milestone. The original
Section 8 ledger undercounted the named baseline commit by three physical,
three nonblank, and eight implementation lines; the table above is the fresh
blob-by-blob recount. The source diff in the declared production directories
contains 4,300 additions and 8,944 deletions across 51 files; that raw diff
also includes the one-line build-manifest change. The only enabling production
change outside the governing ConSan directories is the 22-line
`InstructionSequence` extension from Macro-slice 2. Charging that extension in
full still leaves a conservative **4,485 net implementation lines removed**.

No implementation was displaced into tests, generated sources, scripts,
comments, data tables, or metaprogramming. The two additional production files
are the 48-line typed enum-vocabulary mechanism and the implementation-free
architecture tombstone retained because this campaign could not delete a file.
All four modes and all five targets remain registered and exercised.

The accepted macro replacements and retained bug fix account for the
implementation change as follows:

| Macro-slice | Replacement | Net implementation change |
| ---: | --- | ---: |
| 2 | repeated instruction construction -> one typed transaction | -1,644 in the governing scope; -1,622 including its external helper cost |
| 3 | parallel enum arrays and spelling switches -> one vocabulary authority | -601 |
| 4 | retained independent fail-closed bug fix after abort | +2 |
| 5 | SuperCollider LDS private reservoir -> common direct reservoir | -1,244 |
| 7 | split fault plan/re-resolution -> one resolved mutation transaction | -519 |
| 8 | repeated scalar range reconstruction -> one placement domain | -501 |

Macro-slices 1, 4, and 6 were reverted when complete or representative
cutovers failed their stated economics. Their experimental additions do not
contribute to the final size result.

### 11.2 Actual component and dataflow shape

The static transformer now has an exhaustive, exclusive, forward-only build
graph:

```text
contracts -> targets -> analysis -> transform -> validation -> orchestration
```

- **Contracts** own classifiers, normalized semantic vocabularies, input
  layout, observation policy, and immutable common value types.
- **Targets** turn concrete gfx identity into normalized capabilities and
  target operations. They do not choose a mode.
- **Analysis** decodes the program once into program, synchronization, and
  fault-selection inventories without depending on mutation state.
- **Transform** owns mode planning and emission, shared placement and resource
  mechanics, routing, descriptor growth, fault application, perturbation, and
  patch publication.
- **Validation** independently reconstructs original and replacement behavior,
  descriptor effects, patch geometry, and relay ownership. It may read
  analysis and transformation products but is not visible to either producer.
- **Orchestration** composes those stages and publishes the reviewed result.

The CMake manifest assigns every production `.cpp` to exactly one component,
rejects duplicate ownership, and rejects an unowned source. The architecture-
boundary gate enforces the important reverse-dependency prohibitions rather
than leaving this graph as documentation only.

The runtime side is separately layered as report trust, report snapshot,
mode-local decoding, mode-local analysis, mode-local rendering, and the HSA
hook coordinator. It consumes the transformer report contract; it does not
reach back into placement or mutation internals.

The effective dataflow is consequently:

```text
request + object
  -> normalized target profile and immutable program inventories
  -> selected mode policy and typed resource/placement plans
  -> shared mutation and emission mechanisms
  -> patch proof + candidate bytes
  -> independent final validation
  -> public transform result
  -> runtime snapshot/decode/analyze/render pipeline
```

The five successful replacements shortened this flow rather than adding
facades beside it. Instruction words have one construction transaction;
closed vocabularies have one iterable/spelling authority; direct relay
reservoirs have one relocation representation; fault selection flows directly
into mutation; and scalar searches consume one owner-aware availability
domain.

### 11.3 Mode locality and sharing

Record/Replay, Sampled, and InlineShadow register their policy and resource
demand in `consan_moi_record_replay.cpp`, `consan_moi_sampled.cpp`, and
`consan_moi_inline_shadow.cpp`. Their evidence bodies, report layouts, report
contracts, and specialized emitters remain in correspondingly mode-named
files. SuperCollider owns its LDS and FLAT protocols and its report planning in
SuperCollider-named files. The host report pipeline follows the same rule with
mode-named decoder, analyzer, and renderer sources.

The common layer contains mechanisms rather than a shadow implementation of
each mode: instruction transactions, scalar availability, direct reservoir
relocation, descriptor mutation, patch publication, target-address
normalization, shared barrier placement, and report transport. Mode owners
select typed policy or supply a semantic body; common mechanisms do not reopen
that policy through a broad request, an architecture value, or a mode switch.
The boundary test specifically prevents mode-local probe emission, scratch
demand, report capacity, private-owner policy, and report-layout policy from
returning to common files.

There is no production file for a Record/Replay-by-gfx, Sampled-by-gfx, or
InlineShadow-by-gfx implementation. The only filenames combining a mode and a
gfx family are the three SuperCollider target-operation providers; they are
architecture providers named for gfx9, gfx11/gfx12, and gfx1250, and contain
target encoding/capability behavior rather than duplicated SuperCollider
algorithms. Thus the tree has not acquired an `N * M` mode/architecture
matrix.

### 11.4 Architecture locality

Concrete target behavior is physically concentrated in gfx-named profile and
target-operation files: program analysis, fault operations, SuperCollider
operations, validation operations, gfx1250 LDS/VGPR-bank behavior, and the
per-target capability profiles. Shared and mode-local code consumes normalized
target facts and operation packages.

A raw-architecture-token scan of common and mode files found references only
in explanatory comments and stable diagnostics, not behavioral dispatch.
Mode-provider sources are mechanically forbidden from including generated ISA
architecture headers or naming `ROCJITSU_CODE_ARCH_*`. Target providers are in
turn forbidden from selecting Record/Replay, Sampled, InlineShadow, or
SuperCollider policy. The hypothetical-target test proves that a new target
can register normalized analysis without changing a mode, while the
hypothetical-mode test proves the converse. This makes architecture-specific
code skippable and preserves a direct extension path.

### 11.5 Independent validation and absence of rewrite residue

Final validation remains a separate component and reconstructs its verdict
from pristine bytes, replacement bytes, immutable inventories, patch geometry,
and mutation proofs. It does not consume fault resolver pointers, trust a
producer's placement verdict, or call mutation infrastructure. Macro-slice 4
demonstrated that replacing mirrored instruction construction with explicit
decoded proofs would strengthen one aspect of independence but add 617
physical lines. That experiment was reverted rather than retained as an
uneconomic second validation language; the existing independent byte,
semantic, graph, descriptor, and ownership proofs remain intact.

Searches of the production tree and the boundary gate confirm that the split
fault-planning/application APIs, dense-relay experiment, decoded-proof helpers,
special LDS reservoir representation, parallel enum spelling authorities, and
split scalar validators have not survived. The three abandoned experiments
had all experimental implementation reverted. There is no material comparison
switch, migration adapter, alternate emitter, duplicate authority, or inactive
framework left in production. The one empty source tombstone carries no
implementation and is declared explicitly by the exhaustive source-ownership
manifest.

### 11.6 Bugs and regression coverage

One pre-existing behavior defect was exposed during the work. If every
expected compare-exchange in an InlineShadow versioned-release transaction was
corrupted, final validation dereferenced an empty position list instead of
failing closed. The owner fix reports the missing transaction semantics, and
`FinalValidationPinsVersionedCausalReleaseTransaction` now corrupts every
claim and asserts that diagnostic. No other pre-existing behavior defect was
found during the accepted replacements.

Tests that existed solely to construct the deleted internal fault-plan bridge
were retired with that private API. Their behavioral obligations remain in the
fault request, selection, exact-byte mutation, rollback, composition, and
adversarial final-validation tests. Macro-slice 5 added direct coverage for
caller-selected reservoir donor bounds; Macro-slices 7 and 8 strengthened the
architecture gate to forbid resurrection of the displaced authorities; and
the instruction-sequence test directly pins the two newly supported EXEC
branch encodings. The final registration count is two below the baseline:
representation-specific fault-plan and LDS-reservoir tests disappeared with
their deleted production authorities, while replacement tests exercise the
resolved fault transaction, common direct reservoir, donor bounds, enum
vocabularies, and instruction transaction. The correct/incorrect device
matrix, its mode/target rows, and its 635 physical gfx1201 rows were not
reduced.

### 11.7 Fresh whole-tree candidate disposition

The final scan reconsidered the initial examples and the other large semantic
regions rather than stopping at Macro-slice 8:

- **MOI emission:** Macro-slice 2 harvested the repeated optional-builder and
  local-fixup protocol. Surviving branch offsets connect separately placed
  text regions and expose placement semantics; straight-line fragments retain
  named semantic subexpressions. Neither is a second instruction transaction.
- **SuperCollider:** Macro-slice 5 removed the duplicate direct reservoir.
  Remaining LDS and FLAT volume implements different protocols: LDS owns dense
  generated islands and multi-owner/max-flow policy, while FLAT owns
  direct/local/appended placement and runtime address-space gating. A common
  body would be a union-shaped mode framework.
- **MOI placement:** Macro-slice 8 removed repeated range accounting and owner
  reconstruction. Remaining fallback order, lifetime distinctions, resource
  media, explicit overrides, and target/mode eligibility are the placement
  problem itself. Differential tests proved that owner-wide partial placement
  and RDNA4 object-wide spill coordination cannot be collapsed into the
  component-local fallback.
- **Independent validation:** the complete Macro-slice 4 experiment showed
  that explicit decoded proofs restate rather than eliminate the surviving
  obligations. A smaller proof representation would be required before this
  region becomes a credible replacement.
- **Dense relay routing:** the complete Macro-slice 6 experiment converged all
  synchronization users and the access emitter but removed only 147 net lines.
  Call-return, derived-key, clone-local, explicit-key, tagged-SCC,
  dependency-wait, and relocated-host variants are semantic inputs, not copies
  hidden by the current organization.
- **Branch-only routing:** exact-batch, exact-pair, and bounded greedy paths
  preserve different feasibility and work-budget contracts. Fixed
  source/destination pairs create capacity-sharing commodities, and deferred
  owner selection is a group-activation cost; an ordinary additive flow model
  cannot replace those semantics.
- **Program inventory:** kernel and non-dispatchable-function containers share
  range and statistics fields, but descriptor/dispatch/ownership differences
  leave only a few dozen paired traversal lines. A unified container would add
  filtered views and does not expose 500 deletable lines.
- **Fault and synchronization inventories:** both project identity and decoded
  geometry, but fault sites additionally include non-synchronizing LDS and
  unsupported ordinary-memory candidates with stable user-facing diagnostics.
  Eliminating that projection would widen synchronization events into a
  mutation/synchronization union for an estimated 250--350-line deletion.
- **Host reports, hook registries, options, and synchronization graphs:** the
  host report stages are already mode-local; hook registries own different
  lifetimes; options are a thin aggregate of typed input contracts; and the
  barrier, atomic, ordinary-memory, execution-owner, and move-destination
  graphs prove different relations. None currently presents a replacement-
  sized duplicate authority.

These are economic and semantic dispositions, not claims that the code can
never become smaller. They show that no currently identified replacement both
belongs in this campaign and clears its approximately 500-line floor without
introducing a broader union, filtering layer, proof DSL, or second authority.
Manufacturing 493 lines of deletion to hit the stretch number would therefore
make the design less clear, contrary to the charter.

### 11.8 Final validation evidence and conclusion

At the final tree:

- the local TheRock toolchain/runtime rebuild completed successfully at
  `-j16`;
- `ConSan.ArchitectureBoundaries` passed;
- the complete nonphysical ConSan matrix passed **4,787/4,787** in 197.99
  seconds, covering all four modes and all five emulated targets;
- the complete nonphysical RocJitsu matrix processed 9,572 registered entries
  with **9,562 passes, zero failures**, one disabled test, and nine expected
  environment-dependent skips in 238.20 seconds; and
- the deliberately selected physical gfx1201 ConSan milestone passed
  **635/635** serially at `-j1` in 109.85 seconds.

The audit therefore finds every completion criterion in Section 9 satisfied.
The sixth refactoring closes at 83,516 governing implementation lines, with
4,507 lines removed, five accepted subsystem replacements, three economically
failed experiments fully reverted, preserved mode and target coverage,
independent final validation, and no clearly superior remaining macro
replacement identified by the fresh whole-tree read.
