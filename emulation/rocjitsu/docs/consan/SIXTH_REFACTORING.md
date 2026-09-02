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
architecture locality, and test coverage. It also reduced the starting 91,450
production implementation lines to 88,015 lines. That is real progress, but the
recent slope is not sufficient:

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
review floor** and a **5,000-line stretch milestone**, measured from the 88,015
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
| Physical production lines | 102,167 |
| Nonblank production lines | 95,753 |
| Production implementation lines | **88,015** |
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
   the 88,015-line baseline, without scope loss or metric displacement.
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
| Physical ConSan production lines | 102,167 | 100,517 | **-1,650** |
| Nonblank ConSan production lines | 95,753 | 94,112 | **-1,641** |
| ConSan implementation lines | 88,015 | 86,379 | **-1,636** |

The physical source diff contains 2,326 additions and 3,979 deletions in the
declared ConSan scope. The generic `InstructionSequence` owner outside that
scope adds 22 net implementation lines, so the conservative campaign result
including its full enabling cost is **1,614 net production implementation
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
