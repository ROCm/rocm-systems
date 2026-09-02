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
