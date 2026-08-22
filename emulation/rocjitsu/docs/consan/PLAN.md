# ConSan Work Plan

This document tracks the remaining work required to turn ConSan from a
multi-architecture prototype into a maintainable, reviewable sanitizer. The
target set is CDNA3 (`gfx942`), CDNA4 (`gfx950`), CDNA5 (`gfx1250`), RDNA3
(`gfx1100`), and RDNA4 (`gfx1201`).

The work now has two phases. First, finish the behavioral safety net and freeze
a trustworthy current-tip validation baseline. Then use that evidence to
replace the prototype without preserving its accidental architecture-specific
structure.

## 1. Preparation work

**Status: active; test-suite expansion and end-to-end qualification remain**

At the 2026-08-22 plan checkpoint, the freshly fetched `origin/develop` still
resolves to `550548c58ae9`. It is already an ancestor of the current branch
through merge commit `cf1df8ec82`, so the requested integration checkpoint is
current and Git has no new merge to perform; no empty merge should be
manufactured. Before execution resumes and again before final qualification,
recheck this relationship. If `origin/develop` has advanced, integrate it with
a merge commit. That integration is not complete until every resulting build
or test failure is repaired and each behavioral integration defect is
protected by a focused host-unit or device regression.

The checked-in device-conformance tier, its behavioral coverage, and its
remaining gaps are tracked in [PLAN_DEVICE_TESTS.md](PLAN_DEVICE_TESTS.md).
The end-to-end procedure and acceptance rules are defined in
[VALIDATION.md](VALIDATION.md), with current target evidence in
[STATUS_CDNA4.md](STATUS_CDNA4.md) and
[STATUS_GFX1250.md](STATUS_GFX1250.md).

This phase is preparation for the production replacement, not an attempt to
perfect the prototype indefinitely. **Its primary deliverable is comprehensive
host-unit and checked-in device test suites, not the fixes made to the prototype
along the way.** New regressions distilled from end-to-end investigations and
the systematic expansion or transport of existing tests are equally part of
that deliverable. This behavioral safety net is the durable asset that will
make the section 2 replacement automatically iterable. It must preserve
behavior that matters to users, expose current target limitations precisely,
and distinguish a safe new abstraction from a regression without depending on
the prototype's layout or implementation choices. The working-loop rules below
are binding preparation requirements and feed the exit criteria; they are not
merely suggested ways of organizing prototype fixes.

### Acknowledged working contract

The following rules jointly define the preparation loop and its priorities.
None is optional merely because another rule currently offers faster visible
progress. After this plan-only checkpoint, the first execution step is to
resolve and, if necessary, merge the freshly fetched `origin/develop`, then
repair and regression-test any integration breakage before resuming the
validation loop.

1. Keep a current base. At each qualification checkpoint, resolve the freshly
   fetched `origin/develop`. If it is newer, merge it with a merge commit, then
   repair every resulting build, host-unit, device, and relevant end-to-end
   failure before treating later validation evidence as current. Add a focused
   host-unit or device regression for every behavioral defect exposed by the
   merge. If `origin/develop` is already an ancestor, record the checkpoint as
   current rather than manufacturing an empty merge.
2. Keep both [STATUS_CDNA4.md](STATUS_CDNA4.md) and
   [STATUS_GFX1250.md](STATUS_GFX1250.md) live. Update the relevant ledger in
   the same change whenever an investigation changes what is known about a
   cell; the documents must describe the state at every revision, not merely
   the intended end state.
3. Use one global lexicographic priority across both ledgers. Between engine
   columns, work Record/Replay first, then Sampled, then SuperCollider, then
   Inline Shadow. Within the active column, lift the floor horizontally: tackle
   red cells before orange, orange before yellow, and yellow before relying on
   or revalidating green. In particular, do not use a lower-priority engine's
   red cells to bypass unfinished higher-priority-engine cells merely because
   red looks worse than orange. Keep CDNA4 and gfx1250 moving under that
   ordering rather than completing one architecture before returning to the
   other. Until CDNA4 has caught up, bias approximately 75% of investigation
   time to CDNA4 and 25% to gfx1250 without starving either ledger. Tactical
   latency may change which currently useful cell is taken next, but it does
   not remove deferred higher-priority cells. After all applicable cells are
   green, re-run every green cell on one reviewed revision.
4. Let useful evidence, not waiting time, drive iteration. Debug abnormal
   slowness when that is actionable; otherwise record and defer a slow cell and
   move to the next useful target. A tactically slow higher-priority cell must
   not prevent progress on other cells, but it remains an explicit exit item.
5. For every defect or useful end-to-end idiom encountered, make the durable
   result a focused host-unit test, checked-in device test, or both. Studying an
   end-to-end workload is not finished when its prototype fix works or its
   ledger cell changes color: actively look for and reduce its relevant
   compiler, ISA, resource-pressure, control-flow, memory, or synchronization
   behavior into a quick checked-in contract. Use a host unit for isolated
   analysis, policy, ABI, or runtime semantics; use adjacent correct/incorrect
   device workloads when transformed device behavior is at issue; use both when
   the defect crosses that boundary. The prototype fix and a status-cell
   promotion support that deliverable; neither replaces it. If an investigation
   genuinely cannot yield a focused checked-in regression, record the concrete
   reason instead of silently losing the evidence. Coverage expansion is also a
   deliverable when it exposes no new prototype defect and changes no ledger
   color.
6. Generalize every new and existing test by behavior rather than by its source
   architecture. Translate target-native device code as necessary--the device
   code need not be textually identical--and cover every semantically
   applicable CDNA3/4/5 and RDNA3/4 target. Historical
   `gfx950`-to-`gfx942` transport is only one example, not the boundary. The
   portable asset is the behavioral idea, not identical ISA bytes: transport
   across generations and between CDNA and RDNA wherever an equivalent native
   idiom exists. Treat the suite-wide audit of already-existing tests as
   explicit preparation work, not only as a rule applied to tests added from
   now on.
7. Use fast host tests and parallel RocJitsu targets for the tight loop. It is
   acceptable to omit serialized physical `gfx950` temporarily when it
   dominates latency, but run it at regular checkpoints, after relevant native
   changes, and for final qualification. Periodically run every host, emulated
   device, and physical-device test even when the usual inner loop is narrower.

### Remaining work

1. At later qualification checkpoints, fetch and merge any newer
   `origin/develop` with a merge commit. Repair every resulting build or test
   failure and add a focused host-unit or device regression for each behavioral
   defect exposed by the integration before accepting the merge. The freshly
   fetched `550548c58ae9` checkpoint is already incorporated; do not create an
   empty merge solely to restate it.
2. Continue making both the conventional host-unit suite and the checked-in
   device suite comprehensive. Close meaningful gaps identified by the
   capability audit, [VALIDATION.md](VALIDATION.md), or Aorta. Device scenarios
   must use adjacent correct/incorrect behavioral pairs and assert exact
   workload and diagnostic outcomes rather than prototype layout.
3. Work both current validation ledgers:
   [STATUS_CDNA4.md](STATUS_CDNA4.md) and
   [STATUS_GFX1250.md](STATUS_GFX1250.md). Across engine columns, prioritize
   Record/Replay first, then Sampled, then SuperCollider, then Inline Shadow.
   Within that ordering, lift the floor horizontally: resolve red cells before
   orange, orange before yellow, and yellow before green. Select work across
   both architectures under those rules rather than serializing the project by
   architecture. While CDNA4 remains behind, use the standing approximate
   75% CDNA4 / 25% gfx1250 time allocation without allowing either ledger to
   stall. Once every applicable cell is green, make a fresh global pass over
   all green cells rather than relying on accumulated historical evidence.
4. Treat every end-to-end investigation as an opportunity to create a fast,
   clean regression. Reduce the relevant compiler, instruction, resource,
   control-flow, or synchronization idiom into a host-unit test, a checked-in
   device correct/incorrect pair, or both as appropriate. Host units should
   isolate analysis, policy, ABI, and runtime contracts; device pairs should
   exercise transformed code and assert both the clean result and the intended
   diagnostic. The test is the preparation-phase deliverable; a prototype fix
   or ledger improvement without durable test coverage is incomplete, even
   when the corresponding end-to-end cell passes. When no faithful reduction
   is possible, leave an explicit rationale in the relevant plan or status
   ledger.
5. Make a deliberate, suite-wide pass over all existing ConSan host and device
   tests to identify architecture-local tests whose underlying idea is
   cross-cutting. Transport each useful contract to every semantically
   applicable target, adapting the device code and target-native instruction
   forms as needed. The originating architecture does not define the coverage
   boundary: do not stop at historical `gfx950`-to-`gfx942` ports. Consider
   CDNA3/4/5 and RDNA3/4 for every behavior, and record a capability-based
   reason for each target where the contract is not applicable.
6. Stay tactical about latency. If a reproducer is abnormally slow, first ask
   whether the slowness itself is a defect that can be debugged or reduced.
   Avoid repeated long waits that provide no new information; defer a slow cell
   when another priority cell offers a faster path to useful coverage, while
   keeping the deferred cell explicit in its status ledger.
7. Use a two-speed test cadence. During tight iteration, favor targeted host
   tests and the parallel RocJitsu configurations. If serialized physical
   `gfx950` rows dominate device-suite latency, omit them temporarily and
   accept the bounded risk of a short-lived physical-only regression. Run the
   physical tier at regular checkpoints, after changes that plausibly affect
   native behavior, and as part of every full qualification pass.
8. Run the complete checked-in device matrix periodically in one CTest
   invocation and keep it fully green on RocJitsu `gfx942`, `gfx950`,
   `gfx1250`, `gfx1100`, and `gfx1201`, plus physical `gfx950`. Refresh the
   whole-suite timing after the latest tranche and interpret it against the
   documented 5--20-minute heuristic on the reference host at that host's
   configured parallelism. The current 118.75-second run is below the lower
   review threshold, so the residual coverage audit remains active; do not add
   artificial work merely to consume time.
9. Produce a `gfx950` counterpart to
   [GFX1201_EMPIRICAL_STUDY.md](GFX1201_EMPIRICAL_STUDY.md), based on fresh
   correctness, coverage, fault-detection, overhead, containment, and
   implementation-complexity evidence for all four modes.

### Exit criteria

- The branch includes the already-fetched current `origin/develop` and remains
  current at the final qualification point. All integration breakage is
  repaired with focused regression coverage.
- The host-unit and device suites comprehensively cover the behavior required
  by the current end-to-end corpus, Aorta, and the target capability audit.
  Every fixed defect has a regression, and every device scenario remains
  independent of prototype implementation details. Prototype fixes and status
  promotions are not counted as preparation deliverables on their own.
- The existing-test transport audit is complete. Every cross-cutting contract
  runs on each semantically applicable CDNA3/4/5 and RDNA3/4 target, or has a
  documented capability-based not-applicable disposition.
- Every applicable cell in both the CDNA4 and gfx1250 ledgers is green under the
  stated Record/Replay-first and lift-the-floor strategy, followed by a fresh
  global revalidation of all green cells at one reviewed revision.
- One reviewed revision passes the entire conventional host-unit suite and the
  checked-in simulator/physical device matrix. Faster simulator-focused
  iteration is permitted, but the physical `gfx950` tier and post-run health
  checks pass at regular milestones and at final qualification.
- The checked-in `gfx950` empirical study records the behavior and tradeoffs the
  production implementation must preserve.

## 2. Replace the prototype with a production-grade implementation

**Status: planned; begin after the preparation baseline is frozen**

### Problem

The current implementation is a prototype, not production-grade code. It was
assembled by multiple AI agents whose mandate favored adding whatever code was
needed to make local progress, without a corresponding requirement to reuse,
generalize, simplify, or minimize. The agents did not share a coherent view of
the whole implementation, and the code did not pass through a normal human
review process.

The result is a very large body of code with substantial architecture-specific
duplication, repeated policy decisions, ad hoc solutions, and weak layering.
Work that raises the implementation to a smaller set of concepts and cleanly
separates those concepts has not yet been done. This makes correctness hard to
reason about, review difficult, and cross-architecture regressions likely.

The goal is not to polish the prototype in place. The goal is to replace it,
incrementally and safely, with an implementation suitable for human review and
long-term maintenance.

### Review and design principles

Human reviewers should be able to see that the replacement:

1. **Minimizes added code.** Prefer deletion, reuse, tables, and existing
   RocJitsu facilities over new parallel mechanisms. Any net growth must carry
   a clear semantic justification.
2. **Expresses first-principles meaning.** Code should state architectural and
   sanitizer invariants directly. It should not encode a history of individual
   workload failures as unexplained branches and constants.
3. **Maximizes sharing across architectures.** Equivalent semantics must use
   shared analysis, planning, engine, and validation paths. Target-specific code
   should be limited to genuine ISA and ABI differences.
4. **Uses well-layered abstractions.** Decoding, semantic inventory, policy,
   resource planning, lowering, runtime integration, and report interpretation
   must have explicit boundaries and independently testable contracts.

### Proposed architecture

Converge on layers with one-directional dependencies:

1. **Target capability and decode adapters** translate target-native
   instructions and ABI facts into a shared semantic vocabulary.
2. **Target-neutral semantic inventory** represents accesses, barriers,
   atomics, fences, ownership, control flow, confidence, and typed exclusions.
3. **Engine-independent selection and resource planning** decides what can be
   instrumented and allocates registers, spills, private storage, LDS, and
   report capacity without embedding target emission.
4. **Engine semantics** implement SuperCollider, Record/Replay, Sampled, and
   Inline Shadow in terms of the shared inventory and resource plan.
5. **Small target lowerers/builders** encode only the instructions, calling
   conventions, wait behavior, and other details that genuinely differ by ISA.
6. **Code-object and dispatch integration** applies validated transformations,
   updates descriptors and segment sizes transactionally, and owns rollback and
   failure containment.
7. **Stable reporting and replay** consumes a documented ABI independently of
   target encoding details.

Use typed data and capability tables in place of repeated architecture
conditionals. An abstraction should correspond to a real invariant and normally
have multiple consumers; do not replace duplicated code with a generic layer
that merely hides the duplication. Reuse RocJitsu's existing decoder,
instruction-builder, code-object, relocation, spilling, and dispatch machinery
wherever their contracts are sufficient. Improve those shared facilities when
the missing capability is general rather than creating a ConSan-only copy.

### Migration approach

- First inventory the current implementation by responsibility, dependency,
  architecture, and line count. Identify duplicated algorithms, repeated
  encodings, mixed-layer functions, and prototype-only paths.
- Write down invariants and interfaces before moving code. Use the host tests,
  device-conformance matrix, and end-to-end validation to distinguish required
  behavior from accidental implementation details.
- Replace one vertical slice at a time. Keep each change reviewable, preserve
  behavior with focused tests, run all architectures under RocJitsu, and run the
  native device tier before removing the old path.
- Delete the superseded implementation in the same series that introduces its
  replacement. Do not leave permanent old/new forks or architecture-specific
  compatibility paths without an explicit removal plan.
- Generalize from at least two architectures whenever possible, then validate
  the abstraction on all five. Treat a new architecture branch in a shared
  layer as a design-review event.
- Keep commits and reviews conceptually narrow. Document non-obvious ISA and ABI
  facts at their boundary, and make first-principles reasoning visible to a
  reviewer without requiring knowledge of the prototype's history.

### Completion criteria

- The production path follows the layered architecture above, with no cyclic or
  hidden dependency from shared semantics back into target-specific emission.
- Equivalent behavior on the five architectures is implemented once; remaining
  target-specific code has a documented architectural reason.
- Prototype duplication and superseded code are removed, producing a material
  reduction in total code and in the architecture-specific share. Record these
  measurements rather than relying on subjective impressions.
- Adding or changing a supported architecture primarily affects its capability,
  decode, and lowering adapters, not every ConSan engine.
- Host, device-conformance, physical, and end-to-end gates pass at the same
  reviewed revision.
- The resulting patch series is small enough in concept and organization for
  meaningful human review, with documented invariants and no dependence on
  unexplained generated boilerplate.

## Execution order

1. Finish the preparation exit criteria: one green and timed whole-device
   matrix, reduced regressions for newly fixed defects, a bounded current-tip
   `gfx950` four-mode campaign, and the `gfx950` empirical study.
2. Freeze the behavioral, capability, and target-specific invariants that the
   replacement must preserve.
3. Replace prototype layers incrementally, deleting each superseded path in the
   same series that introduces its production replacement.
4. Run the full checked-in matrix continuously and repeat the expensive
   end-to-end campaigns at qualification milestones.
