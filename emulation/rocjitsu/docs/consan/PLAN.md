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

**Status: active. The tests are the primary deliverable.**

This phase exists to build the behavioral safety net for the production rewrite
in section 2. The current prototype will soon be replaced, so prototype repairs
and E2E status promotions are supporting work, not the durable output. The
durable output is a comprehensive, implementation-independent set of host-unit
and checked-in device tests that can drive and automatically validate that
rewrite. An E2E status promotion or prototype repair without a corresponding
test-coverage disposition is therefore not complete preparation work: extract
the durable regression when possible, or record why no faithful checked-in
reduction exists.

This phase is judged primarily by the breadth and quality of that automated
safety net, not by the number of E2E cells promoted or prototype fixes landed.
The status campaigns are both qualification work and a systematic source of
real user behavior from which to derive the tests that will make section 2
safe to iterate on.

The checked-in device tier and its coverage gaps are tracked in
[PLAN_DEVICE_TESTS.md](PLAN_DEVICE_TESTS.md). E2E procedures and acceptance
rules live in [VALIDATION.md](VALIDATION.md); current evidence lives in
[STATUS_CDNA4.md](STATUS_CDNA4.md) and
[STATUS_GFX1250.md](STATUS_GFX1250.md). The Aorta workloads are another source
of user-relevant behavior that the checked-in suites must cover.

The immediate first execution action, after this planning checkpoint, is to
integrate the newly fetched `origin/develop` and establish a green
post-integration baseline. Resolve the fetched ref when execution begins rather
than baking a transient commit ID into this plan. If it has advanced, integrate
it with a merge commit; if it is already an ancestor, record the verified no-op
instead of manufacturing an empty merge. Repair all resulting build and test
failures, and protect every behavioral integration fix with a focused host-unit
or device regression before accepting new validation evidence. Build failures,
host-test failures, emulated-device failures, and physical-device failures are
all integration breakage for this purpose. Repeat this check at later
qualification checkpoints.

### Working contract

1. **Grow tests as the main product.** Expand the conventional host-unit suite
   and the checked-in device suite together. The principal measure of progress
   in this phase is durable behavioral coverage, not the number of prototype
   patches or promoted E2E cells. Every behavioral fix is incomplete until it
   has an appropriate host regression, device regression, or both. Device tests
   use adjacent correct/incorrect workloads: the correct workload must produce
   exact correct results and no diagnostic, while the incorrect workload must
   produce the expected diagnostic. Tests must assert observable behavior
   rather than the prototype's current structure so that they remain useful
   while section 2 replaces that implementation.
2. **Use E2E validation as a test-discovery loop.** Work both
   [STATUS_CDNA4.md](STATUS_CDNA4.md), using physical `gfx950`, and
   [STATUS_GFX1250.md](STATUS_GFX1250.md), using RocJitsu emulation. Every
   investigation should identify the relevant compiler, ISA,
   resource-pressure, control-flow, memory, or synchronization idiom and
   distill it into a quick checked-in host or device contract, even when the
   E2E workload already passes and no prototype bug is found. Every studied
   case must end with an explicit test disposition: a new or identified host
   regression, a new or identified device contract, both where appropriate, or
   a concrete reason that no faithful checked-in reduction exists. This also
   applies when the E2E cell is tactically deferred. Keep each status file
   current in the same change that alters its evidence.
3. **Apply one global priority order.** Across both ledgers, select the engine
   column first: Record/Replay, then Sampled, then SuperCollider, then Inline
   Shadow. Within the active column, lift the floor: red before orange, orange
   before yellow, and yellow before green. In other words, the ordering is
   lexicographic: engine priority first, then lowest color across both ledgers
   for that engine. This is the default selection order, not permission to
   ignore either ledger. Keep both architectures moving; use
   the standing approximate 75% CDNA4 / 25% gfx1250 catch-up bias while CDNA4
   remains behind, without starving gfx1250. Once every applicable cell is
   green, make a fresh global pass over every green cell on one reviewed
   revision. Latency-driven exceptions are explicit scheduling deferrals, not
   silent changes to this priority or exemptions from final qualification.
4. **Be tactical about slow tests.** Do not spend repeated long iterations
   merely waiting for an abnormally slow reproducer. Investigate the slowness
   when it may itself reveal a hang, performance defect, or smaller reproducer;
   otherwise bound the experiment, record the evidence and a concrete return
   point, defer the cell, and move to another useful item. Prefer fixing
   everything that can be learned without repeated long waits before rotating
   away. Deferral changes scheduling but never removes the cell from the exit
   criteria. An abnormally long run is itself something to diagnose when that
   investigation is more productive than repeatedly enduring the delay.
5. **Generalize behavioral ideas across architectures.** Make a deliberate,
   suite-wide pass over both new and already-existing ConSan host and device
   tests. For every contract, consider all five supported targets: CDNA3
   (`gfx942`), CDNA4 (`gfx950`), CDNA5 (`gfx1250`), RDNA3 (`gfx1100`), and
   RDNA4 (`gfx1201`). Port the behavioral idea wherever it is semantically
   applicable. Treat portability of the behavioral idea as the default
   hypothesis, since most sanitizer behavior is cross-cutting. A transport need
   not reuse identical device code: preserve the behavioral contract while
   adapting target-native code, instruction forms, and ISA details as
   necessary. Historical `gfx950`-to-`gfx942` ports are examples, not a
   boundary; cross-generation and CDNA/RDNA transports are expected for
   cross-cutting behavior. Record a capability-based reason
   wherever a port is genuinely not applicable. The audit is not complete if a
   test merely remains on the architecture where it was first discovered:
   every host and device contract needs an explicit per-target disposition,
   and every meaningful missing transport must be implemented.
6. **Use a two-speed test cadence.** In the inner loop, favor targeted host
   tests and all parallel RocJitsu-emulated configurations. It is acceptable to
   omit the serialized physical `gfx950` tier temporarily when it accounts for
   most of the device-suite latency, explicitly accepting the short-lived risk
   of a physical-only regression in exchange for faster iteration. Do not
   disable that tier, redesign the checked-in tests around its omission, treat
   an omitted run as passing, or use fast-loop evidence alone to promote a cell
   whose acceptance requires physical execution. Keep it separately filterable
   for fast iteration, then run physical `gfx950`, every emulator target, and
   the complete host suite periodically, after relevant native-sensitive
   changes, and at final qualification.

### Per-investigation loop

1. Select a cell using the engine-first, lowest-color-next ordering across both
   live status ledgers, subject to the CDNA4 catch-up bias and tactical latency
   exceptions above.
2. Reproduce enough E2E behavior to understand the semantic idiom or bug. Debug
   unusual latency when useful; otherwise preserve the evidence and rotate.
3. Add the focused host regression, adjacent correct/incorrect device pair, or
   both. Consider transports of the same behavioral idea to all five targets.
4. Make only the prototype repair needed to satisfy the new contract, then run
   the targeted fast host and RocJitsu tests.
5. Update the affected status ledger in the same change. A status promotion or
   prototype fix without its durable regression is not a completed iteration.
6. At regular checkpoints, discharge fast-loop omissions by running the full
   host, emulated-device, and physical-device tiers.

### Remaining work

1. As the next execution step, run the requested integration of the freshly
   fetched `origin/develop` and establish that the post-integration build, host
   tests, emulated-device tests, and physical-device tests are green. If the
   fetched ref has advanced, create the requested merge commit; if it is already
   integrated, verify and record that fact. For every future ref advance,
   create a merge commit, repair all resulting breakage, and add focused
   regressions for behavioral integration defects.
2. Continue the global E2E loop over both
   [STATUS_CDNA4.md](STATUS_CDNA4.md) and
   [STATUS_GFX1250.md](STATUS_GFX1250.md), following Record/Replay, Sampled,
   SuperCollider, Inline Shadow order and lifting red, orange, yellow, then
   green within each active engine. Keep slow deferred cells explicit and
   return to them before exit.
3. Continue extracting quick, implementation-independent coverage from every
   useful E2E investigation, [VALIDATION.md](VALIDATION.md), and Aorta. Close
   both host-unit and device gaps; do not count a prototype repair alone as
   progress toward the principal deliverable.
4. Make a suite-wide inventory of all existing host and device behavioral
   contracts. For each, record an applicable, transported, or
   capability-based not-applicable disposition on `gfx942`, `gfx950`,
   `gfx1250`, `gfx1100`, and `gfx1201`, then implement every meaningful missing
   transport.
5. Periodically run the complete checked-in device matrix in one CTest
   invocation and keep it green on all five RocJitsu targets plus physical
   `gfx950`. Recheck its wall-clock time against the 5--20-minute coverage and
   practicality heuristic in [PLAN_DEVICE_TESTS.md](PLAN_DEVICE_TESTS.md). A
   run below five minutes is a prompt to reassess coverage, not a reason to add
   artificial work; a run above twenty minutes is a prompt to remove redundant
   cases without weakening behavior.
6. Produce a `gfx950` counterpart to
   [GFX1201_EMPIRICAL_STUDY.md](GFX1201_EMPIRICAL_STUDY.md), with fresh
   correctness, coverage, fault-detection, overhead, containment, and
   implementation-complexity evidence for all four engines.
7. After every applicable E2E cell is green, revalidate all green CDNA4 and
   gfx1250 cells on one reviewed revision and run the complete host,
   RocJitsu-device, and physical-gfx950 suites on that same revision.

### Exit criteria

- The final reviewed revision includes the latest fetched `origin/develop`.
  All integration breakage is repaired, and every behavioral integration fix
  has focused regression coverage.
- The host-unit and checked-in device suites comprehensively cover the relevant
  behavior in the current E2E corpus, Aorta, and the target-capability audit.
  Every fixed defect has an implementation-independent regression; device
  scenarios retain their exact correct/incorrect behavioral contracts.
- The existing-test audit is complete. Every cross-cutting contract runs on
  each semantically applicable CDNA3/4/5 and RDNA3/4 target, or has a documented
  capability-based exclusion; no contract is stranded on its source
  architecture merely because that is where its motivating E2E failure was
  first observed.
- Every applicable cell in the CDNA4 and gfx1250 ledgers is green, followed by
  a fresh global revalidation of all green cells on one reviewed revision;
  this includes every cell tactically deferred because of abnormal latency.
- That revision passes the complete conventional host suite and the full
  RocJitsu/physical device matrix. Physical `gfx950` and post-run health checks
  pass at regular milestones and final qualification even though inner-loop
  runs may omit them.
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
