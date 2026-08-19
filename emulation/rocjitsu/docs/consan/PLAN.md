# ConSan Work Plan

This document tracks the three major tasks required to turn ConSan from a
multi-architecture prototype into a maintainable, reviewable sanitizer. The
target set is CDNA3 (`gfx942`), CDNA4 (`gfx950`), CDNA5 (`gfx1250`), RDNA3
(`gfx1100`), and RDNA4 (`gfx1201`).

The three tasks reinforce one another. The checked-in device tier supplies the
regression safety needed to finish `gfx950` and restructure the implementation;
the `gfx950` work supplies realistic reductions for that tier; and the
production rewrite must make the cross-architecture tests exercise shared code
rather than five loosely related implementations.

## 1. Establish a checked-in device-conformance tier

**Status: common tier implemented; capability-specific expansion remains**

The detailed remaining test work, including the required correct/incorrect
pairing for every scenario, is tracked in
[PLAN_DEVICE_TESTS.md](PLAN_DEVICE_TESTS.md).

### Problem

ConSan currently has a large host-test suite and separate end-to-end validation
campaigns. Host tests efficiently cover decoding, semantic analysis,
instrumentation planning, instruction emission, resource allocation, and the
MOI models, but they cannot establish that the complete hook, code-object,
dispatch, device-execution, and reporting path works.

The end-to-end workloads in [VALIDATION.md](VALIDATION.md) provide that stronger
evidence, but their models and assets are too large to check into this tree and
are expensive to run under emulation. Consequently, they cannot be the routine
cross-architecture regression gate for day-to-day development. This makes it
too easy to improve one architecture while silently regressing another.

The existing `ConSanGfx*Sim` tests are the beginning of the missing middle tier,
but their coverage is uneven and reflects architecture bring-up history more
than a systematic device contract.

### Proposed approach

Create a first-class **device-conformance** tier made from small, checked-in,
self-checking device programs. These tests must use production code-object
interception and ConSan instrumentation, execute actual target device code, and
validate application output, instrumentation coverage, completeness, and
diagnostics. They should require no external model, data set, or prebuilt
artifact.

The same scenarios should run through RocJitsu for every supported architecture
and run natively whenever matching physical hardware is available. Simulator
and physical execution are two backends for the same test contract, not two
unrelated suites. Emulation provides portable integration coverage; native
execution remains necessary for runtime- and hardware-owned behavior.

The observable contract must survive the Part 3 replacement. Device tests
therefore assert exact workload results and semantic sanitizer outcomes, not
prototype mechanisms such as patch counts, chosen registers, helper layouts,
instruction sequences, or code-cave placement.

The common cross-target suite should cover at least:

- clean execution and preservation of an exact workload oracle;
- racy and ordered native-LDS and group-FLAT accesses;
- exact overlap, adjacent non-overlap, and multidimensional ownership;
- workgroup barriers and deliberately broken barrier ordering;
- atomic release/acquire communication and deliberately broken atomic ordering;
- shared helpers, multiple execution owners, and nontrivial control flow;
- register pressure, forced spills, zero-EXEC paths, and live-register
  preservation;
- dynamic private stacks and private/group segment growth;
- expected access, barrier, atomic, and fence coverage;
- required diagnostics, forbidden diagnostics, overflow, and dynamic
  completeness; and
- post-instrumentation health on physical devices.

Run the semantic core under the SuperCollider flavor and all three MOI engines:
Record/Replay, Sampled, and Inline Shadow. Do so wherever the capability
contract says the combination applies. Add small target-specific extensions
for real semantic differences, such as CDNA
singleton barriers, RDNA4 split barriers, `gfx1250` cluster barriers and
ordered LDS atomics, target-specific LDS widths, and native VGLOBAL forms.
Equal semantic coverage matters; equal raw test counts do not.

Make registration table-driven. Each scenario should declare its applicable
targets and engines, expected coverage and diagnostic outcome, target-specific
requirements, and the end-to-end workload or regression from which it was
reduced. Audit this manifest against the typed capability contract so that a
new supported form cannot silently lack device coverage.

When end-to-end validation exposes a defect, reduce it to the smallest device
fixture that retains the relevant compiler, resource, control-flow, or
synchronization shape. Add that fixture on every semantically applicable
target, not just the architecture where the defect was found.

### Cross-architecture exploration strategy

Performing five independent end-to-end workload explorations would consume too
much time and would tend to produce five different test suites. Instead, do the
primary exploration on the three architectures that give the most useful
initial spread:

- CDNA5 / `gfx1250`;
- RDNA4 / `gfx1201` (`gfx12`); and
- CDNA4 / `gfx950`.

Abstract the device-level properties discovered there and generalize them to
RDNA3 / `gfx1100` and CDNA3 / `gfx942`. This is not a one-way mapping: useful
reductions should be cross-pollinated across all architectures whenever their
semantics apply, including between CDNA and RDNA. Architecture-specific
exceptions must be justified by the capability contract rather than by the
origin of the test.

### Completion criteria

- A documented, bounded device-conformance command runs every supported target
  through RocJitsu without external workload repositories.
- The matching native suite runs on any locally available supported device.
- Every common semantic capability and every target extension has an explicit
  device-test disposition: covered, not applicable, or a tracked gap.
- Tests fail if the intended kernel is not patched, coverage is incomplete, an
  oracle changes, or diagnostics differ from the declared contract.
- The suite is small enough to run routinely while developing ConSan.
- End-to-end campaigns remain the final qualification authority; the new tier
  does not claim to replace them.

## 2. Finish the basic `gfx950` prototype

**Status: active**

### Problem

`gfx950` support has substantial implementation and historical validation
evidence, but it is not yet a reliable current-tip prototype across the four
ConSan modes. The present compact physical baseline passes five of nine tests;
Inline Shadow forced spill, barrier forced spill, atomic forced spill, and
dynamic-stack Inline Shadow currently cause deterministic GPU memory-access
faults. Their RocJitsu counterparts pass, which also demonstrates why both
physical and simulated device coverage are required.

Historical artifacts are useful diagnostic evidence but do not establish the
state of the current branch. The `gfx950` workload matrix must be reassessed
and fixed on the current source and hook.

### Proposed approach

1. Reproduce and isolate the four compact physical failures. Determine whether
   each fault originates in instruction lowering, spill layout, private/group
   segment growth, kernel-descriptor rewriting, dispatch metadata, or runtime
   integration. Convert every fix into a shared simulator/native
   device-conformance regression.
2. Establish clean, current-tip baselines for the registered `gfx950`
   workloads before instrumentation. Record exact source, build, runtime, hook,
   device, command, timeout, and oracle provenance.
3. Qualify each workload under the SuperCollider flavor and all three MOI
   engines: Record/Replay, Sampled, and Inline Shadow.
4. For every workload/mode cell, require the workload oracle, intended
   code-object identity, nonzero applicable instrumentation, static and dynamic
   coverage accounting, expected diagnostics, bounded execution and memory,
   containment, cleanup, and post-run GPU health.
5. Exercise reviewed fault cases in addition to clean runs. A mode is not
   qualified merely because it preserves clean output; its claimed detection
   behavior and known limitations must also be demonstrated.
6. Measure overhead only after correctness and completeness are established.
   Treat rejection, timeout, GPU fault, silent non-instrumentation, and
   incomplete coverage as distinct outcomes rather than collapsing them into a
   generic failure.
7. Update the `gfx950` status ledger from fresh artifacts. Do not promote a
   workload cell using evidence from a different revision, architecture, or
   instrumentation profile.
8. Produce a `gfx950` counterpart to
   [GFX1201_EMPIRICAL_STUDY.md](GFX1201_EMPIRICAL_STUDY.md). Base its comparison
   of the four modes on fresh, reproducible `gfx950` correctness, coverage,
   fault-detection, overhead, containment, and implementation-complexity
   evidence rather than carrying over conclusions from `gfx1201`.

Start with the compact reproductions and checked-in HIP workloads, then proceed
through the portable hip-moi cases and the larger workloads registered by
`consan_validation.py`. Prefer fixes in shared semantic, resource, and runtime
layers; introduce a `gfx950` special case only when the ISA or ABI genuinely
requires one.

### Completion criteria

- All compact `gfx950` physical device-conformance tests pass without a GPU
  fault and leave the device healthy.
- The checked-in simulator and physical suites agree wherever they exercise the
  same contract, with documented exceptions for simulator limitations.
- Every registered `gfx950` end-to-end workload has a fresh result for all four
  modes.
- Supported cells meet the correctness, coverage, fault, containment, resource,
  timeout, and provenance requirements in [VALIDATION.md](VALIDATION.md).
- A checked-in `gfx950` empirical study, with generated supporting results where
  appropriate, documents the evidence and derives target-specific guidance
  comparable in scope to the existing `gfx1201` study.
- Any remaining unsupported cell has a precise, reproducible technical reason
  and a reduced checked-in regression when possible.

## 3. Replace the prototype with a production-grade implementation

**Status: planned; begin incrementally once the device-conformance safety net is
usable**

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

1. Establish the common device-conformance manifest and close the current
   compact `gfx950` physical failures.
2. Reduce findings from `gfx1250`, `gfx1201`, and `gfx950` workload exploration
   into portable device tests, then generalize and cross-pollinate them to
   `gfx1100` and `gfx942`.
3. Complete the current-tip `gfx950` four-mode end-to-end qualification.
4. Use the resulting multi-architecture safety net to replace prototype layers
   incrementally, deleting superseded code as each production layer lands.
5. Re-run the full device matrix continuously and the expensive end-to-end
   campaigns at qualification milestones.
