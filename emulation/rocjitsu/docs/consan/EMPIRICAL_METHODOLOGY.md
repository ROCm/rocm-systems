# ConSan empirical correctness and detection methodology

This document defines cross-cutting corpus, provenance, fault-detection, and
recommendation rules for ConSan evaluation on real AMDGPU kernels. Correctness
qualification is owned by [VALIDATION.md](validation/VALIDATION.md), while all
performance measurement is owned by the separate
[BENCHMARK.md](benchmark/BENCHMARK.md).

The methodology answers two separate questions:

1. Can the mode instrument and run an unmodified real workload completely?
2. Which precommitted concurrency defects does it diagnose, and with what
   uncertainty?

A mode is never tuned around a selected fault and then presented as an ordinary
configuration. Rejected workload/mode pairs and qualified misses remain part of
the result.

## Corpus admission

A target study must cover at least four independent source families. The corpus
should include production-shaped handwritten or generated kernels, directly
owned kernels with reviewable faults, a kernel with an independent CPU or
library oracle, and at least one compiler- or generator-produced final code
object. Focused microbenchmarks may explain a result but do not replace the
real-workload requirement.

Every admitted workload has:

- a bounded, production-relevant launch shape;
- an independent correctness oracle;
- a stable native command and input;
- pinned source, executable, and code-object identities;
- at least one ConSan-visible supported site; and
- a fixed per-process timeout and post-run device-health probe.

Tiny shapes used only to make a kernel compile or finish quickly are not
performance evidence. A smaller shape is valid only when it retains the
production execution structure being studied and that choice is recorded.

Ordinary measurements use the documented default behavior of the selected
mode. A site cap, kernel filter, manually selected register, forced spill,
reduced report, synchronization override, or static sampling selector
disqualifies an ordinary-support row. Parameter sweeps may be reported as
sensitivity experiments, separately from the ordinary result.

## Clean admission gate

Before fault trials, each workload/mode pair must:

1. pass the independent workload oracle;
2. report an applicable code object and expected kernel identity;
3. patch every admitted supported site without a hidden coverage limiter;
4. report static, dynamic, and analytical completeness;
5. emit no unexpected diagnostic, overflow, or dropped evidence; and
6. finish within the declared bound while leaving the device healthy.

A pair that fails any condition remains an explicit rejected result. It is not
timed, dropped from the corpus, replaced by an easier kernel, or made admissible
by weakening the full-coverage requirement.

## Frozen provenance

Every campaign records at least:

- the RocJITsu commit, dirty-tree state, hook hash, and build configuration;
- workload repository commits, submodule states, local patches, exact commands,
  inputs, launch dimensions, and correctness settings;
- executable and code-object hashes plus kernel identities;
- TheRock SDK, compiler, HIP runtime, kernel driver, and firmware identities;
- framework and generator package versions and module paths for generated
  workloads;
- selected KFD node, PCI identity, target, and visible-device environment;
- warmups, iterations, process count, timeout, and randomization seed; and
- available device temperature, clock, health, and competing-process
  observations.

Record resolved workgroup/cell strides and offsets, achieved bank geometry,
FLAT provenance policy, owner/dispatch operating point, and host-epoch selection
alongside the preset name. Names alone do not preserve meaning across policy
changes. Interpret completeness against the
[implemented assumptions](DESIGN.md#heuristics-and-their-failure-directions);
it does not independently validate those assumptions.

Required identity fields fail closed. Optional machine observations record an
explicit unavailable or failed-probe state rather than disappearing.

Create a new artifact root after any source, hook, binary, input, fault,
methodology, schema, or failed-preparation change. Never merge samples from
different checkpoints into one result. Resume may reuse complete rows only
when every stable identity and campaign setting still matches.

## Detection protocol

The fault corpus may combine exact final-ISA mutations with source-level faults
in directly owned kernels. Every fault is selected before the final campaign
and records:

- semantic class and mode applicability;
- expected independent-oracle effect;
- allowed diagnostic outcome;
- code-object hash, kernel, instruction offset, occurrence, original bytes, and
  replacement bytes for an ISA mutation; or
- source diff and disassembled binary delta for a source mutation.

An instrumentation-independent argument must establish the missing ordering
and execution of both conflicting endpoints before the row is admitted. Valid
evidence includes source/dataflow proof backed by final ISA, a harness access
witness, or a native fault-only control. An oracle failure is not required in
every trial because diagnosing a race before it changes the numerical output is
valuable.

Each fresh, contained process records attempted, mutation-admitted, and reached
counts separately. A reached trial has complete installation evidence plus at
least one of:

- a detector-owned diagnostic;
- an independent-oracle manifestation;
- a runtime access witness; or
- reviewed final-ISA proof that the completed workload unconditionally executed
  the selected instruction.

Process completion alone is not a reach witness. A timeout without direct reach
evidence does not enter the detection-rate denominator.

A final estimate uses at least 30 independent reached trials per applicable
fault/mode pair and reports hits, reached trials, and a Wilson 95-percent
confidence interval. Extend a predeclared row to 100 reached trials when the
30-trial interval cannot distinguish the recommendation categories. A
deterministic claim requires every final trial to diagnose the fault.

Report oracle manifestation separately from ConSan detection. Also record
contained failures, timeouts, post-trial device health, clean-run unexpected
diagnostics, and ConSan runtime stride and offset. A reached but
schedule-masked fault remains an explicit qualified miss.

## Implementation-complexity protocol

Source-line counts are an inventory, not a synthetic quality score. For each
mode, record:

- mode-dedicated and shared production/test surfaces;
- analysis, target-specific emission, relocation, register, spill, and
  descriptor paths;
- persistent and per-dispatch ABI state plus memory bounds;
- causal-window and synchronization models, or SuperCollider mismatch evidence;
- overflow, incompleteness, containment, and cleanup states;
- supported semantic families and typed exclusions; and
- target-porting and regression-test obligations.

Report qualitative maintenance risk with a source-grounded explanation.

## Recommendation rule

Use a Pareto argument across admission, correctness, detection value, GPU cost,
retained state, and implementation complexity. Do not average unrelated metrics
into one score.

- **Ordinary mode:** clean and complete across the required corpus, low
  false-positive risk, useful detection on important real faults, and acceptable
  operational and maintenance cost.
- **Supported opt-in:** unique useful evidence with materially higher
  operational complexity or nondeterminism.
- **Narrowed mode:** useful only for precisely stated semantic families,
  workload scales, or diagnostic workflows.
- **Remove from ordinary support:** dominated in actionable detection and cost,
  or unable to meet correctness and completeness without workload-specific
  controls.

Operational cost may inform a support recommendation, but performance claims
and thresholds come only from the benchmark contract. Preserve uncertainty and
qualified misses rather than tuning a mode until it detects the chosen fault.
