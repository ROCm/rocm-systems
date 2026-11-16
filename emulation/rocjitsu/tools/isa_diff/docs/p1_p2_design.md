# rj_isa_diff P1/P2: retained design (not implemented)

The handoff (section 14) asks P1 to be designed without blocking P0, and P2 to be a separate
task. This document records the design and the seams in the P0 code that it relies on.
**None of it is implemented.** P0 results never depend on it, and `proof_status` stays
`NOT_RUN`.

## P1: scalar control flow and proofs

### Scalar branches and multi-exit regions

- **Region shape.** A region becomes a single-entry, multi-exit subgraph of each side's
  CFG. Its blocks are reached from the region entry without leaving the region.
- **Exits.** The `exit_mapping` field (today one fixed sentence) becomes a list of pairs
  `(old exit, new exit, R_out)`. Exits are matched through common successor anchors,
  the same matched-instruction anchors the LCS alignment already validates for branch
  targets (`DiffResult::old_to_new`).
  - An exit that has no partner is `INCONCLUSIVE_MAPPING`, not a guess.
  - Exits may be taken with different PCs (handoff 5.3): only the logical exit is
    compared.
- **Runner.** `RegionRunner` today executes straight-line words. It would execute a
  small CFG instead: follow taken/not-taken edges inside the region, stop at an exit,
  and report which exit was reached.
  - `PC_WRITTEN` stays a harness error for indirect jumps.
  - `s_cbranch_*` is executed by RocJITsu as today, with the PC interpreted only to pick
    the successor.
- **Comparison.** Both sides must reach corresponding exits. Different exits is a new
  mismatch kind, `exit_divergence`. The comparison uses that exit's `R_out`.
- **Liveness.** `KernelModel` already gives per-instruction live-out sets, so each exit
  has its own live-out.
- **Composition.** The must-equality analysis already propagates per edge. Each exit's
  `R_out` feeds the successor of that exit.

### Scalar loops

- **Option 1: bounded testing.** Run with a step limit derived from the contract, for
  example a loop-count constraint.
  - A step-limit timeout is `INCONCLUSIVE`, never a pass. This holds even when both sides
    time out.
  - A single timeout is never read as non-termination.
- **Option 2: cutpoints.** Pick the loop header on both sides as a cutpoint, with a
  per-iteration relation `R_loop` taken from common-prefix analysis or user assumption.
  - The region is split into entry→header, header→header (body) and header→exit.
  - Each piece is tested under its own contract: the existing region machinery, with
    the header as a common cutpoint.
  - The two sides are not required to run in lockstep, so iteration counts may differ
    (for example after unrolling). Unrolling needs a cutpoint correspondence
    (k old iterations ↔ 1 new), which comes from a user assumption in P1. It is never
    inferred from outputs.
- **Invariants.** A candidate `R_loop` can be checked by bounded testing. It stays
  `ASSUMED` unless proven.

### SemaAST / SMT

- **Reuse.** The repository has a SemaAST pipeline for instruction semantics, used in
  Python code generation. The handoff permits reusing it.
  - Any instruction whose semantics it does not cover returns `unknown`.
  - Concrete `execute` callbacks are never translated into SMT automatically.
- **Query.** For a straight-line P0 region, the query is
  `C ∧ R_in ∧ ¬(R_out ∧ frame)` over bit-vectors for the locations in the contract
  (SCC as a 1-bit vector).
  - `unsat` gives `proof_status = PROVEN_UNDER_CONTRACT`, a new value, together with the
    model version, the contract and the scope.
  - A `sat` model is turned into a concrete `Sample`, replayed through RocJITsu
    (`replay_counterexample`), and reported only if the concrete run reproduces it.
  - A solver timeout is `unknown`.
- **Placement.** This is the "proof" layer of the result model. It never changes
  `test_status`.

### Reachability witnesses

- **Goal.** `witness_scope` already has `REACHABLE_AT_CUT` and
  `OBSERVABLE_KERNEL_DIVERGENCE`. To reach them, a local counterexample needs a
  kernel-entry state that executes the common prefix into the cut state.
- **Diagnostic path.** The P0 path (`--diagnostic-entry-prefix`) executes the prefix
  with arbitrary entry values. P1 would add the following:
  - a dispatch-ABI model for the entry SGPRs: kernarg pointer, workgroup ids, and
    private segment, from the descriptor's `kernel_code_properties` and
    `compute_pgm_rsrc2`;
  - SMEM loads from a modelled kernarg segment;
  - a search (or SMT query over the prefix) for entry values that reproduce the cut
    state of a local counterexample.
- **Outcome.** A reproduced witness upgrades `LOCAL_ONLY` to `REACHABLE_AT_CUT`. Running
  the rest of the kernel on both sides and comparing stores upgrades it to
  `OBSERVABLE_KERNEL_DIVERGENCE`.

## P2: separate tasks

### VALU and inactive lanes (implemented after P0)

Plain VALU is now executed: per-lane VGPR state, inactive lanes compared, per-side MODE,
lane-parallel inputs (see README "VALU support"). What remains for P2:

- byte-granular VGPR liveness, so a 16-bit store does not make the upper half
  observable;
- VGPR register maps for manual regions (SGPR maps exist);
- role-specific lane values for VGPR operands (shift amounts, BFE fields), beyond the
  catalogs;
- AGPRs, which come with MFMA.

### MFMA

- Semantics come from RocJITsu's execution, including the documented fp8 latency
  caveats.
- Accumulators (AGPRs) are added as state.
- Comparison is bit-exact under the strict ISA contract. A tolerance needs a separate,
  explicit refinement contract, never one relaxed after a mismatch (handoff 9).

### Memory, synchronisation and pending effects

Global and scalar memory is now executed (README "Memory"):
- one seeded image per case, shared by both sides;
- stored bytes compared as observable;
- pointer-shaped words for loaded memory bases;
- waits inside regions;
- in-order VMEM in the pending analysis.

What remains for P2:

- **Memory state.** Replace the flat random image with a points-to or bounds contract
  per base, so that aliasing between different bases is an explicit assumption rather
  than an accident of pointer values. Add LDS (per-workgroup), flat, buffer, scratch and
  atomics.
- **Pending effects.** The `pending_effects` analysis finds what is outstanding, and
  completion is eager inside regions. P2 would model completion order instead of
  rejecting conflicts:
  - RocJITsu's memory-wait diagnostics give eager results and separate readiness;
  - checkpoints do not serialise completion dependencies;
  - so the region runner must own the readiness state itself.
- **Synchronisation.** Barriers and waits are executed with a single-wave model and
  reported as limited to it.

### Complete kernel composition

Region results are composed into a kernel-level statement only when every hunk is
covered, every contract is `ESTABLISHED`, and the composition check `HOLDS` along every
path. P0 already computes each of these, so the statement is a conjunction over
`results.json`. It is not reported in P0 because a partial set must never read as a
kernel-level pass.

### Learned input generation (A/B with equal budgets)

- **Interface.** The seam already exists:
  - `InputPolicy::propose(features, contract, history, case)` and `observe(feedback)`;
  - per-case JSONL logs with opcode/effect summaries;
  - bounded novelty buckets.
- **Restrictions.** A learned policy may only propose inputs. The `ConstraintChecker`,
  the contract, the observable set and the program stay outside its reach (handoff 8).
- **Evaluation.** Equal-budget A/B against `BaselinePolicy`: same regions, same case
  counts, same mandatory sweep.
  - The metric is deduplicated, replay-confirmed mismatches per unit of cost, and
    distinct buckets.
  - Raw coverage of values is not a metric.
- **Ordering.** Training comes after the contract and executor are trusted. It is
  never a way to hide their gaps.
