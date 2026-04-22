#ifndef HOTSWAP_TRANSPILER_C5_PREDICATE_CHAIN_CLASSIFIER_HPP
#define HOTSWAP_TRANSPILER_C5_PREDICATE_CHAIN_CLASSIFIER_HPP

#include <string>

namespace llvm {
class Function;
} // namespace llvm

namespace transpiler {

// ============================================================================
// Narrow-O1 post-mem2reg IR-level classifier for the Class-5
// predicate-chain class. See hotswap/docs/modrep-predicate-chain.md §5
// (O1) for the design and §9.6 for the Phase-2 narrowing rationale.
// ============================================================================
//
// PROBLEM. Under modulo-replication (`ModuloReplicationProjection`),
// a wave32 source kernel's `llvm.amdgcn.workitem.id.x()` value
// reaches an `icmp` against a compile-time constant K with
// `0 < K <= W_s - 1`, where the icmp result gates a global-memory
// side effect. The predicate is structurally "which lane-position-
// within-source-wave am I" — source wave 0's lane L and target
// replica-1's lane L+W_s share the same source EXEC bit but have
// different architectural `tid` values, so `tid < K` evaluates
// differently between the replicas. Scan-stage guards
// (`tid >= 2^s` for Kogge-Stone), half-wave broadcasts
// (`if tid < W_s/2`), and quad-level masks (`tid & 3`) all fit
// this shape. The `canary_bpermute_scan_fp32` recipe
// (`compare_correctness` Triton suite) is the canary that pins it.
//
// NARROWING RULE (the key principled bit — Phase-2 evidence in
// `modrep-predicate-chain.md` §9.6). "Refuse any unmasked
// `tid → icmp → side-effect`" — the literal O1 wording — would also
// refuse currently-passing baselines (`vecadd_f16`, `rope_fp32`,
// elementwise Triton kernels) whose IR has structurally identical
// shapes but compares against a *dynamic* kernarg (`tid < N` with
// N a runtime size). The constant-K-only rule distinguishes
// scan-stage / broadcast / lane-position gates (refuse) from
// bounds-against-kernarg (don't refuse), with the soundness
// direction preserved: false positives would refuse a safe kernel
// that happens to compare `tid` against a small compile-time K
// (benign); false negatives would only occur on dynamic-operand
// icmps, which are by construction bounds checks rather than
// lane-position gates.
//
// FORWARD USE-CHAIN WALK. Starting from each
// `@llvm.amdgcn.workitem.id.x()` call, traverse transitive uses:
//
//   * `and %v, K` where K is a compile-time constant with
//     `K <= W_s - 1`: the masked arm. The result folds replica-1's
//     `tid` values back onto `[0, W_s)`; downstream uses cannot
//     reintroduce per-replica divergence. Stop walking this user.
//     (This is the §5.6.2 `wave_id` lift's mask shape and the SPE
//     prelude's own `lane_id & (W_s-1)` — both produce safe values.)
//   * `icmp pred, %v, %C` or `icmp pred, %C, %v` where `%C` is a
//     compile-time constant with `0 < |C| <= W_s - 1`: REFUSE.
//   * `icmp pred, %v, %other` where `%other` is not a compile-time
//     constant, OR the constant exceeds W_s - 1, OR is zero (which
//     is typically a null-check or mask-extraction): treat the
//     icmp result as a bounds-check sink — downstream uses are
//     whatever they are, but the icmp itself is not a lane-position
//     gate. Stop walking this user.
//   * Pure propagators (cast, binop except the masking `and` above,
//     phi, select, gep, freeze, insert/extract{element,value},
//     shufflevector): push the result on the worklist.
//   * Anything else (calls, stores, loads, branches, unknown):
//     sink. Stop walking this user — the user is not an icmp, so
//     the refusal rule doesn't fire here, and the value is either
//     consumed for computation or stored verbatim (both fine under
//     MODREP for the purposes of this class).
//
// DIRECTION GATE. Runs only when `targetWaveSize > sourceWaveSize`.
// Same-wave or narrowing: no replica-1 exists, no lane-position
// ambiguity, the classifier is a structural no-op.
//
// COVERAGE + INTENTIONAL NON-COVERAGE. Per §9.6 of the design doc:
//
//   * `canary_bpermute_scan_fp32`: Kogge-Stone scan stages emit
//     `icmp ult i32 K, %tid` with K in {1, 3, 7, 15} (all
//     `<= W_s - 1 = 31`). Classifier refuses. Flips silent-WRONG
//     to loud-refused — the principled O1 outcome.
//   * `rmsnorm_fp32`, `swiglu_fp32`, `corpus_layernorm_fp32`: their
//     kernel-level icmps compare `tid` against a dynamic kernarg
//     (`%arg4` / `%arg2`), not a compile-time constant. Classifier
//     does NOT refuse. Their actual miscompile is documented
//     separately in §9.5 (convergent-cross-lane-op inactive-lane
//     leak) and §9.6 (MODREP's EXEC-replication broken for
//     `num_warps > 1`) — both orthogonal classes, out of scope
//     for this classifier.
//   * `vecadd_f16`, `rope_fp32`, `canary_dpp_compound_add_fp32`
//     (baselines, currently MATCH): no icmps against compile-time
//     constants ≤ W_s-1. Classifier does NOT refuse. Baselines
//     stay green.
//
// RELATIONSHIP TO §5 O2 (the mask rewrite). O2 is explicitly
// deferred per §9.6's Phase-2 finding that the mask rewrite's
// shape is not semantically correct for the norm-family failing
// recipes nor for the scan-shaped recipe under its actual launch
// configuration. This classifier is refuse-only; there is no
// `RewriteId` entry paired with `ObstructionKind::WorkitemIdPredicateChain`
// today. A future design iteration can add one if the
// predicate-chain class is shown to have a principled rewrite
// that flips specific recipes to MATCH without regressing any
// other recipe.

struct PredicateChainClassifierReport {
  // True iff the classifier refused at least one
  // `workitem.id.x()` call's use chain. When true, the raiser
  // translates this into a
  // `RaiseFailure::crossWavePredicateChain` refusal. When false,
  // every `workitem.id.x()` call in the function is safe under
  // the narrowing rule above.
  bool refused = false;

  // Detail string naming the first failing call's icmp and the
  // constant operand that triggered the refusal. Empty iff
  // `!refused`. Stable-enough-for-lit substring format (see the
  // lit fixtures under `lit_tests/c5_predicate_chain_*`).
  std::string refusalDetail;

  // Number of `@llvm.amdgcn.workitem.id.x()` calls the classifier
  // visited. Informational; used by the lit fixtures to assert
  // "classifier ran at all" in the non-refusal negative case.
  unsigned visitedCalls = 0;
};

// Classify every `@llvm.amdgcn.workitem.id.x()` call in F against
// the narrowing rule above. No-op (returns
// `!refused && visitedCalls == 0`) when either:
//
//   * `targetWaveSize <= sourceWaveSize` — same-wave / narrowing
//     directions have no replica-1 and no predicate-chain risk.
//   * `waveNative == true` — the WaveNativeProjection's
//     `init_whole_wave` + per-lane modeled EXEC model eliminates
//     MODREP's "target wave = R replicas of source wave 0 sharing
//     source EXEC" assumption. Under WaveNative each target lane is
//     its OWN source lane (tid = architectural tid = WG-level
//     source tid for num_warps > 1 kernels; modeled EXEC bit =
//     per-source-lane active state captured from the original HW
//     EXEC at entry). The refusal rationale — "target replica-1
//     evaluates the predicate differently from source wave 0's
//     lane L despite sharing EXEC" — cannot fire because there
//     are no replicas: each target lane evaluates the predicate
//     on its own source-tid, and its store is gated by its own
//     modeled EXEC bit. A kernel that would have been refused
//     under MODREP is safe under WaveNative by construction. See
//     wave-size-translation.md §5.6.1 for the WaveNative model and
//     modrep-predicate-chain.md §9.5 / §9.6 for the MODREP-specific
//     classes the classifier exists to catch.
PredicateChainClassifierReport classifyPredicateChain(
    llvm::Function &F, unsigned sourceWaveSize, unsigned targetWaveSize,
    bool waveNative = false);

} // namespace transpiler

#endif
