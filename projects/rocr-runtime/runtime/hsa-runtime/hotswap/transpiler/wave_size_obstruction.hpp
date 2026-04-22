#ifndef HOTSWAP_TRANSPILER_WAVE_SIZE_OBSTRUCTION_HPP
#define HOTSWAP_TRANSPILER_WAVE_SIZE_OBSTRUCTION_HPP

#include "decoded_inst.hpp"
#include "isa_profile.hpp"
#include "raise_failure.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace transpiler {

struct MCState;

// ============================================================================
// Wave-size obstruction classifier.
//
// Implements the obstruction catalog + 3-outcome decision procedure
// specified in hotswap/docs/wave-size-translation.md §§6–7. Given a
// decoded instruction stream for a source kernel whose target wave
// size differs from the source's, this pass produces an
// `ObstructionReport` whose sites enumerate every construct in the
// kernel that violates the wave-size-obliviousness theorem (see
// wave-size-translation.md §6 for the precise definition), tagged
// with (a) the obstruction class, (b) the rewrite entry in §7's
// landed-rewrites table (if any) that would discharge it, and
// (c) whether the rewrite is implemented in the current raiser.
//
// The decider function `decideProjection` consumes that report and
// returns either a projection to run (outcome a / b) or a structured
// `RaiseFailure` to propagate (outcome c). The projection today is
// always `ModuloReplicationProjection` — wave-size-translation.md
// §2.2's coverage ladder envisions `ThreadLoopProjection` /
// `ScalarizationProjection` as future rungs, but no corpus kernel
// reaches them (see hotswap/docs/gpt-oss-derisking.md §9.1–9.3).
//
// Analysis strategy — syntactic for now, dataflow later.
// ============================================================================
//
// TODO(dataflow-upgrade): The current implementation is a syntactic
// classifier: it walks the decoded instruction stream and flags
// obstruction sites by matching on `SemOp` / `rawMnemonic` / operand
// immediates. This is sound-not-complete for some kinds:
//
//   - MbcntHiLaneIdLeak: matched directly by SemOp
//     (`V_MBCNT_HI_U32_B32`); exact.
//   - OutOfRangeLaneOperand (readlane/writelane bounds): we inspect
//     the lane-operand MCOperand for a constant; static-constant
//     operands are exact, dynamic operands are NOT flagged today
//     (Triton's softmax / matmul use writelane with dynamic
//     operands that happen to be in-bounds at runtime — see
//     gpt-oss-derisking.md §7.1).
//   - Cross-lane shuffles (FullWaveRotate / LaneGroupShuffle /
//     DsSwizzle / DppCrossLane / DsBpermuteGather): matched on
//     SemOp / `rawMnemonic` — exact.
//   - NonCommutativeAtomic: matched by mnemonic substring
//     (`cmpswap`, `atomic_swap`, `atomic_xchg`). Exact at the
//     mnemonic level.
//   - CmpxFromLaneId / SaveExecFromLaneId: the principled check
//     asks "does this v_cmpx / s_*_saveexec's source-operand
//     dataflow chain contain a value derived from
//     `amdgcn.mbcnt.{lo,hi}`?". The syntactic approximation
//     implemented here flags the EXEC writer whenever it co-occurs
//     with any v_mbcnt_* in the same kernel, regardless of whether
//     the mbcnt value actually flows into the EXEC-writer's
//     operands. This over-approximates: kernels that use mbcnt for
//     an unrelated purpose and also have a bounds-check v_cmpx
//     would be flagged and refused, even though modulo-replication
//     would emit correct code for them.
//
// The sound direction of the imprecision is preserved: false
// positives (refuse a safe kernel) are benign; false negatives
// (accept an unsafe kernel) are the silent-miscompile failure mode
// this classifier exists to eliminate. The dataflow upgrade will
// plumb LLVM Uniformity Analysis (plus the operand-aware target hook
// from llvm/llvm-project#137639 to mark `lane_id mod W_s` as
// uniform-across-replicas) over the RAISED IR — i.e. after Phase 2,
// not at the decoded-instruction level — and replace the syntactic
// co-occurrence heuristic with a precise dataflow query.
//
// The syntactic classifier is the minimum viable unit that catches
// every obstruction kind on every kernel in the GPT-OSS /
// hipBLASLt / Gluon corpora (gpt-oss-derisking.md §4); the dataflow
// upgrade is a refinement to shrink the false-positive set, not a
// correctness blocker.

// ----------------------------------------------------------------------------
// Obstruction taxonomy.
//
// Each enum value names the *specific failure mode* the classifier
// detected. The Class 1..4 grouping from hotswap/docs/wave-size-
// translation.md §6 is preserved as the comment headers below; that
// grouping is the stable design-doc cross-reference, but it is not
// part of the code-level identity of an obstruction (a reader should
// not have to bounce to the doc to know what
// `ObstructionKind::DppCrossLane` means).
// ----------------------------------------------------------------------------

enum class ObstructionKind : uint8_t {
  None = 0,

  // ── Class 1 (wave-size-translation.md §6): absolute lane-ID leaks ──
  // The kernel exposes the absolute target-hardware lane position
  // through one of these constructs; under modulo-replication the
  // value diverges from what the wave32 source intended.
  MbcntHiLaneIdLeak,        // v_mbcnt_hi_u32_b32 — no rewrite.
  OutOfRangeLaneOperand,    // v_readlane/writelane with static const operand >= W_s — no rewrite.
  TtmpWaveIdLeak,           // any source read of TTMP8 under cross-widening.
                            // raiser.cpp seeds ttmp8 with `(workitem.id.x >> 5)
                            // << 25` so bits [29:25] carry the per-lane
                            // `wave_id_in_workgroup`. That value is a function
                            // of the *target* absolute lane position, not of
                            // `lane_id mod W_s` — so any downstream computation
                            // that reads ttmp8 and folds the result into an
                            // address / EXEC mask / wave-uniform SGPR is a
                            // Class 1 leak under modulo-replication and a
                            // wave-id collision under wave-native. No rewrite.
  WaveIdLiftScalarized,     // the canonical `s_bfe_u32 sDST, ttmp8, 0x50019`
                            // wave_id extraction (normally rescued by the
                            // handle_sop2.cpp pattern-lift into a per-lane
                            // divergent VGPR value) flows into a
                            // v_writelane_b32 / v_readlane_b32 scalar-source
                            // operand in a kernel that ALSO contains WMMA.
                            // The cross-lane primitives scalarise their
                            // scalar operand via backend readfirstlane, which
                            // collapses the per-source-wave distinction the
                            // lift introduced (source_wave[k]'s wave_id=k
                            // becomes uniform across the target wave = 0).
                            // WMMA forecloses the ThreadLoopProjection escape
                            // hatch (§5.2 requires the full target wave
                            // simultaneously), so there is no correct
                            // projection today. No rewrite.

  WorkitemIdPredicateChain, // post-mem2reg IR-level class:
                            // `llvm.amdgcn.workitem.id.x()` flows into an
                            // `icmp` whose other operand is a compile-time
                            // constant K with 0 < K <= W_s - 1, and the chain
                            // from the intrinsic to the icmp has NOT been
                            // AND-masked by (W_s - 1) somewhere. Such a
                            // predicate is lane-position-scoped (it partitions
                            // lanes by their position within a single source
                            // wave), which is wave-size-sensitive under
                            // modulo-replication: source wave 0's lane L and
                            // target replica-1's lane L+W_s share the same
                            // source EXEC but have different architectural
                            // tids, so a predicate like `tid < 16` evaluates
                            // differently between the replicas and the store-
                            // gating it controls commits to different slots.
                            // Only caught by the IR-level classifier in
                            // `c5_predicate_chain_classifier.{hpp,cpp}`,
                            // NOT by the MC-level obstruction walk
                            // (`workitem.id.x` is emitted by the raiser's
                            // Phase-4 init + handler lifts, never as a
                            // source-side SemOp). `buildObstructionReport`
                            // must never tag a `DecodedInst` with this kind.
                            // See hotswap/docs/modrep-predicate-chain.md §5
                            // (narrow-O1) for the principled derivation and
                            // §9.6 for the Phase-2 finding that narrowed the
                            // classifier from "any unmasked `tid → icmp →
                            // side-effect`" to "compile-time K ≤ W_s-1" so
                            // baseline Triton recipes
                            // (`vecadd_f16`, `rope_fp32`,
                            // `canary_dpp_compound_add_fp32`) don't refuse.

  // ── Class 2 (wave-size-translation.md §6): cross-lane shuffles whose
  //                                            semantics bake in the wave width ──
  FullWaveRotate,           // v_permlane64_b32 — no wave32 analogue, unrewritable.
  LaneGroupShuffle,         // permlane16 / permlanex16 / permlane*_swap_b32 — wave-size-translation.md §5.3 rows P2 / P4 (and the pending-table P4 entry for permlane32_swap).
  DsSwizzle,                // ds_swizzle_b32 — wave-size-translation.md §5.3 row P6.
  DppCrossLane,             // any `_dpp` variant — wave-size-translation.md §5.3 row P5.
  DsBpermuteGather,         // ds_bpermute_b32 — wave-size-translation.md §5.3 row P1 (handler landed).

  // ── Class 3 (wave-size-translation.md §6): replica races on shared state ──
  // Modulo-replication introduces racers on the same address from
  // target lanes i and i + W_s; for non-commutative atomics this
  // produces an outcome the source program never expressed.
  NonCommutativeAtomic,     // atomic_cmpswap / atomic_swap / atomic_xchg — no rewrite.

  // ── Class 4 (wave-size-translation.md §6): lane-predicated EXEC writes ──
  // The EXEC mask the kernel writes depends on the absolute lane
  // position; under modulo-replication the projection does not
  // reproduce the source's intent.
  CmpxFromLaneId,           // v_cmpx co-located with v_mbcnt_* — syntactic approximation.
  SaveExecFromLaneId,       // s_*_saveexec_b32 co-located with v_mbcnt_* — same shape.
};

// Identifier for the rewrite rule that would discharge an obstruction
// site. Names follow the "P-item" convention enumerated in the
// cross-lane rewrite table at hotswap/docs/wave-size-translation.md
// §5.3 (and partitioned into landed / pending / unrewritable in §7).
enum class RewriteId : uint8_t {
  None = 0,                 // no rewrite available (outcome-c class).
  P1_DsBpermute,            // llvm.amdgcn.ds.bpermute lift.
  P2_PermLane16,            // llvm.amdgcn.permlane16 lift.
  P3_PermLane64,            // (reserved; v_permlane64 has no rewrite, see C2_PermLane64).
  P4_PermLaneSwap,          // LDS round-trip or permlane16-pair lowering for *_swap variants.
  P5_DppModifier,           // llvm.amdgcn.update.dpp lift.
  P6_DsSwizzle,             // llvm.amdgcn.ds.swizzle lift.
  LaneOpBoundsValidator,    // raise-time operand-range check for readlane/writelane.
  PostRaiseCrossLaneRewrite,// post-mem2reg rewrite of cross-widen-divergent
                            // writelane/readlane sites into select / ds.bpermute
                            // (rewrite_cross_lane_divergent.{hpp,cpp}, flagged on
                            // via `--enable-writelane-rewrite`). Tags the
                            // WaveIdLiftScalarized site as "implemented rewrite
                            // available" instead of "refuse outright" so the
                            // classifier lets the kernel through to Phase 6.5.
};

// Human-readable short label for an `ObstructionKind` — used in the
// classifier's diagnostic trace and in lit-test STDERR matches.
// Stable enough to assert on substrings (see lit_tests/c1_*..c4_*).
const char *obstructionKindName(ObstructionKind k);
const char *rewriteIdName(RewriteId r);

// ----------------------------------------------------------------------------
// One matched obstruction in the decoded stream.
// ----------------------------------------------------------------------------

struct ObstructionSite {
  // The instruction that triggered the match. Valid for the lifetime of
  // the DecodedInst stream passed to buildObstructionReport.
  const DecodedInst *inst = nullptr;
  ObstructionKind kind = ObstructionKind::None;
  RewriteId rewrite = RewriteId::None;
  // True iff the rewrite identified by `rewrite` is implemented in the
  // current raiser (handler lifts through the right intrinsic). Set per
  // the implementation status audited in wave_size_obstruction.cpp. The
  // decider uses this bit to choose between (a)/(b) "emit" and (c)
  // "refuse with pending-rewrite diagnostic".
  bool rewriteImplemented = false;
  // Short human-readable detail (e.g. "operand value 48 >= W_s=32").
  // Empty if the class/mnemonic alone is sufficient context.
  std::string detail;
};

// ----------------------------------------------------------------------------
// Aggregate report for a single kernel.
// ----------------------------------------------------------------------------

struct ObstructionReport {
  // Per LLVM coding standards, omit the explicit inline-element
  // count — `SmallVector<T>` picks a default suited to `sizeof(T)`.
  // GPT-OSS / hipBLASLt / Gluon corpora typically produce a handful
  // of sites per kernel (gpt-oss-derisking.md §5 worst case is ~13
  // in `bitmatrix_metadata_compute_stage1`); the default inline
  // buffer covers that comfortably.
  llvm::SmallVector<ObstructionSite> sites;

  // True iff any site has `rewrite == RewriteId::None` — i.e. we saw
  // an obstruction for which no rewrite in §4's table applies. This is
  // the outcome-(c) condition.
  bool hasUnrewritable() const;

  // True iff every site has a rewrite AND at least one is not
  // implemented in the current raiser. Outcome-(c) today (loud abort
  // with "rewrite pending" diagnostic), outcome-(b) once the matching
  // handler lands.
  bool hasPendingRewrite() const;

  // True iff the report is empty OR every site is
  // `rewriteImplemented = true`. Outcome-(a) / (b).
  bool isOblivious() const;

  // Return the first site whose rewrite is RewriteId::None.
  const ObstructionSite *firstUnrewritable() const;

  // Return the first site whose rewrite is non-None but not
  // implemented.
  const ObstructionSite *firstPending() const;
};

// ----------------------------------------------------------------------------
// Build an obstruction report for a kernel.
//
// Returns an empty report if src.waveSize == tgt.waveSize (same-wave
// translation has no wave-size obligations). Otherwise walks `insts`
// in decoded order and appends one site per matched obstruction.
//
// The walk is O(n) in the instruction count; memory is O(sites).
// Independent of IR emission — safe to run in the pre-translation
// phase before any LLVM module construction.
// ----------------------------------------------------------------------------

// `enableWritelaneRewrite` opts the classifier into treating the
// `WaveIdLiftScalarized` three-way co-occurrence as a site with an
// *implemented* post-raise rewrite (RewriteId::PostRaiseCrossLaneRewrite),
// not an unrewritable refusal. Default **true** as of the Triton-
// corpus graduation (see raiser.hpp for the full rationale); callers
// that want to pin the pre-rewrite REFUSE contract (lit fixtures for
// the `c1_wave_id_lift_scalarized` REFUSE sibling, etc.) pass `false`
// explicitly. See wave-size-translation.md §5.6.3.
ObstructionReport buildObstructionReport(llvm::ArrayRef<DecodedInst> insts,
                                          const MCState &mc,
                                          const ISAProfile &src,
                                          const ISAProfile &tgt,
                                          bool enableWritelaneRewrite = true);

// ----------------------------------------------------------------------------
// Render the report into a human-readable trace. Intended for
// LLVM_DEBUG (DEBUG_TYPE="wave-size-obstruction") in normal operation
// and for the pre-translation abort diagnostic when a refusal fires.
// The format is stable enough for lit tests to assert on substrings:
//
//   transpiler: projection decision for kernel '<name>':
//     source: <src-isa> (waveN) -> target: <tgt-isa> (waveM), R=<R>
//     obstructions found:
//       <class>: <count> site(s) [first @ offset 0x<hex>: <mnemonic>]
//         rewrite: <RewriteId> [implemented|pending]
//     outcome: <a|b|c>
//
// Returned as a std::string so the caller can route it to either
// LLVM_DEBUG or errs().
// ----------------------------------------------------------------------------

std::string renderObstructionTrace(const ObstructionReport &report,
                                    llvm::StringRef kernelName,
                                    llvm::StringRef srcIsa,
                                    llvm::StringRef tgtIsa,
                                    unsigned srcWaveSize, unsigned tgtWaveSize);

// ----------------------------------------------------------------------------
// Pick the first refusal-worthy site and package it into a structured
// RaiseFailure. Returns a RaiseFailure with `reason = None` iff the
// report is oblivious (no refusal needed). Caller's responsibility to
// have routed the trace through LLVM_DEBUG / errs() before calling.
// ----------------------------------------------------------------------------

RaiseFailure selectFailureFromReport(const ObstructionReport &report);

} // namespace transpiler

#endif
