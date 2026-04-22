// Unit tests for the narrow-O1 Class-5 predicate-chain classifier.
// See hotswap/docs/modrep-predicate-chain.md §5 (O1) and
// transpiler/c5_predicate_chain_classifier.{hpp,cpp}.
//
// The tests synthesise small LLVM Function IR modules directly (no
// code-object round-trip) so the classifier is audited in isolation
// from the raiser's MC-level pipeline. This matches the approach
// AGENTS.md (Missing targeted tests) recommends for
// `WaveProjection` / `ModuloReplicationProjection` primitives:
// build a tiny IR module, run the primitive, assert the verdict.
//
// Coverage matrix (each test listed corresponds to one row):
//
//   * TidDirectSmallConstRefuses — baseline refusal path: direct
//     `tid → icmp ult tid, 15 → br → side effect`. Constant K=15
//     is within (0, W_s-1=31]. Classifier must refuse.
//   * TidMaskedBeforeCmpAccepts — non-refusal path: `tid → and tid,
//     31 → icmp ult masked, 15 → br → side effect`. The mask
//     collapses replica-1 onto `[0, W_s)` before the predicate,
//     so the classifier must not refuse.
//   * TidSmallConstZeroAccepts — the `tid != 0` / `tid == 0` null-
//     check idiom. K=0 is NOT in (0, W_s-1], so classifier must
//     not refuse (zero is a mask-extraction / null-check shape,
//     not a lane-position gate).
//   * TidLargeConstAccepts — `icmp ult tid, 64` with K=64 > W_s-1.
//     Classifier must treat this as a bounds check and not refuse.
//     Pins the baseline-non-refusal contract for kernels whose
//     bounds check compares against a constant ≥ W_t.
//   * TidDynamicCmpAccepts — `icmp ult tid, %kernarg_N`. The
//     non-tid operand is a runtime SGPR value, not a
//     compile-time constant. Pins the baseline-non-refusal
//     contract for the vecadd_f16 / rope_fp32 shape (bounds
//     check vs. a kernarg).
//   * SameWaveDirectionGate — same-wave (source = target) MUST
//     return `!refused && visitedCalls == 0`. Pins the
//     direction gate.
//   * NarrowingDirectionGate — wave64 → wave32 (narrowing)
//     MUST return `!refused && visitedCalls == 0`. Pins the
//     direction gate.
//   * WaveNativeProjectionGate — same refusal-shaped kernel as
//     TidDirectSmallConstRefuses, but invoked with
//     `waveNative = true`. MUST return `!refused` while STILL
//     populating `observedSites` (the walk runs so raiser.cpp
//     can emit `LLVM_DEBUG` attribution breadcrumbs — the
//     classifier's `waveNative` gate SUPPRESSES refusal, it does
//     not skip the walk). Pins the structural projection gate:
//     under WaveNativeProjection each target lane is its own
//     source lane, so the MODREP replica-1 rationale does not
//     apply, but we still need the site list for debug
//     attribution. Protects the `enableWaveNative` thread-
//     through in raiser.cpp Phase 6.6 (and the LLVM_DEBUG
//     emission under WaveNative) from a future refactor that
//     accidentally drops the parameter or the walk.
//   * CrossSubtreeMaskedVsUnmaskedAccepts — `icmp f(tid)_unmasked,
//     g(tid)_masked` where both operands are tid-derived via
//     different subtrees: unmasked `tid+1` vs masked `tid & 15`.
//     Pins the FALSIFIED cross-subtree-refusal theory: the
//     classifier used to refuse this shape in an earlier
//     iteration, but compare_correctness (2026-04-21) showed
//     Triton's bounds-check idiom
//     (`icmp sgt tid-unmasked, tid-masked`) is in the baseline-
//     MATCH set under MODREP for `ult`/`ugt`/`slt`/`sgt`. Only
//     `eq`/`ne` variants would actually diverge (a different
//     per-predicate rule the classifier does not implement
//     today). The test pins the non-refusal so a future
//     iteration re-introducing the cross-subtree rule without
//     per-predicate reasoning fails here.
//   * IntrinsicPropagatorRefuses — `icmp @llvm.umin(tid, 100), 15`:
//     tid flows through a `umin` numeric intrinsic (not a
//     BinaryOperator). Pins the #8 intrinsic-propagator audit —
//     without explicit enumeration of numeric intrinsics as
//     propagators, the walk would stop at the `umin` call and
//     miss the downstream C5 site.
//   * NoCallsIsNoOp — function with no `workitem.id.x` intrinsic
//     call returns `!refused && visitedCalls == 0`.
//   * PhiPropagatesTidDerivation — tid flows through a phi whose
//     OTHER incoming is undef / constant; the masked arm MUST
//     still refuse through the phi chain. Pins that phi is a
//     pure propagator in the walk and doesn't break tid-
//     derivation tracking.
//   * MaskedPhiThroughUnmaskedArmRefuses — tid flows through a
//     phi with one UNMASKED arm (`%tid` directly) and one
//     MASKED arm (`%vand = and %tid, 31`). The classifier must
//     still refuse because the unmasked arm exists on the chain
//     — the `isSourceWaveMaskAnd` gate only stops propagation
//     through the AND user itself, not through phi unions with
//     the AND's consumer. This matches the shape observed in
//     `swiglu_fp32`'s IR (phi-arm asymmetry); the fixture
//     is not refused end-to-end because its icmp constant is
//     dynamic, which is caught by the `TidDynamicCmpAccepts`
//     test above.
//
// The tests use LLVM's IRBuilder + in-memory Module rather than
// `parseAssemblyString` to avoid a MemoryBuffer / LLVM-text-IR
// dependency. Each test constructs a tiny kernel-like function,
// calls `classifyPredicateChain`, and asserts the boolean verdict
// + the `visitedCalls` counter.

#include "../c5_predicate_chain_classifier.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using transpiler::classifyPredicateChain;
using transpiler::PredicateChainClassifierReport;

namespace {

// Canonical source / target wave sizes for all positive-direction
// tests: wave32 source → wave64 target, matching the gfx1250 →
// gfx942/gfx950 cross-widening this class exists to catch.
constexpr unsigned kSrcWs = 32;
constexpr unsigned kTgtWs = 64;

// Build a minimal module + function skeleton. Caller fills in the
// entry block via the returned IRBuilder. The function signature
// is `void kernel(i32* out, i32 bound)` — two args cover the
// common shapes the tests exercise (a store pointer and a dynamic
// bound).
struct Harness {
  LLVMContext ctx;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  IRBuilder<> B;
  BasicBlock *entry = nullptr;

  Harness() : B(ctx) {
    M = std::make_unique<Module>("c5_predicate_chain_test", ctx);
    auto *i32Ty = Type::getInt32Ty(ctx);
    auto *ptrI32 = PointerType::get(ctx, /*AS=*/1);
    auto *fnTy = FunctionType::get(Type::getVoidTy(ctx),
                                   {ptrI32, i32Ty},
                                   /*isVarArg=*/false);
    F = Function::Create(fnTy, Function::ExternalLinkage, "kernel",
                          M.get());
    entry = BasicBlock::Create(ctx, "entry", F);
    B.SetInsertPoint(entry);
  }

  // Emit a `call i32 @llvm.amdgcn.workitem.id.x()` into the current
  // insert block.
  CallInst *emitTid() {
    Function *intrin = Intrinsic::getOrInsertDeclaration(
        M.get(), Intrinsic::amdgcn_workitem_id_x);
    return B.CreateCall(intrin, {}, "tid");
  }

  // Emit a store that gates off a branch `br i1 %cond, spe_do, spe_skip`,
  // returning a handle on the original insert block's terminator-free
  // continuation so the caller can inspect the resulting function.
  // Wiring: we create `spe_do` and `spe_skip`, put a `store` in
  // `spe_do`, branch from the current block on `%cond`, and leave
  // the insert point at `spe_skip`.
  void emitStoreGate(Value *cond) {
    BasicBlock *spe_do = BasicBlock::Create(ctx, "spe_do", F);
    BasicBlock *spe_skip = BasicBlock::Create(ctx, "spe_skip", F);
    B.CreateCondBr(cond, spe_do, spe_skip);
    B.SetInsertPoint(spe_do);
    auto *i32Ty = Type::getInt32Ty(ctx);
    auto *ptrI32 = PointerType::get(ctx, /*AS=*/1);
    Value *outPtr = F->getArg(0);
    (void)ptrI32;
    B.CreateStore(ConstantInt::get(i32Ty, 42), outPtr);
    B.CreateBr(spe_skip);
    B.SetInsertPoint(spe_skip);
  }

  // Close the function with a `ret void`.
  void finish() {
    if (B.GetInsertBlock()->empty() ||
        !B.GetInsertBlock()->getTerminator())
      B.CreateRetVoid();
  }
};

} // namespace

// ---------------------------------------------------------------------
// Refusal path: direct `tid → icmp ult tid, 15 → br → store`.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidDirectSmallConstRefuses) {
  Harness H;
  Value *tid = H.emitTid();
  Value *cmp = H.B.CreateICmpULT(
      tid, ConstantInt::get(Type::getInt32Ty(H.ctx), 15), "c5_cmp");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
  EXPECT_NE(report.refusalDetail.find("compile-time constant 15"),
            std::string::npos)
      << "refusalDetail='" << report.refusalDetail << "'";
}

// ---------------------------------------------------------------------
// Non-refusal: `tid → and tid, 31 → icmp ult masked, 15 → br → store`.
// The mask collapses replica-1 onto [0, W_s); classifier stops walking.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidMaskedBeforeCmpAccepts) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *masked = H.B.CreateAnd(tid, ConstantInt::get(i32Ty, 31),
                                 "tid_masked");
  Value *cmp = H.B.CreateICmpULT(
      masked, ConstantInt::get(i32Ty, 15), "c5_cmp_masked");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
  EXPECT_TRUE(report.refusalDetail.empty());
}

// ---------------------------------------------------------------------
// Non-refusal: `icmp eq tid, 0` (null-check / mask-extract idiom).
// K=0 is structurally a different shape from a lane-position gate
// and the classifier intentionally does not refuse it.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidSmallConstZeroAccepts) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *cmp = H.B.CreateICmpEQ(
      tid, ConstantInt::get(i32Ty, 0), "c5_cmp_zero");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Non-refusal: `icmp ult tid, 64` (K=64 > W_s-1=31). The classifier
// treats this as a bounds check rather than a lane-position gate.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidLargeConstAccepts) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *cmp = H.B.CreateICmpULT(
      tid, ConstantInt::get(i32Ty, 64), "c5_cmp_large");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Non-refusal: `icmp ult tid, %arg1` (dynamic kernarg bound). Matches
// the vecadd_f16 / rope_fp32 baseline shape from compare_correctness.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidDynamicCmpAccepts) {
  Harness H;
  Value *tid = H.emitTid();
  Value *bound = H.F->getArg(1); // dynamic i32 kernarg
  Value *cmp = H.B.CreateICmpULT(tid, bound, "c5_cmp_dynamic");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Direction gate: source == target. Must no-op.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, SameWaveDirectionGate) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *cmp = H.B.CreateICmpULT(
      tid, ConstantInt::get(i32Ty, 15), "c5_cmp_same_wave");
  H.emitStoreGate(cmp);
  H.finish();

  // Same-wave (wave32 → wave32): direction gate triggers the early-
  // return in classifyPredicateChain BEFORE any tid-chain walk. The
  // kernel IR still contains the lane-position-scoped predicate,
  // but the classifier must stay quiet because modulo-replication
  // is a no-op when src == tgt.
  auto report = classifyPredicateChain(*H.F, kSrcWs, /*tgt=*/kSrcWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 0u);
}

// ---------------------------------------------------------------------
// Direction gate: narrowing (wave64 source → wave32 target). Must no-op.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, NarrowingDirectionGate) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *cmp = H.B.CreateICmpULT(
      tid, ConstantInt::get(i32Ty, 15), "c5_cmp_narrow");
  H.emitStoreGate(cmp);
  H.finish();

  // Narrowing (wave64 → wave32): same reason as same-wave — MODREP
  // has no replica-1, so the predicate evaluates consistently
  // across every target lane. Classifier must not touch it.
  auto report = classifyPredicateChain(*H.F, /*src=*/kTgtWs,
                                        /*tgt=*/kSrcWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 0u);
}

// ---------------------------------------------------------------------
// Projection gate: WaveNativeProjection MUST suppress the refusal,
// even on the exact IR shape that the default MODREP path would
// refuse. Pins the structural projection gate documented on the
// `waveNative` parameter in the header — regressions that accidentally
// drop the plumbing from loader/executable.cpp or raiser.cpp Phase 6.6
// fail this test.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, WaveNativeProjectionGate) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *cmp = H.B.CreateICmpULT(
      tid, ConstantInt::get(i32Ty, 15), "c5_cmp_wave_native");
  H.emitStoreGate(cmp);
  H.finish();

  // Sanity: under MODREP (waveNative=false) this exact IR refuses.
  // Narrows the WaveNative assertion to "the flag specifically is
  // what turns the refusal off", not "the IR happens to be safe".
  auto modrepReport =
      classifyPredicateChain(*H.F, kSrcWs, kTgtWs, /*waveNative=*/false);
  EXPECT_TRUE(modrepReport.refused);

  auto waveNativeReport =
      classifyPredicateChain(*H.F, kSrcWs, kTgtWs, /*waveNative=*/true);
  EXPECT_FALSE(waveNativeReport.refused);
  // Walk still runs — `visitedCalls` reflects the tid call count,
  // and `observedSites` names the C5 shape so raiser.cpp can emit
  // LLVM_DEBUG attribution. Only the refusal itself is suppressed.
  EXPECT_EQ(waveNativeReport.visitedCalls, 1u);
  EXPECT_EQ(waveNativeReport.observedSites.size(), 1u);
  EXPECT_TRUE(waveNativeReport.refusalDetail.empty());
}

// ---------------------------------------------------------------------
// Functions with no `workitem.id.x()` call must return the empty
// report. Pins the initial site collection's zero-case.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, NoCallsIsNoOp) {
  Harness H;
  // No tid read — just a constant-gated store. Nothing for the
  // classifier to walk.
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *cmp = H.B.CreateICmpULT(
      H.F->getArg(1), ConstantInt::get(i32Ty, 15), "not_a_tid");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 0u);
}

// ---------------------------------------------------------------------
// Refusal path through a phi node: `tid → phi [tid, entry] [tid, other]
// → icmp ult %phi, 15 → br → store`. Phi is a pure propagator per the
// classifier's walk rules; the refusal must fire through it.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, PhiPropagatesTidDerivation) {
  Harness H;
  Value *tid = H.emitTid();

  // Synthesise a two-predecessor phi: branch unconditionally from
  // entry to `join`, but also create a dummy `other` block that
  // merges in. This gives us a real phi whose only incoming is
  // `tid` from both sides.
  BasicBlock *other = BasicBlock::Create(H.ctx, "other", H.F);
  BasicBlock *join = BasicBlock::Create(H.ctx, "join", H.F);
  H.B.CreateBr(join);

  H.B.SetInsertPoint(other);
  H.B.CreateBr(join);

  H.B.SetInsertPoint(join);
  auto *phi = H.B.CreatePHI(tid->getType(), 2, "tid_phi");
  phi->addIncoming(tid, H.entry);
  phi->addIncoming(tid, other);

  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *cmp = H.B.CreateICmpULT(
      phi, ConstantInt::get(i32Ty, 15), "c5_cmp_phi");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Refusal through a phi with one MASKED and one UNMASKED arm: the
// classifier must still refuse because the unmasked arm reaches the
// icmp. This pins the behaviour observed in `swiglu_fp32`'s IR
// (phi-arm asymmetry from Triton's SPE diamond): the SPE
// diamond produces `phi [vand, spe_do], [tid, entry]` where the
// entry arm is unmasked, and the downstream icmp against the phi
// result is reachable from an unmasked tid path. (The swiglu end-
// to-end kernel is NOT refused because its icmp constant is
// dynamic; `TidDynamicCmpAccepts` pins that. This test pins the
// phi-arm asymmetry in isolation on a compile-time K so the
// refusal-through-phi semantics stay stable under future walker
// refactors.)
// ---------------------------------------------------------------------
TEST(C5PredicateChain, MaskedPhiThroughUnmaskedArmRefuses) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);

  BasicBlock *maskedArm = BasicBlock::Create(H.ctx, "masked_arm", H.F);
  BasicBlock *join = BasicBlock::Create(H.ctx, "join", H.F);

  // Split on an arbitrary predicate that is NOT tid-derived, so the
  // branch itself is not a C5 site — only the phi at `join` is what
  // the classifier must audit. We use `arg1 != 0` for this.
  Value *splitCond = H.B.CreateICmpNE(
      H.F->getArg(1), ConstantInt::get(i32Ty, 0), "split_cond");
  H.B.CreateCondBr(splitCond, maskedArm, join);

  H.B.SetInsertPoint(maskedArm);
  Value *vand = H.B.CreateAnd(tid, ConstantInt::get(i32Ty, 31),
                               "vand");
  H.B.CreateBr(join);

  H.B.SetInsertPoint(join);
  auto *phi = H.B.CreatePHI(tid->getType(), 2, "tid_phi");
  phi->addIncoming(tid, H.entry);       // unmasked arm
  phi->addIncoming(vand, maskedArm);    // masked arm

  Value *cmp = H.B.CreateICmpULT(
      phi, ConstantInt::get(i32Ty, 15), "c5_cmp_phi_mixed");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Cross-subtree non-refusal: `icmp (tid+1)_unmasked, (tid&15)_masked`.
// Both icmp operands are tid-derived via different walks: operand 0
// is `add tid, 1` (unmasked), operand 1 is `and tid, 15` (masked).
// An earlier iteration of the classifier refused this shape on the
// theory that replica-0's `(L+1, L&15)` and replica-1's
// `(L+33, L&15)` evaluate the predicate differently. For
// `ult`/`ugt`/`slt`/`sgt` the specific value ranges make the
// predicate evaluate identically across replicas — e.g.
// `icmp ult (L+1), (L&15)` is false for every L in [0, 31] and
// `icmp ult (L+33), (L&15)` is also false for every L in [0, 31],
// so both replicas agree. Only `eq`/`ne` variants genuinely
// diverge (`icmp eq tid, (tid&15)` is true iff `L < 16`, which
// differs between replicas); that per-predicate tightening is not
// implemented today. compare_correctness 2026-04-21 confirmed
// `vecadd_f16`, `corpus_add_fp32`, `corpus_asin_fp32`, and
// `canary_dpp_reduce_fp32` all emit this shape under Triton's
// `icmp sgt tid_unmasked, tid_masked` bounds-check idiom and pass
// MATCH end-to-end under MODREP.
//
// This test pins the non-refusal so a future iteration that
// re-introduces the cross-subtree rule without per-predicate
// reasoning fails here.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, CrossSubtreeMaskedVsUnmaskedAccepts) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Value *plus1 = H.B.CreateAdd(tid, ConstantInt::get(i32Ty, 1),
                                "tid_plus_1");
  Value *masked = H.B.CreateAnd(tid, ConstantInt::get(i32Ty, 15),
                                 "tid_masked_15");
  Value *cmp = H.B.CreateICmpULT(plus1, masked, "c5_cmp_cross_subtree");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
  EXPECT_TRUE(report.refusalDetail.empty());
  EXPECT_TRUE(report.observedSites.empty());
}

// ---------------------------------------------------------------------
// Intrinsic-propagator refusal: tid flows through `@llvm.umin(tid,
// 100)` and then into an `icmp ult %umin, 15`. Pins #8 — without
// explicit enumeration of numeric intrinsics as propagators the
// walk would stop at the `umin` call and the downstream C5 icmp
// would be missed.
//
// Semantic check: `umin(tid, 100)` is NOT a mask (it clamps at 100,
// not at W_s-1=31). For replica-0 L in [0,31]: result = L. For
// replica-1 L+32 in [32, 63]: result = L+32 (since both < 100). So
// the umin output still carries per-replica divergence — classifier
// must treat it as unmasked-propagator.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, IntrinsicPropagatorRefuses) {
  Harness H;
  Value *tid = H.emitTid();
  auto *i32Ty = Type::getInt32Ty(H.ctx);
  Function *uminDecl = Intrinsic::getOrInsertDeclaration(
      H.M.get(), Intrinsic::umin, {i32Ty});
  Value *umin = H.B.CreateCall(
      uminDecl, {tid, ConstantInt::get(i32Ty, 100)}, "umin_tid_100");
  Value *cmp = H.B.CreateICmpULT(
      umin, ConstantInt::get(i32Ty, 15), "c5_cmp_through_umin");
  H.emitStoreGate(cmp);
  H.finish();

  auto report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(report.refused);
  EXPECT_EQ(report.visitedCalls, 1u);
  EXPECT_NE(report.refusalDetail.find("compile-time constant 15"),
            std::string::npos)
      << "refusalDetail='" << report.refusalDetail << "'";
}
