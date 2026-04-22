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
//     `swiglu_fp32`'s IR (§9.6 Phase-2 inspection); the fixture
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
// (Phase-2 inspection; modrep-predicate-chain.md §9.6): the SPE
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
