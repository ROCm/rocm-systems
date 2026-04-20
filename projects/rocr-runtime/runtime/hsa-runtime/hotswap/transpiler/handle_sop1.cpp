#include "handlers.hpp"
#include "sem_op_attrs.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

namespace {

// Lower an analysis-enumerated indirect dispatch (the runtime i64
// value `targetInt` matches one of `targets` by setpc-analysis
// construction) into a cascade of cmp+br terminators rooted at the
// IRBuilder's current insertion block:
//
//   currBB:                                        ; B's current insert pt
//     %cmp_0 = icmp eq i64 %targetInt, <target_offset_0>
//     br i1 %cmp_0, label %bb_T0, label %dispatch_<off>_1
//   dispatch_<off>_1:
//     %cmp_1 = icmp eq i64 %targetInt, <target_offset_1>
//     br i1 %cmp_1, label %bb_T1, label %dispatch_<off>_2
//   ...
//   dispatch_<off>_{N-1}:
//     %cmp_{N-1} = icmp eq i64 %targetInt, <target_offset_{N-1}>
//     br i1 %cmp_{N-1}, label %bb_T{N-1},
//                       label %dispatch_<off>_unreachable
//   dispatch_<off>_unreachable:
//     unreachable
//
// On return the builder is positioned at the END of the unreachable
// block (after its terminator). Callers in this file return from the
// enclosing handler immediately afterwards, so no further code is
// emitted.
//
// Why a cascade and not `indirectbr` / `switch`:
//   LLVM's `FixIrreducible` pass (Transforms/Utils/FixIrreducible.cpp,
//   relied on by AMDGPU's structurizer) only handles `UncondBrInst`,
//   `CondBrInst` and `CallBrInst` as predecessors of an irreducible
//   cycle header — it `llvm_unreachable`s for any other terminator.
//   Tensilelite-shaped lifted CFGs (kernels using `s_swappc_b64` for
//   activation-function dispatch) place the dispatch block inside an
//   irreducible cycle, so an `indirectbr` (or `switch`) terminator
//   there crashes llc with "unsupported block terminator". A cascade
//   of `br` is FixIrreducible-compatible.
//
// Why we compare against an integer marker (target offset) rather
// than a `blockaddress` pointer:
//   The raiser's chain-terminator hook stores a per-predecessor marker
//   into the ret-pair SGPRs. An earlier revision of this fix stored
//   `ptrtoint(blockaddress(@kernel, %bb_<retAddr>)) to i64` so the
//   cascade could compare against a `blockaddress` constant and let
//   LLVM's SCCP+InstCombine fold the cmp to `i1 true` on the hot
//   path. In practice the hi/lo split imposed by `storeSGPR64` (AMDGPU
//   SGPR pairs are two i32 halves joined back with shl/or at the
//   dispatch site) defeats that fold across phi joins, and the
//   `BlockAddress` SDNode survives into AMDGPU ISel — which has no
//   pattern for materialising a `BlockAddress` as an i64 register
//   value (there is no relocation for "address of arbitrary BB inside
//   a kernel"). llc then aborts with
//     `LLVM ERROR: Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
//   Using the target's source-MC byte offset as a plain i64 marker
//   sidesteps the issue entirely: the marker is a normal integer
//   constant on every contributing predecessor path, folds cleanly
//   through mem2reg + SCCP + InstCombine, and `BlockAddress` only
//   appears as the `label` operand of the `br`, which DOES have a
//   codegen pattern (normal conditional branch). The hot-path folded
//   shape is identical to before (SimplifyCFG collapses the cascade
//   to a direct branch); the cold path does a bounded runtime integer
//   equality check before reaching the trap BB.
//
// `targetInt` must be of `ctx.i64Ty`; we assert this to catch
// regressions that forget to unpack the SGPR pair to i64 before
// calling.
//
// `targets` MUST be non-empty (the analysis never produces an empty
// dispatch set; the caller refuses Unresolvable sites earlier).
//
// `siteOffset` is the source-MC byte offset of the dispatching
// instruction; it is embedded in dispatch BB names to keep them
// unique across multiple dispatch sites in the same kernel.
void emitEnumeratedDispatch(RaiseContext &ctx, Value *targetInt,
                            ArrayRef<uint64_t> targets,
                            uint64_t siteOffset) {
  assert(!targets.empty() && "enumerated dispatch needs ≥1 target");
  assert(targetInt->getType() == ctx.i64Ty &&
         "enumerated dispatch expects i64 target marker");

  SmallString<32> sitePrefixStorage;
  raw_svector_ostream(sitePrefixStorage) << "dispatch_0x"
                                         << utohexstr(siteOffset);
  StringRef sitePrefix = sitePrefixStorage;

  IRBuilder<> &B = ctx.B;

  // Pre-create the unreachable trap block so we can name it
  // deterministically and reference it from the last cascade step.
  BasicBlock *unreachableBB = BasicBlock::Create(
      ctx.C, sitePrefix.str() + "_unreachable", ctx.kernel);

  for (size_t i = 0; i < targets.size(); ++i) {
    BasicBlock *targetBB = ctx.lookupBB(targets[i]);
    Constant *markerCI = ConstantInt::get(ctx.i64Ty, targets[i]);
    SmallString<48> cmpName;
    raw_svector_ostream(cmpName) << sitePrefix << "_cmp_" << i;
    Value *cmp = B.CreateICmpEQ(targetInt, markerCI, cmpName);

    BasicBlock *fallthroughBB;
    if (i + 1 < targets.size()) {
      SmallString<48> nextName;
      raw_svector_ostream(nextName) << sitePrefix << "_" << (i + 1);
      fallthroughBB = BasicBlock::Create(ctx.C, nextName, ctx.kernel);
    } else {
      fallthroughBB = unreachableBB;
    }
    B.CreateCondBr(cmp, targetBB, fallthroughBB);
    B.SetInsertPoint(fallthroughBB);
  }

  // Builder is now positioned at the start of unreachableBB. Emit the
  // unreachable terminator. The block is a BlockAddress-free terminal
  // sink — no other code emits into it.
  B.CreateUnreachable();
}

} // namespace

// SPE attribute registrations. Every SemOp listed here has been audited
// to route EXEC writes through `regs.storeExec` — directly for the
// SAVEEXEC family, via `writeReg{32,64,ExecWidth}` → `storeExec` for
// S_MOV_B{32,64} and S_NOT_B{32,64}. See AGENTS.md's SPE audit note
// before touching this list.
ArrayRef<SemOpAttrSpec> getHandlerSOP1Attrs() {
  static constexpr SemOpAttrSpec kAttrs[] = {
      {SemOp::S_MOV_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_MOV_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_NOT_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_NOT_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_AND_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_OR_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_XOR_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_ANDN2_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_ORN2_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
  };
  return kAttrs;
}

HandlerResult handleSOP1(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  if (sop == SemOp::S_MOV_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MOV_B64) {
    ctx.regs.writeReg64(ctx.B, op.dst(), op.src64(0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_AND_SAVEEXEC_B32) {
    Value *oldExec = ctx.regs.loadExec(ctx.B);
    Value *src = op.srcExecWidth(0);
    ctx.regs.writeRegExecWidth(ctx.B, op.dst(), oldExec);
    Value *newExec = ctx.B.CreateAnd(oldExec, src, "new_exec");
    ctx.regs.storeExec(ctx.B, newExec);
    hr.sccResult = newExec;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_OR_SAVEEXEC_B32) {
    Value *oldExec = ctx.regs.loadExec(ctx.B);
    Value *src = op.srcExecWidth(0);
    ctx.regs.writeRegExecWidth(ctx.B, op.dst(), oldExec);
    Value *newExec = ctx.B.CreateOr(oldExec, src, "new_exec");
    ctx.regs.storeExec(ctx.B, newExec);
    hr.sccResult = newExec;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_XOR_SAVEEXEC_B32) {
    Value *oldExec = ctx.regs.loadExec(ctx.B);
    Value *src = op.srcExecWidth(0);
    ctx.regs.writeRegExecWidth(ctx.B, op.dst(), oldExec);
    Value *newExec = ctx.B.CreateXor(oldExec, src, "new_exec");
    ctx.regs.storeExec(ctx.B, newExec);
    hr.sccResult = newExec;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ANDN2_SAVEEXEC_B32) {
    Value *oldExec = ctx.regs.loadExec(ctx.B);
    Value *src = op.srcExecWidth(0);
    ctx.regs.writeRegExecWidth(ctx.B, op.dst(), oldExec);
    Value *newExec = ctx.B.CreateAnd(oldExec, ctx.B.CreateNot(src), "new_exec");
    ctx.regs.storeExec(ctx.B, newExec);
    hr.sccResult = newExec;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ORN2_SAVEEXEC_B32) {
    Value *oldExec = ctx.regs.loadExec(ctx.B);
    Value *src = op.srcExecWidth(0);
    ctx.regs.writeRegExecWidth(ctx.B, op.dst(), oldExec);
    Value *newExec = ctx.B.CreateOr(oldExec, ctx.B.CreateNot(src), "new_exec");
    ctx.regs.storeExec(ctx.B, newExec);
    hr.sccResult = newExec;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_GETPC_B64) {
    // Stub: the destination's symbolic PC is irrelevant for raised
    // IR. For Pattern A chains, the chain's binary value is never
    // read after we emit the `br label %target`. For Pattern B call
    // sites, the call-site rewrite in raiser.cpp overwrites the
    // ret-pair with a `blockaddress` after the chain's high-half
    // terminator runs, so the binary PC the chain would otherwise
    // produce is also discarded. Writing zero keeps SROA happy and
    // surfaces any stray downstream read as an obvious-zero use that
    // would crash the verifier rather than silently miscompile.
    ctx.regs.writeReg64(ctx.B, op.dst(), ConstantInt::get(ctx.i64Ty, 0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SET_PC_I64) {
    // Look up the static analysis classification (Pattern A direct,
    // Pattern B enumerated-dispatch, or Unresolvable). Both patterns
    // emit a terminator into the current BB (and Pattern B / DispatchSet
    // also append a chain of dispatch sub-blocks via
    // `emitEnumeratedDispatch`); the raiser's BB-layout phase has
    // already promoted the next linear offset to a leader so subsequent
    // instructions land in their own BBs.
    if (!ctx.setpcAnalysis) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SOP1",
          "s_set_pc_i64 reached without a SetPcAnalysis "
          "(raiser pipeline is missing the Phase 1.1 step)");
      return hr;
    }
    auto it = ctx.setpcAnalysis->setpcSites.find(di.offset);
    if (it == ctx.setpcAnalysis->setpcSites.end()) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SOP1",
          "s_set_pc_i64 site not classified by SetPcAnalysis");
      return hr;
    }
    const SetPcSiteInfo &info = it->second;
    switch (info.kind) {
    case SetPcSiteInfo::Kind::DirectA: {
      ctx.B.CreateBr(ctx.lookupBB(info.directTarget));
      hr.handled = true;
      return hr;
    }
    case SetPcSiteInfo::Kind::IndirectB:
    case SetPcSiteInfo::Kind::DispatchSet: {
      // Both shapes lower to the same enumerated-dispatch cascade:
      // read the source SGPR pair as i64 (it holds the per-predecessor
      // marker — the resolved target's source-MC byte offset, written
      // either by the call-site chain-terminator hook in raiser.cpp for
      // IndirectB, or by the dispatch-target chain-terminator hook for
      // DispatchSet), then emit a cmp+br cascade against each
      // enumerated target offset. See `emitEnumeratedDispatch` above
      // for why this is a cascade of integer equality compares and not
      // `indirectbr` / a ptr-equality check against `blockaddress`.
      // The classification difference is purely semantic (return vs.
      // forward dispatch); the lowering mechanism is identical.
      Value *retVal = ctx.regs.loadSGPR64(
          ctx.B, static_cast<int>(info.indirectRetPairLowReg));
      retVal->setName("ret_pc_marker");
      emitEnumeratedDispatch(ctx, retVal, info.indirectTargets,
                             di.offset);
      hr.handled = true;
      return hr;
    }
    case SetPcSiteInfo::Kind::Unresolvable:
      hr.failure = RaiseFailure::unsupportedShape(di, "SOP1",
                                                  info.refusalReason);
      return hr;
    }
    hr.failure = RaiseFailure::unsupportedShape(
        di, "SOP1", "s_set_pc_i64 SetPcSiteInfo::Kind not handled");
    return hr;
  }
  if (sop == SemOp::S_SWAP_PC_I64) {
    // Branch-and-link. setpc_analysis classifies the call-target
    // pair (ssrc) as DirectA (chain resolves the absolute callee
    // offset intra-block), DispatchSet (inter-block dataflow
    // enumerates a bounded set of callee/branch targets reaching
    // this site through distinct CFG paths — the tensilelite
    // "activation function dispatcher" shape), or Unresolvable (the
    // pair's value cannot be statically enumerated).
    //
    // For both DirectA and DispatchSet we materialise
    // `blockaddress(@kernel, %BB_returnAddr)` cast to i64 into sdst
    // BEFORE the terminator (so a downstream Pattern B
    // `s_set_pc_i64 sdst` in the callee can consume that
    // blockaddress via its enumerated-dispatch cascade). The
    // terminator itself is `br label %BB_callee` for DirectA or a
    // cmp+br cascade against `[list]` (via
    // `emitEnumeratedDispatch`) for DispatchSet. The
    // chain-terminator hook in raiser.cpp has already rewritten
    // ssrc to hold the matching BlockAddress on every contributing
    // CFG path, so each cascade `icmp eq` resolves to a constant
    // after mem2reg + SCCP rather than running a true runtime check.
    //
    // IndirectB on a swap_pc is NOT a valid classification: by
    // construction, IndirectB describes a return-side use (the pair
    // was written by some caller's chain terminator in a different
    // block) and a swap_pc reading such a pair would be a
    // function-pointer dispatch through a return slot. The analysis
    // never produces IndirectB for a swap_pc site (the source pair
    // is the call target, not a return address), so we refuse
    // loudly if it ever appears.
    //
    // Unresolvable is refused loudly with the analysis's diagnostic.
    // See semop.hpp's S_SWAP_PC_I64 doc for the lowering contract.
    if (!ctx.setpcAnalysis) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SOP1",
          "s_swap_pc_i64 reached without a SetPcAnalysis "
          "(raiser pipeline is missing the Phase 1.1 step)");
      return hr;
    }
    auto it = ctx.setpcAnalysis->setpcSites.find(di.offset);
    if (it == ctx.setpcAnalysis->setpcSites.end()) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SOP1",
          "s_swap_pc_i64 site not classified by SetPcAnalysis");
      return hr;
    }
    const SetPcSiteInfo &info = it->second;
    if (info.kind == SetPcSiteInfo::Kind::Unresolvable) {
      hr.failure = RaiseFailure::unsupportedShape(di, "SOP1",
                                                  info.refusalReason);
      return hr;
    }
    if (info.kind == SetPcSiteInfo::Kind::IndirectB) {
      // Defensive: the analysis should never produce IndirectB for
      // a swap_pc site (a swap_pc's source pair is a call target,
      // not a return slot — IndirectB is the return-side use of
      // such a pair). If it ever does, refuse loudly so the
      // mismatch surfaces rather than silently mis-lowering.
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SOP1",
          "s_swap_pc_i64 classified as IndirectB by setpc_analysis "
          "(unexpected — IndirectB is the return-side classification "
          "for s_set_pc_i64; a swap_pc reaching this code path "
          "indicates an analysis invariant violation)");
      return hr;
    }
    // Materialise the return address marker (the offset of the BB
    // immediately after the swap) into sdst on both DirectA and
    // DispatchSet paths. Phase 1 of setpc_analysis has already
    // promoted `(di.offset + di.size)` to a leader so subsequent
    // linear instructions live in their own BB; we simply write the
    // offset of that BB as a plain i64 constant, and the downstream
    // IndirectB consumer of sdst reads it back and compares it in a
    // cmp+br cascade. See `emitEnumeratedDispatch` above for why we
    // use an integer marker rather than `ptrtoint(blockaddress(...))`
    // (AMDGPU ISel cannot materialise a `BlockAddress` as an i64).
    uint64_t returnAddr = di.offset + di.size;
    // Force the target BB to exist in the lift so the subsequent
    // `br label %bb_<returnAddr>` has a valid destination; we don't
    // use the returned BB pointer here.
    (void)ctx.lookupBB(returnAddr);
    Value *retMarker = ConstantInt::get(ctx.i64Ty, returnAddr);
    ctx.regs.writeReg64(ctx.B, op.dst(), retMarker);

    if (info.kind == SetPcSiteInfo::Kind::DirectA) {
      ctx.B.CreateBr(ctx.lookupBB(info.directTarget));
      hr.handled = true;
      return hr;
    }
    // DispatchSet: emit an enumerated-dispatch cascade through the
    // source pair into the enumerated targets. The source pair holds
    // a per-predecessor i64 marker (the resolved callee's source-MC
    // byte offset), rewritten by the chain-terminator hook in
    // raiser.cpp on each contributing predecessor path.
    Value *callTarget = ctx.regs.loadSGPR64(
        ctx.B, static_cast<int>(info.indirectRetPairLowReg));
    callTarget->setName("swap_call_target_marker");
    emitEnumeratedDispatch(ctx, callTarget, info.indirectTargets,
                           di.offset);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_NOT_B64) {
    hr.sccResult = ctx.B.CreateNot(op.src64(0), "not64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_NOT_B32) {
    hr.sccResult = ctx.B.CreateNot(op.src(0), "not32");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BREV_B32) {
    Function *brev = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::bitreverse, {ctx.i32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateCall(brev, {op.src(0)}, "sbrev"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_FF1_I32_B32) {
    Function *cttz = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::cttz,
                                                       {ctx.i32Ty});
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateCall(cttz, {op.src(0), ConstantInt::getTrue(ctx.i1Ty)},
                         "ff1"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_FF1_I32_B64) {
    Function *cttz64 = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::cttz, {ctx.i64Ty});
    Value *r = ctx.B.CreateCall(
        cttz64, {op.src64(0), ConstantInt::getTrue(ctx.i1Ty)}, "ff1_64");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateTrunc(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // s_ff0_i32_b{32,64} — find first 0 bit (lowest position), -1 if
  // none. SOPInstructions.td:278-279 omits an LLVM ISel pattern, so
  // we lower directly: invert the source and reuse the cttz path
  // shared with V_FFBL_B32 (handle_valu_small_ops.cpp), then patch
  // the all-ones-input case to -1 since llvm.cttz with
  // is_zero_poison=false returns the bitwidth (32 / 64) for a zero
  // input rather than the AMDGPU's -1 sentinel.
  if (sop == SemOp::S_FF0_I32_B32) {
    Function *cttz = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::cttz,
                                                       {ctx.i32Ty});
    Value *src = op.src(0);
    Value *inv = ctx.B.CreateNot(src, "ff0_inv");
    Value *raw = ctx.B.CreateCall(
        cttz, {inv, ConstantInt::getFalse(ctx.i1Ty)}, "ff0_raw");
    Value *isAllOnes = ctx.B.CreateICmpEQ(
        src, ConstantInt::getAllOnesValue(ctx.i32Ty), "ff0_allones");
    Value *res = ctx.B.CreateSelect(
        isAllOnes, ctx.B.getInt32(-1), raw, "ff0");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_FF0_I32_B64) {
    Function *cttz64 = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::cttz, {ctx.i64Ty});
    Value *src64 = op.src64(0);
    Value *inv = ctx.B.CreateNot(src64, "ff0_inv64");
    Value *raw = ctx.B.CreateCall(
        cttz64, {inv, ConstantInt::getFalse(ctx.i1Ty)}, "ff0_raw64");
    Value *rawTrunc = ctx.B.CreateTrunc(raw, ctx.i32Ty, "ff0_raw32");
    Value *isAllOnes = ctx.B.CreateICmpEQ(
        src64, ConstantInt::getAllOnesValue(ctx.i64Ty), "ff0_allones64");
    Value *res = ctx.B.CreateSelect(
        isAllOnes, ctx.B.getInt32(-1), rawTrunc, "ff0_64");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_FLBIT_I32_B64) {
    Function *ctlz64 = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::ctlz, {ctx.i64Ty});
    Value *r = ctx.B.CreateCall(
        ctlz64, {op.src64(0), ConstantInt::getTrue(ctx.i1Ty)}, "flbit64");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateTrunc(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_FLBIT_I32_B32) {
    Function *ctlz = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::ctlz,
                                                      {ctx.i32Ty});
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateCall(ctlz, {op.src(0), ConstantInt::getTrue(ctx.i1Ty)},
                         "flbit"));
    hr.handled = true;
    return hr;
  }
  // s_flbit_i32 / s_flbit_i32_i64 — signed find-leading-bit-not-equal-
  // to-sign-bit. SOPInstructions.td:296-298. Lower via the dedicated
  // llvm.amdgcn.sffbh intrinsic, which is overloaded on the source
  // integer type and selects back to v_ffbh_i32_e32 (or its 64-bit
  // pseudo equivalent) on AMDGPU. Hardware returns -1 for uniform-sign
  // input (0 or all-ones) — the intrinsic shares the same convention,
  // so no explicit zero-fixup is needed.
  if (sop == SemOp::S_FLBIT_I32) {
    Function *sffbh = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_sffbh, {ctx.i32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateCall(sffbh, {op.src(0)}, "sflbit"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_FLBIT_I32_I64) {
    Function *sffbh = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_sffbh, {ctx.i64Ty});
    Value *r = ctx.B.CreateCall(sffbh, {op.src64(0)}, "sflbit64");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateTrunc(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SEXT_I32_I8) {
    Value *v = ctx.B.CreateTrunc(op.src(0), ctx.i8Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateSExt(v, ctx.i32Ty, "sext8"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SEXT_I32_I16) {
    Value *v = ctx.B.CreateTrunc(op.src(0), Type::getInt16Ty(ctx.C));
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateSExt(v, ctx.i32Ty, "sext16"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CVT_F32_U32) {
    Value *r = ctx.B.CreateUIToFP(op.src(0), ctx.f32Ty, "s_cvt_f");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CVT_F32_I32) {
    Value *r = ctx.B.CreateSIToFP(op.src(0), ctx.f32Ty, "s_cvt_f");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CVT_U32_F32) {
    Value *s = ctx.B.CreateBitCast(op.src(0), ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateFPToUI(s, ctx.i32Ty, "s_cvt_u"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CVT_I32_F32) {
    Value *s = ctx.B.CreateBitCast(op.src(0), ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateFPToSI(s, ctx.i32Ty, "s_cvt_i"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ABS_I32) {
    Function *absF =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::abs, {ctx.i32Ty});
    Value *r = ctx.B.CreateCall(absF, {op.src(0), ctx.B.getFalse()}, "s_abs");
    ctx.regs.writeReg32(ctx.B, op.dst(), r);
    hr.handled = true;
    return hr;
  }
  // s_bitset{0,1}_b{32,64}: clear or set a single bit in sdst.
  //   B32: bit index = src0[4:0], dst and tied read are 32-bit.
  //   B64: bit index = src0[5:0], dst and tied read are 64-bit (src0 is
  //        still an SReg_32 per LLVM's `SOP1_64_32` class).
  // These are read-modify-write: the destination's prior value is the
  // tied `sdst_in` operand in TableGen (`SOP1_32` / `SOP1_64_32` with
  // `tied_in=1` and `Constraints = "$sdst = $sdst_in"`), and the bit
  // index arrives in `src0` at src index 0.  SCC is not updated.
  //
  // The MC layer collapses the tied `$sdst_in` slot — the AMDGPU
  // disassembler emits a 2-operand MCInst (`sdst`, `src0`) and the
  // tie is reconstituted only at MachineInstr lowering time. This
  // matches the S_CMOV_B{32,64} pattern below: the prior dst value
  // must be read explicitly via `regs.readReg{32,64}(op.dst())`, not
  // pulled from `op.src(1)`. (The `kKnownTiedIn` audit in
  // decode.cpp keeps `sdst_in` in the *driftCheck* allow-list — i.e.
  // we declare it semantically a real input — but no actual MCInst
  // operand survives disassembly to land in srcMap, so the read has
  // to come from the destination register itself.)
  if (sop == SemOp::S_BITSET0_B32 || sop == SemOp::S_BITSET1_B32 ||
      sop == SemOp::S_BITSET0_B64 || sop == SemOp::S_BITSET1_B64) {
    bool is64 = (sop == SemOp::S_BITSET0_B64 || sop == SemOp::S_BITSET1_B64);
    bool isSet = (sop == SemOp::S_BITSET1_B32 || sop == SemOp::S_BITSET1_B64);
    llvm::Type *ty = is64 ? ctx.i64Ty : ctx.i32Ty;
    // Hardware only consumes low log2(width) bits of the bit-index src;
    // mask explicitly so `shl 1, N` never becomes poison for N >= width.
    Value *bitIdx = ctx.B.CreateAnd(op.src(0),
                                    ConstantInt::get(ctx.i32Ty,
                                                     is64 ? 0x3F : 0x1F));
    if (is64) bitIdx = ctx.B.CreateZExt(bitIdx, ctx.i64Ty);
    Value *mask = ctx.B.CreateShl(ConstantInt::get(ty, 1), bitIdx);
    Value *old = is64 ? ctx.regs.readReg64(ctx.B, op.dst())
                      : ctx.regs.readReg32(ctx.B, op.dst());
    Value *res = isSet
                     ? ctx.B.CreateOr(old, mask, "bitset1")
                     : ctx.B.CreateAnd(old, ctx.B.CreateNot(mask), "bitset0");
    if (is64)
      ctx.regs.writeReg64(ctx.B, op.dst(), res);
    else
      ctx.regs.writeReg32(ctx.B, op.dst(), res);
    hr.handled = true;
    return hr;
  }
  // s_cmov_b{32,64}: scalar conditional move on SCC. Hardware
  // semantics (per the gfx1250 ISA manual; see also
  // SOPInstructions.td `let Uses = [SCC]`):
  //   if (SCC) sdst = src; else sdst stays unchanged
  // SCC is read but not written.
  //
  // LLVM's SOP1_32/SOP1_64 pseudo for S_CMOV_B{32,64} declares
  //   `(outs sdst), (ins src0)`
  // *without* a tied sdst_in input — the dst-on-SCC=0 read-modify
  // is implicit in the hardware encoding rather than modeled at
  // the MachineInstr level. So `op.nSrcs()` is 1 here (just src0)
  // and the prior dst value must be read explicitly via
  // `regs.readReg{32,64}(op.dst())`. The companion S_BITSET ops
  // above are the opposite case: their tied sdst_in is in srcMap
  // at index 1 because LLVM's `kKnownTiedIn` audit (decode.cpp)
  // keeps it. This asymmetry is a property of the LLVM .td
  // definitions, not a transpiler choice.
  if (sop == SemOp::S_CMOV_B32) {
    Value *cond = ctx.regs.loadSCC(ctx.B);
    Value *src = op.src(0);
    Value *oldDst = ctx.regs.readReg32(ctx.B, op.dst());
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateSelect(cond, src, oldDst, "scmov"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CMOV_B64) {
    Value *cond = ctx.regs.loadSCC(ctx.B);
    Value *src = op.src64(0);
    Value *oldDst = ctx.regs.readReg64(ctx.B, op.dst());
    ctx.regs.writeReg64(ctx.B, op.dst(),
                        ctx.B.CreateSelect(cond, src, oldDst, "scmov64"));
    hr.handled = true;
    return hr;
  }
  // S_SET_VGPR_MSB is SOPP format — handled in handleSOPP, not here.
  // GFX12+ `s_barrier_signal` appears in SOP1 encoding; model it as a no-op
  // (the paired SOPP `s_barrier_wait` does the actual rendezvous).
  if (sop == SemOp::S_BARRIER_SIGNAL) {
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
