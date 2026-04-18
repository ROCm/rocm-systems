#include "handlers.hpp"
#include "sem_op_attrs.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace transpiler {

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
    // Pattern B indirectbr, or Unresolvable). Both patterns emit a
    // terminator into the current BB; the raiser's BB-layout phase
    // has already promoted the next linear offset to a leader so
    // subsequent instructions land in their own BBs.
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
    case SetPcSiteInfo::Kind::IndirectB: {
      // Read the ret-pair as i64, cast to ptr (the blockaddress LLVM
      // type is `ptr addrspace(0)` for a BasicBlock; the call-site
      // rewrite stored it via inttoptr → ptrtoint → store, so we mirror
      // here with an inttoptr).
      Value *retVal = ctx.regs.loadSGPR64(
          ctx.B, static_cast<int>(info.indirectRetPairLowReg));
      Value *retPtr = ctx.B.CreateIntToPtr(
          retVal, PointerType::get(ctx.C, 0), "ret_pc_ptr");
      IndirectBrInst *ibr = ctx.B.CreateIndirectBr(
          retPtr, info.indirectTargets.size());
      for (uint64_t addr : info.indirectTargets)
        ibr->addDestination(ctx.lookupBB(addr));
      hr.handled = true;
      return hr;
    }
    case SetPcSiteInfo::Kind::Unresolvable:
      hr.failure = RaiseFailure::unsupportedShape(di, "SOP1",
                                                  info.refusalReason);
      return hr;
    }
    // Defensive: every Kind is handled above; reaching here means a
    // future enum value was added without updating this switch.
    hr.failure = RaiseFailure::unsupportedShape(
        di, "SOP1", "s_set_pc_i64 SetPcSiteInfo::Kind not handled");
    return hr;
  }
  if (sop == SemOp::S_SWAP_PC_I64) {
    // Branch-and-link. setpc_analysis classifies the call-target
    // pair (ssrc) as DirectA (chain resolves the absolute callee
    // offset) or Unresolvable (call target is dynamic-dispatch /
    // chain doesn't reach here). For DirectA we materialise
    // `blockaddress(@kernel, %BB_returnAddr)` cast to i64 into sdst
    // and then emit `br label %BB_callee`; the eventual Pattern B
    // `s_set_pc_i64 sdst` in the callee will consume that
    // blockaddress via its indirectbr enumeration. For everything
    // else we refuse loudly — never silently emit a stub branch.
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
      // Indirect call target (e.g. function-pointer dispatch). We do
      // not enumerate callee candidates today — that requires either
      // a per-kernel subroutine-entry catalogue or cross-block scalar
      // / kernarg-derived target resolution. Refuse loudly with a
      // distinct reason so future work can pivot off this string.
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SOP1",
          "s_swap_pc_i64 with an indirect call-target SGPR pair is "
          "not yet supported (would require enumerating callee "
          "candidates for `indirectbr`); track via the "
          "swap_pc_dynamic_dispatch worklist item");
      return hr;
    }
    // DirectA path: write the return address (a real LLVM
    // BlockAddress constant cast to i64) into sdst BEFORE the
    // branch, then emit the unconditional br to the callee. Both
    // operations land in the swap's own BB; the raiser's BB-layout
    // phase has already promoted (di.offset + di.size) to a leader
    // so subsequent linear instructions live in their own BB.
    uint64_t returnAddr = di.offset + di.size;
    BasicBlock *retBB = ctx.lookupBB(returnAddr);
    Constant *ba = BlockAddress::get(ctx.kernel, retBB);
    Value *baInt = ctx.B.CreatePtrToInt(ba, ctx.i64Ty,
                                          "swap_ret_blockaddr");
    ctx.regs.writeReg64(ctx.B, op.dst(), baInt);
    ctx.B.CreateBr(ctx.lookupBB(info.directTarget));
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
