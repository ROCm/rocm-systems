#include "handlers.hpp"
#include "sem_op_attrs.hpp"

#include "llvm/IR/DerivedTypes.h"
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
    ctx.regs.writeReg64(ctx.B, op.dst(), ConstantInt::get(ctx.i64Ty, 0));
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
  // These are read-modify-write: the destination's prior value arrives
  // as the tied `sdst_in` operand at src index 1 (see `kKnownTiedIn` in
  // raiser.cpp, which explicitly keeps `sdst_in` in srcMap), and the
  // bit index arrives in `src0` at src index 0.  SCC is not updated.
  if (sop == SemOp::S_BITSET0_B32 || sop == SemOp::S_BITSET1_B32 ||
      sop == SemOp::S_BITSET0_B64 || sop == SemOp::S_BITSET1_B64) {
    // Enforce the tied-operand invariant at runtime so that a future
    // raiser change (e.g. dropping `sdst_in` from the srcMap) fails
    // loudly rather than silently computing garbage from an undefined
    // `op.src(1)`.
    assert(op.nSrcs() >= 2 &&
           "s_bitset*: expected src0 and tied sdst_in in srcMap");
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
    Value *old = is64 ? op.src64(1) : op.src(1);
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
