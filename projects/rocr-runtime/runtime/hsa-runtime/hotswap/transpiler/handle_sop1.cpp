#include "handlers.hpp"
#include "raiser.hpp"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace transpiler {

HandlerResult handleSOP1(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result) {
  (void)result;
  HandlerResult hr;
  llvm::StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;
  (void)mn;

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
  if (sop == SemOp::S_SET_VGPR_MSB) {
    hr.handled = true;
    return hr;
  }
  // s_barrier_signal → no-op (the wait emits the actual barrier)
  if (di.mnemonic == "s_barrier_signal") {
    hr.handled = true;
    return hr;
  }
  // s_barrier_wait → emit a full s_barrier for gfx942 synchronization
  if (di.mnemonic == "s_barrier_wait") {
    Function *barrierFn =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_s_barrier);
    ctx.B.CreateCall(barrierFn, {});
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
