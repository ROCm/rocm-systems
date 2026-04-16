#include "handlers.hpp"
#include "raiser.hpp"

#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace transpiler {

HandlerResult handleSOPP(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result) {
  (void)op;
  (void)result;
  HandlerResult hr;
  llvm::StringRef mn(di.mnemonic);
  (void)mn;
  SemOp sop = di.semOp;

  if (sop == SemOp::S_ENDPGM) {
    ctx.B.CreateRetVoid();
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BRANCH) {
    int64_t raw = di.getImm(0);
    int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
    ctx.B.CreateBr(ctx.lookupBB(di.offset + 4 + brOff * 4));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CBRANCH_EXECZ || sop == SemOp::S_CBRANCH_EXECNZ) {
    int64_t raw = di.getImm(0);
    int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
    uint64_t target = di.offset + 4 + brOff * 4;
    BasicBlock *targetBB = ctx.lookupBB(target);
    BasicBlock *fallthroughBB = ctx.lookupBB(di.offset + di.size);
    Value *execVal = ctx.regs.loadExec(ctx.B);
    Value *isZero = ctx.B.CreateICmpEQ(
        execVal, Constant::getNullValue(ctx.regs.execTy), "exec_is_zero");
    if (sop == SemOp::S_CBRANCH_EXECZ)
      ctx.B.CreateCondBr(isZero, targetBB, fallthroughBB);
    else
      ctx.B.CreateCondBr(ctx.B.CreateNot(isZero, "exec_nz"), targetBB,
                         fallthroughBB);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CBRANCH_SCC0 || sop == SemOp::S_CBRANCH_SCC1) {
    int64_t raw = di.getImm(0);
    int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
    uint64_t target = di.offset + 4 + brOff * 4;
    BasicBlock *targetBB = ctx.lookupBB(target);
    BasicBlock *fallthroughBB = ctx.lookupBB(di.offset + di.size);
    Value *sccV = ctx.regs.loadSCC(ctx.B);
    if (sop == SemOp::S_CBRANCH_SCC0)
      sccV = ctx.B.CreateNot(sccV, "not_scc");
    ctx.B.CreateCondBr(sccV, targetBB, fallthroughBB);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CBRANCH_VCCNZ || sop == SemOp::S_CBRANCH_VCCZ) {
    int64_t raw = di.getImm(0);
    int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
    uint64_t target = di.offset + 4 + brOff * 4;
    BasicBlock *targetBB = ctx.lookupBB(target);
    BasicBlock *fallthroughBB = ctx.lookupBB(di.offset + di.size);
    Value *vccV = ctx.regs.loadVCC(ctx.B);
    if (sop == SemOp::S_CBRANCH_VCCZ)
      vccV = ctx.B.CreateNot(vccV, "not_vcc");
    ctx.B.CreateCondBr(vccV, targetBB, fallthroughBB);
    hr.handled = true;
    return hr;
  }
  // s_barrier / s_barrier_wait / s_barrier_signal — emit barrier for gfx942
  // GFX12+ splits s_barrier into signal+wait; both may be SOPP format.
  if (di.mnemonic == "s_barrier" || di.mnemonic == "s_barrier_wait") {
    Function *barrierFn =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_s_barrier);
    ctx.B.CreateCall(barrierFn, {});
    hr.handled = true;
    return hr;
  }
  if (di.mnemonic == "s_barrier_signal") {
    hr.handled = true;
    return hr;
  }
  // All other SOPP instructions (waitcnt, nop, etc.) are no-ops
  hr.handled = true;
  return hr;
}

} // namespace transpiler
