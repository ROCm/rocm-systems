#include "handlers.hpp"

#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace transpiler {

HandlerResult handleSOPP(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op) {
  (void)op;
  HandlerResult hr;
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
  // Barriers. GFX<12 uses a single `s_barrier`; GFX12+ splits it into a
  // separate signal and wait (both SOPP in this format). We model signal as
  // a no-op (the cross-wave rendezvous happens at the wait) and wait (or the
  // legacy unified barrier) as a full LLVM `amdgcn.s.barrier` call.
  if (sop == SemOp::S_BARRIER || sop == SemOp::S_BARRIER_WAIT) {
    Function *barrierFn =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_s_barrier);
    ctx.B.CreateCall(barrierFn, {});
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BARRIER_SIGNAL) {
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SET_VGPR_MSB) {
    // Only the low 8 bits of the immediate carry runtime meaning; the high
    // 8 bits record the previous mode for compiler bookkeeping (see
    // AMDGPULowerVGPREncoding::setMode in LLVM).  The hardware ignores them.
    int64_t imm = di.getImm(0);
    ctx.vgprMSBs = static_cast<uint8_t>(imm & 0xFF);
    hr.handled = true;
    return hr;
  }
  // All other SOPP instructions (waitcnt, nop, etc.) are no-ops
  hr.handled = true;
  return hr;
}

} // namespace transpiler
