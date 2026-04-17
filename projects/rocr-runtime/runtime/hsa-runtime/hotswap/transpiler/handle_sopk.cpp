#include "handlers.hpp"
#include "raiser.hpp"

#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace transpiler {

HandlerResult handleSOPK(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result) {
  (void)result;
  HandlerResult hr;
  SemOp sop = di.semOp;

  // SOPK format: dst = SDST, src(0) = sign-extended 16-bit imm
  if (sop == SemOp::S_MOVK_I32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MULK_I32) {
    Value *dst = ctx.regs.readReg32(ctx.B, op.dst());
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateMul(dst, op.src(0), "mulk"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ADDK_I32) {
    Value *dst = ctx.regs.readReg32(ctx.B, op.dst());
    Value *imm = op.src(0);
    Value *res = ctx.B.CreateAdd(dst, imm, "addk");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    auto *ov = ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {ctx.i32Ty},
                                     {dst, imm});
    ctx.regs.storeSCC(ctx.B, ctx.B.CreateExtractValue(ov, 1));
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }
  // SOPK compares: s_cmpk_XX_i32 / s_cmpk_XX_u32
  if (sop == SemOp::S_CMPK_EQ_I32 || sop == SemOp::S_CMPK_EQ_U32 ||
      sop == SemOp::S_CMPK_LG_I32 || sop == SemOp::S_CMPK_LG_U32 ||
      (sop >= SemOp::S_CMPK_GE_I32 && sop <= SemOp::S_CMPK_LT_U32)) {
    Value *sdst = ctx.regs.readReg32(ctx.B, op.dst());
    Value *imm = op.src(0);
    Value *cmp = nullptr;
    if (sop == SemOp::S_CMPK_EQ_I32 || sop == SemOp::S_CMPK_EQ_U32)
      cmp = ctx.B.CreateICmpEQ(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LG_I32 || sop == SemOp::S_CMPK_LG_U32)
      cmp = ctx.B.CreateICmpNE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GT_I32)
      cmp = ctx.B.CreateICmpSGT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GE_I32)
      cmp = ctx.B.CreateICmpSGE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LT_I32)
      cmp = ctx.B.CreateICmpSLT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LE_I32)
      cmp = ctx.B.CreateICmpSLE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GT_U32)
      cmp = ctx.B.CreateICmpUGT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_GE_U32)
      cmp = ctx.B.CreateICmpUGE(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LT_U32)
      cmp = ctx.B.CreateICmpULT(sdst, imm, "scmpk");
    else if (sop == SemOp::S_CMPK_LE_U32)
      cmp = ctx.B.CreateICmpULE(sdst, imm, "scmpk");
    if (cmp) {
      ctx.regs.storeSCC(ctx.B, cmp);
      hr.sccHandled = true;
      hr.handled = true;
      return hr;
    }
  }
  // s_getreg_b32: reads a hardware config register into an SGPR.
  // We model all hardware registers as zero — the raised IR should not
  // depend on their exact values for correctness.
  if (sop == SemOp::S_GETREG_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ConstantInt::get(ctx.i32Ty, 0));
    hr.handled = true;
    return hr;
  }
  // s_setreg_b32/s_setreg_imm32_b32: writes to a hardware config register.
  // No-op in our model.
  if (sop == SemOp::S_SETREG_IMM32_B32 || sop == SemOp::S_SETREG_B32 ||
      sop == SemOp::S_GETREG_B32) {
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
