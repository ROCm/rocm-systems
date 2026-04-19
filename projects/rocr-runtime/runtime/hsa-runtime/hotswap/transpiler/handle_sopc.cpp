#include "handlers.hpp"

#include "llvm/IR/DerivedTypes.h"

using namespace llvm;

namespace transpiler {

HandlerResult handleSOPC(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  // s_set_gpr_idx_on enables GPR dynamic indexing via M0.
  // In scalar model, we store the index value to M0 and treat this as a
  // control-flow nop. The actual VGPR indexing effect is not modeled.
  if (sop == SemOp::S_SET_GPR_IDX_ON) {
    ParsedReg m0reg;
    m0reg.kind = ParsedReg::M0;
    m0reg.baseIdx = 0;
    ctx.regs.writeReg32(ctx.B, m0reg, op.src(0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SET_GPR_IDX_OFF) {
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SETVSKIP) {
    hr.handled = true;
    return hr;
  }
  // 64-bit unsigned SOPC compares (gfx8+ S_CMP_EQ_U64 / S_CMP_LG_U64).
  // Both operands are full 64-bit SGPR pairs per SOPInstructions.td's
  // `SOPC_CMP_64` record (the only SOPC compare shape that takes
  // 64-bit operands — there are no signed or ordered 64-bit SOPC
  // compares on any AMDGPU generation). Read both with `op.src64`
  // and emit a single `icmp eq/ne i64` into SCC.
  //
  // Test back-reference: lit_tests/s_cmp_eq_u64/ pins the `icmp eq
  // i64` shape this lift produces. A regression that narrowed the
  // operands to i32 (a common shortcut against 64-bit SGPR pairs)
  // would break the corpus's per-thread-mask compares used in
  // tensilelite gemm dispatch.
  if (sop == SemOp::S_CMP_EQ_U64) {
    Value *cmp64 =
        ctx.B.CreateICmpEQ(op.src64(0), op.src64(1), "scmp64");
    ctx.regs.storeSCC(ctx.B, cmp64);
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CMP_LG_U64) {
    Value *cmp64 =
        ctx.B.CreateICmpNE(op.src64(0), op.src64(1), "scmp64");
    ctx.regs.storeSCC(ctx.B, cmp64);
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }

  Value *src0 = op.src(0);
  Value *src1 = op.src(1);
  Value *cmp = nullptr;
  if (sop == SemOp::S_CMP_GT_I32)
    cmp = ctx.B.CreateICmpSGT(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_LT_I32)
    cmp = ctx.B.CreateICmpSLT(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_GE_I32)
    cmp = ctx.B.CreateICmpSGE(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_LE_I32)
    cmp = ctx.B.CreateICmpSLE(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_EQ_U32)
    cmp = ctx.B.CreateICmpEQ(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_LG_U32)
    cmp = ctx.B.CreateICmpNE(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_GE_U32)
    cmp = ctx.B.CreateICmpUGE(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_GT_U32)
    cmp = ctx.B.CreateICmpUGT(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_LT_U32)
    cmp = ctx.B.CreateICmpULT(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_LE_U32)
    cmp = ctx.B.CreateICmpULE(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_EQ_I32)
    cmp = ctx.B.CreateICmpEQ(src0, src1, "scmp");
  else if (sop == SemOp::S_CMP_LG_I32)
    cmp = ctx.B.CreateICmpNE(src0, src1, "scmp");
  // GFX12 scalar FP compares (ordered and unordered variants)
  else if (sop >= SemOp::S_CMP_EQ_F32 && sop <= SemOp::S_CMP_NLG_F32) {
    Value *f0 = ctx.B.CreateBitCast(src0, ctx.f32Ty);
    Value *f1 = ctx.B.CreateBitCast(src1, ctx.f32Ty);
    if (sop == SemOp::S_CMP_EQ_F32)
      cmp = ctx.B.CreateFCmpOEQ(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_LG_F32)
      cmp = ctx.B.CreateFCmpONE(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_GT_F32)
      cmp = ctx.B.CreateFCmpOGT(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_GE_F32)
      cmp = ctx.B.CreateFCmpOGE(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_LT_F32)
      cmp = ctx.B.CreateFCmpOLT(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_LE_F32)
      cmp = ctx.B.CreateFCmpOLE(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_NEQ_F32)
      cmp = ctx.B.CreateFCmpUNE(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_NLT_F32)
      cmp = ctx.B.CreateFCmpUGE(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_NLE_F32)
      cmp = ctx.B.CreateFCmpUGT(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_NGT_F32)
      cmp = ctx.B.CreateFCmpULE(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_NGE_F32)
      cmp = ctx.B.CreateFCmpULT(f0, f1, "scmpf");
    else if (sop == SemOp::S_CMP_NLG_F32)
      cmp = ctx.B.CreateFCmpUEQ(f0, f1, "scmpf");
  } else if (sop >= SemOp::S_CMP_EQ_F16 && sop <= SemOp::S_CMP_NLG_F16) {
    Type *f16Ty = Type::getHalfTy(ctx.C);
    Value *f0 = ctx.B.CreateBitCast(
        ctx.B.CreateTrunc(src0, Type::getInt16Ty(ctx.C)), f16Ty);
    Value *f1 = ctx.B.CreateBitCast(
        ctx.B.CreateTrunc(src1, Type::getInt16Ty(ctx.C)), f16Ty);
    if (sop == SemOp::S_CMP_EQ_F16)
      cmp = ctx.B.CreateFCmpOEQ(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_LG_F16)
      cmp = ctx.B.CreateFCmpONE(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_GT_F16)
      cmp = ctx.B.CreateFCmpOGT(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_GE_F16)
      cmp = ctx.B.CreateFCmpOGE(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_LT_F16)
      cmp = ctx.B.CreateFCmpOLT(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_LE_F16)
      cmp = ctx.B.CreateFCmpOLE(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_NEQ_F16)
      cmp = ctx.B.CreateFCmpUNE(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_NLT_F16)
      cmp = ctx.B.CreateFCmpUGE(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_NLE_F16)
      cmp = ctx.B.CreateFCmpUGT(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_NGT_F16)
      cmp = ctx.B.CreateFCmpULE(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_NGE_F16)
      cmp = ctx.B.CreateFCmpULT(f0, f1, "scmpf16");
    else if (sop == SemOp::S_CMP_NLG_F16)
      cmp = ctx.B.CreateFCmpUEQ(f0, f1, "scmpf16");
  }
  if (cmp) {
    ctx.regs.storeSCC(ctx.B, cmp);
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
