#include "handlers.hpp"
#include "raiser.hpp"

#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace transpiler {

HandlerResult handleSOP2(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result) {
  (void)result;
  HandlerResult hr;
  llvm::StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;
  (void)mn;

  // 32-bit binary ops — auto SCC via sccResult
  if (sop == SemOp::S_AND_B32) {
    hr.sccResult = ctx.B.CreateAnd(op.src(0), op.src(1), "and");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_OR_B32) {
    hr.sccResult = ctx.B.CreateOr(op.src(0), op.src(1), "or");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_LSHL_B32) {
    hr.sccResult = ctx.B.CreateShl(op.src(0), op.src(1), "shl");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_LSHR_B32) {
    hr.sccResult = ctx.B.CreateLShr(op.src(0), op.src(1), "lshr");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ASHR_I32) {
    hr.sccResult = ctx.B.CreateAShr(op.src(0), op.src(1), "ashr");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  // s_add_i32 / s_add_u32 (both SemOp::S_ADD_U32)
  if (sop == SemOp::S_ADD_U32) {                                                // Match by canonical semantic opcode, not raw mnemonic string
    Value *src0 = op.src(0), *src1 = op.src(1);                                 // Read source operands — resolves SGPR, VGPR, or immediate to LLVM Value*
    Value *res = ctx.B.CreateAdd(src0, src1, "add");                             // Emit LLVM IR: %add = add i32 %src0, %src1
    ctx.regs.writeReg32(ctx.B, op.dst(), res);                                   // Store result into destination register's alloca (later promoted to SSA)
    auto *ov = ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {ctx.i32Ty}, // Compute carry-out using LLVM's uadd.with.overflow intrinsic
                                     {src0, src1});
    ctx.regs.storeSCC(ctx.B, ctx.B.CreateExtractValue(ov, 1));                   // Extract the overflow bit and write it to SCC (Scalar Condition Code)
    hr.sccHandled = true;                                                        // Tell the dispatch loop: "I wrote SCC myself, don't auto-compute it"
    hr.handled = true;                                                           // Tell the dispatch loop: "This instruction was successfully raised"
    return hr;
  }
  // s_sub_i32 / s_sub_u32 (both SemOp::S_SUB_U32)
  if (sop == SemOp::S_SUB_U32) {
    Value *src0 = op.src(0), *src1 = op.src(1);
    Value *res = ctx.B.CreateSub(src0, src1, "sub");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    ctx.regs.storeSCC(ctx.B, ctx.B.CreateICmpULT(src0, src1));
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }

  // Special SCC semantics — handler writes SCC explicitly
  if (sop == SemOp::S_ADDC_U32) {
    Value *src0 = op.src(0), *src1 = op.src(1);
    Value *cin = ctx.B.CreateZExt(ctx.regs.loadSCC(ctx.B), ctx.i32Ty);
    Function *uaddOv = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::uadd_with_overflow, {ctx.i32Ty});
    Value *step1 = ctx.B.CreateCall(uaddOv, {src0, src1});
    Value *sum1 = ctx.B.CreateExtractValue(step1, 0);
    Value *c1 = ctx.B.CreateExtractValue(step1, 1);
    Value *step2 = ctx.B.CreateCall(uaddOv, {sum1, cin});
    Value *res = ctx.B.CreateExtractValue(step2, 0, "addc");
    Value *c2 = ctx.B.CreateExtractValue(step2, 1);
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    ctx.regs.storeSCC(ctx.B, ctx.B.CreateOr(c1, c2));
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SUBB_U32) {
    Value *src0 = op.src(0), *src1 = op.src(1);
    Value *borrow = ctx.B.CreateZExt(ctx.regs.loadSCC(ctx.B), ctx.i32Ty);
    Value *res =
        ctx.B.CreateSub(ctx.B.CreateSub(src0, src1), borrow, "subb");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    ctx.regs.storeSCC(ctx.B,
                       ctx.B.CreateOr(ctx.B.CreateICmpULT(src0, src1),
                                      ctx.B.CreateAnd(ctx.B.CreateICmpEQ(src0, src1),
                                                      ctx.regs.loadSCC(ctx.B))));
    hr.sccHandled = true;
    hr.handled = true;
    return hr;
  }

  // No SCC side-effect (di.defsSCC=false for these)
  if (sop == SemOp::S_MUL_I32) {
    ctx.regs.writeReg32(ctx.B, op.dst(),
                        ctx.B.CreateMul(op.src(0), op.src(1), "mul"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MUL_HI_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty),
          *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateTrunc(
            ctx.B.CreateLShr(ctx.B.CreateMul(a, b, "mulhi_wide"), 32), ctx.i32Ty,
            "mulhi"));
    hr.handled = true;
    return hr;
  }
  // GFX12 scalar FP multiply
  if (sop == SemOp::S_MUL_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.src(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.src(1), ctx.f32Ty);
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "s_fmul"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ADD_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.src(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.src(1), ctx.f32Ty);
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "s_fadd"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // GFX12 scalar 64-bit ops
  if (sop == SemOp::S_MUL_U64) {
    ctx.regs.writeReg64(ctx.B, op.dst(),
                        ctx.B.CreateMul(op.src64(0), op.src64(1), "smul64"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ADD_U64 || sop == SemOp::S_ADD_NC_U64) {
    ctx.regs.writeReg64(ctx.B, op.dst(),
                        ctx.B.CreateAdd(op.src64(0), op.src64(1), "sadd64"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MIN_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    hr.sccResult =
        ctx.B.CreateSelect(ctx.B.CreateICmpULT(s0, s1), s0, s1, "smin");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MAX_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    hr.sccResult =
        ctx.B.CreateSelect(ctx.B.CreateICmpUGT(s0, s1), s0, s1, "smax");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MIN_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    hr.sccResult =
        ctx.B.CreateSelect(ctx.B.CreateICmpSLT(s0, s1), s0, s1, "smin");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_MAX_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    hr.sccResult =
        ctx.B.CreateSelect(ctx.B.CreateICmpSGT(s0, s1), s0, s1, "smax");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_LSHL1_ADD_U32) {
    hr.sccResult =
        ctx.B.CreateAdd(ctx.B.CreateShl(op.src(0), 1), op.src(1), "lshl1add");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_LSHL2_ADD_U32) {
    hr.sccResult =
        ctx.B.CreateAdd(ctx.B.CreateShl(op.src(0), 2), op.src(1), "lshl2add");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_LSHL3_ADD_U32) {
    hr.sccResult =
        ctx.B.CreateAdd(ctx.B.CreateShl(op.src(0), 3), op.src(1), "lshl3add");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_LSHL4_ADD_U32) {
    hr.sccResult =
        ctx.B.CreateAdd(ctx.B.CreateShl(op.src(0), 4), op.src(1), "lshl4add");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_XOR_B32) {
    hr.sccResult = ctx.B.CreateXor(op.src(0), op.src(1), "xor");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_XOR_B64) {
    hr.sccResult = ctx.B.CreateXor(op.src64(0), op.src64(1), "xor64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BFM_B64) {
    // s_bfm_b64 dst, width, offset: creates a 64-bit mask with `width` ones
    // starting at `offset`
    Value *width =
        ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0x3F)),
                         ctx.i64Ty);
    Value *offset =
        ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0x3F)),
                         ctx.i64Ty);
    Value *mask = ctx.B.CreateSub(ctx.B.CreateShl(ConstantInt::get(ctx.i64Ty, 1), width),
                                  ConstantInt::get(ctx.i64Ty, 1));
    hr.sccResult = ctx.B.CreateShl(mask, offset, "bfm64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BFM_B32) {
    Value *width = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *offset = ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *mask =
        ctx.B.CreateSub(ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), width),
                        ConstantInt::get(ctx.i32Ty, 1));
    hr.sccResult = ctx.B.CreateShl(mask, offset, "bfm32");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BFE_U32) {
    Value *src = op.src(0), *ctrl = op.src(1);
    Value *offset = ctx.B.CreateAnd(ctrl, ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *width =
        ctx.B.CreateAnd(ctx.B.CreateLShr(ctrl, 16), ConstantInt::get(ctx.i32Ty, 0x7F));
    Value *safeWidth =
        ctx.B.CreateAnd(width, ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *shifted = ctx.B.CreateLShr(src, offset);
    Value *mask = ctx.B.CreateSub(
        ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), safeWidth),
        ConstantInt::get(ctx.i32Ty, 1));
    Value *isGE32 =
        ctx.B.CreateICmpUGE(width, ConstantInt::get(ctx.i32Ty, 32));
    mask = ctx.B.CreateSelect(isGE32, ConstantInt::getSigned(ctx.i32Ty, -1), mask);
    Value *isZero = ctx.B.CreateICmpEQ(width, ConstantInt::get(ctx.i32Ty, 0));
    hr.sccResult = ctx.B.CreateSelect(
        isZero, ConstantInt::get(ctx.i32Ty, 0),
        ctx.B.CreateAnd(shifted, mask, "bfe"));
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_PACK_LL_B32_B16) {
    Value *lo = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFF));
    Value *hi = ctx.B.CreateShl(
        ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFF)), 16);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(lo, hi, "pack_ll"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_PACK_LH_B32_B16) {
    Value *lo = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFF));
    Value *hi =
        ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFF0000u));
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(lo, hi, "pack_lh"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CSELECT_B32) {
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateSelect(ctx.regs.loadSCC(ctx.B), op.src(0), op.src(1),
                           "csel"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CSELECT_B64) {
    ctx.regs.writeReg64(
        ctx.B, op.dst(),
        ctx.B.CreateSelect(ctx.regs.loadSCC(ctx.B), op.src64(0), op.src64(1),
                           "csel"));
    hr.handled = true;
    return hr;
  }

  // 64-bit SOP2 — auto SCC via sccResult
  if (sop == SemOp::S_LSHL_B64) {
    hr.sccResult = ctx.B.CreateShl(op.src64(0), op.src64(1), "shl64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_OR_B64) {
    Value *res = ctx.B.CreateOr(op.src64(0), op.src64(1), "or64");
    ctx.regs.writeReg64(ctx.B, op.dst(), res);
    hr.sccResult = res;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_AND_B64) {
    Value *res = ctx.B.CreateAnd(op.src64(0), op.src64(1), "and64");
    ctx.regs.writeReg64(ctx.B, op.dst(), res);
    hr.sccResult = res;
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ANDN2_B64) {
    hr.sccResult =
        ctx.B.CreateAnd(op.src64(0), ctx.B.CreateNot(op.src64(1)), "andn2_64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ORN2_B64) {
    hr.sccResult =
        ctx.B.CreateOr(op.src64(0), ctx.B.CreateNot(op.src64(1)), "orn2_64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ANDN2_B32) {
    hr.sccResult =
        ctx.B.CreateAnd(op.src(0), ctx.B.CreateNot(op.src(1)), "andn2");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ORN2_B32) {
    hr.sccResult = ctx.B.CreateOr(op.src(0), ctx.B.CreateNot(op.src(1)), "orn2");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
