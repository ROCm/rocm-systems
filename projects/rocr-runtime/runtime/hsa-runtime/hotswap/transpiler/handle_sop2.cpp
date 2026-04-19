#include "handlers.hpp"
#include "sem_op_attrs.hpp"

#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace transpiler {

// SPE attribute registrations. All of these route through
// `writeReg{32,64,ExecWidth}` which dispatch EXEC writes to
// `regs.storeExec`. Audit any addition before landing — see
// AGENTS.md's SPE section.
ArrayRef<SemOpAttrSpec> getHandlerSOP2Attrs() {
  static constexpr SemOpAttrSpec kAttrs[] = {
      {SemOp::S_AND_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_AND_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_OR_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_OR_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_XOR_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_XOR_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_ANDN2_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_ANDN2_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_ORN2_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_ORN2_B64, {/*routesExecThroughStoreExec=*/true}},
      // s_{nand,nor,xnor}_b{32,64}: `dst = ~(src0 OP src1)`. Identical
      // SPE shape to s_{and,or,xor}_b* — they read EXEC-relative scalar
      // sources and dispatch the result through writeReg{32,64}, which
      // routes to storeExec when the destination operand is EXEC. See
      // SOPInstructions.td:789-803 for the LLVM patterns.
      {SemOp::S_NAND_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_NAND_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_NOR_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_NOR_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_XNOR_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_XNOR_B64, {/*routesExecThroughStoreExec=*/true}},
      // s_absdiff_i32 returns an i32 magnitude; in principle the result
      // can target EXEC like any other SOP2 i32 writer, so route through
      // storeExec for safety.
      {SemOp::S_ABSDIFF_I32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_LSHL_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_LSHL_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_LSHR_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_LSHR_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_ASHR_I64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_BFM_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_BFM_B64, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_CSELECT_B32, {/*routesExecThroughStoreExec=*/true}},
      {SemOp::S_CSELECT_B64, {/*routesExecThroughStoreExec=*/true}},
  };
  return kAttrs;
}

HandlerResult handleSOP2(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

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
  // s_mul_hi_i32: signed mul-high. Same widening pattern as
  // S_MUL_HI_U32 above, but sign-extend both operands so the wide
  // multiply produces a signed product. SOPInstructions.td .td
  // pattern is `mulhs SSrc_b32, SSrc_b32` (line ~849); the only
  // operational difference vs S_MUL_HI_U32 (`mulhu`) is the
  // extension semantics on the inputs.
  if (sop == SemOp::S_MUL_HI_I32) {
    Value *a = ctx.B.CreateSExt(op.src(0), ctx.i64Ty),
          *b = ctx.B.CreateSExt(op.src(1), ctx.i64Ty);
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateTrunc(
            ctx.B.CreateLShr(ctx.B.CreateMul(a, b, "mulhi_i_wide"), 32),
            ctx.i32Ty, "mulhi_i"));
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
  // gfx11+ scalar FP subtract. Direct mirror of S_ADD_F32 above;
  // the .td pattern is `any_fsub` (SOPInstructions.td:894). No
  // source modifiers on SOP2 — the operands are bare i32-shaped
  // SGPRs that we reinterpret as f32.
  if (sop == SemOp::S_SUB_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.src(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.src(1), ctx.f32Ty);
    ctx.regs.writeReg32(
        ctx.B, op.dst(),
        ctx.B.CreateBitCast(ctx.B.CreateFSub(s0, s1, "s_fsub"), ctx.i32Ty));
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
  // s_sub_nc_u64: gfx12 64-bit scalar subtract, no carry. Mirror
  // of S_ADD_NC_U64 above. SCC is *not* updated (the `nc` suffix);
  // see SOPInstructions.td 661 (no `Defs = [SCC]`).
  if (sop == SemOp::S_SUB_NC_U64) {
    ctx.regs.writeReg64(ctx.B, op.dst(),
                        ctx.B.CreateSub(op.src64(0), op.src64(1), "ssub64"));
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
  // s_bfe_i32: signed scalar Bit Field Extract.
  //   shift  = ctrl[4:0]
  //   length = ctrl[22:16]
  //   if length == 0: D = 0
  //   elif shift + length < 32:
  //       D = sign_ext((src << (32 - shift - length)) >> (32 - length))
  //   else:
  //       D = (int32)src >> shift   (length saturates to full width)
  // Matches native s_bfe_i32 exactly, including the shift-trick behavior
  // that diverges from a naive "mask and sign-extend from bit (length-1)"
  // implementation when shift + length >= 32.
  //
  // The `shl` / `ashr` amounts can legitimately be out-of-range on the
  // "wrong" side of the isShortEnough select (e.g. `32 - sum` wraps to a
  // huge value when sum >= 32, and `32 - length` is 32 when length == 0).
  // LLVM's select doesn't propagate poison from the unselected branch so
  // it's observationally safe, but we still mask every shift amount to 5
  // bits up front to remove the poison source entirely and keep future
  // optimizer passes from having to prove the guards are sound.
  if (sop == SemOp::S_BFE_I32) {
    Value *src = op.src(0), *ctrl = op.src(1);
    Value *c31 = ConstantInt::get(ctx.i32Ty, 0x1F);
    Value *c32 = ConstantInt::get(ctx.i32Ty, 32);
    Value *shift = ctx.B.CreateAnd(ctrl, c31);
    Value *length = ctx.B.CreateAnd(ctx.B.CreateLShr(ctrl, 16),
                                    ConstantInt::get(ctx.i32Ty, 0x7F));
    Value *sum = ctx.B.CreateAdd(shift, length);
    Value *isShortEnough = ctx.B.CreateICmpULT(sum, c32);
    Value *shlAmt = ctx.B.CreateAnd(ctx.B.CreateSub(c32, sum), c31);
    Value *shiftedLeft = ctx.B.CreateShl(src, shlAmt);
    Value *shrAmt = ctx.B.CreateAnd(ctx.B.CreateSub(c32, length), c31);
    Value *sx = ctx.B.CreateAShr(shiftedLeft, shrAmt, "sbfe_i");
    // Fall-through branch (length saturates): arithmetic right shift by
    // `shift` gives "sign-extended src[31:shift]" in a single op.
    Value *fallthrough = ctx.B.CreateAShr(src, shift, "sbfe_i_sat");
    Value *computed = ctx.B.CreateSelect(isShortEnough, sx, fallthrough);
    Value *isZero = ctx.B.CreateICmpEQ(length,
                                       ConstantInt::get(ctx.i32Ty, 0));
    Value *result = ctx.B.CreateSelect(isZero,
                                       ConstantInt::get(ctx.i32Ty, 0),
                                       computed);
    // sccResult is an i32; downstream code derives SCC as (sccResult != 0),
    // matching the ISA's "SCC = D != 0" for s_bfe_*.
    hr.sccResult = result;
    ctx.regs.writeReg32(ctx.B, op.dst(), result);
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

  // 64-bit SOP2 — auto SCC via sccResult.
  //
  // S_LSHL_B64 / S_LSHR_B64 / S_ASHR_I64 are all `SOP2_64_32` shape per
  // SOPInstructions.td (`SReg_64:$sdst, SSrc_b64:$src0, SSrc_b32:$src1`)
  // — src1 is a SINGLE 32-bit SGPR holding the shift count, not a
  // 64-bit pair. Reading it as i64 via `op.src64(1)` would pull the
  // following SGPR (s_n+1) as garbage in the high half, and LLVM's
  // `lshr/shl/ashr i64 %a, %b` produces poison whenever `%b >= 64`,
  // which a randomly-set bit in s_n+1 will trigger. We read src1 as
  // i32 and zext to i64 so the shift count is bounded to [0, 2^32).
  // The hardware's effective shift modulo (low 6 bits) is preserved
  // by LLVM's IR semantics: any zext'd i32 < 64 yields the same shift
  // result as a direct 64-bit op, and any value >= 64 is undefined in
  // both hardware (per the AMDGPU ISA docs: "shift count is masked to
  // [0,63]") and IR (poison) — but only the IR path makes the boundary
  // observable, so emitting a defensive `urem` here would mask a real
  // source-binary bug rather than reflect hardware. We do NOT mask.
  //
  // Test back-reference: lit_tests/s_lshr_b64_imm/ pins the dominant
  // corpus shape `s_lshr_b64 sdst, src0, IMM` lifting to
  // `%lshr64 = lshr i64 %src0, IMM` (the i32→i64 zext on the
  // immediate constant-folds away). Any change to this branch — the
  // shift-count zext, the i64 dst write, or the value-name
  // `lshr64` — must keep that fixture green.
  if (sop == SemOp::S_LSHL_B64) {
    Value *amt = ctx.B.CreateZExt(op.src(1), ctx.i64Ty, "shamt64");
    hr.sccResult = ctx.B.CreateShl(op.src64(0), amt, "shl64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_LSHR_B64) {
    Value *amt = ctx.B.CreateZExt(op.src(1), ctx.i64Ty, "shamt64");
    hr.sccResult = ctx.B.CreateLShr(op.src64(0), amt, "lshr64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_ASHR_I64) {
    Value *amt = ctx.B.CreateZExt(op.src(1), ctx.i64Ty, "shamt64");
    hr.sccResult = ctx.B.CreateAShr(op.src64(0), amt, "ashr64");
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
  // s_{nand,nor,xnor}_b{32,64} — negated bitops, `dst = ~(src0 OP src1)`.
  // SCC follows writeReg32/64's standard rule (set when result != 0).
  // Each opcode uses the same SOP2 operand triplet (sdst, src0, src1)
  // and identical sign-/zero-extension semantics as their non-negated
  // siblings (S_AND_B32 etc.), so we can reuse op.src/op.src64 directly.
  if (sop == SemOp::S_NAND_B32) {
    hr.sccResult = ctx.B.CreateNot(
        ctx.B.CreateAnd(op.src(0), op.src(1), "and"), "nand");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_NAND_B64) {
    hr.sccResult = ctx.B.CreateNot(
        ctx.B.CreateAnd(op.src64(0), op.src64(1), "and64"), "nand64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_NOR_B32) {
    hr.sccResult = ctx.B.CreateNot(
        ctx.B.CreateOr(op.src(0), op.src(1), "or"), "nor");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_NOR_B64) {
    hr.sccResult = ctx.B.CreateNot(
        ctx.B.CreateOr(op.src64(0), op.src64(1), "or64"), "nor64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_XNOR_B32) {
    hr.sccResult = ctx.B.CreateNot(
        ctx.B.CreateXor(op.src(0), op.src(1), "xor"), "xnor");
    ctx.regs.writeReg32(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_XNOR_B64) {
    hr.sccResult = ctx.B.CreateNot(
        ctx.B.CreateXor(op.src64(0), op.src64(1), "xor64"), "xnor64");
    ctx.regs.writeReg64(ctx.B, op.dst(), hr.sccResult);
    hr.handled = true;
    return hr;
  }
  // s_absdiff_i32 — `dst = |src0 - src1|` on signed i32. The hardware
  // wraps for INT_MIN (since `0 - INT_MIN` overflows back to itself in
  // two's complement), so we lower with `llvm.abs.i32(diff, false)` —
  // is_int_min_poison=false matches the wrapping behaviour exactly.
  // SCC follows writeReg32's standard (set when result != 0).
  if (sop == SemOp::S_ABSDIFF_I32) {
    Value *diff = ctx.B.CreateSub(op.src(0), op.src(1), "absdiff_sub");
    Value *res = ctx.B.CreateIntrinsic(Intrinsic::abs, {ctx.i32Ty},
                                       {diff, ctx.B.getFalse()},
                                       /*FMFSource=*/nullptr, "absdiff");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    hr.sccResult = res;
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
