#include "handle_valu_internal.hpp"
#include "handlers.hpp"
#include "opcode_map.hpp"
#include "wmma_lowering.hpp"

#include "semop.hpp"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace transpiler {
HandlerResult handleVALU(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op) {
  HandlerResult hr;
  // `mn` is retained for diagnostic messages only; dispatch is driven entirely
  // by `sop`, which the OpcodeMap canonicalizer resolves from the DPP/SDWA/e32
  // encoding to the base pseudo before the handler sees it.
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;
  if (sop == SemOp::V_NOP) {
    hr.handled = true;
    return hr;
  }
  // ---- v_mov_b32 ----
  if (sop == SemOp::V_MOV_B32) {
    ctx.writeReg32(op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }
  // ---- Cross-lane primitives (readlane/writelane/permlane/mbcnt/
  //      readfirstlane) — extracted to handle_valu_cross_lane.cpp ----
  {
    HandlerResult sub = handleVALU_CrossLane(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  // ---- Small ops (conversions, F16 arith, single-src F32 transcendentals,
  //      16-bit shifts, V_BFREV_B32 / V_NOT_B32, byte pack) ----
  // Extracted to handle_valu_small_ops.cpp.
  {
    HandlerResult sub = handleVALU_SmallOps(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  // ---- Simple 2-src integer ALU ----
  if (sop == SemOp::V_ADD_NC_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(op.src(0), op.src(1), "vadd"));
    hr.handled = true;
    return hr;
  }
  // GFX9 VOP3-only v_add_i32 / v_sub_i32: plain add/sub when clamp=0,
  // signed saturation (saddsat/ssubsat) when clamp=1.
  if (sop == SemOp::V_ADD_I32 || sop == SemOp::V_SUB_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    int clampIdx = AMDGPU::getNamedOperandIdx(
        di.inst.getOpcode(), AMDGPU::OpName::clamp);
    bool clamped = clampIdx >= 0 && di.isImm(clampIdx) &&
                   di.getImm(clampIdx) != 0;
    if (clamped) {
      Intrinsic::ID satID = (sop == SemOp::V_ADD_I32)
                                ? Intrinsic::sadd_sat
                                : Intrinsic::ssub_sat;
      Value *res = ctx.B.CreateBinaryIntrinsic(satID, s0, s1);
      ctx.writeReg32(op.dst(), res);
    } else {
      Value *res = (sop == SemOp::V_ADD_I32)
                       ? ctx.B.CreateAdd(s0, s1, "vadd_i32")
                       : ctx.B.CreateSub(s0, s1, "vsub_i32");
      ctx.writeReg32(op.dst(), res);
    }
    hr.handled = true;
    return hr;
  }
  // Vector add with carry-out (GFX12: v_add_co_u32; VCC = carry)
  if (sop == SemOp::V_ADD_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateAdd(s0, s1, "vadd_co");
    ctx.writeReg32(op.dst(), res);
    auto *ov = ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {ctx.i32Ty}, {s0, s1});
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateExtractValue(ov, 1));
    hr.handled = true;
    return hr;
  }
  // Vector sub with carry-out (GFX9: v_sub_u32; GFX10+: v_sub_co_u32)
  if (sop == SemOp::V_SUB_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateSub(s0, s1, "vsub_co");
    ctx.writeReg32(op.dst(), res);
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateICmpULT(s0, s1));
    hr.handled = true;
    return hr;
  }
  // Vector reversed sub with carry-out (GFX9: v_subrev_u32; GFX10+: v_subrev_co_u32)
  if (sop == SemOp::V_SUBREV_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateSub(s1, s0, "vsubrev_co");
    ctx.writeReg32(op.dst(), res);
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateICmpULT(s1, s0));
    hr.handled = true;
    return hr;
  }
  // Vector sub with borrow-in/borrow-out (GFX9: v_subb_u32; GFX10+: v_sub_co_ci_u32)
  if (sop == SemOp::V_SUB_CO_CI_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *bin = ctx.B.CreateZExt(ctx.regs.loadVCC(ctx.B), ctx.i32Ty);
    Value *diff1 = ctx.B.CreateSub(s0, s1);
    Value *diff2 = ctx.B.CreateSub(diff1, bin, "vsub_ci");
    Value *b1 = ctx.B.CreateICmpULT(s0, s1);
    Value *b2 = ctx.B.CreateICmpULT(diff1, bin);
    ctx.writeReg32(op.dst(), diff2);
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateOr(b1, b2));
    hr.handled = true;
    return hr;
  }
  // Vector reversed sub with borrow-in/borrow-out (v_subbrev_co_u32)
  if (sop == SemOp::V_SUBREV_CO_CI_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *bin = ctx.B.CreateZExt(ctx.regs.loadVCC(ctx.B), ctx.i32Ty);
    Value *diff1 = ctx.B.CreateSub(s1, s0);
    Value *diff2 = ctx.B.CreateSub(diff1, bin, "vsubrev_ci");
    Value *b1 = ctx.B.CreateICmpULT(s1, s0);
    Value *b2 = ctx.B.CreateICmpULT(diff1, bin);
    ctx.writeReg32(op.dst(), diff2);
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateOr(b1, b2));
    hr.handled = true;
    return hr;
  }
  // Vector add with carry-in/carry-out (GFX12: v_add_co_ci_u32)
  if (sop == SemOp::V_ADD_CO_CI_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *cin = ctx.B.CreateZExt(ctx.regs.loadVCC(ctx.B), ctx.i32Ty);
    Function *uaddOv = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::uadd_with_overflow, {ctx.i32Ty});
    Value *step1 = ctx.B.CreateCall(uaddOv, {s0, s1});
    Value *sum1 = ctx.B.CreateExtractValue(step1, 0);
    Value *c1   = ctx.B.CreateExtractValue(step1, 1);
    Value *step2 = ctx.B.CreateCall(uaddOv, {sum1, cin});
    Value *res   = ctx.B.CreateExtractValue(step2, 0, "vadd_ci");
    Value *c2    = ctx.B.CreateExtractValue(step2, 1);
    ctx.writeReg32(op.dst(), res);
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateOr(c1, c2));
    hr.handled = true;
    return hr;
  }
  // v_mad_co_u64_u32: D.u64 = S0.u32 * S1.u32 + S2.u64, VCC = carry
  if (sop == SemOp::V_MAD_CO_U64_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src64(2), "vmad_co64");
    ctx.writeReg64(op.dst(0), res);
    hr.handled = true;
    return hr;
  }
  // v_mad_u32: D.u32 = S0.u32 * S1.u32 + S2.u32 (no carry)
  if (sop == SemOp::V_MAD_U32) {
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(op.src(0), op.src(1)), op.src(2), "vmad_u32");
    ctx.writeReg32(op.dst(), res);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_OR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(op.src(0), op.src(1), "vor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_AND_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAnd(op.src(0), op.src(1), "vand"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_LO_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateMul(op.src(0), op.src(1), "vmul"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUB_NC_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateSub(op.src(0), op.src(1), "vsub"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUBREV_NC_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateSub(op.src(1), op.src(0), "vsubrev"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_XOR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateXor(op.src(0), op.src(1), "vxor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_XNOR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateNot(ctx.B.CreateXor(op.src(0), op.src(1)), "vxnor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpUGT(s0, s1), s0, s1, "vmax"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpULT(s0, s1), s0, s1, "vmin"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpSGT(s0, s1), s0, s1, "vmax"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpSLT(s0, s1), s0, s1, "vmin"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateLShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "vmulhi"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_I32) {
    Value *a = ctx.B.CreateSExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateSExt(op.src(1), ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateAShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "vmulhi"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_U32_U24) {
    Value *a = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    Value *b = ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    ctx.writeReg32(op.dst(), ctx.B.CreateMul(a, b, "mul24"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_I32_I24) {
    Value *a = ctx.B.CreateShl(op.src(0), 8);
    a = ctx.B.CreateAShr(a, 8);
    Value *b = ctx.B.CreateShl(op.src(1), 8);
    b = ctx.B.CreateAShr(b, 8);
    ctx.writeReg32(op.dst(), ctx.B.CreateMul(a, b, "mul24i"));
    hr.handled = true;
    return hr;
  }
  // v_dot8c_i32_i4: dst += sum of 8 signed 4-bit lane products
  if (sop == SemOp::V_DOT8C_I32_I4) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *acc = ctx.regs.readReg32(ctx.B, op.dst());
    for (int i = 0; i < 8; i++) {
      Value *shift = ConstantInt::get(ctx.i32Ty, i * 4);
      Value *a = ctx.B.CreateAnd(ctx.B.CreateLShr(s0, shift), ConstantInt::get(ctx.i32Ty, 0xF));
      Value *b = ctx.B.CreateAnd(ctx.B.CreateLShr(s1, shift), ConstantInt::get(ctx.i32Ty, 0xF));
      // Sign-extend 4-bit: shift left 28, arithmetic shift right 28
      a = ctx.B.CreateAShr(ctx.B.CreateShl(a, 28), 28);
      b = ctx.B.CreateAShr(ctx.B.CreateShl(b, 28), 28);
      acc = ctx.B.CreateAdd(acc, ctx.B.CreateMul(a, b));
    }
    ctx.writeReg32(op.dst(), acc);
    hr.handled = true;
    return hr;
  }
  // v_dot4c_i32_i8: dst += sum of 4 signed 8-bit lane products
  if (sop == SemOp::V_DOT4C_I32_I8) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Type *i8Ty = Type::getInt8Ty(ctx.C);
    Value *acc = ctx.regs.readReg32(ctx.B, op.dst());
    for (int i = 0; i < 4; i++) {
      Value *shift = ConstantInt::get(ctx.i32Ty, i * 8);
      Value *a = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s0, shift), i8Ty), ctx.i32Ty);
      Value *b = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s1, shift), i8Ty), ctx.i32Ty);
      acc = ctx.B.CreateAdd(acc, ctx.B.CreateMul(a, b));
    }
    ctx.writeReg32(op.dst(), acc);
    hr.handled = true;
    return hr;
  }
  // v_dot2c_i32_i16: dst += src0.lo16 * src1.lo16 + src0.hi16 * src1.hi16
  if (sop == SemOp::V_DOT2C_I32_I16) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a_lo = ctx.B.CreateSExt(ctx.B.CreateTrunc(s0, i16Ty), ctx.i32Ty);
    Value *a_hi = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s0, 16), i16Ty), ctx.i32Ty);
    Value *b_lo = ctx.B.CreateSExt(ctx.B.CreateTrunc(s1, i16Ty), ctx.i32Ty);
    Value *b_hi = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s1, 16), i16Ty), ctx.i32Ty);
    Value *dot = ctx.B.CreateAdd(ctx.B.CreateMul(a_lo, b_lo), ctx.B.CreateMul(a_hi, b_hi));
    Value *acc = ctx.regs.readReg32(ctx.B, op.dst());
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(acc, dot, "dot2c"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_U32_U24) {
    Value *a = ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF)), ctx.i64Ty);
    Value *b = ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF)), ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateLShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "mulhi24"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_I32_I24) {
    Value *a = ctx.B.CreateAShr(ctx.B.CreateShl(op.src(0), 8), 8);
    Value *b = ctx.B.CreateAShr(ctx.B.CreateShl(op.src(1), 8), 8);
    Value *a64 = ctx.B.CreateSExt(a, ctx.i64Ty), *b64 = ctx.B.CreateSExt(b, ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateAShr(ctx.B.CreateMul(a64, b64), 32), ctx.i32Ty, "mulhi24i"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAD_U32_U24) {
    Value *a = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    Value *b = ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src(2), "mad24"));
    hr.handled = true;
    return hr;
  }
  // v_writelane_b32 / v_readlane_b32 are handled in handle_valu_cross_lane.cpp.

  // v_bfe_u32: Bit Field Extract Unsigned
  // D.u = (S0.u >> S1.u[4:0]) & ((1 << S2.u[4:0]) - 1)
  if (sop == SemOp::V_BFE_U32) {
    Value *base = op.src(0), *offset = op.src(1), *width = op.src(2);
    offset = ctx.B.CreateAnd(offset, ConstantInt::get(ctx.i32Ty, 31));
    width = ctx.B.CreateAnd(width, ConstantInt::get(ctx.i32Ty, 31));
    Value *shifted = ctx.B.CreateLShr(base, offset);
    // width is 0-31 after masking, so shl i32 1, width is always valid
    Value *mask = ctx.B.CreateSub(ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), width),
                              ConstantInt::get(ctx.i32Ty, 1));
    Value *isZeroWidth = ctx.B.CreateICmpEQ(width, ConstantInt::get(ctx.i32Ty, 0));
    Value *result = ctx.B.CreateAnd(shifted, mask, "bfe");
    result = ctx.B.CreateSelect(isZeroWidth, ConstantInt::get(ctx.i32Ty, 0), result);
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }
  // v_bfe_i32: signed Bit Field Extract.
  //   D.i = signext(bits [off+width-1 : off] of src), treating src as if
  //         it had been sign-extended past bit 31 first.
  // Implementation: arithmetic right shift by off (fills high bits with
  // src's sign), then mask to `width` low bits and sign-extend from bit
  // (width-1).  Using LShr instead of AShr here would give mask-and-sx
  // only when off+width <= 32; hardware diverges in the wraparound case,
  // so we must use AShr to stay bit-identical to native v_bfe_i32.
  //
  // Note: this is NOT the same formula as s_bfe_i32 — the scalar form
  // uses a shift-trick (`(src << (32-off-w)) >> (32-w)`), the vector
  // form uses mask-and-sign-extend.  The two hardware blocks differ on
  // the wraparound case; do not "unify" them.
  if (sop == SemOp::V_BFE_I32) {
    Value *base = op.src(0), *offset = op.src(1), *width = op.src(2);
    Value *c31 = ConstantInt::get(ctx.i32Ty, 0x1F);
    offset = ctx.B.CreateAnd(offset, c31);
    width = ctx.B.CreateAnd(width, c31);
    Value *shifted = ctx.B.CreateAShr(base, offset);
    // Build a mask of `width` low bits.  For width == 0 the result is 0
    // (nothing to extract), so we special-case that below and use a
    // safe shift amount (1) here to avoid UB in the mask computation.
    Value *widthNonZero = ctx.B.CreateICmpNE(width,
                                             ConstantInt::get(ctx.i32Ty, 0));
    Value *maskShift = ctx.B.CreateSelect(widthNonZero, width,
                                          ConstantInt::get(ctx.i32Ty, 1));
    Value *mask = ctx.B.CreateSub(
        ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), maskShift),
        ConstantInt::get(ctx.i32Ty, 1));
    Value *field = ctx.B.CreateAnd(shifted, mask);
    Value *widthMinus1 = ctx.B.CreateSub(maskShift,
                                         ConstantInt::get(ctx.i32Ty, 1));
    Value *signBit = ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1),
                                     widthMinus1);
    Value *sx = ctx.B.CreateSub(ctx.B.CreateXor(field, signBit), signBit,
                                "vbfe_i");
    Value *result = ctx.B.CreateSelect(widthNonZero, sx,
                                       ConstantInt::get(ctx.i32Ty, 0));
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }
  // v_mbcnt_{lo,hi}_u32_b32 are handled in handle_valu_cross_lane.cpp.
  // ---- 64-bit float ops ----
  if (sop == SemOp::V_ADD_F64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "vadd_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_F64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "vmul_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  // v_rcp_f64: VOP1 transcendental, single F64 source → F64 result.
  // The hardware op is a ~26-bit accurate reciprocal approximation
  // (TRANS-class, WriteTrans64). Lift to `llvm.amdgcn.rcp.f64` so
  // the AMDGPU backend isels straight back to v_rcp_f64 on gfx942
  // (no Newton-Raphson refinement is added). A generic `fdiv 1.0,
  // x` would lower to a software divide sequence here unless `arcp`
  // / fast-math flags are present, which would be a silent
  // semantics change versus the source op. See the V_RCP_F64
  // SemOp comment in semop.hpp for the rationale.
  if (sop == SemOp::V_RCP_F64) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *s = ctx.B.CreateBitCast(op.src64(0), f64Ty);
    Function *rcp = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_rcp, {f64Ty});
    Value *r = ctx.B.CreateCall(rcp, {s}, "vrcp_f64");
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(r, ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FMA_F64 || sop == SemOp::V_FMAC_F64) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *s0, *s1, *s2;
    if (sop == SemOp::V_FMA_F64) {
      s0 = op.src64(0); s1 = op.src64(1); s2 = op.src64(2);
    } else {
      s0 = op.src64(0); s1 = op.src64(1);
      s2 = ctx.B.CreateBitCast(ctx.regs.readReg64(ctx.B, op.dst()), f64Ty);
    }
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    if (s2->getType() != f64Ty) s2 = ctx.B.CreateBitCast(s2, f64Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {f64Ty});
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "vfma_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F64_U32) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateUIToFP(op.src(0), f64Ty, "cvt_f64_u32"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F64_I32) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateSIToFP(op.src(0), f64Ty, "cvt_f64_i32"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_U32_F64) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *v = ctx.B.CreateBitCast(op.src64(0), f64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateFPToUI(v, ctx.i32Ty, "cvt_u32_f64"));
    hr.handled = true;
    return hr;
  }

  // ---- Reversed-operand shifts ----
  if (sop == SemOp::V_LSHRREV_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateLShr(op.src(1), op.src(0), "vlshr"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHLREV_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateShl(op.src(1), op.src(0), "vlshl"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ASHRREV_I32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAShr(op.src(1), op.src(0), "vashr"));
    hr.handled = true;
    return hr;
  }

  // ---- FP ALU (srcF applies VOP3 neg/abs modifiers) ----
  if (sop == SemOp::V_ADD_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "fadd"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "fmul"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUB_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFSub(s0, s1, "fsub"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUBREV_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFSub(s1, s0, "fsubrev"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_F32 || sop == SemOp::V_MAX_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "fmax"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_F32 || sop == SemOp::V_MIN_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "fmin"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAXIMUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maximum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "fmaximum"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MINIMUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minimum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "fminimum"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FMA_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "fma"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FMAC_F32) {
    ParsedReg dstReg = op.dst();
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *dv = ctx.regs.readReg32(ctx.B, dstReg);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (dv->getType() != ctx.f32Ty) dv = ctx.B.CreateBitCast(dv, ctx.f32Ty);
    Function *fmuladd = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fmuladd, {ctx.f32Ty});
    ctx.writeReg32(dstReg, ctx.B.CreateBitCast(ctx.B.CreateCall(fmuladd, {s0, s1, dv}, "fmac"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_fmamk_f32 dst, src0, K, src2: dst = src0 * K + src2
  // Operand order from MC disassembler: srcF(0)=src0, srcF(1)=K (literal),
  // srcF(2)=src2. Same ordering applies to v_fmaak_f32 below.
  if (sop == SemOp::V_FMAMK_F32) {
    Value *s0 = op.srcF(0), *k = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (k->getType() != ctx.f32Ty) k = ctx.B.CreateBitCast(k, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, k, s2}, "fmamk"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_fmaak_f32 dst, src0, src1, K: dst = src0 * src1 + K
  if (sop == SemOp::V_FMAAK_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *k = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (k->getType() != ctx.f32Ty) k = ctx.B.CreateBitCast(k, ctx.f32Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, k}, "fmaak"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- Division helpers (VOP3) ----
  if (sop == SemOp::V_DIV_SCALE_F32) {
    // amdgcn_div_scale(Numerator, Denominator, i1 select_quotient)
    // src2 in the HW instruction equals either src0 or src1 to indicate
    // which is selected. We determine select_quotient by checking if
    // src2 and src0 refer to the same register operand.
    Value *s0 = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.srcF(1), ctx.f32Ty);
    bool selectNumerator = false;
    if (op.isSrcReg(2) && op.isSrcReg(0)) {
      ParsedReg r2 = op.srcReg(2), r0 = op.srcReg(0);
      selectNumerator = (r2.kind == r0.kind && r2.baseIdx == r0.baseIdx);
    }
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_div_scale,
                                                     {ctx.f32Ty});
    Value *r = ctx.B.CreateCall(fn, {s0, s1,
                 selectNumerator ? ctx.B.getTrue() : ctx.B.getFalse()}, "divscale");
    ctx.writeReg32(op.dst(0), ctx.B.CreateBitCast(ctx.B.CreateExtractValue(r, 0), ctx.i32Ty));
    // Write the boolean flag to the actual SDST destination (operand 1):
    // vcc_lo, sN, or null. The kernel saves flags to SGPRs and later
    // restores them to VCC via s_mov_b32 before each v_div_fmas_f32.
    Value *flag = ctx.B.CreateExtractValue(r, 1);
    if (di.numDefs >= 2 && di.isReg(1)) {
      ParsedReg flagDst = op.dst(1);
      if (flagDst.kind == ParsedReg::VCC)
        ctx.regs.storeVCC(ctx.B, flag);
      else if (flagDst.kind == ParsedReg::SGPR && flagDst.baseIdx >= 0)
        ctx.regs.storeSGPR32(ctx.B, flagDst.baseIdx, ctx.B.CreateZExt(flag, ctx.i32Ty));
      // NOREG (null) or unrecognized → discard the flag
    } else {
      ctx.regs.storeVCC(ctx.B, flag);
    }
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_DIV_FIXUP_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.srcF(1), ctx.f32Ty);
    Value *s2 = ctx.B.CreateBitCast(op.srcF(2), ctx.f32Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_div_fixup,
                                                     {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {s0, s1, s2}, "divfixup"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_DIV_FMAS_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.srcF(1), ctx.f32Ty);
    Value *s2 = ctx.B.CreateBitCast(op.srcF(2), ctx.f32Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_div_fmas,
                                                     {ctx.f32Ty});
    Value *vcc = ctx.regs.loadVCC(ctx.B);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {s0, s1, s2, vcc}, "divfmas"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- 3-source integer VOP3 ----
  if (sop == SemOp::V_ADD3_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(ctx.B.CreateAdd(op.src(0), op.src(1)), op.src(2), "vadd3"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_ADD_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(ctx.B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshl_add"));
    hr.handled = true;
    return hr;
  }
  // v_add_lshl_u32: fused three-input "add then shift".
  //   D.u = (S0.u + S1.u) << S2.u[4:0]
  // Unsigned wrap on the add is well-defined (CreateAdd defaults to
  // "may wrap"), matching hardware.  The shift amount must be masked to
  // 5 bits up front — AMDGPU shifts only consume S2[4:0], but LLVM's
  // `shl` with a shift >= bit-width is poison, so an un-masked `op.src(2)`
  // containing any high bits would silently corrupt the IR.  V_ADD_LSHL
  // has no carry-out and writes no SCC/VCC, so this is the whole op.
  if (sop == SemOp::V_ADD_LSHL_U32) {
    Value *sum = ctx.B.CreateAdd(op.src(0), op.src(1));
    Value *shamt = ctx.B.CreateAnd(op.src(2),
                                   ConstantInt::get(ctx.i32Ty, 0x1F));
    ctx.writeReg32(op.dst(), ctx.B.CreateShl(sum, shamt, "vadd_lshl"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_OR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(ctx.B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshlor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_AND_OR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(ctx.B.CreateAnd(op.src(0), op.src(1)), op.src(2), "vandor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_OR3_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(ctx.B.CreateOr(op.src(0), op.src(1)), op.src(2), "vor3"));
    hr.handled = true;
    return hr;
  }
  // VOP3 v_xor3_b32: 3-way xor. Direct mirror of V_OR3_B32 above
  // — the .td iselect pattern is `(xor (xor a, b), c)` (see
  // VOP3Instructions.td:1350); both nested and outer xor lift to
  // plain CreateXor with no source modifiers (B32 ops carry only
  // ABS/NEG-style modifiers on the FP forms, not the bitwise
  // ones).
  if (sop == SemOp::V_XOR3_B32) {
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateXor(ctx.B.CreateXor(op.src(0), op.src(1)),
                                   op.src(2), "vxor3"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_BITOP3_B32 || sop == SemOp::V_BITOP3_B16) {
    // v_bitop3 dst, src0, src1, src2, imm8
    // For each bit position i, dst[i] = LUT[4*src0[i] + 2*src1[i] + src2[i]]
    // Expand as: result = OR of (minterm_i AND expand(LUT[i])) for i in 0..7
    Value *a = op.src(0), *b = op.src(1), *c = op.src(2);
    Value *imm = op.src(3);
    uint64_t lutConst = 0;
    bool lutIsConst = false;
    if (auto *CI = dyn_cast<ConstantInt>(imm)) {
      lutConst = CI->getZExtValue() & 0xFF;
      lutIsConst = true;
    }
    Value *na = ctx.B.CreateNot(a), *nb = ctx.B.CreateNot(b), *nc = ctx.B.CreateNot(c);
    Value *allOnes = ConstantInt::get(ctx.i32Ty, ~0U);
    Value *result = ConstantInt::get(ctx.i32Ty, 0);
    Value *minterms[8] = {
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), nc),  // 0: ~a & ~b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), c),   // 1: ~a & ~b &  c
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), nc),   // 2: ~a &  b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), c),    // 3: ~a &  b &  c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), nc),   // 4:  a & ~b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), c),    // 5:  a & ~b &  c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), nc),    // 6:  a &  b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), c),     // 7:  a &  b &  c
    };
    if (lutIsConst) {
      for (int i = 0; i < 8; i++)
        if (lutConst & (1 << i))
          result = ctx.B.CreateOr(result, minterms[i]);
    } else {
      for (int i = 0; i < 8; i++) {
        Value *bit = ctx.B.CreateAnd(ctx.B.CreateLShr(imm, ConstantInt::get(ctx.i32Ty, i)),
                                 ConstantInt::get(ctx.i32Ty, 1));
        Value *mask = ctx.B.CreateSub(ConstantInt::get(ctx.i32Ty, 0), bit);
        result = ctx.B.CreateOr(result, ctx.B.CreateAnd(minterms[i], mask));
      }
    }
    if (sop == SemOp::V_BITOP3_B16)
      result = ctx.B.CreateAnd(result, ConstantInt::get(ctx.i32Ty, 0xFFFF));
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }
  // VOP3 v_max3_u32: 3-way unsigned max, ternary. The .td pattern
  // is `AMDGPUumax3` = `umax(umax(a,b), c)`; we lift it as the
  // direct ICmp+Select chain (matches the V_MAX_U32 idiom one
  // block above so the intermediate pair `vmax3_lo` reuses the
  // same shape and a future refactor that switches V_MAX_U32 to
  // an llvm.umax intrinsic can lift this in lockstep).
  if (sop == SemOp::V_MAX3_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1), *s2 = op.src(2);
    Value *m01 =
        ctx.B.CreateSelect(ctx.B.CreateICmpUGT(s0, s1), s0, s1, "vmax3_lo");
    ctx.writeReg32(
        op.dst(),
        ctx.B.CreateSelect(ctx.B.CreateICmpUGT(m01, s2), m01, s2, "vmax3"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX3_F32 || sop == SemOp::V_MAX3_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    Value *m01 = ctx.B.CreateCall(maxFn, {s0, s1});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {m01, s2}, "max3"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN3_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    Value *m01 = ctx.B.CreateCall(minFn, {s0, s1});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {m01, s2}, "min3"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MED3_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    Value *mn01 = ctx.B.CreateCall(minFn, {s0, s1});
    Value *mx01 = ctx.B.CreateCall(maxFn, {s0, s1});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {mn01, ctx.B.CreateCall(minFn, {mx01, s2})}, "med3"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_cvt_pkrtz_f16_f32: pack two f32 into <2 x f16> with round-to-zero.
  // Maps directly onto the dedicated hardware intrinsic so the backend
  // keeps the RTZ rounding mode (a plain FPTrunc uses round-to-nearest).
  if (sop == SemOp::V_CVT_PKRTZ_F16_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_cvt_pkrtz);
    Value *v2h = ctx.B.CreateCall(fn, {s0, s1}, "pkrtz");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(v2h, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_cvt_pk_f16_f32: pack two f32 into <2 x f16> with round-to-nearest-even
  // (the default IEEE rounding). No dedicated intrinsic exists; a pair of
  // FPTrunc operations followed by a packed i32 assembly is the canonical
  // lowering and the backend recognises the pattern.
  if (sop == SemOp::V_CVT_PK_F16_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Type *halfTy = Type::getHalfTy(ctx.C);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *h0 = ctx.B.CreateFPTrunc(s0, halfTy, "pk_h0");
    Value *h1 = ctx.B.CreateFPTrunc(s1, halfTy, "pk_h1");
    Value *b0 = ctx.B.CreateZExt(ctx.B.CreateBitCast(h0, i16Ty), ctx.i32Ty);
    Value *b1 = ctx.B.CreateZExt(ctx.B.CreateBitCast(h1, i16Ty), ctx.i32Ty);
    ctx.writeReg32(op.dst(),
        ctx.B.CreateOr(b0, ctx.B.CreateShl(b1, 16), "pk_f16"));
    hr.handled = true;
    return hr;
  }
  // v_cvt_scalef32_pk_fp4_f32 vdst, src0_f32, src1_f32, scale_f32 op_sel:[..]
  //
  // Converts two f32 sources to FP4 and packs them into one of the four 8-bit
  // slots of vdst (selected by op_sel bits 0..3), using a scalar f32 scale.
  // The remaining slots of the old vdst value are preserved — this is
  // captured by the intrinsic's tied `old_vdst` argument.
  if (sop == SemOp::V_CVT_SCALEF32_PK_FP4_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *scale = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (scale->getType() != ctx.f32Ty)
      scale = ctx.B.CreateBitCast(scale, ctx.f32Ty);
    // Extract the destination nibble index from op_sel. LLVM disasm prints
    // op_sel:[0,0,0,0] with the 4th entry being the slot selector.
    int opSel[4] = {0, 0, 0, 0};
    StringRef text(di.fullText);
    auto pos = text.find("op_sel:");
    if (pos != StringRef::npos) {
      auto brk = text.find('[', pos);
      auto end = text.find(']', brk);
      if (brk != StringRef::npos && end != StringRef::npos) {
        StringRef inner = text.slice(brk + 1, end);
        SmallVector<StringRef, 4> parts;
        inner.split(parts, ',');
        for (unsigned i = 0; i < parts.size() && i < 4; i++) {
          int val = 0;
          if (!parts[i].trim().getAsInteger(10, val))
            opSel[i] = val;
        }
      }
    }
    // Dest-slot index is packed as bits[3:2]+bit[0] per the HW op_sel
    // layout (see LLVM's SIInstrInfo::lowerScaleCvt for reference); for the
    // common `op_sel:[0,0,0,0]` form the selector is simply 0.
    unsigned dstSel = (unsigned)opSel[3];
    Value *oldVdst = ctx.regs.readReg32(ctx.B, op.dst());
    Function *fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_cvt_scalef32_pk_fp4_f32);
    Value *r = ctx.B.CreateCall(
        fn, {oldVdst, s0, s1, scale, ConstantInt::get(ctx.i32Ty, dstSel)},
        "scalef32_pk_fp4");
    ctx.writeReg32(op.dst(), r);
    hr.handled = true;
    return hr;
  }
  // v_mov_b64 vdst:64, src:64
  if (sop == SemOp::V_MOV_B64) {
    ctx.writeReg64(op.dst(), op.src64(0));
    hr.handled = true;
    return hr;
  }
  // v_swap_b32 vdstA, vdstB / uses vdstA, vdstB - exchange two VGPRs.
  // MC encoding has two defs (vdst, vdst_in) and two uses (src0, src0_in).
  // The old values of both registers swap: src0 -> vdst and old-vdst ->
  // vdst_in.
  if (sop == SemOp::V_SWAP_B32) {
    // vdst = old src0; vdst_in(== src0's slot) = old vdst.
    ParsedReg dstA = op.dst(0);
    ParsedReg dstB = (di.numDefs >= 2) ? op.dst(1) : op.srcReg(0);
    Value *vA = ctx.regs.readReg32(ctx.B, dstA);
    Value *vB = ctx.regs.readReg32(ctx.B, dstB);
    ctx.writeReg32(dstA, vB);
    ctx.writeReg32(dstB, vA);
    hr.handled = true;
    return hr;
  }
  // v_cvt_f32_bf16: low 16 bits of src are interpreted as bfloat16.
  if (sop == SemOp::V_CVT_F32_BF16) {
    Type *bfTy = Type::getBFloatTy(ctx.C);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *bits = ctx.B.CreateTrunc(op.src(0), i16Ty);
    Value *bf = ctx.B.CreateBitCast(bits, bfTy);
    Value *f = ctx.B.CreateFPExt(bf, ctx.f32Ty, "cvt_bf16");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(f, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_bfm_b32: D = ((1 << src0[4:0]) - 1) << src1[4:0]
  if (sop == SemOp::V_BFM_B32) {
    Value *width  = ctx.B.CreateAnd(op.src(0),
        ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *offset = ctx.B.CreateAnd(op.src(1),
        ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *ones   = ctx.B.CreateSub(
        ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), width),
        ConstantInt::get(ctx.i32Ty, 1));
    // width==0 must yield 0 — the 1<<0 base case would otherwise leave a
    // single bit set. Mask it out explicitly rather than relying on the
    // subtraction underflow.
    Value *isZero = ctx.B.CreateICmpEQ(width, ConstantInt::get(ctx.i32Ty, 0));
    ones = ctx.B.CreateSelect(isZero, ConstantInt::get(ctx.i32Ty, 0), ones);
    ctx.writeReg32(op.dst(), ctx.B.CreateShl(ones, offset, "bfm"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_PK_BF16_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    auto *bfTy = Type::getBFloatTy(ctx.C);
    Value *bf0 = ctx.B.CreateFPTrunc(s0, bfTy, "tobf16_0");
    Value *bf1 = ctx.B.CreateFPTrunc(s1, bfTy, "tobf16_1");
    Value *bits0 = ctx.B.CreateZExt(ctx.B.CreateBitCast(bf0, Type::getInt16Ty(ctx.C)), ctx.i32Ty);
    Value *bits1 = ctx.B.CreateZExt(ctx.B.CreateBitCast(bf1, Type::getInt16Ty(ctx.C)), ctx.i32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(bits0, ctx.B.CreateShl(bits1, 16), "pk_bf16"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_PK_FP8_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    // v_cvt_pk_fp8_f32 packs two f32 into two fp8 values in the low 16 bits.
    // The "old" value and word_sel determine where in the dest the result goes.
    // src2 = old value, src3 (imm) = word_sel.
    // Use the LLVM intrinsic which handles this correctly.
    Value *oldVal = (op.nSrcs() >= 3) ? op.src(2) : ConstantInt::get(ctx.i32Ty, 0);
    bool wordSel = (op.nSrcs() >= 4 && di.isImm(op.srcIdx(3))) ? (op.srcImm(3) != 0) : false;
    Function *cvtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_cvt_pk_fp8_f32);
    ctx.writeReg32(op.dst(), ctx.B.CreateCall(cvtFn,
        {s0, s1, oldVal, ConstantInt::get(ctx.i1Ty, wordSel)}, "pk_fp8"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_PK_BF8_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Value *oldVal = (op.nSrcs() >= 3) ? op.src(2) : ConstantInt::get(ctx.i32Ty, 0);
    bool wordSel = (op.nSrcs() >= 4 && di.isImm(op.srcIdx(3))) ? (op.srcImm(3) != 0) : false;
    Function *cvtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_cvt_pk_bf8_f32);
    ctx.writeReg32(op.dst(), ctx.B.CreateCall(cvtFn,
        {s0, s1, oldVal, ConstantInt::get(ctx.i1Ty, wordSel)}, "pk_bf8"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_PERM_B32) {
    Function *permFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_perm);
    ctx.writeReg32(op.dst(), ctx.B.CreateCall(permFn, {op.src(0), op.src(1), op.src(2)}, "perm"));
    hr.handled = true;
    return hr;
  }

  // ---- 64-bit vector ops ----
  if (sop == SemOp::V_LSHLREV_B64) {
    Value *shamt = op.src(0);
    Value *src = op.src64(1);
    if (src->getType() != ctx.i64Ty) src = ctx.B.CreateBitOrPointerCast(src, ctx.i64Ty);
    Value *shamtExt = ctx.B.CreateZExt(shamt, ctx.i64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateShl(src, shamtExt, "shl"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_ADD_U64) {
    Value *src0 = op.src64(0);
    Value *shamt = op.src(1);
    Value *src2 = op.src64(2);
    if (src0->getType()->isPointerTy()) src0 = ctx.B.CreatePtrToInt(src0, ctx.i64Ty);
    if (src0->getType() != ctx.i64Ty) src0 = ctx.B.CreateBitOrPointerCast(src0, ctx.i64Ty);
    if (src2->getType() != ctx.i64Ty) src2 = ctx.B.CreateBitOrPointerCast(src2, ctx.i64Ty);
    Value *shamtExt = ctx.B.CreateZExt(shamt, ctx.i64Ty);
    Value *shifted = ctx.B.CreateShl(src0, shamtExt);
    ctx.writeReg64(op.dst(), ctx.B.CreateAdd(shifted, src2, "lshl_add"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ADD_NC_U64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    if (s0->getType() != ctx.i64Ty) s0 = ctx.B.CreateBitOrPointerCast(s0, ctx.i64Ty);
    if (s1->getType() != ctx.i64Ty) s1 = ctx.B.CreateBitOrPointerCast(s1, ctx.i64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateAdd(s0, s1, "vadd64"));
    hr.handled = true;
    return hr;
  }

  // ---- v_mad_u64_u32 (2 defs: VDST + SDST, firstSrcIdx=2) ----
  if (sop == SemOp::V_MAD_U64_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src64(2), "vmad64");
    ctx.writeReg64(op.dst(0), res);
    ctx.writeReg64(op.dst(1), ConstantInt::get(ctx.i64Ty, 0));
    hr.handled = true;
    return hr;
  }

  // ---- Vector compares (V_CMP / V_CMPX) ----
  // Extracted to handle_valu_vcmp.cpp.
  {
    HandlerResult sub = handleVALU_Vcmp(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  // ---- VOP3P / WMMA / v_fma_mix_f32 / v_cndmask_b32 ----
  // Extracted to handle_valu_vop3p.cpp.
  {
    HandlerResult sub = handleVALU_VOP3P(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  return hr;
}

} // namespace transpiler
