#include "handlers.hpp"
#include "raiser.hpp"
#include "wmma_lowering.hpp"

#include "semop.hpp"
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
                        OpResolver &op, RaiseResult &result) {
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
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_PERMLANE16_B32 || sop == SemOp::V_PERMLANEX16_B32 ||
      sop == SemOp::V_PERMLANE64_B32) {
    if (di.numDefs >= 1 && di.numSrcs >= 1) {
      ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    }
    hr.handled = true;
    return hr;
  }
  // In our scalar model v_readfirstlane_b32 is a VGPR→SGPR move
  if (sop == SemOp::V_READFIRSTLANE_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }

  // ---- Type conversions ----
  if (sop == SemOp::V_CVT_F32_U32) {
    Value *r = ctx.B.CreateUIToFP(op.src(0), ctx.f32Ty, "cvt");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F32_I32) {
    Value *r = ctx.B.CreateSIToFP(op.src(0), ctx.f32Ty, "cvt");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_U32_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateFPToUI(s, ctx.i32Ty, "cvt"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_I32_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateFPToSI(s, ctx.i32Ty, "cvt"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F16_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *h = ctx.B.CreateFPTrunc(s, Type::getHalfTy(ctx.C), "cvt");
    Value *bits = ctx.B.CreateBitCast(h, Type::getInt16Ty(ctx.C));
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(bits, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *b = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty), ctx.f16Ty);
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateFMul(a, b, "mul_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ADD_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *b = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty), ctx.f16Ty);
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateFAdd(a, b, "add_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUB_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *b = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty), ctx.f16Ty);
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateFSub(a, b, "sub_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *b = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty), ctx.f16Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f16Ty});
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {a, b}, "max_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_pk_fmac_f16: packed <2xf16> fused multiply-accumulate into dst
  if (sop == SemOp::V_PK_FMAC_F16) {
    auto *v2f16 = FixedVectorType::get(ctx.f16Ty, 2);
    Value *s0 = ctx.B.CreateBitCast(op.src(0), v2f16);
    Value *s1 = ctx.B.CreateBitCast(op.src(1), v2f16);
    Value *acc = ctx.B.CreateBitCast(ctx.regs.readReg32(ctx.B, op.dst()), v2f16);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {v2f16});
    Value *res = ctx.B.CreateCall(fma, {s0, s1, acc}, "pk_fmac");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_mac_f16 / v_fmac_f16: dst = src0 * src1 + dst (f16)
  if (sop == SemOp::V_MAC_F16 || sop == SemOp::V_FMAC_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *s0 = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *s1 = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty), ctx.f16Ty);
    Value *acc = ctx.B.CreateBitCast(ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, op.dst()), i16Ty), ctx.f16Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f16Ty});
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, acc}, "mac_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUBREV_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *b = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty), ctx.f16Ty);
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateFSub(b, a, "subrev_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FLOOR_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *s = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::floor, {ctx.f16Ty});
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {s}, "floor_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F16_U16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *s = ctx.B.CreateTrunc(op.src(0), i16Ty);
    Value *res = ctx.B.CreateUIToFP(s, ctx.f16Ty, "cvt_f16_u16");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(ctx.B.CreateBitCast(res, i16Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_U16_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *s = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *res = ctx.B.CreateFPToUI(s, i16Ty, "cvt_u16_f16");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LDEXP_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *s0 = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *s1 = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Function *ldexpFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::ldexp, {ctx.f16Ty, i16Ty});
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateCall(ldexpFn, {s0, s1}, "ldexp_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_F16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty), ctx.f16Ty);
    Value *b = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty), ctx.f16Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f16Ty});
    Value *res = ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {a, b}, "min_f16"), i16Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_U16 || sop == SemOp::V_MIN_U16 ||
      sop == SemOp::V_MAX_I16 || sop == SemOp::V_MIN_I16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a = ctx.B.CreateTrunc(op.src(0), i16Ty);
    Value *b = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Value *cmp;
    if (sop == SemOp::V_MAX_U16)      cmp = ctx.B.CreateICmpUGT(a, b);
    else if (sop == SemOp::V_MIN_U16) cmp = ctx.B.CreateICmpULT(a, b);
    else if (sop == SemOp::V_MAX_I16) cmp = ctx.B.CreateICmpSGT(a, b);
    else                               cmp = ctx.B.CreateICmpSLT(a, b);
    Value *res = ctx.B.CreateSelect(cmp, a, b, "i16sel");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ASHRREV_I16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    // HW uses src0[3:0]; mask to avoid UB from shift >= bitwidth.
    Value *shamt = ctx.B.CreateAnd(ctx.B.CreateTrunc(op.src(0), i16Ty),
                                   ConstantInt::get(i16Ty, 0xF));
    Value *base = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Value *res = ctx.B.CreateAShr(base, shamt, "vashr16");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHRREV_B16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *shamt = ctx.B.CreateAnd(ctx.B.CreateTrunc(op.src(0), i16Ty),
                                   ConstantInt::get(i16Ty, 0xF));
    Value *base = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Value *res = ctx.B.CreateLShr(base, shamt, "vlshr16");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHLREV_B16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *shamt = ctx.B.CreateAnd(ctx.B.CreateTrunc(op.src(0), i16Ty),
                                   ConstantInt::get(i16Ty, 0xF));
    Value *base = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Value *res = ctx.B.CreateShl(base, shamt, "vlshl16");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_PACK_B32_F16) {
    Value *lo = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFF));
    Value *hi = ctx.B.CreateShl(ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFF)), 16);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(lo, hi, "pack_f16"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F32_F16) {
    Value *bits = ctx.B.CreateTrunc(op.src(0), Type::getInt16Ty(ctx.C));
    Value *h = ctx.B.CreateBitCast(bits, Type::getHalfTy(ctx.C));
    Value *f = ctx.B.CreateFPExt(h, ctx.f32Ty, "cvt");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(f, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F32_UBYTE0) {
    Value *byte = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFF));
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F32_UBYTE1) {
    Value *byte = ctx.B.CreateAnd(ctx.B.CreateLShr(op.src(0), 8), ConstantInt::get(ctx.i32Ty, 0xFF));
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F32_UBYTE2) {
    Value *byte = ctx.B.CreateAnd(ctx.B.CreateLShr(op.src(0), 16), ConstantInt::get(ctx.i32Ty, 0xFF));
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F32_UBYTE3) {
    Value *byte = ctx.B.CreateLShr(op.src(0), 24);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_BFREV_B32) {
    Function *brev = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::bitreverse, {ctx.i32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateCall(brev, {op.src(0)}, "bfrev"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_NOT_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateNot(op.src(0), "vnot"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_RCP_IFLAG_F32 || sop == SemOp::V_RCP_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *r = ctx.B.CreateFDiv(ConstantFP::get(ctx.f32Ty, 1.0), s, "rcp");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_EXP_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *exp2Fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::exp2, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(exp2Fn, {s}, "exp"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LOG_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *log2Fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::log2, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(log2Fn, {s}, "log"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LDEXP_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *s1 = op.src(1);
    Function *ldexpFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::ldexp, {ctx.f32Ty, ctx.i32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(ldexpFn, {s0, s1}, "ldexp"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SQRT_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *sqrtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::sqrt, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(sqrtFn, {s}, "sqrt"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_RSQ_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *sqrtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::sqrt, {ctx.f32Ty});
    Value *sq = ctx.B.CreateCall(sqrtFn, {s}, "sqrt");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFDiv(ConstantFP::get(ctx.f32Ty, 1.0), sq, "rsq"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FLOOR_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *floorFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::floor, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(floorFn, {s}, "floor"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CEIL_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *ceilFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::ceil, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(ceilFn, {s}, "ceil"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_TRUNC_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *truncFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::trunc, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(truncFn, {s}, "trunc"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FRACT_F32) {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *floorFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::floor, {ctx.f32Ty});
    Value *fl = ctx.B.CreateCall(floorFn, {s}, "floor");
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFSub(s, fl, "fract"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- Simple 2-src integer ALU ----
  if (sop == SemOp::V_ADD_NC_U32 || sop == SemOp::V_ADD_I32_legacy) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateAdd(op.src(0), op.src(1), "vadd"));
    hr.handled = true;
    return hr;
  }
  // Vector add with carry-out (GFX12: v_add_co_u32; VCC = carry)
  if (sop == SemOp::V_ADD_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateAdd(s0, s1, "vadd_co");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    auto *ov = ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {ctx.i32Ty}, {s0, s1});
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateExtractValue(ov, 1));
    hr.handled = true;
    return hr;
  }
  // Vector sub with carry-out (GFX9: v_sub_u32; GFX10+: v_sub_co_u32)
  if (sop == SemOp::V_SUB_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateSub(s0, s1, "vsub_co");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateICmpULT(s0, s1));
    hr.handled = true;
    return hr;
  }
  // Vector reversed sub with carry-out (GFX9: v_subrev_u32; GFX10+: v_subrev_co_u32)
  if (sop == SemOp::V_SUBREV_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateSub(s1, s0, "vsubrev_co");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
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
    ctx.regs.writeReg32(ctx.B, op.dst(), diff2);
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
    ctx.regs.writeReg32(ctx.B, op.dst(), diff2);
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
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    ctx.regs.storeVCC(ctx.B, ctx.B.CreateOr(c1, c2));
    hr.handled = true;
    return hr;
  }
  // v_mad_co_u64_u32: D.u64 = S0.u32 * S1.u32 + S2.u64, VCC = carry
  if (sop == SemOp::V_MAD_CO_U64_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src64(2), "vmad_co64");
    ctx.regs.writeReg64(ctx.B, op.dst(0), res);
    hr.handled = true;
    return hr;
  }
  // v_mad_u32: D.u32 = S0.u32 * S1.u32 + S2.u32 (no carry)
  if (sop == SemOp::V_MAD_U32) {
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(op.src(0), op.src(1)), op.src(2), "vmad_u32");
    ctx.regs.writeReg32(ctx.B, op.dst(), res);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_OR_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(op.src(0), op.src(1), "vor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_AND_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateAnd(op.src(0), op.src(1), "vand"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_LO_U32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateMul(op.src(0), op.src(1), "vmul"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUB_NC_U32 || sop == SemOp::V_SUB_I32_legacy) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateSub(op.src(0), op.src(1), "vsub"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUBREV_NC_U32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateSub(op.src(1), op.src(0), "vsubrev"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_XOR_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateXor(op.src(0), op.src(1), "vxor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_XNOR_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateNot(ctx.B.CreateXor(op.src(0), op.src(1)), "vxnor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpUGT(s0, s1), s0, s1, "vmax"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpULT(s0, s1), s0, s1, "vmin"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpSGT(s0, s1), s0, s1, "vmax"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpSLT(s0, s1), s0, s1, "vmin"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateTrunc(ctx.B.CreateLShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "vmulhi"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_I32) {
    Value *a = ctx.B.CreateSExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateSExt(op.src(1), ctx.i64Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateTrunc(ctx.B.CreateAShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "vmulhi"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_U32_U24) {
    Value *a = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    Value *b = ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateMul(a, b, "mul24"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_I32_I24) {
    Value *a = ctx.B.CreateShl(op.src(0), 8);
    a = ctx.B.CreateAShr(a, 8);
    Value *b = ctx.B.CreateShl(op.src(1), 8);
    b = ctx.B.CreateAShr(b, 8);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateMul(a, b, "mul24i"));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), acc);
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
    ctx.regs.writeReg32(ctx.B, op.dst(), acc);
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateAdd(acc, dot, "dot2c"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_U32_U24) {
    Value *a = ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF)), ctx.i64Ty);
    Value *b = ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF)), ctx.i64Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateTrunc(ctx.B.CreateLShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "mulhi24"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_I32_I24) {
    Value *a = ctx.B.CreateAShr(ctx.B.CreateShl(op.src(0), 8), 8);
    Value *b = ctx.B.CreateAShr(ctx.B.CreateShl(op.src(1), 8), 8);
    Value *a64 = ctx.B.CreateSExt(a, ctx.i64Ty), *b64 = ctx.B.CreateSExt(b, ctx.i64Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateTrunc(ctx.B.CreateAShr(ctx.B.CreateMul(a64, b64), 32), ctx.i32Ty, "mulhi24i"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAD_U32_U24) {
    Value *a = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    Value *b = ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src(2), "mad24"));
    hr.handled = true;
    return hr;
  }
  // v_writelane_b32 vDST, sSRC, lane: write sSRC into ONE specific lane
  // of vDST.  On hardware, only the lane whose id matches the `lane` operand
  // has its vDST updated; every other lane keeps its vDST unchanged.
  //
  // Used by the compiler as a "register parking" mechanism: SGPR values are
  // stashed in individual lanes of a single VGPR and later recovered by
  // v_readlane_b32.  Because this is a cross-lane communication primitive,
  // it CANNOT be emulated via per-thread private scratch (each lane has its
  // own scratch) nor via a single scalar SSA value (that would clobber the
  // other lanes' content).
  //
  // We emit `llvm.amdgcn.writelane(val, lane, old)` which lowers directly to
  // the hardware v_writelane_b32 on gfx942.  Our per-lane IR model treats a
  // VGPR as a scalar SSA value; the intrinsic returns the new "per-lane
  // scalar" for this lane (either `val` if lane_id==lane, else `old`), so
  // the VGPR's SSA slot carries the correct value for whichever lane we are.
  if (sop == SemOp::V_WRITELANE_B32) {
    ParsedReg dst = op.dst();
    Value *val = op.src(0);
    Value *lane = op.src(1);
    lane = ctx.B.CreateZExtOrTrunc(lane, ctx.i32Ty, "wrlane_idx");

    // First-write pattern: if the compiler's writelane is the first ever
    // assignment to vDst, the non-selected lanes legitimately hold whatever
    // vDst contained before (hardware semantics).  PoisonValue is the correct
    // IR encoding for "unobservable by any correct program" — any downstream
    // use of those lanes before they are written is itself undefined on
    // hardware, so poisoning them cannot introduce a miscompile.
    Value *oldVal = ctx.regs.readReg32(ctx.B, dst);
    if (!oldVal)
      oldVal = PoisonValue::get(ctx.i32Ty);

    Function *wl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_writelane, {ctx.i32Ty});
    Value *newVal = ctx.B.CreateCall(wl, {val, lane, oldVal}, "writelane");
    ctx.regs.writeReg32(ctx.B, dst, newVal);
    hr.handled = true;
    return hr;
  }
  // v_readlane_b32 sDST, vSRC, lane: read a specific lane of a VGPR into a
  // scalar register.  Reverse of writelane parking; also cross-lane, so it
  // must use the native intrinsic (lowers to hardware v_readlane_b32).
  if (sop == SemOp::V_READLANE_B32) {
    ParsedReg srcReg = op.srcReg(0);
    Value *lane = op.src(1);
    lane = ctx.B.CreateZExtOrTrunc(lane, ctx.i32Ty, "rdlane_idx");

    // Reading from a VGPR that the raiser has never observed being written
    // is a raising bug — the program would read hardware-level garbage.  Do
    // not paper over it with poison; fail loudly so the missing write is
    // noticed and fixed.
    Value *src = ctx.regs.readReg32(ctx.B, srcReg);
    if (!src) {
      std::string msg;
      raw_string_ostream os(msg);
      os << "transpiler: v_readlane_b32 from uninitialized v"
         << srcReg.baseIdx << " at offset 0x";
      os.write_hex(di.offset);
      os << "; raiser missed a prior write.";
      errs() << os.str() << "\n";
      report_fatal_error(StringRef(msg));
    }

    Function *rl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_readlane, {ctx.i32Ty});
    Value *val = ctx.B.CreateCall(rl, {src, lane}, "readlane");
    ctx.regs.writeReg32(ctx.B, op.dst(), val);
    hr.handled = true;
    return hr;
  }
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
    ctx.regs.writeReg32(ctx.B, op.dst(), result);
    hr.handled = true;
    return hr;
  }
  // v_mbcnt_lo_u32_b32: Count bits set in src0 below the current lane
  if (sop == SemOp::V_MBCNT_LO_U32_B32) {
    Function *mbcnt = Intrinsic::getOrInsertDeclaration(&ctx.M,
        Intrinsic::amdgcn_mbcnt_lo, {});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateCall(mbcnt, {op.src(0), op.src(1)}, "mbcnt_lo"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MBCNT_HI_U32_B32) {
    Function *mbcnt = Intrinsic::getOrInsertDeclaration(&ctx.M,
        Intrinsic::amdgcn_mbcnt_hi, {});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateCall(mbcnt, {op.src(0), op.src(1)}, "mbcnt_hi"));
    hr.handled = true;
    return hr;
  }
  // ---- 64-bit float ops ----
  if (sop == SemOp::V_ADD_F64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "vadd_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_F64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "vmul_f64"), ctx.i64Ty));
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
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "vfma_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F64_U32) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateUIToFP(op.src(0), f64Ty, "cvt_f64_u32"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F64_I32) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateSIToFP(op.src(0), f64Ty, "cvt_f64_i32"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_U32_F64) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *v = ctx.B.CreateBitCast(op.src64(0), f64Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateFPToUI(v, ctx.i32Ty, "cvt_u32_f64"));
    hr.handled = true;
    return hr;
  }

  // ---- Reversed-operand shifts ----
  if (sop == SemOp::V_LSHRREV_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateLShr(op.src(1), op.src(0), "vlshr"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHLREV_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateShl(op.src(1), op.src(0), "vlshl"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ASHRREV_I32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateAShr(op.src(1), op.src(0), "vashr"));
    hr.handled = true;
    return hr;
  }

  // ---- FP ALU (srcF applies VOP3 neg/abs modifiers) ----
  if (sop == SemOp::V_ADD_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "fadd"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "fmul"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUB_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFSub(s0, s1, "fsub"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUBREV_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFSub(s1, s0, "fsubrev"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_F32 || sop == SemOp::V_MAX_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "fmax"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_F32 || sop == SemOp::V_MIN_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "fmin"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAXIMUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maximum, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "fmaximum"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MINIMUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minimum, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "fminimum"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FMA_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "fma"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, dstReg, ctx.B.CreateBitCast(ctx.B.CreateCall(fmuladd, {s0, s1, dv}, "fmac"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, k, s2}, "fmamk"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, k}, "fmaak"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(0), ctx.B.CreateBitCast(ctx.B.CreateExtractValue(r, 0), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {s0, s1, s2}, "divfixup"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {s0, s1, s2, vcc}, "divfmas"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- 3-source integer VOP3 ----
  if (sop == SemOp::V_ADD3_U32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateAdd(ctx.B.CreateAdd(op.src(0), op.src(1)), op.src(2), "vadd3"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_ADD_U32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateAdd(ctx.B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshl_add"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_OR_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(ctx.B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshlor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_AND_OR_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(ctx.B.CreateAnd(op.src(0), op.src(1)), op.src(2), "vandor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_OR3_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(ctx.B.CreateOr(op.src(0), op.src(1)), op.src(2), "vor3"));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), result);
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {m01, s2}, "max3"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {m01, s2}, "min3"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {mn01, ctx.B.CreateCall(minFn, {mx01, s2})}, "med3"), ctx.i32Ty));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateOr(bits0, ctx.B.CreateShl(bits1, 16), "pk_bf16"));
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateCall(cvtFn,
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
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateCall(cvtFn,
        {s0, s1, oldVal, ConstantInt::get(ctx.i1Ty, wordSel)}, "pk_bf8"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_PERM_B32) {
    Function *permFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_perm);
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateCall(permFn, {op.src(0), op.src(1), op.src(2)}, "perm"));
    hr.handled = true;
    return hr;
  }

  // ---- 64-bit vector ops ----
  if (sop == SemOp::V_LSHLREV_B64) {
    Value *shamt = op.src(0);
    Value *src = op.src64(1);
    if (src->getType() != ctx.i64Ty) src = ctx.B.CreateBitOrPointerCast(src, ctx.i64Ty);
    Value *shamtExt = ctx.B.CreateZExt(shamt, ctx.i64Ty);
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateShl(src, shamtExt, "shl"));
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
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateAdd(shifted, src2, "lshl_add"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ADD_NC_U64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    if (s0->getType() != ctx.i64Ty) s0 = ctx.B.CreateBitOrPointerCast(s0, ctx.i64Ty);
    if (s1->getType() != ctx.i64Ty) s1 = ctx.B.CreateBitOrPointerCast(s1, ctx.i64Ty);
    ctx.regs.writeReg64(ctx.B, op.dst(), ctx.B.CreateAdd(s0, s1, "vadd64"));
    hr.handled = true;
    return hr;
  }

  // ---- v_mad_u64_u32 (2 defs: VDST + SDST, firstSrcIdx=2) ----
  if (sop == SemOp::V_MAD_U64_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src64(2), "vmad64");
    ctx.regs.writeReg64(ctx.B, op.dst(0), res);
    ctx.regs.writeReg64(ctx.B, op.dst(1), ConstantInt::get(ctx.i64Ty, 0));
    hr.handled = true;
    return hr;
  }

  // ---- Vector compares (VOPC e32 and VOP3 e64) ----
  // Helper lambda: map SemOp to ICmp/FCmp predicate for v_cmp_* instructions
  {
    auto vcmpICmpPred = [](SemOp s) -> std::optional<CmpInst::Predicate> {
      switch (s) {
      case SemOp::V_CMP_EQ_U32: case SemOp::V_CMP_EQ_I32:
      case SemOp::V_CMP_EQ_U64: return CmpInst::ICMP_EQ;
      case SemOp::V_CMP_NE_U32: case SemOp::V_CMP_NE_I32:
      case SemOp::V_CMP_NE_U64: return CmpInst::ICMP_NE;
      case SemOp::V_CMP_GT_I32: case SemOp::V_CMP_GT_I64: return CmpInst::ICMP_SGT;
      case SemOp::V_CMP_GE_I32: case SemOp::V_CMP_GE_I64: return CmpInst::ICMP_SGE;
      case SemOp::V_CMP_LT_I32: case SemOp::V_CMP_LT_I64: return CmpInst::ICMP_SLT;
      case SemOp::V_CMP_LE_I32: case SemOp::V_CMP_LE_I64: return CmpInst::ICMP_SLE;
      case SemOp::V_CMP_GT_U32: case SemOp::V_CMP_GT_U64: return CmpInst::ICMP_UGT;
      case SemOp::V_CMP_GE_U32: case SemOp::V_CMP_GE_U64: return CmpInst::ICMP_UGE;
      case SemOp::V_CMP_LT_U32: case SemOp::V_CMP_LT_U64: return CmpInst::ICMP_ULT;
      case SemOp::V_CMP_LE_U32: case SemOp::V_CMP_LE_U64: return CmpInst::ICMP_ULE;
      default: return std::nullopt;
      }
    };
    auto vcmpFCmpPred = [](SemOp s) -> std::optional<CmpInst::Predicate> {
      switch (s) {
      case SemOp::V_CMP_GT_F32: case SemOp::V_CMP_GT_F16: case SemOp::V_CMP_GT_F64: return CmpInst::FCMP_OGT;
      case SemOp::V_CMP_GE_F32: case SemOp::V_CMP_GE_F16: case SemOp::V_CMP_GE_F64: return CmpInst::FCMP_OGE;
      case SemOp::V_CMP_LT_F32: case SemOp::V_CMP_LT_F16: case SemOp::V_CMP_LT_F64: return CmpInst::FCMP_OLT;
      case SemOp::V_CMP_LE_F32: case SemOp::V_CMP_LE_F16: case SemOp::V_CMP_LE_F64: return CmpInst::FCMP_OLE;
      case SemOp::V_CMP_EQ_F32: case SemOp::V_CMP_EQ_F16: case SemOp::V_CMP_EQ_F64: return CmpInst::FCMP_OEQ;
      case SemOp::V_CMP_NE_F32: case SemOp::V_CMP_NE_F16: case SemOp::V_CMP_NE_F64:
      case SemOp::V_CMP_NEQ_F32: case SemOp::V_CMP_NEQ_F16: case SemOp::V_CMP_NEQ_F64:
      case SemOp::V_CMP_LG_F32: case SemOp::V_CMP_LG_F16: case SemOp::V_CMP_LG_F64: return CmpInst::FCMP_ONE;
      case SemOp::V_CMP_NLT_F32: case SemOp::V_CMP_NLT_F16: case SemOp::V_CMP_NLT_F64: return CmpInst::FCMP_UGE;
      case SemOp::V_CMP_NLE_F32: case SemOp::V_CMP_NLE_F16: case SemOp::V_CMP_NLE_F64: return CmpInst::FCMP_UGT;
      case SemOp::V_CMP_NGT_F32: case SemOp::V_CMP_NGT_F16: case SemOp::V_CMP_NGT_F64: return CmpInst::FCMP_ULE;
      case SemOp::V_CMP_NGE_F32: case SemOp::V_CMP_NGE_F16: case SemOp::V_CMP_NGE_F64: return CmpInst::FCMP_ULT;
      case SemOp::V_CMP_U_F32: case SemOp::V_CMP_U_F16: case SemOp::V_CMP_U_F64: return CmpInst::FCMP_UNO;
      case SemOp::V_CMP_O_F32: case SemOp::V_CMP_O_F16: case SemOp::V_CMP_O_F64: return CmpInst::FCMP_ORD;
      case SemOp::V_CMP_NLG_F32: return CmpInst::FCMP_UEQ;
      default: return std::nullopt;
      }
    };
    auto vcmpxICmpPred = [](SemOp s) -> std::optional<CmpInst::Predicate> {
      switch (s) {
      case SemOp::V_CMPX_EQ_U32: case SemOp::V_CMPX_EQ_I32: return CmpInst::ICMP_EQ;
      case SemOp::V_CMPX_NE_U32: case SemOp::V_CMPX_NE_I32: return CmpInst::ICMP_NE;
      case SemOp::V_CMPX_GT_I32: return CmpInst::ICMP_SGT;
      case SemOp::V_CMPX_GE_I32: return CmpInst::ICMP_SGE;
      case SemOp::V_CMPX_LT_I32: return CmpInst::ICMP_SLT;
      case SemOp::V_CMPX_LE_I32: return CmpInst::ICMP_SLE;
      case SemOp::V_CMPX_GT_U32: return CmpInst::ICMP_UGT;
      case SemOp::V_CMPX_GE_U32: return CmpInst::ICMP_UGE;
      case SemOp::V_CMPX_LT_U32: return CmpInst::ICMP_ULT;
      case SemOp::V_CMPX_LE_U32: return CmpInst::ICMP_ULE;
      default: return std::nullopt;
      }
    };
    auto vcmpxFCmpPred = [](SemOp s) -> std::optional<CmpInst::Predicate> {
      switch (s) {
      case SemOp::V_CMPX_GT_F32: case SemOp::V_CMPX_GT_F16: return CmpInst::FCMP_OGT;
      case SemOp::V_CMPX_GE_F32: case SemOp::V_CMPX_GE_F16: return CmpInst::FCMP_OGE;
      case SemOp::V_CMPX_LT_F32: case SemOp::V_CMPX_LT_F16: return CmpInst::FCMP_OLT;
      case SemOp::V_CMPX_LE_F32: case SemOp::V_CMPX_LE_F16: return CmpInst::FCMP_OLE;
      case SemOp::V_CMPX_EQ_F32: case SemOp::V_CMPX_EQ_F16: return CmpInst::FCMP_OEQ;
      case SemOp::V_CMPX_NE_F32: case SemOp::V_CMPX_NE_F16:
      case SemOp::V_CMPX_NEQ_F32: case SemOp::V_CMPX_NEQ_F16:
      case SemOp::V_CMPX_LG_F32: case SemOp::V_CMPX_LG_F16: return CmpInst::FCMP_ONE;
      case SemOp::V_CMPX_NLT_F32: return CmpInst::FCMP_UGE;
      case SemOp::V_CMPX_NLE_F32: return CmpInst::FCMP_UGT;
      case SemOp::V_CMPX_NGT_F32: return CmpInst::FCMP_ULE;
      case SemOp::V_CMPX_NGE_F32: return CmpInst::FCMP_ULT;
      default: return std::nullopt;
      }
    };

    // Determine if this is a 64-bit integer compare
    bool is64 = sop == SemOp::V_CMP_EQ_U64 || sop == SemOp::V_CMP_NE_U64 ||
                sop == SemOp::V_CMP_GT_U64 || sop == SemOp::V_CMP_GE_U64 ||
                sop == SemOp::V_CMP_LT_U64 || sop == SemOp::V_CMP_LE_U64 ||
                sop == SemOp::V_CMP_GT_I64 || sop == SemOp::V_CMP_GE_I64 ||
                sop == SemOp::V_CMP_LT_I64 || sop == SemOp::V_CMP_LE_I64;
    bool isF64 = sop == SemOp::V_CMP_EQ_F64 || sop == SemOp::V_CMP_NE_F64 ||
                 sop == SemOp::V_CMP_NEQ_F64 ||
                 sop == SemOp::V_CMP_GT_F64 || sop == SemOp::V_CMP_GE_F64 ||
                 sop == SemOp::V_CMP_LT_F64 || sop == SemOp::V_CMP_LE_F64 ||
                 sop == SemOp::V_CMP_LG_F64 ||
                 sop == SemOp::V_CMP_NLT_F64 || sop == SemOp::V_CMP_NLE_F64 ||
                 sop == SemOp::V_CMP_NGT_F64 || sop == SemOp::V_CMP_NGE_F64 ||
                 sop == SemOp::V_CMP_U_F64 || sop == SemOp::V_CMP_O_F64;

    // v_cmp_* integer
    if (auto pred = vcmpICmpPred(sop)) {
      Value *s0 = is64 ? op.src64(0) : op.src(0);
      Value *s1 = is64 ? op.src64(1) : op.src(1);
      if (!s0 || !s1) {
        llvm::errs() << "transpiler: " << mn << ": missing operand\n";
        result.failMnemonic = di.mnemonic;
        result.failFormat = "VALU";
        hr.handled = false;
        return hr;
      }
      Value *cmp = ctx.B.CreateICmp(*pred, s0, s1, "vcmp");
      if (di.numDefs >= 1) {
        ParsedReg d = op.dst();
        if (d.kind == ParsedReg::SGPR) {
          Value *mask = ctx.B.CreateSExt(cmp, ctx.regs.execTy);
          ctx.regs.writeRegExecWidth(ctx.B, d, mask);
        } else {
          ctx.regs.storeVCC(ctx.B, cmp);
        }
      } else {
        ctx.regs.storeVCC(ctx.B, cmp);
      }
      hr.handled = true;
    return hr;
    }
    // v_cmp_* float
    if (auto pred = vcmpFCmpPred(sop)) {
      Value *s0, *s1;
      if (isF64) {
        auto *f64Ty = Type::getDoubleTy(ctx.C);
        s0 = ctx.B.CreateBitCast(op.src64(0), f64Ty);
        s1 = ctx.B.CreateBitCast(op.src64(1), f64Ty);
      } else {
        s0 = op.srcF(0); s1 = op.srcF(1);
        bool isF32 = sop >= SemOp::V_CMP_EQ_F32 && sop <= SemOp::V_CMP_O_F32;
        if (isF32) {
          if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
          if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
        }
      }
      Value *cmp = ctx.B.CreateFCmp(*pred, s0, s1, "vcmpf");
      if (di.numDefs >= 1) {
        ParsedReg d = op.dst();
        if (d.kind == ParsedReg::SGPR) {
          Value *mask = ctx.B.CreateSExt(cmp, ctx.regs.execTy);
          ctx.regs.writeRegExecWidth(ctx.B, d, mask);
        } else {
          ctx.regs.storeVCC(ctx.B, cmp);
        }
      } else {
        ctx.regs.storeVCC(ctx.B, cmp);
      }
      hr.handled = true;
    return hr;
    }

    // v_cmpx_* integer: compare and write result to EXEC mask
    if (auto pred = vcmpxICmpPred(sop)) {
      Value *s0 = op.src(0), *s1 = op.src(1);
      if (!s0 || !s1) { llvm::errs() << "transpiler: " << mn << ": missing operand\n"; result.failMnemonic = di.mnemonic;
        result.failFormat = "VALU";
        hr.handled = false;
        return hr; }
      Value *cmp = ctx.B.CreateICmp(*pred, s0, s1, "vcmpx");
      Value *mask = ctx.B.CreateSExt(cmp, ctx.regs.execTy);
      Value *curExec = ctx.regs.loadExec(ctx.B);
      ctx.regs.storeExec(ctx.B, ctx.B.CreateAnd(curExec, mask, "cmpx_exec"));
      hr.handled = true;
    return hr;
    }
    // v_cmpx_* float
    if (auto pred = vcmpxFCmpPred(sop)) {
      Value *s0 = op.srcF(0), *s1 = op.srcF(1);
      bool isF32 = sop >= SemOp::V_CMPX_EQ_F32 && sop <= SemOp::V_CMPX_NGE_F32;
      if (isF32) {
        if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
        if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      }
      Value *cmp = ctx.B.CreateFCmp(*pred, s0, s1, "vcmpxf");
      Value *mask = ctx.B.CreateSExt(cmp, ctx.regs.execTy);
      Value *curExec = ctx.regs.loadExec(ctx.B);
      ctx.regs.storeExec(ctx.B, ctx.B.CreateAnd(curExec, mask, "cmpx_exec"));
      hr.handled = true;
    return hr;
    }
  }

  // ---- VOP3P packed ops (2x fp32 in 2 dwords) ----
  // Handle op_sel_hi, neg_lo, neg_hi modifiers.
  if (sop >= SemOp::V_PK_ADD_F32 && sop <= SemOp::V_PK_MOV_B32) {
    if (sop == SemOp::V_PK_MOV_B32) {
      ctx.regs.writeReg64(ctx.B, op.dst(), op.src64(0));
      hr.handled = true;
    return hr;
    }

    auto *v2f32 = FixedVectorType::get(ctx.f32Ty, 2);

    // Parse VOP3P modifiers from disassembly text.
    int opSelHi[3] = {1, 1, 1};  // default: high lane reads high element
    int negLo[3] = {0, 0, 0};
    int negHi[3] = {0, 0, 0};
    StringRef text(di.fullText);

    auto parseBracketList3 = [](StringRef text, StringRef key, int out[3]) {
      auto pos = text.find(key);
      if (pos == StringRef::npos) return;
      auto brk = text.find('[', pos);
      if (brk == StringRef::npos) return;
      auto end = text.find(']', brk);
      if (end == StringRef::npos) return;
      StringRef inner = text.slice(brk + 1, end);
      SmallVector<StringRef, 3> parts;
      inner.split(parts, ',');
      for (unsigned i = 0; i < parts.size() && i < 3; i++) {
        int val = 0;
        if (!parts[i].trim().getAsInteger(10, val))
          out[i] = val;
      }
    };

    parseBracketList3(text, "op_sel_hi:", opSelHi);
    parseBracketList3(text, "neg_lo:", negLo);
    parseBracketList3(text, "neg_hi:", negHi);

    // Read each source as <2 x f32>, apply element selection and negation.
    // readPkSrc: returns a <2 x f32> with modifiers applied.
    auto readPkSrc = [&](unsigned i) -> Value * {
      if (!op.isSrcReg(i)) {
        llvm::errs() << "transpiler: " << mn << ": non-register source "
               << i << " (immediate in VOP3P not supported)\n";
        return nullptr;
      }
      Value *vec = ctx.regs.readRegVec(ctx.B, op.srcReg(i), v2f32);
      Value *lo = ctx.B.CreateExtractElement(vec, (uint64_t)0);
      Value *hi = ctx.B.CreateExtractElement(vec, (uint64_t)1);
      // op_sel_hi: if 0, high lane uses low element (broadcast)
      if (opSelHi[i] == 0)
        hi = lo;
      // Apply negation
      if (negLo[i])
        lo = ctx.B.CreateFNeg(lo);
      if (negHi[i])
        hi = ctx.B.CreateFNeg(hi);
      Value *r = UndefValue::get(v2f32);
      r = ctx.B.CreateInsertElement(r, lo, (uint64_t)0);
      r = ctx.B.CreateInsertElement(r, hi, (uint64_t)1);
      return r;
    };

    Value *s0 = readPkSrc(0);
    Value *s1 = readPkSrc(1);
    if (!s0 || !s1) {
      result.failMnemonic = di.mnemonic;
        result.failFormat = "VALU";
        hr.handled = false;
        return hr;
    }

    Value *res = nullptr;
    if (sop == SemOp::V_PK_ADD_F32) {
      res = ctx.B.CreateFAdd(s0, s1, "pk_add");
    } else if (sop == SemOp::V_PK_MUL_F32) {
      res = ctx.B.CreateFMul(s0, s1, "pk_mul");
    } else if (sop == SemOp::V_PK_MAX_F32) {
      Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_max");
    } else if (sop == SemOp::V_PK_MIN_F32) {
      Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_min");
    } else if (sop == SemOp::V_PK_FMA_F32) {
      Value *s2 = readPkSrc(2);
      if (!s2) {
        result.failMnemonic = di.mnemonic;
        result.failFormat = "VALU";
        hr.handled = false;
        return hr;
      }
      Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1, s2}, "pk_fma");
    }
    ctx.regs.writeRegVec(ctx.B, op.dst(), res);
    hr.handled = true;
    return hr;
  }

  // ---- WMMA (gfx1250 RDNA4, VOP3P encoding) ----
  // v_wmma_f32_16x16x32_f16: A and B are v16f16, C/D are v8f32
  if (sop == SemOp::V_WMMA_F32_16x16x32_F16) {
    Type *v16f16Ty = FixedVectorType::get(Type::getHalfTy(ctx.C), 16);
    Type *v8f32Ty = FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
    ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, v16f16Ty);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, v16f16Ty);
    Value *c = ctx.regs.readRegVec(ctx.B, srcC, v8f32Ty);

    Value *result_val;
    if (ctx.targetIsa.hasWMMA12) {
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_wmma_f32_16x16x32_f16,
          {v8f32Ty, v16f16Ty});
      result_val = ctx.B.CreateCall(wmmaFn, {
          ctx.B.getFalse(), a,
          ctx.B.getFalse(), b,
          ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
          ctx.B.getFalse(), ctx.B.getFalse()
      }, "wmma");
    } else {
      result_val = emitWMMAtoMFMA(ctx, a, b, c);
    }

    ctx.regs.writeRegVec(ctx.B, dest, result_val);
    hr.handled = true;
    return hr;
  }

  // ---- v_fma_mix_f32: mixed-precision FMA (VOP3P) ----
  // dst = fma(cvt_f32(src0_part), cvt_f32(src1_part), cvt_f32(src2_part))
  // op_sel_hi[i]==1 → source i is f16 (lo/hi selected by op_sel[i])
  // op_sel_hi[i]==0 → source i is full f32
  if (sop == SemOp::V_FMA_MIX_F32) {
    int opSel[3] = {0, 0, 0};
    int opSelHi[3] = {0, 0, 0};
    StringRef text(di.fullText);

    auto parseBracketList = [](StringRef text, StringRef key, int out[3]) {
      auto pos = text.find(key);
      if (pos == StringRef::npos) return;
      auto brk = text.find('[', pos);
      if (brk == StringRef::npos) return;
      auto end = text.find(']', brk);
      if (end == StringRef::npos) return;
      StringRef inner = text.slice(brk + 1, end);
      SmallVector<StringRef, 3> parts;
      inner.split(parts, ',');
      for (unsigned i = 0; i < parts.size() && i < 3; i++) {
        int val = 0;
        if (!parts[i].trim().getAsInteger(10, val))
          out[i] = val;
      }
    };

    parseBracketList(text, "op_sel:", opSel);
    parseBracketList(text, "op_sel_hi:", opSelHi);

    auto readMixSrc = [&](unsigned i) -> Value * {
      Value *raw = op.srcF(i);
      if (opSelHi[i] == 0) {
        if (raw->getType() != ctx.f32Ty) raw = ctx.B.CreateBitCast(raw, ctx.f32Ty);
        return raw;
      }
      if (raw->getType() == ctx.f32Ty) raw = ctx.B.CreateBitCast(raw, ctx.i32Ty);
      Value *bits;
      if (opSel[i] == 0)
        bits = ctx.B.CreateTrunc(raw, Type::getInt16Ty(ctx.C));
      else
        bits = ctx.B.CreateTrunc(ctx.B.CreateLShr(raw, 16), Type::getInt16Ty(ctx.C));
      Value *f16Val = ctx.B.CreateBitCast(bits, ctx.f16Ty);
      return ctx.B.CreateFPExt(f16Val, ctx.f32Ty, "mix_cvt");
    };

    Value *s0 = readMixSrc(0);
    Value *s1 = readMixSrc(1);
    Value *s2 = readMixSrc(2);
    Function *fmaFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.regs.writeReg32(ctx.B, op.dst(), ctx.B.CreateBitCast(
        ctx.B.CreateCall(fmaFn, {s0, s1, s2}, "fma_mix"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- v_cndmask_b32 (VOP2 or VOP3 — srcMap skips modifiers) ----
  if (sop == SemOp::V_CNDMASK_B32) {
    ParsedReg dest = op.dst();
    Value *src0 = op.src(0);
    Value *src1 = op.src(1);
    Value *cond = nullptr;
    if (op.nSrcs() >= 3 && di.isReg(op.srcIdx(2))) {
      ParsedReg condReg = ctx.parseReg(di.getReg(op.srcIdx(2)), op.srcIdx(2));
      if (condReg.kind == ParsedReg::SGPR) {
        Value *condVal = ctx.isa.isWave32() ? (Value *)ctx.regs.loadSGPR32(ctx.B, condReg.baseIdx)
                                        : (Value *)ctx.regs.loadSGPR64(ctx.B, condReg.baseIdx);
        cond = ctx.B.CreateICmpNE(condVal, Constant::getNullValue(condVal->getType()));
      }
      else
        cond = ctx.regs.loadVCC(ctx.B);
    }
    if (!cond) cond = ctx.regs.loadVCC(ctx.B);
    ctx.regs.writeReg32(ctx.B, dest, ctx.B.CreateSelect(cond, src1, src0, "cndmask"));
    hr.handled = true;
    return hr;
  }

  return hr;
}

} // namespace transpiler
