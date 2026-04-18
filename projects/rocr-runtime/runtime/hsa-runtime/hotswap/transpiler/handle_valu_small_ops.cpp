#include "handle_valu_internal.hpp"

#include "semop.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace transpiler {

// "Small ops": conversions (F32↔{U,I}32, F16↔F32, F16↔{U,I}16, byte
// extract), F16 two-src arith (add/sub/mul/min/max/mac/fmac), packed
// F16 fmac, 16-bit min/max and reverse-operand shifts, byte pack,
// V_BFREV_B32 / V_NOT_B32, and F32 single-src transcendentals
// (rcp/exp/log/ldexp/sqrt/rsq/floor/ceil/trunc/fract).
//
// Grouped here because each case is 1-5 lines of IR emission and they
// would bloat the arithmetic / 3-src sub-handlers if interleaved.
// Structured as a switch on SemOp: cases are mutually exclusive and
// ordering is not load-bearing.
HandlerResult handleVALU_SmallOps(RaiseContext &ctx, const DecodedInst &di,
                                   OpResolver &op) {
  HandlerResult hr;
  Type *i16Ty = Type::getInt16Ty(ctx.C);
  Type *halfTy = Type::getHalfTy(ctx.C);

  switch (di.semOp) {
  // ---- F32 <-> integer conversions ----
  case SemOp::V_CVT_F32_U32: {
    Value *r = ctx.B.CreateUIToFP(op.src(0), ctx.f32Ty, "cvt");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F32_I32: {
    Value *r = ctx.B.CreateSIToFP(op.src(0), ctx.f32Ty, "cvt");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_U32_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateFPToUI(s, ctx.i32Ty, "cvt"));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_I32_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateFPToSI(s, ctx.i32Ty, "cvt"));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F16_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *h = ctx.B.CreateFPTrunc(s, halfTy, "cvt");
    Value *bits = ctx.B.CreateBitCast(h, i16Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(bits, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F32_F16: {
    Value *bits = ctx.B.CreateTrunc(op.src(0), i16Ty);
    Value *h = ctx.B.CreateBitCast(bits, halfTy);
    Value *f = ctx.B.CreateFPExt(h, ctx.f32Ty, "cvt");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(f, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F16_U16: {
    Value *s = ctx.B.CreateTrunc(op.src(0), i16Ty);
    Value *res = ctx.B.CreateUIToFP(s, ctx.f16Ty, "cvt_f16_u16");
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateZExt(ctx.B.CreateBitCast(res, i16Ty),
                                    ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_U16_F16: {
    Value *s = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty),
                                    ctx.f16Ty);
    Value *res = ctx.B.CreateFPToUI(s, i16Ty, "cvt_u16_f16");
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F32_UBYTE0: {
    Value *byte = ctx.B.CreateAnd(op.src(0),
                                   ConstantInt::get(ctx.i32Ty, 0xFF));
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F32_UBYTE1: {
    Value *byte = ctx.B.CreateAnd(ctx.B.CreateLShr(op.src(0), 8),
                                   ConstantInt::get(ctx.i32Ty, 0xFF));
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F32_UBYTE2: {
    Value *byte = ctx.B.CreateAnd(ctx.B.CreateLShr(op.src(0), 16),
                                   ConstantInt::get(ctx.i32Ty, 0xFF));
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CVT_F32_UBYTE3: {
    Value *byte = ctx.B.CreateLShr(op.src(0), 24);
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateUIToFP(byte, ctx.f32Ty), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- F16 two-src arith (reused i16 bitcast idiom) ----
  case SemOp::V_MUL_F16:
  case SemOp::V_ADD_F16:
  case SemOp::V_SUB_F16:
  case SemOp::V_SUBREV_F16:
  case SemOp::V_MAX_F16:
  case SemOp::V_MIN_F16: {
    Value *a = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty),
                                    ctx.f16Ty);
    Value *b = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty),
                                    ctx.f16Ty);
    Value *res = nullptr;
    switch (di.semOp) {
    case SemOp::V_MUL_F16:    res = ctx.B.CreateFMul(a, b, "mul_f16"); break;
    case SemOp::V_ADD_F16:    res = ctx.B.CreateFAdd(a, b, "add_f16"); break;
    case SemOp::V_SUB_F16:    res = ctx.B.CreateFSub(a, b, "sub_f16"); break;
    case SemOp::V_SUBREV_F16: res = ctx.B.CreateFSub(b, a, "subrev_f16"); break;
    case SemOp::V_MAX_F16: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::maxnum, {ctx.f16Ty});
      res = ctx.B.CreateCall(fn, {a, b}, "max_f16");
      break;
    }
    case SemOp::V_MIN_F16: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::minnum, {ctx.f16Ty});
      res = ctx.B.CreateCall(fn, {a, b}, "min_f16");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateZExt(ctx.B.CreateBitCast(res, i16Ty),
                                    ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- Packed <2xf16> FMA into dst ----
  case SemOp::V_PK_FMAC_F16: {
    auto *v2f16 = FixedVectorType::get(ctx.f16Ty, 2);
    Value *s0 = ctx.B.CreateBitCast(op.src(0), v2f16);
    Value *s1 = ctx.B.CreateBitCast(op.src(1), v2f16);
    Value *acc = ctx.B.CreateBitCast(ctx.regs.readReg32(ctx.B, op.dst()),
                                      v2f16);
    Function *fma = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fma, {v2f16});
    Value *res = ctx.B.CreateCall(fma, {s0, s1, acc}, "pk_fmac");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- F16 MAC: dst = src0 * src1 + dst ----
  case SemOp::V_MAC_F16:
  case SemOp::V_FMAC_F16: {
    Value *s0 = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty),
                                     ctx.f16Ty);
    Value *s1 = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(1), i16Ty),
                                     ctx.f16Ty);
    Value *acc = ctx.B.CreateBitCast(
        ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, op.dst()), i16Ty),
        ctx.f16Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fma, {ctx.f16Ty});
    Value *res = ctx.B.CreateBitCast(
        ctx.B.CreateCall(fma, {s0, s1, acc}, "mac_f16"), i16Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  case SemOp::V_FLOOR_F16: {
    Value *s = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty),
                                    ctx.f16Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::floor, {ctx.f16Ty});
    Value *res = ctx.B.CreateBitCast(
        ctx.B.CreateCall(fn, {s}, "floor_f16"), i16Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_LDEXP_F16: {
    Value *s0 = ctx.B.CreateBitCast(ctx.B.CreateTrunc(op.srcF(0), i16Ty),
                                     ctx.f16Ty);
    Value *s1 = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Function *ldexpFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::ldexp, {ctx.f16Ty, i16Ty});
    Value *res = ctx.B.CreateBitCast(
        ctx.B.CreateCall(ldexpFn, {s0, s1}, "ldexp_f16"), i16Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- 16-bit integer min/max ----
  case SemOp::V_MAX_U16:
  case SemOp::V_MIN_U16:
  case SemOp::V_MAX_I16:
  case SemOp::V_MIN_I16: {
    Value *a = ctx.B.CreateTrunc(op.src(0), i16Ty);
    Value *b = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Value *cmp = nullptr;
    switch (di.semOp) {
    case SemOp::V_MAX_U16: cmp = ctx.B.CreateICmpUGT(a, b); break;
    case SemOp::V_MIN_U16: cmp = ctx.B.CreateICmpULT(a, b); break;
    case SemOp::V_MAX_I16: cmp = ctx.B.CreateICmpSGT(a, b); break;
    case SemOp::V_MIN_I16: cmp = ctx.B.CreateICmpSLT(a, b); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    Value *res = ctx.B.CreateSelect(cmp, a, b, "i16sel");
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- 16-bit reverse-operand shifts (HW uses src0[3:0]) ----
  case SemOp::V_ASHRREV_I16:
  case SemOp::V_LSHRREV_B16:
  case SemOp::V_LSHLREV_B16: {
    Value *shamt = ctx.B.CreateAnd(ctx.B.CreateTrunc(op.src(0), i16Ty),
                                    ConstantInt::get(i16Ty, 0xF));
    Value *base = ctx.B.CreateTrunc(op.src(1), i16Ty);
    Value *res = nullptr;
    switch (di.semOp) {
    case SemOp::V_ASHRREV_I16: res = ctx.B.CreateAShr(base, shamt, "vashr16"); break;
    case SemOp::V_LSHRREV_B16: res = ctx.B.CreateLShr(base, shamt, "vlshr16"); break;
    case SemOp::V_LSHLREV_B16: res = ctx.B.CreateShl(base, shamt, "vlshl16"); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(res, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  case SemOp::V_PACK_B32_F16: {
    Value *lo = ctx.B.CreateAnd(op.src(0),
                                 ConstantInt::get(ctx.i32Ty, 0xFFFF));
    Value *hi = ctx.B.CreateShl(
        ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFF)), 16);
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(lo, hi, "pack_f16"));
    hr.handled = true;
    return hr;
  }

  // ---- Simple bit-twiddle single-src ----
  case SemOp::V_BFREV_B32: {
    Function *brev = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::bitreverse, {ctx.i32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateCall(brev, {op.src(0)}, "bfrev"));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_NOT_B32: {
    ctx.writeReg32(op.dst(), ctx.B.CreateNot(op.src(0), "vnot"));
    hr.handled = true;
    return hr;
  }

  // ---- F32 single-src transcendentals / rounding ----
  case SemOp::V_RCP_IFLAG_F32:
  case SemOp::V_RCP_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *r = ctx.B.CreateFDiv(ConstantFP::get(ctx.f32Ty, 1.0), s, "rcp");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_EXP_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *exp2Fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::exp2, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(exp2Fn, {s}, "exp"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_LOG_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *log2Fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::log2, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(log2Fn, {s}, "log"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_LDEXP_F32: {
    Value *s0 = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *s1 = op.src(1);
    Function *ldexpFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::ldexp, {ctx.f32Ty, ctx.i32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(ldexpFn, {s0, s1}, "ldexp"),
                       ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_SQRT_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *sqrtFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::sqrt, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(sqrtFn, {s}, "sqrt"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_RSQ_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *sqrtFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::sqrt, {ctx.f32Ty});
    Value *sq = ctx.B.CreateCall(sqrtFn, {s}, "sqrt");
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateFDiv(ConstantFP::get(ctx.f32Ty, 1.0),
                                        sq, "rsq"),
                       ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_FLOOR_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *floorFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::floor, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(floorFn, {s}, "floor"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_CEIL_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *ceilFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::ceil, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(ceilFn, {s}, "ceil"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_TRUNC_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *truncFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::trunc, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(truncFn, {s}, "trunc"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_FRACT_F32: {
    Value *s = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Function *floorFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::floor, {ctx.f32Ty});
    Value *fl = ctx.B.CreateCall(floorFn, {s}, "floor");
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateFSub(s, fl, "fract"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  default:
    break;
  }
  return hr;
}

} // namespace transpiler
