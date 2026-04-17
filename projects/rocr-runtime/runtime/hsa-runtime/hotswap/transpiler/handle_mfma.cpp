#include "handlers.hpp"
#include "raiser.hpp"

#include "semop.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace llvm;

namespace transpiler {
HandlerResult handleMFMA(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op, RaiseResult &result) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  // AGPR move instructions are classified as MFMA format
  if (sop == SemOp::V_ACCVGPR_WRITE_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ACCVGPR_READ_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }

  struct MfmaInfo {
    Intrinsic::ID id;
    Type *srcTy;
    Type *accumTy;
    unsigned srcDwords;
    unsigned accumDwords;
  };

  auto *v4f16Ty  = FixedVectorType::get(Type::getHalfTy(ctx.C), 4);
  auto *v4f32Ty  = FixedVectorType::get(ctx.f32Ty, 4);
  auto *v16f32Ty = FixedVectorType::get(ctx.f32Ty, 16);
  auto *v32f32Ty = FixedVectorType::get(ctx.f32Ty, 32);
  auto *v4i32Ty  = FixedVectorType::get(ctx.i32Ty, 4);
  auto *v16i32Ty = FixedVectorType::get(ctx.i32Ty, 16);
  auto *v32i32Ty = FixedVectorType::get(ctx.i32Ty, 32);
  auto *v2f32Ty  = FixedVectorType::get(ctx.f32Ty, 2);
  auto *v4i16Ty  = FixedVectorType::get(Type::getInt16Ty(ctx.C), 4);
  auto *v8i32Ty  = FixedVectorType::get(ctx.i32Ty, 8);

  // gfx950 scaled MFMA: v_mfma_[scale_]f32_{16x16x128,32x32x64}_f8f6f4
  // These use the llvm.amdgcn.mfma.scale intrinsic with extra scale params.
  // The non-scale version uses scale=0, op_sel=0 (identity scaling).
  if (sop == SemOp::V_MFMA_F32_16x16x128_F8F6F4 || sop == SemOp::V_MFMA_SCALE_F32_16x16x128_F8F6F4 ||
      sop == SemOp::V_MFMA_F32_32x32x64_F8F6F4 || sop == SemOp::V_MFMA_SCALE_F32_32x32x64_F8F6F4) {
    bool is16x16 = sop == SemOp::V_MFMA_F32_16x16x128_F8F6F4 || sop == SemOp::V_MFMA_SCALE_F32_16x16x128_F8F6F4;
    bool isScale = sop == SemOp::V_MFMA_SCALE_F32_16x16x128_F8F6F4 || sop == SemOp::V_MFMA_SCALE_F32_32x32x64_F8F6F4;
    Type *accumTy = is16x16 ? (Type*)v4f32Ty : (Type*)v16f32Ty;
    Type *srcTy = v8i32Ty;
    Intrinsic::ID intrId = is16x16
        ? Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4
        : Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4;

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
    ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;
    Value *a = ctx.regs.readRegVec(ctx.B, srcA, srcTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, srcTy);
    Value *c = ctx.regs.readRegVec(ctx.B, srcC, accumTy);

    // Read cbsz and blgp from source immediates
    int cbsz = 0, blgp = 0;
    for (unsigned k = 3; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        if (cbsz == 0) cbsz = (int)di.getImm(op.srcIdx(k));
        else { blgp = (int)di.getImm(op.srcIdx(k)); break; }
      }
    }

    Value *scaleA = ConstantInt::get(ctx.i32Ty, 0);
    Value *scaleB = ConstantInt::get(ctx.i32Ty, 0);
    int opSelA = 0, opSelB = 0;
    if (isScale) {
      // Scale versions have additional operands for scale registers and op_sel
      // These come from v_mfma_ld_scale_b32, encoded in extra src operands.
      // For now, read from source registers if available.
      unsigned scaleIdx = 3;
      for (unsigned k = 3; k < op.nSrcs(); k++) {
        if (!di.isImm(op.srcIdx(k)) && op.isSrcReg(k)) {
          if (scaleIdx == 3) { scaleA = ctx.regs.readReg32(ctx.B, op.srcReg(k)); scaleIdx++; }
          else { scaleB = ctx.regs.readReg32(ctx.B, op.srcReg(k)); break; }
        }
      }
    }

    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, intrId, {srcTy, srcTy});
    Value *result_val = ctx.B.CreateCall(fn, {
        a, b, c,
        ConstantInt::get(ctx.i32Ty, cbsz), ConstantInt::get(ctx.i32Ty, blgp),
        ConstantInt::get(ctx.i32Ty, opSelA), scaleA,
        ConstantInt::get(ctx.i32Ty, opSelB), scaleB
    }, "mfma_scale");
    ctx.regs.writeRegVec(ctx.B, dest, result_val);
    hr.handled = true;
    return hr;
  }

  auto *v8bf16Ty = FixedVectorType::get(Type::getBFloatTy(ctx.C), 8);
  auto *v8f16Ty  = FixedVectorType::get(Type::getHalfTy(ctx.C), 8);

  // MFMA dispatch is keyed by SemOp: every encoding variant
  // (e32/e64/_vgprcd_/_mac_/subtarget-specific Real) that LLVM emits for the
  // same semantic MFMA shape folds onto a single SemOp in kCanonTable, so the
  // handler only needs one row per shape.
  DenseMap<SemOp, MfmaInfo> mfmaTable = {
    {SemOp::V_MFMA_F32_16x16x16_F16, {Intrinsic::amdgcn_mfma_f32_16x16x16f16,   v4f16Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_F32_32x32x8_F16,  {Intrinsic::amdgcn_mfma_f32_32x32x8f16,    v4f16Ty, v16f32Ty, 2, 16}},
    {SemOp::V_MFMA_F32_16x16x4_F32,  {Intrinsic::amdgcn_mfma_f32_16x16x4f32,    ctx.f32Ty, v16f32Ty, 1, 16}},
    {SemOp::V_MFMA_F32_32x32x1_F32,  {Intrinsic::amdgcn_mfma_f32_32x32x1f32,    ctx.f32Ty, v32f32Ty, 1, 32}},
    {SemOp::V_MFMA_F32_32x32x2_F32,  {Intrinsic::amdgcn_mfma_f32_32x32x2f32,    ctx.f32Ty, v16f32Ty, 1, 16}},
    {SemOp::V_MFMA_F32_4x4x1_F32,    {Intrinsic::amdgcn_mfma_f32_4x4x1f32,      ctx.f32Ty, v4f32Ty,  1, 4}},
    {SemOp::V_MFMA_F32_16x16x1_F32,  {Intrinsic::amdgcn_mfma_f32_16x16x1f32,    ctx.f32Ty, v16f32Ty, 1, 16}},
    {SemOp::V_MFMA_F32_32x32x4_F16,  {Intrinsic::amdgcn_mfma_f32_32x32x4f16,    v4f16Ty, v32f32Ty, 2, 32}},
    {SemOp::V_MFMA_F32_16x16x4_F16,  {Intrinsic::amdgcn_mfma_f32_16x16x4f16,    v4f16Ty, v16f32Ty, 2, 16}},
    {SemOp::V_MFMA_F32_4x4x4_F16,    {Intrinsic::amdgcn_mfma_f32_4x4x4f16,      v4f16Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_I32_16x16x32_I8,  {Intrinsic::amdgcn_mfma_i32_16x16x32_i8,   ctx.i64Ty, v4i32Ty,  2, 4}},
    {SemOp::V_MFMA_I32_32x32x16_I8,  {Intrinsic::amdgcn_mfma_i32_32x32x16_i8,   ctx.i64Ty, v16i32Ty, 2, 16}},
    {SemOp::V_MFMA_F32_16x16x8_XF32, {Intrinsic::amdgcn_mfma_f32_16x16x8_xf32,  v2f32Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_F32_32x32x4_XF32, {Intrinsic::amdgcn_mfma_f32_32x32x4_xf32,  v2f32Ty, v16f32Ty, 2, 16}},
    {SemOp::V_MFMA_I32_32x32x4_I8,   {Intrinsic::amdgcn_mfma_i32_32x32x4i8,     ctx.i32Ty, v32i32Ty, 1, 32}},
    {SemOp::V_MFMA_I32_16x16x4_I8,   {Intrinsic::amdgcn_mfma_i32_16x16x4i8,     ctx.i32Ty, v16i32Ty, 1, 16}},
    {SemOp::V_MFMA_I32_4x4x4_I8,     {Intrinsic::amdgcn_mfma_i32_4x4x4i8,       ctx.i32Ty, v4i32Ty,  1, 4}},
    {SemOp::V_MFMA_F32_32x32x2_BF16, {Intrinsic::amdgcn_mfma_f32_32x32x2bf16,   v4i16Ty, v32f32Ty, 2, 32}},
    {SemOp::V_MFMA_F32_16x16x2_BF16, {Intrinsic::amdgcn_mfma_f32_16x16x2bf16,   v4i16Ty, v16f32Ty, 2, 16}},
    {SemOp::V_MFMA_F32_4x4x2_BF16,   {Intrinsic::amdgcn_mfma_f32_4x4x2bf16,     v4i16Ty, v4f32Ty,  2, 4}},
    // gfx90a "1K" BF16 shapes — shared v4i16 (=2 dwords) src layout.
    {SemOp::V_MFMA_F32_16x16x16_BF16_1K, {Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k, v4i16Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_F32_32x32x8_BF16_1K,  {Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k,  v4i16Ty, v16f32Ty, 2, 16}},
    // gfx950 wider BF16/F16 shapes — v8bf16 / v8f16 (=4 dwords) src layout.
    {SemOp::V_MFMA_F32_16x16x32_BF16, {Intrinsic::amdgcn_mfma_f32_16x16x32_bf16, v8bf16Ty, v4f32Ty,  4, 4}},
    {SemOp::V_MFMA_F32_32x32x16_BF16, {Intrinsic::amdgcn_mfma_f32_32x32x16_bf16, v8bf16Ty, v16f32Ty, 4, 16}},
    {SemOp::V_MFMA_F32_16x16x32_F16,  {Intrinsic::amdgcn_mfma_f32_16x16x32_f16,  v8f16Ty,  v4f32Ty,  4, 4}},
    // gfx940 FP8/BF8 shapes — i64 (=2 dwords) src layout carries 8 packed
    // 8-bit floats.
    {SemOp::V_MFMA_F32_16x16x32_FP8_FP8, {Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8, ctx.i64Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_F32_16x16x32_FP8_BF8, {Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8, ctx.i64Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_F32_16x16x32_BF8_FP8, {Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8, ctx.i64Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_F32_16x16x32_BF8_BF8, {Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8, ctx.i64Ty, v4f32Ty,  2, 4}},
    {SemOp::V_MFMA_F32_32x32x16_FP8_FP8, {Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8, ctx.i64Ty, v16f32Ty, 2, 16}},
    {SemOp::V_MFMA_F32_32x32x16_FP8_BF8, {Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8, ctx.i64Ty, v16f32Ty, 2, 16}},
    {SemOp::V_MFMA_F32_32x32x16_BF8_FP8, {Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8, ctx.i64Ty, v16f32Ty, 2, 16}},
    {SemOp::V_MFMA_F32_32x32x16_BF8_BF8, {Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8, ctx.i64Ty, v16f32Ty, 2, 16}},
  };

  auto it = mfmaTable.find(sop);
  if (it == mfmaTable.end()) {
    result.failMnemonic = di.mnemonic;
    result.failFormat = "MFMA";
    llvm::errs() << "transpiler: Unknown MFMA: " << di.mnemonic << "\n";
    hr.handled = false;
    return hr;
  }

  auto &info = it->second;
  ParsedReg dest = op.dst();
  ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
  // The accumulator (src2) may be tied to the destination in some encodings.
  ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;
  if (srcA.kind == ParsedReg::OTHER || srcB.kind == ParsedReg::OTHER) {
    result.failMnemonic = di.mnemonic; result.failFormat = "MFMA";
    llvm::errs() << "transpiler: MFMA " << mn << ": cannot read source registers\n";
    result.failMnemonic = di.mnemonic;
        result.failFormat = "MFMA";
        hr.handled = false;
        return hr;
  }
  Value *a = ctx.regs.readRegVec(ctx.B, srcA, info.srcTy);
  Value *b = ctx.regs.readRegVec(ctx.B, srcB, info.srcTy);
  Value *c = ctx.regs.readRegVec(ctx.B, srcC, info.accumTy);

  int cbsz = 0, abid = 0, blgp = 0;
  unsigned immIdx = 0;
  for (unsigned k = 3; k < op.nSrcs(); k++) {
    if (di.isImm(op.srcIdx(k))) {
      int64_t v = di.getImm(op.srcIdx(k));
      if (immIdx == 0) cbsz = v;
      else if (immIdx == 1) abid = v;
      else if (immIdx == 2) blgp = v;
      immIdx++;
    }
  }

  Function *mfmaFn = Intrinsic::getOrInsertDeclaration(&ctx.M, info.id);
  ctx.regs.writeRegVec(ctx.B, dest, ctx.B.CreateCall(mfmaFn, {
      a, b, c,
      ConstantInt::get(ctx.i32Ty, cbsz),
      ConstantInt::get(ctx.i32Ty, abid),
      ConstantInt::get(ctx.i32Ty, blgp)
  }, "mfma"));
  hr.handled = true;
  return hr;
}

} // namespace transpiler
