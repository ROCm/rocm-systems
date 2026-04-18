#include "handlers.hpp"

#include "amdgpu_formats.hpp" // SIInstrFlags
#include "opcode_map.hpp"
#include "semop.hpp"
#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

// =========================================================================
// SemOp -> Intrinsic::ID is the one piece of MFMA metadata LLVM does not
// expose as a public MC-level table. The reverse direction (intrinsic ->
// selected instruction) is encoded in the selector patterns but not
// published. Everything else a handler needs -- source element types,
// accumulator vector type, argument count for scaled vs. non-scaled --
// is recoverable from the intrinsic's declared signature via
// `Intrinsic::getType`, so this map carries ID alone.
// =========================================================================
static const DenseMap<SemOp, Intrinsic::ID> &mfmaIntrinsicTable() {
  static const auto *table = new DenseMap<SemOp, Intrinsic::ID>({
      {SemOp::V_MFMA_F32_16x16x16_F16,     Intrinsic::amdgcn_mfma_f32_16x16x16f16},
      {SemOp::V_MFMA_F32_32x32x8_F16,      Intrinsic::amdgcn_mfma_f32_32x32x8f16},
      {SemOp::V_MFMA_F32_16x16x4_F32,      Intrinsic::amdgcn_mfma_f32_16x16x4f32},
      {SemOp::V_MFMA_F32_32x32x1_F32,      Intrinsic::amdgcn_mfma_f32_32x32x1f32},
      {SemOp::V_MFMA_F32_32x32x2_F32,      Intrinsic::amdgcn_mfma_f32_32x32x2f32},
      {SemOp::V_MFMA_F32_4x4x1_F32,        Intrinsic::amdgcn_mfma_f32_4x4x1f32},
      {SemOp::V_MFMA_F32_16x16x1_F32,      Intrinsic::amdgcn_mfma_f32_16x16x1f32},
      {SemOp::V_MFMA_F32_32x32x4_F16,      Intrinsic::amdgcn_mfma_f32_32x32x4f16},
      {SemOp::V_MFMA_F32_16x16x4_F16,      Intrinsic::amdgcn_mfma_f32_16x16x4f16},
      {SemOp::V_MFMA_F32_4x4x4_F16,        Intrinsic::amdgcn_mfma_f32_4x4x4f16},
      {SemOp::V_MFMA_I32_16x16x32_I8,      Intrinsic::amdgcn_mfma_i32_16x16x32_i8},
      {SemOp::V_MFMA_I32_32x32x16_I8,      Intrinsic::amdgcn_mfma_i32_32x32x16_i8},
      {SemOp::V_MFMA_F32_16x16x8_XF32,     Intrinsic::amdgcn_mfma_f32_16x16x8_xf32},
      {SemOp::V_MFMA_F32_32x32x4_XF32,     Intrinsic::amdgcn_mfma_f32_32x32x4_xf32},
      {SemOp::V_MFMA_I32_32x32x4_I8,       Intrinsic::amdgcn_mfma_i32_32x32x4i8},
      {SemOp::V_MFMA_I32_16x16x4_I8,       Intrinsic::amdgcn_mfma_i32_16x16x4i8},
      {SemOp::V_MFMA_I32_4x4x4_I8,         Intrinsic::amdgcn_mfma_i32_4x4x4i8},
      {SemOp::V_MFMA_F32_32x32x2_BF16,     Intrinsic::amdgcn_mfma_f32_32x32x2bf16},
      {SemOp::V_MFMA_F32_16x16x2_BF16,     Intrinsic::amdgcn_mfma_f32_16x16x2bf16},
      {SemOp::V_MFMA_F32_4x4x2_BF16,       Intrinsic::amdgcn_mfma_f32_4x4x2bf16},
      {SemOp::V_MFMA_F32_16x16x16_BF16_1K, Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k},
      {SemOp::V_MFMA_F32_32x32x8_BF16_1K,  Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k},
      {SemOp::V_MFMA_F32_16x16x32_BF16,    Intrinsic::amdgcn_mfma_f32_16x16x32_bf16},
      {SemOp::V_MFMA_F32_32x32x16_BF16,    Intrinsic::amdgcn_mfma_f32_32x32x16_bf16},
      {SemOp::V_MFMA_F32_16x16x32_F16,     Intrinsic::amdgcn_mfma_f32_16x16x32_f16},
      {SemOp::V_MFMA_F32_16x16x32_FP8_FP8, Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8},
      {SemOp::V_MFMA_F32_16x16x32_FP8_BF8, Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8},
      {SemOp::V_MFMA_F32_16x16x32_BF8_FP8, Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8},
      {SemOp::V_MFMA_F32_16x16x32_BF8_BF8, Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8},
      {SemOp::V_MFMA_F32_32x32x16_FP8_FP8, Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8},
      {SemOp::V_MFMA_F32_32x32x16_FP8_BF8, Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8},
      {SemOp::V_MFMA_F32_32x32x16_BF8_FP8, Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8},
      {SemOp::V_MFMA_F32_32x32x16_BF8_BF8, Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8},
      // gfx950 F8F6F4 scaled MFMAs. Intrinsic signature differs from the
      // non-scaled family (9 params instead of 6) and is overloaded on the
      // src AB type; the handler detects this via `FT->getNumParams() > 6`.
      {SemOp::V_MFMA_F32_16x16x128_F8F6F4,       Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4},
      {SemOp::V_MFMA_SCALE_F32_16x16x128_F8F6F4, Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4},
      {SemOp::V_MFMA_F32_32x32x64_F8F6F4,        Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4},
      {SemOp::V_MFMA_SCALE_F32_32x32x64_F8F6F4,  Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4},
  });
  return *table;
}

// Read a named immediate operand, or return `fallback` if the opcode does
// not expose that name. Using `getNamedOperandIdx` instead of positional
// scanning of the trailing source list means any future operand reshuffle
// in AMDGPU TableGen flows in for free.
static int64_t readNamedImm(const DecodedInst &di, AMDGPU::OpName name,
                            int64_t fallback = 0) {
  int idx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), name);
  if (idx < 0 || !di.isImm(idx))
    return fallback;
  return di.getImm(idx);
}

// Read a named register operand as a 32-bit value. Returns `fallback`
// when the named operand is absent or not a register (e.g. when the
// encoding carries an immediate in the same slot).
static Value *readNamedReg32(RaiseContext &ctx, const DecodedInst &di,
                             AMDGPU::OpName name, Value *fallback) {
  int idx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), name);
  if (idx < 0 || !di.isReg(idx))
    return fallback;
  ParsedReg pr = ctx.parseReg(di.getReg(idx), idx);
  if (pr.kind == ParsedReg::OTHER || pr.kind == ParsedReg::NOREG)
    return fallback;
  return ctx.regs.readReg32(ctx.B, pr);
}

HandlerResult handleMFMA(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  // AGPR moves travel through the MFMA format bit but are not MFMA ops.
  if (sop == SemOp::V_ACCVGPR_WRITE_B32 ||
      sop == SemOp::V_ACCVGPR_READ_B32) {
    ctx.writeReg32(op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }

  const auto &table = mfmaIntrinsicTable();
  auto it = table.find(sop);
  if (it == table.end()) {
    hr.failure = RaiseFailure::unsupportedShape(
        di, "MFMA", "no intrinsic mapping for this MFMA SemOp");
    errs() << "transpiler: Unknown MFMA: " << di.mnemonic << "\n";
    return hr;
  }
  const Intrinsic::ID intrId = it->second;

  // Derive the src / accum IR types from the intrinsic signature.
  //
  // Non-scaled MFMA intrinsics (`AMDGPUMfmaIntrinsic<DestTy, SrcABTy>`)
  // are not overloaded: six fixed parameters
  //   (SrcABTy, SrcABTy, DestTy, i32 cbsz, i32 abid, i32 blgp)
  // and `Intrinsic::getType(Ctx, ID)` returns the whole signature.
  //
  // Scaled F8F6F4 MFMAs (`AMDGPUMfmaScaleIntrinsic<DestTy>`) are overloaded
  // on the AB vector type: nine parameters
  //   (anyvec A, anyvec B, DestTy C, i32 cbsz, i32 blgp,
  //    i32 opSelA, i32 scaleA, i32 opSelB, i32 scaleB)
  // and we must supply the overload types. AMDGPU kernels uniformly use
  // a v8i32 A/B layout (the widest F8 case) and select the active format
  // via `cbsz` / `blgp`, so we pass `{v8i32, v8i32}` here.
  auto *v8i32Ty = FixedVectorType::get(ctx.i32Ty, 8);
  SmallVector<Type *, 2> overloads;
  if (Intrinsic::isOverloaded(intrId))
    overloads = {v8i32Ty, v8i32Ty};

  FunctionType *FT = Intrinsic::getType(ctx.C, intrId, overloads);
  const bool isScaled = FT->getNumParams() == 9;
  if (!isScaled && FT->getNumParams() != 6)
    report_fatal_error(Twine("transpiler: unexpected MFMA intrinsic arity ") +
                       Twine(FT->getNumParams()) + " for " + di.mnemonic);

  Type *srcTy = FT->getParamType(0);
  Type *accumTy = FT->getReturnType();

  ParsedReg dest = op.dst();
  ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
  // The accumulator (src2) may be tied to the destination in some encodings.
  ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;
  if (srcA.kind == ParsedReg::OTHER || srcB.kind == ParsedReg::OTHER) {
    hr.failure = RaiseFailure::unsupportedShape(
        di, "MFMA", "cannot classify MFMA source registers");
    errs() << "transpiler: MFMA " << di.mnemonic
           << ": cannot read source registers\n";
    return hr;
  }

  Value *a = ctx.regs.readRegVec(ctx.B, srcA, srcTy);
  Value *b = ctx.regs.readRegVec(ctx.B, srcB, srcTy);
  Value *c = ctx.regs.readRegVec(ctx.B, srcC, accumTy);

  // Immediate modifiers keyed off the authoritative named-operand table.
  // `cbsz` is common to both families; `abid` is non-scaled only; scaled
  // instead carries `blgp` + four scale control operands.
  Value *cbsz = ConstantInt::get(ctx.i32Ty, readNamedImm(di, AMDGPU::OpName::cbsz));
  Value *blgp = ConstantInt::get(ctx.i32Ty, readNamedImm(di, AMDGPU::OpName::blgp));

  Function *mfmaFn = Intrinsic::getOrInsertDeclaration(&ctx.M, intrId, overloads);

  Value *callRet;
  if (isScaled) {
    // The scale operand layout mirrors `ScaledMAIInst` in
    // `VOP3PInstructions.td`: the two scale VGPRs come in as
    // `scale_src0` / `scale_src1`, and the op_sel bits are carried in the
    // repurposed `src0_modifiers` / `src1_modifiers` immediate slots.
    Value *zero = ConstantInt::get(ctx.i32Ty, 0);
    Value *opSelA = ConstantInt::get(
        ctx.i32Ty, readNamedImm(di, AMDGPU::OpName::src0_modifiers));
    Value *opSelB = ConstantInt::get(
        ctx.i32Ty, readNamedImm(di, AMDGPU::OpName::src1_modifiers));
    Value *scaleA =
        readNamedReg32(ctx, di, AMDGPU::OpName::scale_src0, zero);
    Value *scaleB =
        readNamedReg32(ctx, di, AMDGPU::OpName::scale_src1, zero);
    callRet = ctx.B.CreateCall(
        mfmaFn, {a, b, c, cbsz, blgp, opSelA, scaleA, opSelB, scaleB},
        "mfma_scale");
  } else {
    Value *abid = ConstantInt::get(
        ctx.i32Ty, readNamedImm(di, AMDGPU::OpName::abid));
    callRet =
        ctx.B.CreateCall(mfmaFn, {a, b, c, cbsz, abid, blgp}, "mfma");
  }

  ctx.writeRegVec(dest, callRet);
  hr.handled = true;
  return hr;
}

// Drift-detection for the one column of MFMA metadata that stays
// hand-rolled: `SemOp -> Intrinsic::ID`. Every MFMA-format MC opcode
// the decoder can reach at runtime must either (a) have a SemOp entry
// in `mfmaIntrinsicTable()`, or (b) be one of the two AGPR-move
// pseudos that `handleMFMA` short-circuits. Anything else is a silent
// coverage gap: the raiser would hit the "Unknown MFMA" path at
// per-kernel lift time instead of telling us at startup that the
// canon table and the handler table disagree. This is the same
// discipline `initMCState`'s `kMaxSrcs` check uses -- run once, fail
// loudly, no per-kernel surprises.
void verifyMFMACoverage(const MCInstrInfo &MCII, const OpcodeMap &opcMap) {
  const auto &table = mfmaIntrinsicTable();
  for (unsigned opc = 0, end = MCII.getNumOpcodes(); opc < end; ++opc) {
    const MCInstrDesc &desc = MCII.get(opc);
    if (!(desc.TSFlags & llvm::SIInstrFlags::IsMAI))
      continue;
    SemOp sop = opcMap.lookup(opc);
    if (sop == SemOp::Unknown)
      continue; // Opcode not modelled in kCanonTable -- fails at raise
                // time with "Unknown MFMA"; not a drift we own here.
    if (sop == SemOp::V_ACCVGPR_WRITE_B32 ||
        sop == SemOp::V_ACCVGPR_READ_B32)
      continue; // Handled specially above; no intrinsic entry needed.
    if (table.find(sop) == table.end())
      report_fatal_error(
          Twine("transpiler: MFMA-format opcode #") + Twine(opc) +
          " maps to SemOp " + Twine(static_cast<int>(sop)) +
          " but `mfmaIntrinsicTable` has no entry for it. Either add the "
          "Intrinsic::ID row or remove the SemOp from `kCanonTable`.");
  }
}

} // namespace transpiler
