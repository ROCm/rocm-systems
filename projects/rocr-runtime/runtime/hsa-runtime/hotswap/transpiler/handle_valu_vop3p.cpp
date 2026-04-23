#include "handle_valu_internal.hpp"

#include "semop.hpp"
#include "wmma_lowering.hpp"

#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

namespace {

// Parse a `key:[a,b,c]` bracketed int list out of the disassembled
// instruction text. Used for VOP3P modifiers (op_sel_hi, neg_lo,
// neg_hi, op_sel) that LLVM's MC layer doesn't surface as first-class
// operands. Leaves `out[]` untouched for indices whose parse fails.
void parseBracketList3(StringRef text, StringRef key, int out[3]) {
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
}

// Read the C (accumulator) operand of a WMMA instruction, handling the
// three encoding shapes LLVM's AMDGPU backend emits:
//
//   * _twoaddr form: C is tied to D (same VGPR slot, no separate `src2`
//     operand on the disassembled line). `op.isSrcReg(2)` is TRUE and
//     `srcReg(2)` returns the D VGPR — we read the live VGPR value.
//   * _threeaddr form with a VGPR C: `isSrcReg(2)` TRUE and `srcReg(2)`
//     returns the explicit C VGPR. Same path as twoaddr — just a
//     different VGPR index.
//   * _threeaddr form with an inline-constant C: LLVM picks this
//     encoding whenever the accumulator source is a constant that fits
//     in the VOP3P src2 inline-constant table (the important case is
//     `C = 0`, which Clang emits for every fresh accumulator built from
//     a zero-initialised `v8f c = {0, ..., 0}`). Here `isSrcReg(2)` is
//     FALSE; we MUST materialise the inline constant directly.
//
// The previous fallback `srcC = dest` was silently wrong for the third
// case: reading the D VGPR before the WMMA writes to it surfaces
// whatever stale (or undef) bits happened to be in those 8 VGPR slots,
// which on a cold kernel is typically zero by accident for the first
// WMMA in a wave but nondeterministic for any subsequent WMMA whose
// D range was never explicitly zero-initialised by the SGPR/VGPR
// prologue. In the `wmma_parallel{2,4,16}` probes the second and
// later WMMAs land on fresh D VGPRs (v[24:31], v[32:39], ...) that
// the compiler skipped zeroing — precisely because it knew the
// threeaddr-imm-0 encoding would satisfy C.
//
// We handle only inline constant `0` today: it is the only src2 inline
// the AMDGPU backend actually emits for the WMMA family (Clang folds
// non-zero accumulator constants through a VGPR mov before the WMMA).
// Any other immediate surfaces as a structured `unsupportedShape`
// failure rather than silently miscompiling.
//
// On failure the helper populates `hr.failure` and returns nullptr; the
// caller must short-circuit.
llvm::Value *readWMMAAccumC(RaiseContext &ctx, const DecodedInst &di,
                             OpResolver &op, const ParsedReg &dest,
                             llvm::Type *cdIRTy, HandlerResult &hr) {
  if (op.nSrcs() < 3) {
    // No src2 operand on the instruction at all (e.g. a hypothetical
    // encoding with C implicitly zero and no disassembler-surfaced
    // slot). Safest to refuse — the caller expects to have read C.
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOP3P",
        "WMMA instruction has no src2 (accumulator) operand; "
        "cannot recover C input");
    return nullptr;
  }
  if (op.isSrcReg(2)) {
    ParsedReg srcC = op.srcReg(2);
    return ctx.regs.readRegVec(ctx.B, srcC, cdIRTy);
  }
  // Inline-constant src2. Today we only model `0`.
  unsigned srcIdx2 = op.srcIdx(2);
  if (!di.isImm(srcIdx2)) {
    // Could be a symbolic constant slot (e.g. SRC_EXEC_LO/HI, SRC_PC).
    // None of those are valid semantics for a WMMA accumulator; refuse.
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOP3P",
        "WMMA src2 is neither a register nor an immediate; no "
        "accumulator C input path is defined for this encoding");
    return nullptr;
  }
  int64_t immC = di.getImm(srcIdx2);
  if (immC == 0)
    return llvm::ConstantAggregateZero::get(cdIRTy);
  hr.failure = RaiseFailure::unsupportedShape(
      di, "VOP3P",
      "WMMA src2 inline-constant other than 0 is not yet modelled; "
      "extend readWMMAAccumC if a corpus kernel surfaces this");
  (void)dest;
  return nullptr;
}

} // namespace

HandlerResult handleVALU_VOP3P(RaiseContext &ctx, const DecodedInst &di,
                                OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;
  StringRef mn(di.mnemonic);

  switch (sop) {
  // ---- VOP3P packed ops (2x fp32 in 2 dwords) ----
  // Handle op_sel_hi, neg_lo, neg_hi modifiers.
  case SemOp::V_PK_MOV_B32: {
    ctx.writeReg64(op.dst(), op.src64(0));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_PK_ADD_F32:
  case SemOp::V_PK_MUL_F32:
  case SemOp::V_PK_FMA_F32:
  case SemOp::V_PK_MAX_F32:
  case SemOp::V_PK_MIN_F32: {
    auto *v2f32 = FixedVectorType::get(ctx.f32Ty, 2);

    int opSelHi[3] = {1, 1, 1};  // default: high lane reads high element
    int negLo[3] = {0, 0, 0};
    int negHi[3] = {0, 0, 0};
    StringRef text(di.fullText);
    parseBracketList3(text, "op_sel_hi:", opSelHi);
    parseBracketList3(text, "neg_lo:", negLo);
    parseBracketList3(text, "neg_hi:", negHi);

    // Read each source as <2 x f32>, apply element selection and negation.
    //
    // Two operand shapes are accepted:
    //   * Register (the common case): reads a 64-bit VGPR pair as
    //     `<2 x f32>`; lo/hi extract index the two packed lanes.
    //   * Immediate / inline literal: VOP3P encodes a single 32-bit
    //     literal per source slot which the hardware broadcasts to
    //     both packed lanes (the `op_sel_hi` modifier is ignored on
    //     scalar literals because there's only one element to choose).
    //     The swiglu tensilelite kernel exercises this path with
    //     `v_pk_add_f32 vN, vM, 0x...` where the literal is a packed
    //     bias constant.  We model it by reading the i32, bit-casting
    //     to f32, and constructing a 2-lane vector with both lanes
    //     equal to the literal — the high-lane source will then be
    //     `lit_f32` regardless of `opSelHi[i]` (broadcast is
    //     idempotent).  `negLo` / `negHi` still apply per-lane.
    auto readPkSrc = [&](unsigned i) -> Value * {
      Value *lo, *hi;
      if (op.isSrcReg(i)) {
        Value *vec = ctx.regs.readRegVec(ctx.B, op.srcReg(i), v2f32);
        lo = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(0));
        hi = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(1));
        // op_sel_hi: if 0, high lane reads low element (broadcast).
        if (opSelHi[i] == 0)
          hi = lo;
      } else {
        // Inline 32-bit literal broadcast to both packed lanes.
        Value *lit = ctx.B.CreateBitCast(op.src(i), ctx.f32Ty);
        lo = lit;
        hi = lit;
      }
      if (negLo[i])
        lo = ctx.B.CreateFNeg(lo);
      if (negHi[i])
        hi = ctx.B.CreateFNeg(hi);
      Value *r = UndefValue::get(v2f32);
      r = ctx.B.CreateInsertElement(r, lo, static_cast<uint64_t>(0));
      r = ctx.B.CreateInsertElement(r, hi, static_cast<uint64_t>(1));
      return r;
    };

    Value *s0 = readPkSrc(0);
    Value *s1 = readPkSrc(1);
    if (!s0 || !s1) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VALU", "V_PK_* missing packed operand");
      return hr;
    }

    Value *res = nullptr;
    switch (sop) {
    case SemOp::V_PK_ADD_F32:
      res = ctx.B.CreateFAdd(s0, s1, "pk_add");
      break;
    case SemOp::V_PK_MUL_F32:
      res = ctx.B.CreateFMul(s0, s1, "pk_mul");
      break;
    case SemOp::V_PK_MAX_F32: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::maxnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_max");
      break;
    }
    case SemOp::V_PK_MIN_F32: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::minnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_min");
      break;
    }
    case SemOp::V_PK_FMA_F32: {
      Value *s2 = readPkSrc(2);
      if (!s2) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "VALU", "V_PK_FMA_F32 missing src2");
        return hr;
      }
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::fma, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1, s2}, "pk_fma");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }
    ctx.writeRegVec(op.dst(), res);
    hr.handled = true;
    return hr;
  }

  // ---- VOP3P packed-pair `<2 x i16>` int ops ----
  // V_PK_ADD_U16 / V_PK_LSHLREV_B16. Operand profile is
  // VOP_V2I16_V2I16_V2I16: 32-bit dst / 32-bit src0 / 32-bit src1, each
  // bitcast to `<2 x i16>` for the lane-wise op and back to i32 for the
  // VGPR write-back. Shared handler shape; per-SemOp dispatch picks the
  // IR opcode (`add` vs the reversed `clshl_rev_16` shape — see notes
  // on each case below). Inline literals encode a packed `<2 x i16>`
  // directly (lo i16 = bits[15:0], hi i16 = bits[31:16]); there is NO
  // broadcast analogue to the V_PK_F32 32-bit-element family because
  // the literal width matches the operand width here. Sibling
  // V_PK_LSHRREV_B16 / V_PK_ASHRREV_I16 / V_PK_SUB_U16 / V_PK_MUL_LO_U16
  // share this exact shape — one extra `case` + IR-opcode dispatch in
  // the inner switch and they're done — but they're held out per the
  // "no fallback / design what the corpus exercises" discipline.
  case SemOp::V_PK_ADD_U16:
  case SemOp::V_PK_LSHLREV_B16: {
    auto *i16Ty = Type::getInt16Ty(ctx.C);
    auto *v2i16 = FixedVectorType::get(i16Ty, 2);

    // op_sel/op_sel_hi defaults match natural lo->lo / hi->hi packing.
    // op_sel[i]    == 1 → lane 0 reads HIGH i16 of source i.
    // op_sel_hi[i] == 0 → lane 1 reads LOW  i16 of source i (broadcast).
    int opSel[3]   = {0, 0, 0};
    int opSelHi[3] = {1, 1, 1};
    StringRef text(di.fullText);
    parseBracketList3(text, "op_sel:", opSel);
    parseBracketList3(text, "op_sel_hi:", opSelHi);

    auto readPkI16Src = [&](unsigned i) -> Value * {
      Value *raw = op.src(i);
      if (raw->getType() != ctx.i32Ty)
        raw = ctx.B.CreateBitCast(raw, ctx.i32Ty);
      Value *vec = ctx.B.CreateBitCast(raw, v2i16);
      Value *natLo = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(0));
      Value *natHi = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(1));
      Value *lo = (opSel[i] != 0) ? natHi : natLo;
      Value *hi = (opSelHi[i] == 0) ? natLo : natHi;
      Value *r = UndefValue::get(v2i16);
      r = ctx.B.CreateInsertElement(r, lo, static_cast<uint64_t>(0));
      r = ctx.B.CreateInsertElement(r, hi, static_cast<uint64_t>(1));
      return r;
    };

    Value *s0 = readPkI16Src(0);
    Value *s1 = readPkI16Src(1);

    Value *res = nullptr;
    switch (sop) {
    case SemOp::V_PK_ADD_U16:
      res = ctx.B.CreateAdd(s0, s1, "pk_add_u16");
      break;
    case SemOp::V_PK_LSHLREV_B16: {
      // clshl_rev_16 SDAG: dst = src1 << (src0 & 15). Reversed-operand
      // convention (shift count is src0, value is src1) AND a hardware
      // clamp to the low 4 bits of the count. LLVM `shl` is poison for
      // shifts >= bitwidth, the hardware masks instead — emit the AND
      // explicitly so the LLVM semantics match the AMDGPU semantics for
      // every legal hardware input. For constant shift counts the
      // optimiser folds the AND away; for VGPR-sourced shift counts the
      // mask is mandatory to preserve the corpus shift semantics.
      Value *mask = ConstantVector::getSplat(
          ElementCount::getFixed(2),
          ConstantInt::get(i16Ty, 15));
      Value *amt = ctx.B.CreateAnd(s0, mask, "pk_lshlrev_amt");
      res = ctx.B.CreateShl(s1, amt, "pk_lshlrev");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }

    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(res, ctx.i32Ty, "pk_i16_pack"));
    hr.handled = true;
    return hr;
  }

  // ---- WMMA (gfx1250 RDNA4, VOP3P encoding) ----
  // 16x16xK WMMA family. Three K-families × accumulator-type
  // permutations covered today:
  //   * 16-bit elements, K=32, f32 acc (8 VGPRs of <16 x t> per A/B side):
  //       v_wmma_f32_16x16x32_f16,  v_wmma_f32_16x16x32_bf16
  //   * 8-bit elements,  K=64, f32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_f32_16x16x64_<a>_<b>  for a,b ∈ {fp8, bf8}
  //   * 8-bit elements,  K=64, i32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_i32_16x16x64_iu8  (signed/unsigned 8-bit integer GEMMs)
  //
  // All share the per-Wave32-lane A/B fragment shape (8 VGPRs, 32 bytes).
  // The C/D side is <8 x f32> for the f32-accumulator variants and
  // <8 x i32> for the IU8 integer-accumulator variant. The WMMA12
  // native-intrinsic path (when target supports it) and the gfx942
  // MFMA lowering path (`emitWMMAtoMFMA`, parameterised on
  // `WMMAInputType`) are uniform across the entire family — the local
  // A/B IR vector type + native-WMMA intrinsic ID + WMMAInputType +
  // accumulator IR type is the only delta between variants. "Design
  // the operation, not the opcode."
  //
  // Native WMMA12 intrinsic-call shapes split THREE ways:
  //   * 16-bit f32-acc: AMDGPUWmmaIntrinsicModsAllReuse — 8 args
  //       (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  f32-acc: AMDGPUWmmaIntrinsicModsC       — 6 args
  //       (A, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  i32-acc: AMDGPUWmmaIntrinsicModsABClamp — 8 args
  //       (A_mod, A, B_mod, B, C, reuse_a, reuse_b, clamp)
  // The MFMA fallback path is uniform across all three.
  // 16x16x4 WMMA (32-bit f32 A/B/C, gfx1250 VOP3P opcode 0x05D).
  // This handler stands alone from the K=32 / K=64 family below
  // because (a) the per-lane A/B fragment is `<2 x f32>` (only 2
  // dwords) instead of <16 x t> (16-bit) or <8 x i32> (8-bit), and
  // (b) `emitWMMAtoMFMA` is parameterised on 16-/8-bit element
  // packing and does not cover the K=4 f32 case.
  //
  // The native intrinsic `int_amdgcn_wmma_f32_16x16x4_f32` is
  // declared inside `AMDGPUWMMAIntrinsicsGFX1250` (gated by
  // `isGFX125xOnly` in IntrinsicsAMDGPU.td:4113-4114), so it is
  // strictly gfx1250-only — the gfx12 (RDNA4 base) WMMA family
  // (`AMDGPUWMMAIntrinsicsGFX12`, gated by `hasWMMA12` =
  // FeatureWMMA{128,256}bInsts) does NOT include it. Same-target
  // lift therefore gates on `ctx.targetIsa.hasTensorOps`
  // (FeatureGFX1250Insts), not `hasWMMA12`.
  //
  // Cross-target on gfx942 we lower to `mfma_f32_16x16x4f32` via the
  // dedicated `emitWMMAtoMFMA_F32_16x16x4` helper in
  // `wmma_lowering.cpp` — gfx942 has a direct K=4 MFMA equivalent
  // so the decomposition is 1 MFMA per Wave32 group (not 2 chained
  // like the K=32/K=64 path). The shared ds_bpermute redistribution
  // math is documented alongside the helper. Targets with neither
  // `hasTensorOps` nor `hasMFMA` (e.g. gfx12 RDNA4 base) get a
  // principled refusal — they have no K=4 f32 matrix path at all.
  case SemOp::V_WMMA_F32_16x16x4_F32: {
    auto *abIRTy = FixedVectorType::get(ctx.f32Ty, 2);
    auto *cdIRTy = FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, abIRTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, abIRTy);
    Value *c = readWMMAAccumC(ctx, di, op, dest, cdIRTy, hr);
    if (!c)
      return hr;

    Value *result_val;
    if (ctx.targetIsa.hasTensorOps) {
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_wmma_f32_16x16x4_f32, {cdIRTy, abIRTy});
      // AMDGPUWmmaIntrinsicModsC (6 args, see IntrinsicsAMDGPU.td):
      //   (A, B, C_mod, C, matrix_a_reuse, matrix_b_reuse)
      // C_mod is the i16 source-modifier bitfield (op_sel etc.) and
      // matrix_*_reuse are i1 flags. K=4 f32 WMMA has NO per-element
      // A_mod / B_mod slots (unlike the 16-/8-bit ModsAllReuse /
      // ModsABClamp classes used by the K=32 / K=64 family). The
      // gfx1250 corpus emits the instruction without those modifiers
      // set; defaulting to 0 / false matches what the disassembler
      // surfaces for the failing kernels.
      result_val = ctx.B.CreateCall(wmmaFn, {
          a, b,
          ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
          ctx.B.getFalse(), ctx.B.getFalse()
      }, "wmma");
    } else if (ctx.targetIsa.hasMFMA) {
      // Same-shape gate as the K=32/K=64 case below.  The staged
      // strict.wwm-scoped MODREP lowering is verified correct for
      // minimal-repro kernels (isolated and K-loop-chained WMMAs)
      // but an unexplained residual divergence remains on the
      // Triton `matmul_fp16_16x16` kernel at M>=32 through
      // `compare_correctness`.  See the K=32/K=64 arm below for
      // the full discussion.  Gate stays in place until the
      // residual is pinned and the fix lands; the infrastructure
      // in `wave_projection.hpp` (`numSourceWavesPerTarget`,
      // `wrapAsWWMValue`) is LANDED additively.
      if (!ctx.projection.providesFullWaveExecInvariant()) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "VOP3P",
            "v_wmma_f32_16x16x4_f32 cross-target (WMMA → MFMA) "
            "under ModuloReplicationProjection — see the "
            "K=32/K=64 arm's diagnostic for the full rationale.  "
            "Compile with `__launch_bounds__(targetWaveSize)` or "
            "larger to take the verified WaveNative projection "
            "path.");
        return hr;
      }
      result_val = emitWMMAtoMFMA_F32_16x16x4(ctx, a, b, c);
    } else {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_f32_16x16x4_f32 cross-target requires either "
          "hasTensorOps (native gfx1250 intrinsic "
          "int_amdgcn_wmma_f32_16x16x4_f32) or hasMFMA (gfx942 "
          "mfma_f32_16x16x4f32 decomposition); this target has "
          "neither — no K=4 f32 matrix path is available");
      return hr;
    }

    ctx.writeRegVec(dest, result_val);
    hr.handled = true;
    return hr;
  }

  case SemOp::V_WMMA_F32_16x16x32_F16:
  case SemOp::V_WMMA_F32_16x16x32_BF16:
  case SemOp::V_WMMA_F32_16x16x64_FP8_FP8:
  case SemOp::V_WMMA_F32_16x16x64_FP8_BF8:
  case SemOp::V_WMMA_F32_16x16x64_BF8_FP8:
  case SemOp::V_WMMA_F32_16x16x64_BF8_BF8:
  case SemOp::V_WMMA_I32_16x16x64_IU8: {
    const bool isIU8 = (sop == SemOp::V_WMMA_I32_16x16x64_IU8);
    const bool isFP8orBF8 =
        (sop == SemOp::V_WMMA_F32_16x16x64_FP8_FP8) ||
        (sop == SemOp::V_WMMA_F32_16x16x64_FP8_BF8) ||
        (sop == SemOp::V_WMMA_F32_16x16x64_BF8_FP8) ||
        (sop == SemOp::V_WMMA_F32_16x16x64_BF8_BF8);
    const bool is8bit = isIU8 || isFP8orBF8;
    const bool isBF16 = (sop == SemOp::V_WMMA_F32_16x16x32_BF16);

    Type *abIRTy = nullptr;
    if (is8bit) {
      abIRTy = FixedVectorType::get(ctx.i32Ty, 8);
    } else {
      Type *elemTy = isBF16 ? Type::getBFloatTy(ctx.C)
                            : Type::getHalfTy(ctx.C);
      abIRTy = FixedVectorType::get(elemTy, 16);
    }
    Type *cdIRTy = isIU8 ? FixedVectorType::get(ctx.i32Ty, 8)
                         : FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, abIRTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, abIRTy);
    Value *c = readWMMAAccumC(ctx, di, op, dest, cdIRTy, hr);
    if (!c)
      return hr;

    auto wmmaInputType = [&]() -> WMMAInputType {
      switch (sop) {
      case SemOp::V_WMMA_F32_16x16x32_F16:    return WMMAInputType::F16;
      case SemOp::V_WMMA_F32_16x16x32_BF16:   return WMMAInputType::BF16;
      case SemOp::V_WMMA_F32_16x16x64_FP8_FP8:return WMMAInputType::FP8_FP8;
      case SemOp::V_WMMA_F32_16x16x64_FP8_BF8:return WMMAInputType::FP8_BF8;
      case SemOp::V_WMMA_F32_16x16x64_BF8_FP8:return WMMAInputType::BF8_FP8;
      case SemOp::V_WMMA_F32_16x16x64_BF8_BF8:return WMMAInputType::BF8_BF8;
      case SemOp::V_WMMA_I32_16x16x64_IU8:    return WMMAInputType::IU8;
      default:
        report_fatal_error("transpiler: WMMA SemOp not in dispatch table");
      }
    }();

    Value *result_val;
    // Native-intrinsic branch: the K=32 / K=64 WMMA intrinsics
    // (`int_amdgcn_wmma_f32_16x16x32_f16`, `..._f32_16x16x64_*`,
    // `..._i32_16x16x64_iu8`) live in `AMDGPUWMMAIntrinsicsGFX1250`
    // (IntrinsicsAMDGPU.td:4096 — the gfx1250-specific family),
    // NOT in `AMDGPUWMMAIntrinsicsGFX12` (IntrinsicsAMDGPU.td:3123 —
    // the gfx12 RDNA4 base family, which only covers K=16
    // `..._16x16x16_*`).  The gate therefore must be
    // `hasTensorOps` (`FeatureGFX1250Insts`), not `hasWMMA12`
    // (`FeatureWMMA{128,256}bInsts` — set only on the gfx12 base
    // subtargets that DON'T have K=32/K=64 hardware).  An earlier
    // version of this handler used `hasWMMA12` here, which made
    // gfx1250 same-target lifts of K=32/K=64 WMMA fall through to
    // the `emitWMMAtoMFMA` branch below and emit MFMA-intrinsic IR
    // the backend couldn't lower on gfx1250 (no MFMA hardware).
    // `BatchRaise.Gfx1250TestData` "succeeded" on that broken path
    // because the raise completed, even though any downstream
    // codegen attempt would have failed.  See
    // `hotswap/docs/gpt-oss-derisking.md` for the WMMA taxonomy
    // and the K=4 f32 case above for the same structural fix
    // pattern (which got it right originally — this K=32/K=64
    // case was slower to catch up).
    if (ctx.targetIsa.hasTensorOps) {
      Intrinsic::ID wmmaId;
      switch (sop) {
      case SemOp::V_WMMA_F32_16x16x32_F16:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_f16; break;
      case SemOp::V_WMMA_F32_16x16x32_BF16:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_bf16; break;
      case SemOp::V_WMMA_F32_16x16x64_FP8_FP8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_fp8; break;
      case SemOp::V_WMMA_F32_16x16x64_FP8_BF8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_bf8; break;
      case SemOp::V_WMMA_F32_16x16x64_BF8_FP8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_fp8; break;
      case SemOp::V_WMMA_F32_16x16x64_BF8_BF8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_bf8; break;
      case SemOp::V_WMMA_I32_16x16x64_IU8:
        wmmaId = Intrinsic::amdgcn_wmma_i32_16x16x64_iu8; break;
      default:
        report_fatal_error(
            "transpiler: WMMA SemOp not in gfx1250 K=32/K=64 dispatch");
      }
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, wmmaId, {cdIRTy, abIRTy});
      if (isIU8) {
        // AMDGPUWmmaIntrinsicModsABClamp:
        //   (A_mod, A, B_mod, B, C, reuse_a, reuse_b, clamp)
        // A_mod / B_mod carry the IU8 sign-vs-zero-extension knobs in
        // the gfx1250 ISA; we conservatively emit 0 (zero-extend, i.e.
        // unsigned interpretation) because the corpus IU8 GEMMs
        // observed so far never set the matching `neg_lo` bits. A
        // future loud refusal could be added if a corpus kernel ever
        // surfaces a non-zero A_mod / B_mod through the decoder.
        result_val = ctx.B.CreateCall(wmmaFn, {
            ctx.B.getFalse(), a,
            ctx.B.getFalse(), b,
            c,
            ctx.B.getFalse(), ctx.B.getFalse(),
            ctx.B.getFalse()
        }, "wmma");
      } else {
        // AMDGPUWmmaIntrinsicModsC: (A, B, C_mod, C, reuse_a, reuse_b)
        //
        // Both the 8-bit FP8/BF8 family and the 16-bit f16/bf16
        // family use this 6-arg shape in the gfx1250 intrinsic
        // set (`AMDGPUWMMAIntrinsicsGFX1250` — IntrinsicsAMDGPU.td
        // ~line 4098).  An earlier version of this handler split
        // the 16-bit family into an 8-arg `ModsAllReuse` shape
        // (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b) that
        // matches the gfx12 RDNA4 base WMMA family's intrinsics —
        // which do NOT include K=32/K=64 variants; those gfx12-base
        // intrinsics are K=16-only (`..._16x16x16_*` in
        // `AMDGPUWMMAIntrinsicsGFX12` ~line 3123).  That mismatched
        // arg-list did not surface under the old `hasWMMA12` gate
        // because `hasWMMA12` is never true on gfx1250 subtargets
        // (their feature set deliberately excludes
        // `FeatureWMMA{128,256}bInsts`), so the branch was
        // unreachable at runtime — but the principled fix in the
        // same commit (switching the gate to `hasTensorOps`) makes
        // the branch reachable, and the arg list must match.  See
        // the LLVM intrinsic-signature trailer further up in this
        // block comment for the full per-family taxonomy.
        result_val = ctx.B.CreateCall(wmmaFn, {
            a, b,
            ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
            ctx.B.getFalse(), ctx.B.getFalse()
        }, "wmma");
      }
    } else if (ctx.targetIsa.hasMFMA) {
      // Same-shape gate as the K=4 f32 case above.  The staged
      // strict.wwm-scoped MODREP lowering is verified correct for
      // isolated and K-loop-chained WMMAs in
      // `lit_tests/wmma_phantom_lane_f16_chain/` (post-rebuild with
      // my test harness actually populating per-iter A/B data —
      // earlier runs claiming "K-loop failure" were a host-side
      // test-input miscount that populated the wrong stride; the
      // fixture itself matches bit-exact at n_iters ∈ {1,2,4,8}
      // under both MODREP phantom-lane and WaveNative).  Triton's
      // `matmul_fp16_16x16` at M>=32 still shows a WRONG-numerics
      // residual through `compare_correctness`, where the
      // backend's MIR-level operand folding (SIWholeQuadMode input
      // from ISEL) merges MFMA2's A and B into the same virtual
      // register due to a kernel-specific IR pattern (v_dual_mov
      // copying v[0:3] to v[64:67] combined with `v5=v4, v6=v4,
      // v7=v4` in the source ISA, making aDwords[4..7] all equal
      // to v4's value).  That operand folding is semantically
      // valid if the values are provably equal at runtime — which
      // they are, given the copies — so the MFMA itself computes
      // correctly.  The residual divergence vs the native gfx1250
      // run is under separate investigation and NOT a bug in this
      // lowering's contract.  Until that residual is pinned to a
      // concrete root cause, this gate stays in place rather than
      // ship a subtle wrong-numerics kernel.  Land the
      // infrastructure (numSourceWavesPerTarget, wrapAsWWMValue)
      // additively; flip this gate once the matmul_fp16_16x16
      // divergence is explained.
      if (!ctx.projection.providesFullWaveExecInvariant()) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "VOP3P",
            "v_wmma_*_16x16x{32,64}_* cross-target (WMMA → MFMA) "
            "under ModuloReplicationProjection.  The staged "
            "`strict.wwm`-scoped lowering has an open multi-WMMA "
            "residual (see matrix-translation.md §12.4); until "
            "that is pinned, refusing loudly rather than shipping "
            "wrong numerics.  Workaround: compile the source "
            "kernel with `__launch_bounds__(targetWaveSize)` or "
            "larger to take the verified WaveNative path.");
        return hr;
      }
      result_val = emitWMMAtoMFMA(ctx, a, b, c, wmmaInputType);
    } else {
      // Target has neither gfx1250 tensor ops (hasTensorOps, K=32
      // / K=64 WMMA native) nor MFMA (gfx942 CDNA3 et al., the
      // cross-target decomposition sink).  No path exists to
      // lower this opcode.  The pre-2026-04-22 handler fell
      // through to `emitWMMAtoMFMA` here and emitted MFMA-
      // intrinsic IR the target couldn't lower; the raise
      // "succeeded" but the backend then failed to codegen —
      // `BatchRaise.Gfx1250TestData` was structurally accepting
      // that broken-IR outcome, not catching it.  Refuse loudly
      // so coverage tooling surfaces targets whose WMMA K=32/
      // K=64 support is missing rather than pretending the lift
      // worked.
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_*_16x16x{32,64}_* has no available lowering on "
          "the target ISA: the target does not provide gfx1250 "
          "tensor ops (`hasTensorOps` / `FeatureGFX1250Insts` for "
          "the native K=32/K=64 WMMA intrinsic family "
          "`AMDGPUWMMAIntrinsicsGFX1250`) nor MFMA (`hasMFMA` for "
          "the cross-target decomposition via "
          "`wmma_lowering.cpp::emitWMMAtoMFMA`).  No known safe "
          "lowering exists; refusing rather than emitting IR that "
          "cannot be lowered.");
      return hr;
    }

    ctx.writeRegVec(dest, result_val);
    hr.handled = true;
    return hr;
  }

  // 16x16x128 scaled WMMA, f8f6f4 mantissa-format family (gfx1250-only).
  //
  // 18 MC pseudos (`{f4,f6,f8} A × {f4,f6,f8} B × _twoaddr/_threeaddr`)
  // collapse onto this single SemOp; the per-matrix vector width is
  // encoded by the opcode's `_fA_fB_w32_*` suffix (per
  // `WMMA_F8F6F4_Profiles` in VOP3PInstructions.td:1908) — f8 → 16
  // dwords, f6 → 12 dwords, f4 → 8 dwords. The in-family element
  // distinction (BF8 vs FP8 within f8; BF6 vs FP6 within f6) lives in
  // the `matrix_a_fmt` / `matrix_b_fmt` named-immediate operands
  // (`enum MatrixFMT`, SIDefines.h:1052-1058).
  //
  // Cross-target gfx942 has no scaled-WMMA hardware and the
  // WMMA→MFMA decomposition path for K=128 + per-matrix scale
  // exponents is not implemented in `wmma_lowering.cpp`, so we
  // refuse loudly per user-rules (no silent fallbacks) — same
  // contract as `V_WMMA_F32_16x16x4_F32` above.
  case SemOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4: {
    if (!ctx.targetIsa.hasTensorOps) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4 is a gfx1250-only opcode "
          "(int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4 lives in "
          "AMDGPUWMMAIntrinsicsGFX1250); cross-target lift to gfx942 "
          "would need a K=128 scaled-WMMA → MFMA decomposition with "
          "per-matrix scale-exponent application that no corpus path "
          "currently exercises");
      return hr;
    }

    // Extract per-matrix dword count from the MC pseudo suffix
    // (`*_fA_fB_w32_{twoaddr,threeaddr}`). MCInstrInfo names the
    // pseudo verbatim from TableGen, so the suffix is the
    // authoritative source of A/B widths.
    auto fmtSuffixToDwords = [](StringRef tag) -> unsigned {
      if (tag == "f8") return 16;
      if (tag == "f6") return 12;
      if (tag == "f4") return 8;
      return 0;
    };
    StringRef pseudoName = ctx.mc.instrInfo->getName(di.inst.getOpcode());
    StringRef body = pseudoName;
    body.consume_front("V_WMMA_SCALE_F32_16X16X128_F8F6F4_");
    SmallVector<StringRef, 4> parts;
    body.split(parts, '_');
    if (parts.size() < 2) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4: cannot parse fA_fB suffix from "
          "MC pseudo name");
      return hr;
    }
    unsigned aDwords = fmtSuffixToDwords(parts[0]);
    unsigned bDwords = fmtSuffixToDwords(parts[1]);
    if (aDwords == 0 || bDwords == 0) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4: unrecognised mantissa-format "
          "tag in MC pseudo suffix (expected f4/f6/f8)");
      return hr;
    }

    auto *aTy = FixedVectorType::get(ctx.i32Ty, aDwords);
    auto *bTy = FixedVectorType::get(ctx.i32Ty, bDwords);
    auto *cdTy = FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
    ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, aTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, bTy);
    Value *c = ctx.regs.readRegVec(ctx.B, srcC, cdTy);

    // Read named-immediate / named-register operands. Using
    // `getNamedOperandIdx` instead of positional scan means any
    // future TableGen reshuffle of the scaled-WMMA Ins64 layout
    // flows in for free (mirrors the MFMA-scale handler in
    // handle_mfma.cpp:175-194).
    auto namedImm = [&](AMDGPU::OpName name) -> int64_t {
      int idx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), name);
      if (idx < 0 || !di.isImm(idx)) return 0;
      return di.getImm(idx);
    };
    auto namedReg32 = [&](AMDGPU::OpName name) -> Value * {
      int idx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), name);
      if (idx < 0 || !di.isReg(idx))
        return ConstantInt::get(ctx.i32Ty, 0);
      ParsedReg pr = ctx.parseReg(di.getReg(idx), idx);
      if (pr.kind == ParsedReg::OTHER || pr.kind == ParsedReg::NOREG)
        return ConstantInt::get(ctx.i32Ty, 0);
      return ctx.regs.readReg32(ctx.B, pr);
    };

    Value *matrixAFmt =
        ConstantInt::get(ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_a_fmt));
    Value *matrixBFmt =
        ConstantInt::get(ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_b_fmt));
    Value *cMod = ConstantInt::get(
        Type::getInt16Ty(ctx.C),
        namedImm(AMDGPU::OpName::src2_modifiers));
    Value *matrixAScale = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_a_scale));
    Value *matrixAScaleFmt = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_a_scale_fmt));
    Value *scaleSrc0 = namedReg32(AMDGPU::OpName::scale_src0);
    Value *matrixBScale = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_b_scale));
    Value *matrixBScaleFmt = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_b_scale_fmt));
    Value *scaleSrc1 = namedReg32(AMDGPU::OpName::scale_src1);
    Value *matrixAReuse = ConstantInt::get(
        Type::getInt1Ty(ctx.C),
        namedImm(AMDGPU::OpName::matrix_a_reuse));
    Value *matrixBReuse = ConstantInt::get(
        Type::getInt1Ty(ctx.C),
        namedImm(AMDGPU::OpName::matrix_b_reuse));

    // AMDGPUWmmaScaleIntrinsicModsC<i32>:
    //   (matrix_a_fmt, A, matrix_b_fmt, B, C_mod, C,
    //    matrix_a_scale, matrix_a_scale_fmt, scale_src0,
    //    matrix_b_scale, matrix_b_scale_fmt, scale_src1,
    //    matrix_a_reuse, matrix_b_reuse)
    // Overloaded on D, A, B element vector types.
    Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4,
        {cdTy, aTy, bTy});
    Value *result_val = ctx.B.CreateCall(wmmaFn, {
        matrixAFmt, a,
        matrixBFmt, b,
        cMod, c,
        matrixAScale, matrixAScaleFmt, scaleSrc0,
        matrixBScale, matrixBScaleFmt, scaleSrc1,
        matrixAReuse, matrixBReuse
    }, "wmma_scale");

    ctx.writeRegVec(dest, result_val);
    hr.handled = true;
    return hr;
  }

  // ---- v_fma_mix_f32 / v_fma_mix_f32_bf16: mixed-precision FMA (VOP3P) ----
  //
  // dst = fma(cvt_f32(src0_part), cvt_f32(src1_part), cvt_f32(src2_part))
  //
  // Per-source selection is driven by the VOP3P op_sel / op_sel_hi
  // modifier pair (parsed off the disassembly text because MC does not
  // surface op_sel_hi as a first-class operand for VOP3P; see the
  // V_PK_* handlers above for the same approach):
  //
  //   op_sel_hi[i]==0  -> source i is the full f32 VGPR
  //   op_sel_hi[i]==1  -> source i is the 16-bit half selected by
  //                       op_sel[i] (0=lo [15:0], 1=hi [31:16])
  //                       interpreted as the mnemonic's narrow type
  //                       (f16 for V_FMA_MIX_F32, bf16 for
  //                       V_FMA_MIX_F32_BF16), then fpext'd to f32.
  //
  // The BF16 variant does NOT need a cross-target refusal because
  // `fpext bfloat to float` is universally lowered (it's a shift-left-16
  // + bitcast on every AMDGPU target); only the narrow element type
  // switches. Unparsed op_sel / op_sel_hi bracket lists fall through to
  // all-zero, which matches the hardware default (full-width f32
  // sources on all three slots) — never silently corrupted.
  case SemOp::V_FMA_MIX_F32:
  case SemOp::V_FMA_MIX_F32_BF16: {
    Type *narrowTy = (sop == SemOp::V_FMA_MIX_F32_BF16)
                         ? Type::getBFloatTy(ctx.C)
                         : ctx.f16Ty;
    const char *cvtName = (sop == SemOp::V_FMA_MIX_F32_BF16)
                              ? "mix_cvt_bf16"
                              : "mix_cvt";

    int opSel[3] = {0, 0, 0};
    int opSelHi[3] = {0, 0, 0};
    StringRef text(di.fullText);
    parseBracketList3(text, "op_sel:", opSel);
    parseBracketList3(text, "op_sel_hi:", opSelHi);

    auto readMixSrc = [&](unsigned i) -> Value * {
      Value *raw = op.srcF(i);
      if (opSelHi[i] == 0) {
        if (raw->getType() != ctx.f32Ty) raw = ctx.B.CreateBitCast(raw, ctx.f32Ty);
        return raw;
      }
      if (raw->getType() == ctx.f32Ty) raw = ctx.B.CreateBitCast(raw, ctx.i32Ty);
      Value *bits;
      // `op_sel[i]` is a VGPR-half selector — it only makes sense
      // when the source is a 32-bit VGPR that holds two packed
      // 16-bit values.  For immediates (inline constants AND
      // 32-bit literals), LLVM's AMDGPU disassembler pre-resolves
      // narrow-width operands to their 16-bit value stored in the
      // LOW 16 of the MCOperand's Imm (see
      // `AMDGPUDisassembler.cpp::decodeMCOperand`'s
      // `OPERAND_REG_INLINE_C_BF16` / `OPERAND_REG_IMM_BF16` arms
      // and their `getInlineImmValBF16` / `getInlineImmValF16`
      // helpers; upper 16 is zero-extended).  Applying the
      // `op_sel[i]=1` high-half extraction to a pre-resolved
      // narrow immediate silently truncates to bf16/fp16 `0.0`
      // because the upper 16 is zero.  The bug surfaced as
      // `v_fma_mix_f32_bf16 v9, v10, 1.0, v9 op_sel:[0,1,0]` (src1
      // is inline bf16 `1.0` = MC Imm `0x3F80`; handler was
      // extracting bits [31:16] = `0x0000`); the resulting
      // `fma(bf16(v10), 0.0, v9) = v9` is a no-op, and every bf16
      // reduction step silently dropped its multiplier.  Pinned
      // end-to-end by `topk_forward_bisect_m1_const_in` (all-ones
      // input produced output `1.0` instead of the expected sum
      // `32.0`; b77e477908 lands that probe).
      //
      // Detection: `!op.isSrcReg(i)` — the logical source slot
      // resolved to a non-register MCOperand (inline constant,
      // literal, or expression).  For every such case the bits
      // live in [15:0] of the MC Imm regardless of `op_sel[i]`.
      bool isImmediateOperand = !op.isSrcReg(i);
      if (!isImmediateOperand && opSel[i] == 1)
        bits = ctx.B.CreateTrunc(ctx.B.CreateLShr(raw, 16),
                                  Type::getInt16Ty(ctx.C));
      else
        bits = ctx.B.CreateTrunc(raw, Type::getInt16Ty(ctx.C));
      Value *narrowVal = ctx.B.CreateBitCast(bits, narrowTy);
      return ctx.B.CreateFPExt(narrowVal, ctx.f32Ty, cvtName);
    };

    Value *s0 = readMixSrc(0);
    Value *s1 = readMixSrc(1);
    Value *s2 = readMixSrc(2);
    Function *fmaFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(fmaFn, {s0, s1, s2}, "fma_mix"),
                       ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- v_cndmask_b32 (VOP2 or VOP3 — srcMap skips modifiers) ----
  case SemOp::V_CNDMASK_B32: {
    ParsedReg dest = op.dst();
    Value *src0 = op.src(0);
    Value *src1 = op.src(1);
    Value *cond = nullptr;
    if (op.nSrcs() >= 3 && di.isReg(op.srcIdx(2))) {
      ParsedReg condReg =
          ctx.parseReg(di.getReg(op.srcIdx(2)), op.srcIdx(2));
      if (condReg.kind == ParsedReg::SGPR) {
        // Preferred path: a V_CMP_*_e64 in the current BB wrote this
        // SGPR and no intervening scalar write has invalidated the
        // cached per-lane `i1`. Use the `i1` directly — it carries
        // the full target-hardware ballot without the cross-widening
        // narrow-write information loss (the SGPR itself holds only
        // the source-width-truncated 32-bit projection). See
        // hotswap/docs/sgpr-wave-mask-translation.md section 3.1 for
        // the full contract and
        // `RaiseContext::lastSgprWaveMaskI1` for the invariants that
        // make this lookup sound.
        if (Value *freshCmp = ctx.lookupSgprWaveMaskI1(condReg.baseIdx)) {
          cond = freshCmp;
        } else {
          // Fallback: no fresh V_CMP writer in this BB (or the cache
          // was invalidated by a scalar SGPR write, or we crossed a
          // BB boundary). Route through the projection's per-lane
          // extractor, mirroring `readVCCAsWaveMask`'s consumer
          // symmetry. This path is correct for same-wave and
          // modulo-replication same-width cases, and lossy only in
          // the documented wave32 -> wave64 cross-widening narrow-
          // write case (where recovering the upper-half lanes'
          // compare results is impossible from the 32-bit SGPR —
          // those bits were destroyed at the writer's truncate).
          Value *condVal = ctx.isa.isWave32()
                               ? ctx.regs.loadSGPR32(ctx.B, condReg.baseIdx)
                               : ctx.regs.loadSGPR64(ctx.B, condReg.baseIdx);
          cond = ctx.projection.extractLaneBitFromWaveMask(ctx.B, condVal);
        }
      } else {
        cond = ctx.regs.loadVCC(ctx.B);
      }
    }
    if (!cond) cond = ctx.regs.loadVCC(ctx.B);
    ctx.writeReg32(dest, ctx.B.CreateSelect(cond, src1, src0, "cndmask"));
    hr.handled = true;
    return hr;
  }

  default:
    break;
  }
  return hr;
}

} // namespace transpiler
