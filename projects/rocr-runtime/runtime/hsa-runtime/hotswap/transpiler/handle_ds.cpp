#include "handlers.hpp"

#include "semop.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace transpiler {
HandlerResult handleDS(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  // Map SemOp to {dwords, loadBits, isSigned} for DS read/write
  auto dsClassify = [](SemOp s) -> std::tuple<int, int, bool> {
    switch (s) {
    case SemOp::DS_READ_B128:  case SemOp::DS_WRITE_B128:
    case SemOp::DS_READ2_B64:  case SemOp::DS_WRITE2_B64:
      return {4, 128, false};
    case SemOp::DS_READ_B64:   case SemOp::DS_WRITE_B64:
    case SemOp::DS_READ2_B32:  case SemOp::DS_WRITE2_B32:
      return {2, 64, false};
    case SemOp::DS_READ_B32:   case SemOp::DS_WRITE_B32:
      return {1, 32, false};
    case SemOp::DS_READ_U16:   case SemOp::DS_WRITE_B16:
      return {0, 16, false};
    case SemOp::DS_READ_I16:
      return {0, 16, true};
    case SemOp::DS_READ_U8:    case SemOp::DS_WRITE_B8:
      return {0, 8, false};
    case SemOp::DS_READ_I8:
      return {0, 8, true};
    default: return {-1, 0, false};
    }
  };
  // ds_load_tr16_b128: LDS transpose load for Wave32 with 16-bit elements.
  //
  // Hardware behaviour (gfx1250, Wave32):
  //   Each lane provides a base address via VGPR + immediate offset.  The
  //   hardware reads 128 bits (8 × i16) from each lane's LDS address, then
  //   transposes the data across lanes within groups of 8.
  //
  //   Per the CDNA4 ISA doc §11.4 (gfx950 DS_READ_B64_TR_B16):
  //   "Read N bits of data per lane from data share. Interpret the data as
  //    a matrix with 16 bit elements and transpose the matrix."
  //   "Each lane (one VGPR) holds 4 consecutive M or N values."
  //
  //   The transpose converts M-contiguous LDS data into the WMMA/MFMA
  //   register layout where each thread holds K-contiguous elements.
  //
  // Software emulation (for gfx942 which lacks transpose loads):
  //   1. Contiguous 128-bit load (4 × i32) from each lane's address.
  //   2. 8×8 cross-lane transpose via ds_bpermute within groups of 8 lanes:
  //        result[lane][elem] = raw[group_base + elem][lane_in_group]
  //      This exchanges rows and columns so that each lane, which started
  //      with 8 values from consecutive M positions for one K column, now
  //      holds 8 values from different K columns for its M position.
  // gfx950 ds_read_b64_tr_b16: LDS transpose read, returns 64 bits as
  // v4i16 (4 × i16). Emit the LLVM intrinsic so the backend can lower it
  // to the correct instruction for the target ISA (native on gfx950,
  // software-emulated on targets that lack it).
  if (sop == SemOp::DS_READ_B64_TR_B16) {
    Value *addr = ctx.B.CreateZExt(op.src(0), ctx.i64Ty, "ds_addr");
    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        int64_t imm = di.getImm(op.srcIdx(k));
        if (imm != 0)
          addr = ctx.B.CreateAdd(addr, ConstantInt::get(ctx.i64Ty, imm), "ds_off");
        break;
      }
    }
    auto *ldsPtrTy = PointerType::get(ctx.C, 3);
    Value *ptr = ctx.B.CreateIntToPtr(addr, ldsPtrTy, "tr64_ptr");
    auto *v4i16Ty = FixedVectorType::get(Type::getInt16Ty(ctx.C), 4);
    Function *trFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_read_tr16_b64, {v4i16Ty});
    Value *trResult = ctx.B.CreateCall(trFn, {ptr}, "tr64_ld");
    Value *asDwords = ctx.B.CreateBitCast(
        trResult, FixedVectorType::get(ctx.i32Ty, 2), "tr64_dw");
    ParsedReg dest = op.dst();
    ctx.writeRegVec(dest, asDwords);
    hr.handled = true;
    return hr;
  }
  // gfx950 ds_read_b64_tr_b8: 64 bits of 8-bit data laid out 8x8 inside the
  // wave, then transposed across lanes. Use the LLVM intrinsic (v2i32) so
  // the backend selects the native instruction on gfx950 and emulates it
  // elsewhere.
  if (sop == SemOp::DS_READ_B64_TR_B8) {
    Value *addr = ctx.B.CreateZExt(op.src(0), ctx.i64Ty, "ds_addr");
    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        int64_t imm = di.getImm(op.srcIdx(k));
        if (imm != 0)
          addr = ctx.B.CreateAdd(addr, ConstantInt::get(ctx.i64Ty, imm), "ds_off");
        break;
      }
    }
    auto *ldsPtrTy = PointerType::get(ctx.C, 3);
    Value *ptr = ctx.B.CreateIntToPtr(addr, ldsPtrTy, "tr8_ptr");
    auto *v2i32Ty = FixedVectorType::get(ctx.i32Ty, 2);
    Function *trFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_read_tr8_b64, {v2i32Ty});
    Value *trResult = ctx.B.CreateCall(trFn, {ptr}, "tr8_ld");
    ctx.writeRegVec(op.dst(), trResult);
    hr.handled = true;
    return hr;
  }

  if (sop == SemOp::DS_LOAD_TR16_B128) {
    Value *addr = ctx.B.CreateZExt(op.src(0), ctx.i64Ty, "ds_addr");
    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        int64_t imm = di.getImm(op.srcIdx(k));
        if (imm != 0)
          addr = ctx.B.CreateAdd(addr, ConstantInt::get(ctx.i64Ty, imm), "ds_off");
        break;
      }
    }

    Type *ptrLdsTy = PointerType::get(ctx.C, 3);
    auto *v4i32Ty = FixedVectorType::get(ctx.i32Ty, 4);

    Value *ptr = ctx.B.CreateIntToPtr(addr, ptrLdsTy, "tr_ptr");
    Value *loaded = ctx.B.CreateLoad(v4i32Ty, ptr, "tr_load");

    // Optimized transpose via LDS re-reads.
    // Instead of 32 bpermute+select per transpose (which causes register
    // pressure and spills), bpermute only the base address from each source
    // lane, then do direct i16 LDS loads for the transposed elements.
    // This is 8 bpermute + 8 LDS loads = 16 ops vs 32 bpermute + selects.

    // Compute the 32-bit LDS base address (before the 128-bit load).
    Value *addr32 = op.src(0);
    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        int64_t imm = di.getImm(op.srcIdx(k));
        if (imm != 0)
          addr32 = ctx.B.CreateAdd(addr32,
                       ConstantInt::get(ctx.i32Ty, imm), "ds_off32");
        break;
      }
    }

    Function *mbcntLo = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_mbcnt_lo);
    Function *mbcntHi = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_mbcnt_hi);
    Value *allOnes = ConstantInt::getSigned(ctx.i32Ty, -1);
    Value *zero32 = ConstantInt::get(ctx.i32Ty, 0);
    Value *lo = ctx.B.CreateCall(mbcntLo, {allOnes, zero32}, "lane_lo");
    Value *laneId = ctx.B.CreateCall(mbcntHi, {allOnes, lo}, "lane_id");

    // L_in_group = lane_id % 8
    Value *lInGroup = ctx.B.CreateAnd(laneId, ctx.B.getInt32(7), "l_in_grp");
    // group_base = (lane_id / 8) * 8
    Value *groupBase = ctx.B.CreateAnd(laneId,
        ctx.B.CreateNot(ctx.B.getInt32(7)), "grp_base");
    // Byte offset for element L_in_group (each i16 = 2 bytes)
    Value *elemOff = ctx.B.CreateShl(lInGroup, ctx.B.getInt32(1), "elem_off");

    Function *bperm = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_bpermute);

    auto *i16Ty = Type::getInt16Ty(ctx.C);

    Value *outDw[4];
    for (unsigned j = 0; j < 4; j++) {
      Value *srcLo = ctx.B.CreateAdd(groupBase, ctx.B.getInt32(2 * j));
      Value *srcHi = ctx.B.CreateAdd(groupBase, ctx.B.getInt32(2 * j + 1));

      // Get source lane's LDS base address via ds_bpermute.
      Value *baseLo = ctx.B.CreateCall(bperm,
          {ctx.B.CreateShl(srcLo, ctx.B.getInt32(2)), addr32}, "bp_base_lo");
      Value *baseHi = ctx.B.CreateCall(bperm,
          {ctx.B.CreateShl(srcHi, ctx.B.getInt32(2)), addr32}, "bp_base_hi");

      // LDS address for element L_in_group in source lane's contiguous data.
      Value *ldAddrLo = ctx.B.CreateAdd(baseLo, elemOff, "ld_addr_lo");
      Value *ldAddrHi = ctx.B.CreateAdd(baseHi, elemOff, "ld_addr_hi");

      // Load i16 from LDS (address space 3).
      Value *ptrLo = ctx.B.CreateIntToPtr(
          ctx.B.CreateZExt(ldAddrLo, ctx.i64Ty), ptrLdsTy, "tr_p_lo");
      Value *ptrHi = ctx.B.CreateIntToPtr(
          ctx.B.CreateZExt(ldAddrHi, ctx.i64Ty), ptrLdsTy, "tr_p_hi");
      Value *valLo = ctx.B.CreateLoad(i16Ty, ptrLo, "tr_lo");
      Value *valHi = ctx.B.CreateLoad(i16Ty, ptrHi, "tr_hi");

      // Pack two i16 into one i32: (hi << 16) | lo
      Value *lo32 = ctx.B.CreateZExt(valLo, ctx.i32Ty);
      Value *hi32 = ctx.B.CreateZExt(valHi, ctx.i32Ty);
      outDw[j] = ctx.B.CreateOr(
          ctx.B.CreateShl(hi32, ctx.B.getInt32(16)), lo32, "tr_out");
    }

    ParsedReg dest = op.dst();
    for (unsigned j = 0; j < 4; j++)
      ctx.storeVGPR32(dest.baseIdx + j, outDw[j]);

    hr.handled = true;
    return hr;
  }
  bool isDsRead = sop >= SemOp::DS_READ_B32 && sop <= SemOp::DS_READ_I8;
  bool isDsWrite = sop >= SemOp::DS_WRITE_B32 && sop <= SemOp::DS_WRITE_B8;
  if (isDsRead || isDsWrite) {
    auto [dwords, loadBits, isSigned] = dsClassify(sop);

    Value *addr = ctx.B.CreateZExt(op.src(0), ctx.i64Ty, "ds_addr");

    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        int64_t imm = di.getImm(op.srcIdx(k));
        if (imm != 0)
          addr = ctx.B.CreateAdd(addr, ConstantInt::get(ctx.i64Ty, imm), "ds_off");
        break;
      }
    }

    Value *ptr = ctx.B.CreateIntToPtr(addr, PointerType::get(ctx.C, 3));

    if (isDsRead) {
      ParsedReg dest = op.dst();
      if (dwords == 0) {
        Type *memTy = Type::getIntNTy(ctx.C, loadBits);
        Value *v = ctx.B.CreateLoad(memTy, ptr, "ds_ld");
        ctx.writeReg32(dest, isSigned ? ctx.B.CreateSExt(v, ctx.i32Ty)
                                      : ctx.B.CreateZExt(v, ctx.i32Ty));
      } else if (dwords == 1) {
        ctx.writeReg32(dest, ctx.B.CreateLoad(ctx.i32Ty, ptr, "ds_ld"));
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        ctx.writeRegVec(dest, ctx.B.CreateLoad(vecTy, ptr, "ds_ld"));
      }
      hr.handled = true;
    return hr;
    }
    if (isDsWrite) {
      ParsedReg stData = op.srcReg(1);
      if (dwords == 0) {
        Type *memTy = Type::getIntNTy(ctx.C, loadBits);
        Value *val = ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, stData), memTy);
        ctx.emitUnderExec([&] { ctx.B.CreateStore(val, ptr); });
      } else if (dwords == 1) {
        Value *val = ctx.regs.readReg32(ctx.B, stData);
        ctx.emitUnderExec([&] { ctx.B.CreateStore(val, ptr); });
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        Value *val = ctx.regs.readRegVec(ctx.B, stData, vecTy);
        ctx.emitUnderExec([&] { ctx.B.CreateStore(val, ptr); });
      }
      hr.handled = true;
    return hr;
    }
  }
  if (sop == SemOp::DS_BPERMUTE_B32) {
    // Backwards permute: per-lane GATHER. Each lane reads the `src1`
    // value from a *source* lane whose index is `src0 >> 2` (the
    // selector is pre-scaled by the source compiler to match DS byte
    // addressing). The previous handler was an identity copy of
    // src(1), which silently collapsed `__shfl_xor(x, 1)` to `x` and
    // any shuffle-based reduction to "every lane keeps its own
    // partial" — exactly the `lane_swap` / `block_sum_shfl`
    // compare_correctness failures. Lower through the native
    // `amdgcn.ds_bpermute` intrinsic so the backend emits the real
    // cross-lane gather on the target ISA.
    //
    // MODREP: wave-width projection. `amdgcn.ds_bpermute` on the
    // target masks the selector by `target_wave_bits - 1` in hardware.
    // For gfx1250 wave32 → gfx942 wave64 lifts, a source selector of
    // `(lane ^ k) * 4` with `k < 32` stays inside the low 32 lanes,
    // so the natural wave64 behaviour partitions the wave into two
    // independent 32-lane halves that each reproduce the source's
    // wave32 shuffle — a clean match under modulo-replication.
    //
    //   Selectors this policy covers:
    //     * `__shfl_xor(x, k)` with `k < sourceWave` — `lane_swap`
    //       (k=1) and the intra-warp phase of `block_sum_shfl`
    //       (k ∈ {16,8,4,2,1}).
    //     * `__shfl_down(x, k)` with `k < sourceWave`.
    //     * `__shfl(x, srcLane)` with `srcLane < sourceWave`.
    //
    //   Selectors this policy does NOT cover:
    //     * `k ≥ sourceWave` — would step lanes from the "upper" half
    //       of the target wave into the "lower" half of the source
    //       replica, which is not what the source kernel meant. A
    //       proper lift needs SPMDification or same-wave execution.
    //     * Selectors derived from global thread/lane indices where
    //       the source's `mbcnt`-based lane ID cannot be re-expressed
    //       in target-wave terms (see lane_swap / block_sum_shfl
    //       residual numerical mismatches in RESULTS.md).
    //
    // Grep for MODREP when revisiting the cross-wave policy; these
    // assumptions will need to be either formalised (with a gate that
    // rejects selectors violating them) or generalised (SPMDification).
    //
    // EXEC gating. We emit the intrinsic *outside* `emitUnderExec`.
    // `amdgcn.ds_bpermute` is convergent — all lanes of the hardware
    // wave must participate or the result in inactive lanes is
    // undefined. For lanes that are inactive in the source kernel,
    // their `src1` input is the ambient VGPR value (possibly the
    // 0xA5A5… sentinel), but since no lane *reads* from an inactive
    // lane under a correct selector, the undef propagation does not
    // affect the active-lane outputs. If a future handler needs to
    // emit `ds_bpermute` on a value that was written inside an SPE
    // diamond, it must first broadcast the diamond result through a
    // `readfirstlane` / explicit VGPR move outside the diamond,
    // otherwise the cross-lane read will pick up `undef`.
    Value *index = op.src(0);
    Value *src = op.src(1);
    Function *bperm = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_bpermute);
    Value *gathered = ctx.B.CreateCall(bperm, {index, src}, "bperm");
    ctx.writeReg32(op.dst(), gathered);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::DS_SWIZZLE_B32) {
    // CROSS_LANE_SURVEY.md item P6 (pending). Refuse the lift until
    // the handler emits `llvm.amdgcn.ds.swizzle` with the offset
    // immediate plumbed through. The wave-size classifier (Phase
    // 1.4.5) catches this earlier in the cross-wave case and reports
    // the more specific `cross-wave-shuffle-rewrite-pending`
    // diagnostic; this same-wave fallback exists so that a same-wave
    // raise also fails loudly rather than silently same-laning the
    // cross-lane shuffle.
    hr.failure = RaiseFailure::unsupportedShape(
        di, "DS",
        "ds_swizzle_b32 — CROSS_LANE_SURVEY.md P6 lift not implemented");
    return hr;
  }
  return hr;
}

} // namespace transpiler
