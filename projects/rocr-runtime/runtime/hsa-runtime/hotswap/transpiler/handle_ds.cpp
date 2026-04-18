#include "handlers.hpp"

#include "semop.hpp"
#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx
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
    // 96-bit (3 x i32) LDS load/store. gfx11+ asm spellings are
    // `ds_load_b96` / `ds_store_b96`; LLVM MC keeps the legacy
    // `DS_READ_B96` / `DS_WRITE_B96` pseudo names. The generic
    // `vecTy = <3 x i32>` path below handles the lift; the gfx942
    // backend lowers the resulting `load <3 x i32>` /
    // `store <3 x i32>` to either a native ds_read_b96/ds_write_b96
    // (gfx9 inherits the `_vi` Real form from DSInstructions.td) or
    // splits into 3x ds_read_b32/ds_write_b32 — both correct.
    case SemOp::DS_READ_B96:   case SemOp::DS_WRITE_B96:
      return {3, 96, false};
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
  // 64-bit transposed LDS load with 8-bit elements (gfx950
  // `ds_read_b64_tr_b8` and its gfx1250 spelling `ds_load_tr8_b64`).
  //
  // Hardware behaviour (CDNA4 ISA §11.4 / RDNA4 gfx1250 spec):
  //   Each lane provides a base address via VGPR + immediate offset.
  //   The hardware reads 64 bits (8 x i8) from each lane's LDS
  //   address, then transposes across 8-lane groups so each lane
  //   post-transpose holds 8 i8 values that originally lived at the
  //   same intra-group element offset across 8 different source
  //   lanes:
  //     result[lane][k] = raw[group_base + k][l_in_group]   k = 0..7
  //   where group_base = lane & ~7 and l_in_group = lane & 7.
  //   The transpose converts M-contiguous LDS data into the
  //   WMMA/MFMA register layout where each thread holds K-contiguous
  //   elements.
  //
  // Software emulation (target gfx942 has no isel pattern for either
  // intrinsic — neither HasGFX950Insts nor isGFX1250Plus):
  //   1. Compute lane_id, group_base, l_in_group via
  //      mbcnt_lo/mbcnt_hi.
  //   2. For each output dword j (0..1, each holds 4 i8 values):
  //      For each in-dword element i (0..3):
  //        a. src_lane = group_base + (4 * j + i)
  //        b. base    = ds_bpermute(src_lane * 4, addr32)
  //        c. val_i8  = LOAD_I8(base + l_in_group)
  //        d. accumulate into outDw[j] at byte slot i.
  //   The direct byte-offset addressing into the source lane's
  //   contiguous data is byte-correct for i8 elements; no `* sizeof`
  //   scaling is needed because each element is one byte wide. Total:
  //   8 ds_bpermute + 8 i8 LDS loads + 8 OR/shifts per lane.
  //
  // EXEC gating: ds_bpermute is convergent and must run with all
  // hardware lanes participating; emit OUTSIDE emitUnderExec, same
  // contract as DS_LOAD_TR16_B128. Inactive-lane reads of the result
  // are not propagated to side effects under a correct source kernel
  // (each lane only reads its own packed-i8 result).
  auto emitDsLoadTr8B64 = [&]() {
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

    Value *lInGroup = ctx.B.CreateAnd(laneId, ctx.B.getInt32(7),
                                       "l_in_grp");
    Value *groupBase = ctx.B.CreateAnd(laneId,
        ctx.B.CreateNot(ctx.B.getInt32(7)), "grp_base");
    // Each i8 = 1 byte, so the per-element byte offset is just
    // l_in_group; no shift needed.

    Function *bperm = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_bpermute);

    Type *ptrLdsTy = PointerType::get(ctx.C, 3);
    auto *i8Ty = Type::getInt8Ty(ctx.C);

    // 2 output dwords, each holds 4 i8 values from 4 different
    // source lanes within the 8-lane transpose group.
    Value *outDw[2];
    for (unsigned j = 0; j < 2; j++) {
      Value *acc = ConstantInt::get(ctx.i32Ty, 0);
      for (unsigned i = 0; i < 4; i++) {
        Value *srcLane = ctx.B.CreateAdd(groupBase,
                            ctx.B.getInt32(4 * j + i));
        // ds_bpermute selector is byte-addressed (lane_id << 2).
        Value *base = ctx.B.CreateCall(bperm,
            {ctx.B.CreateShl(srcLane, ctx.B.getInt32(2)), addr32},
            "bp_base");
        Value *ldAddr = ctx.B.CreateAdd(base, lInGroup, "ld_addr");
        Value *ptr = ctx.B.CreateIntToPtr(
            ctx.B.CreateZExt(ldAddr, ctx.i64Ty), ptrLdsTy, "tr8_p");
        Value *valI8 = ctx.B.CreateLoad(i8Ty, ptr, "tr8_b");
        Value *valI32 = ctx.B.CreateZExt(valI8, ctx.i32Ty);
        Value *shifted = ctx.B.CreateShl(valI32,
                            ctx.B.getInt32(8 * i));
        acc = ctx.B.CreateOr(acc, shifted, "tr8_pack");
      }
      outDw[j] = acc;
    }

    ParsedReg dest = op.dst();
    for (unsigned j = 0; j < 2; j++)
      ctx.storeVGPR32(dest.baseIdx + j, outDw[j]);

    hr.handled = true;
  };

  if (sop == SemOp::DS_READ_B64_TR_B8 ||
      sop == SemOp::DS_LOAD_TR8_B64) {
    emitDsLoadTr8B64();
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
    // CROSS_LANE_SURVEY.md item P6 — lift `ds_swizzle_b32` through
    // `llvm.amdgcn.ds.swizzle`. The intrinsic signature is
    //   declare i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 immarg %offset)
    // (`ImmArg<ArgIndex<1>>`), so the second operand MUST be a
    // ConstantInt — we read the 16-bit `offset` field from the MC
    // operand table and materialise it as `i32 immarg`.
    //
    // MODREP: wave-width projection. The 16-bit swizzle imm encodes
    // one of seven swizzle modes (SIDefines.h `Swizzle::Id`),
    // partitioned into four valid top-nibble/byte envelopes plus a
    // RESERVED region:
    //
    //   QUAD_PERM   (top byte = 0x80)   — independent 4-lane quads.
    //   BITMASK_PERM (top bit = 0)      — per-lane swizzle via three
    //                                      5-bit AND/OR/XOR masks;
    //                                      includes BROADCAST / SWAP
    //                                      / REVERSE sub-encodings.
    //   FFT_MODE    (top nibble = 0xE)  — 5-bit FFT swizzle selector.
    //   ROTATE_MODE (top nibble = 0xC)  — 5-bit size + 1-bit dir.
    //   RESERVED    (top nibble in
    //                {0x9,0xA,0xB,0xD,0xF}) — undefined semantics.
    //
    // All four valid envelopes are wave-size-oblivious under modulo-
    // replication. The argument is the same for all four: wave64
    // ds_swizzle hardware preserves bit 5 of the lane id, so each
    // 32-lane half independently performs the same permutation. The
    // bit-5 preservation is a hardware contract documented neither
    // in IntrinsicsAMDGPU.td nor in AMDGPUUsage.rst, but it is a
    // wave64-GPU-family-wide property:
    //
    //   * The encoding (16-bit `Swizzle::EncBits`) has been stable
    //     since gfx7 (the first GPU with `ds_swizzle_b32`); the
    //     same single MC opcode (DS_SWIZZLE_B32) covers every
    //     wave64 target the transpiler can emit to (gfx7 hawaii,
    //     gfx8 fiji, gfx9 vega, gfx90a MI200, gfx942 CDNA3, gfx950).
    //   * Upstream LLVM's `test/CodeGen/AMDGPU/llvm.amdgcn.ds.
    //     swizzle.ll` runs on hawaii (gfx7 wave64) and fiji (gfx8
    //     wave64) and asserts the intrinsic lowers to a SINGLE
    //     `ds_swizzle_b32 ... offset:swizzle(...)` instruction with
    //     no extra arithmetic. If wave64 hardware did not preserve
    //     bit 5, the upper-half lanes would need fixup code to
    //     match the lower-half pattern, and that test (and every
    //     downstream wave64 user since gfx7) would have caught it.
    //   * Verified directly on gfx942 (CDNA3) wave64 across all
    //     four envelopes via inline-asm probes (results below); the
    //     observed per-32-lane-half independence is consistent with
    //     LLVM's wave64-family-wide assumption.
    //
    // The combined argument — encoding stability + LLVM's wave64-
    // family-wide implicit reliance + direct empirical verification
    // on gfx942 — is what justifies applying the safe-set decision
    // to every wave64 target the transpiler emits to. If the
    // contract is ever re-questioned for a specific target,
    // regenerate the probes from the documentation below for that
    // target.
    //
    // Per-envelope empirical evidence (gfx942 wave64):
    //
    //   * BROADCAST(32, 5) (imm=0x00A0, BITMASK_PERM): lanes 0..31
    //     → lane 5, lanes 32..63 → lane 37 (= 32+5).
    //   * SWAP-1           (imm=0x041F, BITMASK_PERM): lane 32↔33,
    //     34↔35, …, 62↔63. Each 32-lane half pairs internally.
    //   * FFT 0x00         (imm=0xE000): 5-bit-reverse within each
    //     32-lane half (lanes 32..63 produce lower-half-result + 32).
    //   * FFT 0x10         (imm=0xE010): 4-bit-reverse within each
    //     16-lane group, again per-half-independent.
    //   * FFT 0x1F         (imm=0xE01F): identity within each half.
    //   * ROTATE-LEFT-1    (imm=0xC020): lane 31 → lane 0 (lower
    //     half), lane 63 → lane 32 (upper half).
    //   * ROTATE-LEFT-31   (imm=0xC3E0): lane 0 → lane 31 (lower
    //     half), lane 32 → lane 63 (upper half).
    //
    // Reproduction: small HIP probes that issue `ds_swizzle_b32` via
    // inline asm against per-lane lane-id VGPRs and print each
    // lane's destination. Every probe confirms the upper-half
    // independence the modulo-replication argument requires.
    //
    // CI regression gate: `Gfx1250Gpu.DsSwizzle` (in
    // tests/gfx1250_gpu_test.cpp) lifts the committed
    // `test_data/gfx1250/ds_swizzle_gfx1250.hsaco` (a wave32
    // source kernel using `ds_swizzle_b32 offset:swizzle(SWAP,2)`)
    // and runs it on gfx942 wave64, verifying the per-32-lane-half
    // XOR-2 BITMASK_PERM swizzle pattern across all 64 lanes. The
    // SWAP-2 pattern is intentionally distinct from the GPT-OSS
    // `sum_bitmatrix_rows` corpus pattern (SWAP-1) so the test
    // catches imm-extraction bugs that happen to round-trip
    // SWAP-1.
    //
    // The classifier (`dsSwizzleSafeForModRep`) accepts all four
    // valid envelopes and refuses only the RESERVED top-nibble
    // region, where hardware semantics are undefined and a silent
    // lift would map the source kernel's imm onto whatever the
    // wave64 backend happens to do. Phase 1.4.5 surfaces the
    // refusal with a `RESERVED top-nibble envelope` detail line so
    // operators can identify the offending imm at triage time.
    //
    // Same-wave raises bypass the classifier (it early-returns when
    // src/tgt wave sizes match) and reach here unconditionally; for
    // same-wave every imm is correct by construction (the source and
    // target hardware execute the same `ds_swizzle_b32`
    // byte-for-byte), so the same intrinsic emit covers both paths.
    //
    // gfx942 backend support: confirmed via a separate verification
    // run — the lifted IR `call i32 @llvm.amdgcn.ds.swizzle(i32 %v,
    // i32 1055)` lowers through `llc -mcpu=gfx942
    // -mattr=+wavefrontsize64` to a direct `ds_swizzle_b32 ...
    // offset:swizzle(SWAP,1)` instruction with `.wavefront_size: 64`
    // metadata. No emulation needed (unlike P2's permlane16 path,
    // where the gfx942 backend lacked native isel and we routed
    // through ds_bpermute).
    //
    // EXEC gating: `amdgcn.ds.swizzle` is convergent (`isConvergent`
    // in DSInstructions.td); same convergence reasoning as the
    // `ds_bpermute` handler above applies — emit OUTSIDE
    // `emitUnderExec` so all hardware lanes participate, and trust
    // that inactive-lane reads of the result do not feed any
    // observable side effect under a correct source kernel.
    // The 16-bit imm is extracted once at decode time into
    // `di.dsSwizzleImm` (see `decode.cpp::decodeDsSwizzleImm`); the
    // decoder enforces the unsigned 16-bit range and refuses to
    // populate the field on missing / non-immediate / out-of-range
    // operands. A `!di.hasDsSwizzleImm` here means the decoder
    // rejected this exact site, which the cross-wave classifier
    // already mirrors as a refusal — but a same-wave raise bypasses
    // the classifier and still reaches us, so refuse loudly here too
    // for symmetry with the cross-wave path.
    if (!di.hasDsSwizzleImm) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "DS",
          "ds_swizzle_b32 missing/invalid OpName::offset immediate "
          "operand — decoder rejected the 16-bit imm");
      return hr;
    }
    // Read the data input via `OpName::addr` rather than positional
    // `op.src(0)`. ds_swizzle_b32's MCInst layout per
    // DSInstructions.td's DS_1A_RET base class is `(outs vdst), (ins
    // VGPR_32:$addr, Offset:$offset, gds)`; `buildSrcMap` (decode.cpp)
    // walks all non-modifier post-vdst operands, so srcMap[0] = $addr,
    // srcMap[1] = $offset, srcMap[2] = $gds. Using `op.src(0)` would
    // therefore work, but the named lookup is more explicit about
    // which operand is the data and is robust to future srcMap
    // refactors. (The other DS handlers in this file still use
    // positional `op.src(0)` for consistency with their existing
    // patterns; the named-lookup audit there is a system-wide cleanup
    // outside the scope of P6.)
    int addrIdx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(),
                                              AMDGPU::OpName::addr);
    if (addrIdx < 0 ||
        (unsigned)addrIdx >= di.inst.getNumOperands() ||
        !di.inst.getOperand((unsigned)addrIdx).isReg()) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "DS",
          "ds_swizzle_b32 missing OpName::addr VGPR operand — operand "
          "table mismatch");
      return hr;
    }
    Value *src = ctx.readOp32(di, (unsigned)addrIdx);
    Function *swiz = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_swizzle);
    Value *result = ctx.B.CreateCall(
        swiz,
        {src, ConstantInt::get(ctx.i32Ty, di.dsSwizzleImm)},
        "ds_swiz");
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
