#include "handlers.hpp"
#include "mubuf_addr.hpp"

#include "semop.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace transpiler {
HandlerResult handleMUBUF(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  // d16Half encodes D16 partial-write target: 0 = full-write (regular
  // load), 1 = low half (`_D16`), 2 = high half (`_D16_HI`). The
  // partial-write loads zero/sign-extend the loaded sub-dword to i16
  // and merge into the named half of the destination VGPR, preserving
  // the other half. See BUFInstructions.td:1155-1177 (predicate
  // `D16PreservesUnusedBits`).
  auto mubufClassify = [](SemOp s)
      -> std::tuple<bool, bool, int, int, bool, bool, int> {
    // returns {isLoad, isStore, dwords, loadBits, isSubDword, isSigned, d16Half}
    switch (s) {
    case SemOp::BUFFER_LOAD_DWORD:    return {true, false, 1, 32, false, false, 0};
    case SemOp::BUFFER_LOAD_DWORDX2:  return {true, false, 2, 64, false, false, 0};
    case SemOp::BUFFER_LOAD_DWORDX3:  return {true, false, 3, 96, false, false, 0};
    case SemOp::BUFFER_LOAD_DWORDX4:  return {true, false, 4, 128, false, false, 0};
    case SemOp::BUFFER_LOAD_UBYTE:    return {true, false, 1, 8, true, false, 0};
    case SemOp::BUFFER_LOAD_SBYTE:    return {true, false, 1, 8, true, true, 0};
    case SemOp::BUFFER_LOAD_USHORT:   return {true, false, 1, 16, true, false, 0};
    case SemOp::BUFFER_LOAD_SSHORT:   return {true, false, 1, 16, true, true, 0};
    case SemOp::BUFFER_LOAD_SHORT_D16:     return {true, false, 1, 16, true, false, 1};
    case SemOp::BUFFER_LOAD_SHORT_D16_HI:  return {true, false, 1, 16, true, false, 2};
    case SemOp::BUFFER_LOAD_UBYTE_D16:     return {true, false, 1, 8,  true, false, 1};
    case SemOp::BUFFER_LOAD_UBYTE_D16_HI:  return {true, false, 1, 8,  true, false, 2};
    case SemOp::BUFFER_LOAD_SBYTE_D16:     return {true, false, 1, 8,  true, true,  1};
    case SemOp::BUFFER_LOAD_SBYTE_D16_HI:  return {true, false, 1, 8,  true, true,  2};
    case SemOp::BUFFER_STORE_DWORD:   return {false, true, 1, 32, false, false, 0};
    case SemOp::BUFFER_STORE_DWORDX2: return {false, true, 2, 64, false, false, 0};
    case SemOp::BUFFER_STORE_DWORDX3: return {false, true, 3, 96, false, false, 0};
    case SemOp::BUFFER_STORE_DWORDX4: return {false, true, 4, 128, false, false, 0};
    case SemOp::BUFFER_STORE_BYTE:    return {false, true, 1, 8, true, false, 0};
    case SemOp::BUFFER_STORE_SHORT:   return {false, true, 1, 16, true, false, 0};
    default: return {false, false, 0, 0, false, false, 0};
    }
  };
  auto [isLoad, isStore, dwords, loadBits, isSubDword, isBufSigned, d16Half] =
      mubufClassify(sop);
  if (isLoad || isStore) {
    // Use gfx942 buffer intrinsics directly. The hardware handles OOB:
    // loads return 0, stores are silently dropped. This avoids the
    // flat-memory lowering that requires conditional branches (which
    // break under LLVM -O1+ SIMT optimizations).
    MubufAddr mbuf = decodeMubufAddr(ctx, di, op, isStore, "MUBUF");
    // For loads, vdata is the dst; for stores it's the first VGPR src
    // (captured into mbuf.stData by the decoder).
    ParsedReg vdata = isStore ? mbuf.stData : op.dst(0);
    Value *srd = mbuf.srd;
    Value *voffset = mbuf.voffset;
    Value *soffset = mbuf.soffset;
    Value *auxFlags = mbuf.auxFlags;

    if (isLoad) {
      if (isSubDword) {
        // Load the sub-dword datum and zero/sign-extend to i32. For
        // plain ushort/sbyte/etc. (`d16Half == 0`) we then write the
        // whole VGPR; for D16 partial-write loads we merge with the
        // prior dst (see comment block above mubufClassify).
        Type *memTy = (loadBits == 8) ? Type::getInt8Ty(ctx.C)
                                      : Type::getInt16Ty(ctx.C);
        Function *bufLd = Intrinsic::getOrInsertDeclaration(
            &ctx.M, Intrinsic::amdgcn_raw_buffer_load, {memTy});
        Value *loaded = ctx.B.CreateCall(bufLd,
            {srd, voffset, soffset, auxFlags}, "buf_ld");
        if (d16Half == 0) {
          Value *ext = isBufSigned ? ctx.B.CreateSExt(loaded, ctx.i32Ty)
                                   : ctx.B.CreateZExt(loaded, ctx.i32Ty);
          ctx.writeReg32(vdata, ext);
        } else {
          // Partial-write: extend to i16 (sign for `_SBYTE_D16*`,
          // zero for `_UBYTE_D16*` / `_SHORT_D16*`), zext to i32 so
          // the high half of the i32 is exactly zero before merging.
          Value *ext16 = loaded;
          if (loadBits == 8) {
            ext16 = isBufSigned
                        ? ctx.B.CreateSExt(loaded, Type::getInt16Ty(ctx.C))
                        : ctx.B.CreateZExt(loaded, Type::getInt16Ty(ctx.C));
          }
          Value *ext32 = ctx.B.CreateZExt(ext16, ctx.i32Ty);
          Value *prior = ctx.regs.readReg32(ctx.B, vdata);
          Value *merged;
          if (d16Half == 1) {
            // _D16: place datum in lo 16, preserve hi 16 of prior.
            Value *priorHi =
                ctx.B.CreateAnd(prior, ConstantInt::get(ctx.i32Ty, 0xFFFF0000));
            merged = ctx.B.CreateOr(priorHi, ext32, "d16_lo_merge");
          } else {
            // _D16_HI: place datum in hi 16, preserve lo 16 of prior.
            Value *priorLo =
                ctx.B.CreateAnd(prior, ConstantInt::get(ctx.i32Ty, 0x0000FFFF));
            Value *shifted =
                ctx.B.CreateShl(ext32, ConstantInt::get(ctx.i32Ty, 16));
            merged = ctx.B.CreateOr(priorLo, shifted, "d16_hi_merge");
          }
          ctx.writeReg32(vdata, merged);
        }
      } else if (dwords == 1) {
        Function *bufLd = Intrinsic::getOrInsertDeclaration(
            &ctx.M,
            Intrinsic::amdgcn_raw_buffer_load,
            {ctx.i32Ty});
        Value *loaded = ctx.B.CreateCall(bufLd,
            {srd, voffset, soffset, auxFlags}, "buf_ld");
        ctx.writeReg32(vdata, loaded);
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        Function *bufLd = Intrinsic::getOrInsertDeclaration(
            &ctx.M,
            Intrinsic::amdgcn_raw_buffer_load,
            {vecTy});
        Value *loaded = ctx.B.CreateCall(bufLd,
            {srd, voffset, soffset, auxFlags}, "buf_ld");
        ctx.writeRegVec(vdata, loaded);
      }
      hr.handled = true;
    return hr;
    }
    if (isStore) {
      // Use the gfx942 buffer-store intrinsic directly, exactly
      // mirroring the load path above. The hardware's BUFFER unit
      // handles OOB silently (the write is dropped when the per-lane
      // offset is >= num_records), so no software OOB sink is needed.
      //
      // The earlier implementation lowered every BUFFER_STORE_* to a
      // generic `store` against an `addrspacecast(alloca i32, addrspace(5))`
      // OOB sink (selected via `select i1 oob, sink, real`). That was
      // wrong on three independent axes:
      //
      //   1. Size mismatch. The sink alloca was always `i32` (4 B), but
      //      `BUFFER_STORE_DWORDX4` writes 16 B. For OOB lanes the
      //      flat_store_dwordx4 walked 12 B past the sink and into
      //      either the next per-thread scratch slot or unmapped
      //      scratch — root cause of the gfx1250 Triton vector-add
      //      SIGSEGV (R1).
      //
      //   2. Forced scratch enablement. Adding any `addrspace(5)`
      //      alloca that survives PromoteMemToReg makes the AMDGPU
      //      backend emit `.amdhsa_enable_private_segment 1` plus
      //      `.amdhsa_private_segment_fixed_size > 0`. Salmon's KD
      //      doesn't request `flat_scratch_init` (we model only the
      //      source ABI's user-SGPR set), so on entry FLAT_SCRATCH is
      //      undefined; any flat instruction touching the scratch
      //      aperture (including the OOB sink path) is a fault waiting
      //      to happen. Native gfx942 Triton emits `buffer_store_*`
      //      directly and reports `private_segment_fixed_size 0` /
      //      `enable_private_segment 0` — confirming hardware OOB
      //      handling is the right primitive.
      //
      //   3. Asymmetric with the load path. Loads already go through
      //      `amdgcn.raw.buffer.load` and rely on hardware OOB clamp.
      //      Routing stores through a software select+sink was an
      //      avoidable divergence whose only justification ("avoids
      //      flat-memory lowering with conditional branches breaking
      //      under -O1+ SIMT optimisations" — comment block above)
      //      doesn't apply when we use the buffer intrinsic itself.
      //
      // EXEC gating: like the load, the call is emitted unconditionally;
      // the hardware EXEC mask suppresses inactive-lane writes natively.
      // We still wrap in `emitUnderExec` so that within an
      // already-narrowed IR-level lane window (e.g. an inner if-then
      // branch handled earlier in the kernel) the store body is
      // dominated by the handler's lane-active diamond, keeping
      // EXEC and IR-level dominance in sync — same pattern other
      // store handlers (DS, scratch) use.
      Type *storeTy;
      Value *val;
      if (isSubDword) {
        storeTy = Type::getIntNTy(ctx.C, loadBits);
        val = ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, vdata), storeTy);
      } else if (dwords == 1) {
        storeTy = ctx.i32Ty;
        val = ctx.regs.readReg32(ctx.B, vdata);
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        storeTy = vecTy;
        val = ctx.regs.readRegVec(ctx.B, vdata, vecTy);
      }
      Function *bufSt = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_raw_buffer_store, {storeTy});
      ctx.emitUnderExec([&] {
        ctx.B.CreateCall(bufSt, {val, srd, voffset, soffset, auxFlags});
      });
      hr.handled = true;
    return hr;
    }
  }

  // ---- Buffer load to LDS (buffer_load_dword lds, ...) ----
  // Data goes directly to LDS at M0 + vaddr, not to a VGPR.
  // Model as: tmp = raw_buffer_load; ds_write(LDS[M0], tmp)
  if (sop == SemOp::BUFFER_LOAD_DWORD_LDS ||
      sop == SemOp::BUFFER_LOAD_DWORDX2_LDS ||
      sop == SemOp::BUFFER_LOAD_DWORDX4_LDS) {
    int dwords = (sop == SemOp::BUFFER_LOAD_DWORDX4_LDS) ? 4
               : (sop == SemOp::BUFFER_LOAD_DWORDX2_LDS) ? 2 : 1;

    MubufAddr mbuf = decodeMubufAddr(ctx, di, op, /*isStore=*/false,
                                      "MUBUF_LDS");

    // Load from buffer into a temp value.
    Type *ldTy = (dwords == 1) ? (Type *)ctx.i32Ty
                               : (Type *)FixedVectorType::get(ctx.i32Ty, dwords);
    Function *bufLd = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_raw_buffer_load, {ldTy});
    Value *loaded = ctx.B.CreateCall(
        bufLd, {mbuf.srd, mbuf.voffset, mbuf.soffset, mbuf.auxFlags},
        "lds_buf_ld");

    // Store to LDS at address from M0.
    ParsedReg m0Reg; m0Reg.kind = ParsedReg::M0; m0Reg.baseIdx = 0;
    Value *ldsAddr = ctx.regs.readReg32(ctx.B, m0Reg);
    auto *ldsPtrTy = PointerType::get(ctx.C, 3);
    Value *ldsPtr = ctx.B.CreateIntToPtr(ldsAddr, ldsPtrTy);
    ctx.emitUnderExec([&] { ctx.B.CreateStore(loaded, ldsPtr); });

    hr.handled = true;
    return hr;
  }

  // ---- Buffer atomics ----
  if (sop >= SemOp::BUFFER_ATOMIC_ADD && sop <= SemOp::BUFFER_ATOMIC_PK_ADD_F16) {
    assert(((di.tsFlags & SIInstrFlags::IsAtomicRet) != 0) == (di.numDefs > 0) &&
           "buffer atomic: IsAtomicRet disagrees with numDefs");
    MubufAtomicAddr atomic =
        decodeMubufAtomicAddr(ctx, di, op, "buffer_atomic");
    Value *gep = atomic.ptr;
    Value *data = ctx.regs.readReg32(ctx.B, op.dst(0));

    AtomicRMWInst::BinOp atomicOp;
    Type *atomicTy = ctx.i32Ty;
    bool isFP = false;
    switch (sop) {
    case SemOp::BUFFER_ATOMIC_ADD: atomicOp = AtomicRMWInst::Add; break;
    case SemOp::BUFFER_ATOMIC_SUB: atomicOp = AtomicRMWInst::Sub; break;
    case SemOp::BUFFER_ATOMIC_AND: atomicOp = AtomicRMWInst::And; break;
    case SemOp::BUFFER_ATOMIC_OR:  atomicOp = AtomicRMWInst::Or; break;
    case SemOp::BUFFER_ATOMIC_XOR: atomicOp = AtomicRMWInst::Xor; break;
    case SemOp::BUFFER_ATOMIC_ADD_F32:
      atomicOp = AtomicRMWInst::FAdd; atomicTy = ctx.f32Ty; isFP = true; break;
    case SemOp::BUFFER_ATOMIC_PK_ADD_BF16:
      atomicOp = AtomicRMWInst::FAdd;
      atomicTy = FixedVectorType::get(Type::getBFloatTy(ctx.C), 2); isFP = true; break;
    case SemOp::BUFFER_ATOMIC_PK_ADD_F16:
      atomicOp = AtomicRMWInst::FAdd;
      atomicTy = FixedVectorType::get(Type::getHalfTy(ctx.C), 2); isFP = true; break;
    default:
      llvm::errs() << "transpiler: Unsupported buffer atomic: " << mn << "\n";
      hr.failure = RaiseFailure::unsupportedShape(di, "MUBUF",
                                                   "unsupported buffer atomic");
      return hr;
    }
    if (isFP) data = ctx.B.CreateBitCast(data, atomicTy);
    ctx.emitUnderExec([&] {
      ctx.B.CreateAtomicRMW(atomicOp, gep, data, MaybeAlign(),
                            AtomicOrdering::Monotonic);
    });
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
