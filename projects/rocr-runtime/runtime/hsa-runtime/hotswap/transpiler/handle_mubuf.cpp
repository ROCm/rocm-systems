#include "handlers.hpp"
#include "raiser.hpp"

#include "semop.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace transpiler {
HandlerResult handleMUBUF(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op, RaiseResult &result) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  auto mubufClassify = [](SemOp s) -> std::tuple<bool, bool, int, int, bool, bool> {
    // returns {isLoad, isStore, dwords, loadBits, isSubDword, isSigned}
    switch (s) {
    case SemOp::BUFFER_LOAD_DWORD:    return {true, false, 1, 32, false, false};
    case SemOp::BUFFER_LOAD_DWORDX2:  return {true, false, 2, 64, false, false};
    case SemOp::BUFFER_LOAD_DWORDX3:  return {true, false, 3, 96, false, false};
    case SemOp::BUFFER_LOAD_DWORDX4:  return {true, false, 4, 128, false, false};
    case SemOp::BUFFER_LOAD_UBYTE:    return {true, false, 1, 8, true, false};
    case SemOp::BUFFER_LOAD_SBYTE:    return {true, false, 1, 8, true, true};
    case SemOp::BUFFER_LOAD_USHORT:   return {true, false, 1, 16, true, false};
    case SemOp::BUFFER_LOAD_SSHORT:   return {true, false, 1, 16, true, true};
    case SemOp::BUFFER_LOAD_SHORT_D16:     return {true, false, 1, 16, true, false};
    case SemOp::BUFFER_LOAD_SHORT_D16_HI: return {true, false, 1, 16, true, false};
    case SemOp::BUFFER_STORE_DWORD:   return {false, true, 1, 32, false, false};
    case SemOp::BUFFER_STORE_DWORDX2: return {false, true, 2, 64, false, false};
    case SemOp::BUFFER_STORE_DWORDX3: return {false, true, 3, 96, false, false};
    case SemOp::BUFFER_STORE_DWORDX4: return {false, true, 4, 128, false, false};
    case SemOp::BUFFER_STORE_BYTE:    return {false, true, 1, 8, true, false};
    case SemOp::BUFFER_STORE_SHORT:   return {false, true, 1, 16, true, false};
    default: return {false, false, 0, 0, false, false};
    }
  };
  auto [isLoad, isStore, dwords, loadBits, isSubDword, isBufSigned] = mubufClassify(sop);
  if (isLoad || isStore) {

    // Scan source operands by register kind to handle both MUBUF and
    // VBUFFER encodings (where vaddr and srsrc order may differ).
    ParsedReg vdata = op.dst(0);
    ParsedReg srsrc{}, vaddrReg{}, soffReg{};
    bool haveSrsrc = false, haveVaddr = false, haveSoff = false;
    int64_t immOff = 0;
    int vgprSrcCount = 0;

    for (unsigned k = 0; k < op.nSrcs(); k++) {
      unsigned idx = op.srcIdx(k);
      if (di.isReg(idx)) {
        ParsedReg pr = op.srcReg(k);
        if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 && !haveSrsrc) {
          srsrc = pr; haveSrsrc = true;
        } else if (pr.kind == ParsedReg::VGPR) {
          vgprSrcCount++;
          // For stores (numDefs==0), the first VGPR source is vdata
          // (the stored value, already captured via op.dst(0)).
          // The second VGPR source is the actual buffer offset (vaddr).
          if (isStore && vgprSrcCount == 1)
            continue;
          if (!haveVaddr) {
            vaddrReg = pr; haveVaddr = true;
          }
        } else if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 && !haveSoff) {
          soffReg = pr; haveSoff = true;
        }
      } else if (di.isImm(idx)) {
        int64_t v = di.getImm(idx);
        if (v != 0 && immOff == 0)
          immOff = v;
      }
    }

    if (!haveSrsrc) {
      llvm::errs() << "transpiler: MUBUF: no SRSRC found for " << mn << "\n";
      result.failMnemonic = di.mnemonic;
        result.failFormat = "MUBUF";
        hr.handled = false;
        return hr;
    }

    // Use gfx942 buffer intrinsics directly. The hardware handles
    // OOB: loads return 0, stores are silently dropped. This avoids
    // the flat-memory lowering that requires conditional branches
    // (which break under LLVM -O1+ SIMT optimizations).
    Value *dw0 = ctx.regs.readReg32(ctx.B, srsrc);
    ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
    ParsedReg srsrc2 = srsrc; srsrc2.baseIdx = srsrc.baseIdx + 2;
    Value *dw1 = ctx.regs.readReg32(ctx.B, srsrc1);
    Value *dw2 = ctx.regs.readReg32(ctx.B, srsrc2);
    if (!dw0 || !dw1 || !dw2) {
      llvm::errs() << "transpiler: MUBUF: cannot read SRSRC for " << mn << "\n";
      result.failMnemonic = di.mnemonic;
        result.failFormat = "MUBUF";
        hr.handled = false;
        return hr;
    }

    // Build a gfx942-compatible raw buffer descriptor <4 x i32>.
    // Word 0: base_lo
    // Word 1: base_hi (only low 16 bits are address; clear stride/flags)
    // Word 2: num_records (byte count)
    // Word 3: 0 (raw buffer, TYPE=0, no format conversion)
    // Use readfirstlane to force SRD words into SGPRs, avoiding
    // the costly waterfall loop and incorrect register allocation.
    Function *readfirstlane = Intrinsic::getOrInsertDeclaration(
        &ctx.M,
        Intrinsic::amdgcn_readfirstlane, {ctx.i32Ty});
    Value *cleanDw1 = ctx.B.CreateAnd(dw1, ConstantInt::get(ctx.i32Ty, 0xFFFF));
    Value *srdW0 = ctx.B.CreateCall(readfirstlane, {dw0}, "srd_w0");
    Value *srdW1 = ctx.B.CreateCall(readfirstlane, {cleanDw1}, "srd_w1");
    Value *srdW2 = ctx.B.CreateCall(readfirstlane, {dw2}, "srd_w2");
    Value *word3 = ConstantInt::get(ctx.i32Ty, 0);
    Value *srd = UndefValue::get(FixedVectorType::get(ctx.i32Ty, 4));
    srd = ctx.B.CreateInsertElement(srd, srdW0, (uint64_t)0);
    srd = ctx.B.CreateInsertElement(srd, srdW1, (uint64_t)1);
    srd = ctx.B.CreateInsertElement(srd, srdW2, (uint64_t)2);
    srd = ctx.B.CreateInsertElement(srd, word3, (uint64_t)3);

    // Compute the per-lane VGPR offset (i32)
    Value *voffset = ConstantInt::get(ctx.i32Ty, 0);
    if (haveVaddr)
      voffset = ctx.B.CreateAdd(voffset, ctx.regs.readReg32(ctx.B, vaddrReg));
    if (immOff != 0)
      voffset = ctx.B.CreateAdd(voffset, ConstantInt::get(ctx.i32Ty, (int32_t)immOff));

    // SGPR offset
    Value *soffset = ConstantInt::get(ctx.i32Ty, 0);
    if (haveSoff)
      soffset = ctx.regs.readReg32(ctx.B, soffReg);

    Value *auxFlags = ConstantInt::get(ctx.i32Ty, 0);

    if (isLoad) {
      if (isSubDword) {
        if (loadBits == 8) {
          Function *bufLdI8 = Intrinsic::getOrInsertDeclaration(
              &ctx.M,
              Intrinsic::amdgcn_raw_buffer_load,
              {Type::getInt8Ty(ctx.C)});
          Value *loaded = ctx.B.CreateCall(bufLdI8,
              {srd, voffset, soffset, auxFlags}, "buf_ld");
          Value *ext = isBufSigned ? ctx.B.CreateSExt(loaded, ctx.i32Ty)
                                   : ctx.B.CreateZExt(loaded, ctx.i32Ty);
          ctx.regs.writeReg32(ctx.B, vdata, ext);
        } else {
          Function *bufLdI16 = Intrinsic::getOrInsertDeclaration(
              &ctx.M,
              Intrinsic::amdgcn_raw_buffer_load,
              {Type::getInt16Ty(ctx.C)});
          Value *loaded = ctx.B.CreateCall(bufLdI16,
              {srd, voffset, soffset, auxFlags}, "buf_ld");
          Value *ext = isBufSigned ? ctx.B.CreateSExt(loaded, ctx.i32Ty)
                                   : ctx.B.CreateZExt(loaded, ctx.i32Ty);
          ctx.regs.writeReg32(ctx.B, vdata, ext);
        }
      } else if (dwords == 1) {
        Function *bufLd = Intrinsic::getOrInsertDeclaration(
            &ctx.M,
            Intrinsic::amdgcn_raw_buffer_load,
            {ctx.i32Ty});
        Value *loaded = ctx.B.CreateCall(bufLd,
            {srd, voffset, soffset, auxFlags}, "buf_ld");
        ctx.regs.writeReg32(ctx.B, vdata, loaded);
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        Function *bufLd = Intrinsic::getOrInsertDeclaration(
            &ctx.M,
            Intrinsic::amdgcn_raw_buffer_load,
            {vecTy});
        Value *loaded = ctx.B.CreateCall(bufLd,
            {srd, voffset, soffset, auxFlags}, "buf_ld");
        ctx.regs.writeRegVec(ctx.B, vdata, loaded);
      }
      hr.handled = true;
    return hr;
    }
    if (isStore) {
      // Flat store with OOB sink: redirect out-of-bounds writes to
      // private scratch memory to avoid illegal memory access faults.
      Value *lo = ctx.B.CreateZExt(dw0, ctx.i64Ty);
      Value *hi = ctx.B.CreateAnd(ctx.B.CreateZExt(dw1, ctx.i64Ty),
                               ConstantInt::get(ctx.i64Ty, 0xFFFF));
      Value *basePtr = ctx.B.CreateOr(lo, ctx.B.CreateShl(hi, 32), "buf_base");
      Value *totalOff = ctx.B.CreateZExt(voffset, ctx.i64Ty);
      if (haveSoff)
        totalOff = ctx.B.CreateAdd(totalOff, ctx.B.CreateZExt(soffset, ctx.i64Ty));
      Value *numRec = ctx.B.CreateZExt(dw2, ctx.i64Ty);
      Value *oob = ctx.B.CreateICmpUGE(totalOff, numRec, "buf_oob");

      Value *realAddr = ctx.B.CreateAdd(basePtr, totalOff, "buf_addr");
      Value *realPtr = ctx.B.CreateIntToPtr(realAddr, PointerType::get(ctx.C, 0));

      Function *parentFn = ctx.kernel;
      IRBuilder<> entryB(&parentFn->getEntryBlock(),
                          parentFn->getEntryBlock().getFirstInsertionPt());
      AllocaInst *sink = entryB.CreateAlloca(ctx.i32Ty, /*AddrSpace=*/5,
                                              nullptr, "oob_sink");
      sink->setAlignment(Align(4));
      Value *sinkFlat = ctx.B.CreateAddrSpaceCast(sink, PointerType::get(ctx.C, 0));
      Value *storePtr = ctx.B.CreateSelect(oob, sinkFlat, realPtr,
                                        "store_ptr");

      ParsedReg storeData = op.dst(0);
      if (isSubDword) {
        Type *memTy = Type::getIntNTy(ctx.C, loadBits);
        Value *val = ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, storeData), memTy);
        ctx.B.CreateStore(val, storePtr);
      } else if (dwords == 1) {
        ctx.B.CreateStore(ctx.regs.readReg32(ctx.B, storeData), storePtr);
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        ctx.B.CreateStore(ctx.regs.readRegVec(ctx.B, storeData, vecTy), storePtr);
      }
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

    ParsedReg srsrcReg{}, vaddrReg{};
    bool haveSrsrc = false, haveVaddr = false, haveSoff = false;
    ParsedReg soffReg{};
    int64_t immOff = 0;

    for (unsigned k = 0; k < op.nSrcs(); k++) {
      unsigned idx = op.srcIdx(k);
      if (di.isReg(idx)) {
        ParsedReg pr = op.srcReg(k);
        if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 && !haveSrsrc) {
          srsrcReg = pr; haveSrsrc = true;
        } else if (pr.kind == ParsedReg::VGPR && !haveVaddr) {
          vaddrReg = pr; haveVaddr = true;
        } else if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 && !haveSoff) {
          soffReg = pr; haveSoff = true;
        }
      } else if (di.isImm(idx)) {
        int64_t v = di.getImm(idx);
        if (v != 0 && immOff == 0)
          immOff = v;
      }
    }

    if (!haveSrsrc) {
      llvm::errs() << "transpiler: MUBUF_LDS: no SRSRC for " << mn << "\n";
      result.failMnemonic = di.mnemonic;
      result.failFormat = "MUBUF";
      hr.handled = false;
      return hr;
    }

    // Build SRD <4 x i32>
    Value *dw0 = ctx.regs.readReg32(ctx.B, srsrcReg);
    ParsedReg s1 = srsrcReg; s1.baseIdx = srsrcReg.baseIdx + 1;
    ParsedReg s2 = srsrcReg; s2.baseIdx = srsrcReg.baseIdx + 2;
    Value *dw1 = ctx.regs.readReg32(ctx.B, s1);
    Value *dw2 = ctx.regs.readReg32(ctx.B, s2);
    Function *readfirstlane = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_readfirstlane, {ctx.i32Ty});
    Value *cleanDw1 = ctx.B.CreateAnd(dw1, ConstantInt::get(ctx.i32Ty, 0xFFFF));
    Value *srd = UndefValue::get(FixedVectorType::get(ctx.i32Ty, 4));
    srd = ctx.B.CreateInsertElement(srd, ctx.B.CreateCall(readfirstlane, {dw0}), (uint64_t)0);
    srd = ctx.B.CreateInsertElement(srd, ctx.B.CreateCall(readfirstlane, {cleanDw1}), (uint64_t)1);
    srd = ctx.B.CreateInsertElement(srd, ctx.B.CreateCall(readfirstlane, {dw2}), (uint64_t)2);
    srd = ctx.B.CreateInsertElement(srd, ConstantInt::get(ctx.i32Ty, 0), (uint64_t)3);

    Value *voffset = ConstantInt::get(ctx.i32Ty, 0);
    if (haveVaddr)
      voffset = ctx.B.CreateAdd(voffset, ctx.regs.readReg32(ctx.B, vaddrReg));
    if (immOff != 0)
      voffset = ctx.B.CreateAdd(voffset, ConstantInt::get(ctx.i32Ty, (int32_t)immOff));
    Value *soffset = haveSoff ? ctx.regs.readReg32(ctx.B, soffReg)
                              : ConstantInt::get(ctx.i32Ty, 0);
    Value *auxFlags = ConstantInt::get(ctx.i32Ty, 0);

    // Load from buffer into temp value(s)
    Type *ldTy = (dwords == 1) ? (Type *)ctx.i32Ty
                               : (Type *)FixedVectorType::get(ctx.i32Ty, dwords);
    Function *bufLd = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_raw_buffer_load, {ldTy});
    Value *loaded = ctx.B.CreateCall(bufLd, {srd, voffset, soffset, auxFlags}, "lds_buf_ld");

    // Store to LDS at address from M0
    ParsedReg m0Reg; m0Reg.kind = ParsedReg::M0; m0Reg.baseIdx = 0;
    Value *ldsAddr = ctx.regs.readReg32(ctx.B, m0Reg);
    auto *ldsPtrTy = PointerType::get(ctx.C, 3);
    Value *ldsPtr = ctx.B.CreateIntToPtr(ldsAddr, ldsPtrTy);
    ctx.B.CreateStore(loaded, ldsPtr);

    hr.handled = true;
    return hr;
  }

  // ---- Buffer atomics ----
  if (sop >= SemOp::BUFFER_ATOMIC_ADD && sop <= SemOp::BUFFER_ATOMIC_PK_ADD_F16) {
    ParsedReg srsrc = op.srcReg(0);
    Value *dw0 = ctx.regs.readReg32(ctx.B, srsrc);
    ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
    Value *dw1 = ctx.regs.readReg32(ctx.B, srsrc1);
    if (!dw0 || !dw1) {
      llvm::errs() << "transpiler: buffer_atomic: cannot read SRSRC\n";
      result.failMnemonic = di.mnemonic;
        result.failFormat = "MUBUF";
        hr.handled = false;
        return hr;
    }
    Value *lo = ctx.B.CreateZExt(dw0, ctx.i64Ty);
    Value *hi = ctx.B.CreateAnd(ctx.B.CreateZExt(dw1, ctx.i64Ty), ConstantInt::get(ctx.i64Ty, 0xFFFF));
    Value *ptr = ctx.B.CreateOr(lo, ctx.B.CreateShl(hi, 32), "buf_base");
    Value *gep = ctx.B.CreateIntToPtr(ptr, PointerType::get(ctx.C, 0));
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
      result.failMnemonic = di.mnemonic;
        result.failFormat = "MUBUF";
        hr.handled = false;
        return hr;
    }
    if (isFP) data = ctx.B.CreateBitCast(data, atomicTy);
    ctx.B.CreateAtomicRMW(atomicOp, gep, data, MaybeAlign(), AtomicOrdering::Monotonic);
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
