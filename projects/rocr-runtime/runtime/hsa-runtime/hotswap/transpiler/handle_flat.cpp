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
#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>

using namespace llvm;

namespace transpiler {
HandlerResult handleFLAT(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op, RaiseResult &result) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  if (sop == SemOp::GLOBAL_LOAD_USHORT || sop == SemOp::GLOBAL_LOAD_SHORT_D16_HI ||
      sop == SemOp::GLOBAL_LOAD_SSHORT || sop == SemOp::GLOBAL_LOAD_UBYTE ||
      sop == SemOp::GLOBAL_LOAD_SBYTE) {
    ParsedReg dest = op.dst();
    Value *addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
    if (addr->getType() != ctx.ptrGlobalTy) addr = ctx.B.CreateIntToPtr(addr, ctx.ptrGlobalTy);
    int64_t memOffset = 0;
    for (unsigned k = 1; k < op.nSrcs(); k++)
      if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
        memOffset = di.getImm(op.srcIdx(k));
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));

    bool isByte = sop == SemOp::GLOBAL_LOAD_UBYTE || sop == SemOp::GLOBAL_LOAD_SBYTE;
    Type *loadTy = isByte ? ctx.i8Ty : Type::getInt16Ty(ctx.C);
    Value *loaded = ctx.B.CreateLoad(loadTy, addr, "gload_sub");
    bool isUnsigned = sop == SemOp::GLOBAL_LOAD_UBYTE || sop == SemOp::GLOBAL_LOAD_USHORT;
    Value *ext = isUnsigned ? ctx.B.CreateZExt(loaded, ctx.i32Ty)
                            : ctx.B.CreateSExt(loaded, ctx.i32Ty);
    if (sop == SemOp::GLOBAL_LOAD_SHORT_D16_HI) {
      Value *prev = ctx.regs.readReg32(ctx.B, dest);
      ext = ctx.B.CreateOr(ctx.B.CreateAnd(prev, ConstantInt::get(ctx.i32Ty, 0xFFFF)),
                       ctx.B.CreateShl(ext, 16), "d16hi");
    }
    ctx.writeReg32(dest, ext);
    hr.handled = true;
    return hr;
  }

  if (sop == SemOp::GLOBAL_LOAD_DWORD || sop == SemOp::GLOBAL_LOAD_DWORDX2 ||
      sop == SemOp::GLOBAL_LOAD_DWORDX3 || sop == SemOp::GLOBAL_LOAD_DWORDX4) {
    int loadDwords = 1;
    if (sop == SemOp::GLOBAL_LOAD_DWORDX2) loadDwords = 2;
    else if (sop == SemOp::GLOBAL_LOAD_DWORDX3) loadDwords = 3;
    else if (sop == SemOp::GLOBAL_LOAD_DWORDX4) loadDwords = 4;

    ParsedReg dest = op.dst();

    // Detect SADDR form: operands are saddr(SGPR64), vaddr(VGPR32), offset, [flags]
    // vs plain form: vaddr(VGPR64), offset
    // The SADDR form has src[0]=SGPR, src[1]=VGPR.
    Value *addr = nullptr;
    bool hasSaddr = false;
    if (op.nSrcs() >= 2 && op.isSrcReg(0) && op.isSrcReg(1)) {
      ParsedReg r0 = op.srcReg(0);
      ParsedReg r1 = op.srcReg(1);
      if (r0.kind == ParsedReg::SGPR && r1.kind == ParsedReg::VGPR) {
        hasSaddr = true;
        Value *saddr = ctx.regs.readReg64(ctx.B, r0);
        Value *vaddr = ctx.B.CreateSExt(ctx.regs.readReg32(ctx.B, r1), ctx.i64Ty, "voff_sext");
        if (di.fullText.find("scale_offset") != std::string::npos) {
          int elemBytes = loadDwords * 4;
          vaddr = ctx.B.CreateMul(vaddr, ConstantInt::get(ctx.i64Ty, elemBytes), "scaled_voff");
        }
        addr = ctx.B.CreateAdd(saddr, vaddr, "saddr_vaddr");
      }
    }
    if (!addr) addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
    if (addr->getType() != ctx.ptrGlobalTy) addr = ctx.B.CreateIntToPtr(addr, ctx.ptrGlobalTy);

    // The first immediate operand after the registers is the memory offset.
    int64_t memOffset = 0;
    unsigned immStart = hasSaddr ? 2 : 1;
    for (unsigned k = immStart; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        memOffset = di.getImm(op.srcIdx(k));
        break;
      }
    }
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));

    if (loadDwords == 1) {
      ctx.writeReg32(dest, ctx.B.CreateBitCast(ctx.B.CreateLoad(ctx.f32Ty, addr, "gload"), ctx.i32Ty));
    } else {
      Type *vecTy = FixedVectorType::get(ctx.i32Ty, loadDwords);
      Value *loaded = ctx.B.CreateLoad(vecTy, addr, "gload");
      for (int d = 0; d < loadDwords; d++) {
        ParsedReg sub = dest;
        sub.baseIdx = dest.baseIdx + d;
        sub.width = 1;
        ctx.writeReg32(sub, ctx.B.CreateExtractElement(loaded, ctx.B.getInt32(d)));
      }
    }
    hr.handled = true;
    return hr;
  }

  if (sop == SemOp::GLOBAL_STORE_BYTE || sop == SemOp::GLOBAL_STORE_SHORT ||
      sop == SemOp::GLOBAL_STORE_SHORT_D16_HI || sop == SemOp::GLOBAL_STORE_DWORD ||
      sop == SemOp::GLOBAL_STORE_DWORDX2 || sop == SemOp::GLOBAL_STORE_DWORDX3 ||
      sop == SemOp::GLOBAL_STORE_DWORDX4) {
    int storeDwords = 1;
    int storeBits = 32;
    if (sop == SemOp::GLOBAL_STORE_DWORDX4) storeDwords = 4;
    else if (sop == SemOp::GLOBAL_STORE_DWORDX3) storeDwords = 3;
    else if (sop == SemOp::GLOBAL_STORE_DWORDX2) storeDwords = 2;
    else if (sop == SemOp::GLOBAL_STORE_DWORD) storeDwords = 1;
    else if (sop == SemOp::GLOBAL_STORE_SHORT ||
             sop == SemOp::GLOBAL_STORE_SHORT_D16_HI) {
      storeBits = 16;
      storeDwords = 0;
    } else if (sop == SemOp::GLOBAL_STORE_BYTE) {
      storeBits = 8;
      storeDwords = 0;
    }

    // global_store SADDR form: vaddr(VGPR32), vdata(VGPR), saddr(SGPR64), offset, [flags]
    // Plain form: vaddr(VGPR64), vdata(VGPR), offset
    Value *addr = nullptr;
    ParsedReg stData;
    bool hasSaddr = false;

    // Check for SADDR form: src[0]=VGPR(vaddr), src[1]=VGPR(data), src[2]=SGPR(saddr)
    if (op.nSrcs() >= 3 && op.isSrcReg(0) && op.isSrcReg(1) && op.isSrcReg(2)) {
      ParsedReg r0 = op.srcReg(0);
      ParsedReg r1 = op.srcReg(1);
      ParsedReg r2 = op.srcReg(2);
      if (r0.kind == ParsedReg::VGPR && r2.kind == ParsedReg::SGPR) {
        hasSaddr = true;
        Value *saddr = ctx.regs.readReg64(ctx.B, r2);
        Value *vaddr = ctx.B.CreateSExt(ctx.regs.readReg32(ctx.B, r0), ctx.i64Ty, "st_voff_sext");
        if (di.fullText.find("scale_offset") != std::string::npos) {
          int elemBytes = std::max(storeDwords, 1) * 4;
          if (storeBits < 32) elemBytes = storeBits / 8;
          vaddr = ctx.B.CreateMul(vaddr, ConstantInt::get(ctx.i64Ty, elemBytes), "st_scaled_voff");
        }
        addr = ctx.B.CreateAdd(saddr, vaddr, "st_saddr_vaddr");
        stData = r1;
      }
    }
    if (!addr) {
      addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
      stData = op.srcReg(1);
    }
    if (addr->getType() != ctx.ptrGlobalTy) addr = ctx.B.CreateIntToPtr(addr, ctx.ptrGlobalTy);
    int64_t memOffset = 0;
    unsigned immStart = hasSaddr ? 3 : 2;
    for (unsigned k = immStart; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        memOffset = di.getImm(op.srcIdx(k));
        break;
      }
    }
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));

    if (storeDwords == 0) {
      Type *memTy = Type::getIntNTy(ctx.C, storeBits);
      Value *val = ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, stData), memTy);
      ctx.emitUnderExec([&] { ctx.B.CreateStore(val, addr); });
    } else if (storeDwords == 1) {
      Value *val = ctx.regs.readReg32(ctx.B, stData);
      ctx.emitUnderExec([&] { ctx.B.CreateStore(val, addr); });
    } else {
      auto *vecTy = FixedVectorType::get(ctx.i32Ty, storeDwords);
      Value *val = ctx.regs.readRegVec(ctx.B, stData, vecTy);
      ctx.emitUnderExec([&] { ctx.B.CreateStore(val, addr); });
    }
    hr.handled = true;
    return hr;
  }

  // flat_load/flat_store — same structure as global but uses flat address space
  if (sop == SemOp::FLAT_LOAD_USHORT || sop == SemOp::FLAT_LOAD_SSHORT ||
      sop == SemOp::FLAT_LOAD_UBYTE || sop == SemOp::FLAT_LOAD_SBYTE) {
    ParsedReg dest = op.dst();
    Value *addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
    Type *ptrFlatTy = PointerType::get(ctx.C, 0);
    if (addr->getType() != ptrFlatTy) addr = ctx.B.CreateIntToPtr(addr, ptrFlatTy);
    int64_t memOffset = 0;
    for (unsigned k = 1; k < op.nSrcs(); k++)
      if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
        memOffset = di.getImm(op.srcIdx(k));
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));
    bool isByte = sop == SemOp::FLAT_LOAD_UBYTE || sop == SemOp::FLAT_LOAD_SBYTE;
    Type *loadTy = isByte ? ctx.i8Ty : Type::getInt16Ty(ctx.C);
    Value *loaded = ctx.B.CreateLoad(loadTy, addr, "flat_load_sub");
    bool isUnsigned = sop == SemOp::FLAT_LOAD_UBYTE || sop == SemOp::FLAT_LOAD_USHORT;
    Value *ext = isUnsigned ? ctx.B.CreateZExt(loaded, ctx.i32Ty) : ctx.B.CreateSExt(loaded, ctx.i32Ty);
    ctx.writeReg32(dest, ext);
    hr.handled = true;
    return hr;
  }

  if (sop == SemOp::FLAT_LOAD_DWORD || sop == SemOp::FLAT_LOAD_DWORDX2 ||
      sop == SemOp::FLAT_LOAD_DWORDX3 || sop == SemOp::FLAT_LOAD_DWORDX4) {
    int loadDwords = 1;
    if (sop == SemOp::FLAT_LOAD_DWORDX2) loadDwords = 2;
    else if (sop == SemOp::FLAT_LOAD_DWORDX4) loadDwords = 4;
    else if (sop == SemOp::FLAT_LOAD_DWORDX3) loadDwords = 3;

    ParsedReg dest = op.dst();
    Value *addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
    Type *ptrFlatTy = PointerType::get(ctx.C, 0);
    if (addr->getType() != ptrFlatTy) addr = ctx.B.CreateIntToPtr(addr, ptrFlatTy);
    int64_t memOffset = 0;
    for (unsigned k = 1; k < op.nSrcs(); k++)
      if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
        memOffset = di.getImm(op.srcIdx(k));
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));

    if (loadDwords == 1) {
      ctx.writeReg32(dest, ctx.B.CreateBitCast(ctx.B.CreateLoad(ctx.f32Ty, addr, "flat_load"), ctx.i32Ty));
    } else {
      Type *vecTy = FixedVectorType::get(ctx.i32Ty, loadDwords);
      Value *loaded = ctx.B.CreateLoad(vecTy, addr, "flat_load");
      for (int d = 0; d < loadDwords; d++) {
        ParsedReg sub = dest; sub.baseIdx = dest.baseIdx + d; sub.width = 1;
        ctx.writeReg32(sub, ctx.B.CreateExtractElement(loaded, ctx.B.getInt32(d)));
      }
    }
    hr.handled = true;
    return hr;
  }

  if (sop == SemOp::FLAT_STORE_DWORD || sop == SemOp::FLAT_STORE_DWORDX2 ||
      sop == SemOp::FLAT_STORE_DWORDX3 || sop == SemOp::FLAT_STORE_DWORDX4 ||
      sop == SemOp::FLAT_STORE_BYTE || sop == SemOp::FLAT_STORE_SHORT ||
      sop == SemOp::FLAT_STORE_SHORT_D16_HI) {
    int storeDwords = 1;
    int storeBits = 32;
    if (sop == SemOp::FLAT_STORE_DWORDX4) storeDwords = 4;
    else if (sop == SemOp::FLAT_STORE_DWORDX3) storeDwords = 3;
    else if (sop == SemOp::FLAT_STORE_DWORDX2) storeDwords = 2;
    else if (sop == SemOp::FLAT_STORE_DWORD) storeDwords = 1;
    else if (sop == SemOp::FLAT_STORE_SHORT ||
             sop == SemOp::FLAT_STORE_SHORT_D16_HI) { storeBits = 16; storeDwords = 0; }
    else if (sop == SemOp::FLAT_STORE_BYTE) { storeBits = 8; storeDwords = 0; }

    Value *addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
    Type *ptrFlatTy = PointerType::get(ctx.C, 0);
    if (addr->getType() != ptrFlatTy) addr = ctx.B.CreateIntToPtr(addr, ptrFlatTy);
    int64_t memOffset = 0;
    for (unsigned k = 2; k < op.nSrcs(); k++)
      if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
        memOffset = di.getImm(op.srcIdx(k));
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));

    ParsedReg stData = op.srcReg(1);
    if (storeDwords == 0) {
      Type *memTy = Type::getIntNTy(ctx.C, storeBits);
      Value *val = ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, stData), memTy);
      ctx.emitUnderExec([&] { ctx.B.CreateStore(val, addr); });
    } else if (storeDwords == 1) {
      Value *val = ctx.regs.readReg32(ctx.B, stData);
      ctx.emitUnderExec([&] { ctx.B.CreateStore(val, addr); });
    } else {
      auto *vecTy = FixedVectorType::get(ctx.i32Ty, storeDwords);
      Value *val = ctx.regs.readRegVec(ctx.B, stData, vecTy);
      ctx.emitUnderExec([&] { ctx.B.CreateStore(val, addr); });
    }
    hr.handled = true;
    return hr;
  }

  // flat_atomic_* — same as global_atomic but flat address space
  if (sop >= SemOp::FLAT_ATOMIC_ADD && sop <= SemOp::FLAT_ATOMIC_ADD_F32) {
    // Contract: the RTN/non-RTN collapse in OpcodeMap relies on
    // IsAtomicRet <=> (numDefs > 0) to decide result writeback below.
    assert(((di.tsFlags & SIInstrFlags::IsAtomicRet) != 0) == (di.numDefs > 0) &&
           "flat atomic: IsAtomicRet disagrees with numDefs");
    ParsedReg addrReg = op.srcReg(0);
    Value *addr = ctx.regs.readReg64(ctx.B, addrReg);
    Type *ptrFlatTy = PointerType::get(ctx.C, 0);
    if (addr->getType() != ptrFlatTy) addr = ctx.B.CreateIntToPtr(addr, ptrFlatTy);
    int64_t memOffset = 0;
    unsigned dataIdx = 1;
    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
        memOffset = di.getImm(op.srcIdx(k));
      else if (di.isReg(op.srcIdx(k)))
        dataIdx = k;
    }
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));
    Value *data = ctx.regs.readReg32(ctx.B, op.srcReg(dataIdx));

    if (sop == SemOp::FLAT_ATOMIC_CMPSWAP) {
      Value *cmpVal = ctx.regs.readReg32(ctx.B, op.srcReg(dataIdx));
      ParsedReg srcPair = op.srcReg(dataIdx);
      ParsedReg newReg = srcPair; newReg.baseIdx += 1; newReg.width = 1;
      Value *newVal = ctx.regs.readReg32(ctx.B, newReg);
      ctx.emitUnderExec([&] {
        auto *cas = ctx.B.CreateAtomicCmpXchg(
            addr, cmpVal, newVal, MaybeAlign(),
            AtomicOrdering::SequentiallyConsistent,
            AtomicOrdering::SequentiallyConsistent);
        if (di.numDefs > 0)
          ctx.regs.writeReg32(ctx.B, op.dst(),
                              ctx.B.CreateExtractValue(cas, 0));
      });
      hr.handled = true;
    return hr;
    }

    AtomicRMWInst::BinOp atomicOp;
    Type *atomicTy = ctx.i32Ty;
    bool isFP = false;
    switch (sop) {
    case SemOp::FLAT_ATOMIC_ADD:  atomicOp = AtomicRMWInst::Add; break;
    case SemOp::FLAT_ATOMIC_SUB:  atomicOp = AtomicRMWInst::Sub; break;
    case SemOp::FLAT_ATOMIC_AND:  atomicOp = AtomicRMWInst::And; break;
    case SemOp::FLAT_ATOMIC_OR:   atomicOp = AtomicRMWInst::Or; break;
    case SemOp::FLAT_ATOMIC_XOR:  atomicOp = AtomicRMWInst::Xor; break;
    case SemOp::FLAT_ATOMIC_SMIN: atomicOp = AtomicRMWInst::Min; break;
    case SemOp::FLAT_ATOMIC_SMAX: atomicOp = AtomicRMWInst::Max; break;
    case SemOp::FLAT_ATOMIC_UMIN: atomicOp = AtomicRMWInst::UMin; break;
    case SemOp::FLAT_ATOMIC_UMAX: atomicOp = AtomicRMWInst::UMax; break;
    case SemOp::FLAT_ATOMIC_SWAP: atomicOp = AtomicRMWInst::Xchg; break;
    case SemOp::FLAT_ATOMIC_ADD_F32:
      atomicOp = AtomicRMWInst::FAdd; isFP = true;
      data = ctx.B.CreateBitCast(data, ctx.f32Ty); atomicTy = ctx.f32Ty; break;
    default:
      result.failMnemonic = di.mnemonic; result.failFormat = "FLAT";
      llvm::errs() << "transpiler: Unhandled flat atomic: " << mn << "\n";
      result.failMnemonic = di.mnemonic;
        result.failFormat = "FLAT";
        hr.handled = false;
        return hr;
    }
    ctx.emitUnderExec([&] {
      auto *rmw = ctx.B.CreateAtomicRMW(
          atomicOp, addr, data, MaybeAlign(),
          AtomicOrdering::SequentiallyConsistent);
      if (di.numDefs > 0) {
        Value *retVal = rmw;
        if (isFP) retVal = ctx.B.CreateBitCast(retVal, ctx.i32Ty);
        ctx.regs.writeReg32(ctx.B, op.dst(), retVal);
      }
    });
    hr.handled = true;
    return hr;
  }

  // ---- Global atomics ----
  if (sop >= SemOp::GLOBAL_ATOMIC_ADD && sop <= SemOp::GLOBAL_ATOMIC_PK_ADD_F16) {
    assert(((di.tsFlags & SIInstrFlags::IsAtomicRet) != 0) == (di.numDefs > 0) &&
           "global atomic: IsAtomicRet disagrees with numDefs");
    ParsedReg addrReg = op.srcReg(0);
    Value *addr = ctx.regs.readReg64(ctx.B, addrReg);
    if (addr->getType() != ctx.ptrGlobalTy) addr = ctx.B.CreateIntToPtr(addr, ctx.ptrGlobalTy);
    int64_t memOffset = 0;
    unsigned dataIdx = 1;
    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
        memOffset = di.getImm(op.srcIdx(k));
      else if (di.isReg(op.srcIdx(k)))
        dataIdx = k;
    }
    if (memOffset != 0) addr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));
    Value *data = ctx.regs.readReg32(ctx.B, op.srcReg(dataIdx));

    if (sop == SemOp::GLOBAL_ATOMIC_CMPSWAP) {
      Value *cmpVal = data;
      ParsedReg srcPair = op.srcReg(dataIdx);
      ParsedReg newReg = srcPair; newReg.baseIdx += 1; newReg.width = 1;
      Value *newVal = ctx.regs.readReg32(ctx.B, newReg);
      ctx.emitUnderExec([&] {
        auto *cas = ctx.B.CreateAtomicCmpXchg(
            addr, cmpVal, newVal, MaybeAlign(), AtomicOrdering::Monotonic,
            AtomicOrdering::Monotonic);
        if (di.numDefs > 0)
          ctx.regs.writeReg32(ctx.B, op.dst(),
                              ctx.B.CreateExtractValue(cas, 0));
      });
      hr.handled = true;
    return hr;
    }

    AtomicRMWInst::BinOp atomicOp;
    Type *atomicTy = ctx.i32Ty;
    bool isFP = false;
    switch (sop) {
    case SemOp::GLOBAL_ATOMIC_ADD:  atomicOp = AtomicRMWInst::Add; break;
    case SemOp::GLOBAL_ATOMIC_SUB:  atomicOp = AtomicRMWInst::Sub; break;
    case SemOp::GLOBAL_ATOMIC_AND:  atomicOp = AtomicRMWInst::And; break;
    case SemOp::GLOBAL_ATOMIC_OR:   atomicOp = AtomicRMWInst::Or; break;
    case SemOp::GLOBAL_ATOMIC_XOR:  atomicOp = AtomicRMWInst::Xor; break;
    case SemOp::GLOBAL_ATOMIC_SMIN: atomicOp = AtomicRMWInst::Min; break;
    case SemOp::GLOBAL_ATOMIC_SMAX: atomicOp = AtomicRMWInst::Max; break;
    case SemOp::GLOBAL_ATOMIC_UMIN: atomicOp = AtomicRMWInst::UMin; break;
    case SemOp::GLOBAL_ATOMIC_UMAX: atomicOp = AtomicRMWInst::UMax; break;
    case SemOp::GLOBAL_ATOMIC_SWAP: atomicOp = AtomicRMWInst::Xchg; break;
    case SemOp::GLOBAL_ATOMIC_ADD_F32:
      atomicOp = AtomicRMWInst::FAdd; atomicTy = ctx.f32Ty; isFP = true; break;
    case SemOp::GLOBAL_ATOMIC_PK_ADD_BF16:
      atomicOp = AtomicRMWInst::FAdd;
      atomicTy = FixedVectorType::get(Type::getBFloatTy(ctx.C), 2); isFP = true; break;
    case SemOp::GLOBAL_ATOMIC_PK_ADD_F16:
      atomicOp = AtomicRMWInst::FAdd;
      atomicTy = FixedVectorType::get(Type::getHalfTy(ctx.C), 2); isFP = true; break;
    default:
      llvm::errs() << "transpiler: Unsupported global atomic variant: " << mn << "\n";
      result.failMnemonic = di.mnemonic;
        result.failFormat = "FLAT";
        hr.handled = false;
        return hr;
    }
    if (isFP) data = ctx.B.CreateBitCast(data, atomicTy);
    ctx.emitUnderExec([&] {
      Value *prev = ctx.B.CreateAtomicRMW(atomicOp, addr, data, MaybeAlign(),
                                          AtomicOrdering::Monotonic);
      if (di.numDefs > 0) {
        if (isFP) prev = ctx.B.CreateBitCast(prev, ctx.i32Ty);
        ctx.regs.writeReg32(ctx.B, op.dst(), prev);
      }
    });
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
