#include "reg_file.hpp"

#include "isa_profile.hpp"
#include "wave_projection.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::SGPR_32RegClassID, TTMP_32RegClassID

#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace llvm;

namespace transpiler {

namespace {

// Human-readable name for a ParsedReg::Kind. Used only in fatal-error
// messages, so an accidentally-missed case here surfaces as a slightly
// uglier diagnostic (`"?<9>"`) rather than a silent bug.
const char *kindName(ParsedReg::Kind k) {
  switch (k) {
  case ParsedReg::SGPR:       return "SGPR";
  case ParsedReg::VGPR:       return "VGPR";
  case ParsedReg::AGPR:       return "AGPR";
  case ParsedReg::VCC:        return "VCC";
  case ParsedReg::EXEC:       return "EXEC";
  case ParsedReg::SCC:        return "SCC";
  case ParsedReg::MODE:       return "MODE";
  case ParsedReg::M0:         return "M0";
  case ParsedReg::FLAT_SCR:   return "FLAT_SCR";
  case ParsedReg::TTMP:       return "TTMP";
  case ParsedReg::LDS_DIRECT: return "LDS_DIRECT";
  case ParsedReg::NOREG:      return "NOREG";
  case ParsedReg::OTHER:      return "OTHER";
  }
  return "<invalid>";
}

[[noreturn]] void failUnhandledKind(const char *fn, ParsedReg pr) {
  report_fatal_error(Twine("transpiler: ") + fn +
                     ": unhandled ParsedReg kind " + kindName(pr.kind) +
                     " (baseIdx=" + Twine(pr.baseIdx) +
                     ", width=" + Twine(pr.width) + "). Caller must "
                     "handle NOREG / OTHER / MODE before dispatching "
                     "through readReg32/readReg64.");
}

} // namespace

void AllocaRegFile::init(IRBuilder<> &B, Type *i32Ty, Type *i1Ty,
                          const ISAProfile &isa, const MCRegisterInfo &MRI,
                          const WaveProjection &proj) {
  projection = &proj;
  execTy = isa.isWave32() ? (Type *)i32Ty : (Type *)B.getInt64Ty();

  const unsigned nSGPR = MRI.getRegClass(AMDGPU::SGPR_32RegClassID).getNumRegs();
  sgpr.assign(nSGPR, nullptr);
  for (unsigned i = 0; i < nSGPR; ++i)
    sgpr[i] = B.CreateAlloca(i32Ty, nullptr, "sgpr" + std::to_string(i));

  // VGPR storage is explicitly oversized relative to TableGen's VGPR_32
  // class (see `kVGPRCap` docs in reg_file.hpp). AGPR storage mirrors
  // the VGPR size because AGPRs share the same index space under MFMA
  // encoding conventions.
  vgpr.assign(kVGPRCap, nullptr);
  for (unsigned i = 0; i < kVGPRCap; ++i)
    vgpr[i] = B.CreateAlloca(i32Ty, nullptr, "vgpr" + std::to_string(i));

  if (isa.hasAGPR) {
    agpr.assign(kVGPRCap, nullptr);
    for (unsigned i = 0; i < kVGPRCap; ++i)
      agpr[i] = B.CreateAlloca(i32Ty, nullptr, "agpr" + std::to_string(i));
  }

  // Condition-carrying scalar registers are initialised to zero so that a
  // read-before-write (raiser bug or unhandled instruction) yields a
  // deterministic "false / inactive" value rather than `undef`/poison.
  // Poison here would silently destroy SPE predication (the entry block
  // already reads EXEC, and VCC/SCC feed downstream branches). EXEC is
  // the only register initialised to all-ones — that is the architectural
  // boot state of a dispatched wave and is load-bearing for every
  // subsequent `emitLaneActiveBit` call.
  vcc = B.CreateAlloca(i1Ty, nullptr, "vcc");
  B.CreateStore(ConstantInt::getFalse(i1Ty), vcc);
  scc = B.CreateAlloca(i1Ty, nullptr, "scc");
  B.CreateStore(ConstantInt::getFalse(i1Ty), scc);
  exec = B.CreateAlloca(execTy, nullptr, "exec");
  B.CreateStore(ConstantInt::getSigned(execTy, -1), exec);
  m0 = B.CreateAlloca(i32Ty, nullptr, "m0");
  B.CreateStore(ConstantInt::get(i32Ty, 0), m0);
  flatScr[0] = B.CreateAlloca(i32Ty, nullptr, "flat_scr_lo");
  flatScr[1] = B.CreateAlloca(i32Ty, nullptr, "flat_scr_hi");

  const unsigned nTTMP = MRI.getRegClass(AMDGPU::TTMP_32RegClassID).getNumRegs();
  ttmp.assign(nTTMP, nullptr);
  for (unsigned i = 0; i < nTTMP; ++i)
    ttmp[i] = B.CreateAlloca(i32Ty, nullptr, "ttmp" + std::to_string(i));
}

void AllocaRegFile::storeSGPR32(IRBuilder<> &B, int idx, Value *v) {
  Type *i32Ty = B.getInt32Ty();
  if (v->getType() != i32Ty)
    v = B.CreateBitCast(v, i32Ty);
  B.CreateStore(v, sgpr[idx]);
}

namespace {

// Shared OOB check for the per-class load/store helpers. Fatal-errors
// instead of returning undef / crashing inside LLVM — a caller that
// hands us an out-of-range `idx` has already produced an invalid
// ParsedReg, which is always a raiser bug rather than a kernel bug.
[[noreturn]] void failOOB(const char *fn, int idx, size_t cap) {
  report_fatal_error(Twine("transpiler: ") + fn + " idx=" + Twine(idx) +
                     " out of range [0.." + Twine(cap) +
                     ") or null; raiser bug, not a kernel bug — the "
                     "caller produced an invalid ParsedReg.");
}

template <typename Bank>
void assertInBank(const char *fn, const Bank &bank, int idx) {
  if (idx < 0 || (unsigned)idx >= bank.size() || !bank[idx])
    failOOB(fn, idx, bank.size());
}

// 64-bit reads touch idx and idx+1; both must be in range.
template <typename Bank>
void assertPairInBank(const char *fn, const Bank &bank, int idx) {
  if (idx < 0 || (unsigned)(idx + 1) >= bank.size() || !bank[idx] ||
      !bank[idx + 1])
    failOOB(fn, idx, bank.size());
}

} // namespace

Value *AllocaRegFile::loadSGPR32(IRBuilder<> &B, int idx) {
  assertInBank("loadSGPR32", sgpr, idx);
  return B.CreateLoad(B.getInt32Ty(), sgpr[idx]);
}

void AllocaRegFile::storeSGPR64(IRBuilder<> &B, int idx, Value *v) {
  assertPairInBank("storeSGPR64", sgpr, idx);
  Type *i32Ty = B.getInt32Ty();
  Type *i64Ty = B.getInt64Ty();
  if (v->getType()->isPointerTy())
    v = B.CreatePtrToInt(v, i64Ty);
  if (v->getType() != i64Ty)
    v = B.CreateBitCast(v, i64Ty);
  Value *lo = B.CreateTrunc(v, i32Ty);
  Value *hi = B.CreateTrunc(B.CreateLShr(v, 32), i32Ty);
  B.CreateStore(lo, sgpr[idx]);
  B.CreateStore(hi, sgpr[idx + 1]);
}

Value *AllocaRegFile::loadSGPR64(IRBuilder<> &B, int idx) {
  assertPairInBank("loadSGPR64", sgpr, idx);
  Type *i32Ty = B.getInt32Ty();
  Type *i64Ty = B.getInt64Ty();
  Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, sgpr[idx]), i64Ty);
  Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, sgpr[idx + 1]), i64Ty);
  return B.CreateOr(lo, B.CreateShl(hi, 32));
}

void AllocaRegFile::storeVGPR32(IRBuilder<> &B, int idx, Value *v) {
  assertInBank("storeVGPR32", vgpr, idx);
  Type *i32Ty = B.getInt32Ty();
  if (v->getType() != i32Ty) {
    if (v->getType()->isPointerTy())
      v = B.CreatePtrToInt(v, B.getInt64Ty());
    if (v->getType() == B.getFloatTy())
      v = B.CreateBitCast(v, i32Ty);
    else if (v->getType() != i32Ty)
      v = B.CreateTrunc(v, i32Ty);
  }
  B.CreateStore(v, vgpr[idx]);
}

Value *AllocaRegFile::loadVGPR32(IRBuilder<> &B, int idx) {
  assertInBank("loadVGPR32", vgpr, idx);
  return B.CreateLoad(B.getInt32Ty(), vgpr[idx]);
}

void AllocaRegFile::storeVGPR64(IRBuilder<> &B, int idx, Value *v) {
  assertPairInBank("storeVGPR64", vgpr, idx);
  Type *i32Ty = B.getInt32Ty();
  Type *i64Ty = B.getInt64Ty();
  if (v->getType()->isPointerTy())
    v = B.CreatePtrToInt(v, i64Ty);
  if (v->getType() != i64Ty)
    v = B.CreateBitCast(v, i64Ty);
  Value *lo = B.CreateTrunc(v, i32Ty);
  Value *hi = B.CreateTrunc(B.CreateLShr(v, 32), i32Ty);
  B.CreateStore(lo, vgpr[idx]);
  B.CreateStore(hi, vgpr[idx + 1]);
}

Value *AllocaRegFile::loadVGPR64(IRBuilder<> &B, int idx) {
  assertPairInBank("loadVGPR64", vgpr, idx);
  Type *i32Ty = B.getInt32Ty();
  Type *i64Ty = B.getInt64Ty();
  Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, vgpr[idx]), i64Ty);
  Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, vgpr[idx + 1]), i64Ty);
  return B.CreateOr(lo, B.CreateShl(hi, 32));
}

void AllocaRegFile::storeAGPR32(IRBuilder<> &B, int idx, Value *v) {
  assertInBank("storeAGPR32", agpr, idx);
  Type *i32Ty = B.getInt32Ty();
  if (v->getType() != i32Ty)
    v = B.CreateBitCast(v, i32Ty);
  B.CreateStore(v, agpr[idx]);
}

Value *AllocaRegFile::loadAGPR32(IRBuilder<> &B, int idx) {
  assertInBank("loadAGPR32", agpr, idx);
  return B.CreateLoad(B.getInt32Ty(), agpr[idx]);
}

void AllocaRegFile::storeVCC(IRBuilder<> &B, Value *v) {
  if (v->getType() != B.getInt1Ty())
    v = B.CreateICmpNE(v, Constant::getNullValue(v->getType()));
  B.CreateStore(v, vcc);
}

Value *AllocaRegFile::loadVCC(IRBuilder<> &B) {
  return B.CreateLoad(B.getInt1Ty(), vcc);
}

void AllocaRegFile::storeSCC(IRBuilder<> &B, Value *v) {
  if (v->getType() != B.getInt1Ty())
    v = B.CreateICmpNE(v, Constant::getNullValue(v->getType()));
  B.CreateStore(v, scc);
}

Value *AllocaRegFile::loadSCC(IRBuilder<> &B) {
  return B.CreateLoad(B.getInt1Ty(), scc);
}

Value *AllocaRegFile::loadExec(IRBuilder<> &B) {
  return B.CreateLoad(execTy, exec, "exec_val");
}

void AllocaRegFile::storeExec(IRBuilder<> &B, Value *v) {
  if (v->getType() != execTy)
    v = B.CreateBitOrPointerCast(v, execTy);
  B.CreateStore(v, exec);
  if (onExecWritten)
    onExecWritten();
}

Value *AllocaRegFile::readVCCAsWaveMask(IRBuilder<> &B, Type *resultTy) {
  assert(projection && "readVCCAsWaveMask requires a WaveProjection — "
                        "call init() before using this reg-file");
  return projection->ballotI1ToWidth(B, loadVCC(B), resultTy, "vcc_ballot");
}

Value *AllocaRegFile::readReg32(IRBuilder<> &B, ParsedReg pr) {
  if (pr.kind == ParsedReg::SGPR) return loadSGPR32(B, pr.baseIdx);
  if (pr.kind == ParsedReg::VGPR) return loadVGPR32(B, pr.baseIdx);
  if (pr.kind == ParsedReg::AGPR) return loadAGPR32(B, pr.baseIdx);
  // VCC as a 32-bit scalar read: must go through the wave-mask ballot,
  // NOT a sign-extension of the local i1. Callers that want a per-lane
  // i1 (e.g. predicating a compute op) should call `loadVCC` directly.
  // This path is hit by handlers that bypass `RaiseContext::readOp32`
  // (which performs the same routing); keeping the routing centralised
  // here prevents silent miscompiles when a new handler adds a direct
  // `regs.readReg32(VCC)` call. See `readVCCAsWaveMask` for the ballot
  // invariant (must be emitted at wave-level / full EXEC).
  if (pr.kind == ParsedReg::VCC)
    return readVCCAsWaveMask(B, B.getInt32Ty());
  if (pr.kind == ParsedReg::EXEC) {
    Value *v = loadExec(B);
    Type *i32Ty = B.getInt32Ty();
    if (v->getType() == i32Ty)
      return v;
    // wave64 EXEC is i64; pick the correct half when reading a 32-bit
    // slice. width==2 reads are handled by readReg64 / readExecWidth;
    // this path is the width==1 case where baseIdx selects LO/HI.
    if (pr.width >= 2)
      return B.CreateTrunc(v, i32Ty, "exec_lo");
    if (pr.baseIdx == 1)
      v = B.CreateLShr(v, 32, "exec_hi_shr");
    return B.CreateTrunc(v, i32Ty, pr.baseIdx == 1 ? "exec_hi" : "exec_lo");
  }
  if (pr.kind == ParsedReg::SCC)
    return B.CreateZExt(loadSCC(B), B.getInt32Ty());
  if (pr.kind == ParsedReg::M0)
    return B.CreateLoad(B.getInt32Ty(), m0, "m0_val");
  if (pr.kind == ParsedReg::FLAT_SCR)
    return B.CreateLoad(B.getInt32Ty(), flatScr[0], "fscr_val");
  if (pr.kind == ParsedReg::TTMP && pr.baseIdx >= 0 &&
      (unsigned)pr.baseIdx < ttmp.size())
    return B.CreateLoad(B.getInt32Ty(), ttmp[pr.baseIdx], "ttmp_val");
  // GFX9 src_lds_direct (encoding 254): reads one dword from LDS at the
  // byte address in M0. There is NO auto-increment of M0 on GFX9 — the
  // kernel manages M0 explicitly between reads. (GFX11+ DSDIR
  // `lds_direct_load` does auto-increment; if we ever raise GFX11+
  // kernels that use DSDIR, the increment must be modeled separately.)
  if (pr.kind == ParsedReg::LDS_DIRECT) {
    auto *i32Ty = B.getInt32Ty();
    Value *addr = B.CreateLoad(i32Ty, m0, "m0_lds_off");
    auto *ldsPtr = PointerType::get(i32Ty->getContext(), 3);
    Value *ptr = B.CreateIntToPtr(addr, ldsPtr, "lds_direct_ptr");
    return B.CreateLoad(i32Ty, ptr, "lds_direct_val");
  }
  failUnhandledKind("readReg32", pr);
}

Value *AllocaRegFile::readReg64(IRBuilder<> &B, ParsedReg pr) {
  if (pr.kind == ParsedReg::SGPR) return loadSGPR64(B, pr.baseIdx);
  if (pr.kind == ParsedReg::VGPR) return loadVGPR64(B, pr.baseIdx);
  // VCC as a 64-bit scalar read: route through the wave-mask ballot.
  // Previous implementations used `SExt(i1 -> i64)`, which replicates
  // the CURRENT LANE's VCC bit across all 64 bits — a silent lie when
  // the consumer expects a wave-level mask (e.g. `s_and_b64 vcc, exec,
  // vcc`). All direct VCC reads must materialise the full per-lane
  // collection via `amdgcn.ballot`.
  if (pr.kind == ParsedReg::VCC)
    return readVCCAsWaveMask(B, B.getInt64Ty());
  if (pr.kind == ParsedReg::EXEC) {
    Value *v = loadExec(B);
    if (v->getType() != B.getInt64Ty())
      v = B.CreateZExt(v, B.getInt64Ty(), "exec_ext");
    return v;
  }
  if (pr.kind == ParsedReg::M0)
    return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), m0, "m0_val"),
                        B.getInt64Ty());
  if (pr.kind == ParsedReg::FLAT_SCR) {
    Type *i32Ty = B.getInt32Ty();
    Type *i64Ty = B.getInt64Ty();
    Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, flatScr[0]), i64Ty);
    Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, flatScr[1]), i64Ty);
    return B.CreateOr(lo, B.CreateShl(hi, 32), "fscr64");
  }
  failUnhandledKind("readReg64", pr);
}

Value *AllocaRegFile::readExecWidth(IRBuilder<> &B) { return loadExec(B); }

void AllocaRegFile::writeExecWidth(IRBuilder<> &B, Value *v) {
  storeExec(B, v);
}

void AllocaRegFile::writeReg32(IRBuilder<> &B, ParsedReg pr, Value *v) {
  if (pr.kind == ParsedReg::SGPR) { storeSGPR32(B, pr.baseIdx, v); return; }
  if (pr.kind == ParsedReg::VGPR) { storeVGPR32(B, pr.baseIdx, v); return; }
  if (pr.kind == ParsedReg::AGPR) { storeAGPR32(B, pr.baseIdx, v); return; }
  if (pr.kind == ParsedReg::EXEC) {
    Type *i32Ty = B.getInt32Ty();
    // Coerce incoming value to i32. storeExec handles width matching to
    // execTy (i32 on wave32 / i64 on wave64). For wave64, a 32-bit write
    // addresses only EXEC_LO or EXEC_HI, so merge with the current value.
    if (v->getType() != i32Ty) {
      if (v->getType()->isPointerTy())
        v = B.CreatePtrToInt(v, B.getInt64Ty());
      if (v->getType() != i32Ty) {
        unsigned bits = v->getType()->getPrimitiveSizeInBits();
        v = bits > 32 ? B.CreateTrunc(v, i32Ty)
                       : B.CreateBitCast(v, i32Ty);
      }
    }
    if (execTy == i32Ty || pr.width >= 2) {
      storeExec(B, v);
    } else {
      Value *cur = loadExec(B);
      Value *v64 = B.CreateZExt(v, execTy);
      Value *merged;
      if (pr.baseIdx == 1) {
        Value *mask = ConstantInt::get(execTy, 0xFFFFFFFFULL);
        merged = B.CreateOr(B.CreateAnd(cur, mask),
                             B.CreateShl(v64, 32), "exec_hi_write");
      } else {
        Value *mask = ConstantInt::get(execTy, 0xFFFFFFFF00000000ULL);
        merged = B.CreateOr(B.CreateAnd(cur, mask), v64, "exec_lo_write");
      }
      storeExec(B, merged);
    }
    return;
  }
  if (pr.kind == ParsedReg::VCC) {
    assert(projection && "writeReg32(VCC) requires a WaveProjection");
    storeVCC(B, projection->extractLaneBitFromWaveMask(B, v));
    return;
  }
  if (pr.kind == ParsedReg::M0) {
    if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
    B.CreateStore(v, m0);
    return;
  }
  if (pr.kind == ParsedReg::FLAT_SCR) {
    if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
    B.CreateStore(v, flatScr[0]);
    return;
  }
  if (pr.kind == ParsedReg::TTMP && pr.baseIdx >= 0 &&
      (unsigned)pr.baseIdx < ttmp.size()) {
    if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
    B.CreateStore(v, ttmp[pr.baseIdx]);
    return;
  }
}

void AllocaRegFile::writeReg64(IRBuilder<> &B, ParsedReg pr, Value *v) {
  if (pr.kind == ParsedReg::SGPR) { storeSGPR64(B, pr.baseIdx, v); return; }
  if (pr.kind == ParsedReg::VGPR) { storeVGPR64(B, pr.baseIdx, v); return; }
  if (pr.kind == ParsedReg::VCC) {
    assert(projection && "writeReg64(VCC) requires a WaveProjection");
    storeVCC(B, projection->extractLaneBitFromWaveMask(B, v));
    return;
  }
  if (pr.kind == ParsedReg::EXEC) {
    storeExec(B, v);
    return;
  }
  if (pr.kind == ParsedReg::FLAT_SCR) {
    Type *i32Ty = B.getInt32Ty();
    Type *i64Ty = B.getInt64Ty();
    if (v->getType() != i64Ty) v = B.CreateBitOrPointerCast(v, i64Ty);
    B.CreateStore(B.CreateTrunc(v, i32Ty), flatScr[0]);
    B.CreateStore(B.CreateTrunc(B.CreateLShr(v, 32), i32Ty), flatScr[1]);
    return;
  }
}

void AllocaRegFile::writeRegExecWidth(IRBuilder<> &B, ParsedReg pr, Value *v) {
  if (pr.kind == ParsedReg::SGPR) {
    if (execTy == B.getInt32Ty())
      storeSGPR32(B, pr.baseIdx, v);
    else
      storeSGPR64(B, pr.baseIdx, v);
    return;
  }
  if (pr.kind == ParsedReg::VCC) {
    assert(projection && "writeRegExecWidth(VCC) requires a WaveProjection");
    storeVCC(B, projection->extractLaneBitFromWaveMask(B, v));
    return;
  }
  if (pr.kind == ParsedReg::EXEC) {
    storeExec(B, v);
    return;
  }
}

Value *AllocaRegFile::readRegVec(IRBuilder<> &B, ParsedReg pr, Type *vecTy) {
  unsigned n = vecTy->isVectorTy()
      ? cast<FixedVectorType>(vecTy)->getNumElements()
      : 1;
  Type *elemTy = vecTy->isVectorTy()
      ? cast<FixedVectorType>(vecTy)->getElementType()
      : vecTy;
  unsigned dwordsPerElem = elemTy->getPrimitiveSizeInBits() / 32;
  if (dwordsPerElem == 0) dwordsPerElem = 1;

  if (n == 1 && !vecTy->isVectorTy() && vecTy->getPrimitiveSizeInBits() <= 32) {
    Value *v = readReg32(B, pr);
    if (v->getType() != vecTy) v = B.CreateBitCast(v, vecTy);
    return v;
  }

  unsigned totalDwords = 0;
  if (elemTy->isFloatTy()) totalDwords = n;
  else if (elemTy->isIntegerTy(32)) totalDwords = n;
  else if (elemTy->isHalfTy()) totalDwords = (n + 1) / 2;
  else totalDwords = (n * elemTy->getPrimitiveSizeInBits() + 31) / 32;

  SmallVector<Value *> dwords;
  for (unsigned i = 0; i < totalDwords; i++) {
    ParsedReg sub = pr;
    sub.baseIdx = pr.baseIdx + i;
    sub.width = 1;
    dwords.push_back(readReg32(B, sub));
  }

  unsigned totalBits = totalDwords * 32;
  Type *intTy = Type::getIntNTy(B.getContext(), totalBits);

  Value *packed = ConstantInt::get(intTy, 0);
  for (unsigned i = 0; i < totalDwords; i++) {
    Value *ext = B.CreateZExt(dwords[i], intTy);
    if (i > 0) ext = B.CreateShl(ext, i * 32);
    packed = B.CreateOr(packed, ext);
  }
  return B.CreateBitCast(packed, vecTy);
}

void AllocaRegFile::writeRegVec(IRBuilder<> &B, ParsedReg pr, Value *v) {
  Type *ty = v->getType();
  unsigned totalBits = ty->getPrimitiveSizeInBits();
  unsigned totalDwords = (totalBits + 31) / 32;

  Type *intTy = Type::getIntNTy(B.getContext(), totalDwords * 32);
  Type *i32Ty = B.getInt32Ty();
  Value *packed = B.CreateBitCast(v, intTy);

  for (unsigned i = 0; i < totalDwords; i++) {
    Value *dw;
    if (i == 0)
      dw = B.CreateTrunc(packed, i32Ty);
    else
      dw = B.CreateTrunc(B.CreateLShr(packed, i * 32), i32Ty);
    ParsedReg sub = pr;
    sub.baseIdx = pr.baseIdx + i;
    sub.width = 1;
    writeReg32(B, sub, dw);
  }
}

void AllocaRegFile::collectAllocas(SmallVectorImpl<AllocaInst *> &out) {
  for (auto *a : sgpr) if (a) out.push_back(a);
  for (auto *a : vgpr) if (a) out.push_back(a);
  for (auto *a : agpr) if (a) out.push_back(a);
  if (vcc) out.push_back(vcc);
  if (scc) out.push_back(scc);
  if (exec) out.push_back(exec);
  if (m0) out.push_back(m0);
  for (auto *a : flatScr) if (a) out.push_back(a);
}

} // namespace transpiler
