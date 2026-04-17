#ifndef HOTSWAP_TRANSPILER_REG_FILE_HPP
#define HOTSWAP_TRANSPILER_REG_FILE_HPP

#include "parsed_reg.hpp"
#include "isa_profile.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include <string>

namespace transpiler {

struct AllocaRegFile {
  static constexpr int MAX_SGPR = 106;
  static constexpr int MAX_VGPR = 512;
  llvm::AllocaInst *sgpr[106] = {};
  llvm::AllocaInst *vgpr[512] = {};
  llvm::AllocaInst *agpr[512] = {};
  llvm::AllocaInst *vcc = nullptr;
  llvm::AllocaInst *scc = nullptr;
  llvm::AllocaInst *exec = nullptr;
  llvm::AllocaInst *m0 = nullptr;
  llvm::AllocaInst *flatScr[2] = {};
  static constexpr int MAX_TTMP = 16;
  llvm::AllocaInst *ttmp[16] = {};
  llvm::Type *execTy = nullptr;

  void init(llvm::IRBuilder<> &B, llvm::Type *i32Ty, llvm::Type *i1Ty, const ISAProfile &isa) {
    execTy = isa.isWave32() ? (llvm::Type *)i32Ty : (llvm::Type *)B.getInt64Ty();
    for (int i = 0; i < MAX_SGPR; i++)
      sgpr[i] = B.CreateAlloca(i32Ty, nullptr, "sgpr" + std::to_string(i));
    for (int i = 0; i < MAX_VGPR; i++)
      vgpr[i] = B.CreateAlloca(i32Ty, nullptr, "vgpr" + std::to_string(i));
    if (isa.hasAGPR) {
      for (int i = 0; i < MAX_VGPR; i++)
        agpr[i] = B.CreateAlloca(i32Ty, nullptr, "agpr" + std::to_string(i));
    }
    vcc = B.CreateAlloca(i1Ty, nullptr, "vcc");
    scc = B.CreateAlloca(i1Ty, nullptr, "scc");
    exec = B.CreateAlloca(execTy, nullptr, "exec");
    B.CreateStore(llvm::ConstantInt::getSigned(execTy, -1), exec);
    m0 = B.CreateAlloca(i32Ty, nullptr, "m0");
    flatScr[0] = B.CreateAlloca(i32Ty, nullptr, "flat_scr_lo");
    flatScr[1] = B.CreateAlloca(i32Ty, nullptr, "flat_scr_hi");
    for (int i = 0; i < MAX_TTMP; i++)
      ttmp[i] = B.CreateAlloca(i32Ty, nullptr, "ttmp" + std::to_string(i));
  }

  void storeSGPR32(llvm::IRBuilder<> &B, int idx, llvm::Value *v) {
    llvm::Type *i32Ty = B.getInt32Ty();
    if (v->getType() != i32Ty)
      v = B.CreateBitCast(v, i32Ty);
    B.CreateStore(v, sgpr[idx]);
  }
  llvm::Value *loadSGPR32(llvm::IRBuilder<> &B, int idx) {
    if (idx < 0 || idx >= MAX_SGPR || !sgpr[idx]) {
      llvm::errs() << "transpiler: BUG: loadSGPR32 idx=" << idx << " out of range or null\n";
      return llvm::UndefValue::get(B.getInt32Ty());
    }
    return B.CreateLoad(B.getInt32Ty(), sgpr[idx]);
  }
  void storeSGPR64(llvm::IRBuilder<> &B, int idx, llvm::Value *v) {
    llvm::Type *i32Ty = B.getInt32Ty();
    llvm::Type *i64Ty = B.getInt64Ty();
    if (v->getType()->isPointerTy())
      v = B.CreatePtrToInt(v, i64Ty);
    if (v->getType() != i64Ty)
      v = B.CreateBitCast(v, i64Ty);
    llvm::Value *lo = B.CreateTrunc(v, i32Ty);
    llvm::Value *hi = B.CreateTrunc(B.CreateLShr(v, 32), i32Ty);
    B.CreateStore(lo, sgpr[idx]);
    B.CreateStore(hi, sgpr[idx + 1]);
  }
  llvm::Value *loadSGPR64(llvm::IRBuilder<> &B, int idx) {
    llvm::Type *i32Ty = B.getInt32Ty();
    llvm::Type *i64Ty = B.getInt64Ty();
    llvm::Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, sgpr[idx]), i64Ty);
    llvm::Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, sgpr[idx + 1]), i64Ty);
    return B.CreateOr(lo, B.CreateShl(hi, 32));
  }
  void storeVGPR32(llvm::IRBuilder<> &B, int idx, llvm::Value *v) {
    llvm::Type *i32Ty = B.getInt32Ty();
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
  llvm::Value *loadVGPR32(llvm::IRBuilder<> &B, int idx) {
    return B.CreateLoad(B.getInt32Ty(), vgpr[idx]);
  }
  void storeVGPR64(llvm::IRBuilder<> &B, int idx, llvm::Value *v) {
    llvm::Type *i32Ty = B.getInt32Ty();
    llvm::Type *i64Ty = B.getInt64Ty();
    if (v->getType()->isPointerTy())
      v = B.CreatePtrToInt(v, i64Ty);
    if (v->getType() != i64Ty)
      v = B.CreateBitCast(v, i64Ty);
    llvm::Value *lo = B.CreateTrunc(v, i32Ty);
    llvm::Value *hi = B.CreateTrunc(B.CreateLShr(v, 32), i32Ty);
    B.CreateStore(lo, vgpr[idx]);
    B.CreateStore(hi, vgpr[idx + 1]);
  }
  llvm::Value *loadVGPR64(llvm::IRBuilder<> &B, int idx) {
    llvm::Type *i32Ty = B.getInt32Ty();
    llvm::Type *i64Ty = B.getInt64Ty();
    llvm::Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, vgpr[idx]), i64Ty);
    llvm::Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, vgpr[idx + 1]), i64Ty);
    return B.CreateOr(lo, B.CreateShl(hi, 32));
  }
  void storeAGPR32(llvm::IRBuilder<> &B, int idx, llvm::Value *v) {
    llvm::Type *i32Ty = B.getInt32Ty();
    if (v->getType() != i32Ty)
      v = B.CreateBitCast(v, i32Ty);
    B.CreateStore(v, agpr[idx]);
  }
  llvm::Value *loadAGPR32(llvm::IRBuilder<> &B, int idx) {
    return B.CreateLoad(B.getInt32Ty(), agpr[idx]);
  }
  void storeVCC(llvm::IRBuilder<> &B, llvm::Value *v) {
    if (v->getType() != B.getInt1Ty())
      v = B.CreateICmpNE(v, llvm::Constant::getNullValue(v->getType()));
    B.CreateStore(v, vcc);
  }
  llvm::Value *loadVCC(llvm::IRBuilder<> &B) {
    return B.CreateLoad(B.getInt1Ty(), vcc);
  }
  void storeSCC(llvm::IRBuilder<> &B, llvm::Value *v) {
    if (v->getType() != B.getInt1Ty())
      v = B.CreateICmpNE(v, llvm::Constant::getNullValue(v->getType()));
    B.CreateStore(v, scc);
  }
  llvm::Value *loadSCC(llvm::IRBuilder<> &B) {
    return B.CreateLoad(B.getInt1Ty(), scc);
  }

  llvm::Value *readReg32(llvm::IRBuilder<> &B, ParsedReg pr) {
    if (pr.kind == ParsedReg::SGPR) return loadSGPR32(B, pr.baseIdx);
    if (pr.kind == ParsedReg::VGPR) return loadVGPR32(B, pr.baseIdx);
    if (pr.kind == ParsedReg::AGPR) return loadAGPR32(B, pr.baseIdx);
    if (pr.kind == ParsedReg::M0) return B.CreateLoad(B.getInt32Ty(), m0, "m0_val");
    if (pr.kind == ParsedReg::FLAT_SCR) return B.CreateLoad(B.getInt32Ty(), flatScr[0], "fscr_val");
    if (pr.kind == ParsedReg::TTMP && pr.baseIdx >= 0 && pr.baseIdx < MAX_TTMP)
      return B.CreateLoad(B.getInt32Ty(), ttmp[pr.baseIdx], "ttmp_val");
    // GFX9 src_lds_direct (encoding 254): reads one dword from LDS at the
    // byte address in M0.  There is NO auto-increment of M0 on GFX9 — the
    // kernel manages M0 explicitly between reads.  (GFX11+ DSDIR
    // `lds_direct_load` does auto-increment; if we ever raise GFX11+
    // kernels that use DSDIR, the increment must be modeled separately.)
    if (pr.kind == ParsedReg::LDS_DIRECT) {
      auto *i32Ty = B.getInt32Ty();
      llvm::Value *addr = B.CreateLoad(i32Ty, m0, "m0_lds_off");
      auto *ldsPtr = llvm::PointerType::get(i32Ty->getContext(), 3);
      llvm::Value *ptr = B.CreateIntToPtr(addr, ldsPtr, "lds_direct_ptr");
      return B.CreateLoad(i32Ty, ptr, "lds_direct_val");
    }
    return nullptr;
  }
  llvm::Value *loadExec(llvm::IRBuilder<> &B) {
    return B.CreateLoad(execTy, exec, "exec_val");
  }
  void storeExec(llvm::IRBuilder<> &B, llvm::Value *v) {
    if (v->getType() != execTy)
      v = B.CreateBitOrPointerCast(v, execTy);
    B.CreateStore(v, exec);
  }

  llvm::Value *readReg64(llvm::IRBuilder<> &B, ParsedReg pr) {
    if (pr.kind == ParsedReg::SGPR) return loadSGPR64(B, pr.baseIdx);
    if (pr.kind == ParsedReg::VGPR) return loadVGPR64(B, pr.baseIdx);
    if (pr.kind == ParsedReg::VCC)
      return B.CreateSExt(loadVCC(B), B.getInt64Ty());
    if (pr.kind == ParsedReg::EXEC) {
      llvm::Value *v = loadExec(B);
      if (v->getType() != B.getInt64Ty())
        v = B.CreateZExt(v, B.getInt64Ty(), "exec_ext");
      return v;
    }
    if (pr.kind == ParsedReg::M0)
      return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), m0, "m0_val"), B.getInt64Ty());
    if (pr.kind == ParsedReg::FLAT_SCR) {
      llvm::Type *i32Ty = B.getInt32Ty();
      llvm::Type *i64Ty = B.getInt64Ty();
      llvm::Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, flatScr[0]), i64Ty);
      llvm::Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, flatScr[1]), i64Ty);
      return B.CreateOr(lo, B.CreateShl(hi, 32), "fscr64");
    }
    return nullptr;
  }

  llvm::Value *readExecWidth(llvm::IRBuilder<> &B) {
    return loadExec(B);
  }
  void writeExecWidth(llvm::IRBuilder<> &B, llvm::Value *v) {
    storeExec(B, v);
  }
  void writeReg32(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v) {
    if (pr.kind == ParsedReg::SGPR) storeSGPR32(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::VGPR) storeVGPR32(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::AGPR) storeAGPR32(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::EXEC) storeExec(B, v);
    else if (pr.kind == ParsedReg::VCC) {
      storeVCC(B, B.CreateICmpNE(v, llvm::Constant::getNullValue(v->getType())));
    }
    else if (pr.kind == ParsedReg::M0) {
      if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
      B.CreateStore(v, m0);
    }
    else if (pr.kind == ParsedReg::FLAT_SCR) {
      if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
      B.CreateStore(v, flatScr[0]);
    }
    else if (pr.kind == ParsedReg::TTMP && pr.baseIdx >= 0 && pr.baseIdx < MAX_TTMP) {
      if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
      B.CreateStore(v, ttmp[pr.baseIdx]);
    }
  }
  void writeReg64(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v) {
    if (pr.kind == ParsedReg::SGPR) storeSGPR64(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::VGPR) storeVGPR64(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::VCC) {
      storeVCC(B, B.CreateICmpNE(v, llvm::Constant::getNullValue(v->getType())));
    }
    else if (pr.kind == ParsedReg::EXEC) {
      storeExec(B, v);
    }
    else if (pr.kind == ParsedReg::FLAT_SCR) {
      llvm::Type *i32Ty = B.getInt32Ty();
      llvm::Type *i64Ty = B.getInt64Ty();
      if (v->getType() != i64Ty) v = B.CreateBitOrPointerCast(v, i64Ty);
      B.CreateStore(B.CreateTrunc(v, i32Ty), flatScr[0]);
      B.CreateStore(B.CreateTrunc(B.CreateLShr(v, 32), i32Ty), flatScr[1]);
    }
  }

  void writeRegExecWidth(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v) {
    if (pr.kind == ParsedReg::SGPR) {
      if (execTy == B.getInt32Ty())
        storeSGPR32(B, pr.baseIdx, v);
      else
        storeSGPR64(B, pr.baseIdx, v);
    } else if (pr.kind == ParsedReg::VCC) {
      storeVCC(B, B.CreateICmpNE(v, llvm::Constant::getNullValue(v->getType())));
    } else if (pr.kind == ParsedReg::EXEC) {
      storeExec(B, v);
    }
  }

  // Read/write N dwords as a vector from contiguous VGPRs/AGPRs
  llvm::Value *readRegVec(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Type *vecTy) {
    unsigned n = vecTy->isVectorTy()
        ? llvm::cast<llvm::FixedVectorType>(vecTy)->getNumElements()
        : 1;
    llvm::Type *elemTy = vecTy->isVectorTy()
        ? llvm::cast<llvm::FixedVectorType>(vecTy)->getElementType()
        : vecTy;
    unsigned dwordsPerElem = elemTy->getPrimitiveSizeInBits() / 32;
    if (dwordsPerElem == 0) dwordsPerElem = 1;

    if (n == 1 && !vecTy->isVectorTy() && vecTy->getPrimitiveSizeInBits() <= 32) {
      llvm::Value *v = readReg32(B, pr);
      if (v->getType() != vecTy) v = B.CreateBitCast(v, vecTy);
      return v;
    }

    unsigned totalDwords = 0;
    if (elemTy->isFloatTy()) totalDwords = n;
    else if (elemTy->isIntegerTy(32)) totalDwords = n;
    else if (elemTy->isHalfTy()) totalDwords = (n + 1) / 2;
    else totalDwords = (n * elemTy->getPrimitiveSizeInBits() + 31) / 32;

    // Load all dwords
    llvm::SmallVector<llvm::Value *, 16> dwords;
    for (unsigned i = 0; i < totalDwords; i++) {
      ParsedReg sub = pr;
      sub.baseIdx = pr.baseIdx + i;
      sub.width = 1;
      dwords.push_back(readReg32(B, sub));
    }

    // Bitcast the dwords into the target vector type
    llvm::Type *i32Ty = B.getInt32Ty();
    unsigned totalBits = totalDwords * 32;
    llvm::Type *intTy = llvm::Type::getIntNTy(B.getContext(), totalBits);

    llvm::Value *packed = llvm::ConstantInt::get(intTy, 0);
    for (unsigned i = 0; i < totalDwords; i++) {
      llvm::Value *ext = B.CreateZExt(dwords[i], intTy);
      if (i > 0) ext = B.CreateShl(ext, i * 32);
      packed = B.CreateOr(packed, ext);
    }
    return B.CreateBitCast(packed, vecTy);
  }

  void writeRegVec(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v) {
    llvm::Type *ty = v->getType();
    unsigned totalBits = ty->getPrimitiveSizeInBits();
    unsigned totalDwords = (totalBits + 31) / 32;

    llvm::Type *intTy = llvm::Type::getIntNTy(B.getContext(), totalDwords * 32);
    llvm::Type *i32Ty = B.getInt32Ty();
    llvm::Value *packed = B.CreateBitCast(v, intTy);

    for (unsigned i = 0; i < totalDwords; i++) {
      llvm::Value *dw;
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

  void collectAllocas(llvm::SmallVectorImpl<llvm::AllocaInst *> &out) {
    for (auto *a : sgpr) if (a) out.push_back(a);
    for (auto *a : vgpr) if (a) out.push_back(a);
    for (auto *a : agpr) if (a) out.push_back(a);
    if (vcc) out.push_back(vcc);
    if (scc) out.push_back(scc);
    if (exec) out.push_back(exec);
    if (m0) out.push_back(m0);
    for (auto *a : flatScr) if (a) out.push_back(a);
  }
};

} // namespace transpiler

#endif
