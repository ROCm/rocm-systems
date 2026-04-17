#ifndef HOTSWAP_TRANSPILER_REG_FILE_HPP
#define HOTSWAP_TRANSPILER_REG_FILE_HPP

#include "parsed_reg.hpp"
#include "isa_profile.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>
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
  // Width of the target's hardware wave-mask (i32 on wave32, i64 on
  // wave64). Distinct from execTy because execTy tracks the *source*
  // ISA — in cross-arch lifts (e.g. gfx1250 wave32 → gfx942 wave64)
  // the raised EXEC alloca keeps the source width, but any intrinsic
  // that directly names the target's wave mask (amdgcn.ballot and the
  // lane-bit shift below) must match the target width so the AMDGPU
  // backend can select it.
  llvm::Type *waveMaskTy = nullptr;

  // Optional invalidation hook fired on every EXEC-mutating store. The
  // owning RaiseContext installs this so its per-instruction
  // `lane_active` memo (see `RaiseContext::emitLaneActiveBit`) stays in
  // sync regardless of which path (high-level `ctx.storeExec`,
  // `ctx.writeReg*(EXEC, …)`, or the handful of handlers that call
  // `regs.storeExec` / `regs.writeRegExecWidth` directly) ends up
  // hitting `storeExec`. Left null outside of the raiser — the reg file
  // is used in contexts where no memo exists (e.g. register-file-only
  // unit tests or EXEC initialisation at function entry).
  //
  // Using std::function (rather than llvm::function_ref) because the hook
  // must own its callable: the RaiseContext installs it once at the start
  // of raising and the regs outlive any stack-resident lambda that would
  // back a function_ref.
  std::function<void()> onExecWritten;

  void init(llvm::IRBuilder<> &B, llvm::Type *i32Ty, llvm::Type *i1Ty,
            const ISAProfile &isa, const ISAProfile &targetIsa) {
    execTy = isa.isWave32() ? (llvm::Type *)i32Ty : (llvm::Type *)B.getInt64Ty();
    waveMaskTy = targetIsa.isWave32() ? (llvm::Type *)i32Ty
                                      : (llvm::Type *)B.getInt64Ty();
    for (int i = 0; i < MAX_SGPR; i++)
      sgpr[i] = B.CreateAlloca(i32Ty, nullptr, "sgpr" + std::to_string(i));
    for (int i = 0; i < MAX_VGPR; i++)
      vgpr[i] = B.CreateAlloca(i32Ty, nullptr, "vgpr" + std::to_string(i));
    if (isa.hasAGPR) {
      for (int i = 0; i < MAX_VGPR; i++)
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
    B.CreateStore(llvm::ConstantInt::getFalse(i1Ty), vcc);
    scc = B.CreateAlloca(i1Ty, nullptr, "scc");
    B.CreateStore(llvm::ConstantInt::getFalse(i1Ty), scc);
    exec = B.CreateAlloca(execTy, nullptr, "exec");
    B.CreateStore(llvm::ConstantInt::getSigned(execTy, -1), exec);
    m0 = B.CreateAlloca(i32Ty, nullptr, "m0");
    B.CreateStore(llvm::ConstantInt::get(i32Ty, 0), m0);
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

  // Compute the current lane's linear index within the wavefront. Uses
  // amdgcn.mbcnt.lo/hi with an all-ones mask: mbcnt counts set bits
  // strictly below the current lane, and with `-1` as the mask that count
  // equals the lane id.  On wave32 only `mbcnt.lo` is required; on wave64
  // the high-half lanes are folded in by `mbcnt.hi`. Return type is i32.
  // Wave-size choice is based on the *target* ISA (waveMaskTy) because
  // the lane id is a runtime property of the hardware this IR will be
  // compiled for, independent of the source ISA we lifted from.
  llvm::Value *emitLaneIdx(llvm::IRBuilder<> &B) {
    llvm::Module *M = B.GetInsertBlock()->getModule();
    llvm::Type *i32Ty = B.getInt32Ty();
    llvm::Function *mbcntLo = llvm::Intrinsic::getOrInsertDeclaration(
        M, llvm::Intrinsic::amdgcn_mbcnt_lo);
    llvm::Value *allOnes = llvm::ConstantInt::getSigned(i32Ty, -1);
    llvm::Value *zero32 = llvm::ConstantInt::get(i32Ty, 0);
    llvm::Value *laneId =
        B.CreateCall(mbcntLo, {allOnes, zero32}, "lane_lo");
    if (waveMaskTy != i32Ty) {
      llvm::Function *mbcntHi = llvm::Intrinsic::getOrInsertDeclaration(
          M, llvm::Intrinsic::amdgcn_mbcnt_hi);
      laneId = B.CreateCall(mbcntHi, {allOnes, laneId}, "lane_id");
    }
    return laneId;
  }

  // Read VCC as a wave-level bit-mask of width `resultTy`. VCC is stored
  // in this model as a per-lane i1 alloca (each lane holds its own SSA
  // i1 after mem-to-reg promotion). Scalar handlers that consume VCC as
  // a whole-wave mask — e.g. `s_and_b64 vcc, exec, vcc` or any path that
  // needs `v_cndmask_b32_e64`'s mask input — need the cross-lane bitwise
  // projection; emit `llvm.amdgcn.ballot.iN` which is the exact
  // primitive that collects per-lane predicates into a scalar mask (and
  // which the AMDGPU backend lowers back to `v_cmp_* / s_mov_b64 exec`
  // pairs on the hardware side). The ballot's overload type must match
  // the target wave mask (waveMaskTy): the backend only has patterns to
  // select ballot.i32 on wave32 hardware and ballot.i64 on wave64.
  // After the ballot we zext or trunc to the caller's `resultTy`, which
  // is driven by the source instruction's operand width and can differ
  // from the target wave size in cross-arch lifts.
  //
  // INVARIANT (important): `amdgcn.ballot` collects the predicate value
  // of every lane whose hardware EXEC bit is set at the call site;
  // inactive lanes contribute 0. This call must therefore be emitted at
  // the enclosing basic block's "outer" control-flow point, not inside
  // an SPE `emitUnderExec` diamond's `active_bb` (where the backend
  // narrows EXEC to the per-lane-active subset). The raiser upholds
  // this by:
  //   (1) All operand reads route through `RaiseContext::readOp{32,64,
  //       ExecWidth}`, which emit in the *current* basic block before
  //       any per-op `emitUnderExec` wrapping happens — that wrapping
  //       fires only around side-effectful writes.
  //   (2) All direct `regs.readReg{32,64}(VCC)` callers (for SGPR-to-
  //       scalar-mask conversions) are in scalar / uniform code paths
  //       that do not themselves cross a narrow-EXEC boundary.
  // Violating this invariant would silently drop the VCC bits of every
  // lane that happens to be inactive at the ballot call site — a
  // category of bug that would be hard to catch without a round-trip
  // reference. Keep it in mind when adding new handlers.
  llvm::Value *readVCCAsWaveMask(llvm::IRBuilder<> &B, llvm::Type *resultTy) {
    llvm::Module *M = B.GetInsertBlock()->getModule();
    llvm::Value *vccI1 = loadVCC(B);
    llvm::Function *ballot = llvm::Intrinsic::getOrInsertDeclaration(
        M, llvm::Intrinsic::amdgcn_ballot, {waveMaskTy});
    llvm::Value *waveMask = B.CreateCall(ballot, {vccI1}, "vcc_ballot");
    unsigned wantedBits = resultTy->getPrimitiveSizeInBits();
    unsigned waveBits = waveMaskTy->getPrimitiveSizeInBits();
    if (wantedBits == waveBits)
      return waveMask;
    if (wantedBits > waveBits)
      return B.CreateZExt(waveMask, resultTy, "vcc_ballot_ext");
    return B.CreateTrunc(waveMask, resultTy, "vcc_ballot_trunc");
  }

  // Project a wave-level bit-mask back onto the current lane's VCC bit.
  // Inverse of readVCCAsWaveMask for the write direction: scalar
  // handlers that compute a new VCC value as a full-wave iN (e.g.
  // `s_and_b64 vcc, exec, s[4:5]`) need to set each lane's VCC bit to
  // `(waveMask >> lane_id) & 1`. We normalise to waveMaskTy before the
  // shift because lane_id is measured in target-wave terms and the
  // shift amount must stay inside [0, waveMaskBits).
  //
  // Per-lane i1 inputs short-circuit to a direct pass-through: some
  // handlers (v_cmp, carry-out) already produce the final per-lane i1
  // and route through writeReg*(VCC, i1). Those must not be reinterpreted
  // as a one-bit wave mask.
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                         llvm::Value *v) {
    if (v->getType() == B.getInt1Ty())
      return v;
    llvm::Type *i64Ty = B.getInt64Ty();
    if (v->getType()->isPointerTy())
      v = B.CreatePtrToInt(v, i64Ty);
    llvm::Type *targetTy = waveMaskTy;
    unsigned srcBits = v->getType()->getPrimitiveSizeInBits();
    unsigned dstBits = targetTy->getPrimitiveSizeInBits();
    if (srcBits < dstBits)
      v = B.CreateZExt(v, targetTy);
    else if (srcBits > dstBits)
      v = B.CreateTrunc(v, targetTy);
    else if (v->getType() != targetTy)
      v = B.CreateBitCast(v, targetTy);
    llvm::Value *laneIdx = emitLaneIdx(B);
    llvm::Value *laneIdxExt =
        B.CreateZExtOrTrunc(laneIdx, targetTy, "vcc_lane_idx");
    llvm::Value *shifted = B.CreateLShr(v, laneIdxExt, "vcc_at_lane");
    llvm::Value *bit = B.CreateAnd(
        shifted, llvm::ConstantInt::get(targetTy, 1), "vcc_lane_bit");
    return B.CreateICmpNE(bit, llvm::ConstantInt::get(targetTy, 0),
                          "vcc_i1");
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
      llvm::Value *v = loadExec(B);
      if (v->getType() != B.getInt32Ty())
        v = B.CreateTrunc(v, B.getInt32Ty(), "exec_lo");
      return v;
    }
    if (pr.kind == ParsedReg::SCC)
      return B.CreateZExt(loadSCC(B), B.getInt32Ty());
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
    if (onExecWritten)
      onExecWritten();
  }

  llvm::Value *readReg64(llvm::IRBuilder<> &B, ParsedReg pr) {
    if (pr.kind == ParsedReg::SGPR) return loadSGPR64(B, pr.baseIdx);
    if (pr.kind == ParsedReg::VGPR) return loadVGPR64(B, pr.baseIdx);
    // VCC as a 64-bit scalar read: route through the wave-mask ballot.
    // Previous implementations used `SExt(i1 -> i64)`, which replicates
    // the CURRENT LANE's VCC bit across all 64 bits — a silent lie when
    // the consumer expects a wave-level mask (e.g. `s_and_b64 vcc, exec,
    // vcc`). All direct VCC reads must materialise the full per-lane
    // collection via `amdgcn.ballot`. See `readVCCAsWaveMask` for the
    // EXEC-full invariant this relies on.
    if (pr.kind == ParsedReg::VCC)
      return readVCCAsWaveMask(B, B.getInt64Ty());
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
      storeVCC(B, extractLaneBitFromWaveMask(B, v));
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
      storeVCC(B, extractLaneBitFromWaveMask(B, v));
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
      storeVCC(B, extractLaneBitFromWaveMask(B, v));
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
