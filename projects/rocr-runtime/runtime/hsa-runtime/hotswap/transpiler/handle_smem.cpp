#include "handlers.hpp"

#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <utility>
#include <vector>

#define DEBUG_TYPE "transpiler"

using namespace llvm;

namespace transpiler {

HandlerResult handleSMEM(RaiseContext &ctx, const DecodedInst &di,
                       OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  if (sop == SemOp::S_LOAD_B32 || sop == SemOp::S_LOAD_B64 ||
      sop == SemOp::S_LOAD_B96 || sop == SemOp::S_LOAD_B128 ||
      sop == SemOp::S_LOAD_B256 || sop == SemOp::S_LOAD_B512) {
    int loadDwords = 1;
    switch (sop) {
    case SemOp::S_LOAD_B32:
      loadDwords = 1;
      break;
    case SemOp::S_LOAD_B64:
      loadDwords = 2;
      break;
    case SemOp::S_LOAD_B96:
      loadDwords = 3;
      break;
    case SemOp::S_LOAD_B128:
      loadDwords = 4;
      break;
    case SemOp::S_LOAD_B256:
      loadDwords = 8;
      break;
    case SemOp::S_LOAD_B512:
      loadDwords = 16;
      break;
    default:
      break;
    }
    int loadBytes = loadDwords * 4;

    ParsedReg dest = op.dst();
    ParsedReg base = op.srcReg(0);

    unsigned offIdx = op.srcIdx(1);
    bool immOffset = di.isImm(offIdx);
    int64_t byteOffset = immOffset ? op.srcImm(1) : 0;
    bool isKernarg = (base.kind == ParsedReg::SGPR && base.baseIdx == 0);

    LLVM_DEBUG(if (isKernarg && immOffset) {
      llvm::dbgs() << "transpiler: SMEM: mn=" << di.mnemonic
                   << " raw=" << di.rawMnemonic << " full=\"" << di.fullText
                   << "\" off=" << byteOffset << "\n";
    });

    if (isKernarg && immOffset && byteOffset < ctx.kernargs.implicitArgsBase) {
      std::vector<std::pair<int, int>> resolved;
      ctx.kernargs.resolveLoad(byteOffset, loadBytes, resolved);
      if (resolved.empty()) {
        llvm::errs() << "transpiler: Cannot resolve kernarg at offset "
                     << byteOffset << "\n";
        hr.failure = RaiseFailure::smemKernargMiss(di);
        return hr;
      }
      int regOff = 0;
      for (auto &[regWidth, pIdx] : resolved) {
        Value *arg = ctx.kernel->getArg(pIdx);
        if (regWidth == 1)
          ctx.regs.storeSGPR32(ctx.B, dest.baseIdx + regOff, arg);
        else if (regWidth == 2)
          ctx.regs.storeSGPR64(ctx.B, dest.baseIdx + regOff, arg);
        regOff += regWidth;
      }
    } else if (isKernarg && immOffset) {
      int implOffset = byteOffset - ctx.kernargs.implicitArgsBase;
      LLVM_DEBUG(llvm::dbgs() << "transpiler: implicit kernarg load: byteOffset="
                              << byteOffset
                              << " implicitArgsBase=" << ctx.kernargs.implicitArgsBase
                              << " implOffset=" << implOffset << "\n");
      Function *fnImplicitArgPtr = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_implicitarg_ptr);
      Value *implPtr =
          ctx.B.CreateCall(fnImplicitArgPtr, {}, "implicitarg_ptr");
      Value *gep = ctx.B.CreateInBoundsGEP(ctx.i8Ty, implPtr,
                                           ctx.B.getInt64(implOffset), "impl_gep");
      ctx.regs.storeSGPR32(ctx.B, dest.baseIdx,
                           ctx.B.CreateLoad(ctx.i32Ty, gep, "impl_load"));
    } else {
      Value *baseAddr = ctx.regs.loadSGPR64(ctx.B, base.baseIdx);
      Value *ptr = ctx.B.CreateIntToPtr(baseAddr, ctx.ptrGlobalTy);
      if (immOffset) {
        if (byteOffset != 0)
          ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, ctx.B.getInt64(byteOffset));
      } else {
        Value *regOff = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
        ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, regOff);
      }
      for (int d = 0; d < loadDwords; d++) {
        Value *ep = (d == 0) ? ptr
                             : ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr,
                                                       ctx.B.getInt64(d * 4));
        ctx.regs.storeSGPR32(ctx.B, dest.baseIdx + d,
                             ctx.B.CreateLoad(ctx.i32Ty, ep, "smem_load"));
      }
    }
    hr.handled = true;
    return hr;
  }

  // s_store_* (scalar store through SGPR base + imm/sgpr offset).
  // MC operand layout: (sdata, sbase, soffset/imm, cpol).
  if (sop == SemOp::S_STORE_B32 || sop == SemOp::S_STORE_B64 ||
      sop == SemOp::S_STORE_B128) {
    int storeDwords = (sop == SemOp::S_STORE_B32)  ? 1
                      : (sop == SemOp::S_STORE_B64) ? 2
                                                    : 4;
    ParsedReg data = op.srcReg(0);
    ParsedReg base = op.srcReg(1);
    if (data.kind != ParsedReg::SGPR || base.kind != ParsedReg::SGPR) {
      llvm::errs() << "transpiler: " << di.mnemonic
                   << ": S_STORE expects SGPR data and base\n";
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SMEM", "S_STORE expects SGPR data and base");
      return hr;
    }
    Value *baseAddr = ctx.regs.loadSGPR64(ctx.B, base.baseIdx);
    Value *ptr = ctx.B.CreateIntToPtr(baseAddr, ctx.ptrGlobalTy);
    if (op.nSrcs() >= 3) {
      unsigned offIdx = op.srcIdx(2);
      if (di.isImm(offIdx)) {
        int64_t off = op.srcImm(2);
        if (off != 0)
          ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, ctx.B.getInt64(off));
      } else if (di.isReg(offIdx)) {
        Value *regOff = ctx.B.CreateZExt(op.src(2), ctx.i64Ty);
        ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, regOff);
      }
    }
    for (int d = 0; d < storeDwords; d++) {
      Value *ep = (d == 0) ? ptr
                           : ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr,
                                                     ctx.B.getInt64(d * 4));
      Value *v = ctx.regs.loadSGPR32(ctx.B, data.baseIdx + d);
      ctx.B.CreateStore(v, ep);
    }
    hr.handled = true;
    return hr;
  }

  // s_atomic_swap: atomic exchange on memory through SGPR pair base pointer.
  // HW only returns the old value when GLC=1; we always write it back,
  // which is conservative (harmless when GLC=0 since the dest is dead).
  if (sop == SemOp::S_ATOMIC_SWAP) {
    assert(((di.tsFlags & SIInstrFlags::IsAtomicRet) != 0) == (di.numDefs > 0) &&
           "s_atomic_swap: IsAtomicRet disagrees with numDefs");
    ParsedReg dataDst = op.dst();
    ParsedReg base = op.srcReg(0);
    Value *data = ctx.regs.readReg32(ctx.B, dataDst);

    Value *baseAddr = ctx.regs.loadSGPR64(ctx.B, base.baseIdx);
    Value *ptr = ctx.B.CreateIntToPtr(baseAddr, ctx.ptrGlobalTy);

    unsigned offIdx = op.srcIdx(1);
    if (di.isImm(offIdx)) {
      int64_t off = op.srcImm(1);
      if (off != 0)
        ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, ctx.B.getInt64(off));
    } else {
      Value *regOff = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
      ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, regOff);
    }

    Value *old = ctx.B.CreateAtomicRMW(AtomicRMWInst::Xchg, ptr, data,
                                       MaybeAlign(), AtomicOrdering::Monotonic);
    ctx.regs.storeSGPR32(ctx.B, dataDst.baseIdx, old);
    hr.handled = true;
    return hr;
  }

  return hr;
}

} // namespace transpiler
