#include "handlers.hpp"

#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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
      // Materialise the load one dword at a time. Per-dword extraction
      // is the only shape that handles every kernarg layout we see in
      // the corpus uniformly:
      //   * scalar args (i32, i64, ptr) — the helper splits 64-bit args
      //     into low/high dwords as needed (B96 over a (ptr, i32)
      //     pair, etc.);
      //   * `by_value` aggregates with size > 8 (Triton tensor-
      //     descriptor structs, tensilelite kernarg blobs) — the
      //     raiser's per-dword decomposition (raiser.cpp) makes every
      //     interior dword addressable as a standalone i32 slot, so a
      //     B96 load that lands inside such a struct (e.g. offset 8
      //     inside an 80-byte arg) materialises correctly without any
      //     aggregate-aware extract logic in this handler.
      // If any dword in the load can't be served (out-of-range offset,
      // partial-overlap with an unsupported slot type, etc.) we refuse
      // loudly with the helper's diagnostic. The no-fallback rule
      // forbids reading uninitialised SGPRs or substituting zero for
      // a missing dword — a kernel that gets a wrong kernarg byte will
      // compute out-of-bounds GPU addresses, and that is the failure
      // mode this refusal exists to surface.
      //
      // Test back-reference: lit_tests/s_load_b96_kernarg/ exercises
      // the by_value-aggregate path end-to-end with an explicit
      // `s_load_b96 s[0:2], s[0:1], 0x4` over a 16-byte by_value;
      // any change to this loop or to `extractKernargDword` in
      // kernarg_layout.cpp must keep that fixture's IR signature
      // and `phi i32 [ %arg{1,2}, ... ]` data-flow pins green.
      for (int d = 0; d < loadDwords; ++d) {
        int dwordOffset = (int)byteOffset + d * 4;
        std::string why;
        Value *v = extractKernargDword(ctx.kernargs, ctx.B, ctx.kernel,
                                       dwordOffset, &why);
        if (!v) {
          llvm::errs() << "transpiler: " << di.mnemonic
                       << " kernarg load loadBytes=" << loadBytes
                       << " byteOffset=" << byteOffset << " dword=" << d
                       << " (offset=" << dwordOffset << "): " << why << "\n";
          hr.failure = RaiseFailure::smemKernargMiss(di);
          return hr;
        }
        ctx.regs.storeSGPR32(ctx.B, dest.baseIdx + d, v);
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
