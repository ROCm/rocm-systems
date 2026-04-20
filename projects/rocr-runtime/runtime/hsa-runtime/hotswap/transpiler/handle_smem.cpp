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

namespace {

// Look up the SGPR index that holds the *low* dword of the source-ISA
// KernargSegmentPtr at kernel entry, via `ctx.userSgprLayout`.
//
// Why this exists: both the dword-granular S_LOAD_B* block and the
// narrow-SMEM (S_LOAD_U8/I8/U16/I16) block need to answer the same
// question — "is this load's sbase the kernarg pointer?" — in order
// to route through `extractKernargDword` (dword path) or to refuse
// (narrow path). The previous implementation hardcoded `baseIdx == 0`,
// which is only correct when KernargSegmentPtr is the first enabled
// user-SGPR source. That held for Triton kernels (where it is the
// only enabled source) but silently broke as soon as a kernel also
// enabled PrivateSegmentBuffer (4 dwords), DispatchPtr (2 dwords),
// or QueuePtr (2 dwords) ahead of it in the canonical
// enable_sgpr_* order — the kernarg pointer then slides up to
// s[2:3], s[6:7], s[8:9], etc., and every `baseIdx == 0` check
// incorrectly rejects it.
//
// The layout object is the single source of truth for the source
// ISA's user-SGPR ABI. It is populated by
// `UserSgprLayout::fromKernelMeta` (which itself aborts loudly if the
// kernel descriptor is missing, so we never fall back to a guessed
// layout), and wired into `RaiseContext` before handler dispatch in
// raiser.cpp. A null pointer here therefore means the raiser failed
// to wire the layout into the context — a wiring bug, not a runtime
// condition — and we surface it with `report_fatal_error`.
//
// Returns -1 if KernargSegmentPtr is disabled in the KD (no corpus
// kernel today, but the caller must still guard against the `-1`
// match to avoid a false-positive "is kernarg" on negative sbase
// indices).
int getKernargPtrSgpr(RaiseContext &ctx) {
  if (ctx.userSgprLayout == nullptr)
    llvm::report_fatal_error(
        "transpiler: handle_smem: RaiseContext::userSgprLayout is null. "
        "The raiser must populate this before dispatching to handlers; "
        "missing wiring is a bug.");
  return ctx.userSgprLayout->kernargSegmentPtrSgpr;
}

} // namespace

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
    int kernargPtrSgpr = getKernargPtrSgpr(ctx);
    bool isKernarg = (base.kind == ParsedReg::SGPR && kernargPtrSgpr >= 0 &&
                      base.baseIdx == kernargPtrSgpr);

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

  // gfx12+ scalar narrow loads: s_load_{u8,i8,u16,i16}. These fetch 1 or 2
  // bytes from a uniform address materialised in an SGPR-pair base and
  // zero/sign-extend the result into a 32-bit SGPR. MC operand shape
  // matches the dword-granular s_load_* family (sbase + imm-or-sgpr
  // offset), so operand decoding mirrors the S_LOAD_B* block above.
  //
  // Design notes (Position-α permissive lift, 2026-04-19):
  // ------------------------------------------------------
  //  * IR shape: `load iN, ptr addrspace(1) %p, align N` + `zext`/`sext`
  //    to i32 → `storeSGPR32`. No AMDGPU-specific intrinsic exists for
  //    narrow scalar loads; the backend's ISel matches the uniform-address
  //    pattern directly.
  //  * Same-target gfx1250 → gfx1250: the backend re-codegens to the
  //    original `s_load_u16` / `s_load_u8` / etc. (identity-preserving).
  //  * Cross-target gfx1250 → gfx942: the backend has no native narrow
  //    SMEM load, so it lowers to VMEM (`global_load_ushort` / ubyte).
  //    The lifted kernel stays correct — the value appears on every lane
  //    with the same content, matching the SMEM broadcast semantics —
  //    but the register class shifts SGPR→VGPR and the memory path
  //    shifts scalar-cache→vector-cache. We permit this demotion rather
  //    than refuse, because `load iN` IR is a semantically well-defined
  //    lift (no silent miscompile), and the backend's VMEM choice is the
  //    architecturally-correct lowering on an ISA without scalar narrow
  //    loads. Consumers that require SMEM uniformity should gate their
  //    lift pipeline on same-target at the raiser level, not here.
  //  * Alignment: explicit `Align(1)` for byte, `Align(2)` for halfword,
  //    mirroring the principled pattern handle_flat.cpp uses for
  //    GLOBAL_LOAD sub-dword. Omitting this would let the IRBuilder
  //    infer ABI alignment, which happens to match today but is fragile
  //    against LLVM default-alignment changes.
  //  * Kernarg-pointer defensive refusal: a narrow load through the
  //    kernarg pointer (sbase == s[0:1]) is refused loudly. Kernarg
  //    layouts in Salmon's metadata are dword-granular (see
  //    `extractKernargDword` in kernarg_layout.cpp); sub-dword extraction
  //    would require special-case bitfield logic that no corpus kernel
  //    exercises today. If a future kernel does hit this shape, a loud
  //    failure is better than silently decomposing a dword slot.
  //
  // Test back-reference: lit_tests/s_load_u16/ exercises the halfword
  // same-target happy path (`s_load_u16 s*, s*, 0x*` → `load i16 align 2 +
  // zext`). The byte (u8/i8) and signed (i8/i16) variants are covered
  // transitively by the shared handler.
  if (sop == SemOp::S_LOAD_U8 || sop == SemOp::S_LOAD_I8 ||
      sop == SemOp::S_LOAD_U16 || sop == SemOp::S_LOAD_I16) {
    bool isHalfWord =
        (sop == SemOp::S_LOAD_U16 || sop == SemOp::S_LOAD_I16);
    bool isSigned =
        (sop == SemOp::S_LOAD_I8 || sop == SemOp::S_LOAD_I16);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Type *narrowTy = isHalfWord ? i16Ty : ctx.i8Ty;
    Align narrowAlign = Align(isHalfWord ? 2 : 1);
    const char *narrowLoadName = isHalfWord ? "smem_load_h" : "smem_load_b";
    const char *extName = isSigned ? "smem_load_sext" : "smem_load_zext";

    ParsedReg dest = op.dst();
    ParsedReg base = op.srcReg(0);

    // Defensive refusal: narrow load against the kernarg pointer would
    // require sub-dword extraction from a dword-granular kernarg layout.
    // Uses the same layout-driven kernarg-pointer detection as the
    // S_LOAD_B* path above (see `getKernargPtrSgpr` for the rationale);
    // keeping the two call sites identical guarantees that whether a
    // given sbase is treated as "the kernarg pointer" is a single
    // source-ABI question, not two independently-drifted heuristics.
    int kernargPtrSgpr = getKernargPtrSgpr(ctx);
    bool isKernarg = (base.kind == ParsedReg::SGPR && kernargPtrSgpr >= 0 &&
                      base.baseIdx == kernargPtrSgpr);
    if (isKernarg) {
      llvm::errs() << "transpiler: " << di.mnemonic
                   << ": narrow scalar load directly off the kernarg pointer "
                      "is not supported (would need sub-dword extraction from "
                      "the dword-granular kernarg layout)\n";
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SMEM",
          "narrow s_load_* against the kernarg pointer would require "
          "sub-dword extraction from the dword-granular kernarg layout");
      return hr;
    }

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

    Value *narrow = ctx.B.CreateAlignedLoad(narrowTy, ptr, narrowAlign,
                                             narrowLoadName);
    Value *ext = isSigned
                     ? ctx.B.CreateSExt(narrow, ctx.i32Ty, extName)
                     : ctx.B.CreateZExt(narrow, ctx.i32Ty, extName);
    ctx.regs.storeSGPR32(ctx.B, dest.baseIdx, ext);
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
