#include "flat_addr.hpp"
#include "handlers.hpp"

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
                        OpResolver &op) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  if (sop == SemOp::GLOBAL_LOAD_USHORT || sop == SemOp::GLOBAL_LOAD_SHORT_D16_HI ||
      sop == SemOp::GLOBAL_LOAD_SSHORT || sop == SemOp::GLOBAL_LOAD_UBYTE ||
      sop == SemOp::GLOBAL_LOAD_SBYTE) {
    ParsedReg dest = op.dst();
    bool isByte = sop == SemOp::GLOBAL_LOAD_UBYTE || sop == SemOp::GLOBAL_LOAD_SBYTE;
    Type *loadTy = isByte ? ctx.i8Ty : Type::getInt16Ty(ctx.C);
    // The ISA only guarantees natural alignment for each sub-dword access:
    // 1 byte for byte loads, 2 bytes for short loads. Using the ABI default
    // alignment here was over-promising for buffers legitimately aligned
    // to the element size.
    Align loadAlign = Align(isByte ? 1 : 2);

    FlatAddr fa = decodeGlobalLoadAddr(ctx, di, op, isByte ? 1 : 2,
                                        "GLOBAL_LOAD sub-dword");
    Value *addr = fa.ptr;
    Value *loaded = ctx.B.CreateAlignedLoad(loadTy, addr, loadAlign, "gload_sub");
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

    FlatAddr fa = decodeGlobalLoadAddr(ctx, di, op, loadDwords * 4,
                                        "GLOBAL_LOAD dword");
    Value *addr = fa.ptr;

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

    // scale_offset on stores scales the per-lane vaddr by the access
    // element size. For sub-dword stores (byte/short) the element size
    // is 1 or 2 bytes; for dword/dwordx{2,3,4} it is 4, 8, 12, or 16
    // bytes — the compiler emits `global_store_dwordx4 … scale_offset`
    // with a lane-index vaddr to lower `out[tid] = vec4` patterns.
    int elemBytes = storeBits < 32 ? (storeBits / 8)
                                    : std::max(storeDwords, 1) * 4;
    FlatAddr fa =
        decodeGlobalStoreAddr(ctx, di, op, elemBytes, "GLOBAL_STORE");
    Value *addr = fa.ptr;
    ParsedReg stData = fa.stData;

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

  // ---------------------------------------------------------------------
  // gfx1250 async global → LDS load (FLAT VFLAT 0x5f-0x62 — b8 / b32 /
  // b64 / b128).
  //
  // Operand layout from `FLAT_Global_Load_LDS_Pseudo<…, IsAsync=1>`
  // (FLATInstructions.td:391-417) is identical across all four widths
  // (only the data byte size — and so the intrinsic ID — varies):
  //
  //   plain (4 srcs): vdst:VGPR_32, vaddr:VGPR_64,            offset, cpol
  //   SADDR (5 srcs): vdst:VGPR_32, saddr:SReg_64, vaddr:VGPR_32, offset, cpol
  //
  // `vdst` is in the *input* list (because `has_vdst = IsAsync = 1`)
  // and carries the per-lane LDS i32 base offset. The intrinsics
  // `int_amdgcn_global_load_async_to_lds_b{8,32,64,128}`
  // (IntrinsicsAMDGPU.td:3939-3946) all share signature
  // `AMDGPUAsyncGlobalLoadToLDS` (line 3904) and consume the LDS
  // pointer as `local_ptr_ty`, so we materialise it via `inttoptr i32
  // → ptr addrspace(3)`. Each lane fires its own write so the call
  // is wrapped in `emitUnderExec`; the intrinsic's
  // `IntrInaccessibleMemOrArgMemOnly` attribute keeps later passes
  // from sinking it across companion `s_wait_asynccnt` barriers.
  //
  // gfx942 has no asynchronous global→LDS DMA channel, so a cross-
  // target lift (gfx1250 → gfx942) is refused loudly. See the
  // SemOp's docstring in `semop.hpp` for the design rationale.
  if (sop == SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B8 ||
      sop == SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B32 ||
      sop == SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B64 ||
      sop == SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B128) {
    if (!ctx.targetIsa.hasTensorOps) {
      llvm::errs()
          << "transpiler: FLAT: " << di.mnemonic
          << " has no equivalent on the compilation target "
          << "(gfx1250 asynccnt unit; LLVM intrinsic "
          << "amdgcn.global.load.async.to.lds.b{8,32,64,128} is gated "
          << "by FeatureGFX1250Insts). A synthesised "
          << "global_load + ds_write pair would alter the wave's "
          << "memory ordering and asynccnt observable state — "
          << "refusing to emit a fallback.\n";
      hr.failure = RaiseFailure::unsupportedShape(
          di, "FLAT",
          "gfx1250-only async global→LDS DMA; no equivalent on "
          "non-gfx1250 compilation target (no asynccnt unit, no "
          "amdgcn.global.load.async.to.lds.b* intrinsic)");
      return hr;
    }

    bool isSaddr = false;
    if (op.nSrcs() == 5) {
      isSaddr = true;
    } else if (op.nSrcs() != 4) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "FLAT",
          "global_load_async_to_lds_b*: expected 4 srcs (plain) or "
          "5 srcs (SADDR) per FLAT_Global_Load_LDS_Pseudo<IsAsync=1>");
      return hr;
    }

    ParsedReg vdstPr = op.srcReg(0);
    if (vdstPr.kind != ParsedReg::VGPR) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "FLAT",
          "global_load_async_to_lds_b*: vdst (LDS-base operand) is "
          "not a VGPR");
      return hr;
    }
    Value *ldsOff = ctx.regs.readReg32(ctx.B, vdstPr);
    Type *ptrLDSTy = PointerType::get(ctx.C, /*addrspace=*/3);
    Value *ldsPtr = ctx.B.CreateIntToPtr(ldsOff, ptrLDSTy, "lds_ptr");

    Value *globalAddr = nullptr;
    unsigned immStart = 0;
    if (isSaddr) {
      ParsedReg saddrPr = op.srcReg(1);
      ParsedReg vaddrPr = op.srcReg(2);
      if (saddrPr.kind != ParsedReg::SGPR ||
          vaddrPr.kind != ParsedReg::VGPR) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "FLAT",
            "global_load_async_to_lds_b* SADDR: expected "
            "(SGPR_64, VGPR_32) for (saddr, vaddr)");
        return hr;
      }
      Value *saddr = ctx.regs.readReg64(ctx.B, saddrPr);
      Value *voff = ctx.B.CreateZExt(
          ctx.regs.readReg32(ctx.B, vaddrPr), ctx.i64Ty, "voff_zext");
      globalAddr = ctx.B.CreateAdd(saddr, voff, "saddr_vaddr");
      immStart = 3;
    } else {
      ParsedReg vaddrPr = op.srcReg(1);
      if (vaddrPr.kind != ParsedReg::VGPR) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "FLAT",
            "global_load_async_to_lds_b* plain: expected VGPR_64 "
            "for vaddr");
        return hr;
      }
      globalAddr = ctx.regs.readReg64(ctx.B, vaddrPr);
      immStart = 2;
    }

    int64_t flatOffset = 0;
    int64_t cpolImm = 0;
    bool sawOffset = false;
    for (unsigned k = immStart; k < op.nSrcs(); ++k) {
      if (!di.isImm(op.srcIdx(k))) continue;
      int64_t v = di.getImm(op.srcIdx(k));
      if (!sawOffset) {
        flatOffset = v;
        sawOffset = true;
      } else {
        cpolImm = v;
      }
    }

    Value *globalPtr = globalAddr;
    if (globalPtr->getType() != ctx.ptrGlobalTy)
      globalPtr = ctx.B.CreateIntToPtr(globalPtr, ctx.ptrGlobalTy);

    Intrinsic::ID iid;
    switch (sop) {
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B8:
      iid = Intrinsic::amdgcn_global_load_async_to_lds_b8; break;
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B32:
      iid = Intrinsic::amdgcn_global_load_async_to_lds_b32; break;
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B64:
      iid = Intrinsic::amdgcn_global_load_async_to_lds_b64; break;
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B128:
      iid = Intrinsic::amdgcn_global_load_async_to_lds_b128; break;
    default:
      llvm_unreachable("dispatch matched async-to-LDS family but width SemOp "
                       "fell through the switch");
    }
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, iid);
    Value *offsetArg = ConstantInt::get(ctx.i32Ty, flatOffset);
    Value *cpolArg = ConstantInt::get(ctx.i32Ty, cpolImm);
    ctx.emitUnderExec([&] {
      ctx.B.CreateCall(fn, {globalPtr, ldsPtr, offsetArg, cpolArg});
    });

    hr.handled = true;
    return hr;
  }

  // ---------------------------------------------------------------------
  // gfx1250 FLAT VMEM prefetch (VFLAT 0x05D — global_prefetch_b8).
  //
  // Operand layout from `FLAT_Prefetch_Pseudo` (FLATInstructions.td
  // :525-553) — note `has_vdst = 0`, so there is no dst slot:
  //
  //   plain (3 srcs): vaddr:VGPR_64,            offset, cpol
  //   SADDR (4 srcs): saddr:SReg_64, vaddr:VGPR_32, offset, cpol
  //
  // Lifts to `int_amdgcn_global_prefetch(globalPtr, cpol)`
  // (IntrinsicsAMDGPU.td:3211); the FLAT `flat_offset` is folded
  // onto the pointer via a non-inbounds GEP before the call (the
  // intrinsic itself takes no offset operand). The call sits
  // OUTSIDE `emitUnderExec` because the intrinsic carries the EXEC
  // mask implicitly through `IntrInaccessibleMemOrArgMemOnly` — a
  // hint with no observable side effect on inactive lanes, so an
  // extra `if-spe-active` guard would gratuitously inflate IR for
  // what hardware executes as a single broadcast hint.
  //
  // gfx942 has no VMEM-prefetch encoding (the intrinsic is gated by
  // `HasVmemPrefInsts`, only set on gfx1250+), so a cross-target
  // lift is refused loudly. See the SemOp's docstring in
  // `semop.hpp` for the design rationale.
  if (sop == SemOp::GLOBAL_PREFETCH_B8) {
    if (!ctx.targetIsa.hasTensorOps) {
      llvm::errs()
          << "transpiler: FLAT: " << di.mnemonic
          << " has no equivalent on the compilation target "
          << "(gfx1250 VMEM-prefetch unit; LLVM intrinsic "
          << "amdgcn.global.prefetch is gated by HasVmemPrefInsts, "
          << "only set on gfx1250+). The closest sibling "
          << "amdgcn.s.prefetch.data requires a uniform SGPR "
          << "pointer which we cannot prove for the divergent "
          << "VGPR address used here without divergence "
          << "analysis — refusing to emit a fallback or silently "
          << "drop the hint.\n";
      hr.failure = RaiseFailure::unsupportedShape(
          di, "FLAT",
          "gfx1250-only VMEM prefetch (HasVmemPrefInsts); no "
          "equivalent on non-gfx1250 compilation target. The "
          "amdgcn.s.prefetch.data sibling requires a uniform "
          "pointer (the VMEM prefetch is divergent), and a silent "
          "drop would mask both the cross-target capability gap "
          "and any pipeline-tuning regression downstream.");
      return hr;
    }

    bool isSaddr = false;
    if (op.nSrcs() == 4) {
      isSaddr = true;
    } else if (op.nSrcs() != 3) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "FLAT",
          "global_prefetch_b8: expected 3 srcs (plain) or 4 srcs "
          "(SADDR) per FLAT_Prefetch_Pseudo");
      return hr;
    }

    Value *globalAddr = nullptr;
    unsigned immStart = 0;
    if (isSaddr) {
      ParsedReg saddrPr = op.srcReg(0);
      ParsedReg vaddrPr = op.srcReg(1);
      if (saddrPr.kind != ParsedReg::SGPR ||
          vaddrPr.kind != ParsedReg::VGPR) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "FLAT",
            "global_prefetch_b8 SADDR: expected (SGPR_64, VGPR_32) "
            "for (saddr, vaddr)");
        return hr;
      }
      Value *saddr = ctx.regs.readReg64(ctx.B, saddrPr);
      Value *voff = ctx.B.CreateZExt(
          ctx.regs.readReg32(ctx.B, vaddrPr), ctx.i64Ty, "voff_zext");
      globalAddr = ctx.B.CreateAdd(saddr, voff, "saddr_vaddr");
      immStart = 2;
    } else {
      ParsedReg vaddrPr = op.srcReg(0);
      if (vaddrPr.kind != ParsedReg::VGPR) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "FLAT",
            "global_prefetch_b8 plain: expected VGPR_64 for vaddr");
        return hr;
      }
      globalAddr = ctx.regs.readReg64(ctx.B, vaddrPr);
      immStart = 1;
    }

    int64_t flatOffset = 0;
    int64_t cpolImm = 0;
    bool sawOffset = false;
    for (unsigned k = immStart; k < op.nSrcs(); ++k) {
      if (!di.isImm(op.srcIdx(k))) continue;
      int64_t v = di.getImm(op.srcIdx(k));
      if (!sawOffset) {
        flatOffset = v;
        sawOffset = true;
      } else {
        cpolImm = v;
      }
    }

    Value *globalPtr = globalAddr;
    if (globalPtr->getType() != ctx.ptrGlobalTy)
      globalPtr = ctx.B.CreateIntToPtr(globalPtr, ctx.ptrGlobalTy);
    if (flatOffset != 0)
      globalPtr = ctx.B.CreateGEP(ctx.i8Ty, globalPtr,
                                   ctx.B.getInt64(flatOffset),
                                   "prefetch_addr");

    Function *fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_global_prefetch);
    Value *cpolArg = ConstantInt::get(ctx.i32Ty, cpolImm);
    ctx.B.CreateCall(fn, {globalPtr, cpolArg});

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
      llvm::errs() << "transpiler: Unhandled flat atomic: " << mn << "\n";
      hr.failure = RaiseFailure::unsupportedShape(di, "FLAT",
                                                   "unhandled flat atomic");
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
      hr.failure = RaiseFailure::unsupportedShape(
          di, "FLAT", "unsupported global atomic variant");
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
