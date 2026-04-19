#include "mubuf_addr.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace transpiler {

namespace {

// Classify source operands of a MUBUF load/store into {srsrc, vaddr,
// soff, imm, vdata}. Keys on `ParsedReg::Kind` rather than position so
// MUBUF and VBUFFER encodings with different operand orders both
// route here. `isStore` controls the skip-first-VGPR rule: stores
// carry vdata (VGPR) ahead of vaddr (VGPR) in the operand list.
struct MubufOps {
  ParsedReg srsrc;
  ParsedReg vaddr;
  ParsedReg soff;
  ParsedReg vdata;
  int64_t immOff = 0;
  bool haveSrsrc = false;
  bool haveVaddr = false;
  bool haveSoff = false;
};

MubufOps classifyMubufOps(const DecodedInst &di, OpResolver &op,
                           bool isStore) {
  MubufOps out;
  int vgprSrcCount = 0;
  for (unsigned k = 0; k < op.nSrcs(); ++k) {
    unsigned idx = op.srcIdx(k);
    if (di.isReg(idx)) {
      ParsedReg pr = op.srcReg(k);
      if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 && !out.haveSrsrc) {
        out.srsrc = pr;
        out.haveSrsrc = true;
      } else if (pr.kind == ParsedReg::VGPR) {
        vgprSrcCount++;
        // For stores, the first VGPR source is vdata (the stored value);
        // the second is the per-lane buffer offset (vaddr).
        if (isStore && vgprSrcCount == 1) {
          out.vdata = pr;
          continue;
        }
        if (!out.haveVaddr) {
          out.vaddr = pr;
          out.haveVaddr = true;
        }
      } else if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 &&
                 !out.haveSoff) {
        out.soff = pr;
        out.haveSoff = true;
      }
    } else if (di.isImm(idx)) {
      int64_t v = di.getImm(idx);
      if (v != 0 && out.immOff == 0)
        out.immOff = v;
    }
  }
  return out;
}

// Read the three consecutive SGPR dwords of a MUBUF/VBUFFER SRSRC
// 128-bit tuple. The fourth word is always 0 (raw buffer, TYPE=0, no
// format conversion) so we don't need to read it from the register
// file.
//
// Returns the raw dwords (dw0 = base_lo, dw1 = base_hi bits plus
// stride/flags, dw2 = num_records); callers that only need the
// packaged raw-buffer descriptor should use `buildMubufSRD` below.
//
// No null-check on the returned Value*s: `AllocaRegFile::readReg32`
// already fails loudly on unhandled ParsedReg kinds and out-of-range
// SGPR indices, so SGPR reads are guaranteed to hand back a real
// Value*.
struct SRSRCDwords {
  Value *dw0;
  Value *dw1;
  Value *dw2;
};

SRSRCDwords readSRSRCDwords(RaiseContext &ctx, ParsedReg srsrc) {
  Value *dw0 = ctx.regs.readReg32(ctx.B, srsrc);
  ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
  ParsedReg srsrc2 = srsrc; srsrc2.baseIdx = srsrc.baseIdx + 2;
  Value *dw1 = ctx.regs.readReg32(ctx.B, srsrc1);
  Value *dw2 = ctx.regs.readReg32(ctx.B, srsrc2);
  return {dw0, dw1, dw2};
}

// Build a gfx942-compatible raw buffer descriptor <4 x i32> from the
// three SRSRC dwords. Each word is routed through
// `amdgcn.readfirstlane` so it lands in an SGPR — the backend would
// otherwise emit a waterfall loop around the intrinsic call.
Value *buildMubufSRD(RaiseContext &ctx, const SRSRCDwords &dw) {
  Function *readfirstlane = Intrinsic::getOrInsertDeclaration(
      &ctx.M, Intrinsic::amdgcn_readfirstlane, {ctx.i32Ty});
  Value *cleanDw1 = ctx.B.CreateAnd(dw.dw1,
                                     ConstantInt::get(ctx.i32Ty, 0xFFFF));
  Value *srdW0 = ctx.B.CreateCall(readfirstlane, {dw.dw0}, "srd_w0");
  Value *srdW1 = ctx.B.CreateCall(readfirstlane, {cleanDw1}, "srd_w1");
  Value *srdW2 = ctx.B.CreateCall(readfirstlane, {dw.dw2}, "srd_w2");
  Value *word3 = ConstantInt::get(ctx.i32Ty, 0);
  Value *srd = UndefValue::get(FixedVectorType::get(ctx.i32Ty, 4));
  srd = ctx.B.CreateInsertElement(srd, srdW0, (uint64_t)0);
  srd = ctx.B.CreateInsertElement(srd, srdW1, (uint64_t)1);
  srd = ctx.B.CreateInsertElement(srd, srdW2, (uint64_t)2);
  srd = ctx.B.CreateInsertElement(srd, word3, (uint64_t)3);
  return srd;
}

} // namespace

MubufAddr decodeMubufAddr(RaiseContext &ctx, const DecodedInst &di,
                          OpResolver &op, bool isStore,
                          StringRef diagLabel) {
  MubufOps m = classifyMubufOps(di, op, isStore);

  if (!m.haveSrsrc) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "transpiler: " << diagLabel << ": no SRSRC found for "
       << di.rawMnemonic;
    report_fatal_error(StringRef(os.str()));
  }

  MubufAddr out;
  SRSRCDwords dw = readSRSRCDwords(ctx, m.srsrc);
  out.srd = buildMubufSRD(ctx, dw);
  out.stData = m.vdata;

  Value *voffset = ConstantInt::get(ctx.i32Ty, 0);
  if (m.haveVaddr)
    voffset = ctx.B.CreateAdd(voffset, ctx.regs.readReg32(ctx.B, m.vaddr));
  if (m.immOff != 0)
    voffset = ctx.B.CreateAdd(voffset,
                               ConstantInt::get(ctx.i32Ty, (int32_t)m.immOff));
  out.voffset = voffset;

  out.soffset = m.haveSoff ? ctx.regs.readReg32(ctx.B, m.soff)
                           : ConstantInt::get(ctx.i32Ty, 0);
  out.auxFlags = ConstantInt::get(ctx.i32Ty, 0);
  return out;
}

MubufAtomicAddr decodeMubufAtomicAddr(RaiseContext &ctx,
                                       const DecodedInst & /*di*/,
                                       OpResolver &op,
                                       StringRef /*diagLabel*/) {
  // `readReg32` fatal's on out-of-range SGPR indices, so no null-check
  // is needed here — the loads either succeed or never return.
  ParsedReg srsrc = op.srcReg(0);
  Value *dw0 = ctx.regs.readReg32(ctx.B, srsrc);
  ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
  Value *dw1 = ctx.regs.readReg32(ctx.B, srsrc1);
  Value *lo = ctx.B.CreateZExt(dw0, ctx.i64Ty);
  Value *hi = ctx.B.CreateAnd(ctx.B.CreateZExt(dw1, ctx.i64Ty),
                               ConstantInt::get(ctx.i64Ty, 0xFFFF));
  Value *ptrInt = ctx.B.CreateOr(lo, ctx.B.CreateShl(hi, 32), "buf_base");
  MubufAtomicAddr out;
  out.ptr = ctx.B.CreateIntToPtr(ptrInt, PointerType::get(ctx.C, 0));
  return out;
}

} // namespace transpiler
