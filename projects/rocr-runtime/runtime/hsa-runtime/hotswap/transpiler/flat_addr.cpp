#include "flat_addr.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace transpiler {

namespace {

// Scan the operand tail (at and after `immStart`) for the first immediate
// and return its value. Any later imms are encoding flags (cpol, th,
// scope) and are ignored. Signed 13-bit offset; already sign-extended
// by the MC layer.
int64_t firstImmOffset(const DecodedInst &di, OpResolver &op,
                       unsigned immStart) {
  for (unsigned k = immStart; k < op.nSrcs(); ++k) {
    if (di.isImm(op.srcIdx(k)))
      return di.getImm(op.srcIdx(k));
  }
  return 0;
}

// Coerce an integer address into a global-AS pointer and apply a signed
// byte offset via a plain (non-inbounds) GEP. The ISA's signed offset
// can legitimately leave the base allocation (e.g. compiler-scheduled
// prefetches, negative strides); `inbounds` would turn that into UB.
Value *toGlobalPtr(RaiseContext &ctx, Value *addr, int64_t memOffset) {
  if (addr->getType() != ctx.ptrGlobalTy)
    addr = ctx.B.CreateIntToPtr(addr, ctx.ptrGlobalTy);
  if (memOffset != 0)
    addr = ctx.B.CreateGEP(ctx.i8Ty, addr, ctx.B.getInt64(memOffset));
  return addr;
}

} // namespace

FlatAddr decodeGlobalLoadAddr(RaiseContext &ctx, const DecodedInst &di,
                               OpResolver &op, int elemBytes,
                               StringRef diagLabel) {
  FlatAddr out;
  Value *addr = nullptr;

  // SADDR form: saddr(SGPR64), vaddr(VGPR32), ... — LLVM MC places the
  // SGPR first in the decoded operand order.
  if (op.nSrcs() >= 2 && op.isSrcReg(0) && op.isSrcReg(1) &&
      op.srcReg(0).kind == ParsedReg::SGPR &&
      op.srcReg(1).kind == ParsedReg::VGPR) {
    out.hasSaddr = true;
    Value *saddr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
    Value *vaddr = ctx.B.CreateSExt(ctx.regs.readReg32(ctx.B, op.srcReg(1)),
                                    ctx.i64Ty, "voff_sext");
    if (di.hasScaleOffset)
      vaddr = ctx.B.CreateMul(vaddr, ConstantInt::get(ctx.i64Ty, elemBytes),
                              "scaled_voff");
    addr = ctx.B.CreateAdd(saddr, vaddr, "saddr_vaddr");
  } else if (op.nSrcs() >= 1 && op.isSrcReg(0) &&
             op.srcReg(0).kind == ParsedReg::VGPR) {
    // Plain form: VGPR64 holds the full per-lane address. Do NOT gate on
    // width — parseReg currently reports tuple VGPRs (e.g. VGPR2_VGPR3)
    // with width=1 on some subtargets; readReg64 walks the sub0/sub1
    // graph itself, so trust the SGPR-vs-VGPR discriminator above and
    // let readReg64 enforce the 64-bit shape.
    addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
  } else {
    std::string msg;
    raw_string_ostream os(msg);
    os << "transpiler: unrecognized " << diagLabel
       << " operand shape (expected plain VGPR64 or SADDR SGPR64+VGPR32): \""
       << di.fullText << "\" (mnemonic=" << di.rawMnemonic << ")";
    report_fatal_error(StringRef(os.str()));
  }

  out.memOffset = firstImmOffset(di, op, out.hasSaddr ? 2 : 1);
  out.ptr = toGlobalPtr(ctx, addr, out.memOffset);
  return out;
}

FlatAddr decodeGlobalStoreAddr(RaiseContext &ctx, const DecodedInst &di,
                                OpResolver &op, int elemBytes,
                                StringRef diagLabel) {
  FlatAddr out;
  Value *addr = nullptr;

  // SADDR form: vaddr(VGPR32), vdata(VGPR*), saddr(SGPR64), ...
  if (op.nSrcs() >= 3 && op.isSrcReg(0) && op.isSrcReg(1) && op.isSrcReg(2) &&
      op.srcReg(0).kind == ParsedReg::VGPR &&
      op.srcReg(1).kind == ParsedReg::VGPR &&
      op.srcReg(2).kind == ParsedReg::SGPR) {
    out.hasSaddr = true;
    Value *saddr = ctx.regs.readReg64(ctx.B, op.srcReg(2));
    Value *vaddr = ctx.B.CreateSExt(ctx.regs.readReg32(ctx.B, op.srcReg(0)),
                                    ctx.i64Ty, "st_voff_sext");
    if (di.hasScaleOffset)
      vaddr = ctx.B.CreateMul(vaddr, ConstantInt::get(ctx.i64Ty, elemBytes),
                              "st_scaled_voff");
    addr = ctx.B.CreateAdd(saddr, vaddr, "st_saddr_vaddr");
    out.stData = op.srcReg(1);
  } else if (op.nSrcs() >= 2 && op.isSrcReg(0) && op.isSrcReg(1) &&
             op.srcReg(0).kind == ParsedReg::VGPR &&
             op.srcReg(1).kind == ParsedReg::VGPR) {
    addr = ctx.regs.readReg64(ctx.B, op.srcReg(0));
    out.stData = op.srcReg(1);
  } else {
    std::string msg;
    raw_string_ostream os(msg);
    os << "transpiler: unrecognized " << diagLabel
       << " operand shape (expected plain VGPR+VGPR or SADDR VGPR+VGPR+SGPR): \""
       << di.fullText << "\" (mnemonic=" << di.rawMnemonic << ")";
    report_fatal_error(StringRef(os.str()));
  }

  out.memOffset = firstImmOffset(di, op, out.hasSaddr ? 3 : 2);
  out.ptr = toGlobalPtr(ctx, addr, out.memOffset);
  return out;
}

} // namespace transpiler
