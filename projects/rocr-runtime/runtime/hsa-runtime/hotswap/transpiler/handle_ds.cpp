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
#include <cstring>
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace transpiler {
HandlerResult handleDS(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op, RaiseResult &result) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  // Map SemOp to {dwords, loadBits, isSigned} for DS read/write
  auto dsClassify = [](SemOp s) -> std::tuple<int, int, bool> {
    switch (s) {
    case SemOp::DS_READ_B128:  case SemOp::DS_WRITE_B128:
    case SemOp::DS_READ2_B64:  case SemOp::DS_WRITE2_B64:
      return {4, 128, false};
    case SemOp::DS_READ_B64:   case SemOp::DS_WRITE_B64:
    case SemOp::DS_READ2_B32:  case SemOp::DS_WRITE2_B32:
      return {2, 64, false};
    case SemOp::DS_READ_B32:   case SemOp::DS_WRITE_B32:
      return {1, 32, false};
    case SemOp::DS_READ_U16:   case SemOp::DS_WRITE_B16:
      return {0, 16, false};
    case SemOp::DS_READ_I16:
      return {0, 16, true};
    case SemOp::DS_READ_U8:    case SemOp::DS_WRITE_B8:
      return {0, 8, false};
    case SemOp::DS_READ_I8:
      return {0, 8, true};
    default: return {-1, 0, false};
    }
  };
  // ds_load_tr16_b128: LDS transpose load — reads 128 bits with transpose
  // Semantically equivalent to a 4-dword LDS read for the purposes of
  // binary translation.
  if (sop == SemOp::DS_LOAD_TR16_B128) {
    Value *addr = ctx.B.CreateZExt(op.src(0), ctx.i64Ty, "ds_addr");
    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        int64_t imm = di.getImm(op.srcIdx(k));
        if (imm != 0)
          addr = ctx.B.CreateAdd(addr, ConstantInt::get(ctx.i64Ty, imm), "ds_off");
        break;
      }
    }
    Value *ptr = ctx.B.CreateIntToPtr(addr, PointerType::get(ctx.C, 3));
    Type *v4i32Ty = FixedVectorType::get(ctx.i32Ty, 4);
    Value *loaded = ctx.B.CreateLoad(v4i32Ty, ptr, "ds_tr16");
    ParsedReg dest = op.dst();
    for (unsigned i = 0; i < 4; i++) {
      Value *elem = ctx.B.CreateExtractElement(loaded, i);
      ctx.regs.storeVGPR32(ctx.B, dest.baseIdx + i, elem);
    }
    hr.handled = true;
    return hr;
  }
  bool isDsRead = sop >= SemOp::DS_READ_B32 && sop <= SemOp::DS_READ_I8;
  bool isDsWrite = sop >= SemOp::DS_WRITE_B32 && sop <= SemOp::DS_WRITE_B8;
  if (isDsRead || isDsWrite) {
    auto [dwords, loadBits, isSigned] = dsClassify(sop);

    Value *addr = ctx.B.CreateZExt(op.src(0), ctx.i64Ty, "ds_addr");

    for (unsigned k = 1; k < op.nSrcs(); k++) {
      if (di.isImm(op.srcIdx(k))) {
        int64_t imm = di.getImm(op.srcIdx(k));
        if (imm != 0)
          addr = ctx.B.CreateAdd(addr, ConstantInt::get(ctx.i64Ty, imm), "ds_off");
        break;
      }
    }

    Value *ptr = ctx.B.CreateIntToPtr(addr, PointerType::get(ctx.C, 3));

    if (isDsRead) {
      ParsedReg dest = op.dst();
      if (dwords == 0) {
        Type *memTy = Type::getIntNTy(ctx.C, loadBits);
        Value *v = ctx.B.CreateLoad(memTy, ptr, "ds_ld");
        ctx.regs.writeReg32(ctx.B, dest, isSigned ? ctx.B.CreateSExt(v, ctx.i32Ty) : ctx.B.CreateZExt(v, ctx.i32Ty));
      } else if (dwords == 1) {
        ctx.regs.writeReg32(ctx.B, dest, ctx.B.CreateLoad(ctx.i32Ty, ptr, "ds_ld"));
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        ctx.regs.writeRegVec(ctx.B, dest, ctx.B.CreateLoad(vecTy, ptr, "ds_ld"));
      }
      hr.handled = true;
    return hr;
    }
    if (isDsWrite) {
      ParsedReg stData = op.srcReg(1);
      if (dwords == 0) {
        Type *memTy = Type::getIntNTy(ctx.C, loadBits);
        ctx.B.CreateStore(ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, stData), memTy), ptr);
      } else if (dwords == 1) {
        ctx.B.CreateStore(ctx.regs.readReg32(ctx.B, stData), ptr);
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        ctx.B.CreateStore(ctx.regs.readRegVec(ctx.B, stData, vecTy), ptr);
      }
      hr.handled = true;
    return hr;
    }
  }
  if (sop == SemOp::DS_BPERMUTE_B32) {
    ctx.regs.writeReg32(ctx.B, op.dst(), op.src(1));
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
