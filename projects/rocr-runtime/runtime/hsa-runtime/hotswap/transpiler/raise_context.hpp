#ifndef HOTSWAP_TRANSPILER_RAISE_CONTEXT_HPP
#define HOTSWAP_TRANSPILER_RAISE_CONTEXT_HPP

#include "decoded_inst.hpp"
#include "isa_profile.hpp"
#include "kernarg_layout.hpp"
#include "mc_state.hpp"
#include "parsed_reg.hpp"
#include "reg_file.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include <map>

namespace transpiler {

struct RaiseResult;

// Shared state threaded through every format handler.
struct RaiseContext {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::IRBuilder<> &B;
  AllocaRegFile &regs;
  const MCState &mc;
  const ISAProfile &isa;
  KernargLayout &kernargs;
  llvm::Function *kernel;

  llvm::Type *i1Ty;
  llvm::Type *i8Ty;
  llvm::Type *i32Ty;
  llvm::Type *i64Ty;
  llvm::Type *f32Ty;
  llvm::Type *f16Ty;
  llvm::Type *ptrGlobalTy;

  std::map<uint64_t, llvm::BasicBlock *> &offsetToBB;

  llvm::BasicBlock *lookupBB(uint64_t addr);

  ParsedReg parseReg(unsigned reg) const;

  // Operand reading — mirrors the lambdas in the original raiseToIR.
  llvm::Value *readOp32(const DecodedInst &di, unsigned opIdx);
  llvm::Value *readOp64(const DecodedInst &di, unsigned opIdx);
  llvm::Value *readOpExecWidth(const DecodedInst &di, unsigned opIdx);
};

// Return value from every format handler.
struct HandlerResult {
  bool handled = false;
  llvm::Value *sccResult = nullptr;
  bool sccHandled = false;
};

// Reads source operands via srcMap, skipping VOP3 modifiers.
struct OpResolver {
  RaiseContext &ctx;
  const DecodedInst &di;

  unsigned srcIdx(unsigned i) const {
    assert(i < di.numSrcs && "source index out of range");
    return di.srcMap[i];
  }
  unsigned nSrcs() const { return di.numSrcs; }

  unsigned srcMod(unsigned i) const {
    unsigned modIdx = di.modMap[i];
    if (modIdx == UINT_MAX) return 0;
    if (!di.isImm(modIdx)) return 0;
    return (unsigned)(di.getImm(modIdx) & 0xF);
  }

  llvm::Value *applyMods(unsigned i, llvm::Value *v) {
    unsigned mods = srcMod(i);
    if (mods == 0) return v;
    bool isI32 = (v->getType() == ctx.i32Ty);
    if (isI32) v = ctx.B.CreateBitCast(v, ctx.f32Ty);
    if (mods & 2)
      v = ctx.B.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, v, nullptr, "abs");
    if (mods & 1)
      v = ctx.B.CreateFNeg(v, "neg");
    if (isI32) v = ctx.B.CreateBitCast(v, ctx.i32Ty);
    return v;
  }

  llvm::Value *src(unsigned i) { return ctx.readOp32(di, srcIdx(i)); }
  llvm::Value *srcF(unsigned i) { return applyMods(i, ctx.readOp32(di, srcIdx(i))); }
  llvm::Value *src64(unsigned i) { return ctx.readOp64(di, srcIdx(i)); }
  llvm::Value *srcExecWidth(unsigned i) { return ctx.readOpExecWidth(di, srcIdx(i)); }
  int64_t srcImm(unsigned i) { return di.getImm(srcIdx(i)); }

  ParsedReg dst(unsigned i = 0) { return ctx.parseReg(di.getReg(i)); }
  bool isSrcReg(unsigned i) { return di.isReg(srcIdx(i)); }

  ParsedReg srcReg(unsigned i) {
    unsigned idx = srcIdx(i);
    if (!di.isReg(idx)) {
      ParsedReg pr;
      pr.kind = ParsedReg::OTHER;
      return pr;
    }
    return ctx.parseReg(di.getReg(idx));
  }
};

} // namespace transpiler

#endif
