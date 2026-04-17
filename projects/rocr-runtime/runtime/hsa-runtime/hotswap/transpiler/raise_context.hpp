#ifndef HOTSWAP_TRANSPILER_RAISE_CONTEXT_HPP
#define HOTSWAP_TRANSPILER_RAISE_CONTEXT_HPP

#include "decoded_inst.hpp"
#include "isa_profile.hpp"
#include "kernarg_layout.hpp"
#include "mc_state.hpp"
#include "parsed_reg.hpp"
#include "reg_file.hpp"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegister.h"

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
  const ISAProfile &isa;       // source ISA (for disassembly / instruction semantics)
  ISAProfile targetIsa;        // compilation target ISA (for code generation decisions)
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

  // gfx1250 s_set_vgpr_msb state: only the LOW 8 bits of the instruction's
  // 16-bit immediate carry runtime meaning.  They encode the MSB bit pair for
  // every operand slot of the next ALU instruction:
  //
  //   [1:0]  src0 MSB        [3:2]  src1 MSB
  //   [5:4]  src2 MSB        [7:6]  vdst MSB
  //
  // Each 2-bit field adds (value * 256) to the corresponding VGPR index.
  //
  // The HIGH 8 bits of the `s_set_vgpr_msb` immediate record the PREVIOUS
  // mode value for compiler bookkeeping (see LLVM's AMDGPULowerVGPREncoding
  // pass, `setMode()`: "Record previous mode into high 8 bits of the
  // immediate."). The hardware ignores them, so we mask to 8 bits on store.
  //
  // For VOPD dual-issue instructions, the X-op and Y-op share the same MSB
  // pair per operand slot: LLVM asserts that if X-op and Y-op both reference
  // the same slot, they must carry identical MSBs ("Invalid VOPD pair was
  // created" in `computeMode`).  So applying the same 8-bit state to both
  // halves is correct.
  uint8_t vgprMSBs = 0;

  // Per-instruction VGPR index adjustment, indexed by MCInst operand index.
  // Computed from vgprMSBs before each instruction dispatch.
  static constexpr unsigned kMaxOps = 16;
  unsigned currentVGPRAdjust[kMaxOps] = {};

  // Compute currentVGPRAdjust for the given instruction based on vgprMSBs.
  void computeVGPRAdjust(const DecodedInst &di);

  llvm::BasicBlock *lookupBB(uint64_t addr);

  ParsedReg parseReg(llvm::MCRegister reg, int mciOpIdx = -1) const;

  // Operand reading — mirrors the lambdas in the original raiseToIR.
  llvm::Value *readOp32(const DecodedInst &di, unsigned opIdx);
  llvm::Value *readOp64(const DecodedInst &di, unsigned opIdx);
  llvm::Value *readOpExecWidth(const DecodedInst &di, unsigned opIdx);

  // ==== SIMT Predicated Execution (SPE) helpers (see SPE_DESIGN.md). =====
  //
  // emitLaneActiveBit() returns an i1 true iff the current lane's bit in the
  // EXEC-mask alloca is set. Wave-size-aware via targetIsa: the lane id is
  // built from llvm.amdgcn.mbcnt.lo for wave32 and mbcnt.lo+mbcnt.hi for
  // wave64. The alloca itself is the current SSA-tracked EXEC value, so
  // uniform code (EXEC provably -1 by SROA) folds this to `true`.
  //
  // Caching: the result is memoised for the duration of a single decoded
  // instruction's dispatch (`cachedLaneActive` + `cachedLaneActiveBB`).
  // Handlers that emit multiple `emitUnderExec` diamonds for one source
  // instruction (e.g. a multi-dword MUBUF store writing four VGPRs) reuse
  // the same `lane_active` i1 instead of re-emitting the mbcnt / lshr /
  // and / icmp chain. Invalidated on:
  //   - Entry to every new source instruction (`resetLaneActiveCache`),
  //     because intervening instructions may have written EXEC.
  //   - Insertion-block change within a handler (the cached i1 no longer
  //     dominates emission points in a new BB).
  //   - Any explicit EXEC write via `ctx.storeExec`.
  // LLVM's mem2reg + CSE would clean up the redundancy anyway; the cache
  // keeps the raw raised IR readable for lit tests that FileCheck the
  // unoptimised output shape.
  llvm::Value *emitLaneActiveBit();

  // Invalidate the lane_active memoisation. Called by the main raiser loop
  // between instructions and by `storeExec`. Handlers that know they have
  // mutated EXEC through a lower-level path (e.g. the few places that call
  // `regs.storeExec` directly) must also invoke this.
  void resetLaneActiveCache() {
    cachedLaneActive = nullptr;
    cachedLaneActiveBB = nullptr;
  }

  // Wrap `regs.storeExec` with cache invalidation. Handlers should prefer
  // this over `regs.storeExec` so the lane_active memo is always
  // consistent with the live EXEC value.
  void storeExec(llvm::Value *v) {
    regs.storeExec(B, v);
    resetLaneActiveCache();
  }

  // Predicated register-commit API. VGPR/AGPR writes are per-lane side
  // effects and MUST be wrapped in an emitUnderExec diamond so inactive
  // lanes keep their prior VGPR value; SGPR/VCC/SCC/EXEC/M0/FLAT_SCR/TTMP
  // writes are wave-level and pass through unchanged. Handlers should
  // call these instead of reaching into `regs.write*` directly.
  //
  // `storeVGPR32` / `storeVGPR64` / `storeAGPR32` are the direct-index
  // variants used when a handler already knows the register index (e.g.
  // VOPD's dstX/dstY pair or DS-transpose's contiguous sub-VGPR loop).
  void writeReg32(ParsedReg pr, llvm::Value *v);
  void writeReg64(ParsedReg pr, llvm::Value *v);
  void writeRegVec(ParsedReg pr, llvm::Value *v);
  void writeRegExecWidth(ParsedReg pr, llvm::Value *v);
  void storeVGPR32(int idx, llvm::Value *v);
  void storeVGPR64(int idx, llvm::Value *v);
  void storeAGPR32(int idx, llvm::Value *v);

  // emitUnderExec(body) wraps `body()` in an `if (lane_active)` diamond:
  //
  //   %active = emitLaneActiveBit()
  //   br i1 %active, label %exec_do, label %exec_skip
  //   exec_do:
  //     body()                 (whatever side-effectful IR the handler emits)
  //     br label %exec_skip    (only if body() did not itself terminate)
  //   exec_skip:
  //     ...                    (insertion point on return)
  //
  // Because %active is data-dependent on workitem.id.x, LLVM's divergence
  // analysis treats the branch as divergent and the AMDGPU backend
  // rematerialises hardware-level v_cmpx around the do-block. Uniform code
  // collapses: when %active folds to `true` the diamond vanishes, so this
  // is a no-op in IR size + codegen terms for non-divergent sites.
  //
  // On return, the builder's insertion point is at the start of %exec_skip,
  // so subsequent handler emission continues in the skip block (which is
  // topologically the "after" of the wrapped op, exactly like before).
  void emitUnderExec(llvm::function_ref<void()> body);

  // Memoised lane_active for this instruction's emission. Kept as public
  // members (rather than `private:`) so RaiseContext remains an aggregate
  // and can be brace-initialised from the raiser. Mutate only via
  // `resetLaneActiveCache` / `emitLaneActiveBit`.
  llvm::Value *cachedLaneActive = nullptr;
  llvm::BasicBlock *cachedLaneActiveBB = nullptr;
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

  ParsedReg dst(unsigned i = 0) { return ctx.parseReg(di.getReg(i), i); }
  bool isSrcReg(unsigned i) { return di.isReg(srcIdx(i)); }

  ParsedReg srcReg(unsigned i) {
    unsigned idx = srcIdx(i);
    if (!di.isReg(idx)) {
      ParsedReg pr;
      pr.kind = ParsedReg::OTHER;
      return pr;
    }
    return ctx.parseReg(di.getReg(idx), idx);
  }
};

} // namespace transpiler

#endif
