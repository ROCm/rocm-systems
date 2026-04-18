#ifndef HOTSWAP_TRANSPILER_REG_FILE_HPP
#define HOTSWAP_TRANSPILER_REG_FILE_HPP

#include "parsed_reg.hpp"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
class MCRegisterInfo;
} // namespace llvm

namespace transpiler {

struct ISAProfile;
class WaveProjection;

// Per-register alloca-based representation of the AMDGPU register file.
//
// Every architectural register gets its own i32 alloca (or i1 for VCC/SCC,
// i32/i64 for EXEC depending on wave width). `PromoteMemToReg` in
// `raiser.cpp` Phase 6 lifts these allocas to proper SSA with phis after
// the handler pass finishes. Keeping per-register state in allocas lets
// handlers emit straight-line loads/stores during dispatch without
// worrying about SSA construction.
//
// Storage is sized at `init()` time:
//
//   * SGPR bank: `MRI.getRegClass(AMDGPU::SGPR_32RegClassID).getNumRegs()`
//     — the authoritative count from TableGen for the live subtarget (106
//     on every AMDGPU subtarget today).
//   * TTMP bank: `MRI.getRegClass(AMDGPU::TTMP_32RegClassID).getNumRegs()`
//     — 16 on every AMDGPU subtarget.
//   * VGPR / AGPR bank: `kVGPRCap` (below). NOT sourced from the register
//     class because gfx1250 `S_SET_VGPR_MSB` extends the runtime-
//     addressable VGPR index range beyond the TableGen class size (256):
//     the raiser needs storage for every index a kernel can reach under
//     MSB replay, not just the indices the assembler names directly.
//     Keeping the cap explicit documents the extra storage as
//     intentional.
//
//     Sized to match gfx1250's `1024-addressable-vgprs` subtarget
//     feature (LLVM AMDGPU.td:1120 `def 1024AddressableVGPRs`):
//     S_SET_VGPR_MSB encodes a 2-bit MSB pair per operand slot, each
//     contributing `value * 256` to that slot's VGPR index, so the
//     maximum reachable index is `255 (8-bit base) + 3*256 = 1023`.
//     A smaller cap (the previous 512) crashed legitimate gfx1250
//     tensilelite kernels that issue `s_set_vgpr_msb 0x80` followed
//     by `v_mov_b32 v72 /*v584*/` (vdst MSB=2 → effective index
//     `72 + 512 = 584`); see `failOOB` in reg_file.cpp.
struct AllocaRegFile {
  // See class-level comment for rationale.
  static constexpr unsigned kVGPRCap = 1024;

  llvm::SmallVector<llvm::AllocaInst *> sgpr;
  llvm::SmallVector<llvm::AllocaInst *> vgpr;
  llvm::SmallVector<llvm::AllocaInst *> agpr;
  llvm::SmallVector<llvm::AllocaInst *> ttmp;
  llvm::AllocaInst *vcc = nullptr;
  llvm::AllocaInst *scc = nullptr;
  llvm::AllocaInst *exec = nullptr;
  llvm::AllocaInst *m0 = nullptr;
  llvm::AllocaInst *flatScr[2] = {};

  // Width of the EXEC alloca — tracks the *source* ISA wave width
  // (i32 on wave32 source, i64 on wave64 source). Distinct from the
  // target-hardware wave mask width owned by `WaveProjection`.
  llvm::Type *execTy = nullptr;

  // Non-owning pointer to the cross-wave projection policy. Used by
  // VCC read/write paths inside the reg file (ballot for per-lane-i1 →
  // wave-mask, and the inverse for scalar → per-lane stores). Left null
  // outside the raiser — the reg file is used in contexts where no
  // projection exists (e.g. register-file-only unit tests or EXEC
  // initialisation at function entry) and any code path that would
  // need the projection (VCC wave-mask round-trip) is then unreachable.
  const WaveProjection *projection = nullptr;

  // Invalidation hook fired on every EXEC-mutating store. The owning
  // RaiseContext installs this so its per-instruction `lane_active`
  // memo stays in sync regardless of which path hits `storeExec`.
  //
  // `unique_function` owns the callable (so the lambda's lifetime can
  // be decoupled from the stack frame that installs it) while still
  // being move-only — a correctness nudge toward "installed exactly
  // once, not copied around."
  llvm::unique_function<void()> onExecWritten;

  // Initialise storage.
  //
  // `MRI` is queried for the architectural SGPR_32 / TTMP_32 register-
  // class sizes. `ISAProfile::hasAGPR` selects whether to allocate
  // AGPR slots. `proj` is stored as a non-owning pointer for use by
  // VCC read/write paths.
  void init(llvm::IRBuilder<> &B, llvm::Type *i32Ty, llvm::Type *i1Ty,
            const ISAProfile &isa, const llvm::MCRegisterInfo &MRI,
            const WaveProjection &proj);

  // Direct per-class store/load helpers. All are predicated on `idx`
  // being in range for the corresponding class; bounds failures surface
  // via `errs()` + an undef fallback for reads, and are otherwise
  // caught at higher levels by the raiser's allocation pass.
  void storeSGPR32(llvm::IRBuilder<> &B, int idx, llvm::Value *v);
  llvm::Value *loadSGPR32(llvm::IRBuilder<> &B, int idx);
  void storeSGPR64(llvm::IRBuilder<> &B, int idx, llvm::Value *v);
  llvm::Value *loadSGPR64(llvm::IRBuilder<> &B, int idx);
  void storeVGPR32(llvm::IRBuilder<> &B, int idx, llvm::Value *v);
  llvm::Value *loadVGPR32(llvm::IRBuilder<> &B, int idx);
  void storeVGPR64(llvm::IRBuilder<> &B, int idx, llvm::Value *v);
  llvm::Value *loadVGPR64(llvm::IRBuilder<> &B, int idx);
  void storeAGPR32(llvm::IRBuilder<> &B, int idx, llvm::Value *v);
  llvm::Value *loadAGPR32(llvm::IRBuilder<> &B, int idx);

  void storeVCC(llvm::IRBuilder<> &B, llvm::Value *v);
  llvm::Value *loadVCC(llvm::IRBuilder<> &B);
  void storeSCC(llvm::IRBuilder<> &B, llvm::Value *v);
  llvm::Value *loadSCC(llvm::IRBuilder<> &B);
  llvm::Value *loadExec(llvm::IRBuilder<> &B);
  void storeExec(llvm::IRBuilder<> &B, llvm::Value *v);

  // Read VCC as a wave-level bit-mask of width `resultTy`. Routed
  // through `WaveProjection::ballotI1ToWidth`; see
  // `wave_projection.hpp` for the outer-EXEC-only invariant this
  // relies on.
  llvm::Value *readVCCAsWaveMask(llvm::IRBuilder<> &B, llvm::Type *resultTy);

  // Generic read/write by ParsedReg.
  llvm::Value *readReg32(llvm::IRBuilder<> &B, ParsedReg pr);
  llvm::Value *readReg64(llvm::IRBuilder<> &B, ParsedReg pr);
  llvm::Value *readExecWidth(llvm::IRBuilder<> &B);
  void writeExecWidth(llvm::IRBuilder<> &B, llvm::Value *v);
  void writeReg32(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v);
  void writeReg64(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v);
  void writeRegExecWidth(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v);

  // Read/write N dwords as a vector from contiguous VGPRs/AGPRs.
  llvm::Value *readRegVec(llvm::IRBuilder<> &B, ParsedReg pr,
                          llvm::Type *vecTy);
  void writeRegVec(llvm::IRBuilder<> &B, ParsedReg pr, llvm::Value *v);

  // Populate `out` with every alloca the raiser emitted, for feeding
  // into `PromoteMemToReg`.
  void collectAllocas(llvm::SmallVectorImpl<llvm::AllocaInst *> &out);
};

} // namespace transpiler

#endif
