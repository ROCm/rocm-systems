//===-- aegisbit/ScratchRegisters.h - Scratch Register Allocation *-C++-*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Scratch register set used by the trampoline emitters. Extracted from
/// Types.h so that ownership is clear: the factory methods describe a
/// static allocation strategy against a KernelDescriptor, while the
/// instance methods (refineScratchVGPRs / setupAccVGPRSpill /
/// setupScratchSpill) need a decoded CFG and therefore live in a
/// dedicated translation unit.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_SCRATCH_REGISTERS_H
#define AEGISBIT_SCRATCH_REGISTERS_H

#include "aegisbit/RegisterHelper.h"
#include "aegisbit/Types.h"

#include <cstdint>

namespace aegisbit {

class Disassembler;

/// Scratch registers allocated above the kernel's declared register count.
/// The hardware guarantees these are not used by the kernel.
///
/// Two modes:
///   Empty trampoline  — needs 2 SGPRs + 1 VGPR  (fromDescriptor)
///   Instrumented      — needs 6 SGPRs + 3 VGPRs (fromDescriptorInstrumented)
///
/// Due to SGPR/VGPR granularity (8), both modes bump the descriptor
/// by the same amount (one granularity step each = +8 of each).
struct ScratchRegisters {
  // --- Core registers (used by both empty and instrumented) ---

  unsigned ReturnAddrSGPR = 0;   ///< s[N]   — return addr lo
  unsigned ReturnAddrSGPRHi = 0; ///< s[N+1] — return addr hi
  unsigned ScratchVGPR = 0;      ///< v[M]   — save RA via writelane/readlane

  // --- Additional registers for instrumented trampoline ---

  unsigned ScratchSGPR = 0;      ///< s[N+2] — general-purpose scalar scratch
  unsigned ExecSaveSGPRLo = 0;   ///< s[N+3] — save exec_lo during atomic
  unsigned ExecSaveSGPRHi = 0;   ///< s[N+4] — save exec_hi during atomic
  unsigned SAddrTempSGPR = 0;    ///< s[N+5] — SADDR temp for large-offset scratch ops
  unsigned SwapTargetSGPR = 0;   ///< s[N+6] — swappc target addr lo
  unsigned SwapTargetSGPRHi = 0; ///< s[N+7] — swappc target addr hi
  unsigned LaneVGPR = 0;         ///< v[M+1] — lane ID / per-lane write offset
  unsigned TempVGPR = 0;         ///< v[M+2] — temp for atomic value

  uint32_t ExtraVGPRs = 0;      ///< Requested extra VGPRs (descriptor delta)
  uint32_t ExtraSGPRs = 0;      ///< Requested extra SGPRs (descriptor delta)
  bool HasAccumVGPRs = false;   ///< True if kernel uses AccVGPRs (gfx90a/gfx942)
  bool NeedsAccVGPRSpill = false; ///< True if victim VGPRs must be spilled to AccVGPRs (DEPRECATED)
  bool NeedsScratchSpill = false; ///< True if victim VGPRs must be spilled to scratch memory
  bool ZeroSGPR = false;         ///< True if using VCC-temp mode (no extra SGPRs allocated)
  bool UseSwapPC = false;        ///< True if using s_swappc_b64 shared-body mode

  /// AccVGPR spill slots for saving victim VGPRs (LLVM AGPR register numbers).
  /// Used when all regular VGPRs are referenced and we must spill/restore.
  /// DEPRECATED: Use scratch memory spill instead for robustness.
  unsigned SpillAGPR0 = 0;      ///< Spill slot for ScratchVGPR
  unsigned SpillAGPR1 = 0;      ///< Spill slot for LaneVGPR
  unsigned SpillAGPR2 = 0;      ///< Spill slot for TempVGPR
  unsigned SpillAGPR3 = 0;      ///< Spill slot for AddrVGPR during counting loop

  /// Scratch memory spill configuration.
  /// Used for AccVGPR kernels where we spill victim VGPRs to scratch memory
  /// instead of AccVGPR slots. More robust than AccVGPR spill.
  uint32_t ScratchSpillOffset = 0;  ///< Byte offset within scratch for our spill area
  uint32_t ExtraScratchBytes = 0;   ///< Additional scratch bytes needed (12 = 3 VGPRs × 4 bytes)

  /// Raw SGPR/VGPR indices (not LLVM register numbers) for the first free
  /// register. Useful for encoding instructions that take raw indices.
  unsigned FirstFreeSGPRIdx = 0;
  unsigned FirstFreeVGPRIdx = 0;

  //===--------------------------------------------------------------------===//
  // Factory methods
  //===--------------------------------------------------------------------===//

  /// Minimal allocation for empty trampoline (2 SGPR + 1 VGPR).
  static ScratchRegisters fromDescriptor(const KernelDescriptor &KD);

  /// Full allocation for instrumented trampoline (6 SGPR + 3 VGPR).
  static ScratchRegisters
  fromDescriptorInstrumented(const KernelDescriptor &KD);

  /// SwapPC allocation (8 SGPR + 3 VGPR) for unlimited-range shared body.
  static ScratchRegisters fromDescriptorSwapPC(const KernelDescriptor &KD);

  /// Zero-extra-SGPR allocation for kernels that max out SGPR capacity.
  /// Uses VCC_LO as sole temp SGPR and requires GPU atomics support.
  static ScratchRegisters fromDescriptorZeroSGPR(const KernelDescriptor &KD);

  //===--------------------------------------------------------------------===//
  // CFG-aware refinement / spill setup
  //===--------------------------------------------------------------------===//

  /// Refine scratch VGPRs for AccVGPR kernels by scanning the CFG for
  /// VGPRs never referenced by any instruction in v0..v(AccumOffset-1).
  /// \returns true if 3 free VGPRs were found.
  bool refineScratchVGPRs(const ControlFlowGraph &CFG,
                          const Disassembler &Disasm,
                          uint32_t AccumOffset);

  /// Set up AccVGPR spill/restore when all regular VGPRs are in use.
  /// DEPRECATED: Use setupScratchSpill() instead for robustness.
  void setupAccVGPRSpill(uint32_t AccumOffset, uint32_t VGPRCount);

  /// Set up scratch-memory spill/restore for AccVGPR kernels.
  void setupScratchSpill(uint32_t AccumOffset, uint32_t CurrentScratchSize);

  bool isValid() const {
    if (ZeroSGPR)
      return ScratchVGPR != 0;
    return ReturnAddrSGPR != 0 && ReturnAddrSGPRHi != 0 && ScratchVGPR != 0;
  }
};

} // namespace aegisbit

#endif // AEGISBIT_SCRATCH_REGISTERS_H
