//===-- aegisbit/PayloadCompiler.h - LLVM IR Payload Compiler ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compiles trampoline payloads from LLVM IR through the AMDGPU backend.
/// The compiler's GCNHazardRecognizer inserts all necessary hazard NOPs,
/// eliminating hand-assembled ISA hazard workarounds.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_PAYLOAD_COMPILER_H
#define AEGISBIT_PAYLOAD_COMPILER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
} // namespace llvm

namespace aegisbit {

/// Compiles LLVM IR modules to raw AMDGPU machine code bytes.
///
/// Each compiled payload is a position-independent blob of instructions
/// (branches use PC-relative offsets) that can be spliced directly into
/// trampoline byte streams. The prologue (s_waitcnt) and epilogue
/// (s_setpc_b64) are stripped so the blob contains only the payload logic.
class PayloadCompiler {
public:
  /// Create a PayloadCompiler targeting the given GPU architecture.
  /// \param GPUArch  e.g. "gfx942", "gfx950"
  static llvm::Expected<std::unique_ptr<PayloadCompiler>>
  create(llvm::StringRef GPUArch);

  ~PayloadCompiler();

  /// Compile an LLVM IR module to raw .text section bytes.
  /// Strips the function prologue (s_waitcnt) and epilogue (s_setpc_b64).
  llvm::Expected<std::vector<uint8_t>>
  compile(std::unique_ptr<llvm::Module> M);

  /// Get the LLVMContext for IR construction.
  llvm::LLVMContext &getContext();

  /// Get the target GPU architecture string.
  const std::string &getGPUArch() const;

  /// Build LLVM IR for the unique cache-line counting loop.
  ///
  /// Function signature (amdgpu_gfx calling convention):
  ///   i32 @aegis_count_unique(i32 %addr_val, i32 inreg %exec_lo,
  ///                           i32 inreg %exec_hi)
  ///
  /// Arguments:
  ///   %addr_val  — divergent VGPR value (per-lane address), passed in v0
  ///   %exec_lo   — low 32 bits of active lane mask, passed in s0
  ///   %exec_hi   — high 32 bits of active lane mask, passed in s1
  ///
  /// Returns the number of unique values across active lanes in v0.
  static std::unique_ptr<llvm::Module>
  buildCountingLoop(llvm::LLVMContext &Ctx);

  /// Build LLVM IR for the max-popcount-per-value counting loop.
  ///
  /// Same structure as buildCountingLoop but returns the maximum number of
  /// lanes that share any single value, rather than the number of unique values.
  /// Used for LDS bank conflict measurement: the result equals the conflict
  /// cycle count (assuming no intra-bank address broadcasts).
  ///
  /// Function signature identical to buildCountingLoop.
  /// Returns max(popcount(match_mask)) across all unique values in v0→s0.
  static std::unique_ptr<llvm::Module>
  buildMaxPopCountLoop(llvm::LLVMContext &Ctx);

  /// Build LLVM IR for the atomic counter accumulation.
  ///
  /// Function signature (amdgpu_gfx calling convention):
  ///   void @aegis_atomic_accum(ptr addrspace(1) inreg %base,
  ///                            i32 inreg %count)
  ///
  /// Arguments:
  ///   %base  — pointer to per-site counter pair (8 bytes), passed in s[0:1]
  ///   %count — unique cache-line count to add, passed in s2
  ///
  /// Layout at %base: [u32 total_cache_lines, u32 total_samples]
  ///
  /// \param UseAtomics  If true, uses atomicrmw add (for fine-grained memory).
  ///                    If false, uses non-atomic load-add-store.
  static std::unique_ptr<llvm::Module>
  buildAtomicAccumulator(llvm::LLVMContext &Ctx, bool UseAtomics);

private:
  PayloadCompiler();

  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::TargetMachine> TM;
  std::string Arch;
};

} // namespace aegisbit

#endif // AEGISBIT_PAYLOAD_COMPILER_H
