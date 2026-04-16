//===-- aegisbit/TrampolineBridge.h - Trampoline Bridge ----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Orchestrator for kernel instrumentation: discovers sites, allocates
/// islands, emits trampoline bodies, and manages relay stubs.
///
/// Composes ISAEncoder, IslandAllocator, TrampolineEmitter, and RelayEmitter
/// to produce patched kernel code.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TRAMPOLINE_BRIDGE_H
#define AEGISBIT_TRAMPOLINE_BRIDGE_H

#include "aegisbit/InstrumentationPlan.h"
#include "aegisbit/ISAEncoder.h"
#include "aegisbit/TrampolineTypes.h"
#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aegisbit {

class Disassembler;

//===----------------------------------------------------------------------===//
// TrampolineBridge
//===----------------------------------------------------------------------===//

class TrampolineBridge {
public:
  static llvm::Expected<std::unique_ptr<TrampolineBridge>>
  create(llvm::StringRef GPUArch, Disassembler &Disasm);

  ~TrampolineBridge();

  /// Code ranges in .text occupied by other functions (must not be overwritten).
  using OccupiedRanges = std::vector<std::pair<uint64_t, uint64_t>>;

  /// Build an empty trampoline bridge using above-the-count registers.
  llvm::Expected<BridgeResult>
  buildEmpty(llvm::ArrayRef<uint8_t> Code, uint64_t BaseAddr,
             uint64_t TextSectionSize,
             const std::vector<InstrumentationSite> &Sites,
             const ScratchRegisters &Scratch,
             uint64_t PreKernelSpace = 0,
             const OccupiedRanges &Occupied = {});

  /// Build an instrumented trampoline with per-lane address capture.
  llvm::Expected<BridgeResult>
  buildInstrumented(llvm::ArrayRef<uint8_t> Code, uint64_t BaseAddr,
                    uint64_t TextSectionSize,
                    const std::vector<InstrumentationSite> &Sites,
                    const InstrumentationPlan &Plan,
                    const ScratchRegisters &Scratch,
                    const TraceConfig &Trace,
                    uint64_t PreKernelSpace = 0,
                    const OccupiedRanges &Occupied = {});

  /// Compatibility overload for existing call sites that still derive behavior
  /// from scratch/trace inputs directly.
  llvm::Expected<BridgeResult>
  buildInstrumented(llvm::ArrayRef<uint8_t> Code, uint64_t BaseAddr,
                    uint64_t TextSectionSize,
                    const std::vector<InstrumentationSite> &Sites,
                    const ScratchRegisters &Scratch,
                    const TraceConfig &Trace,
                    uint64_t PreKernelSpace = 0,
                    const OccupiedRanges &Occupied = {});

  /// Find all memory sites in a kernel's CFG.
  /// Delegates to SiteAnalyzer::findMemorySites.
  static std::vector<InstrumentationSite>
  findMemorySites(const ControlFlowGraph &CFG, uint64_t BaseAddr,
                  Disassembler &Disasm,
                  const ScratchRegisters &Scratch = ScratchRegisters{},
                  bool SupportsGPUAtomics = false);

  /// Compute pre-spill drain values for each site.
  /// Delegates to SiteAnalyzer::computePreSpillDrainValues.
  static void computePreSpillDrainValues(
      const ControlFlowGraph &CFG,
      std::vector<InstrumentationSite> &Sites,
      const ScratchRegisters &Scratch,
      Disassembler &Disasm);

  static constexpr int64_t MaxBranchRange = 128 * 1024;
  static constexpr uint64_t ExecBufferSize = 64ULL * 1024 * 1024;

private:
  TrampolineBridge() = default;

  std::unique_ptr<ISAEncoder> Enc;
  std::string Arch;
};

} // namespace aegisbit

#endif // AEGISBIT_TRAMPOLINE_BRIDGE_H
