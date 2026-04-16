//===-- aegisbit/TrampolineEmitter.h - Trampoline Body Codegen -----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Generates the byte sequences for individual trampoline bodies:
/// prologue (state save), payload (OnGpuReduce or FullCapture), epilogue
/// (state restore), spill/restore. Does not handle island placement --
/// that is IslandAllocator's responsibility.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TRAMPOLINE_EMITTER_H
#define AEGISBIT_TRAMPOLINE_EMITTER_H

#include "aegisbit/InstrumentationPlan.h"
#include "aegisbit/ISAEncoder.h"
#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace aegisbit {

class TrampolineEmitter {
public:
  /// Create an emitter, compiling payload blobs once for the given strategy.
  static llvm::Expected<std::unique_ptr<TrampolineEmitter>>
  create(ISAEncoder &Enc, llvm::StringRef Arch, const TraceConfig &Trace);

  ~TrampolineEmitter();

  /// Emit a direct trampoline body for a single site.
  llvm::Expected<std::vector<uint8_t>>
  emitDirectBody(const InstrumentationSite &Site,
                 const InstrumentationPlan &Plan,
                 const ScratchRegisters &Scratch,
                 const TraceConfig &Trace,
                 unsigned RetAddrSGPRPair,
                 uint32_t SiteIdx);

  /// Emit a relay trampoline body for a single site.
  llvm::Expected<std::vector<uint8_t>>
  emitRelayBody(const InstrumentationSite &Site,
                const InstrumentationPlan &Plan,
                const ScratchRegisters &Scratch,
                const TraceConfig &Trace,
                unsigned RetAddrSGPRPair,
                uint32_t SiteIdx);

  /// Emit a shared trampoline body that handles all sites via packed parameters.
  /// IsLDS selects the VMEM (shift 7) or LDS (shift 2 + mask) variant.
  /// maxPreSpillVmWait is the max PreSpillVmWait across all sites.
  /// GetpcToReturnTableOffset is the signed byte offset from the s_getpc_b64's
  /// "next instruction" PC to the start of the return table.
  llvm::Expected<std::vector<uint8_t>>
  emitSharedBody(const ScratchRegisters &Scratch,
                 const TraceConfig &Trace,
                 bool IsLDS,
                 unsigned RetAddrSGPRPair,
                 unsigned maxPreSpillVmWait,
                 int32_t GetpcToReturnTableOffset);

  /// Emit a per-site dispatch entry: s_mov packed_info + s_branch to shared body.
  llvm::Expected<std::vector<uint8_t>>
  emitDispatchEntry(const ScratchRegisters &Scratch,
                    uint32_t SiteIdx, uint32_t AddrVGPRIdx, bool IsGlobal,
                    int16_t BranchDwordOffset);

  /// Emit a per-site return entry (16 bytes uniform stride):
  ///   s_and_b32 ScratchSGPR, ReturnAddrSGPR, 1   (restore SCC from bit 0)
  ///   <displaced instruction> [+ s_nop if 4 bytes]
  ///   s_setpc_b64 RetAddrSGPRPair                  (return to kernel)
  llvm::Expected<std::vector<uint8_t>>
  emitReturnEntry(llvm::ArrayRef<uint8_t> DisplacedBytes,
                  unsigned RetAddrSGPRPair,
                  unsigned ScratchSGPR,
                  unsigned ReturnAddrSGPR);

  llvm::Expected<std::vector<uint8_t>>
  emitSwapPCPreamble(const ScratchRegisters &Scratch,
                     int32_t GetpcToDispatchTable);

  /// Emit a scratch_store_dword, using SADDR mode if Offset > 4095.
  /// When SADDR is needed, TempSGPR is loaded with Offset and used as
  /// the base address register. Small (0/4/8) remainders use the immediate.
  llvm::Error appendScratchStore(std::vector<uint8_t> &TB,
                                  unsigned VDataVGPR, uint32_t Offset,
                                  unsigned TempSGPR);

  /// Emit a scratch_load_dword, using SADDR mode if Offset > 4095.
  llvm::Error appendScratchLoad(std::vector<uint8_t> &TB,
                                 unsigned VDstVGPR, uint32_t Offset,
                                 unsigned TempSGPR);

private:
  TrampolineEmitter() = default;

  llvm::Expected<std::vector<uint8_t>>
  emitBodyForPath(const InstrumentationSite &Site,
                  const InstrumentationPlan &Plan,
                  const ScratchRegisters &Scratch,
                  const TraceConfig &Trace,
                  bool UseRelay,
                  unsigned RetAddrSGPRPair,
                  uint32_t SiteIdx);

  ISAEncoder *Enc = nullptr;

  std::vector<uint8_t> CountingBytes;
  std::vector<uint8_t> LDSCountingBytes;
  std::vector<uint8_t> AtomicBytesAtomic;
  std::vector<uint8_t> AtomicBytesNonAtomic;
};

} // namespace aegisbit

#endif // AEGISBIT_TRAMPOLINE_EMITTER_H
