//===-- BridgeHelpers.h - Shared TrampolineBridge helpers --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal helpers shared by the three bridge strategies (SharedBody,
/// SwapPCSharedBody, Adaptive). These functions factor the pieces that were
/// previously duplicated across the three big `buildXxxPath` functions.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_SRC_TRANSFORM_BRIDGE_HELPERS_H
#define AEGISBIT_SRC_TRANSFORM_BRIDGE_HELPERS_H

#include "aegisbit/TrampolineBridge.h"
#include "aegisbit/TrampolineTypes.h"
#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace aegisbit {

class ISAEncoder;
class TrampolineEmitter;

/// Align `TextSectionSize` up to 256 bytes and step past any occupied
/// range that overlaps the aligned start. Used by every shared-body path
/// to pick the first byte of the instrumentation island.
uint64_t alignIslandStart(uint64_t TextSectionSize,
                          const TrampolineBridge::OccupiedRanges &Occupied);

/// Summary statistics over a site vector used by all strategies to size
/// the shared body(ies).
struct SiteSummary {
  unsigned MaxPreSpillVmWait = 0;
  bool HasVMEM = false;
  bool HasLDS = false;
};
SiteSummary summarizeSites(const std::vector<InstrumentationSite> &Sites);

/// Emit the VMEM/LDS shared body blobs.
/// \param VMEMBodyOffsetFromTail  Offset (in bytes) from the end of the
///        per-site dispatch jump block to the VMEM body start. Passed to
///        the emitter so its short s_branch can encode the jump correctly.
struct SharedBodies {
  std::vector<uint8_t> VMEM;
  std::vector<uint8_t> LDS;
};
llvm::Expected<SharedBodies>
emitSharedBodies(TrampolineEmitter &Emitter, const ScratchRegisters &Scratch,
                 const TraceConfig &Trace, unsigned RetAddrSGPRPair,
                 const SiteSummary &Summary);

/// Build the per-site dispatch table (12 bytes per site).
/// \param IslandStart  Absolute address of the dispatch table in memory.
/// \param VMEMBodyAbs  Absolute address of the VMEM shared body.
/// \param LDSBodyAbs   Absolute address of the LDS shared body.
llvm::Expected<std::vector<uint8_t>> buildDispatchTable(
    TrampolineEmitter &Emitter, const ScratchRegisters &Scratch,
    const std::vector<InstrumentationSite> &Sites, uint64_t IslandStart,
    uint64_t DispatchTableOffsetInIsland, uint64_t VMEMBodyAbs,
    uint64_t LDSBodyAbs);

/// Build the per-site return table. Each entry contains a copy of the
/// displaced instruction followed by s_setpc_b64 back to the kernel.
llvm::Expected<std::vector<uint8_t>> buildReturnTable(
    TrampolineEmitter &Emitter, const ScratchRegisters &Scratch,
    const std::vector<InstrumentationSite> &Sites,
    llvm::ArrayRef<uint8_t> Code, unsigned RetAddrSGPRPair);

} // namespace aegisbit

#endif // AEGISBIT_SRC_TRANSFORM_BRIDGE_HELPERS_H
