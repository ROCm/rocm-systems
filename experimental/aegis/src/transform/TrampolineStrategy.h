//===-- TrampolineStrategy.h - Abstract trampoline strategy ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal header (not part of the public API). Declares the
/// `TrampolineStrategy` interface and the `BridgeInputs` aggregate that
/// TrampolineBridge uses to dispatch instrumented builds. Each concrete
/// strategy — SharedBody, SwapPCSharedBody, Adaptive — lives in its own
/// translation unit under src/transform/.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_SRC_TRANSFORM_TRAMPOLINE_STRATEGY_H
#define AEGISBIT_SRC_TRANSFORM_TRAMPOLINE_STRATEGY_H

#include "aegisbit/InstrumentationPlan.h"
#include "aegisbit/TrampolineBridge.h"
#include "aegisbit/TrampolineTypes.h"
#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aegisbit {

class ISAEncoder;
class TrampolineEmitter;

/// Aggregate of all inputs a strategy needs to produce a BridgeResult.
/// Keeping them in one struct makes strategy signatures stable and makes
/// unit tests easy to construct.
struct BridgeInputs {
  ISAEncoder *Enc = nullptr;            ///< Owned by TrampolineBridge
  TrampolineEmitter *Emitter = nullptr; ///< Owned by TrampolineBridge
  llvm::ArrayRef<uint8_t> Code;         ///< Original kernel bytes
  uint64_t BaseAddr = 0;                ///< Kernel start address within .text
  uint64_t TextSectionSize = 0;         ///< Effective end of .text for layout
  const std::vector<InstrumentationSite> *Sites = nullptr;
  const InstrumentationPlan *Plan = nullptr;
  const ScratchRegisters *Scratch = nullptr;
  const TraceConfig *Trace = nullptr;
  unsigned RetAddrSGPRPair = 0;         ///< Encoded SGPR pair or 0 (ZeroSGPR)
  uint64_t PreKernelSpace = 0;          ///< Pre-kernel relay stub slack
  const TrampolineBridge::OccupiedRanges *Occupied = nullptr;
};

/// Abstract strategy that emits a trampoline layout for a single kernel.
/// Concrete subclasses: SharedBodyStrategy, SwapPCSharedBodyStrategy,
/// AdaptiveStrategy.
class TrampolineStrategy {
public:
  virtual ~TrampolineStrategy();

  /// Produce a BridgeResult for the given inputs.
  virtual llvm::Expected<BridgeResult> build(const BridgeInputs &In) = 0;
};

/// Factory selecting a strategy based on Plan.Jump.
std::unique_ptr<TrampolineStrategy>
createStrategyForPlan(const InstrumentationPlan &Plan);

} // namespace aegisbit

#endif // AEGISBIT_SRC_TRANSFORM_TRAMPOLINE_STRATEGY_H
