// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_LIVENESS_H
#define WaveASM_TRANSFORMS_LIVENESS_H

#include "mlir/IR/Value.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace waveasm {

//===----------------------------------------------------------------------===//
// Live Range
//===----------------------------------------------------------------------===//

/// Represents the lifetime of a virtual register in the instruction stream.
struct LiveRange {
  /// The virtual register value
  mlir::Value reg;

  /// Instruction index of definition
  int64_t start = 0;

  /// Instruction index of last use
  int64_t end = 0;

  /// Number of consecutive registers (for register ranges)
  int64_t size = 1;

  /// Required physical alignment (1, 2, 4, etc.)
  int64_t alignment = 1;

  /// Register class (VGPR or SGPR)
  RegClass regClass = RegClass::VGPR;

  /// Tied equivalence class ID. Values that must share the same physical
  /// register (loop block args, results, condition iter_args, MFMA acc ties)
  /// belong to the same class. -1 means the value is standalone (not tied).
  /// The pressure computation counts each class only once; the allocator
  /// uses this to enforce coalescing constraints.
  int64_t tiedClassId = -1;

  /// Check if this range overlaps with another
  bool overlaps(const LiveRange &other) const {
    return !(end < other.start || other.end < start);
  }

  /// Check if an instruction index is within this range
  bool contains(int64_t point) const { return start <= point && point <= end; }

  /// Get the duration of this range
  int64_t length() const { return end - start + 1; }

  /// Whether this value is tied to other values sharing a physical register.
  bool isTied() const { return tiedClassId >= 0; }

  /// Validate invariants: end >= start, size > 0, alignment > 0
  bool isValid() const { return end >= start && size > 0 && alignment > 0; }

  /// Assert that invariants hold (for debugging)
  void assertValid() const {
    assert(end >= start && "Invalid live range: end < start");
    assert(size > 0 && "Invalid live range: size must be positive");
    assert(alignment > 0 && "Invalid live range: alignment must be positive");
  }
};

//===----------------------------------------------------------------------===//
// Tied Value Equivalence Classes
//===----------------------------------------------------------------------===//

/// A single equivalence class of values that share a physical register.
///
/// Loop block args, loop results, condition iter_args, and init args are all
/// tied to the same physical register. MFMA accumulator ties also create
/// equivalence classes. Instead of zeroing sizes on secondary members (which
/// violates LiveRange invariants), we group them into classes and let the
/// pressure computation count each class exactly once.
struct TiedClass {
  /// Unique ID for this class (index into TiedValueClasses::classes).
  int64_t id = 0;

  /// All values in this class.
  llvm::SmallVector<mlir::Value> members;

  /// The canonical representative (the value whose range envelope is used
  /// for pressure). Typically the loop body block arg.
  mlir::Value canonical;

  /// Physical register size for this class (all members have the same size).
  int64_t size = 1;

  /// Alignment requirement.
  int64_t alignment = 1;

  /// Register class.
  RegClass regClass = RegClass::VGPR;

  /// Envelope range: the union of all member live ranges.
  /// Pressure is counted once over [envelopeStart, envelopeEnd].
  int64_t envelopeStart = 0;
  int64_t envelopeEnd = 0;
};

/// Collection of all tied equivalence classes for a program.
///
/// Built during liveness analysis. Consumed by both the pressure sweep
/// (to avoid double-counting) and the register allocator (for tie
/// constraints). This is the single source of truth for tied-value
/// relationships, replacing ad-hoc size zeroing and ad-hoc tiedPairs maps.
struct TiedValueClasses {
  /// All equivalence classes.
  llvm::SmallVector<TiedClass> classes;

  /// Map from Value to its class ID. Values not in any class are absent.
  llvm::DenseMap<mlir::Value, int64_t> valueToClassId;

  /// Map from Value to the value it should be tied to during allocation.
  /// This is the flattened "result -> operand" map that the allocator needs.
  /// For a class {init_arg, block_arg, iter_arg, loop_result}, this contains:
  ///   block_arg -> init_arg
  ///   iter_arg  -> block_arg
  ///   loop_result -> block_arg
  llvm::DenseMap<mlir::Value, mlir::Value> tiedPairs;

  /// Get the class for a value, or nullptr if the value is standalone.
  const TiedClass *getClass(mlir::Value value) const {
    auto it = valueToClassId.find(value);
    if (it == valueToClassId.end())
      return nullptr;
    return &classes[it->second];
  }

  /// Check whether a value is the canonical representative of its class.
  bool isCanonical(mlir::Value value) const {
    auto it = valueToClassId.find(value);
    if (it == valueToClassId.end())
      return true; // standalone values are trivially canonical
    return classes[it->second].canonical == value;
  }
};

//===----------------------------------------------------------------------===//
// Liveness Information
//===----------------------------------------------------------------------===//

/// Complete liveness analysis results
struct LivenessInfo {
  /// Live ranges for all virtual registers
  llvm::DenseMap<mlir::Value, LiveRange> ranges;

  /// VGPR ranges sorted by start point (for linear scan)
  llvm::SmallVector<LiveRange> vregRanges;

  /// SGPR ranges sorted by start point (for linear scan)
  llvm::SmallVector<LiveRange> sregRanges;

  /// AGPR ranges sorted by start point (for linear scan)
  llvm::SmallVector<LiveRange> aregRanges;

  /// Tied value equivalence classes. Built during liveness analysis.
  /// Used by pressure computation (count each class once) and by the
  /// register allocator (for tie constraints).
  TiedValueClasses tiedClasses;

  /// Definition points for each register
  llvm::DenseMap<mlir::Value, int64_t> defPoints;

  /// Use points for each register
  llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> usePoints;

  /// Maximum VGPR pressure (peak overlapping VGPRs)
  int64_t maxVRegPressure = 0;

  /// Maximum SGPR pressure (peak overlapping SGPRs)
  int64_t maxSRegPressure = 0;

  /// Maximum AGPR pressure (peak overlapping AGPRs)
  int64_t maxARegPressure = 0;

  /// Get live range for a value
  const LiveRange *getRange(mlir::Value value) const {
    auto it = ranges.find(value);
    return it != ranges.end() ? &it->second : nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Region Utilities
//===----------------------------------------------------------------------===//

/// Collect all operations in a block and its nested regions into a flat list.
/// Operations are visited in pre-order: parent first, then children.
/// This is used by liveness analysis, hazard mitigation, and other passes
/// that need a linearized view of the instruction stream.
void collectOpsRecursive(mlir::Block &block,
                         llvm::SmallVectorImpl<mlir::Operation *> &ops);

//===----------------------------------------------------------------------===//
// Liveness Analysis Functions
//===----------------------------------------------------------------------===//

/// Compute liveness information for a kernel program
/// @param program The kernel program to analyze
/// @return Complete liveness information
LivenessInfo computeLiveness(ProgramOp program);

/// Compute maximum register pressure using sweep algorithm.
/// This is the class-aware version: values belonging to tied equivalence
/// classes are counted once per class (using the class envelope), not once
/// per member. Standalone values are counted normally.
int64_t computeMaxPressure(llvm::ArrayRef<LiveRange> ranges,
                           const TiedValueClasses &tiedClasses,
                           int64_t *peakPoint = nullptr);

/// Dump detailed peak pressure information for diagnostics.
/// Shows which values are live at the peak point, categorized by defining op.
/// Output is gated behind LLVM_DEBUG (use -debug-only=waveasm-liveness).
void dumpPeakPressureInfo(const LivenessInfo &info,
                          llvm::ArrayRef<mlir::Operation *> ops,
                          RegClass regClass);

/// Validate that the program is in SSA form
/// Each virtual register should be defined exactly once
/// (with exceptions for loop control and accumulator registers)
mlir::LogicalResult validateSSA(ProgramOp program,
                                llvm::DenseSet<int64_t> loopControlSRegs = {},
                                llvm::DenseSet<int64_t> accumulatorVRegs = {});

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_LIVENESS_H
