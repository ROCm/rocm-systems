// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_REGALLOC_H
#define WaveASM_TRANSFORMS_REGALLOC_H

#include "mlir/IR/Value.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Liveness.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <memory>
#include <optional>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Allocation Statistics
//===----------------------------------------------------------------------===//

/// Statistics from register allocation
struct AllocationStats {
  int64_t peakVGPRs = 0;
  int64_t peakSGPRs = 0;
  int64_t peakAGPRs = 0;
  int64_t totalVRegs = 0;
  int64_t totalSRegs = 0;
  int64_t totalARegs = 0;
  int64_t rangesAllocated = 0;
  int64_t rangesExpired = 0;
};

//===----------------------------------------------------------------------===//
// Physical Mapping (Pure SSA)
//===----------------------------------------------------------------------===//

/// Maps SSA Values to physical register indices
class PhysicalMapping {
public:
  /// Get physical register index for a Value
  int64_t getPhysReg(mlir::Value value) const {
    auto it = valueToPhysReg.find(value);
    return it != valueToPhysReg.end() ? it->second : -1;
  }

  /// Set mapping for a Value
  void setPhysReg(mlir::Value value, int64_t physIdx) {
    valueToPhysReg[value] = physIdx;
  }

  /// Check if a Value is mapped
  bool hasMapping(mlir::Value value) const {
    return valueToPhysReg.contains(value);
  }

  /// Direct access for LinearScanRegAlloc
  llvm::DenseMap<mlir::Value, int64_t> valueToPhysReg;
};

//===----------------------------------------------------------------------===//
// Register Pool
//===----------------------------------------------------------------------===//

/// Pool of available physical registers.
/// Uses a BitVector for O(1) per-register operations and cache-friendly
/// scanning. Typical GPU register files (256 VGPRs, 102 SGPRs, 256 AGPRs)
/// fit in a few 64-bit words.
class RegPool {
public:
  RegPool(RegClass regClass, int64_t maxRegs,
          const llvm::DenseSet<int64_t> &reserved)
      : regClass(regClass), maxRegs(maxRegs), free(maxRegs, true) {
    for (int64_t reg : reserved) {
      if (reg >= 0 && reg < maxRegs) {
        free.reset(reg);
        ++currentUsage;
      }
    }
    updatePeak();
  }

  /// O(1) free-check via bit test.
  bool isFree(int64_t reg) const {
    return reg >= 0 && reg < maxRegs && free.test(reg);
  }

  /// Reserve a specific register range (for precoloring / re-reservation).
  void reserve(int64_t reg, int64_t size) {
    for (int64_t idx = 0; idx < size; ++idx) {
      int64_t target = reg + idx;
      if (target >= 0 && target < maxRegs && free.test(target)) {
        free.reset(target);
        ++currentUsage;
      }
    }
    updatePeak();
  }

  /// Allocate the lowest-numbered free register.
  /// Returns -1 if no register is available.
  int64_t allocSingle() {
    int64_t idx = free.find_first();
    if (idx < 0)
      return -1;
    free.reset(idx);
    ++currentUsage;
    updatePeak();
    return idx;
  }

  /// Allocate a contiguous range of registers with the given alignment.
  /// Scans by alignment stride for O(maxRegs/alignment) candidate checks.
  /// Returns base register index, or -1 if allocation fails.
  int64_t allocRange(int64_t size, int64_t alignment) {
    if (size <= 0)
      return -1;
    assert(alignment > 0 && "alignment must be positive");

    for (int64_t candidate = 0; candidate + size <= maxRegs;
         candidate += alignment) {
      bool allFree = true;
      for (int64_t offset = 0; offset < size; ++offset) {
        if (!free.test(candidate + offset)) {
          allFree = false;
          break;
        }
      }
      if (allFree) {
        for (int64_t offset = 0; offset < size; ++offset)
          free.reset(candidate + offset);
        currentUsage += size;
        updatePeak();
        return candidate;
      }
    }
    return -1;
  }

  /// Allocate the highest-numbered free register below ceiling.
  int64_t allocSingleFromTop(int64_t ceiling = -1) {
    int64_t cap = (ceiling > 0 && ceiling <= maxRegs) ? ceiling : maxRegs;
    for (int64_t reg = cap - 1; reg >= 0; --reg) {
      if (free.test(reg)) {
        free.reset(reg);
        ++currentUsage;
        updatePeak();
        return reg;
      }
    }
    return -1;
  }

  /// Allocate a contiguous range from the top of a capped region.
  /// Scans from `ceiling` downward to pack long-lived values at the top
  /// of the USED range, not the top of the entire register file.
  int64_t allocRangeFromTop(int64_t size, int64_t alignment,
                            int64_t ceiling = -1) {
    if (size <= 0)
      return -1;
    assert(alignment > 0 && "alignment must be positive");

    int64_t cap = (ceiling > 0 && ceiling <= maxRegs) ? ceiling : maxRegs;
    if (size > cap)
      return -1;
    int64_t highestBase = ((cap - size) / alignment) * alignment;
    for (int64_t candidate = highestBase; candidate >= 0;
         candidate -= alignment) {
      bool allFree = true;
      for (int64_t offset = 0; offset < size; ++offset) {
        if (!free.test(candidate + offset)) {
          allFree = false;
          break;
        }
      }
      if (allFree) {
        for (int64_t offset = 0; offset < size; ++offset)
          free.reset(candidate + offset);
        currentUsage += size;
        updatePeak();
        return candidate;
      }
    }
    return -1;
  }

  /// Free a single register back to the pool.
  void freeSingle(int64_t reg) {
    if (reg < 0 || reg >= maxRegs || free.test(reg))
      return;
    free.set(reg);
    --currentUsage;
  }

  /// Free a contiguous range of registers.
  void freeRange(int64_t base, int64_t size) {
    for (int64_t offset = 0; offset < size; ++offset) {
      freeSingle(base + offset);
    }
  }

  int64_t getPeakUsage() const { return peak; }

  int64_t getCurrentUsage() const { return currentUsage; }

  RegClass getRegClass() const { return regClass; }

private:
  void updatePeak() { peak = std::max(peak, currentUsage); }

  RegClass regClass;
  int64_t maxRegs;
  llvm::BitVector free;
  int64_t currentUsage = 0;
  int64_t peak = 0;
};

//===----------------------------------------------------------------------===//
// Allocation Strategy
//===----------------------------------------------------------------------===//

/// Pluggable strategy for choosing how to allocate a physical register for a
/// given live range. Subclasses override `allocate()` to implement heuristics
/// such as bidirectional allocation, best-fit, etc.
///
/// Returning std::nullopt signals "no preference" and lets the caller fall
/// through to the default bottom-up allocation.
class AllocationStrategy {
public:
  virtual ~AllocationStrategy() = default;

  /// Try to allocate a physical register for the given live range.
  /// Returns the allocated register index, or std::nullopt to fall back to
  /// the default bottom-up allocation.
  virtual std::optional<int64_t> allocate(RegPool &pool, const LiveRange &range,
                                          llvm::ArrayRef<LiveRange> allRanges,
                                          int64_t maxPressure) = 0;
};

/// Allocate long-lived multi-register VGPR ranges from the top of the
/// expected register usage and short-lived ranges from the bottom. This
/// separates interleaved buffer_load (prefetch) and ds_read (consumed)
/// destinations into contiguous regions, reducing fragmentation.
class BidirectionalStrategy : public AllocationStrategy {
public:
  explicit BidirectionalStrategy(int64_t thresholdPct = 75)
      : thresholdPct(thresholdPct) {}

  std::optional<int64_t> allocate(RegPool &pool, const LiveRange &range,
                                  llvm::ArrayRef<LiveRange> allRanges,
                                  int64_t maxPressure) override;

private:
  int64_t thresholdPct;
};

//===----------------------------------------------------------------------===//
// Linear Scan Register Allocator (Pure SSA)
//===----------------------------------------------------------------------===//

/// Linear scan register allocator for pure SSA IR
class LinearScanRegAlloc {
public:
  LinearScanRegAlloc(int64_t maxVGPRs, int64_t maxSGPRs, int64_t maxAGPRs,
                     const llvm::DenseSet<int64_t> &reservedVGPRs,
                     const llvm::DenseSet<int64_t> &reservedSGPRs,
                     const llvm::DenseSet<int64_t> &reservedAGPRs)
      : maxVGPRs(maxVGPRs), maxSGPRs(maxSGPRs), maxAGPRs(maxAGPRs),
        reservedVGPRs(reservedVGPRs), reservedSGPRs(reservedSGPRs),
        reservedAGPRs(reservedAGPRs) {}

  /// Precolor a Value to a specific physical register (for ABI args)
  void precolorValue(mlir::Value value, int64_t physIdx) {
    precoloredValues[value] = physIdx;
  }

  /// Add a tied operand constraint: result must get same physical reg as
  /// operand Used for MFMA accumulator tying where result overwrites the
  /// accumulator
  void addTiedOperand(mlir::Value result, mlir::Value operand) {
    tiedOperands[result] = operand;
  }

  /// Set the allocation strategy used for VGPR ranges. When null (or not
  /// set), all ranges use bottom-up allocation.
  void setVGPRStrategy(std::unique_ptr<AllocationStrategy> strategy) {
    vgprStrategy = std::move(strategy);
  }

  /// Run allocation on a kernel program
  /// Returns the physical mapping and statistics, or failure if allocation
  /// fails
  mlir::FailureOr<std::pair<PhysicalMapping, AllocationStats>>
  allocate(ProgramOp program);

private:
  /// Process active ranges, expiring those that end before currentPoint
  void expireRanges(
      llvm::SmallVector<std::tuple<int64_t, LiveRange, int64_t>> &active,
      int64_t currentPoint, RegPool &pool, AllocationStats &stats);

  int64_t maxVGPRs;
  int64_t maxSGPRs;
  int64_t maxAGPRs;
  llvm::DenseSet<int64_t> reservedVGPRs;
  llvm::DenseSet<int64_t> reservedSGPRs;
  llvm::DenseSet<int64_t> reservedAGPRs;
  llvm::DenseMap<mlir::Value, int64_t> precoloredValues;
  llvm::DenseMap<mlir::Value, mlir::Value> tiedOperands; // result -> operand
  std::unique_ptr<AllocationStrategy> vgprStrategy;
};

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_REGALLOC_H
