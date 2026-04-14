// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// ActiveRange - Represents an allocated range in the active list
//===----------------------------------------------------------------------===//

namespace {

/// Represents an active live range during linear scan register allocation.
/// The active list is sorted by endPoint to efficiently expire ranges.
/// When a range ends (endPoint < currentPoint), its physical register is freed.
struct ActiveRange {
  int64_t endPoint; ///< Program point where this range ends (exclusive)
  LiveRange range;  ///< The original live range (contains Value, start, size)
  int64_t physReg;  ///< Allocated physical register index

  bool operator<(const ActiveRange &other) const {
    return endPoint < other.endPoint;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Expire ranges that ended before currentPoint, returning registers to pool.
static void expireRanges(llvm::SmallVectorImpl<ActiveRange> &active,
                         int64_t currentPoint, RegPool &pool,
                         AllocationStats &stats) {
  active.erase(std::remove_if(active.begin(), active.end(),
                              [&](const ActiveRange &entry) {
                                if (entry.endPoint < currentPoint) {
                                  assert(entry.range.size > 0 &&
                                         "Cannot free zero-sized range");
                                  pool.freeRange(entry.physReg,
                                                 entry.range.size);
                                  stats.rangesExpired++;
                                  return true;
                                }
                                return false;
                              }),
               active.end());
}

/// Insert a new active range while maintaining sorted order by end point.
/// Uses binary search for O(log n) insertion position finding.
static void insertActiveRange(llvm::SmallVectorImpl<ActiveRange> &active,
                              ActiveRange newRange) {
  auto insertPos = std::lower_bound(active.begin(), active.end(), newRange);
  active.insert(insertPos, newRange);
}

/// Try to allocate a physical register from the pool (lowest-first).
static std::optional<int64_t> tryAllocate(RegPool &pool, int64_t size,
                                          int64_t alignment) {
  int64_t physReg =
      (size == 1) ? pool.allocSingle() : pool.allocRange(size, alignment);
  if (physReg < 0)
    return std::nullopt;
  return physReg;
}

//===----------------------------------------------------------------------===//
// Concrete Allocation Strategies
//===----------------------------------------------------------------------===//

std::optional<int64_t>
BidirectionalStrategy::allocate(RegPool &pool, const LiveRange &range,
                                llvm::ArrayRef<LiveRange> allRanges,
                                int64_t maxPressure) {
  if (pool.getRegClass() != RegClass::VGPR || range.size <= 1)
    return std::nullopt;

  int64_t rangeLength = range.end - range.start;
  // Ranges whose length exceeds 75% of the program span are allocated from
  // the top. This targets buffer_load prefetch values (which span almost the
  // entire loop body) while leaving ds_read values (consumed within one half)
  // at the bottom. The ceiling is maxPressure (peak simultaneous VGPRs from
  // liveness), not maxRegs, to avoid allocating into the AGPR region.
  int64_t programEnd = 0;
  for (const auto &lr : allRanges)
    programEnd = std::max(programEnd, lr.end);
  int64_t threshold = (programEnd * thresholdPct) / 100;
  if (rangeLength > threshold) {
    int64_t physReg =
        pool.allocRangeFromTop(range.size, range.alignment, maxPressure);
    if (physReg >= 0)
      return physReg;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Register Class Allocation
//===----------------------------------------------------------------------===//

/// Allocate registers for a single register class (VGPR, SGPR, or AGPR).
/// This is the core linear scan algorithm, parameterized by register class.
/// An optional AllocationStrategy is consulted before the default bottom-up
/// allocation; returning std::nullopt falls through to tryAllocate.
static LogicalResult allocateRegClass(
    ArrayRef<LiveRange> ranges, RegPool &pool, PhysicalMapping &mapping,
    AllocationStats &stats, const llvm::DenseMap<Value, Value> &tiedOperands,
    const llvm::DenseMap<Value, int64_t> &precoloredValues,
    llvm::StringRef regClassName, ProgramOp program, int64_t maxRegs,
    int64_t maxPressure, AllocationStrategy *strategy) {

  llvm::SmallVector<ActiveRange> active;

  for (const LiveRange &range : ranges) {
    // Skip precolored values - they're already mapped
    if (precoloredValues.contains(range.reg)) {
      continue;
    }

    // Expire finished ranges, returning registers to the pool
    expireRanges(active, range.start, pool, stats);

    std::optional<int64_t> physReg;

    // Check if this value is tied to another value (must share same phys reg).
    // This handles MFMA accumulator tying AND while loop block arg coalescing.
    {
      auto tiedIt = tiedOperands.find(range.reg);
      if (tiedIt != tiedOperands.end()) {
        Value tiedTo = tiedIt->second;
        auto mappingIt = mapping.valueToPhysReg.find(tiedTo);
        if (mappingIt != mapping.valueToPhysReg.end()) {
          physReg = mappingIt->second;
          mapping.valueToPhysReg[range.reg] = *physReg;

          // Extend the physical register's lifetime to cover the tied result.
          // The tied-to operand may have a shorter lifetime (e.g., %55 ends
          // at op2), but the tied result (%56) may live longer (used in
          // iteration 2). Without this extension, the physical register would
          // be freed too early when the tied-to operand expires.
          bool foundInActive = false;
          for (size_t i = 0; i < active.size(); ++i) {
            if (active[i].physReg == *physReg) {
              foundInActive = true;
              if (range.end > active[i].endPoint) {
                // Update end point and re-sort the affected portion
                active[i].endPoint = range.end;
                active[i].range = range;
                // Re-sort: since we only increased one element's key,
                // bubble it forward to maintain sorted order by endPoint
                while (i + 1 < active.size() &&
                       active[i].endPoint > active[i + 1].endPoint) {
                  std::swap(active[i], active[i + 1]);
                  ++i;
                }
              }
              break;
            }
          }

          if (!foundInActive) {
            // Two cases reach here:
            // (a) Precolored tying (MFMA): the tied-to value is precolored
            //     and was never in the active list. Its physReg is already
            //     reserved in the pool. pool.reserve is a safe no-op.
            // (b) Loop boundary: the tied-to virtual value expired from
            //     active and its registers were returned to the pool.
            //     The physReg MUST still be free (not re-allocated).
            bool tiedToPrecolored = precoloredValues.contains(tiedTo);
            assert((tiedToPrecolored || pool.isFree(*physReg)) &&
                   "Tied register was re-allocated before re-reservation");
            pool.reserve(*physReg, range.size);
            insertActiveRange(active, {range.end, range, *physReg});
          }

          stats.rangesAllocated++;
          continue;
        }
        // If tied-to not yet allocated, fall through to normal allocation
      }
    }

    // Consult the pluggable strategy (if any) before the bottom-up fallback.
    if (strategy)
      physReg = strategy->allocate(pool, range, ranges, maxPressure);
    if (!physReg)
      physReg = tryAllocate(pool, range.size, range.alignment);

    if (!physReg) {
      InFlightDiagnostic diag = mlir::emitError(range.reg.getLoc())
                                << "Failed to allocate " << regClassName
                                << ": kernel requires " << maxPressure
                                << " but only " << maxRegs << " are available";
      diag.attachNote(range.reg.getLoc())
          << "Register spilling is not supported; reduce register pressure "
             "(e.g., smaller tile sizes, fewer unrolled iterations).";
      return failure();
    }

    // Record mapping: Value -> physical register
    mapping.valueToPhysReg[range.reg] = *physReg;

    // Add to active list, maintaining sorted order by end point
    insertActiveRange(active, {range.end, range, *physReg});

    stats.rangesAllocated++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main Allocation Algorithm (Pure SSA)
//===----------------------------------------------------------------------===//

FailureOr<std::pair<PhysicalMapping, AllocationStats>>
LinearScanRegAlloc::allocate(ProgramOp program) {
  PhysicalMapping mapping;
  AllocationStats stats;

  // Step 1: Validate SSA
  if (failed(validateSSA(program))) {
    return program.emitOpError() << "SSA validation failed before allocation";
  }

  // Step 2: Compute liveness (builds tied equivalence classes)
  LivenessInfo liveness = computeLiveness(program);

  // Merge loop tied pairs from liveness into the allocator's tiedOperands.
  // The liveness analysis builds TiedValueClasses with a tiedPairs map
  // that captures loop block_arg/init_arg/iter_arg/result relationships.
  // MFMA ties were added externally via addTiedOperand() and are already
  // in tiedOperands; loop ties come from liveness and are merged here.
  for (const auto &[result, operand] : liveness.tiedClasses.tiedPairs) {
    if (!tiedOperands.contains(result)) {
      tiedOperands[result] = operand;
    }
  }

  stats.totalVRegs = liveness.vregRanges.size();
  stats.totalSRegs = liveness.sregRanges.size();
  stats.totalARegs = liveness.aregRanges.size();

  // Step 3: Create register pools with reserved registers
  RegPool vgprPool(RegClass::VGPR, maxVGPRs, reservedVGPRs);
  RegPool sgprPool(RegClass::SGPR, maxSGPRs, reservedSGPRs);
  RegPool agprPool(RegClass::AGPR, maxAGPRs, reservedAGPRs);

  // Step 4: Handle precolored values (from ABI args like tid, kernarg)
  for (const auto &[value, physIdx] : precoloredValues) {
    if (isVGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      vgprPool.reserve(physIdx, getRegSize(value.getType()));
    } else if (isAGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      agprPool.reserve(physIdx, getRegSize(value.getType()));
    } else if (isSGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      sgprPool.reserve(physIdx, getRegSize(value.getType()));
    }
  }

  // Step 5: Allocate VGPRs using linear scan (with optional strategy)
  if (failed(allocateRegClass(liveness.vregRanges, vgprPool, mapping, stats,
                              tiedOperands, precoloredValues, "VGPR", program,
                              maxVGPRs, liveness.maxVRegPressure,
                              vgprStrategy.get()))) {
    return failure();
  }
  stats.peakVGPRs = vgprPool.getPeakUsage();

  // Step 6: Allocate SGPRs using linear scan (no strategy, always bottom-up)
  if (failed(allocateRegClass(liveness.sregRanges, sgprPool, mapping, stats,
                              tiedOperands, precoloredValues, "SGPR", program,
                              maxSGPRs, liveness.maxSRegPressure,
                              /*strategy=*/nullptr))) {
    return failure();
  }
  stats.peakSGPRs = sgprPool.getPeakUsage();

  // Step 7: Allocate AGPRs using linear scan (no strategy, always bottom-up)
  if (failed(allocateRegClass(liveness.aregRanges, agprPool, mapping, stats,
                              tiedOperands, precoloredValues, "AGPR", program,
                              maxAGPRs, liveness.maxARegPressure,
                              /*strategy=*/nullptr))) {
    return failure();
  }
  stats.peakAGPRs = agprPool.getPeakUsage();

  return std::make_pair(mapping, stats);
}
