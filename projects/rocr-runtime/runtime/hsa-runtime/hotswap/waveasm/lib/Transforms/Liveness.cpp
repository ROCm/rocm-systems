// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "waveasm-liveness"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Iter-arg classification helpers
//===----------------------------------------------------------------------===//

/// Return true when iterArg at position i is a block_arg of bodyBlock at a
/// different position (the LDS double-buffer ping-pong swap pattern).
static bool isSwapPatternIterArg(Value iterArg, Block &bodyBlock, unsigned i) {
  if (auto ba = dyn_cast<BlockArgument>(iterArg))
    if (ba.getOwner() == &bodyBlock && ba.getArgNumber() != i)
      return true;
  return false;
}

/// Detect a write-after-read (WAR) hazard between an iter_arg and the
/// block_arg it feeds back into.
///
/// If the iter_arg is defined at a point where the block_arg still has
/// subsequent uses, tying them to the same physical register would let
/// the in-place update clobber a value that later instructions still need.
///
/// This is critical for unrolled loops where CSE can merge an
/// affine.apply result (e.g. arg8+2 for a bounds check) with the IV
/// increment (also arg8+2 for step=2).  After merging, the single
/// S_ADD_U32 is placed in the middle of the body; if it shares a
/// register with the IV block_arg, the second unrolled copy's use of
/// the original IV reads the wrong value.
static bool hasWARHazard(Value iterArg, Value blockArg,
                         const LivenessInfo &info) {
  if (isa<BlockArgument>(iterArg))
    return false;
  auto *defOp = iterArg.getDefiningOp();
  assert(defOp && "non-BlockArgument Value must have a defining op");

  // Single-element buffer loads (ubyte/sbyte/ushort/sshort): the block_arg
  // is consumed indirectly through vector.bitcast / vector.extract that
  // share the same physical register. These transitive uses don't appear
  // in usePoints, so unconditionally untie to be safe.
  auto opName = defOp->getName().getStringRef();
  if (opName.contains("buffer_load") && !opName.contains("_lds")) {
    if (opName.contains("_ubyte") || opName.contains("_sbyte") ||
        opName.contains("_ushort") || opName.contains("_sshort")) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  WAR hazard (single-element load): " << opName << "\n");
      return true;
    }
  }

  // General case: check for def/use overlap. If the iter_arg is defined
  // at a point where block_arg still has subsequent uses (strictly after,
  // since same-point means read-before-write), tying them creates a WAR
  // hazard.
  auto iterDefIt = info.defPoints.find(iterArg);
  auto baUseIt = info.usePoints.find(blockArg);
  if (iterDefIt != info.defPoints.end() && baUseIt != info.usePoints.end()) {
    int64_t iterDef = iterDefIt->second;
    for (int64_t usePoint : baUseIt->second) {
      if (usePoint > iterDef)
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Region Utilities
//===----------------------------------------------------------------------===//

void collectOpsRecursive(Block &block,
                         llvm::SmallVectorImpl<Operation *> &ops) {
  for (Operation &op : block) {
    ops.push_back(&op);
    for (Region &region : op.getRegions()) {
      for (Block &nestedBlock : region) {
        collectOpsRecursive(nestedBlock, ops);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pressure Computation
//===----------------------------------------------------------------------===//

int64_t computeMaxPressure(llvm::ArrayRef<LiveRange> ranges,
                           const TiedValueClasses &tiedClasses,
                           int64_t *peakPoint) {
  if (ranges.empty()) {
    if (peakPoint)
      *peakPoint = 0;
    return 0;
  }

  // Build sweep events. For tied equivalence classes, emit ONE set of
  // events using the class envelope (the union of all member ranges)
  // and the class size. For standalone values, emit events normally.
  // This ensures each physical register is counted exactly once.
  llvm::SmallVector<std::tuple<int64_t, int64_t, bool>> events;
  llvm::DenseSet<int64_t> emittedClassIds;

  for (const auto &r : ranges) {
    if (r.tiedClassId >= 0) {
      // Tied value: emit events only for the class envelope, once per class.
      if (emittedClassIds.insert(r.tiedClassId).second) {
        const TiedClass &cls = tiedClasses.classes[r.tiedClassId];
        events.push_back({cls.envelopeStart, cls.size, true});
        events.push_back({cls.envelopeEnd + 1, -cls.size, false});
      }
    } else {
      // Standalone value: emit events using its own range and size.
      events.push_back({r.start, r.size, true});
      events.push_back({r.end + 1, -r.size, false});
    }
  }

  // Sort by point, then ends before starts at the same point
  llvm::sort(events, [](const auto &a, const auto &b) {
    if (std::get<0>(a) != std::get<0>(b))
      return std::get<0>(a) < std::get<0>(b);
    return !std::get<2>(a) && std::get<2>(b);
  });

  int64_t currentPressure = 0;
  int64_t maxPressure = 0;
  int64_t bestPoint = 0;

  constexpr int64_t kMaxReasonablePressure = 1000000;

  for (const auto &[point, delta, isStart] : events) {
    currentPressure += delta;

    assert(currentPressure >= 0 &&
           "Negative register pressure - possible overflow or bug");
    assert(currentPressure < kMaxReasonablePressure &&
           "Register pressure exceeds reasonable bounds - possible overflow");

    if (currentPressure > maxPressure) {
      maxPressure = currentPressure;
      bestPoint = point;
    }
  }

  if (peakPoint)
    *peakPoint = bestPoint;
  return maxPressure;
}

void dumpPeakPressureInfo(const LivenessInfo &info,
                          llvm::ArrayRef<Operation *> ops, RegClass regClass) {
  llvm::ArrayRef<LiveRange> ranges;
  const char *className = "VGPR";
  if (regClass == RegClass::VGPR) {
    ranges = info.vregRanges;
    className = "VGPR";
  } else if (regClass == RegClass::SGPR) {
    ranges = info.sregRanges;
    className = "SGPR";
  } else {
    ranges = info.aregRanges;
    className = "AGPR";
  }
  if (ranges.empty())
    return;

  int64_t peakPoint = 0;
  int64_t maxPressure =
      computeMaxPressure(ranges, info.tiedClasses, &peakPoint);

  // Collect all ranges alive at the peak point
  struct LiveAtPeak {
    Value value;
    int64_t start;
    int64_t end;
    int64_t size;
    int64_t length;
  };
  llvm::SmallVector<LiveAtPeak> liveAtPeak;
  for (const auto &r : ranges) {
    if (r.start <= peakPoint && r.end >= peakPoint) {
      liveAtPeak.push_back({r.reg, r.start, r.end, r.size, r.end - r.start});
    }
  }

  llvm::sort(liveAtPeak, [](const LiveAtPeak &a, const LiveAtPeak &b) {
    return a.length > b.length;
  });

  llvm::StringRef peakOpName = "<unknown>";
  if (peakPoint >= 0 && peakPoint < static_cast<int64_t>(ops.size())) {
    peakOpName = ops[peakPoint]->getName().getStringRef();
  }

  int64_t totalRegs = 0;
  for (const auto &l : liveAtPeak)
    totalRegs += l.size;

  LLVM_DEBUG({
    llvm::dbgs()
        << "\n============================================================\n";
    llvm::dbgs() << className << " Peak Pressure Analysis\n";
    llvm::dbgs()
        << "============================================================\n";
    llvm::dbgs() << "Peak point: " << peakPoint << " (op: " << peakOpName
                 << ")\n";
    llvm::dbgs() << "Peak pressure: " << maxPressure << " " << className
                 << "s\n";
    llvm::dbgs() << "Live ranges at peak: " << liveAtPeak.size()
                 << " (total regs: " << totalRegs << ")\n\n";

    llvm::StringMap<int64_t> categoryRegs;
    llvm::StringMap<int64_t> categoryCounts;
    for (const auto &l : liveAtPeak) {
      llvm::StringRef cat = "<block_arg>";
      if (auto defOp = l.value.getDefiningOp()) {
        cat = defOp->getName().getStringRef();
      }
      categoryRegs[cat] += l.size;
      categoryCounts[cat]++;
    }

    llvm::dbgs() << "By defining op type:\n";
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>> cats;
    for (const auto &kv : categoryRegs)
      cats.push_back({kv.first(), kv.second});
    llvm::sort(
        cats, [](const auto &a, const auto &b) { return a.second > b.second; });
    for (const auto &[cat, regs] : cats) {
      llvm::dbgs() << "  " << cat << ": " << regs << " regs ("
                   << categoryCounts[cat] << " values)\n";
    }

    constexpr size_t kTopN = 10;
    llvm::dbgs() << "\nTop " << kTopN << " longest-lived ranges at peak:\n";
    for (size_t i = 0; i < std::min(liveAtPeak.size(), kTopN); ++i) {
      const auto &l = liveAtPeak[i];
      llvm::StringRef defName = "<block_arg>";
      if (auto defOp = l.value.getDefiningOp())
        defName = defOp->getName().getStringRef();
      llvm::dbgs() << "  " << (i + 1) << ". [" << l.start << ", " << l.end
                   << "] size=" << l.size << " len=" << l.length
                   << " def=" << defName << "\n";
    }
    llvm::dbgs()
        << "============================================================\n\n";
  });
}

//===----------------------------------------------------------------------===//
// Main Liveness Computation (Pure SSA)
//===----------------------------------------------------------------------===//

LivenessInfo computeLiveness(ProgramOp program) {
  LivenessInfo info;

  // Collect all operations in order, recursively walking into regions
  llvm::SmallVector<Operation *> ops;
  collectOpsRecursive(program.getBodyBlock(), ops);

  if (ops.empty())
    return info;

  // Build op-to-index map for range extension
  llvm::DenseMap<Operation *, int64_t> opToIdx;
  for (int64_t idx = 0; idx < static_cast<int64_t>(ops.size()); ++idx) {
    opToIdx[ops[idx]] = idx;
  }

  // Pass 1: Collect def and use points from instructions.
  // Also include block arguments of loop ops as definitions.
  for (int64_t idx = 0; idx < static_cast<int64_t>(ops.size()); ++idx) {
    Operation *op = ops[idx];

    // Process defs: results are definitions.
    // For LoopOp results, the def point should be AFTER the loop body,
    // not at the LoopOp itself. Loop results are only available after
    // the loop exits, so their live ranges should not overlap with the
    // loop body. Using the LoopOp index would inflate register pressure
    // by keeping these results "live" throughout the entire loop.
    if (isa<LoopOp>(op)) {
      // Find the next sibling op after this LoopOp in the parent block.
      // If there is no next sibling (loop is block-terminating), use idx + 1
      // as a synthetic "after loop" point so loop results still get def points.
      int64_t loopResultDefPoint = idx + 1;
      Operation *nextOp = op->getNextNode();
      if (nextOp) {
        auto nextIt = opToIdx.find(nextOp);
        if (nextIt != opToIdx.end()) {
          loopResultDefPoint = nextIt->second;
        }
      }
      for (Value def : op->getResults()) {
        if (isVirtualRegType(def.getType())) {
          if (!info.defPoints.contains(def)) {
            info.defPoints[def] = loopResultDefPoint;
          }
        }
      }
    } else {
      for (Value def : op->getResults()) {
        if (isVirtualRegType(def.getType())) {
          if (!info.defPoints.contains(def)) {
            info.defPoints[def] = idx;
          }
        }
      }
    }

    // Process uses: operands are uses
    for (Value use : op->getOperands()) {
      if (isVirtualRegType(use.getType())) {
        info.usePoints[use].push_back(idx);
      }
    }

    // Block arguments of while ops are defs at the loop op index
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (isVirtualRegType(arg.getType())) {
            if (!info.defPoints.contains(arg)) {
              info.defPoints[arg] = idx;
            }
          }
        }
      }
    }
  }

  // Pass 2: Build basic live ranges from def/use points
  for (const auto &[value, defPoint] : info.defPoints) {
    LiveRange range;
    range.reg = value;
    range.start = defPoint;

    // Find last use
    auto useIt = info.usePoints.find(value);
    if (useIt != info.usePoints.end() && !useIt->second.empty()) {
      range.end = *std::max_element(useIt->second.begin(), useIt->second.end());
    } else {
      // No uses: range is just the definition point
      range.end = defPoint;
    }

    // For loop op block args, extend live range to cover entire loop body.
    // Also extend to the first op after the LoopOp so that the block_arg's
    // register stays reserved through the loop exit transition. Loop results
    // are tied to block args (they share the same physical register), and
    // their def points are one position after the loop body's terminator.
    // Without this extension, the block_arg would be freed one point before
    // the loop result re-claims the register, causing a re-allocation that
    // leads to register pressure inflation from fragmentation.
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (parentOp && isa<LoopOp>(parentOp)) {
        Block *body = blockArg.getOwner();
        Operation *terminator = body->getTerminator();
        if (terminator) {
          auto termIt = opToIdx.find(terminator);
          if (termIt != opToIdx.end()) {
            range.end = std::max(range.end, termIt->second);
          }
        }

        // Extend to the next sibling op after the LoopOp. This bridges
        // the gap between the block_arg (which ends at the terminator)
        // and the loop_result (which starts at the next sibling). Since
        // they're tied, the physical register is shared and this doesn't
        // actually increase physical register usage. Without this, the
        // register is freed and immediately re-reserved, which causes
        // fragmentation and allocation failure.
        Operation *nextSibling = parentOp->getNextNode();
        if (nextSibling) {
          auto nextIt = opToIdx.find(nextSibling);
          if (nextIt != opToIdx.end()) {
            range.end = std::max(range.end, nextIt->second);
          }
        }
      }
    }

    // Get size and alignment from type
    Type ty = value.getType();
    range.size = getRegSize(ty);
    range.alignment = getRegAlignment(ty);
    if (auto regClass = getRegClass(ty)) {
      range.regClass = *regClass;
    }

    info.ranges[value] = range;
  }

  // Pass 2b: Extend live ranges for values used inside loop bodies.
  // Any value used inside a loop body is used on EVERY iteration, so its
  // live range must extend from its definition to the end of the loop body.
  // We cannot shorten this to just the last use point because the linear
  // scan allocator processes the loop body only once — if we freed the
  // register after the use point, a later op could take it, but the next
  // iteration would still read from the original register.
  for (const auto &[value, defPoint] : info.defPoints) {
    auto it = info.ranges.find(value);
    if (it == info.ranges.end())
      continue;

    // Check all use points for this value. If any use is inside a loop body,
    // extend the range to cover the entire loop body.
    auto useIt = info.usePoints.find(value);
    if (useIt == info.usePoints.end())
      continue;

    for (int64_t useIdx : useIt->second) {
      if (useIdx >= static_cast<int64_t>(ops.size()))
        continue;
      Operation *useOp = ops[useIdx];

      // Walk up parent chain to find enclosing loop ops
      Operation *parent = useOp->getParentOp();
      while (parent && !isa<ProgramOp>(parent)) {
        if (auto loopOp = dyn_cast<LoopOp>(parent)) {
          // Check if the value is defined inside the loop body
          // (at any nesting depth). Values defined inside are recomputed
          // each iteration and should keep their natural live ranges
          // within the iteration. Only values defined OUTSIDE need
          // extension across the loop.
          bool definedInside = false;
          if (auto defOp = value.getDefiningOp()) {
            // Check if defOp is anywhere inside the loop's region,
            // not just a direct child. This handles values defined
            // inside nested if/loop ops within the loop body.
            definedInside = loopOp->isProperAncestor(defOp);
          } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
            // BlockArguments don't have a defining op. Check if the
            // block argument's parent op is the loop or nested inside it.
            Operation *argParentOp = blockArg.getOwner()->getParentOp();
            definedInside = (argParentOp == loopOp.getOperation()) ||
                            loopOp->isProperAncestor(argParentOp);
          }

          if (!definedInside) {
            // Extend end to cover the entire loop body (value is
            // used every iteration, must survive until loop exits)
            Block &body = loopOp.getBodyBlock();
            Operation *terminator = body.getTerminator();
            if (terminator) {
              auto termIt = opToIdx.find(terminator);
              if (termIt != opToIdx.end()) {
                it->second.end = std::max(it->second.end, termIt->second);
              }
            }
            // Extend start back to the loop op
            auto loopIt = opToIdx.find(loopOp.getOperation());
            if (loopIt != opToIdx.end()) {
              it->second.start = std::min(it->second.start, loopIt->second);
            }
          }
        }
        parent = parent->getParentOp();
      }
    }
  }

  // Note: Pass 3 (CFG-based backward dataflow liveness extension) has been
  // removed. It was needed for the old label-based control flow path where
  // loop back-edges were represented as explicit branch instructions. With
  // region-based control flow (LoopOp/IfOp), Pass 2 and Pass 2b above
  // already handle all necessary live range extensions by directly inspecting
  // the region structure.

  // Pass 3a: Pack group pass -- treat pack inputs as sub-registers of the
  // pack result.
  //
  // waveasm.pack emits no assembly; it is a register allocation directive
  // declaring that N inputs form a contiguous register block. The pack result
  // already gets a correct contiguous allocation via allocRange, but pack
  // inputs would otherwise get independent allocations to arbitrary registers.
  //
  // Fix: extend the pack result's live range to cover the full lifetime of
  // all inputs (both defs and uses), then remove pack inputs from the
  // allocation worklists entirely. A post-pass in LinearScanPass assigns
  // input[i].physReg = result.physReg + i.
  llvm::DenseSet<Value> claimedPackInputs;
  program.walk([&](PackOp packOp) {
    Value packResult = packOp.getResult();
    auto resultIt = info.ranges.find(packResult);
    assert(resultIt != info.ranges.end() &&
           "pack result must have a live range");

    for (auto [idx, input] : llvm::enumerate(packOp.getElements())) {
      bool isExtract = input.getDefiningOp<ExtractOp>() != nullptr;
      bool isDuplicate = claimedPackInputs.contains(input);

      if (isDuplicate || isExtract) {
        // Insert a copy for:
        // - Duplicate inputs: same value claimed by another PackOp slot or
        //   repeated within this PackOp. A single SSA value can only occupy
        //   one physical register.
        // - ExtractOp inputs: ExtractOp is non-emitting and aliases
        //   source[offset]. The PackOp post-pass assigns input[i] =
        //   result_base + i, which may differ from the extract's physical
        //   register. Without a copy, the SRD word is silently wrong.
        OpBuilder builder(packOp);
        Value copy;
        if (isSGPRType(input.getType()))
          copy = S_MOV_B32::create(builder, packOp.getLoc(), input.getType(),
                                   input);
        else
          copy = V_MOV_B32::create(builder, packOp.getLoc(), input.getType(),
                                   input);
        packOp.setOperand(idx, copy);

        // If this is a first-seen extract, also remove its range from the
        // allocator worklist (the ExtractOp walk in LinearScanPass will
        // assign the correct physReg based on the source vector).
        auto inputIt = info.ranges.find(input);
        if (!isDuplicate && inputIt != info.ranges.end()) {
          info.ranges.erase(inputIt);
          claimedPackInputs.insert(input);
        }
      } else {
        auto inputIt = info.ranges.find(input);
        if (inputIt != info.ranges.end()) {
          // First PackOp to claim this input: extend the pack result's range
          // to cover the input's full lifetime, then remove from the
          // allocator worklist.
          resultIt->second.start =
              std::min(resultIt->second.start, inputIt->second.start);
          resultIt->second.end =
              std::max(resultIt->second.end, inputIt->second.end);
          info.ranges.erase(inputIt);
          claimedPackInputs.insert(input);
        }
      }
    }
  });

  // Pass 3a2: InsertOp pass -- treat insert result as an alias of the source
  // vector, but keep the inserted value in the allocator worklist.
  //
  // waveasm.insert declares that the result is the same contiguous register
  // block as the source with one element overwritten. The result aliases the
  // source (removed from worklist), but the inserted value must get its own
  // independent allocation. A post-pass in LinearScanPass emits a mov from
  // the inserted value's physreg into source_physreg + index.
  //
  // Why not alias the inserted value to source[index]? Because that would
  // clobber the source slot at the inserted value's def point, not at the
  // InsertOp's position. Any use of the source between the def and the
  // InsertOp would see the wrong value.
  //
  // Chained inserts (%b = insert into %a where %a = insert into %src) require
  // walking the chain to find the root source that still has a live range,
  // since intermediate results are already erased.
  program.walk([&](InsertOp insertOp) {
    Value insertResult = insertOp.getResult();

    // Walk the InsertOp chain to find the root source with a live range.
    Value source = insertOp.getVector();
    auto sourceIt = info.ranges.find(source);
    while (sourceIt == info.ranges.end()) {
      auto defOp = source.getDefiningOp<InsertOp>();
      if (!defOp) {
        // Chain root has no live range -- unexpected, but not fatal.
        // LinearScanPass will skip the insert (no physreg to alias).
        insertOp.emitWarning("insert chain root has no live range");
        return;
      }
      source = defOp.getVector();
      sourceIt = info.ranges.find(source);
    }

    // Extend root source range to cover the insert result's lifetime.
    auto resultIt = info.ranges.find(insertResult);
    if (resultIt != info.ranges.end()) {
      sourceIt->second.start =
          std::min(sourceIt->second.start, resultIt->second.start);
      sourceIt->second.end =
          std::max(sourceIt->second.end, resultIt->second.end);
      info.ranges.erase(resultIt);
    }

    // The inserted value stays in the worklist -- it gets its own allocation.
    // LinearScanPass emits a mov at the InsertOp's position.
  });

  // Pass 3b: Build tied equivalence classes for pressure de-duplication.
  //
  // LoopOp results, condition iter_args, and block args are all tied to the
  // same physical register. Instead of zeroing sizes (which violates
  // LiveRange invariants and risks allocator asserts), we group them into
  // equivalence classes. Each class has an envelope range (the union of all
  // member ranges) and the pressure sweep counts each class exactly once.
  //
  // All LiveRange::size values remain correct for the allocator. The
  // tiedClassId field on each range identifies its class membership.
  auto &tc = info.tiedClasses;

  program.walk([&](LoopOp loopOp) {
    Block &bodyBlock = loopOp.getBodyBlock();
    auto condOp = dyn_cast<ConditionOp>(bodyBlock.getTerminator());
    if (!condOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pass 3b: LoopOp body has non-ConditionOp terminator ("
                 << bodyBlock.getTerminator()->getName() << "), skipping "
                 << "tied-value class construction.\n");
      return;
    }

    for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
      BlockArgument blockArg = bodyBlock.getArgument(i);
      auto baIt = info.ranges.find(blockArg);
      if (baIt == info.ranges.end())
        continue;

      // Check if this block arg is already in a class (e.g., from an
      // MFMA tie on a nested loop). If so, extend that class.
      int64_t classId = -1;
      auto existingIt = tc.valueToClassId.find(blockArg);
      if (existingIt != tc.valueToClassId.end()) {
        classId = existingIt->second;
      }

      // Collect all values that share this physical register.
      llvm::SmallVector<Value> members;
      members.push_back(blockArg);

      // Check if init arg can be coalesced with its block arg. If the
      // init arg has uses after the loop, coalescing would let the loop
      // body modify the shared register and corrupt the post-loop value.
      bool initArgCanCoalesce = true;
      if (i < loopOp.getInitArgs().size()) {
        Value initArg = loopOp.getInitArgs()[i];
        auto initRangeIt = info.ranges.find(initArg);
        auto loopPosIt = opToIdx.find(loopOp.getOperation());
        if (initRangeIt != info.ranges.end() && loopPosIt != opToIdx.end() &&
            initRangeIt->second.end > loopPosIt->second) {
          initArgCanCoalesce = false;
          LLVM_DEBUG(llvm::dbgs()
                     << "  Skipping init arg coalescing for block arg " << i
                     << ": init arg has post-loop uses (range end "
                     << initRangeIt->second.end << " > loop position "
                     << loopPosIt->second << ")\n");
        }
      }

      // Init arg -> block arg (only if init arg has no post-loop uses).
      if (initArgCanCoalesce && i < loopOp.getInitArgs().size()) {
        Value initArg = loopOp.getInitArgs()[i];
        if (info.ranges.contains(initArg))
          members.push_back(initArg);
      }

      // Loop result -> block arg
      if (i < loopOp->getNumResults()) {
        Value loopResult = loopOp->getResult(i);
        if (info.ranges.contains(loopResult))
          members.push_back(loopResult);
      }

      // Condition iter_arg -> block arg.
      // Skip swap patterns and WAR hazards so the allocator keeps them
      // in separate registers (see hasWARHazard).
      if (i < condOp.getIterArgs().size()) {
        Value iterArg = condOp.getIterArgs()[i];
        bool skip = isSwapPatternIterArg(iterArg, bodyBlock, i) ||
                    hasWARHazard(iterArg, blockArg, info);
        if (!skip && info.ranges.contains(iterArg))
          members.push_back(iterArg);
      }

      if (members.size() <= 1)
        continue; // No ties to form a class.

      // Create or extend the class.
      if (classId < 0) {
        classId = static_cast<int64_t>(tc.classes.size());
        tc.classes.push_back({});
        tc.classes.back().id = classId;
        tc.classes.back().canonical = blockArg;
        tc.classes.back().size = baIt->second.size;
        tc.classes.back().alignment = baIt->second.alignment;
        tc.classes.back().regClass = baIt->second.regClass;
        tc.classes.back().envelopeStart = baIt->second.start;
        tc.classes.back().envelopeEnd = baIt->second.end;
      }

      TiedClass &cls = tc.classes[classId];

      // Add all members to the class and compute envelope.
      for (Value member : members) {
        if (tc.valueToClassId.contains(member))
          continue; // Already in this or another class.
        tc.valueToClassId[member] = classId;
        cls.members.push_back(member);

        auto rangeIt = info.ranges.find(member);
        if (rangeIt != info.ranges.end()) {
          cls.envelopeStart =
              std::min(cls.envelopeStart, rangeIt->second.start);
          cls.envelopeEnd = std::max(cls.envelopeEnd, rangeIt->second.end);
          // Tag the range with its class ID.
          rangeIt->second.tiedClassId = classId;
        }
      }

      // Build tiedPairs for the allocator:
      //   block_arg -> init_arg   (only if no post-loop uses)
      //   iter_arg  -> block_arg
      //   loop_result -> block_arg
      if (initArgCanCoalesce && i < loopOp.getInitArgs().size()) {
        Value initArg = loopOp.getInitArgs()[i];
        if (info.ranges.contains(initArg))
          tc.tiedPairs[blockArg] = initArg;
      }
      if (i < condOp.getIterArgs().size()) {
        Value iterArg = condOp.getIterArgs()[i];
        bool skip = isSwapPatternIterArg(iterArg, bodyBlock, i) ||
                    hasWARHazard(iterArg, blockArg, info);
        if (!skip && info.ranges.contains(iterArg) &&
            !tc.tiedPairs.contains(iterArg))
          tc.tiedPairs[iterArg] = blockArg;
      }
      if (i < loopOp->getNumResults()) {
        Value loopResult = loopOp->getResult(i);
        if (info.ranges.contains(loopResult))
          tc.tiedPairs[loopResult] = blockArg;
      }
    }
  });

  // Pass 4: Categorize ranges by register class and sort by start
  for (const auto &[value, range] : info.ranges) {
    if (range.regClass == RegClass::VGPR) {
      info.vregRanges.push_back(range);
    } else if (range.regClass == RegClass::SGPR) {
      info.sregRanges.push_back(range);
    } else if (range.regClass == RegClass::AGPR) {
      info.aregRanges.push_back(range);
    }
  }

  // Sort by (start, -size, -alignment, -end) for linear scan.
  // At the same start point, allocate larger and more constrained ranges
  // first to reduce fragmentation — a 16-wide aligned range has very few
  // valid slots, so giving it priority prevents smaller ranges from
  // blocking its only valid position. Final tie-break: longer ranges first,
  // since they are harder to fit around existing allocations.
  auto sortByStart = [](const LiveRange &a, const LiveRange &b) {
    if (a.start != b.start)
      return a.start < b.start;
    if (a.size != b.size)
      return a.size > b.size;
    if (a.alignment != b.alignment)
      return a.alignment > b.alignment;
    return a.end > b.end;
  };

  llvm::sort(info.vregRanges, sortByStart);
  llvm::sort(info.sregRanges, sortByStart);
  llvm::sort(info.aregRanges, sortByStart);

  // Pass 5: Compute pressure (class-aware: each tied class counted once)
  info.maxVRegPressure = computeMaxPressure(info.vregRanges, info.tiedClasses);
  info.maxSRegPressure = computeMaxPressure(info.sregRanges, info.tiedClasses);
  info.maxARegPressure = computeMaxPressure(info.aregRanges, info.tiedClasses);

  LLVM_DEBUG({
    dumpPeakPressureInfo(info, ops, RegClass::VGPR);
    dumpPeakPressureInfo(info, ops, RegClass::SGPR);
    dumpPeakPressureInfo(info, ops, RegClass::AGPR);
  });

  return info;
}

//===----------------------------------------------------------------------===//
// SSA Validation (Pure SSA - simplified)
//===----------------------------------------------------------------------===//

LogicalResult validateSSA(ProgramOp program,
                          llvm::DenseSet<int64_t> /*loopControlSRegs*/,
                          llvm::DenseSet<int64_t> /*accumulatorVRegs*/) {
  // In pure SSA, MLIR already enforces single definition for each Value
  // This function can validate additional constraints if needed
  llvm::DenseSet<Value> definitions;

  for (Operation &op : program.getBodyBlock()) {
    for (Value def : op.getResults()) {
      if (!isVirtualRegType(def.getType()))
        continue;

      auto [it, inserted] = definitions.insert(def);
      if (!inserted) {
        // This shouldn't happen in valid MLIR SSA
        return op.emitOpError()
               << "SSA violation: value defined multiple times.";
      }
    }
  }

  return success();
}

} // namespace waveasm
