// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Ticketing Pass - Insert s_waitcnt instructions for memory synchronization
//
// This pass implements a ticket-based tracking system for VMEM and LGKM
// memory operations. Each memory operation is assigned a monotonically
// increasing ticket. When a value is used, the pass computes the minimum
// wait threshold and inserts s_waitcnt if required.
//
// Key optimizations over a naive approach:
// 1. Incremental wait tracking: when new loads are issued after a wait,
//    we increment the last-wait threshold instead of resetting it. This
//    correctly recognizes that values loaded BEFORE the last wait are still
//    ready, avoiding unnecessary waitcnt insertions.
// 2. Latency-covering for LGKM: for ds_read (LDS) loads, if enough
//    instructions separate the load from its use, we skip fine-grained
//    lgkmcnt(N>0) waits since the data has certainly arrived. This matches
//    how hand-tuned kernels (e.g., AITER) rely on latency hiding.
// 3. Barrier handling uses "full drain" checks (hasOutstandingVmem/Lgkm)
//    to correctly insert vmcnt(0)/lgkmcnt(0) before barriers.
// 4. Loop back-edge (ConditionOp) does NOT reset wait tracking, allowing
//    waits from the end of one iteration to carry to the next, eliminating
//    redundant waitcnts at loop tops.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMINSERTWAITCNT
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

//===----------------------------------------------------------------------===//
// Ticketing System
//===----------------------------------------------------------------------===//

/// Tracks memory operation tickets for waitcnt insertion.
///
/// The ticket system maps each load operation to a monotonically increasing
/// ticket number. The hardware vmcnt/lgkmcnt counters track outstanding
/// loads in FIFO order, so vmcnt(N) means "wait until at most N loads are
/// outstanding" (the N most recent). To wait for ticket T, we need
/// vmcnt(lastTicket - T), which allows (lastTicket - T) newer loads to
/// remain outstanding while guaranteeing T (and older) have completed.
class Ticketing {
public:
  Ticketing() = default;

  /// Issue a new VMEM ticket, returns the ticket number.
  /// A new load shifts the meaning of previous waits: if we previously
  /// waited for vmcnt(N), that now effectively means vmcnt(N+1) since
  /// one more operation is outstanding. We increment (not reset) the
  /// last wait value to reflect this.
  int64_t nextVmemTicket() {
    ++lastVmemTicket;
    if (lastVmemWait.has_value())
      ++(*lastVmemWait);
    return lastVmemTicket;
  }

  /// Issue a new LGKM ticket, returns the ticket number.
  int64_t nextLgkmTicket() {
    ++lastLgkmTicket;
    if (lastLgkmWait.has_value())
      ++(*lastLgkmWait);
    return lastLgkmTicket;
  }

  /// Get current VMEM ticket (last issued)
  int64_t getLastVmemTicket() const { return lastVmemTicket; }

  /// Get current LGKM ticket (last issued)
  int64_t getLastLgkmTicket() const { return lastLgkmTicket; }

  /// Compute the vmcnt threshold needed to wait for a given ticket.
  /// Returns std::nullopt if no wait is needed (already satisfied).
  std::optional<int64_t> computeVmemWait(int64_t requiredTicket) {
    if (requiredTicket < 0 || lastVmemTicket < 0)
      return std::nullopt;

    int64_t threshold = std::max(int64_t(0), lastVmemTicket - requiredTicket);

    if (lastVmemWait.has_value() && *lastVmemWait <= threshold)
      return std::nullopt;

    return threshold;
  }

  /// Compute the lgkmcnt threshold needed to wait for a given ticket.
  std::optional<int64_t> computeLgkmWait(int64_t requiredTicket) {
    if (requiredTicket < 0 || lastLgkmTicket < 0)
      return std::nullopt;

    int64_t threshold = std::max(int64_t(0), lastLgkmTicket - requiredTicket);

    if (lastLgkmWait.has_value() && *lastLgkmWait <= threshold)
      return std::nullopt;

    return threshold;
  }

  /// Check if any VMEM operations are outstanding since the last full drain.
  /// Used for barrier pre-waits where we need vmcnt(0).
  bool hasOutstandingVmem() const {
    if (lastVmemTicket < 0)
      return false;
    return !lastVmemWait.has_value() || *lastVmemWait > 0;
  }

  /// Check if any LGKM operations are outstanding since the last full drain.
  bool hasOutstandingLgkm() const {
    if (lastLgkmTicket < 0)
      return false;
    return !lastLgkmWait.has_value() || *lastLgkmWait > 0;
  }

  /// Record that a vmem wait was emitted with the given threshold
  void observeVmemWait(int64_t threshold) {
    if (!lastVmemWait.has_value() || threshold < *lastVmemWait)
      lastVmemWait = threshold;
  }

  /// Record that a lgkm wait was emitted with the given threshold
  void observeLgkmWait(int64_t threshold) {
    if (!lastLgkmWait.has_value() || threshold < *lastLgkmWait)
      lastLgkmWait = threshold;
  }

  /// Reset wait tracking (e.g., at loop entry from outside)
  void resetWaits() {
    lastVmemWait = std::nullopt;
    lastLgkmWait = std::nullopt;
  }

private:
  int64_t lastVmemTicket = -1;
  int64_t lastLgkmTicket = -1;
  std::optional<int64_t> lastVmemWait;
  std::optional<int64_t> lastLgkmWait;
};

//===----------------------------------------------------------------------===//
// Memory Operation Classification
//===----------------------------------------------------------------------===//

enum class MemOpKind {
  None,
  VmemLoad,  // buffer_load, global_load, flat_load
  VmemStore, // buffer_store, global_store, flat_store
  LgkmLoad,  // ds_read, s_load
  LgkmStore  // ds_write
};

/// Classify an operation by its memory type using MLIR op types.
MemOpKind classifyMemOp(const Operation *op) {
  // VMEM loads (buffer, global, flat)
  if (isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
          BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE,
          BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT, GLOBAL_LOAD_DWORD,
          GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4,
          GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_USHORT,
          GLOBAL_LOAD_SSHORT, FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2,
          FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4,
          // Gather-to-LDS: VMEM load directly to LDS (still uses VMEM path)
          BUFFER_LOAD_DWORD_LDS, BUFFER_LOAD_DWORDX4_LDS>(op))
    return MemOpKind::VmemLoad;

  // VMEM stores (buffer, global, flat)
  if (isa<BUFFER_STORE_DWORD, BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX3,
          BUFFER_STORE_DWORDX4, BUFFER_STORE_BYTE, BUFFER_STORE_SHORT,
          GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORDX2, GLOBAL_STORE_DWORDX3,
          GLOBAL_STORE_DWORDX4, GLOBAL_STORE_BYTE, GLOBAL_STORE_SHORT,
          FLAT_STORE_DWORD, FLAT_STORE_DWORDX2, FLAT_STORE_DWORDX3,
          FLAT_STORE_DWORDX4>(op))
    return MemOpKind::VmemStore;

  // LDS loads (ds_read)
  if (isa<DS_READ_B32, DS_READ_B64, DS_READ_B128, DS_READ2_B32, DS_READ2_B64,
          DS_READ_U8, DS_READ_I8, DS_READ_U16, DS_READ_I16>(op))
    return MemOpKind::LgkmLoad;

  // LDS stores (ds_write)
  if (isa<DS_WRITE_B32, DS_WRITE_B64, DS_WRITE_B128, DS_WRITE2_B32,
          DS_WRITE2_B64, DS_WRITE_B8, DS_WRITE_B16>(op))
    return MemOpKind::LgkmStore;

  // Scalar loads (s_load) - also use LGKM counter
  if (isa<S_LOAD_DWORD, S_LOAD_DWORDX2, S_LOAD_DWORDX4, S_LOAD_DWORDX8,
          S_LOAD_DWORDX16>(op))
    return MemOpKind::LgkmLoad;

  return MemOpKind::None;
}

//===----------------------------------------------------------------------===//
// InsertWaitcnt Pass
//===----------------------------------------------------------------------===//

/// Per-program mutable state shared across all handler methods.
struct WaitcntState {
  Ticketing ticketing;
  llvm::DenseMap<Value, std::pair<MemOpKind, int64_t>> valueTickets;
  llvm::DenseMap<Value, int64_t> loadDefIndex;
  int64_t opIndex = 0;
  int64_t maxLgkmcnt = 15;
  int64_t maxVmcnt = 63;

  int64_t capLgkmcnt(int64_t v) const {
    return v > maxLgkmcnt ? int64_t(0) : v;
  }
  int64_t capVmcnt(int64_t v) const { return v > maxVmcnt ? int64_t(0) : v; }
};

struct InsertWaitcntPass
    : public waveasm::impl::WAVEASMInsertWaitcntBase<InsertWaitcntPass> {
  using WAVEASMInsertWaitcntBase::WAVEASMInsertWaitcntBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    module->walk([&](ProgramOp program) { processProgram(program); });
  }

private:
  /// Minimum number of IR operations between an LGKM load and its use for
  /// latency covering. ds_read (LDS) latency is ~20-40 cycles on CDNA.
  ///
  /// HEURISTIC ASSUMPTION: This counts IR operations, not hardware cycles.
  /// A single MFMA is 32+ cycles but counts as 1 op. This is conservative
  /// for MFMA-heavy kernels (where 40 ops >> 40 cycles) but borderline for
  /// kernels with only short VALU ops. For correctness, the threshold is
  /// intentionally set at the high end of LDS latency; if this proves too
  /// aggressive for VALU-only sequences, reduce to ~20.
  static constexpr int64_t kMinOpsForLgkmLatency = 40;

  unsigned numVmemOps = 0;
  unsigned numLgkmOps = 0;
  unsigned numWaitcntInserted = 0;

  //===--------------------------------------------------------------------===//
  // Handler: observe pre-existing waitcnt instructions
  //===--------------------------------------------------------------------===//

  /// Returns true if the op was a waitcnt and was consumed.
  bool observeExistingWaitcnt(Operation *op, WaitcntState &st) {
    if (auto waitcnt = dyn_cast<S_WAITCNT>(op)) {
      if (auto a = waitcnt.getVmcntAttr())
        st.ticketing.observeVmemWait(a.getValue().getSExtValue());
      if (auto a = waitcnt.getLgkmcntAttr())
        st.ticketing.observeLgkmWait(a.getValue().getSExtValue());
      return true;
    }
    if (auto waitcnt = dyn_cast<S_WAITCNT_VMCNT>(op)) {
      if (auto a = waitcnt.getCountAttr())
        st.ticketing.observeVmemWait(a.getValue().getSExtValue());
      return true;
    }
    if (auto waitcnt = dyn_cast<S_WAITCNT_LGKMCNT>(op)) {
      if (auto a = waitcnt.getCountAttr())
        st.ticketing.observeLgkmWait(a.getValue().getSExtValue());
      return true;
    }
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Handler: barrier synchronization
  //===--------------------------------------------------------------------===//

  /// Returns true if the op was a barrier and was consumed.
  bool handleBarrier(Operation *op, WaitcntState &st) {
    if (!isa<S_BARRIER>(op))
      return false;

    bool needVmem = st.ticketing.hasOutstandingVmem();
    bool needLgkm = st.ticketing.hasOutstandingLgkm();

    if (needVmem || needLgkm) {
      OpBuilder builder(op->getContext());
      builder.setInsertionPoint(op);
      if (needVmem && needLgkm) {
        S_WAITCNT::create(builder, op->getLoc(), builder.getI32IntegerAttr(0),
                          builder.getI32IntegerAttr(0), IntegerAttr());
      } else if (needVmem) {
        S_WAITCNT_VMCNT::create(builder, op->getLoc(), 0);
      } else {
        S_WAITCNT_LGKMCNT::create(builder, op->getLoc(), 0);
      }
      numWaitcntInserted++;
    }

    st.ticketing.observeVmemWait(0);
    st.ticketing.observeLgkmWait(0);
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Handler: memory operations (loads and stores)
  //===--------------------------------------------------------------------===//

  void handleMemoryOp(Operation *op, MemOpKind kind, WaitcntState &st) {
    // For stores, check operands BEFORE incrementing ticket counter
    if (kind == MemOpKind::VmemStore || kind == MemOpKind::LgkmStore)
      emitWaitsForStoreOperands(op, st);

    // Only LOADS increment the counter (vmcnt/lgkmcnt track loads, not stores)
    int64_t ticket = -1;
    if (kind == MemOpKind::VmemLoad) {
      ticket = st.ticketing.nextVmemTicket();
      numVmemOps++;
    } else if (kind == MemOpKind::LgkmLoad) {
      ticket = st.ticketing.nextLgkmTicket();
      numLgkmOps++;
    } else {
      numVmemOps++;
    }

    for (Value result : op->getResults()) {
      st.valueTickets[result] = {kind, ticket};
      if (kind == MemOpKind::VmemLoad || kind == MemOpKind::LgkmLoad)
        st.loadDefIndex[result] = st.opIndex;
    }

    if (insertAfterLoads &&
        (kind == MemOpKind::VmemLoad || kind == MemOpKind::LgkmLoad)) {
      OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(op);
      if (kind == MemOpKind::VmemLoad) {
        S_WAITCNT_VMCNT::create(builder, op->getLoc(),
                                builder.getI32IntegerAttr(0));
        st.ticketing.observeVmemWait(0);
      } else {
        S_WAITCNT_LGKMCNT::create(builder, op->getLoc(),
                                  builder.getI32IntegerAttr(0));
        st.ticketing.observeLgkmWait(0);
      }
      numWaitcntInserted++;
    }
  }

  void emitWaitsForStoreOperands(Operation *op, WaitcntState &st) {
    std::optional<int64_t> neededVmcnt;
    std::optional<int64_t> neededLgkmcnt;

    for (Value operand : op->getOperands()) {
      auto it = st.valueTickets.find(operand);
      if (it == st.valueTickets.end())
        continue;
      auto [opKind, opTicket] = it->second;
      if (opKind == MemOpKind::VmemLoad) {
        auto wait = st.ticketing.computeVmemWait(opTicket);
        if (wait.has_value() && (!neededVmcnt || *wait < *neededVmcnt))
          neededVmcnt = wait;
      } else if (opKind == MemOpKind::LgkmLoad) {
        auto wait = st.ticketing.computeLgkmWait(opTicket);
        if (wait.has_value() && (!neededLgkmcnt || *wait < *neededLgkmcnt))
          neededLgkmcnt = wait;
      }
    }

    if (!neededVmcnt && !neededLgkmcnt)
      return;

    emitWaitcnt(op, neededVmcnt, neededLgkmcnt, st);
  }

  //===--------------------------------------------------------------------===//
  // Handler: non-memory operations (VALU, SALU, etc.)
  //===--------------------------------------------------------------------===//

  void handleNonMemoryOp(Operation *op, WaitcntState &st) {
    std::optional<int64_t> neededVmcnt;
    std::optional<int64_t> neededLgkmcnt;

    for (Value operand : op->getOperands()) {
      auto it = st.valueTickets.find(operand);
      if (it == st.valueTickets.end())
        continue;
      auto [opKind, ticket] = it->second;

      if (opKind == MemOpKind::VmemLoad) {
        auto wait = st.ticketing.computeVmemWait(ticket);
        if (wait.has_value() && (!neededVmcnt || *wait < *neededVmcnt))
          neededVmcnt = wait;
      } else if (opKind == MemOpKind::LgkmLoad) {
        auto wait = st.ticketing.computeLgkmWait(ticket);
        if (wait.has_value()) {
          if (*wait > 0) {
            auto defIt = st.loadDefIndex.find(operand);
            if (defIt != st.loadDefIndex.end()) {
              int64_t gap = st.opIndex - defIt->second;
              if (gap >= kMinOpsForLgkmLatency)
                continue;
            }
          }
          if (!neededLgkmcnt || *wait < *neededLgkmcnt)
            neededLgkmcnt = wait;
        }
      }
    }

    if (neededVmcnt || neededLgkmcnt)
      emitWaitcnt(op, neededVmcnt, neededLgkmcnt, st);
  }

  //===--------------------------------------------------------------------===//
  // Handler: loop boundary reset
  //===--------------------------------------------------------------------===//

  void handleLoopBoundary(Operation *op, WaitcntState &st) {
    // Reset at loop ENTRY (LoopOp) since pending waits from before the loop
    // may not be valid inside. Do NOT reset at ConditionOp (loop back-edge)
    // because waits from the end of one iteration ARE valid at the start
    // of the next -- the hardware state is continuous.
    if (isa<LoopOp>(op) || op->getName().getStringRef().contains("branch")) {
      st.ticketing.resetWaits();
    }
  }

  //===--------------------------------------------------------------------===//
  // Shared helper: emit s_waitcnt with cap and tracking
  //===--------------------------------------------------------------------===//

  void emitWaitcnt(Operation *op, std::optional<int64_t> neededVmcnt,
                   std::optional<int64_t> neededLgkmcnt, WaitcntState &st) {
    OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);

    int64_t vmcntVal = neededVmcnt ? st.capVmcnt(*neededVmcnt) : 0;
    int64_t lgkmcntVal = neededLgkmcnt ? st.capLgkmcnt(*neededLgkmcnt) : 0;

    if (neededVmcnt && neededLgkmcnt) {
      S_WAITCNT::create(builder, op->getLoc(),
                        builder.getI32IntegerAttr(vmcntVal),
                        builder.getI32IntegerAttr(lgkmcntVal), IntegerAttr());
      st.ticketing.observeVmemWait(vmcntVal);
      st.ticketing.observeLgkmWait(lgkmcntVal);
    } else if (neededVmcnt) {
      S_WAITCNT_VMCNT::create(builder, op->getLoc(),
                              builder.getI32IntegerAttr(vmcntVal));
      st.ticketing.observeVmemWait(vmcntVal);
    } else {
      S_WAITCNT_LGKMCNT::create(builder, op->getLoc(),
                                builder.getI32IntegerAttr(lgkmcntVal));
      st.ticketing.observeLgkmWait(lgkmcntVal);
    }
    numWaitcntInserted++;
  }

  //===--------------------------------------------------------------------===//
  // Top-level per-program dispatcher
  //===--------------------------------------------------------------------===//

  void processProgram(ProgramOp program) {
    WaitcntState st;

    if (auto targetAttr = program.getTarget()) {
      TargetAttrInterface targetKind = targetAttr.getTargetKind();
      st.maxLgkmcnt = targetKind.getMaxLgkmcnt();
      st.maxVmcnt = targetKind.getMaxVmcnt();
    }

    llvm::SmallVector<Operation *> allOps;
    collectOpsRecursive(program.getBodyBlock(), allOps);

    if (ticketedWaitcnt) {
      for (Operation *op : allOps) {
        ++st.opIndex;

        if (observeExistingWaitcnt(op, st))
          continue;
        if (handleBarrier(op, st))
          continue;

        MemOpKind kind = classifyMemOp(op);
        if (kind != MemOpKind::None) {
          handleMemoryOp(op, kind, st);
          continue;
        }

        handleNonMemoryOp(op, st);
        handleLoopBoundary(op, st);
      }
    }

    combineAdjacentWaitcnts(program);
  }

  /// Merge any adjacent waitcnt-like operations into a single S_WAITCNT.
  ///
  /// For each field (vmcnt/lgkmcnt/expcnt), the merged op keeps the minimum
  /// threshold contributed by the pair, which is equivalent to executing both
  /// waits back-to-back without intervening instructions.
  void combineAdjacentWaitcnts(ProgramOp program) {
    // Helpers to extract individual counter values from waitcnt ops.
    auto getVmcntFromOp = [](Operation *op) -> int64_t {
      if (auto w = dyn_cast<S_WAITCNT_VMCNT>(op))
        return w.getCountAttr().getValue().getSExtValue();
      if (auto w = dyn_cast<S_WAITCNT>(op)) {
        if (auto attr = w.getVmcntAttr())
          return attr.getValue().getSExtValue();
      }
      return -1;
    };
    auto getLgkmcntFromOp = [](Operation *op) -> int64_t {
      if (auto w = dyn_cast<S_WAITCNT_LGKMCNT>(op))
        return w.getCountAttr().getValue().getSExtValue();
      if (auto w = dyn_cast<S_WAITCNT>(op)) {
        if (auto attr = w.getLgkmcntAttr())
          return attr.getValue().getSExtValue();
      }
      return -1;
    };
    auto getExpcntFromOp = [](Operation *op) -> int64_t {
      if (auto w = dyn_cast<S_WAITCNT>(op)) {
        if (auto attr = w.getExpcntAttr())
          return attr.getValue().getSExtValue();
      }
      return -1;
    };

    auto isWaitcntLike = [](Operation *op) {
      return isa<S_WAITCNT, S_WAITCNT_VMCNT, S_WAITCNT_LGKMCNT>(op);
    };

    llvm::SmallVector<std::pair<Operation *, Operation *>> toCombine;

    llvm::SmallVector<Operation *> combineOps;
    collectOpsRecursive(program.getBodyBlock(), combineOps);

    for (Operation *op : combineOps) {
      Operation *next = op->getNextNode();
      if (!next || !isWaitcntLike(op) || !isWaitcntLike(next))
        continue;
      toCombine.push_back({op, next});
    }

    llvm::DenseSet<Operation *> erased;
    for (auto [first, second] : toCombine) {
      if (erased.contains(first) || erased.contains(second))
        continue;

      // Collect fields from both ops, taking min where both contribute.
      int64_t vm1 = getVmcntFromOp(first), vm2 = getVmcntFromOp(second);
      int64_t lk1 = getLgkmcntFromOp(first), lk2 = getLgkmcntFromOp(second);
      int64_t ex1 = getExpcntFromOp(first), ex2 = getExpcntFromOp(second);

      auto merge = [](int64_t a, int64_t b) -> int64_t {
        if (a < 0)
          return b;
        if (b < 0)
          return a;
        return std::min(a, b);
      };
      int64_t vmcntVal = merge(vm1, vm2);
      int64_t lgkmcntVal = merge(lk1, lk2);
      int64_t expcntVal = merge(ex1, ex2);

      OpBuilder builder(first->getContext());
      builder.setInsertionPoint(first);

      auto optAttr = [&](int64_t v) -> IntegerAttr {
        return v >= 0 ? builder.getI32IntegerAttr(v) : IntegerAttr();
      };
      S_WAITCNT::create(builder, first->getLoc(), optAttr(vmcntVal),
                        optAttr(lgkmcntVal), optAttr(expcntVal));

      erased.insert(first);
      erased.insert(second);
      second->erase();
      first->erase();
    }
  }
};

} // namespace
