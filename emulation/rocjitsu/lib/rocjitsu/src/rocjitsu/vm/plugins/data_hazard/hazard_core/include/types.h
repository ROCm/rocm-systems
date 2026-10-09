// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Simulator-agnostic type definitions for hazard detection.
//
// This header defines the core types shared by simulator frontend adapters. It
// deliberately excludes frontend-specific callback, warning, and reporting types
// to maintain simulator independence.
//
// Defines:
//   - WaitCntType      Enum of GPU wait-counter categories (VMEM, SMEM,
//                      LDS, STORE, etc.) used to classify pending ops.
//   - PendingAsyncOp   Lightweight record of an in-flight async operation
//                      (instruction id, PC, wait type, access size).
//   - WaveState        Per-wave tracking: pending-op maps, FIFO queues for
//                      each counter type, and dedup sets for reported hazards.
//   - LdsHazardKey     Dedup key for intra-wave LDS RAW hazards.
//   - RegisterHazardKey Dedup key for register RAW/WAR/WAW hazards.

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace hazard_core {

// =============================================================================
// Wait counter types (mapped from opcode detection)
// =============================================================================

enum class WaitCntType {
  NONE,
  VMEM,  // Vector memory loads (s_wait_loadcnt)
  SMEM,  // Scalar memory loads (s_wait_kmcnt)
  LDS,   // Local Data Share operations (s_wait_dscnt)
  STORE, // Vector memory stores (s_wait_storecnt)
  // The one counter gfx9-style targets keep scalar memory and LDS on
  // (s_waitcnt lgkmcnt). Only a drain names it: an operation is outstanding on
  // SMEM or LDS, and this counter retires the two as the single sequence the
  // hardware counts them as.
  LGKM,
  XCNT, // Address translation counter (s_wait_xcnt) — tracks XACK/XNACK, not data completion
  // TODO: Implement tracking for these wait counter types when needed
  ASYNC, // Async copy operations (s_wait_asynccnt) - not yet tracked
  TENSOR // Tensor DMA operations (s_wait_tensorcnt) — tensor_load_to_lds, tensor_store_from_lds
};

// =============================================================================
// Constants
// =============================================================================

constexpr uint32_t MAX_VGPR_INDEX = 255;
constexpr uint32_t MAX_ACC_VGPR_INDEX = 255;
constexpr uint32_t MAX_SGPR_INDEX = 105;
constexpr uint32_t BYTES_PER_DWORD = 4;
constexpr uint32_t MAX_ACCESS_SIZE_BYTES = 1024 * 1024; // 1MB sanity limit

constexpr uint32_t bytes_to_dwords(uint32_t bytes) {
  return (bytes + BYTES_PER_DWORD - 1) / BYTES_PER_DWORD;
}

// =============================================================================
// Per-wave state for tracking pending async operations
// =============================================================================

// Simulator-agnostic identifier for instructions, waves, workgroups, and
// dispatches. Frontend adapters convert native identifiers to this type at the
// API boundary.
using EntityId = uint64_t;

/// Every lane of a wave. A VGPR holds one value per lane, and an access with no
/// lane mask is taken to touch all of them.
constexpr uint64_t ALL_LANES = ~uint64_t{0};

/// The lanes an event's exec_mask names; 0 means the frontend gave none.
constexpr uint64_t lanes_or_all(uint64_t exec_mask) {
  return exec_mask != 0 ? exec_mask : ALL_LANES;
}

struct PendingAsyncOp {
  EntityId instruction_id = 0;
  uint64_t pc = 0;
  WaitCntType wait_type = WaitCntType::NONE;
  uint32_t size = 0;
  // True when this is the DScnt-side entry of a flat load (which also has
  // a LOADcnt-side entry in pending_vgpr_writes). When the LOADcnt side is
  // cleared, this DScnt-only pending is safe and should not fire a RAW.
  bool is_flat_ds = false;
  // Lanes of a vector register the operation writes or reads. Another access
  // conflicts with it only on a lane both have active. ALL_LANES for scalar
  // registers, LDS and anything a frontend reported without lanes.
  uint64_t lane_mask = ALL_LANES;
  // NOTE: raw_isa is intentionally omitted here. Raw instruction bytes are
  // frontend/reporting metadata, not hazard detection state.
};

// LDS hazard deduplication key: (producer_instruction_id, consumer_instruction_id).
// One report per (producer, consumer) instruction pair regardless of how many
// addresses they conflict on.
struct LdsHazardKey {
  EntityId producer_instruction_id;
  EntityId consumer_instruction_id;

  bool operator<(const LdsHazardKey &other) const {
    if (producer_instruction_id != other.producer_instruction_id)
      return producer_instruction_id < other.producer_instruction_id;
    return consumer_instruction_id < other.consumer_instruction_id;
  }
};

enum class HazardRegisterKind {
  Vector,
  AccumVector,
  Scalar,
};

struct WawHazardKey {
  EntityId instruction_id;
  HazardRegisterKind register_kind;
  uint32_t register_index;

  bool operator<(const WawHazardKey &other) const {
    if (instruction_id != other.instruction_id)
      return instruction_id < other.instruction_id;
    if (register_kind != other.register_kind)
      return register_kind < other.register_kind;
    return register_index < other.register_index;
  }
};

// Secondary dedup key applied on top of the per-register sets below, following
// the same policy as LdsHazardKey: one report per (producer, consumer) pair
// however many registers of a wide operand conflict, because a single wait
// resolves all of them. A consumer conflicting with two different producers
// still reports once per producer, since those are distinct defects.
struct RegisterHazardKey {
  EntityId producer_instruction_id;
  EntityId consumer_instruction_id;
  HazardRegisterKind register_kind;

  bool operator<(const RegisterHazardKey &other) const {
    if (producer_instruction_id != other.producer_instruction_id)
      return producer_instruction_id < other.producer_instruction_id;
    if (consumer_instruction_id != other.consumer_instruction_id)
      return consumer_instruction_id < other.consumer_instruction_id;
    return register_kind < other.register_kind;
  }
};

/// A FIFO of pending LDS ranges, (start address, op), oldest first.
///
/// The frontend reports LDS accesses one lane at a time. push_range folds a
/// lane into an entry of the same instruction whose range it overlaps or
/// abuts, so a contiguous wave access holds one entry rather than one per
/// lane, and a later access scans one entry rather than one per lane. Entries
/// of one instruction share everything but their range and retire together,
/// so a merged entry covers exactly the bytes its lanes did.
class LdsRangeQueue {
public:
  using Entry = std::pair<uint32_t, PendingAsyncOp>;
  /// How many of an instruction's most recent entries a new lane range is
  /// tried against before it gets an entry of its own. Lanes usually arrive in
  /// address order, so the newest entry almost always absorbs the next lane; a
  /// short window also catches lanes swizzled within a row.
  static constexpr int COALESCE_WINDOW = 8;

  auto begin() const { return entries_.begin(); }
  auto end() const { return entries_.end(); }
  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }
  const Entry &front() const { return entries_.front(); }
  const Entry &operator[](size_t i) const { return entries_[i]; }

  void clear() { entries_.clear(); }

  /// Append an entry as is, with no folding.
  void push_back(const Entry &entry) { entries_.push_back(entry); }

  /// Append one lane's range, folding it into the instruction's recent entries.
  void push_range(uint32_t address, const PendingAsyncOp &op) {
    uint64_t lo = address;
    uint64_t hi = lo + op.size;
    // Absorb every recent entry of this instruction the range touches. Taking
    // one can bridge the gap to another -- a swizzled lane landing between two
    // runs -- so the window is searched again until nothing more joins.
    for (bool merged = true; merged;) {
      merged = false;
      int tried = 0;
      for (auto it = entries_.end(); it != entries_.begin() && tried < COALESCE_WINDOW; ++tried) {
        --it;
        const auto &[start, pending] = *it;
        if (pending.instruction_id != op.instruction_id)
          break;
        if (pending.wait_type != op.wait_type)
          continue;
        const uint64_t end = uint64_t{start} + pending.size;
        if (lo > end || start > hi)
          continue;
        const uint64_t new_lo = std::min<uint64_t>(lo, start);
        const uint64_t new_hi = std::max(hi, end);
        if (new_hi - new_lo > std::numeric_limits<uint32_t>::max())
          continue;
        lo = new_lo;
        hi = new_hi;
        entries_.erase(it);
        merged = true;
        break;
      }
    }

    PendingAsyncOp entry = op;
    entry.size = static_cast<uint32_t>(hi - lo);
    entries_.push_back({static_cast<uint32_t>(lo), entry});
  }

  /// Drop every entry @p pred selects.
  template <typename Pred> void remove_if(Pred pred) {
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), pred), entries_.end());
  }

private:
  std::deque<Entry> entries_;
};

struct WaveState {
  // === RAW hazard tracking (reads before async writes complete) ===
  std::unordered_map<uint32_t, PendingAsyncOp> pending_vgpr_writes;
  std::unordered_map<uint32_t, PendingAsyncOp> pending_sgpr_writes;

  // FIFO queues for proper waitcnt N handling (RAW).
  //
  // An instruction leaves one entry per register or address it touches, all
  // carrying its instruction id. The counter these drain against counts issued
  // instructions, so a wait of N keeps the N newest *instructions*, and every
  // entry of an instruction retires with it: global_load_dwordx4 v[0:3] leaves
  // four entries in one LOADcnt slot, and s_wait_loadcnt 1 leaves all four
  // pending. See detail/fifo_ops.h for the drain itself.
  //
  // Design trade-off for LDS: We use address ranges rather than a per-byte bitmap.
  // Ranges are memory-efficient for typical contiguous accesses, while a
  // bitmap would require 8KB per wave (LDS is 64KB). Ranges may have rare
  // false positives on adjacent non-overlapping accesses.
  std::deque<std::pair<uint32_t, PendingAsyncOp>> vmem_load_fifo;
  std::deque<std::pair<uint32_t, PendingAsyncOp>> smem_load_fifo;
  LdsRangeQueue lds_fifo;

  // DScnt-side tracking for flat load VGPR writes.
  // Flat instructions use both LOADcnt and DScnt; the VGPR destination is
  // not safe until both counters are satisfied. These structures mirror
  // vmem_load_fifo / pending_vgpr_writes but are drained by s_wait_dscnt.
  std::unordered_map<uint32_t, PendingAsyncOp> pending_vgpr_writes_ds;
  std::deque<std::pair<uint32_t, PendingAsyncOp>> flat_vgpr_ds_fifo;

  // === Accumulator VGPR tracking (separate architectural namespace from VGPR) ===
  std::unordered_map<uint32_t, PendingAsyncOp> pending_acc_vgpr_writes;
  std::deque<std::pair<uint32_t, PendingAsyncOp>> acc_vmem_load_fifo;
  std::unordered_map<uint32_t, PendingAsyncOp> pending_acc_vgpr_writes_ds;
  std::deque<std::pair<uint32_t, PendingAsyncOp>> flat_acc_vgpr_ds_fifo;
  std::unordered_map<uint32_t, PendingAsyncOp> pending_acc_vgpr_reads;
  std::deque<std::pair<uint32_t, PendingAsyncOp>> acc_vmem_store_fifo;

  // === WAR hazard tracking (writes before async reads complete) ===
  std::unordered_map<uint32_t, PendingAsyncOp> pending_vgpr_reads;

  // FIFO for store operations (WAR)
  std::deque<std::pair<uint32_t, PendingAsyncOp>> vmem_store_fifo;

  // Vector memory stores in flight, held for wait-counter occupancy alone. A
  // store leaves no pending register state behind because its data sources are
  // captured at issue, but it does hold a memory counter slot, and where one
  // counter tracks loads and stores together that slot delays the retirement of
  // older loads. One entry per issued store instruction, oldest first.
  std::deque<PendingAsyncOp> vmem_store_ops;

  // Pending async LDS reads (DS loads in flight): address -> op.
  // WAR hazard: a DS write to an overlapping address before s_wait_dscnt
  // could corrupt LDS before the in-flight read captures it.
  LdsRangeQueue lds_read_fifo;

  // TENSORcnt-tracked LDS writes (tensor_load_to_lds, tensor_store_from_lds)
  LdsRangeQueue tensor_lds_fifo;

  // === Deduplication ===
  std::set<WawHazardKey> reported_raw_hazards;
  std::set<WawHazardKey> reported_war_hazards;
  std::set<WawHazardKey> reported_waw_hazards;
  std::set<LdsHazardKey> reported_lds_hazards;

  // Collapse the per-register duplicates a wide operand would otherwise
  // produce. Checked in addition to the per-register sets above, never
  // instead of them, and kept separate per hazard kind so that a RAW report
  // does not suppress a WAR or WAW on the same instruction pair.
  std::set<RegisterHazardKey> reported_raw_pairs;
  std::set<RegisterHazardKey> reported_war_pairs;
  std::set<RegisterHazardKey> reported_waw_pairs;

  void reset() {
    pending_vgpr_writes.clear();
    pending_sgpr_writes.clear();
    vmem_load_fifo.clear();
    smem_load_fifo.clear();
    lds_fifo.clear();
    pending_vgpr_writes_ds.clear();
    flat_vgpr_ds_fifo.clear();
    pending_acc_vgpr_writes.clear();
    acc_vmem_load_fifo.clear();
    pending_acc_vgpr_writes_ds.clear();
    flat_acc_vgpr_ds_fifo.clear();
    pending_acc_vgpr_reads.clear();
    acc_vmem_store_fifo.clear();
    pending_vgpr_reads.clear();
    vmem_store_fifo.clear();
    vmem_store_ops.clear();
    lds_read_fifo.clear();
    tensor_lds_fifo.clear();
    reported_raw_hazards.clear();
    reported_war_hazards.clear();
    reported_waw_hazards.clear();
    reported_lds_hazards.clear();
    reported_raw_pairs.clear();
    reported_war_pairs.clear();
    reported_waw_pairs.clear();
  }
};

} // namespace hazard_core
