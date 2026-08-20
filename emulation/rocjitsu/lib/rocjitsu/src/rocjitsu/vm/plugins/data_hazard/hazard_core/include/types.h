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

#include <cstdint>
#include <deque>
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
  XCNT,  // Address translation counter (s_wait_xcnt) — tracks XACK/XNACK, not data completion
  // TODO: Implement tracking for these wait counter types when needed
  ASYNC,  // Async copy operations (s_wait_asynccnt) - not yet tracked
  TENSOR, // Tensor DMA operations (s_wait_tensorcnt) — tensor_load_to_lds, tensor_store_from_lds
  // Wavegroup semaphore synchronization (s_sema_wait). Unlike the entries
  // above this is not a per-wave counter: it orders LDS accesses between the
  // waves of a wavegroup, so it has no FIFO to drain and reaches the engine as
  // WaitAction::is_wavegroup_semaphore_wait rather than as a WaitCounterClear.
  SEMA
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

struct PendingAsyncOp {
  EntityId instruction_id = 0;
  uint64_t pc = 0;
  WaitCntType wait_type = WaitCntType::NONE;
  uint32_t size = 0;
  // True when this is the DScnt-side entry of a flat load (which also has
  // a LOADcnt-side entry in pending_vgpr_writes). When the LOADcnt side is
  // cleared, this DScnt-only pending is safe and should not fire a RAW.
  bool is_flat_ds = false;
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

struct WaveState {
  // === RAW hazard tracking (reads before async writes complete) ===
  std::unordered_map<uint32_t, PendingAsyncOp> pending_vgpr_writes;
  std::unordered_map<uint32_t, PendingAsyncOp> pending_sgpr_writes;

  // FIFO queues for proper waitcnt N handling (RAW)
  // Design trade-off for LDS: We use address ranges rather than a per-byte bitmap.
  // Ranges are memory-efficient for typical contiguous accesses, while a
  // bitmap would require 8KB per wave (LDS is 64KB). Ranges may have rare
  // false positives on adjacent non-overlapping accesses.
  std::deque<std::pair<uint32_t, PendingAsyncOp>> vmem_load_fifo;
  std::deque<std::pair<uint32_t, PendingAsyncOp>> smem_load_fifo;
  std::deque<std::pair<uint32_t, PendingAsyncOp>> lds_fifo; // (start_addr, op)

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

  // Pending async LDS reads (DS loads in flight): address -> op.
  // WAR hazard: a DS write to an overlapping address before s_wait_dscnt
  // could corrupt LDS before the in-flight read captures it.
  std::deque<std::pair<uint32_t, PendingAsyncOp>> lds_read_fifo; // (start_addr, op)

  // TENSORcnt-tracked LDS writes (tensor_load_to_lds, tensor_store_from_lds)
  std::deque<std::pair<uint32_t, PendingAsyncOp>> tensor_lds_fifo;

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
