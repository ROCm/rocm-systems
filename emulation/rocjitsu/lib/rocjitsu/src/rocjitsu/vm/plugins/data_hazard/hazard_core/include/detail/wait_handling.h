// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Wait instruction handling utilities for hazard detection.
//
// This header declares pure utility functions for processing GPU wait
// instructions and managing pending async operation state. These functions
// are simulator-agnostic and shared by frontend adapters.
//
// Functions:
//   - find_pending_lds_write:        Find overlapping pending DScnt LDS write
//   - find_pending_tensor_lds_write: Find overlapping pending TENSORcnt LDS write
//   - find_pending_lds_read:         Find overlapping pending LDS read
//   - clear_pending_ops:             Drain FIFOs by counter type (wait N semantics)
//   - get_wait_suggestion:           Get human-readable wait instruction suggestion

#pragma once

#include "fifo_ops.h"
#include "hazard_events.h"
#include "types.h"
#include "wait_suggestion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hazard_core {

/// Drain the entries of one counter from an address-keyed FIFO. An instruction
/// scattering several accesses still occupies a single counter slot, so whole
/// instructions retire together.
inline void clear_address_fifo(std::deque<std::pair<uint32_t, PendingAsyncOp>> &fifo,
                               uint32_t keep_count, WaitCntType wait_type) {
  const auto matches = [wait_type](const auto &entry) {
    return entry.second.wait_type == wait_type;
  };
  const std::vector<uint64_t> retiring = instructions_to_retire(fifo, keep_count, matches);
  if (retiring.empty())
    return;

  fifo.erase(std::remove_if(fifo.begin(), fifo.end(),
                            [&](const auto &entry) {
                              return matches(entry) &&
                                     is_retiring(retiring, entry.second.instruction_id);
                            }),
             fifo.end());
}

/// Count the stores a wait covers, and retire the ones it drains.
///
/// Which counter a store is outstanding on is an architectural difference the
/// frontend resolves per store: gfx9 and CDNA count vector stores on the same
/// vmcnt as loads, gfx10 and later on a vscnt/storecnt of their own. Both
/// drains therefore look at the same queues and take only the entries tagged
/// with the counter they are clearing.
inline void note_store_ops(const WaveState &wave, WaitCntType counter,
                           const std::function<void(EntityId)> &note) {
  for (const auto &op : wave.vmem_store_ops) {
    if (op.wait_type == counter)
      note(op.instruction_id);
  }
  for (const auto &entry : wave.vmem_store_fifo) {
    if (entry.second.wait_type == counter)
      note(entry.second.instruction_id);
  }
  for (const auto &entry : wave.acc_vmem_store_fifo) {
    if (entry.second.wait_type == counter)
      note(entry.second.instruction_id);
  }
}

inline void retire_store_ops(WaveState &wave, const std::vector<uint64_t> &retiring) {
  if (retiring.empty())
    return;

  wave.vmem_store_ops.erase(std::remove_if(wave.vmem_store_ops.begin(), wave.vmem_store_ops.end(),
                                           [&retiring](const PendingAsyncOp &op) {
                                             return is_retiring(retiring, op.instruction_id);
                                           }),
                            wave.vmem_store_ops.end());
  retire_register_entries(wave.vmem_store_fifo, wave.pending_vgpr_reads, retiring);
  retire_register_entries(wave.acc_vmem_store_fifo, wave.pending_acc_vgpr_reads, retiring);
}

/// Drain LOADcnt, which spans the VGPR loads, the accumulator VGPR loads, the
/// VMEM-tracked LDS writes of global_load_lds, and — where one counter tracks
/// loads and stores together — the outstanding stores. The counter counts
/// instructions across all of them, so they are drained as one sequence
/// ordered by instruction id. Stores taking up slots of this counter is what
/// lets a gfx9 s_waitcnt vmcnt(2) retire a load that two later stores sit
/// behind.
inline void clear_vmem_pending_ops(WaveState &wave, uint32_t keep_count) {
  std::vector<uint64_t> outstanding;
  const auto note = [&outstanding](EntityId instruction_id) {
    if (std::find(outstanding.begin(), outstanding.end(), instruction_id) == outstanding.end())
      outstanding.push_back(instruction_id);
  };

  for (const auto &entry : wave.vmem_load_fifo)
    note(entry.second.instruction_id);
  for (const auto &entry : wave.acc_vmem_load_fifo)
    note(entry.second.instruction_id);
  for (const auto &entry : wave.lds_fifo) {
    if (entry.second.wait_type == WaitCntType::VMEM)
      note(entry.second.instruction_id);
  }
  note_store_ops(wave, WaitCntType::VMEM, note);

  if (outstanding.size() <= keep_count)
    return;

  // Ids increase with program order, which the per-FIFO walk above does not
  // preserve across FIFOs.
  std::sort(outstanding.begin(), outstanding.end());
  outstanding.resize(outstanding.size() - keep_count);

  retire_register_entries(wave.vmem_load_fifo, wave.pending_vgpr_writes, outstanding);
  retire_register_entries(wave.acc_vmem_load_fifo, wave.pending_acc_vgpr_writes, outstanding);

  wave.lds_fifo.erase(std::remove_if(wave.lds_fifo.begin(), wave.lds_fifo.end(),
                                     [&outstanding](const auto &entry) {
                                       return entry.second.wait_type == WaitCntType::VMEM &&
                                              is_retiring(outstanding, entry.second.instruction_id);
                                     }),
                      wave.lds_fifo.end());

  retire_store_ops(wave, outstanding);
}

/// Drain STOREcnt: the stores held for counter occupancy, plus the source
/// registers a frontend chose to keep live for the duration of a store.
inline void clear_store_pending_ops(WaveState &wave, uint32_t keep_count) {
  std::vector<uint64_t> outstanding;
  const auto note = [&outstanding](EntityId instruction_id) {
    if (std::find(outstanding.begin(), outstanding.end(), instruction_id) == outstanding.end())
      outstanding.push_back(instruction_id);
  };
  note_store_ops(wave, WaitCntType::STORE, note);

  if (outstanding.size() <= keep_count)
    return;

  std::sort(outstanding.begin(), outstanding.end());
  outstanding.resize(outstanding.size() - keep_count);

  retire_store_ops(wave, outstanding);
}

/// True when [address, address + size) and [other_address, other_address + other_size) intersect.
inline bool ranges_overlap(uint32_t address, uint32_t size, uint32_t other_address,
                           uint32_t other_size) {
  const uint64_t start = address;
  const uint64_t end = start + size;
  const uint64_t other_start = other_address;
  const uint64_t other_end = other_start + other_size;
  return start < other_end && other_start < end;
}

/// Find the first entry in an address-keyed FIFO whose range overlaps [address, address + size).
///
/// @param fifo    FIFO of (address, pending op) entries
/// @param address Start address of the access
/// @param size    Size of the access in bytes
/// @return Pointer to the first overlapping pending op, or nullptr if none
inline const PendingAsyncOp *
find_overlapping_pending_op(const std::deque<std::pair<uint32_t, PendingAsyncOp>> &fifo,
                            uint32_t address, uint32_t size) {
  for (const auto &entry : fifo) {
    if (ranges_overlap(address, size, entry.first, entry.second.size))
      return &entry.second;
  }
  return nullptr;
}

/// Find a pending LDS write (DScnt-tracked) that overlaps the given address range.
///
/// @param wave    Pointer to wave state (may be null)
/// @param address Start address of the LDS access
/// @param size    Size of the LDS access in bytes
/// @return Pointer to the first overlapping pending op, or nullptr if none
inline const PendingAsyncOp *find_pending_lds_write(const WaveState *wave, uint32_t address,
                                                    uint32_t size) {
  if (!wave)
    return nullptr;
  return find_overlapping_pending_op(wave->lds_fifo, address, size);
}

inline const PendingAsyncOp *find_pending_lds_write(WaveState *wave, uint32_t address,
                                                    uint32_t size) {
  return find_pending_lds_write(static_cast<const WaveState *>(wave), address, size);
}

/// Find a pending tensor LDS write (TENSORcnt-tracked) that overlaps the given address range.
///
/// @param wave    Pointer to wave state (may be null)
/// @param address Start address of the LDS access
/// @param size    Size of the LDS access in bytes
/// @return Pointer to the first overlapping pending op, or nullptr if none
inline const PendingAsyncOp *find_pending_tensor_lds_write(const WaveState *wave, uint32_t address,
                                                           uint32_t size) {
  if (!wave)
    return nullptr;
  return find_overlapping_pending_op(wave->tensor_lds_fifo, address, size);
}

inline const PendingAsyncOp *find_pending_tensor_lds_write(WaveState *wave, uint32_t address,
                                                           uint32_t size) {
  return find_pending_tensor_lds_write(static_cast<const WaveState *>(wave), address, size);
}

/// Find a pending LDS read that overlaps the given address range.
/// Used for WAR hazard detection: a write to an address with a pending read.
///
/// @param wave    Pointer to wave state (may be null)
/// @param address Start address of the LDS access
/// @param size    Size of the LDS access in bytes
/// @return Pointer to the first overlapping pending op, or nullptr if none
inline const PendingAsyncOp *find_pending_lds_read(const WaveState *wave, uint32_t address,
                                                   uint32_t size) {
  if (!wave)
    return nullptr;
  return find_overlapping_pending_op(wave->lds_read_fifo, address, size);
}

inline const PendingAsyncOp *find_pending_lds_read(WaveState *wave, uint32_t address,
                                                   uint32_t size) {
  return find_pending_lds_read(static_cast<const WaveState *>(wave), address, size);
}

/// Clear pending operations from the specified counter's FIFO, implementing
/// "wait N" semantics where the N most recent operations are kept.
///
/// This runs on the simulator's callback path, so an unsupported counter is
/// reported to the caller rather than thrown.
///
/// @param wave       Pointer to wave state (may be null, no-op if null)
/// @param type       Which wait counter type to clear (VMEM, SMEM, LDS, STORE, TENSOR)
/// @param keep_count Number of most recent operations to keep (0 = clear all)
/// @return False when @p type has no FIFO to drain (NONE, or ASYNC which is not
///         implemented yet); the wave is left untouched in that case
inline bool clear_pending_ops(WaveState *wave, WaitCntType type, uint32_t keep_count) {
  if (!wave)
    return true;

  switch (type) {
  case WaitCntType::VMEM:
    clear_vmem_pending_ops(*wave, keep_count);
    break;

  case WaitCntType::SMEM:
    // Scalar memory operations can complete out of order, so only a full
    // drain is a precise dependency boundary for SGPR results.
    if (keep_count == 0)
      clear_register_fifo(wave->smem_load_fifo, wave->pending_sgpr_writes, keep_count);
    break;

  case WaitCntType::STORE:
    clear_store_pending_ops(*wave, keep_count);
    break;

  case WaitCntType::LDS: {
    clear_address_fifo(wave->lds_fifo, keep_count, WaitCntType::LDS);

    clear_register_fifo(wave->flat_vgpr_ds_fifo, wave->pending_vgpr_writes_ds, keep_count);
    clear_register_fifo(wave->flat_acc_vgpr_ds_fifo, wave->pending_acc_vgpr_writes_ds, keep_count);

    clear_address_fifo(wave->lds_read_fifo, keep_count, WaitCntType::LDS);
    break;
  }

  case WaitCntType::TENSOR:
    clear_address_fifo(wave->tensor_lds_fifo, keep_count, WaitCntType::TENSOR);
    break;

  case WaitCntType::XCNT:
    // Address translation counter: no data hazard FIFO to drain.
    break;

  case WaitCntType::NONE:
  case WaitCntType::ASYNC:
    return false;
  }
  return true;
}

inline bool clear_pending_ops(WaveState *wave, const WaitCounterClear &counter) {
  return clear_pending_ops(wave, counter.kind, counter.keep_count);
}

} // namespace hazard_core
