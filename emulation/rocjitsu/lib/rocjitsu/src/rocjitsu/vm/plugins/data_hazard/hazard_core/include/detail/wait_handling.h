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
#include "hash_utils.h"
#include "hazard_events.h"
#include "types.h"
#include "wait_suggestion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hazard_core {

inline void clear_address_fifo(std::deque<std::pair<uint32_t, PendingAsyncOp>> &fifo,
                               uint32_t keep_count, WaitCntType wait_type) {
  if (keep_count == 0) {
    fifo.erase(std::remove_if(
                   fifo.begin(), fifo.end(),
                   [wait_type](const auto &entry) { return entry.second.wait_type == wait_type; }),
               fifo.end());
    return;
  }

  uint32_t matching_entries = 0;
  for (const auto &entry : fifo) {
    if (entry.second.wait_type == wait_type)
      ++matching_entries;
  }

  if (keep_count >= matching_entries)
    return;

  uint32_t entries_to_remove = matching_entries - keep_count;
  fifo.erase(std::remove_if(fifo.begin(), fifo.end(),
                            [&](const auto &entry) {
                              if (entries_to_remove == 0 || entry.second.wait_type != wait_type)
                                return false;
                              --entries_to_remove;
                              return true;
                            }),
             fifo.end());
}

enum class VmemPendingSource {
  Register,
  Lds,
};

struct VmemPendingRef {
  VmemPendingSource source;
  uint32_t key;
  EntityId instruction_id;

  bool operator==(const VmemPendingRef &other) const {
    return source == other.source && key == other.key && instruction_id == other.instruction_id;
  }
};

struct VmemPendingRefHash {
  size_t operator()(const VmemPendingRef &ref) const noexcept {
    size_t seed = std::hash<int>{}(static_cast<int>(ref.source));
    hash_combine(seed, ref.key);
    hash_combine(seed, ref.instruction_id);
    return seed;
  }
};

using VmemPendingRefCounts = std::unordered_map<VmemPendingRef, uint32_t, VmemPendingRefHash>;

inline bool consume_vmem_pending_ref(VmemPendingRefCounts &counts, const VmemPendingRef &ref) {
  auto it = counts.find(ref);
  if (it == counts.end() || it->second == 0)
    return false;
  --it->second;
  return true;
}

inline void clear_vmem_pending_ops(WaveState &wave, uint32_t keep_count) {
  std::vector<VmemPendingRef> pending;
  pending.reserve(wave.vmem_load_fifo.size() + wave.acc_vmem_load_fifo.size() +
                  wave.lds_fifo.size());

  for (const auto &entry : wave.vmem_load_fifo)
    pending.push_back({VmemPendingSource::Register, entry.first, entry.second.instruction_id});
  for (const auto &entry : wave.acc_vmem_load_fifo)
    pending.push_back({VmemPendingSource::Register, entry.first, entry.second.instruction_id});
  for (const auto &entry : wave.lds_fifo) {
    if (entry.second.wait_type == WaitCntType::VMEM)
      pending.push_back({VmemPendingSource::Lds, entry.first, entry.second.instruction_id});
  }

  if (keep_count >= pending.size())
    return;

  std::stable_sort(pending.begin(), pending.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.instruction_id < rhs.instruction_id;
  });

  VmemPendingRefCounts to_remove;
  to_remove.reserve(pending.size() - keep_count);
  for (auto it = pending.begin(); it != pending.end() - keep_count; ++it)
    ++to_remove[*it];

  auto drain_register_fifo = [&](std::deque<std::pair<uint32_t, PendingAsyncOp>> &load_fifo,
                                 std::unordered_map<uint32_t, PendingAsyncOp> &pending_writes) {
    std::vector<uint32_t> removed_registers;
    std::deque<std::pair<uint32_t, PendingAsyncOp>> kept_load_fifo;
    for (auto &entry : load_fifo) {
      VmemPendingRef ref{VmemPendingSource::Register, entry.first, entry.second.instruction_id};
      if (consume_vmem_pending_ref(to_remove, ref)) {
        removed_registers.push_back(entry.first);
        continue;
      }
      kept_load_fifo.push_back(std::move(entry));
    }
    load_fifo = std::move(kept_load_fifo);

    for (uint32_t reg : removed_registers) {
      const bool still_pending =
          std::any_of(load_fifo.begin(), load_fifo.end(),
                      [reg](const auto &entry) { return entry.first == reg; });
      if (!still_pending)
        pending_writes.erase(reg);
    }
  };

  drain_register_fifo(wave.vmem_load_fifo, wave.pending_vgpr_writes);
  drain_register_fifo(wave.acc_vmem_load_fifo, wave.pending_acc_vgpr_writes);

  decltype(wave.lds_fifo) kept_lds_fifo;
  for (auto &entry : wave.lds_fifo) {
    VmemPendingRef ref{VmemPendingSource::Lds, entry.first, entry.second.instruction_id};
    if (entry.second.wait_type == WaitCntType::VMEM && consume_vmem_pending_ref(to_remove, ref))
      continue;
    kept_lds_fifo.push_back(std::move(entry));
  }
  wave.lds_fifo = std::move(kept_lds_fifo);
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
    clear_register_fifo(wave->vmem_store_fifo, wave->pending_vgpr_reads, keep_count);
    clear_register_fifo(wave->acc_vmem_store_fifo, wave->pending_acc_vgpr_reads, keep_count);
    break;

  case WaitCntType::LDS: {
    clear_address_fifo(wave->lds_fifo, keep_count, WaitCntType::LDS);

    clear_register_fifo(wave->flat_vgpr_ds_fifo, wave->pending_vgpr_writes_ds, keep_count);
    clear_register_fifo(wave->flat_acc_vgpr_ds_fifo, wave->pending_acc_vgpr_writes_ds, keep_count);

    auto &read_fifo = wave->lds_read_fifo;
    if (keep_count == 0) {
      read_fifo.clear();
    } else {
      while (read_fifo.size() > keep_count)
        read_fifo.pop_front();
    }
    break;
  }

  case WaitCntType::TENSOR: {
    auto &tensor_fifo = wave->tensor_lds_fifo;
    if (keep_count == 0) {
      tensor_fifo.clear();
    } else {
      while (tensor_fifo.size() > keep_count)
        tensor_fifo.pop_front();
    }
    break;
  }

  case WaitCntType::XCNT:
    // Address translation counter: no data hazard FIFO to drain.
    break;

  case WaitCntType::NONE:
  case WaitCntType::ASYNC:
    return false;
  }
  return true;
}

} // namespace hazard_core
