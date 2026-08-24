// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// FIFO operations for hazard detection.
//
// This header provides template utilities for managing the FIFO queues that
// track in-flight async operations. These FIFOs implement correct "wait N"
// semantics where the N oldest operations complete first.
//
// clear_register_fifo<FifoType, MapType>
//   Drains the oldest entries from a register FIFO (e.g. vmem_load_fifo)
//   and removes the corresponding entries from the pending-op map, honoring
//   a keep_count so that "wait N" semantics are preserved.
//
// FIFO Drain Semantics:
//   GPU wait instructions like s_wait_loadcnt N mean "wait until at most N
//   operations remain pending". So:
//   - wait 0: drain all operations (nothing pending after)
//   - wait 1: keep the 1 most recent operation, drain all older ones
//   - wait N: keep the N most recent operations, drain older ones
//
//   The counter counts *issued instructions*, not the registers or addresses
//   one instruction touches: global_load_dwordx4 occupies a single LOADcnt slot
//   while leaving four entries behind, one per destination register. Draining
//   therefore works in units of instruction id, and every entry belonging to a
//   retired instruction retires with it. Counting raw entries instead would let
//   s_wait_loadcnt 1 retire three of that load's four destinations while the
//   load is still in flight, and reads of them would go unreported.
//
//   The FIFO is ordered oldest-first (front = oldest). When clearing:
//   1. Collect the distinct instruction ids present, oldest first
//   2. Drop every entry belonging to the oldest ids beyond keep_count
//   3. Remove from pending map only if no remaining FIFO entry references it
//      (handles cases where same register appears multiple times)
//
// This header is simulator-agnostic - it only operates on the FIFO and map
// data structures via templates, so frontend adapters can share the same
// wait-drain semantics.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// =============================================================================
// FIFO clearing helper
// =============================================================================

/// Instruction that left a FIFO entry behind. Entries are keyed by register or
/// address in most FIFOs and bare pending operations in the ones that track
/// counter occupancy alone, so both shapes are read the same way.
template <typename Key, typename Op>
uint64_t entry_instruction_id(const std::pair<Key, Op> &entry) {
  return entry.second.instruction_id;
}

template <typename Op> uint64_t entry_instruction_id(const Op &op) { return op.instruction_id; }

/// Instruction ids a "wait N" retires, oldest first, counting one slot per
/// issued instruction however many FIFO entries that instruction left behind.
///
/// @tparam FifoType A deque of (key, PendingAsyncOp) pairs or of bare ops
/// @tparam MatchFn  Predicate on an entry selecting the ones this counter owns
///
/// @param fifo       The FIFO to inspect
/// @param keep_count Number of most recent operations to keep (0 = retire all)
/// @param matches    Only entries satisfying this take up a counter slot
/// @return The ids to retire; empty when the counter is already at or below
///         @p keep_count
template <typename FifoType, typename MatchFn>
std::vector<uint64_t> instructions_to_retire(const FifoType &fifo, uint32_t keep_count,
                                             MatchFn matches) {
  std::vector<uint64_t> outstanding;
  for (const auto &entry : fifo) {
    if (!matches(entry))
      continue;
    const uint64_t id = entry_instruction_id(entry);
    if (std::find(outstanding.begin(), outstanding.end(), id) == outstanding.end())
      outstanding.push_back(id);
  }

  if (outstanding.size() <= keep_count)
    return {};

  outstanding.resize(outstanding.size() - keep_count);
  return outstanding;
}

/// True when @p id is one of the instructions @p retiring names.
inline bool is_retiring(const std::vector<uint64_t> &retiring, uint64_t id) {
  return std::find(retiring.begin(), retiring.end(), id) != retiring.end();
}

/// Drop every entry the named instructions left in a register FIFO, and clear
/// the registers they were the last pending writer or reader of.
///
/// @tparam FifoType  A deque of (register_index, PendingAsyncOp) pairs
/// @tparam MapType   An unordered_map from register_index to PendingAsyncOp
///
/// @param fifo        The FIFO queue to drain (modified in place)
/// @param pending_map The pending operation map (modified in place)
/// @param retiring    Instruction ids to retire, as from instructions_to_retire
template <typename FifoType, typename MapType>
void retire_register_entries(FifoType &fifo, MapType &pending_map,
                             const std::vector<uint64_t> &retiring) {
  if (retiring.empty())
    return;

  std::vector<uint32_t> retired_registers;
  FifoType kept;
  for (auto &entry : fifo) {
    if (is_retiring(retiring, entry.second.instruction_id)) {
      retired_registers.push_back(entry.first);
      continue;
    }
    kept.push_back(std::move(entry));
  }
  fifo = std::move(kept);

  for (uint32_t reg : retired_registers) {
    const bool still_pending = std::any_of(fifo.begin(), fifo.end(),
                                           [reg](const auto &entry) { return entry.first == reg; });
    if (!still_pending)
      pending_map.erase(reg);
  }
}

/// Drains the oldest operations from a register FIFO, keeping only `keep_count`
/// most recent ones. Updates the pending map to reflect completed operations.
///
/// @tparam FifoType  A deque of (register_index, PendingAsyncOp) pairs
/// @tparam MapType   An unordered_map from register_index to PendingAsyncOp
///
/// @param fifo        The FIFO queue to drain (modified in place)
/// @param pending_map The pending operation map (modified in place)
/// @param keep_count  Number of most recent operations to keep (0 = clear all)
///
/// Edge case handling:
///   A register may appear multiple times in the FIFO (e.g., two consecutive
///   loads to the same VGPR). A register is removed from the pending map only
///   once no remaining FIFO entry references it.
template <typename FifoType, typename MapType>
void clear_register_fifo(FifoType &fifo, MapType &pending_map, uint32_t keep_count) {
  retire_register_entries(
      fifo, pending_map,
      instructions_to_retire(fifo, keep_count, [](const auto &) { return true; }));
}
