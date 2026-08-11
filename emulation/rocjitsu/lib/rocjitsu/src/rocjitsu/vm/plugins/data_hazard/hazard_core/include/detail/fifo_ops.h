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
//   - wait 0: drain all entries (nothing pending after)
//   - wait 1: keep the 1 most recent entry, drain all older ones
//   - wait N: keep the N most recent entries, drain older ones
//
//   The FIFO is ordered oldest-first (front = oldest). When clearing:
//   1. Calculate how many entries to remove: fifo.size() - keep_count
//   2. Pop entries from the front (oldest first)
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

// =============================================================================
// FIFO clearing helper
// =============================================================================

/// Drains the oldest entries from a register FIFO, keeping only `keep_count`
/// most recent entries. Updates the pending map to reflect completed operations.
///
/// @tparam FifoType  A deque of (register_index, PendingAsyncOp) pairs
/// @tparam MapType   An unordered_map from register_index to PendingAsyncOp
///
/// @param fifo        The FIFO queue to drain (modified in place)
/// @param pending_map The pending operation map (modified in place)
/// @param keep_count  Number of most recent entries to keep (0 = clear all)
///
/// Edge case handling:
///   A register may appear multiple times in the FIFO (e.g., two consecutive
///   loads to the same VGPR). When popping an entry, we only remove from the
///   pending map if no remaining FIFO entry references that register.
template <typename FifoType, typename MapType>
void clear_register_fifo(FifoType &fifo, MapType &pending_map, uint32_t keep_count) {
  if (keep_count == 0) {
    // Fast path: clear everything
    for (const auto &entry : fifo)
      pending_map.erase(entry.first);
    fifo.clear();
  } else if (keep_count < fifo.size()) {
    // Partial drain: remove oldest entries until only keep_count remain
    std::size_t to_clear = fifo.size() - keep_count;
    for (std::size_t i = 0; i < to_clear; ++i) {
      uint32_t reg = fifo.front().first;
      fifo.pop_front();
      // Check if this register still has pending operations in the FIFO
      bool still_pending = std::any_of(fifo.begin(), fifo.end(),
                                       [reg](const auto &entry) { return entry.first == reg; });
      if (!still_pending)
        pending_map.erase(reg);
    }
  }
  // If keep_count >= fifo.size(), nothing to do
}
