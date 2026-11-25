// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>

namespace simdojo {

/// Identifies the same underlying storage across aliases in a running model.
/// Its owner assigns one ID and passes it to each mapping and subrange.
/// Zero denotes anonymous storage.
using BackingStoreId = uint64_t;

/// Allocate a fresh nonzero ID. The caller establishes which aliases share it.
[[nodiscard]] inline BackingStoreId allocate_backing_store_id() noexcept {
  static std::atomic<BackingStoreId> next{1};
  auto current = next.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<BackingStoreId>::max()) {
    if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
      return current;
    }
  }
  std::terminate();
}

} // namespace simdojo
