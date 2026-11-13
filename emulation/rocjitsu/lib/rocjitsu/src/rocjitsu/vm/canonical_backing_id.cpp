// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/canonical_backing_id.h"

#include <atomic>
#include <exception>
#include <limits>

namespace rocjitsu {

uint64_t allocate_canonical_backing_id() noexcept {
  static std::atomic<uint64_t> next{1};
  uint64_t current = next.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<uint64_t>::max()) {
    if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
      return current;
  }
  std::terminate();
}

} // namespace rocjitsu
