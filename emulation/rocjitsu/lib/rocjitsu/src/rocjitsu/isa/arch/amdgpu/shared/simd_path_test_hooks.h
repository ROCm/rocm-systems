// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"

#include <atomic>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Observe successful execution of the SimdFastPath families on this thread.
/// @details Uninstrumented SIMD routes (including F32/F64 helpers) are invisible:
/// an empty observation does not prove scalar execution, and a single recorded
/// family does not rule out untracked routes. Like util::set_force_scalar_for_testing,
/// this observes only execution linked into the caller's module. Nested observers
/// restore the previous destination; events belong only to the innermost observer.
class ScopedSimdFastPathTracker {
public:
  ScopedSimdFastPathTracker() : previous_sink_(detail::active_simd_fast_path_bits) {
    detail::active_simd_fast_path_bits = &executed_paths_;
    // The counter gates TLS access, not publication: the sink is thread-local.
    detail::simd_fast_path_observer_count.fetch_add(1, std::memory_order_relaxed);
  }

  ~ScopedSimdFastPathTracker() {
    detail::active_simd_fast_path_bits = previous_sink_;
    detail::simd_fast_path_observer_count.fetch_sub(1, std::memory_order_relaxed);
  }

  ScopedSimdFastPathTracker(const ScopedSimdFastPathTracker &) = delete;
  ScopedSimdFastPathTracker &operator=(const ScopedSimdFastPathTracker &) = delete;

  [[nodiscard]] bool tracked_path_executed(SimdFastPath path) const {
    return (executed_paths_ & path_bit(path)) != 0;
  }

  [[nodiscard]] bool only_tracked_path_executed(SimdFastPath path) const {
    return executed_paths_ == path_bit(path);
  }

  [[nodiscard]] bool no_tracked_path_executed() const { return executed_paths_ == 0; }

private:
  static constexpr uint64_t path_bit(SimdFastPath path) {
    return uint64_t{1} << static_cast<uint8_t>(path);
  }

  uint64_t executed_paths_ = 0;
  uint64_t *previous_sink_;
};

} // namespace rocjitsu::amdgpu
