// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PATH_TEST_HOOKS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PATH_TEST_HOOKS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

/// Test-only tracker for successful SIMD fast-path execution on this thread.
///
/// Like util::set_force_scalar_for_testing, this observes only execution linked
/// into the caller's module. Nested observers restore the previous destination.
class ScopedSimdFastPathTracker {
public:
  ScopedSimdFastPathTracker() : previous_sink_(detail::active_simd_fast_path_bits) {
    detail::active_simd_fast_path_bits = &executed_paths_;
  }

  ~ScopedSimdFastPathTracker() { detail::active_simd_fast_path_bits = previous_sink_; }

  ScopedSimdFastPathTracker(const ScopedSimdFastPathTracker &) = delete;
  ScopedSimdFastPathTracker &operator=(const ScopedSimdFastPathTracker &) = delete;

  [[nodiscard]] bool was_executed(SimdFastPath path) const {
    return (executed_paths_ & path_bit(path)) != 0;
  }

  [[nodiscard]] bool only_was_executed(SimdFastPath path) const {
    return executed_paths_ == path_bit(path);
  }

  [[nodiscard]] bool none_was_executed() const { return executed_paths_ == 0; }

private:
  static constexpr uint64_t path_bit(SimdFastPath path) {
    return uint64_t{1} << static_cast<uint8_t>(path);
  }

  uint64_t executed_paths_ = 0;
  uint64_t *previous_sink_;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PATH_TEST_HOOKS_H_
