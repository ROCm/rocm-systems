// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_
#define ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_

/// @file hsa_clock.h
/// @brief Shared synthetic HSA system-clock timestamp helper.

#include <chrono>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Return a monotonic timestamp in the guest HSA system-clock domain.
///
/// @details `LinuxKfd::fill_get_clock_counters_ioctl` exposes a synthetic
/// 1 GHz nanosecond clock. Dispatch profiling timestamps stored in
/// `amd_signal_t` use the same HSA system-clock domain, so HIP event timing can
/// subtract values written by the command processor and completion tracker
/// directly.
inline uint64_t hsa_system_timestamp() {
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_
