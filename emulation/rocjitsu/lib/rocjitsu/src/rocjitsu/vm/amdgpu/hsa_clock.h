// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_
#define ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_

/// @file hsa_clock.h
/// @brief Shared synthetic HSA system-clock timestamp helper.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/isa_family.h"

#include <chrono>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Return a monotonic timestamp in the guest HSA system-clock domain.
///
inline uint64_t hsa_system_timestamp() {
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

/// Architectural RTC frequency in MHz. Other models use a nanosecond device clock.
constexpr uint32_t gpu_clock_mhz(rocjitsu::IsaFamily arch) {
  switch (arch) {
  case rocjitsu::IsaFamily::Cdna1:
  case rocjitsu::IsaFamily::Cdna2:
    return 25;
  case rocjitsu::IsaFamily::Cdna3:
  case rocjitsu::IsaFamily::Cdna4:
    return 100;
  default:
    return 1000;
  }
}

constexpr uint64_t gpu_clock_from_ns(uint64_t ns, uint32_t frequency_mhz) {
  return ns / (1000 / frequency_mhz);
}

/// GPU profiling packets and clock calibration share the steady-clock epoch.
inline uint64_t gpu_timestamp(rocjitsu::IsaFamily arch) {
  return gpu_clock_from_ns(hsa_system_timestamp(), gpu_clock_mhz(arch));
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_
