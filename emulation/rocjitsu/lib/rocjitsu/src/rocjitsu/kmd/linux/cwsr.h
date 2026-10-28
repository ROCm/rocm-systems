// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cwsr.h
/// @brief KFD topology and device-capacity adaptation for the shared CWSR codec.

#ifndef ROCJITSU_KMD_LINUX_CWSR_H_
#define ROCJITSU_KMD_LINUX_CWSR_H_

#include "rocjitsu/config/kfd_device_config.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/vm/amdgpu/cwsr.h"

#include <algorithm>
#include <cstdint>

namespace rocjitsu::kmd {

/// @brief Select the codec architecture from the GC version used by KFD.
/// @details DBG_TRAP has a SoC architecture, while topology receives a
/// gfx_target_version and translates it to the GC version. Keep those
/// spellings adjacent; KfdTopologyTest.ArchAndGcSpellingsOfTheCwsrGateAgree
/// pins their exact agreement.
/// \NPI update this alongside amdgpu::cwsr_layout_modelled() when the codec learns a
/// new record layout.
/// @param gc_ip_version Packed GC hardware IP version, as
///        gc_ip_version_for_gfx_target_version() produces.
/// @returns The exact modelled architecture, or INVALID for unsupported ones.
constexpr rj_code_arch_t cwsr_arch_for_gc_ip_version(uint32_t gc_ip_version) {
  if (gc_ip_version == make_gc_ip_version(9, 4, 3)) // gfx942
    return ROCJITSU_CODE_ARCH_CDNA3;
  if (gc_ip_version == make_gc_ip_version(9, 5, 0)) // gfx950
    return ROCJITSU_CODE_ARCH_CDNA4;
  if (gc_ip_version == make_gc_ip_version(12, 1, 0)) // gfx1250
    return ROCJITSU_CODE_ARCH_CDNA5;
  return ROCJITSU_CODE_ARCH_INVALID;
}

/// @brief The same layout selection as amdgpu::cwsr_layout_kind(), for KFD topology.
constexpr amdgpu::CwsrLayoutKind cwsr_layout_kind_for_gc_ip_version(uint32_t gc_ip_version) {
  return amdgpu::cwsr_layout_kind(cwsr_arch_for_gc_ip_version(gc_ip_version));
}

constexpr bool cwsr_layout_modelled_for_gc_ip_version(uint32_t gc_ip_version) {
  return cwsr_layout_kind_for_gc_ip_version(gc_ip_version) != amdgpu::CwsrLayoutKind::Unsupported;
}

/// @brief Whether a device has the geometry needed for an @p arch capacity.
/// @details The caller separately decides whether this device is published.
constexpr bool cwsr_capacity_eligible_device(const config::KfdDeviceConfig &device,
                                             rj_code_arch_t arch) {
  return device.simd_count != 0 && device.simd_per_cu != 0 && device.max_waves_per_simd != 0 &&
         cwsr_arch_for_gc_ip_version(
             gc_ip_version_for_gfx_target_version(device.gfx_target_version)) == arch;
}

[[nodiscard]] constexpr amdgpu::CwsrProducerLimits
cwsr_producer_limits_for_device(const config::KfdDeviceConfig &device, uint32_t num_xcc) {
  const rj_code_arch_t arch =
      cwsr_arch_for_gc_ip_version(gc_ip_version_for_gfx_target_version(device.gfx_target_version));
  if (num_xcc == 0 || !cwsr_capacity_eligible_device(device, arch))
    return {};
  const uint64_t kfd_cus =
      ((uint64_t{device.simd_count} + device.simd_per_cu - 1) / device.simd_per_cu + num_xcc - 1) /
      num_xcc;
  // KFD can advertise only enabled SIMDs, while the simulator can stop waves
  // in every configured CU slot. Bound the producer's complete XCC.
  const uint64_t arrays = std::max(1u, device.num_shader_arrays_per_engine);
  if (device.num_shader_engines != 0 &&
      arrays > amdgpu::kCwsrMaxAlignedBytes / device.num_shader_engines)
    return {};
  const uint64_t shader_arrays = uint64_t{device.num_shader_engines} * arrays;
  if (shader_arrays != 0 && device.num_cu_per_sh > amdgpu::kCwsrMaxAlignedBytes / shader_arrays)
    return {};
  const uint64_t simulated_cus = shader_arrays * device.num_cu_per_sh;
  const uint64_t active_cus = std::max(kfd_cus, simulated_cus);
  const uint64_t slots_per_cu = std::max(uint64_t{device.simd_per_cu} * device.max_waves_per_simd,
                                         uint64_t{device.max_slots_scratch_cu});
  if (active_cus == 0 || slots_per_cu == 0 ||
      active_cus > amdgpu::kCwsrMaxAlignedBytes / slots_per_cu)
    return {};
  return {active_cus, active_cus * slots_per_cu};
}

/// @brief Size one XCC's save area from the KFD device description.
[[nodiscard]] inline amdgpu::CwsrCapacity
cwsr_capacity_for_device(const config::KfdDeviceConfig &device, uint32_t num_xcc) {
  const rj_code_arch_t arch =
      cwsr_arch_for_gc_ip_version(gc_ip_version_for_gfx_target_version(device.gfx_target_version));
  if (num_xcc == 0 || !cwsr_capacity_eligible_device(device, arch))
    return {};
  return amdgpu::cwsr_capacity_for_producer(arch, cwsr_producer_limits_for_device(device, num_xcc),
                                            device.lds_size_kb);
}

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_LINUX_CWSR_H_
