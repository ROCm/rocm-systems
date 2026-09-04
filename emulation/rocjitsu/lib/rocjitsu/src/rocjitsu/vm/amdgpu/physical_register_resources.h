// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file physical_register_resources.h
/// @brief Physical SIMD register-file capacities and allocation granules.

#ifndef ROCJITSU_VM_AMDGPU_PHYSICAL_REGISTER_RESOURCES_H_
#define ROCJITSU_VM_AMDGPU_PHYSICAL_REGISTER_RESOURCES_H_

#include "rocjitsu/base/api.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Descriptor-derived VGPR allocation carried with one wave.
///
/// @details `total` is the physical allocation charged to SIMD occupancy. On
/// CDNA2-4 that unified allocation is split at COMPUTE_PGM_RSRC3.ACCUM_OFFSET:
/// `ordinary` names the v0-based prefix and `accumulator` names the acc0-based
/// suffix. Other architectures use only the ordinary component.
struct WaveVgprAllocation {
  uint32_t total = 0;
  uint32_t ordinary = 0;
  uint32_t accumulator = 0;
};

/// @brief Register resources owned by one physical SIMD and the CU layout.
///
/// VGPR counts use register names at @ref native_wave_size lanes. A Wave64
/// allocation therefore consumes twice as many physical lanes as the same
/// Wave32 allocation on a Wave32-native target. SGPR occupancy stopped limiting
/// waves with GFX10; those targets retain the physical size for description but
/// do not use it as an admission constraint.
struct PhysicalRegisterProperties {
  uint32_t simds_per_cu = 0;
  uint32_t max_waves_per_simd = 0;
  uint32_t native_wave_size = 0;
  uint32_t vgprs_per_simd = 0;
  uint32_t vgpr_alloc_granule_wave32 = 0;
  uint32_t vgpr_alloc_granule_wave64 = 0;
  uint32_t sgprs_per_simd = 800;
  uint32_t sgpr_alloc_granule = 16;
  bool sgpr_occupancy_limited = true;
};

/// @brief Return physical register-file parameters for an ISA family.
///
/// Values follow LLVM AMDGPU's occupancy model
/// (`AMDGPUBaseInfo::{getTotalNumVGPRs,getTotalNumSGPRs,
/// getVGPRAllocGranule,getSGPRAllocGranule}`) and RocJitsu's modeled CU layout.
[[nodiscard]] constexpr PhysicalRegisterProperties
physical_register_properties(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return {.simds_per_cu = 4,
            .max_waves_per_simd = 10,
            .native_wave_size = 64,
            .vgprs_per_simd = 256,
            .vgpr_alloc_granule_wave64 = 4};
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return {.simds_per_cu = 4,
            .max_waves_per_simd = 8,
            .native_wave_size = 64,
            .vgprs_per_simd = 512,
            .vgpr_alloc_granule_wave64 = 8};
  case ROCJITSU_CODE_ARCH_RDNA1:
    return {.simds_per_cu = 2,
            .max_waves_per_simd = 20,
            .native_wave_size = 32,
            .vgprs_per_simd = 1024,
            .vgpr_alloc_granule_wave32 = 8,
            .vgpr_alloc_granule_wave64 = 4,
            .sgpr_occupancy_limited = false};
  case ROCJITSU_CODE_ARCH_RDNA2:
    return {.simds_per_cu = 2,
            .max_waves_per_simd = 16,
            .native_wave_size = 32,
            .vgprs_per_simd = 1024,
            .vgpr_alloc_granule_wave32 = 16,
            .vgpr_alloc_granule_wave64 = 8,
            .sgpr_occupancy_limited = false};
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return {.simds_per_cu = 2,
            .max_waves_per_simd = 16,
            .native_wave_size = 32,
            .vgprs_per_simd = 1536,
            .vgpr_alloc_granule_wave32 = 24,
            .vgpr_alloc_granule_wave64 = 12,
            .sgpr_occupancy_limited = false};
  case ROCJITSU_CODE_ARCH_CDNA5:
    return {.simds_per_cu = 4,
            .max_waves_per_simd = 16,
            .native_wave_size = 32,
            .vgprs_per_simd = 1536,
            .vgpr_alloc_granule_wave32 = 24,
            .sgpr_occupancy_limited = false};
  default:
    return {};
  }
}

[[nodiscard]] constexpr uint32_t round_up_registers(uint32_t count, uint32_t granule) {
  return count == 0 || granule == 0 ? 0 : ((count + granule - 1) / granule) * granule;
}

/// @brief Physical VGPR units consumed by one wave.
[[nodiscard]] constexpr uint32_t physical_vgpr_units(rj_code_arch_t arch, uint32_t count,
                                                     uint32_t wave_size) {
  const auto properties = physical_register_properties(arch);
  const uint32_t granule = wave_size == 32   ? properties.vgpr_alloc_granule_wave32
                           : wave_size == 64 ? properties.vgpr_alloc_granule_wave64
                                             : 0;
  const uint32_t rounded = round_up_registers(count, granule);
  if (rounded == 0 || properties.native_wave_size == 0 || wave_size < properties.native_wave_size)
    return 0;
  return rounded * (wave_size / properties.native_wave_size);
}

/// @brief Physical SGPR units consumed by one wave, or zero when SGPRs do not
/// constrain occupancy for this architecture.
[[nodiscard]] constexpr uint32_t physical_sgpr_units(rj_code_arch_t arch, uint32_t count) {
  const auto properties = physical_register_properties(arch);
  if (!properties.sgpr_occupancy_limited)
    return 0;
  return round_up_registers(count, properties.sgpr_alloc_granule);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_PHYSICAL_REGISTER_RESOURCES_H_
