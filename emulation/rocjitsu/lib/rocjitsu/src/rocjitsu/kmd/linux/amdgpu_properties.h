// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_AMDGPU_PROPERTIES_H_
#define ROCJITSU_KMD_LINUX_AMDGPU_PROPERTIES_H_

#include "rocjitsu/code/rj_code.h"

#include <cstdint>

namespace rocjitsu::kmd {

inline constexpr uint32_t kAmdgpuVramTypeHbm = 6;
inline constexpr uint32_t kAmdgpuVramTypeGddr6 = 9;

inline constexpr uint32_t gb_addr_config_for_arch(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return 0x545;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return 0x8200545;
  default:
    return 0;
  }
}

inline constexpr uint32_t gb_addr_config_for_gfx_target_version(uint32_t gfx_target_version) {
  if (gfx_target_version >= 120000 && gfx_target_version < 130000)
    return gb_addr_config_for_arch(ROCJITSU_CODE_ARCH_RDNA4);
  if (gfx_target_version >= 110000 && gfx_target_version < 120000)
    return gb_addr_config_for_arch(ROCJITSU_CODE_ARCH_RDNA3);
  return 0;
}

inline constexpr uint32_t drm_shader_engine_count(uint32_t kfd_array_count,
                                                  uint32_t arrays_per_engine) {
  return arrays_per_engine == 0 ? kfd_array_count : kfd_array_count / arrays_per_engine;
}

inline constexpr uint32_t drm_quad_shader_pipe_count(uint32_t kfd_array_count) {
  return kfd_array_count;
}

inline constexpr uint32_t drm_cu_active_number(uint32_t kfd_array_count,
                                               uint32_t cu_per_shader_array) {
  return kfd_array_count * cu_per_shader_array;
}

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_LINUX_AMDGPU_PROPERTIES_H_
