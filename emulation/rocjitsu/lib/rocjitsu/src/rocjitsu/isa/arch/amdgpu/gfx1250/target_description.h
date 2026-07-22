// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_DESCRIPTION_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_DESCRIPTION_H_

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::gfx1250 {

inline constexpr std::array<IsaGpuTargetDescription, 1> gpu_targets{{
    {ROCJITSU_CODE_TARGET_GFX1250, "gfx1250", EF_AMDGPU_MACH_AMDGCN_GFX1250},
}};

constexpr IsaTargetDescription make_target_description(bool supports_execution) {
  return {
      .id = "gfx1250",
      .architecture_id = ROCJITSU_CODE_ARCH_GFX1250,
      .gpu_targets = gpu_targets,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::gfx1250

#endif // ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_DESCRIPTION_H_
