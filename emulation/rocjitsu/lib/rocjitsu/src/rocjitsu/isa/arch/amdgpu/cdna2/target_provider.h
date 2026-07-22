// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_TARGET_PROVIDER_H_

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna2 {

inline constexpr std::array<std::string_view, 1> target_aliases{"gfx90a"};
inline constexpr std::array<IsaGpuTargetDescription, 1> target_gpu_targets{{
    {ROCJITSU_CODE_TARGET_GFX90A, "gfx90a", EF_AMDGPU_MACH_AMDGCN_GFX90A},
}};
inline constexpr IsaTargetDescription target_description{
    .id = "cdna2",
    .aliases = target_aliases,
    .architecture_id = ROCJITSU_CODE_ARCH_CDNA2,
    .gpu_targets = target_gpu_targets,
    .supports_execution = true,
};

IsaTargetRegistryError register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::cdna2

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::cdna2::register_target)
#endif
