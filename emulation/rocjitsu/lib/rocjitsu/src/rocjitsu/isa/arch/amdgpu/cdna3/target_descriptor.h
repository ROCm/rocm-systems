// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna3 {

inline constexpr std::array<std::string_view, 1> kTargetAliases{"gfx942"};
inline constexpr std::array<IsaGpuTargetDescription, 1> kModelGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX942, "gfx942", EF_AMDGPU_MACH_AMDGCN_GFX942},
}};
inline constexpr std::array<IsaGpuTargetDescription, 1> kExecutionGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX942,
     "gfx942",
     EF_AMDGPU_MACH_AMDGCN_GFX942,
     0,
     {.execution_implemented = true}},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(std::span<const IsaGpuTargetDescription> gpu_targets,
                       bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "cdna3",
      .aliases = kTargetAliases,
      .architecture_id = ROCJITSU_CODE_ARCH_CDNA3,
      .gpu_targets = gpu_targets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::cdna3
