// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna3 {

inline constexpr std::array<std::string_view, 1> kTargetAliases{"gfx942"};
inline constexpr std::array<IsaGpuTargetDescription, 1> kTargetGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX942, "gfx942", EF_AMDGPU_MACH_AMDGCN_GFX942},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "cdna3",
      .aliases = kTargetAliases,
      .architecture_id = ROCJITSU_CODE_ARCH_CDNA3,
      .gpu_targets = kTargetGpuTargets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::cdna3
