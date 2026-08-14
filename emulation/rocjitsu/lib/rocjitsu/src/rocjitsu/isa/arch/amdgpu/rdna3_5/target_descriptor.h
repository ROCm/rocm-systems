// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna3_5 {

inline constexpr std::array<std::string_view, 2> kTargetAliases{"gfx1150", "gfx1151"};
inline constexpr std::array<IsaGpuTargetDescription, 2> kTargetGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX1150, "gfx1150", EF_AMDGPU_MACH_AMDGCN_GFX1150},
    {ROCJITSU_CODE_TARGET_GFX1151, "gfx1151", EF_AMDGPU_MACH_AMDGCN_GFX1151},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "rdna3_5",
      .aliases = kTargetAliases,
      .architecture_id = ROCJITSU_CODE_ARCH_RDNA3_5,
      .gpu_targets = kTargetGpuTargets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::rdna3_5
