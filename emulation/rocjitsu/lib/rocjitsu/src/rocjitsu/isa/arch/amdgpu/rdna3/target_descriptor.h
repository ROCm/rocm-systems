// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna3 {

inline constexpr std::array<std::string_view, 1> kTargetAliases{"gfx1100"};
inline constexpr std::array<IsaGpuTargetDescription, 1> kTargetGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX1100, "gfx1100", EF_AMDGPU_MACH_AMDGCN_GFX1100},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "rdna3",
      .aliases = kTargetAliases,
      .architecture_id = ROCJITSU_CODE_ARCH_RDNA3,
      .gpu_targets = kTargetGpuTargets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::rdna3
