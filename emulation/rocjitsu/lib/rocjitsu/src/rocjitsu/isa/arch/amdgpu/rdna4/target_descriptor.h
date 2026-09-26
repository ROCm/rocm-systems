// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna4 {

inline constexpr std::array<std::string_view, 2> kTargetAliases{"gfx1200", "gfx1201"};
inline constexpr std::array<IsaGpuTargetDescription, 2> kTargetGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX1200, "gfx1200", EF_AMDGPU_MACH_AMDGCN_GFX1200},
    {ROCJITSU_CODE_TARGET_GFX1201, "gfx1201", EF_AMDGPU_MACH_AMDGCN_GFX1201},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "rdna4",
      .aliases = kTargetAliases,
      .architecture_id = ROCJITSU_CODE_ARCH_RDNA4,
      .gpu_targets = kTargetGpuTargets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::rdna4
