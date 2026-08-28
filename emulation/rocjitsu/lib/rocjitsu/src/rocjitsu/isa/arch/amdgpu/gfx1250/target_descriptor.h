// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::gfx1250 {

inline constexpr std::array<IsaGpuTargetDescription, 1> kGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX1250, "gfx1250", EF_AMDGPU_MACH_AMDGCN_GFX1250},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "gfx1250",
      .architecture_id = ROCJITSU_CODE_ARCH_GFX1250,
      .gpu_targets = kGpuTargets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::gfx1250
