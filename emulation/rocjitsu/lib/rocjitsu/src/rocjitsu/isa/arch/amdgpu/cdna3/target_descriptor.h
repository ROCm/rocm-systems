// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna3 {

inline constexpr std::array<std::string_view, 1> kTargetAliases{"gfx942"};
inline constexpr std::array<IsaTargetBinding, 1> kModelGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX942, "gfx942", EF_AMDGPU_MACH_AMDGCN_GFX942},
}};
inline constexpr std::array<IsaTargetBinding, 1> kExecutionGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX942,
     "gfx942",
     EF_AMDGPU_MACH_AMDGCN_GFX942,
     0,
     {.execution_implemented = true}},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(std::span<const IsaTargetBinding> bindings, bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "cdna3",
      .aliases = kTargetAliases,
      .family = rocjitsu::IsaFamily::Cdna3,
      .bindings = bindings,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::cdna3
