// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::rdna2 {

inline constexpr std::array<std::string_view, 1> kTargetAliases{"gfx1030"};
inline constexpr std::array<IsaTargetBinding, 1> kModelTargets{{
    {ROCJITSU_CODE_TARGET_GFX1030, "gfx1030", EF_AMDGPU_MACH_AMDGCN_GFX1030, 100300},
}};
inline constexpr std::array<IsaTargetBinding, 1> kExecutionTargets{{
    {ROCJITSU_CODE_TARGET_GFX1030,
     "gfx1030",
     EF_AMDGPU_MACH_AMDGCN_GFX1030,
     100300,
     {.execution_implemented = true}},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "rdna2",
      .aliases = kTargetAliases,
      .family = rocjitsu::IsaFamily::Rdna2,
      .bindings = supports_execution ? kExecutionTargets : kModelTargets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::rdna2
