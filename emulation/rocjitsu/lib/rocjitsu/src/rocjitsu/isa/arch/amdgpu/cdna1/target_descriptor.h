// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::cdna1 {

inline constexpr std::array<std::string_view, 1> kTargetAliases{"gfx908"};
inline constexpr std::array<IsaTargetBinding, 1> kModelTargets{{
    {ROCJITSU_CODE_TARGET_GFX908, "gfx908", EF_AMDGPU_MACH_AMDGCN_GFX908, 90008},
}};
inline constexpr std::array<IsaTargetBinding, 1> kExecutionTargets{{
    {ROCJITSU_CODE_TARGET_GFX908,
     "gfx908",
     EF_AMDGPU_MACH_AMDGCN_GFX908,
     90008,
     {.execution_implemented = true}},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "cdna1",
      .aliases = kTargetAliases,
      .family = rocjitsu::IsaFamily::Cdna1,
      .bindings = supports_execution ? kExecutionTargets : kModelTargets,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::cdna1
