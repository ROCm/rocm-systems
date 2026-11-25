// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna3 {

inline constexpr std::array<std::string_view, 1> kTargetAliases = {"gfx1100"};
inline constexpr std::array<IsaTargetBinding, 1> kModelGpuTargets = {{
    {.public_id = ROCJITSU_CODE_TARGET_GFX1100,
     .id = "gfx1100",
     .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
     .gfx_target_version = 110000},
}};
inline constexpr std::array<IsaTargetBinding, 1> kExecutionGpuTargets = {{
    {.public_id = ROCJITSU_CODE_TARGET_GFX1100,
     .id = "gfx1100",
     .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
     .gfx_target_version = 110000,
     .capabilities = {.execution_implemented = true}},
}};

constexpr IsaTargetDescriptor
make_target_descriptor(std::span<const IsaTargetBinding> bindings, bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "rdna3",
      .aliases = kTargetAliases,
      .family = rocjitsu::IsaFamily::Rdna3,
      .bindings = bindings,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::rdna3
