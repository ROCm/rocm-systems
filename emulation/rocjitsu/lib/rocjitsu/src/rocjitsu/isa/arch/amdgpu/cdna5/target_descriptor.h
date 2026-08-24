// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/isa_features.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna5 {

inline constexpr std::array<std::string_view, 2> kTargetAliases{"gfx1250", "gfx1251"};
inline constexpr IsaGpuTargetDescription kGfx1250Target{
    ROCJITSU_CODE_TARGET_GFX1250,
    "gfx1250",
    EF_AMDGPU_MACH_AMDGCN_GFX1250,
    120500,
    {.instruction_features = kGfx1250IsaFeatures,
     .setreg_vgpr_msb_fixup = true,
     .execution_implemented = true}};
inline constexpr IsaGpuTargetDescription kGfx1251Target{
    ROCJITSU_CODE_TARGET_GFX1251,
    "gfx1251",
    EF_AMDGPU_MACH_AMDGCN_GFX1251,
    120501,
    {.instruction_features = kGfx1251IsaFeatures,
     .setreg_vgpr_msb_fixup = false,
     .execution_implemented = false}};
inline constexpr std::array<IsaGpuTargetDescription, 2> kGpuTargets{
    {kGfx1250Target, kGfx1251Target}};

constexpr IsaGpuTargetDescription without_execution(IsaGpuTargetDescription target) {
  target.capabilities.execution_implemented = false;
  return target;
}

inline constexpr std::array<IsaGpuTargetDescription, 2> kModelGpuTargets{
    {without_execution(kGfx1250Target), without_execution(kGfx1251Target)}};

constexpr IsaTargetDescriptor
make_target_descriptor(std::span<const IsaGpuTargetDescription> gpu_targets,
                       bool supports_execution, IsaTargetDescriptor::DecoderFactory decoder_factory,
                       IsaTargetDescriptor::VariantDecoderFactory variant_decoder_factory) {
  return {
      .id = "cdna5",
      .aliases = kTargetAliases,
      .architecture_id = ROCJITSU_CODE_ARCH_CDNA5,
      .gpu_targets = gpu_targets,
      .default_gpu_target = ROCJITSU_CODE_TARGET_GFX1250,
      .decoder_factory = decoder_factory,
      .variant_decoder_factory = variant_decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::cdna5
