// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_TARGET_PROVIDER_H_

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna3 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr std::array<std::string_view, 1> kTargetAliases{"gfx1100"};
inline constexpr std::array<IsaGpuTargetDescription, 1> kTargetGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX1100, "gfx1100", EF_AMDGPU_MACH_AMDGCN_GFX1100},
}};
inline constexpr IsaTargetDescriptor kTargetDescriptor{
    .id = "rdna3",
    .aliases = kTargetAliases,
    .architecture_id = ROCJITSU_CODE_ARCH_RDNA3,
    .gpu_targets = kTargetGpuTargets,
    .decoder_factory = &create_target_decoder,
    .supports_execution = true,
};

} // namespace rocjitsu::rdna3

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::rdna3::kTargetDescriptor)
#endif
