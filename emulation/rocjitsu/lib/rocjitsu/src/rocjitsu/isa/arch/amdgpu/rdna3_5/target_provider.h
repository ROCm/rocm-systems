// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_TARGET_PROVIDER_H_

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna3_5 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr std::array<std::string_view, 2> kTargetAliases{"gfx1150", "gfx1151"};
inline constexpr std::array<IsaGpuTargetDescription, 2> kTargetGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX1150, "gfx1150", EF_AMDGPU_MACH_AMDGCN_GFX1150},
    {ROCJITSU_CODE_TARGET_GFX1151, "gfx1151", EF_AMDGPU_MACH_AMDGCN_GFX1151},
}};
inline constexpr IsaTargetDescriptor kTargetDescriptor{
    .id = "rdna3_5",
    .aliases = kTargetAliases,
    .architecture_id = ROCJITSU_CODE_ARCH_RDNA3_5,
    .gpu_targets = kTargetGpuTargets,
    .decoder_factory = &create_target_decoder,
    .supports_execution = true,
};

} // namespace rocjitsu::rdna3_5

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::rdna3_5::kTargetDescriptor)
#endif
