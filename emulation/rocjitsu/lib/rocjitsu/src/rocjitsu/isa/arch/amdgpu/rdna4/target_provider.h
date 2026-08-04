// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_TARGET_PROVIDER_H_

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna4 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr std::array<std::string_view, 2> kTargetAliases{"gfx1200", "gfx1201"};
inline constexpr std::array<IsaGpuTargetDescription, 2> kTargetGpuTargets{{
    {ROCJITSU_CODE_TARGET_GFX1200, "gfx1200", EF_AMDGPU_MACH_AMDGCN_GFX1200},
    {ROCJITSU_CODE_TARGET_GFX1201, "gfx1201", EF_AMDGPU_MACH_AMDGCN_GFX1201},
}};
inline constexpr IsaTargetDescriptor kTargetDescriptor{
    .id = "rdna4",
    .aliases = kTargetAliases,
    .architecture_id = ROCJITSU_CODE_ARCH_RDNA4,
    .gpu_targets = kTargetGpuTargets,
    .decoder_factory = &create_target_decoder,
    .supports_execution = true,
};

} // namespace rocjitsu::rdna4

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::rdna4::kTargetDescriptor)
#endif
