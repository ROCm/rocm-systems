// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_PROVIDER_H_

#include "rocjitsu/isa/arch/amdgpu/gfx1250/target_descriptor.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::gfx1250 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

/// Full execution alternative; do not combine it with the model-only provider
/// in the same registry.
inline constexpr IsaTargetDescriptor kTargetDescriptor =
    make_target_descriptor(true, &create_target_decoder);

} // namespace rocjitsu::gfx1250

#endif // ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::gfx1250::kTargetDescriptor)
#endif
