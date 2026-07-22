// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_MODEL_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_MODEL_TARGET_PROVIDER_H_

#include "rocjitsu/isa/arch/amdgpu/gfx1250/target_description.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::gfx1250 {

/// Model-only alternative; do not combine it with the full execution provider
/// in the same registry.
inline constexpr IsaTargetDescription model_target_description = make_target_description(false);

IsaTargetRegistryError register_model_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::gfx1250

#endif // ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_MODEL_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::gfx1250::register_model_target)
#endif
