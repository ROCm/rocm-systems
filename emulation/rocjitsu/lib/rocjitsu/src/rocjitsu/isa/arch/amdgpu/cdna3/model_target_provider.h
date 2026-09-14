// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_MODEL_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_MODEL_TARGET_PROVIDER_H_

#include "rocjitsu/isa/arch/amdgpu/cdna3/target_descriptor.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::cdna3 {

std::unique_ptr<rocjitsu::Decoder> create_model_target_decoder();

inline constexpr IsaTargetDescriptor kModelTargetDescriptor =
    make_target_descriptor(kModelGpuTargets, false, &create_model_target_decoder);

} // namespace rocjitsu::cdna3

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_MODEL_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::cdna3::kModelTargetDescriptor)
#endif
