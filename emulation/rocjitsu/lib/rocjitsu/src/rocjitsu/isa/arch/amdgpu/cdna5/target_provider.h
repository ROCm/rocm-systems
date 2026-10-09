// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_TARGET_PROVIDER_H_

#include "rocjitsu/isa/arch/amdgpu/cdna5/target_descriptor.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::cdna5 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();
std::unique_ptr<rocjitsu::Decoder> create_target_decoder(const IsaGpuTargetDescription &gpu_target);

/// Full execution alternative; do not combine it with the model-only provider
/// in the same registry.
inline constexpr IsaTargetDescriptor kTargetDescriptor = make_target_descriptor(
    kGpuTargets, true, static_cast<IsaTargetDescriptor::DecoderFactory>(&create_target_decoder),
    static_cast<IsaTargetDescriptor::VariantDecoderFactory>(&create_target_decoder));

} // namespace rocjitsu::cdna5

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::cdna5::kTargetDescriptor)
#endif
