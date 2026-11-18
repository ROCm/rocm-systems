// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_TARGET_PROVIDER_H_

#include "rocjitsu/isa/arch/amdgpu/cdna4/target_descriptor.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::cdna4 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr IsaTargetDescriptor kTargetDescriptor =
    make_target_descriptor(kExecutionGpuTargets, true, &create_target_decoder);

} // namespace rocjitsu::cdna4

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::cdna4::kTargetDescriptor)
#endif
