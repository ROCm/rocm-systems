// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA1_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA1_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::rdna1 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr IsaTargetDescriptor kTargetDescriptor{
    .id = "rdna1",
    .architecture_id = ROCJITSU_CODE_ARCH_RDNA1,
    .decoder_factory = &create_target_decoder,
    .supports_execution = true,
};

} // namespace rocjitsu::rdna1

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA1_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::rdna1::kTargetDescriptor)
#endif
