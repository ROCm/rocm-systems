// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::rdna3_5 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr IsaTargetDescriptor kTargetDescriptor{
    .id = "rdna3_5",
    .architecture_id = ROCJITSU_CODE_ARCH_RDNA3_5,
    .decoder_factory = &create_target_decoder,
    .supports_execution = true,
};

} // namespace rocjitsu::rdna3_5

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::rdna3_5::kTargetDescriptor)
#endif
