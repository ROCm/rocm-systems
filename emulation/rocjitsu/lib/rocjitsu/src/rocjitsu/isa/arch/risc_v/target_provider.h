// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_RISC_V_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::risc_v {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr std::array<std::string_view, 1> kTargetAliases{"rv64i"};
inline constexpr IsaTargetDescriptor kTargetDescriptor{
    .id = "risc-v",
    .aliases = kTargetAliases,
    .architecture_id = ROCJITSU_CODE_ARCH_RV64I,
    .decoder_factory = &create_target_decoder,
    .supports_execution = true,
};

} // namespace rocjitsu::risc_v

#endif // ROCJITSU_ISA_ARCH_RISC_V_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::risc_v::kTargetDescriptor)
#endif
