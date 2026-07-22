// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_RISC_V_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::risc_v {

inline constexpr std::array<std::string_view, 1> target_aliases{"rv64i"};
inline constexpr IsaTargetDescription target_description{
    .id = "risc-v",
    .aliases = target_aliases,
    .architecture_id = ROCJITSU_CODE_ARCH_RV64I,
    .supports_execution = true,
};

IsaTargetRegistryError register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::risc_v

#endif // ROCJITSU_ISA_ARCH_RISC_V_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::risc_v::register_target)
#endif
