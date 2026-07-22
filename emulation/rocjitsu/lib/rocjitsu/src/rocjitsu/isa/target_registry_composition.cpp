// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_registry_composition.cpp
/// @brief Checked-in implementation shared by static registry compositions.

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#ifndef RJ_ISA_TARGET_HEADERS
#error "RJ_ISA_TARGET_HEADERS must name a generated provider-header list"
#endif

#include RJ_ISA_TARGET_HEADERS

namespace rocjitsu {

const IsaTargetRegistry &default_isa_target_registry() {
  static const IsaTargetRegistry registry = [] {
#define ROCJITSU_GET_ISA_TARGET_REGISTRATION(provider) provider,
    constexpr IsaTargetProvider providers[] = {
#include RJ_ISA_TARGET_HEADERS
    };
#undef ROCJITSU_GET_ISA_TARGET_REGISTRATION
    return make_isa_target_registry(providers);
  }();
  return registry;
}

std::unique_ptr<Decoder> Decoder::create(rj_code_arch_t arch) {
  return create(default_isa_target_registry(), arch);
}

} // namespace rocjitsu
