// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_target_ops.cpp
/// @brief Narrow registry for independent encoded-mutation validation.

#include "rocjitsu/code/patch/consan/consan_validation_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_validation_target_ops_internal.h"

namespace rocjitsu {
namespace {

using ValidationFunction = ConSanEncodedMutationValidation (*)(std::span<const uint8_t>,
                                                               std::span<const uint8_t>);

[[nodiscard]] ConSanEncodedMutationValidation dispatch_validation(rj_code_arch_t arch,
                                                                  ValidationFunction gfx12,
                                                                  std::span<const uint8_t> before,
                                                                  std::span<const uint8_t> after) {
  if (consan_uses_gfx12_encoding(arch))
    return gfx12(before, after);
  return ConSanEncodedMutationValidation::UnsupportedInstructionEncoding;
}

} // namespace

ConSanEncodedMutationValidation validate_consan_ordinary_global_address_mutation(
    rj_code_arch_t arch, std::span<const uint8_t> before, std::span<const uint8_t> after) {
  return dispatch_validation(
      arch, consan_validation_target_detail::validate_gfx12_ordinary_global_address_mutation,
      before, after);
}

ConSanEncodedMutationValidation
validate_consan_ordinary_global_scope_mutation(rj_code_arch_t arch, std::span<const uint8_t> before,
                                               std::span<const uint8_t> after) {
  return dispatch_validation(
      arch, consan_validation_target_detail::validate_gfx12_ordinary_global_scope_mutation, before,
      after);
}

ConSanEncodedMutationValidation
validate_consan_atomic_address_mutation(rj_code_arch_t arch, std::span<const uint8_t> before,
                                        std::span<const uint8_t> after) {
  return dispatch_validation(
      arch, consan_validation_target_detail::validate_gfx12_atomic_address_mutation, before, after);
}

ConSanEncodedMutationValidation
validate_consan_atomic_scope_mutation(rj_code_arch_t arch, std::span<const uint8_t> before,
                                      std::span<const uint8_t> after) {
  return dispatch_validation(
      arch, consan_validation_target_detail::validate_gfx12_atomic_scope_mutation, before, after);
}

} // namespace rocjitsu
