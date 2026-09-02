// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_target_ops.cpp
/// @brief Narrow registry for independent encoded-mutation validation.

#include "rocjitsu/code/patch/consan/consan_validation_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"

namespace rocjitsu::consan_validation_target_detail {
[[nodiscard]] ConSanEncodedMutationValidation
validate_gfx12_encoded_mutation(ConSanEncodedMutationKind kind, std::span<const uint8_t> before,
                                std::span<const uint8_t> after);
[[nodiscard]] ConSanDescriptorResourceDeltaValidation
validate_gfx9_cdna_descriptor_resource_delta(const ConSanTargetProfile &target,
                                             const ConSanDescriptorResourceDeltaInput &input);
[[nodiscard]] bool validate_gfx9_cdna_dependency(ConSanDependencyKind kind,
                                                 const Instruction &instruction);
[[nodiscard]] bool validate_gfx11_dependency(ConSanDependencyKind kind,
                                             const Instruction &instruction);
[[nodiscard]] bool validate_gfx12_dependency(ConSanDependencyKind kind,
                                             const Instruction &instruction);
} // namespace rocjitsu::consan_validation_target_detail

namespace rocjitsu {
ConSanEncodedMutationValidation validate_consan_encoded_mutation(rj_code_arch_t arch,
                                                                 ConSanEncodedMutationKind kind,
                                                                 std::span<const uint8_t> before,
                                                                 std::span<const uint8_t> after) {
  if (consan_uses_gfx12_encoding(arch))
    return consan_validation_target_detail::validate_gfx12_encoded_mutation(kind, before, after);
  return ConSanEncodedMutationValidation::UnsupportedInstructionEncoding;
}

ConSanDescriptorResourceDeltaValidation
validate_consan_descriptor_resource_delta(rj_code_arch_t arch,
                                          const ConSanDescriptorResourceDeltaInput &input) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target && consan_uses_gfx9_cdna_encoding(arch))
    return consan_validation_target_detail::validate_gfx9_cdna_descriptor_resource_delta(*target,
                                                                                         input);
  return {.valid = true, .normalized_rsrc3 = input.replacement_rsrc3};
}

bool validate_consan_dependency(rj_code_arch_t arch, ConSanDependencyKind kind,
                                const Instruction &instruction) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (!target)
    return false;
  switch (target->encoding_family) {
  case ConSanEncodingFamily::Gfx9Cdna3:
  case ConSanEncodingFamily::Gfx9Cdna4:
    return consan_validation_target_detail::validate_gfx9_cdna_dependency(kind, instruction);
  case ConSanEncodingFamily::Gfx11:
    return consan_validation_target_detail::validate_gfx11_dependency(kind, instruction);
  case ConSanEncodingFamily::Gfx12:
    return consan_validation_target_detail::validate_gfx12_dependency(kind, instruction);
  }
  return false;
}

} // namespace rocjitsu
