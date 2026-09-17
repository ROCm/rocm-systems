// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_target_ops.cpp
/// @brief Narrow registry for independent encoded-mutation validation.

#include "rocjitsu/code/patch/consan/targets/consan_validation_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"

namespace rocjitsu::consan::validation_target_detail {
[[nodiscard]] EncodedMutationValidation
validate_rdna4_cdna5_encoded_mutation(EncodedMutationKind kind, std::span<const uint8_t> before,
                                      std::span<const uint8_t> after);
[[nodiscard]] DescriptorResourceDeltaValidation
validate_cdna3_cdna4_descriptor_resource_delta(const TargetProfile &target,
                                               const DescriptorResourceDeltaInput &input);
} // namespace rocjitsu::consan::validation_target_detail

namespace rocjitsu::consan {
EncodedMutationValidation validate_encoded_mutation(rj_code_arch_t arch, EncodedMutationKind kind,
                                                    std::span<const uint8_t> before,
                                                    std::span<const uint8_t> after) {
  if (arch_is_rdna4_or_cdna5(arch))
    return validation_target_detail::validate_rdna4_cdna5_encoded_mutation(kind, before, after);
  return EncodedMutationValidation::UnsupportedInstructionEncoding;
}

DescriptorResourceDeltaValidation
validate_descriptor_resource_delta(rj_code_arch_t arch, const DescriptorResourceDeltaInput &input) {
  const TargetProfile *target = target_profile(arch);
  if (target && arch_is_cdna3_or_cdna4(arch))
    return validation_target_detail::validate_cdna3_cdna4_descriptor_resource_delta(*target, input);
  return {.valid = true, .normalized_rsrc3 = input.replacement_rsrc3};
}

} // namespace rocjitsu::consan
