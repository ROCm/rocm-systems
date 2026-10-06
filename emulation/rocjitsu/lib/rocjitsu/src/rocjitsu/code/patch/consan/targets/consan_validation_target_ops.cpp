// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_target_ops.cpp
/// @brief Narrow registry for independent encoded-mutation validation.

#include "rocjitsu/code/patch/consan/targets/consan_validation_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <cstring>

namespace rocjitsu::consan::validation_target_detail {
[[nodiscard]] EncodedMutationValidation
validate_rdna4_cdna5_encoded_mutation(EncodedMutationKind kind, std::span<const uint8_t> before,
                                      std::span<const uint8_t> after,
                                      bool allow_atomic_observation);
[[nodiscard]] DescriptorResourceDeltaValidation
validate_cdna3_cdna4_descriptor_resource_delta(const TargetProfile &target,
                                               const DescriptorResourceDeltaInput &input);
} // namespace rocjitsu::consan::validation_target_detail

namespace rocjitsu::consan {
bool validate_atomic_order_boundary_rewrite(rj_code_arch_t arch, std::span<const uint8_t> before,
                                            std::span<const uint8_t> after) {
  if (before.empty() || before.size() != after.size() || before.size() % sizeof(uint32_t) != 0)
    return false;
  const uint32_t nop = build_s_nop(0, arch);
  for (size_t offset = 0; offset < before.size(); offset += sizeof(uint32_t)) {
    uint32_t before_word = 0;
    uint32_t after_word = 0;
    std::memcpy(&before_word, before.data() + offset, sizeof(before_word));
    std::memcpy(&after_word, after.data() + offset, sizeof(after_word));
    if (before.size() == sizeof(uint32_t)) {
      const auto wait = classify_wait_instruction({}, before_word, arch);
      if (wait.drains_load) {
        const auto load_only = instrumentation::build_s_wait_global_load0(arch);
        return wait.drains_lds && load_only && after_word == *load_only;
      }
    }
    if (after_word != nop)
      return false;
  }
  return true;
}

EncodedMutationValidation validate_encoded_mutation(rj_code_arch_t arch, EncodedMutationKind kind,
                                                    std::span<const uint8_t> before,
                                                    std::span<const uint8_t> after,
                                                    bool allow_atomic_observation) {
  if (arch_is_rdna4_or_cdna5(arch))
    return validation_target_detail::validate_rdna4_cdna5_encoded_mutation(
        kind, before, after, allow_atomic_observation && arch == ROCJITSU_CODE_ARCH_RDNA4);
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
