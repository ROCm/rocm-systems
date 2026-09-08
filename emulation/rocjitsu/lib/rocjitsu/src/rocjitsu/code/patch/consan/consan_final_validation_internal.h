// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_final_validation_internal.h
/// @brief Narrow common context shared by independent final validators.

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_validation_inventory.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"
#include "rocjitsu/isa/decoder.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::consan_validation_detail {

/// Immutable proof facets consumed by final validation.
struct ConSanFinalValidationInput {
  explicit ConSanFinalValidationInput(const ConSanTransformArtifacts &artifacts)
      : program_inventory(artifacts.program_inventory), coverage_ledger(artifacts.coverage_ledger),
        mutation(artifacts.mutation), resource_plans(artifacts.resource_plans),
        moi_operating_point(artifacts.moi_operating_point), patches(artifacts.patches),
        text_relocation(artifacts.text_relocation), replacement(artifacts.replacement) {}

  const ProgramInventory &program_inventory;
  const ConSanCoverageLedger &coverage_ledger;
  const ConSanMutationOutcome &mutation;
  std::span<const ConSanCandidateResourcePlan> resource_plans;
  const ConSanMoiOperatingPoint &moi_operating_point;
  std::span<const ConSanPatchInfo> patches;
  const std::optional<ConSanTextRelocationProof> &text_relocation;
  std::span<const uint8_t> replacement;

  [[nodiscard]] const ConSanObservationPlan &observation_plan() const {
    return coverage_ledger.observation_plan();
  }
};

/// Immutable executable bytes used by every independent proof pass.
struct ValidationText {
  std::span<const uint8_t> bytes;
  bool available = false;

  [[nodiscard]] static ValidationText from(const AmdGpuCodeObject &code_object) {
    if (code_object.text_sections().size() != 1u)
      return {};
    const Section *section = code_object.text_sections().front();
    return {
        .bytes = {reinterpret_cast<const uint8_t *>(section->data()), section->size()},
        .available = true,
    };
  }

  [[nodiscard]] std::optional<uint32_t> word(uint64_t offset) const {
    if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
      return std::nullopt;
    uint32_t result = 0;
    std::memcpy(&result, bytes.data() + offset, sizeof(result));
    return result;
  }

  [[nodiscard]] bool matches(uint64_t offset, std::span<const uint32_t> words) const {
    const uint64_t size = words.size() * sizeof(uint32_t);
    return offset <= bytes.size() && size <= bytes.size() - offset &&
           std::memcmp(bytes.data() + offset, words.data(), size) == 0;
  }
};

/// Universal parse and target context shared by independent validators.
struct FinalValidationEnvironment {
  const AmdGpuCodeObject &original;
  const AmdGpuCodeObject &replacement;
  ValidationText original_text;
  ValidationText replacement_text;
  const ConSanTargetProfile *target_profile = nullptr;
  std::unique_ptr<Decoder> decoder;

  FinalValidationEnvironment(const AmdGpuCodeObject &original, const AmdGpuCodeObject &replacement)
      : original(original), replacement(replacement), original_text(ValidationText::from(original)),
        replacement_text(ValidationText::from(replacement)),
        target_profile(consan_target_profile(replacement.target_id())),
        decoder(target_profile ? Decoder::create(target_profile->arch) : nullptr) {}

  [[nodiscard]] rj_code_arch_t arch() const {
    return target_profile ? target_profile->arch : ROCJITSU_CODE_ARCH_INVALID;
  }

  [[nodiscard]] bool require_target(std::vector<std::string> &errors,
                                    std::string_view profile_error,
                                    std::string_view decoder_error = {}) const {
    if (target_profile == nullptr) {
      errors.emplace_back(profile_error);
      return false;
    }
    if (!decoder_error.empty() && !decoder) {
      errors.emplace_back(decoder_error);
      return false;
    }
    return true;
  }
};

[[nodiscard]] inline bool has_consan_patch_phase(const ConSanFinalValidationInput &result,
                                                 ConSanPatchPhase phase) {
  return std::ranges::find(result.patches, phase, &ConSanPatchInfo::phase) != result.patches.end();
}

void validate_supercollider_final_semantics(const FinalValidationEnvironment &environment,
                                            const ConSanFinalValidationInput &result,
                                            const ConSanPristineValidationInventory *pristine,
                                            std::vector<std::string> &errors);

void validate_inline_shadow_final_semantics(const FinalValidationEnvironment &environment,
                                            const ConSanFinalValidationInput &result,
                                            uint64_t expected_moi_report_dispatch_id,
                                            std::vector<std::string> &errors);

} // namespace rocjitsu::consan_validation_detail
