// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_text_relocation.h
/// @brief Whole-text placement transaction for inline ConSan probe programs.

#pragma once

#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

struct ConSanInlineTextRewrite {
  uint64_t source_offset = 0;
  std::vector<uint32_t> words;
};

struct ConSanRelocatedText {
  std::vector<uint8_t> image;
  std::vector<TranslatedTextPlacement> placements;
  uint64_t source_text_size = 0;
  uint64_t relocated_text_size = 0;
};

/// Atomically stage descriptor mutations and inline programs against the
/// current composition image. Mode-local builders never publish partial bytes.
[[nodiscard]] bool stage_consan_text_rewrites(
    const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
    const ConSanDescriptorMutationBatch &descriptor_mutations,
    const ConSanDescriptorMutationPolicy &descriptor_policy, std::string_view subject,
    std::vector<ConSanStagedTextRewrite> rewrites, ConSanTransformArtifacts &result);

/// Relocate the executable program once while expanding selected instructions
/// inline. The source text remains as an unreachable prefix so ConSan's
/// independent inventory keeps stable pristine coordinates.
[[nodiscard]] std::optional<ConSanRelocatedText> relocate_consan_text(
    std::span<const uint8_t> descriptor_patched_image, rj_code_arch_t arch,
    std::span<const ConSanInlineTextRewrite> rewrites,
    const ConSanPatchedImageGrowthLimit &growth_limit, const ConSanCodeObjectId &input_id,
    std::string_view operation, std::span<const SourceTextCodeRange> additional_code_ranges,
    std::vector<std::string> &errors,
    std::optional<ConSanTransformFailureCause> *failure_cause = nullptr);

/// Commit all staged access programs through one object-wide relocation.
[[nodiscard]] bool finalize_consan_text_rewrites(std::span<const uint8_t> descriptor_image,
                                                 rj_code_arch_t arch,
                                                 const ConSanOptions &options,
                                                 std::string_view subject,
                                                 ConSanTransformArtifacts &result);

} // namespace rocjitsu
