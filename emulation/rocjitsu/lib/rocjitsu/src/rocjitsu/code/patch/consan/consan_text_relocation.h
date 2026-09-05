// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_text_relocation.h
/// @brief Whole-text placement transaction for inline ConSan probe programs.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

/// Split an already-assembled semantic program around its single guest operation.
/// The returned fragment contains no entry or return routing.
[[nodiscard]] std::optional<ConSanTextFragment> make_consan_around_text_fragment(
    std::vector<uint32_t> words, uint32_t guest_offset, uint32_t guest_size,
    std::span<const ConSanProbeIntentId> intent_ids, ConSanRuntimeStaticMapping runtime_mapping,
    ConSanPatchInfo patch, std::vector<std::string> &errors, std::string_view subject,
    std::optional<uint32_t> emitted_guest_size = std::nullopt);

/// Atomically stage descriptor mutations and inline programs against the
/// current composition image. Mode-local builders never publish partial bytes.
[[nodiscard]] bool
stage_consan_text_rewrites(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                           const ConSanDescriptorMutationBatch &descriptor_mutations,
                           const ConSanDescriptorMutationPolicy &descriptor_policy,
                           std::string_view subject, std::vector<ConSanTextFragment> fragments,
                           ConSanTransformArtifacts &result);

/// Add position-independent fragments after a caller has staged its exact
/// descriptor mutations in `result.replacement`.
[[nodiscard]] bool stage_consan_text_fragments(std::vector<ConSanTextFragment> fragments,
                                               ConSanTransformArtifacts &result);

/// Append fragments to a private candidate transaction.  Final placement and
/// publication remain deferred until that candidate has been accepted.
void append_consan_text_fragments(std::vector<ConSanTextFragment> fragments,
                                  std::vector<ConSanTextFragment> &transaction);

/// Commit all staged access programs through one object-wide relocation.
[[nodiscard]] bool finalize_consan_text_rewrites(std::span<const uint8_t> descriptor_image,
                                                 rj_code_arch_t arch,
                                                 ConSanPatchedImageGrowthLimit growth_limit,
                                                 std::string_view subject,
                                                 ConSanTransformArtifacts &result);

} // namespace rocjitsu
