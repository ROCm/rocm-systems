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

namespace rocjitsu::consan {

/// Split an already-assembled semantic program around its single guest operation.
/// The returned fragment contains no entry or return routing.
[[nodiscard]] std::optional<TextFragment>
make_around_text_fragment(std::vector<uint32_t> words, uint32_t guest_offset, uint32_t guest_size,
                          std::span<const ProbeIntentId> intent_ids,
                          StaticAccessMappings runtime_mapping, PatchInfo patch,
                          std::vector<std::string> &errors, std::string_view subject,
                          std::optional<uint32_t> emitted_guest_size = std::nullopt,
                          bool replacement_preserves_source_span = false);

/// Atomically stage descriptor mutations and inline programs against the
/// current composition image. Mode-local builders never publish partial bytes.
[[nodiscard]] bool stage_text_rewrites(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                       const DescriptorMutationBatch &descriptor_mutations,
                                       const DescriptorMutationPolicy &descriptor_policy,
                                       std::string_view subject,
                                       std::vector<TextFragment> fragments,
                                       TransformArtifacts &result);

/// Add position-independent fragments after a caller has staged its exact
/// descriptor mutations in `result.replacement`.
[[nodiscard]] bool stage_text_fragments(std::vector<TextFragment> fragments,
                                        TransformArtifacts &result);

/// Append fragments to a private candidate transaction.  Final placement and
/// publication remain deferred until that candidate has been accepted.
void append_text_fragments(std::vector<TextFragment> fragments,
                           std::vector<TextFragment> &transaction);

/// Commit all staged access programs through one object-wide relocation.
[[nodiscard]] bool finalize_text_rewrites(std::span<const uint8_t> descriptor_image,
                                          rj_code_arch_t arch, PatchedImageGrowthLimit growth_limit,
                                          std::string_view subject, TransformArtifacts &result);

} // namespace rocjitsu::consan
