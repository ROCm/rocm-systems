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

/// Atomically stage descriptor mutations and inline programs against the
/// current composition image. Mode-local builders never publish partial bytes.
[[nodiscard]] bool
stage_consan_text_rewrites(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                           const ConSanDescriptorMutationBatch &descriptor_mutations,
                           const ConSanDescriptorMutationPolicy &descriptor_policy,
                           std::string_view subject, std::vector<ConSanTextFragment> fragments,
                           ConSanTransformArtifacts &result);

/// Commit all staged access programs through one object-wide relocation.
[[nodiscard]] bool finalize_consan_text_rewrites(std::span<const uint8_t> descriptor_image,
                                                 rj_code_arch_t arch, const ConSanOptions &options,
                                                 std::string_view subject,
                                                 ConSanTransformArtifacts &result);

} // namespace rocjitsu
