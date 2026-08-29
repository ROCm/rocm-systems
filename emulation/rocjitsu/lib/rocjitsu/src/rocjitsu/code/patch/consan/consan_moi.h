// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi.h
/// @brief MOI instrumentation mode entry points for ConSan DBI patching.

#pragma once

#include "rocjitsu/code/patch/consan/consan_lowering_types.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_contract.h"

namespace rocjitsu {

struct ConSanLoweringObservation;

[[nodiscard]] ConSanTransformArtifacts
try_patch_consan_moi(ConSanTransformArtifacts result, const MoiOptions &options,
                     std::span<const uint8_t> code_object_bytes, rj_code_arch_t arch,
                     ConSanLoweringExecution *execution = nullptr);

/// Re-run only MOI planning, lowering, and validation from a semantic
/// inventory produced for the same bytes and engine. This is used after a
/// runtime-sized report buffer and, optionally, a live fault selection become
/// available.
[[nodiscard]] ConSanTransformArtifacts retry_patch_consan_moi_from_inventory(
    ConSanTransformArtifacts inventory, ConSanOptions bound_options,
    std::span<const uint8_t> code_object_bytes, ConSanLoweringExecution *execution = nullptr,
    const ConSanLoweringObservation *observation = nullptr);

} // namespace rocjitsu
