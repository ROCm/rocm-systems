// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_final_validation.h
/// @brief Private final-validation boundary for ConSan lowering.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <span>

namespace rocjitsu {

struct ConSanLoweringExecution;

/// Apply the opt-in fail-closed abort rewrite for unmatched barrier waits.
/// This remains adjacent to final proof because its patch records are
/// immediately validated by the same transaction.
void try_apply_unmatched_barrier_wait_abort(std::span<const uint8_t> original_bytes,
                                            const ConSanOptions &options,
                                            ConSanTransformArtifacts &result);

/// Convert a staged transformation into its terminal result after independent
/// structural, semantic, resource, ABI, and mutation proof validation.
[[nodiscard]] ConSanTransformArtifacts
finalize_consan_result(ConSanTransformArtifacts result, std::span<const uint8_t> original_bytes,
                       uint64_t expected_moi_report_dispatch_id = 0,
                       bool validating_carried_composite_stage = false,
                       const ConSanPerturbationPlanningState *carried_perturbation = nullptr,
                       ConSanLoweringExecution *execution = nullptr);

} // namespace rocjitsu
