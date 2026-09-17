// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_instrumentation.h
/// @brief ConSan instrumentation mode entry points for ConSan DBI patching.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_lowering_types.h"

namespace rocjitsu::consan {

struct LoweringObservation;

/// Immutable semantic state retained while automatic runtime resources are
/// bound. Mutation plans, resource attempts, emitted patches, candidate bytes,
/// and operating-point state are deliberately excluded.
struct RetryInventory {
  ProgramInventory program_inventory;
  CoverageLedger coverage_ledger;
  std::vector<FaultSite> fault_sites;
  std::vector<BarrierMoveDestination> barrier_move_destinations;
};

[[nodiscard]] TransformArtifacts try_patch(TransformArtifacts result, const Options &options,
                                           const OperatingPoint &initial_operating_point,
                                           std::span<const uint8_t> code_object_bytes,
                                           rj_code_arch_t arch);

/// Re-run only ConSan planning, lowering, and validation from the immutable
/// semantic products prepared for the same bytes and mode. This is used
/// after a runtime-sized report buffer becomes available.
[[nodiscard]] TransformArtifacts
retry_patch_from_inventory(RetryInventory inventory, Options bound_options,
                           std::span<const uint8_t> code_object_bytes,
                           const LoweringObservation *observation = nullptr);

} // namespace rocjitsu::consan
