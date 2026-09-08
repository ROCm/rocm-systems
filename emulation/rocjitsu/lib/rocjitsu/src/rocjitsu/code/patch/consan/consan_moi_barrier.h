// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_barrier.h
/// @brief Compiled MOI barrier lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

#include <string>
#include <vector>

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::vector<consan_detail::MoiBarrierEvidenceSitePlan>
build_moi_barrier_evidence_site_plans(const ProgramInventory &inventory,
                                      const ConSanObservationPlan &observation,
                                      ConSanProbeIntentKind evidence_kind,
                                      std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
