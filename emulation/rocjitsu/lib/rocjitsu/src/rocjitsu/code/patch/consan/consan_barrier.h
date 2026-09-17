// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_barrier.h
/// @brief Compiled ConSan barrier lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_evidence_planning.h"
#include "rocjitsu/code/patch/consan/consan_register_allocation.h"

#include <string>
#include <vector>

namespace rocjitsu::consan::detail {

[[nodiscard]] std::vector<detail::BarrierEvidenceSitePlan>
build_barrier_evidence_site_plans(const ProgramInventory &inventory,
                                  const ObservationPlan &observation, ProbeIntentKind evidence_kind,
                                  std::vector<std::string> &errors);

} // namespace rocjitsu::consan::detail
