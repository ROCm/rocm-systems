// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_probe_planning.h
/// @brief Shared guest-state preservation plans for ConSan probes.

#pragma once

#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_register_allocation.h"

namespace rocjitsu::consan::detail {

/// Complete placement-to-emission preservation handoff for one ConSan probe.
struct PlannedProbeResources {
  ResolvedScratchPlan resources;
  std::optional<VgprSpillSequence> spill;
  std::optional<SgprSpillSequence> scalar_spill;
  std::optional<PrivateStateLayout> private_layout;
  uint32_t required_private_bytes = 0;
};

[[nodiscard]] std::optional<PlannedProbeResources>
plan_probe_resources(const ProgramInventory &inventory, ResolvedScratchPlan resources,
                     const Request &request, const BoundRuntimeResources &bound_resources,
                     const OperatingPoint &point, SpillManagers &spill_managers,
                     rj_code_arch_t arch, std::optional<PrivateStateLayout> private_layout,
                     bool scalar_spill_required, std::vector<std::string> &warnings,
                     std::optional<uint32_t> active_private_segment_size = std::nullopt);

void note_probe_private_requirements(DescriptorPrivateRequirements &requirements,
                                     const PlannedProbeResources &probe);

void note_probe_patch_info(PatchAbiEffects &effects, const PlannedProbeResources &probe);

} // namespace rocjitsu::consan::detail
