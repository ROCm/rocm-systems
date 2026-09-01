// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_probe_planning.h
/// @brief Shared guest-state preservation plans for MOI probes.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"

namespace rocjitsu::consan_moi_impl {

/// Complete placement-to-emission preservation handoff for one MOI probe.
struct MoiPlannedProbeResources {
  ResolvedMoiScratchPlan resources;
  std::optional<VgprSpillSequence> spill;
  std::optional<SgprSpillSequence> scalar_spill;
  std::optional<MoiPrivateEpochLayout> private_layout;
  uint32_t required_private_bytes = 0;
};

[[nodiscard]] std::optional<MoiPlannedProbeResources>
plan_moi_probe_resources(const ProgramInventory &inventory, ResolvedMoiScratchPlan resources,
                         const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
                         const ConSanMoiOperatingPoint &point,
                         const MoiObjectModeSemantics &mode_semantics,
                         MoiSpillManagers &spill_managers, rj_code_arch_t arch,
                         std::optional<MoiPrivateEpochLayout> private_layout,
                         bool scalar_spill_required, std::vector<std::string> &warnings,
                         std::optional<uint32_t> active_private_segment_size = std::nullopt);

void note_moi_probe_private_requirements(MoiDescriptorPrivateRequirements &requirements,
                                         const MoiPlannedProbeResources &probe);

void note_moi_probe_patch_info(ConSanPatchAbiEffects &effects,
                               const MoiPlannedProbeResources &probe);

} // namespace rocjitsu::consan_moi_impl
