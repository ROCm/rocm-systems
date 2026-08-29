// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_record_planning.h
/// @brief Record/Replay event planning contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::consan_moi_impl {

using consan_detail::MoiWorkitemOwnerDerivationPlan;

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &resources,
                                const MoiRecordEventEmissionPlan &plan, rj_code_arch_t arch);

[[nodiscard]] std::optional<MoiRecordEventEmissionPlan> resolve_moi_record_event_emission_plan(
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, uint16_t scratch_vgpr, rj_code_arch_t arch);

struct MoiPlannedRecordEvent : MoiPlannedProbeResources {
  MoiRecordEventEmissionPlan emission;
  std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
};

[[nodiscard]] std::optional<MoiPlannedRecordEvent>
plan_moi_record_event(std::span<const uint8_t> bytes, const ProgramInventory &inventory,
                      std::vector<std::string> &warnings, const ResolvedMoiScratchPlan &resources,
                      const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
                      const ConSanMoiOperatingPoint &base_point,
                      const ConSanMoiOperatingPoint &allocation, MoiSpillManagers &spill_managers,
                      rj_code_arch_t arch, std::string_view warning_context,
                      std::optional<uint32_t> active_private_segment_size = std::nullopt);

} // namespace rocjitsu::consan_moi_impl
