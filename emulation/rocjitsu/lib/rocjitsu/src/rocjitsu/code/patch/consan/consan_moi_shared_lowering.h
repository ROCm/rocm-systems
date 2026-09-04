// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_shared_lowering.h
/// @brief Shared MOI layout, descriptor mutation, spill, and emission contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] ConSanDescriptorMutationPolicy
moi_descriptor_mutation_policy(const RuntimeCapabilities *capabilities, rj_code_arch_t arch);

[[nodiscard]] std::optional<uint16_t>
common_moi_workitem_owner_shift(std::span<const uint8_t> image,
                                const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
                                std::vector<std::string> &warnings);

[[nodiscard]] std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan>
resolve_moi_private_workitem_owner(std::span<const uint8_t> image,
                                   const ResolvedMoiScratchPlan &resources,
                                   const ConSanMoiPrivateStateLayout &layout, rj_code_arch_t arch,
                                   std::vector<std::string> &warnings);

[[nodiscard]] std::optional<uint64_t>
common_moi_record_owner_descriptor(std::span<const uint8_t> image,
                                   const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
                                   std::vector<std::string> &warnings);

[[nodiscard]] std::optional<ConSanMoiWorkgroupShadowLayout> common_moi_workgroup_shadow_layout(
    std::span<const uint8_t> image, const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
    const RuntimeCapabilities &capabilities, std::vector<std::string> &warnings);

/// Mode-owned persistent values required in one private entry-state layout.
struct MoiPrivateStateDemand {
  bool owner = false;
  bool workgroup_key = false;
  bool exact_workgroup = false;
  bool dispatch_id = false;

  bool operator==(const MoiPrivateStateDemand &) const = default;
};

[[nodiscard]] std::optional<ConSanMoiPrivateStateLayout>
build_moi_private_state_layout(const ProgramInventory &program_inventory,
                               const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
                               std::vector<std::string> &warnings, MoiPrivateStateDemand demand);

/// Reuse a descriptor-local layout, including a prior failed resolution.
/// Multi-owner sites omit the key and are resolved independently.
class MoiPrivateStateLayoutCache {
public:
  [[nodiscard]] std::optional<ConSanMoiPrivateStateLayout>
  resolve(std::optional<uint64_t> descriptor, const ProgramInventory &program_inventory,
          const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
          std::vector<std::string> &warnings, MoiPrivateStateDemand demand);

private:
  std::map<std::pair<uint64_t, uint8_t>, std::optional<ConSanMoiPrivateStateLayout>> layouts_;
};

[[nodiscard]] std::optional<VgprSpillSequence> build_moi_spill_sequence(
    const ProgramInventory &program_inventory, const ResolvedMoiScratchPlan &resources,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, const MoiObjectModeSemantics &mode_semantics,
    MoiSpillManagers &managers, rj_code_arch_t arch, std::vector<std::string> &warnings,
    std::optional<uint32_t> private_layout_base = std::nullopt);

[[nodiscard]] bool apply_moi_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &active_code_object,
    const ProgramInventory &program_inventory, const MoiDescriptorVgprRequirements &vgprs,
    const MoiDescriptorSgprRequirements &sgprs,
    const MoiDescriptorPrivateRequirements &private_segment_bytes,
    const MoiDescriptorLdsRequirements *group_segment_bytes,
    const RuntimeCapabilities *capabilities, rj_code_arch_t arch, std::string_view subject,
    std::vector<std::string> &errors);

[[nodiscard]] bool apply_moi_descriptor_requirements(
    std::vector<uint8_t> &image, const ProgramInventory &program_inventory,
    const MoiDescriptorVgprRequirements &vgprs, const MoiDescriptorSgprRequirements &sgprs,
    const MoiDescriptorPrivateRequirements &private_segment_bytes,
    const MoiDescriptorLdsRequirements *group_segment_bytes,
    const RuntimeCapabilities *capabilities, rj_code_arch_t arch, std::string_view subject,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
