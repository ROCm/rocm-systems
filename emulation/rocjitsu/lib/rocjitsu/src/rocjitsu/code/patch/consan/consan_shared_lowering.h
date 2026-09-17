// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_shared_lowering.h
/// @brief Shared ConSan layout, descriptor mutation, spill, and emission contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_packed_fields.h"
#include "rocjitsu/code/patch/consan/consan_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_register_allocation.h"

namespace rocjitsu::consan::detail {

[[nodiscard]] DescriptorMutationPolicy
descriptor_mutation_policy(const RuntimeCapabilities *capabilities, rj_code_arch_t arch);

[[nodiscard]] std::optional<uint16_t>
common_workitem_owner_shift(std::span<const uint8_t> image, const ResolvedScratchPlan &resources,
                            rj_code_arch_t arch, std::vector<std::string> &warnings);

[[nodiscard]] std::optional<uint64_t>
common_record_owner_descriptor(std::span<const uint8_t> image, const ResolvedScratchPlan &resources,
                               rj_code_arch_t arch, std::vector<std::string> &warnings);

/// Persistent values required in one private entry-state layout.
struct PrivateStateDemand {
  bool owner = false;
  bool exact_workgroup = false;
  bool dispatch_id = false;

  bool operator==(const PrivateStateDemand &) const = default;
};

[[nodiscard]] std::optional<PrivateStateLayout>
build_private_state_layout(const ProgramInventory &program_inventory,
                           const ResolvedScratchPlan &resources, rj_code_arch_t arch,
                           std::vector<std::string> &warnings, PrivateStateDemand demand);

/// Reuse a descriptor-local layout, including a prior failed resolution.
/// Multi-owner sites omit the key and are resolved independently.
class PrivateStateLayoutCache {
public:
  [[nodiscard]] std::optional<PrivateStateLayout>
  resolve(std::optional<uint64_t> descriptor, const ProgramInventory &program_inventory,
          const ResolvedScratchPlan &resources, rj_code_arch_t arch,
          std::vector<std::string> &warnings, PrivateStateDemand demand);

private:
  std::map<std::pair<uint64_t, uint8_t>, std::optional<PrivateStateLayout>> layouts_;
};

[[nodiscard]] std::optional<VgprSpillSequence> build_spill_sequence(
    const ProgramInventory &program_inventory, const ResolvedScratchPlan &resources,
    const Request &request, const BoundRuntimeResources &bound_resources,
    const OperatingPoint &point, SpillManagers &managers, rj_code_arch_t arch,
    std::vector<std::string> &warnings, std::optional<uint32_t> private_layout_base = std::nullopt);

[[nodiscard]] bool apply_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &active_code_object,
    const ProgramInventory &program_inventory, const DescriptorVgprRequirements &vgprs,
    const DescriptorSgprRequirements &sgprs,
    const DescriptorPrivateRequirements &private_segment_bytes,
    const DescriptorLdsRequirements *group_segment_bytes, const RuntimeCapabilities *capabilities,
    rj_code_arch_t arch, std::string_view subject, std::vector<std::string> &errors);

[[nodiscard]] bool apply_descriptor_requirements(
    std::vector<uint8_t> &image, const ProgramInventory &program_inventory,
    const DescriptorVgprRequirements &vgprs, const DescriptorSgprRequirements &sgprs,
    const DescriptorPrivateRequirements &private_segment_bytes,
    const DescriptorLdsRequirements *group_segment_bytes, const RuntimeCapabilities *capabilities,
    rj_code_arch_t arch, std::string_view subject, std::vector<std::string> &errors);

} // namespace rocjitsu::consan::detail
