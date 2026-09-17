// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_shared_lowering.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::consan {

using detail::WorkitemOwnerDerivationPlan;
namespace detail {
[[nodiscard]] std::optional<uint8_t> descriptor_workitem_id_dimensions(const KD &descriptor) {
  const uint32_t encoded =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID);
  if (encoded > 2u)
    return std::nullopt;
  return static_cast<uint8_t>(encoded + 1u);
}

[[nodiscard]] std::optional<uint16_t>
common_workitem_owner_shift(std::span<const uint8_t> image, const ResolvedScratchPlan &resources,
                            rj_code_arch_t arch, std::vector<std::string> &warnings) {
  std::optional<uint16_t> common_shift;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    std::vector<std::string> errors;
    const auto shift = descriptor_owner_shift(image, descriptor_offset, arch, errors);
    if (!shift) {
      warnings.insert(warnings.end(), errors.begin(), errors.end());
      return std::nullopt;
    }
    if (common_shift && *common_shift != *shift) {
      warnings.emplace_back("ConSan shared private owner has incompatible owner wave sizes");
      return std::nullopt;
    }
    common_shift = shift;
  }
  if (!common_shift)
    warnings.emplace_back("ConSan private owner requires an owning kernel descriptor");
  return common_shift;
}

[[nodiscard]] std::optional<WorkitemOwnerDerivationPlan>
resolve_private_workitem_owner(std::span<const uint8_t> image, const ResolvedScratchPlan &resources,
                               const PrivateStateLayout &layout, rj_code_arch_t arch,
                               std::vector<std::string> &warnings) {
  if (!layout.owner_offset) {
    warnings.emplace_back("ConSan private owner requires an entry-captured owner slot");
    return std::nullopt;
  }
  const auto common_shift = common_workitem_owner_shift(image, resources, arch, warnings);
  if (!common_shift)
    return std::nullopt;
  return WorkitemOwnerDerivationPlan{.entry_workitem_x_private_offset = *layout.owner_offset,
                                     .wave_size_shift = *common_shift};
}

[[nodiscard]] std::optional<uint64_t>
common_record_owner_descriptor(std::span<const uint8_t> image, const ResolvedScratchPlan &resources,
                               rj_code_arch_t arch, std::vector<std::string> &warnings) {
  std::optional<uint16_t> common_shift;
  std::optional<WorkgroupSources> common_workgroup_sources;
  std::optional<uint64_t> representative;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    std::vector<std::string> errors;
    const auto shift = descriptor_owner_shift(image, descriptor_offset, arch, errors);
    const auto workgroup_sources =
        descriptor_workgroup_sources(image, descriptor_offset, arch, errors);
    if (!shift || !workgroup_sources) {
      warnings.insert(warnings.end(), errors.begin(), errors.end());
      return std::nullopt;
    }
    const bool owner_shift_mismatch = common_shift && *common_shift != *shift;
    const bool workgroup_source_mismatch =
        common_workgroup_sources && *common_workgroup_sources != *workgroup_sources;
    if (owner_shift_mismatch || workgroup_source_mismatch) {
      warnings.emplace_back("ConSan shared record owner has incompatible descriptor ABI "
                            "inputs: " +
                            std::string(owner_shift_mismatch && workgroup_source_mismatch
                                            ? "owner shift and workgroup sources"
                                        : owner_shift_mismatch ? "owner shift"
                                                               : "workgroup sources"));
      return std::nullopt;
    }
    common_shift = shift;
    common_workgroup_sources = workgroup_sources;
    if (!representative)
      representative = descriptor_offset;
  }
  if (!representative)
    warnings.emplace_back("ConSan shared record owner has no owning descriptor");
  return representative;
}

[[nodiscard]] std::optional<PrivateStateLayout>
build_private_state_layout(const ProgramInventory &program_inventory,
                           const ResolvedScratchPlan &resources, rj_code_arch_t arch,
                           std::vector<std::string> &warnings, PrivateStateDemand demand) {
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan private epoch requires an owning kernel descriptor");
    return std::nullopt;
  }
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ProgramContainer *kernel = program_inventory.find_kernel_by_descriptor(descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan private epoch references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false)) {
      warnings.emplace_back("ConSan private epoch does not support a dynamic-stack owning kernel");
      return std::nullopt;
    }
  }
  const uint64_t epoch_offset =
      util::align_up(resources.original_private_segment_size, SpillManager::kDbiZoneAlignment);
  uint64_t persistent_end = epoch_offset + SpillManager::kSlotBytes;
  const std::optional<uint64_t> owner_offset =
      demand.owner ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (owner_offset)
    persistent_end += SpillManager::kSlotBytes;
  const std::optional<uint64_t> dispatch_id_offset =
      demand.dispatch_id ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (dispatch_id_offset)
    persistent_end += 2u * SpillManager::kSlotBytes;
  PersistentWorkgroupPrivateOffsets exact_workgroup_offsets;
  std::optional<std::array<uint64_t, 4>> exact_workgroup_offset_values;
  if (demand.exact_workgroup) {
    const bool include_cluster_workgroup_id =
        arch_has_cluster_facilities(arch) &&
        std::ranges::any_of(resources.owner_descriptor_file_offsets,
                            [&](uint64_t descriptor_offset) {
                              const ProgramContainer *kernel =
                                  program_inventory.find_kernel_by_descriptor(descriptor_offset);
                              return kernel != nullptr && kernel->uses_cluster_workgroup_id;
                            });
    exact_workgroup_offset_values = std::array<uint64_t, 4>{
        persistent_end,
        persistent_end + SpillManager::kSlotBytes,
        persistent_end + 2u * SpillManager::kSlotBytes,
        include_cluster_workgroup_id ? persistent_end + 3u * SpillManager::kSlotBytes : 0u,
    };
    persistent_end += (include_cluster_workgroup_id ? 4u : 3u) * SpillManager::kSlotBytes;
  }
  const auto private_limit = address_free_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back(
        "ConSan private epoch has no address-free scratch support for this architecture");
    return std::nullopt;
  }
  if (persistent_end > std::numeric_limits<uint32_t>::max()) {
    warnings.emplace_back("ConSan private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  if (exact_workgroup_offset_values) {
    exact_workgroup_offsets = PersistentWorkgroupPrivateOffsets{
        static_cast<uint32_t>((*exact_workgroup_offset_values)[0]),
        static_cast<uint32_t>((*exact_workgroup_offset_values)[1]),
        static_cast<uint32_t>((*exact_workgroup_offset_values)[2]),
        (*exact_workgroup_offset_values)[3] != 0u
            ? std::optional<uint32_t>(static_cast<uint32_t>((*exact_workgroup_offset_values)[3]))
            : std::nullopt};
  }
  const auto ephemeral_base =
      normalize_address_free_private_size(arch, static_cast<uint32_t>(persistent_end));
  if (!ephemeral_base) {
    warnings.emplace_back("ConSan private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  return PrivateStateLayout{
      .epoch_offset = static_cast<uint32_t>(epoch_offset),
      .owner_offset = owner_offset ? std::optional<uint32_t>(static_cast<uint32_t>(*owner_offset))
                                   : std::nullopt,
      .dispatch_id_offset =
          dispatch_id_offset ? std::optional<uint32_t>(static_cast<uint32_t>(*dispatch_id_offset))
                             : std::nullopt,
      .exact_workgroup_offsets = exact_workgroup_offsets,
      .persistent_state_end = static_cast<uint32_t>(persistent_end),
      .ephemeral_base = *ephemeral_base};
}

std::optional<PrivateStateLayout>
PrivateStateLayoutCache::resolve(std::optional<uint64_t> descriptor,
                                 const ProgramInventory &program_inventory,
                                 const ResolvedScratchPlan &resources, rj_code_arch_t arch,
                                 std::vector<std::string> &warnings, PrivateStateDemand demand) {
  if (!descriptor)
    return build_private_state_layout(program_inventory, resources, arch, warnings, demand);
  const uint8_t demand_key = static_cast<uint8_t>(demand.owner) |
                             static_cast<uint8_t>(demand.exact_workgroup) << 1u |
                             static_cast<uint8_t>(demand.dispatch_id) << 2u;
  auto [cached, inserted] = layouts_.try_emplace(std::pair{*descriptor, demand_key}, std::nullopt);
  if (inserted)
    cached->second =
        build_private_state_layout(program_inventory, resources, arch, warnings, demand);
  return cached->second;
}

[[nodiscard]] std::optional<VgprSpillSequence> build_spill_sequence(
    const ProgramInventory &program_inventory, const ResolvedScratchPlan &resources,
    const Request &request, const BoundRuntimeResources &bound_resources,
    const OperatingPoint &point, SpillManagers &managers, rj_code_arch_t arch,
    std::vector<std::string> &warnings, std::optional<uint32_t> private_layout_base) {
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan spill requires an owning kernel descriptor");
    return std::nullopt;
  }
  const auto private_limit = address_free_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back("ConSan spill has no address-free scratch support for this architecture");
    return std::nullopt;
  }

  bool has_dynamic_owner = false;
  bool has_fixed_owner = false;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ProgramContainer *kernel = program_inventory.find_kernel_by_descriptor(descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan spill references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false))
      has_dynamic_owner = true;
    else
      has_fixed_owner = true;
  }
  const bool needs_dynamic_scalar_frame = has_dynamic_owner && (point.has_scalar_spill());
  if (resources.source != RegisterAllocationSource::SpillRequired && !needs_dynamic_scalar_frame) {
    return std::nullopt;
  }

  const bool fixed_lane_scalar_reservoir =
      !has_dynamic_owner && point.exec_save_sgpr && (point.has_scalar_spill());
  const uint16_t spill_vgpr_count =
      static_cast<uint16_t>(resources.count + static_cast<uint16_t>(fixed_lane_scalar_reservoir));
  if (spill_vgpr_count < resources.count ||
      static_cast<uint32_t>(resources.base) + spill_vgpr_count > REGISTER_SET_MAX_VGPRS ||
      (fixed_lane_scalar_reservoir &&
       resources.required_vgpr_count < static_cast<uint32_t>(resources.base) + spill_vgpr_count)) {
    warnings.emplace_back("ConSan scalar reservoir is outside its planned VGPR allocation");
    return std::nullopt;
  }

  if (has_dynamic_owner) {
    if (has_fixed_owner) {
      warnings.emplace_back(
          "ConSan spill cannot share one frame recipe across dynamic and fixed owners");
      return std::nullopt;
    }
    if (!is_capability_arch(arch)) {
      warnings.emplace_back("ConSan dynamic-stack spill has no backend for this target and mode");
      return std::nullopt;
    }
    if (!point.dynamic_stack_spill) {
      warnings.emplace_back("ConSan dynamic-stack spill was not included in scalar-state planning");
      return std::nullopt;
    }
    if (!point.exec_save_sgpr) {
      warnings.emplace_back("ConSan dynamic-stack spill has no EXEC-save scalar window");
      return std::nullopt;
    }
    const bool spill_backed_scalar_window = point.has_scalar_spill();
    const ScalarAbiPlan scalar_abi = plan_scalar_abi(scalar_preservation_state(point));
    if (!scalar_abi.special_state || (spill_backed_scalar_window && !point.scalar_spill_setup)) {
      warnings.emplace_back("ConSan dynamic-stack spill has no SCC-save register");
      return std::nullopt;
    }
    const uint16_t saved_scc_sgpr = spill_backed_scalar_window
                                        ? point.scalar_spill_setup->temporaries.scc_save_sgpr
                                        : scalar_abi.special_state->scc_save_sgpr;
    constexpr uint16_t saved_frame_offset = kDynamicStackFrameSaveSgprOffset;
    if (!spill_backed_scalar_window &&
        (saved_frame_offset >=
         exec_save_sgpr_count(resolve_exec_save_requirement(request, bound_resources, point),
                              arch))) {
      warnings.emplace_back("ConSan dynamic-stack spill has no reserved frame-base save slot");
      return std::nullopt;
    }
    const uint16_t saved_frame_base_sgpr =
        spill_backed_scalar_window
            ? point.scalar_spill_setup->temporaries.frame_base_sgpr
            : static_cast<uint16_t>(*point.exec_save_sgpr + saved_frame_offset);
    const uint32_t additional_frame_bytes =
        spill_backed_scalar_window
            ? static_cast<uint32_t>(exec_save_sgpr_count(
                  resolve_exec_save_requirement(request, bound_resources, point), arch)) *
                  SpillManager::kSlotBytes
            : 0u;
    auto spill = build_dynamic_stack_vgpr_spill_sequence(
        resources.base, resources.count, kDynamicStackTopSgpr, kDynamicFrameBaseSgpr,
        saved_frame_base_sgpr, saved_scc_sgpr, arch, additional_frame_bytes);
    const uint64_t requested_private_bytes =
        static_cast<uint64_t>(resources.original_private_segment_size) +
        static_cast<uint64_t>(resources.count) * SpillManager::kSlotBytes + additional_frame_bytes;
    const auto required_private_bytes =
        requested_private_bytes <= std::numeric_limits<uint32_t>::max()
            ? normalize_address_free_private_size(arch,
                                                  static_cast<uint32_t>(requested_private_bytes))
            : std::nullopt;
    if (!spill || !required_private_bytes) {
      warnings.emplace_back("ConSan could not encode the dynamic-stack VGPR spill frame");
      return std::nullopt;
    }
    // The offsets are frame-relative, but the loader still needs enough
    // backing scratch for the extra maximum stack depth.
    spill->total_private_bytes = *required_private_bytes;
    return spill;
  }

  if (resources.owner_descriptor_file_offsets.size() > 1) {
    // Shared text needs one immediate layout. Start above the caller's
    // persistent-state boundary when supplied, otherwise above the maximum
    // original private size. That keeps one save/fill sequence legal for
    // every owner without overlapping epoch, owner, or sequence state.
    SpillManager shared_manager(
        private_layout_base.value_or(resources.original_private_segment_size), *private_limit);
    auto spill = build_vgpr_spill_sequence(shared_manager, resources.base, spill_vgpr_count, arch);
    if (!spill)
      warnings.emplace_back("ConSan could not encode the shared-owner VGPR spill window");
    return spill;
  }

  const uint64_t descriptor_offset = resources.owner_descriptor_file_offsets.front();
  auto [it, inserted] = managers.try_emplace(
      descriptor_offset, private_layout_base.value_or(resources.original_private_segment_size),
      *private_limit);
  (void)inserted;
  auto spill = build_vgpr_spill_sequence(it->second, resources.base, spill_vgpr_count, arch);
  if (!spill)
    warnings.emplace_back("ConSan could not encode or reserve the planned VGPR spill window");
  return spill;
}

const DescriptorLdsRequirements kNoLdsRequirements;

[[nodiscard]] DescriptorMutationPolicy
descriptor_mutation_policy(const RuntimeCapabilities *capabilities, rj_code_arch_t arch) {
  DescriptorMutationPolicy policy{
      .maximum_ordinary_vgpr_count = kMaxVgprs,
      .inventory_proves_empty_accumulator_bank = true,
      .maximum_group_segment_bytes = std::nullopt,
  };
  if (capabilities != nullptr) {
    policy.maximum_group_segment_bytes = max_workgroup_lds_bytes(*capabilities, arch);
  }
  return policy;
}

bool apply_descriptor_requirements(CodeObjectPatcher &patcher,
                                   const AmdGpuCodeObject &active_code_object,
                                   const ProgramInventory &program_inventory,
                                   const DescriptorVgprRequirements &vgprs,
                                   const DescriptorSgprRequirements &sgprs,
                                   const DescriptorPrivateRequirements &private_segment_bytes,
                                   const DescriptorLdsRequirements *group_segment_bytes,
                                   const RuntimeCapabilities *capabilities, rj_code_arch_t arch,
                                   std::string_view subject, std::vector<std::string> &errors) {
  return apply_descriptor_mutations_to_patcher(
      patcher, program_inventory, active_code_object,
      {vgprs, sgprs, private_segment_bytes,
       group_segment_bytes == nullptr ? kNoLdsRequirements : *group_segment_bytes},
      descriptor_mutation_policy(capabilities, arch), arch, subject, errors);
}

bool apply_descriptor_requirements(std::vector<uint8_t> &image,
                                   const ProgramInventory &program_inventory,
                                   const DescriptorVgprRequirements &vgprs,
                                   const DescriptorSgprRequirements &sgprs,
                                   const DescriptorPrivateRequirements &private_segment_bytes,
                                   const DescriptorLdsRequirements *group_segment_bytes,
                                   const RuntimeCapabilities *capabilities, rj_code_arch_t arch,
                                   std::string_view subject, std::vector<std::string> &errors) {
  return apply_descriptor_mutations_to_bytes(
      image, program_inventory,
      {vgprs, sgprs, private_segment_bytes,
       group_segment_bytes == nullptr ? kNoLdsRequirements : *group_segment_bytes},
      descriptor_mutation_policy(capabilities, arch), arch, subject, errors);
}

} // namespace detail
} // namespace rocjitsu::consan
