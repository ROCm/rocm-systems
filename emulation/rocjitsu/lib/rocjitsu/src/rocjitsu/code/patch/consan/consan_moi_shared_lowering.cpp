// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
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

namespace rocjitsu {

using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
namespace consan_moi_impl {
[[nodiscard]] std::optional<uint8_t> moi_descriptor_workitem_id_dimensions(const KD &descriptor) {
  const uint32_t encoded =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID);
  if (encoded > 2u)
    return std::nullopt;
  return static_cast<uint8_t>(encoded + 1u);
}

[[nodiscard]] std::optional<uint16_t>
common_moi_workitem_owner_shift(std::span<const uint8_t> image,
                                const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
                                std::vector<std::string> &warnings) {
  std::optional<uint16_t> common_shift;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    std::vector<std::string> errors;
    const auto shift = moi_descriptor_owner_shift(image, descriptor_offset, arch, errors);
    if (!shift) {
      warnings.insert(warnings.end(), errors.begin(), errors.end());
      return std::nullopt;
    }
    if (common_shift && *common_shift != *shift) {
      warnings.emplace_back("ConSan MOI shared private owner has incompatible owner wave sizes");
      return std::nullopt;
    }
    common_shift = shift;
  }
  if (!common_shift)
    warnings.emplace_back("ConSan MOI private owner requires an owning kernel descriptor");
  return common_shift;
}

[[nodiscard]] std::optional<MoiWorkitemOwnerDerivationPlan>
resolve_moi_private_workitem_owner(std::span<const uint8_t> image,
                                   const ResolvedMoiScratchPlan &resources,
                                   const ConSanMoiPrivateStateLayout &layout, rj_code_arch_t arch,
                                   std::vector<std::string> &warnings) {
  if (!layout.owner_offset) {
    warnings.emplace_back("ConSan MOI private owner requires an entry-captured owner slot");
    return std::nullopt;
  }
  const auto common_shift = common_moi_workitem_owner_shift(image, resources, arch, warnings);
  if (!common_shift)
    return std::nullopt;
  return MoiWorkitemOwnerDerivationPlan{.entry_workitem_x_private_offset = *layout.owner_offset,
                                        .wave_size_shift = *common_shift};
}

[[nodiscard]] std::optional<uint64_t>
common_moi_record_owner_descriptor(std::span<const uint8_t> image,
                                   const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
                                   std::vector<std::string> &warnings) {
  std::optional<uint16_t> common_shift;
  std::optional<ConSanMoiWorkgroupSources> common_workgroup_sources;
  std::optional<uint64_t> representative;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    std::vector<std::string> errors;
    const auto shift = moi_descriptor_owner_shift(image, descriptor_offset, arch, errors);
    const auto workgroup_sources =
        moi_descriptor_workgroup_sources(image, descriptor_offset, arch, errors);
    if (!shift || !workgroup_sources) {
      warnings.insert(warnings.end(), errors.begin(), errors.end());
      return std::nullopt;
    }
    const bool owner_shift_mismatch = common_shift && *common_shift != *shift;
    const bool workgroup_source_mismatch =
        common_workgroup_sources && *common_workgroup_sources != *workgroup_sources;
    if (owner_shift_mismatch || workgroup_source_mismatch) {
      warnings.emplace_back("ConSan MOI shared record owner has incompatible descriptor ABI "
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
    warnings.emplace_back("ConSan MOI shared record owner has no owning descriptor");
  return representative;
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupShadowLayout> common_moi_workgroup_shadow_layout(
    std::span<const uint8_t> image, const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
    const RuntimeCapabilities &capabilities, std::vector<std::string> &warnings) {
  std::optional<ConSanMoiWorkgroupShadowLayout> common;
  const uint32_t max_workgroup_lds_bytes = consan_moi_max_workgroup_lds_bytes(capabilities, arch);
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const auto descriptor = read_kernel_descriptor(image, descriptor_offset);
    if (!descriptor) {
      warnings.emplace_back("ConSan MOI workgroup shadow descriptor exceeds ELF bytes");
      return std::nullopt;
    }
    auto layout = plan_consan_moi_workgroup_shadow(descriptor->group_segment_fixed_size,
                                                   max_workgroup_lds_bytes);
    if (!layout) {
      warnings.emplace_back(
          "ConSan MOI workgroup shadow requires fixed LDS that fits with its exact mirror");
      return std::nullopt;
    }
    // Exact-byte provenance uses the canonical full-width cell on every
    // target. CDNA entry clearing completes before any instrumented access;
    // RDNA4 can instead use its packed first-use bitmap. If either full layout
    // does not fit, the caller uses the semantically equivalent external table.
    if (consan_uses_gfx12_encoding(arch)) {
      if (auto lazy_layout = plan_consan_moi_lazy_workgroup_shadow(
              descriptor->group_segment_fixed_size, max_workgroup_lds_bytes)) {
        layout = lazy_layout;
      }
    }
    const auto workitem_id_dimensions = moi_descriptor_workitem_id_dimensions(*descriptor);
    if (!workitem_id_dimensions) {
      warnings.emplace_back("ConSan MOI workgroup shadow has invalid workitem-ID dimensions");
      return std::nullopt;
    }
    layout->workitem_id_dimensions = *workitem_id_dimensions;
    if (common && (common->base != layout->base || common->size != layout->size ||
                   common->validity_base != layout->validity_base ||
                   common->validity_size != layout->validity_size ||
                   common->lazy_initialization != layout->lazy_initialization ||
                   common->workitem_id_dimensions != layout->workitem_id_dimensions)) {
      warnings.emplace_back(
          "ConSan MOI shared access text has incompatible owning-kernel LDS layouts or IDs");
      return std::nullopt;
    }
    common = layout;
  }
  if (!common)
    warnings.emplace_back("ConSan MOI workgroup shadow has no owning kernel descriptor");
  return common;
}

[[nodiscard]] std::optional<ConSanMoiPrivateStateLayout>
build_moi_private_state_layout(const ProgramInventory &program_inventory,
                               const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
                               std::vector<std::string> &warnings, MoiPrivateStateDemand demand) {
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan MOI private epoch requires an owning kernel descriptor");
    return std::nullopt;
  }
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ConSanKernelInfo *kernel = program_inventory.find_kernel_by_descriptor(descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan MOI private epoch references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false)) {
      warnings.emplace_back(
          "ConSan MOI private epoch does not support a dynamic-stack owning kernel");
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
  const std::optional<uint64_t> workgroup_key_offset =
      demand.workgroup_key ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (workgroup_key_offset)
    persistent_end += SpillManager::kSlotBytes;
  const std::optional<uint64_t> dispatch_id_offset =
      demand.dispatch_id ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (dispatch_id_offset)
    persistent_end += 2u * SpillManager::kSlotBytes;
  ConSanMoiPersistentWorkgroupPrivateOffsets exact_workgroup_offsets;
  std::optional<std::array<uint64_t, 4>> exact_workgroup_offset_values;
  if (demand.exact_workgroup) {
    const bool include_cluster_workgroup_id =
        consan_arch_has_cluster_facilities(arch) &&
        std::ranges::any_of(resources.owner_descriptor_file_offsets,
                            [&](uint64_t descriptor_offset) {
                              const ConSanKernelInfo *kernel =
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
  const auto private_limit = consan_address_free_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back(
        "ConSan MOI private epoch has no address-free scratch support for this architecture");
    return std::nullopt;
  }
  if (persistent_end > std::numeric_limits<uint32_t>::max()) {
    warnings.emplace_back("ConSan MOI private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  if (exact_workgroup_offset_values) {
    exact_workgroup_offsets = ConSanMoiPersistentWorkgroupPrivateOffsets{
        static_cast<uint32_t>((*exact_workgroup_offset_values)[0]),
        static_cast<uint32_t>((*exact_workgroup_offset_values)[1]),
        static_cast<uint32_t>((*exact_workgroup_offset_values)[2]),
        (*exact_workgroup_offset_values)[3] != 0u
            ? std::optional<uint32_t>(
                  static_cast<uint32_t>((*exact_workgroup_offset_values)[3]))
            : std::nullopt};
  }
  const auto ephemeral_base =
      consan_normalize_address_free_private_size(arch, static_cast<uint32_t>(persistent_end));
  if (!ephemeral_base) {
    warnings.emplace_back("ConSan MOI private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  return ConSanMoiPrivateStateLayout{
      .epoch_offset = static_cast<uint32_t>(epoch_offset),
      .owner_offset = owner_offset ? std::optional<uint32_t>(static_cast<uint32_t>(*owner_offset))
                                   : std::nullopt,
      .workgroup_key_offset =
          workgroup_key_offset
              ? std::optional<uint32_t>(static_cast<uint32_t>(*workgroup_key_offset))
              : std::nullopt,
      .dispatch_id_offset =
          dispatch_id_offset ? std::optional<uint32_t>(static_cast<uint32_t>(*dispatch_id_offset))
                             : std::nullopt,
      .exact_workgroup_offsets = exact_workgroup_offsets,
      .persistent_state_end = static_cast<uint32_t>(persistent_end),
      .ephemeral_base = *ephemeral_base};
}

std::optional<ConSanMoiPrivateStateLayout> MoiPrivateStateLayoutCache::resolve(
    std::optional<uint64_t> descriptor, const ProgramInventory &program_inventory,
    const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
    std::vector<std::string> &warnings, MoiPrivateStateDemand demand) {
  if (!descriptor)
    return build_moi_private_state_layout(program_inventory, resources, arch, warnings, demand);
  const uint8_t demand_key = static_cast<uint8_t>(demand.owner) |
                             static_cast<uint8_t>(demand.workgroup_key) << 1u |
                             static_cast<uint8_t>(demand.exact_workgroup) << 2u |
                             static_cast<uint8_t>(demand.dispatch_id) << 3u;
  auto [cached, inserted] = layouts_.try_emplace(std::pair{*descriptor, demand_key}, std::nullopt);
  if (inserted)
    cached->second =
        build_moi_private_state_layout(program_inventory, resources, arch, warnings, demand);
  return cached->second;
}

[[nodiscard]] bool append_sampled_private_owner_epoch_load(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes, uint64_t descriptor_file_offset,
    bool automatic_private_epoch, const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs,
    const ConSanMoiPrivateStateLayout &layout, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  if (!automatic_private_epoch || !owner_epoch_vgprs.owner || !owner_epoch_vgprs.epoch ||
      !layout.owner_offset) {
    errors.emplace_back("ConSan MOI sampled sync has an invalid private owner/epoch plan");
    return false;
  }
  const auto owner_shift = moi_descriptor_owner_shift(bytes, descriptor_file_offset, arch, errors);
  const auto load_epoch =
      instrumentation::build_private_load_b32(*owner_epoch_vgprs.epoch, layout.epoch_offset, arch);
  const auto load_owner =
      instrumentation::build_private_load_b32(*owner_epoch_vgprs.owner, *layout.owner_offset, arch);
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  const auto owner =
      owner_shift ? instrumentation::build_v_lshrrev_b32(*owner_epoch_vgprs.owner,
                                                         scalar_positive_inline_u32(*owner_shift),
                                                         *owner_epoch_vgprs.owner, arch)
                  : std::nullopt;
  if (!owner_shift || !load_epoch || !load_owner || !wait || !owner) {
    errors.emplace_back("ConSan MOI sampled sync could not load private owner/epoch state");
    return false;
  }
  words.insert(words.end(), load_epoch->begin(), load_epoch->end());
  words.insert(words.end(), load_owner->begin(), load_owner->end());
  words.push_back(*wait);
  words.push_back(*owner);
  return true;
}

[[nodiscard]] std::optional<VgprSpillSequence> build_moi_spill_sequence(
    const ProgramInventory &program_inventory, const ResolvedMoiScratchPlan &resources,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, const MoiObjectModeSemantics &mode_semantics,
    MoiSpillManagers &managers, rj_code_arch_t arch, std::vector<std::string> &warnings,
    std::optional<uint32_t> private_layout_base) {
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan MOI spill requires an owning kernel descriptor");
    return std::nullopt;
  }
  const auto private_limit = consan_address_free_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back(
        "ConSan MOI spill has no address-free scratch support for this architecture");
    return std::nullopt;
  }

  bool has_dynamic_owner = false;
  bool has_fixed_owner = false;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ConSanKernelInfo *kernel = program_inventory.find_kernel_by_descriptor(descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan MOI spill references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false))
      has_dynamic_owner = true;
    else
      has_fixed_owner = true;
  }
  const bool needs_dynamic_scalar_frame = has_dynamic_owner && (point.has_moi_scalar_spill());
  if (resources.source != ConSanRegisterAllocationSource::SpillRequired &&
      !needs_dynamic_scalar_frame) {
    return std::nullopt;
  }

  const bool fixed_lane_scalar_reservoir =
      !has_dynamic_owner && point.moi_exec_save_sgpr && (point.has_moi_scalar_spill());
  const uint16_t spill_vgpr_count =
      static_cast<uint16_t>(resources.count + static_cast<uint16_t>(fixed_lane_scalar_reservoir));
  if (spill_vgpr_count < resources.count ||
      static_cast<uint32_t>(resources.base) + spill_vgpr_count > REGISTER_SET_MAX_VGPRS ||
      (fixed_lane_scalar_reservoir &&
       resources.required_vgpr_count < static_cast<uint32_t>(resources.base) + spill_vgpr_count)) {
    warnings.emplace_back("ConSan MOI scalar reservoir is outside its planned VGPR allocation");
    return std::nullopt;
  }

  if (has_dynamic_owner) {
    if (has_fixed_owner) {
      warnings.emplace_back(
          "ConSan MOI spill cannot share one frame recipe across dynamic and fixed owners");
      return std::nullopt;
    }
    if (!plan_moi_dynamic_stack_spill(request.moi_engine, arch).backend_supported) {
      warnings.emplace_back(
          "ConSan MOI dynamic-stack spill has no backend for this target and engine");
      return std::nullopt;
    }
    if (!point.moi_dynamic_stack_spill) {
      warnings.emplace_back(
          "ConSan MOI dynamic-stack spill was not included in scalar-state planning");
      return std::nullopt;
    }
    if (!point.moi_exec_save_sgpr) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no EXEC-save scalar window");
      return std::nullopt;
    }
    const bool spill_backed_scalar_window = point.has_moi_scalar_spill();
    const MoiScalarAbiPlan scalar_abi =
        plan_moi_scalar_abi(request.moi_engine, moi_scalar_routing_state(point));
    if (!scalar_abi.special_state || (spill_backed_scalar_window && !scalar_abi.indirect_jump)) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no SCC-save register");
      return std::nullopt;
    }
    const uint16_t saved_scc_sgpr = spill_backed_scalar_window
                                        ? scalar_abi.indirect_jump->scc_save_sgpr
                                        : scalar_abi.special_state->scc_save_sgpr;
    const auto saved_frame_offset = moi_dynamic_stack_frame_save_sgpr_offset(request.moi_engine);
    if (!spill_backed_scalar_window &&
        (!saved_frame_offset ||
         *saved_frame_offset >=
             moi_exec_save_sgpr_count(
                 resolve_moi_exec_save_requirement(request, bound_resources, point, mode_semantics),
                 arch))) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no reserved frame-base save slot");
      return std::nullopt;
    }
    const uint16_t saved_frame_base_sgpr =
        spill_backed_scalar_window
            ? scalar_abi.indirect_jump->pc_sgpr
            : static_cast<uint16_t>(*point.moi_exec_save_sgpr + *saved_frame_offset);
    const uint32_t additional_frame_bytes =
        spill_backed_scalar_window ? static_cast<uint32_t>(moi_exec_save_sgpr_count(
                                         resolve_moi_exec_save_requirement(request, bound_resources,
                                                                           point, mode_semantics),
                                         arch)) *
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
            ? consan_normalize_address_free_private_size(
                  arch, static_cast<uint32_t>(requested_private_bytes))
            : std::nullopt;
    if (!spill || !required_private_bytes) {
      warnings.emplace_back("ConSan MOI could not encode the dynamic-stack VGPR spill frame");
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
      warnings.emplace_back("ConSan MOI could not encode the shared-owner VGPR spill window");
    return spill;
  }

  const uint64_t descriptor_offset = resources.owner_descriptor_file_offsets.front();
  auto [it, inserted] = managers.try_emplace(
      descriptor_offset, private_layout_base.value_or(resources.original_private_segment_size),
      *private_limit);
  (void)inserted;
  auto spill = build_vgpr_spill_sequence(it->second, resources.base, spill_vgpr_count, arch);
  if (!spill)
    warnings.emplace_back("ConSan MOI could not encode or reserve the planned VGPR spill window");
  return spill;
}

namespace {

const MoiDescriptorLdsRequirements kNoMoiLdsRequirements;

[[nodiscard]] ConSanDescriptorMutationPolicy
moi_descriptor_policy(const RuntimeCapabilities *capabilities, rj_code_arch_t arch) {
  ConSanDescriptorMutationPolicy policy{
      .maximum_ordinary_vgpr_count = kMaxVgprs,
      .inventory_proves_empty_accumulator_bank = true,
      .maximum_group_segment_bytes = std::nullopt,
  };
  if (capabilities != nullptr) {
    policy.maximum_group_segment_bytes = consan_moi_max_workgroup_lds_bytes(*capabilities, arch);
  }
  return policy;
}

} // namespace

bool apply_moi_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &active_code_object,
    const ProgramInventory &program_inventory, const MoiDescriptorVgprRequirements &vgprs,
    const MoiDescriptorSgprRequirements &sgprs,
    const MoiDescriptorPrivateRequirements &private_segment_bytes,
    const MoiDescriptorLdsRequirements *group_segment_bytes,
    const RuntimeCapabilities *capabilities, rj_code_arch_t arch, std::string_view subject,
    std::vector<std::string> &errors) {
  return apply_consan_descriptor_mutations_to_patcher(
      patcher, program_inventory, active_code_object,
      {vgprs, sgprs, private_segment_bytes,
       group_segment_bytes == nullptr ? kNoMoiLdsRequirements : *group_segment_bytes},
      moi_descriptor_policy(capabilities, arch), arch, subject, errors);
}

bool apply_moi_descriptor_requirements(
    std::vector<uint8_t> &image, const ProgramInventory &program_inventory,
    const MoiDescriptorVgprRequirements &vgprs, const MoiDescriptorSgprRequirements &sgprs,
    const MoiDescriptorPrivateRequirements &private_segment_bytes,
    const MoiDescriptorLdsRequirements *group_segment_bytes,
    const RuntimeCapabilities *capabilities, rj_code_arch_t arch, std::string_view subject,
    std::vector<std::string> &errors) {
  return apply_consan_descriptor_mutations_to_bytes(
      image, program_inventory,
      {vgprs, sgprs, private_segment_bytes,
       group_segment_bytes == nullptr ? kNoMoiLdsRequirements : *group_segment_bytes},
      moi_descriptor_policy(capabilities, arch), arch, subject, errors);
}


[[nodiscard]] bool append_inline_shadow_owner_field(
    std::vector<uint32_t> &words, const MoiInlineShadowOwnerFieldPlan &plan, uint16_t low_vgpr,
    uint16_t tmp_vgpr, uint16_t owner_backup_vgpr, rj_code_arch_t arch,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    std::vector<std::string> &errors) {
  if (plan.owner_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *plan.owner_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (plan.persistent_owner_sgpr) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *plan.persistent_owner_sgpr, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (!plan.automatic_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no persistent owner representation");
    return false;
  }

  switch (plan.private_owner_source) {
  case ConSanMoiOwnerSource::Automatic:
    errors.emplace_back("ConSan MOI inline-shadow probe has an unresolved automatic owner source");
    return false;
  case ConSanMoiOwnerSource::WorkitemId: {
    if (!owner_derivation || !owner_derivation->entry_workitem_x_private_offset) {
      errors.emplace_back(
          "ConSan MOI private owner has no persistent descriptor-compatible workitem-id state");
      return false;
    }
    const auto owner = build_moi_workitem_owner_derivation(*owner_derivation, tmp_vgpr, arch,
                                                           "inline-shadow probe", errors);
    if (!owner)
      return false;
    words.insert(words.end(), owner->words.begin(), owner->words.end());
    break;
  }
  case ConSanMoiOwnerSource::HwId: {
    if (!plan.resident_wave_owner_sgpr) {
      errors.emplace_back("ConSan MOI private owner hw_id source requires "
                          "RJ_CONSAN_MOI_OWNER_SGPR");
      return false;
    }
    if (plan.borrowed_resident_wave_owner_sgpr) {
      const auto save = instrumentation::build_v_writelane_b32(
          owner_backup_vgpr, *plan.resident_wave_owner_sgpr, 0u, arch);
      if (!save) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not save its borrowed owner scalar");
        return false;
      }
      words.insert(words.end(), save->begin(), save->end());
    }
    const ConSanTargetProfile *target = consan_target_profile(arch);
    const consan_detail::MoiResidentWaveOwnerRequest owner_request{
        .destination_sgpr = *plan.resident_wave_owner_sgpr,
        .one_based = true,
    };
    if (target == nullptr ||
        !consan_detail::append_moi_resident_wave_owner(words, owner_request, *target)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not encode its resident-wave owner");
      return false;
    }
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *plan.resident_wave_owner_sgpr, arch));
    if (plan.borrowed_resident_wave_owner_sgpr) {
      const auto restore = instrumentation::build_v_readlane_b32(*plan.resident_wave_owner_sgpr,
                                                                 owner_backup_vgpr, 0u, arch);
      const auto wait = instrumentation::build_valu_to_salu_dependency_wait(arch);
      if (!restore || !wait) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not restore its borrowed owner scalar");
        return false;
      }
      words.insert(words.end(), restore->begin(), restore->end());
      words.push_back(*wait);
    }
    break;
  }
  }

  return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                       consan_moi_exact_shadow::owner_shift,
                                       consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
}

[[nodiscard]] bool append_inline_shadow_epoch_field(std::vector<uint32_t> &words,
                                                    const MoiInlineShadowEpochFieldPlan &plan,
                                                    std::optional<uint32_t> private_epoch_offset,
                                                    uint16_t low_vgpr, uint16_t tmp_vgpr,
                                                    rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  if (plan.epoch_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *plan.epoch_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (plan.persistent_epoch_sgpr) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *plan.persistent_epoch_sgpr, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (!plan.automatic_private_epoch || !private_epoch_offset) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no persistent epoch representation");
    return false;
  }
  const auto load = instrumentation::build_private_load_b32(tmp_vgpr, *private_epoch_offset, arch);
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(load, wait)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not load private epoch state");
    return false;
  }
  return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                       consan_moi_exact_shadow::epoch_shift,
                                       consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
}

[[nodiscard]] bool
append_extract_exact_shadow_generation(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                       uint16_t packed_low_vgpr, uint16_t packed_high_vgpr,
                                       uint16_t temporary_vgpr, rj_code_arch_t arch) {
  constexpr uint32_t kLowBits = 32u - consan_moi_exact_shadow::generation_shift;
  constexpr uint32_t kHighBits = consan_moi_exact_shadow::generation_bits - kLowBits;
  // Read the high fragment first: the destination is intentionally allowed to
  // alias packed_high_vgpr in the compact diagnostic register assignment.
  const auto high = instrumentation::build_v_and_b32_literal(
      temporary_vgpr, (uint32_t{1} << kHighBits) - 1u, packed_high_vgpr, arch);
  const auto low = instrumentation::build_v_lshrrev_b32(
      destination_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::generation_shift),
      packed_low_vgpr, arch);
  const auto shift_high = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(kLowBits), temporary_vgpr, arch);
  const auto combine = instrumentation::build_v_add_u32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), temporary_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(high, low, shift_high, combine);
}

[[nodiscard]] bool
append_add_exact_shadow_generation(std::vector<uint32_t> &words, uint16_t packed_low_vgpr,
                                   uint16_t packed_high_vgpr, uint16_t generation_vgpr,
                                   uint16_t temporary_vgpr, rj_code_arch_t arch) {
  constexpr uint32_t kLowBits = 32u - consan_moi_exact_shadow::generation_shift;
  constexpr uint32_t kHighBits = consan_moi_exact_shadow::generation_bits - kLowBits;
  const auto low = instrumentation::build_v_and_b32_literal(
      temporary_vgpr, (uint32_t{1} << kLowBits) - 1u, generation_vgpr, arch);
  const auto shift_low = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::generation_shift),
      temporary_vgpr, arch);
  const auto add_low = instrumentation::build_v_add_u32(
      packed_low_vgpr, vector_source_vgpr(packed_low_vgpr), temporary_vgpr, arch);
  const auto high = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(kLowBits), generation_vgpr, arch);
  const auto mask_high = instrumentation::build_v_and_b32_literal(
      temporary_vgpr, (uint32_t{1} << kHighBits) - 1u, temporary_vgpr, arch);
  const auto add_high = instrumentation::build_v_add_u32(
      packed_high_vgpr, vector_source_vgpr(packed_high_vgpr), temporary_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(low, shift_low, add_low, high, mask_high, add_high);
}

[[nodiscard]] MoiWorkgroupKeyRegisterPlan
moi_workgroup_key_register_plan(const ConSanMoiOperatingPoint &point) {
  return MoiWorkgroupKeyRegisterPlan{
      .exec_save_sgpr = point.moi_exec_save_sgpr,
      .cached_key_vgpr = point.moi_workgroup_key_vgpr,
      .cached_key_sgpr = point.moi_persistent_sgprs.workgroup_key,
  };
}

/// Emit a call-return comparison whose expected PC is rebuilt from a
/// canonical get-PC sequence. Revision translation can then relocate the
/// expected caller independently of the dispatcher and its in-kernel entry
/// island. The target profile owns the choice of PC-delta representation; the
/// router owns only the semantic comparison.
[[nodiscard]] bool append_moi_call_return_match(std::vector<uint32_t> &words,
                                                uint64_t words_text_offset,
                                                uint64_t caller_return_text_offset,
                                                uint16_t pc_sgpr, uint16_t call_return_sgpr,
                                                rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return false;
  const uint64_t getpc_text_offset =
      words_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  words.push_back(build_s_getpc_b64(pc_sgpr, arch));
  const uint64_t pc_after_getpc = getpc_text_offset + sizeof(uint32_t);
  if (caller_return_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  const int64_t delta =
      static_cast<int64_t>(caller_return_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta, /*minimum_words=*/3u,
                               /*prefer_literal64=*/target->direct_call_form ==
                                   ConSanDirectCallForm::SCallI64))
    return false;
  const auto compare = instrumentation::build_s_cmp_eq_u32(call_return_sgpr, pc_sgpr, arch);
  if (!compare)
    return false;
  words.push_back(*compare);
  return true;
}

/// Emit the owner-local dense entry island, dispatcher, optional relocated
/// host entry, and original-site route anchors shared by all three access
/// engines.
///
/// `PlannedPatch` is deliberately structural: the engines retain distinct
/// evidence-body plans, but expose the same `candidate`, `resources`,
/// `placement`, and `entry_island_offset` facts to this routing mechanism.
/// The boolean parameter preserves the Record/Replay-versus-Sampled policy for
/// equal indirect-PC and call-return assignments on explicit-key targets.
/// Inline Shadow selects its distinct scalar ABI through the request and
/// owner-bound operating point; its
/// SCC-tagged explicit key and indirect-PC dependency wait remain explicit
/// branches of this shared routing mechanism. Evidence construction and
/// per-site resource admission remain outside this function.
} // namespace consan_moi_impl
} // namespace rocjitsu
