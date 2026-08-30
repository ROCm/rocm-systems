// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

using consan_detail::build_moi_relocated_guest_access_words;
using consan_detail::has_recent_saveexec;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_atomic_or_u32_literal;
using consan_moi_detail::append_dynamic_record_event_index_store;
using consan_moi_detail::append_dynamic_record_store_moi_report_dispatch_id_pair;
using consan_moi_detail::append_dynamic_record_store_u32_literal;
using consan_moi_detail::append_dynamic_record_store_u32_scalar_src;
using consan_moi_detail::append_dynamic_record_store_u32_vgpr;
using consan_moi_detail::append_dynamic_record_store_workgroup_source;
using consan_moi_detail::append_store_moi_report_dispatch_id_pair;
using consan_moi_detail::append_store_u32_literal;
using consan_moi_detail::ConSanMoiLiteralDispatchIdPolicy;
using consan_moi_detail::ConSanMoiRecordEmitter;
using consan_moi_detail::kAccessRecordLayout;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;
using consan_moi_detail::record_replay_requires_entry_workgroup_capture;
using consan_moi_detail::record_replay_uses_automatic_banked_capture;

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

[[nodiscard]] bool moi_record_uses_private_owner(const ConSanRequest &request,
                                                 const ConSanMoiOperatingPoint &point) {
  return moi_initializes_owner_epoch(request, point) && point.automatic_moi_private_epoch &&
         request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId && !point.moi_owner_vgpr &&
         !point.moi_persistent_sgprs.owner;
}

[[nodiscard]] std::optional<MoiWorkitemOwnerDerivationPlan> resolve_moi_private_workitem_owner(
    std::span<const uint8_t> image, const ResolvedMoiScratchPlan &resources,
    const MoiPrivateEpochLayout &layout, rj_code_arch_t arch, std::vector<std::string> &warnings) {
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
    // target. The former generation tag avoided relying on entry clearing, but
    // it occupies the field now carrying local byte provenance. CDNA entry
    // clearing completes before any instrumented access, so the full eager
    // mirror establishes the same validity invariant without that tag. RDNA4
    // can instead use its packed first-use bitmap. If either full layout does
    // not fit, the caller uses the semantically equivalent external table.
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
                   common->compact != layout->compact ||
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

[[nodiscard]] std::optional<MoiPrivateEpochLayout> build_moi_private_epoch_layout(
    const ProgramInventory &program_inventory, const ResolvedMoiScratchPlan &resources,
    rj_code_arch_t arch, std::vector<std::string> &warnings, bool include_owner,
    bool include_workgroup_key, bool include_record_replay_workgroup, bool include_dispatch_id) {
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
      include_owner ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (owner_offset)
    persistent_end += SpillManager::kSlotBytes;
  const std::optional<uint64_t> workgroup_key_offset =
      include_workgroup_key ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (workgroup_key_offset)
    persistent_end += SpillManager::kSlotBytes;
  const std::optional<uint64_t> dispatch_id_offset =
      include_dispatch_id ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (dispatch_id_offset)
    persistent_end += 2u * SpillManager::kSlotBytes;
  ConSanMoiPersistentWorkgroupPrivateOffsets record_replay_workgroup_offsets;
  std::optional<std::array<uint64_t, 4>> record_replay_workgroup_offset_values;
  if (include_record_replay_workgroup) {
    const bool include_cluster_workgroup_id =
        consan_arch_has_cluster_facilities(arch) &&
        std::ranges::any_of(resources.owner_descriptor_file_offsets,
                            [&](uint64_t descriptor_offset) {
                              const ConSanKernelInfo *kernel =
                                  program_inventory.find_kernel_by_descriptor(descriptor_offset);
                              return kernel != nullptr && kernel->uses_cluster_workgroup_id;
                            });
    record_replay_workgroup_offset_values = std::array<uint64_t, 4>{
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
  if (record_replay_workgroup_offset_values) {
    record_replay_workgroup_offsets = {
        .x = static_cast<uint32_t>((*record_replay_workgroup_offset_values)[0]),
        .y = static_cast<uint32_t>((*record_replay_workgroup_offset_values)[1]),
        .z = static_cast<uint32_t>((*record_replay_workgroup_offset_values)[2]),
        .cluster_workgroup_id = (*record_replay_workgroup_offset_values)[3] != 0u
                                    ? std::optional<uint32_t>(static_cast<uint32_t>(
                                          (*record_replay_workgroup_offset_values)[3]))
                                    : std::nullopt,
    };
  }
  const auto ephemeral_base =
      consan_normalize_address_free_private_size(arch, static_cast<uint32_t>(persistent_end));
  if (!ephemeral_base) {
    warnings.emplace_back("ConSan MOI private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  return MoiPrivateEpochLayout{
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
      .record_replay_workgroup_offsets = record_replay_workgroup_offsets,
      .persistent_state_end = static_cast<uint32_t>(persistent_end),
      .ephemeral_base = *ephemeral_base};
}

[[nodiscard]] std::optional<MoiPrivateEpochLayout> build_sampled_private_epoch_layout(
    const ProgramInventory &program_inventory, const ResolvedMoiScratchPlan &resources,
    const ConSanRequest &request, rj_code_arch_t arch, std::vector<std::string> &warnings) {
  return build_moi_private_epoch_layout(
      program_inventory, resources, arch, warnings,
      /*include_owner=*/request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
      /*include_workgroup_key=*/false,
      /*include_record_replay_workgroup=*/
      record_replay_requires_entry_workgroup_capture(request.moi_engine));
}

[[nodiscard]] bool append_sampled_private_owner_epoch_load(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes, uint64_t descriptor_file_offset,
    const ConSanMoiOperatingPoint &point, const MoiPrivateEpochLayout &layout, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  if (!point.automatic_moi_private_epoch || !point.moi_owner_vgpr || !point.moi_epoch_vgpr ||
      !layout.owner_offset) {
    errors.emplace_back("ConSan MOI sampled sync has an invalid private owner/epoch plan");
    return false;
  }
  const auto owner_shift = moi_descriptor_owner_shift(bytes, descriptor_file_offset, arch, errors);
  const auto load_epoch =
      instrumentation::build_private_load_b32(*point.moi_epoch_vgpr, layout.epoch_offset, arch);
  const auto load_owner =
      instrumentation::build_private_load_b32(*point.moi_owner_vgpr, *layout.owner_offset, arch);
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  const auto owner =
      owner_shift ? instrumentation::build_v_lshrrev_b32(*point.moi_owner_vgpr,
                                                         scalar_positive_inline_u32(*owner_shift),
                                                         *point.moi_owner_vgpr, arch)
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
    const ConSanMoiOperatingPoint &point, MoiSpillManagers &managers, rj_code_arch_t arch,
    std::vector<std::string> &warnings, std::optional<uint32_t> private_layout_base) {
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
  const bool needs_dynamic_scalar_frame =
      has_dynamic_owner &&
      (point.automatic_moi_inline_sgpr_spill || point.automatic_moi_record_replay_sgpr_spill);
  if (resources.source != ConSanRegisterAllocationSource::SpillRequired &&
      !needs_dynamic_scalar_frame) {
    return std::nullopt;
  }

  const bool fixed_lane_scalar_reservoir =
      !has_dynamic_owner && point.moi_exec_save_sgpr &&
      (point.automatic_moi_inline_sgpr_spill || point.automatic_moi_record_replay_sgpr_spill);
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
    if (!moi_supports_dynamic_stack_spill(arch, request.moi_engine)) {
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
    const bool spill_backed_scalar_window =
        point.automatic_moi_inline_sgpr_spill || point.automatic_moi_record_replay_sgpr_spill;
    const auto special_state = moi_special_state_sgprs(request, point);
    const auto indirect_state = moi_indirect_jump_sgprs(request, point);
    if (!special_state || (spill_backed_scalar_window && !indirect_state)) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no SCC-save register");
      return std::nullopt;
    }
    const uint16_t saved_scc_sgpr =
        spill_backed_scalar_window ? indirect_state->scc_save_sgpr : special_state->scc_save_sgpr;
    const auto saved_frame_offset = moi_dynamic_stack_frame_save_sgpr_offset(request.moi_engine);
    if (!spill_backed_scalar_window &&
        (!saved_frame_offset ||
         *saved_frame_offset >=
             moi_exec_save_sgpr_count(
                 resolve_moi_exec_save_requirement(request, bound_resources, point), arch))) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no reserved frame-base save slot");
      return std::nullopt;
    }
    const uint16_t saved_frame_base_sgpr =
        spill_backed_scalar_window
            ? indirect_state->pc_sgpr
            : static_cast<uint16_t>(*point.moi_exec_save_sgpr + *saved_frame_offset);
    const uint32_t additional_frame_bytes =
        spill_backed_scalar_window
            ? static_cast<uint32_t>(moi_exec_save_sgpr_count(
                  resolve_moi_exec_save_requirement(request, bound_resources, point), arch)) *
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

// Descriptor patches are applied after text growth, so the active descriptor
// offset must be recovered by stable kernel name. Large generated libraries
// have thousands of long, common-prefix names; indexing both sides once keeps
// each descriptor requirement lookup constant-time.
class MoiActiveKernelResolver {
public:
  MoiActiveKernelResolver(const ConSanTransformArtifacts &result,
                          const AmdGpuCodeObject &code_object) {
    result_kernels_by_descriptor_.reserve(result.program_inventory.kernels().size());
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
      result_kernels_by_descriptor_.emplace(kernel.descriptor_file_offset, &kernel);
    active_kernels_by_name_.reserve(code_object.kernels().size());
    for (const AmdGpuKernelInfo &kernel : code_object.kernels())
      active_kernels_by_name_.emplace(kernel.name, &kernel);
  }

  [[nodiscard]] const ConSanKernelInfo *result_kernel(uint64_t descriptor_offset) const {
    const auto it = result_kernels_by_descriptor_.find(descriptor_offset);
    return it == result_kernels_by_descriptor_.end() ? nullptr : it->second;
  }

  [[nodiscard]] const AmdGpuKernelInfo *active_kernel(const ConSanKernelInfo &kernel) const {
    const auto it = active_kernels_by_name_.find(kernel.name);
    return it == active_kernels_by_name_.end() ? nullptr : it->second;
  }

private:
  std::unordered_map<uint64_t, const ConSanKernelInfo *> result_kernels_by_descriptor_;
  std::unordered_map<std::string_view, const AmdGpuKernelInfo *> active_kernels_by_name_;
};

[[nodiscard]] bool apply_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object, std::span<const uint8_t> image,
    const ConSanTransformArtifacts &result, const MoiDescriptorVgprRequirements &requirements,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  const MoiActiveKernelResolver kernel_resolver(result, code_object);
  for (const auto &[descriptor_offset, required_count] : requirements) {
    const ConSanKernelInfo *kernel = kernel_resolver.result_kernel(descriptor_offset);
    if (kernel == nullptr) {
      errors.emplace_back("ConSan MOI resource plan references an unknown kernel descriptor");
      return false;
    }
    const AmdGpuKernelInfo *active_kernel = kernel_resolver.active_kernel(*kernel);
    if (active_kernel == nullptr) {
      errors.emplace_back("ConSan MOI resource plan could not resolve an active descriptor");
      return false;
    }
    ConSanKernelInfo active = *kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    if (!grow_moi_kernel_descriptor_vgprs(patcher, image, active, required_count, arch, errors))
      return false;
  }
  return true;
}

[[nodiscard]] bool apply_spill_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object, std::span<const uint8_t> image,
    const ConSanTransformArtifacts &result, const MoiDescriptorPrivateRequirements &requirements,
    std::vector<std::string> &errors) {
  (void)image;
  const MoiActiveKernelResolver kernel_resolver(result, code_object);
  for (const auto &[descriptor_offset, required_private_bytes] : requirements) {
    // A lane-backed fixed-stack scalar save needs no private-memory backing.
    // Callers retain a uniform requirement map across lane and memory
    // representations, so zero is an intentional no-op rather than an
    // invalid request to the descriptor spill updater.
    if (required_private_bytes == 0u)
      continue;
    const ConSanKernelInfo *kernel = kernel_resolver.result_kernel(descriptor_offset);
    const AmdGpuKernelInfo *active_kernel =
        kernel == nullptr ? nullptr : kernel_resolver.active_kernel(*kernel);
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    if (kernel == nullptr || active_kernel == nullptr) {
      errors.emplace_back("ConSan MOI spill descriptor exceeds ELF bytes or has no kernel owner");
      return false;
    }
    auto descriptor = read_kernel_descriptor(current_image, active_kernel->descriptor_file_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan MOI spill descriptor exceeds ELF bytes or has no kernel owner");
      return false;
    }
    const SpillDescriptorUpdate update =
        update_kernel_descriptor_for_spills(*descriptor, required_private_bytes);
    if (update != SpillDescriptorUpdate::Updated && update != SpillDescriptorUpdate::Unchanged) {
      errors.emplace_back("ConSan MOI could not grow the owning kernel's private spill segment");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(active_kernel->descriptor_file_offset, *descriptor)) {
      errors.emplace_back("ConSan MOI could not patch the owning kernel spill descriptor");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool apply_sgpr_descriptor_requirements(
    CodeObjectPatcher &patcher, const ConSanTransformArtifacts &result,
    const AmdGpuCodeObject &code_object, const MoiDescriptorSgprRequirements &requirements,
    std::vector<std::string> &errors) {
  const MoiActiveKernelResolver kernel_resolver(result, code_object);
  for (const auto &[descriptor_offset, required_count] : requirements) {
    const ConSanKernelInfo *kernel = kernel_resolver.result_kernel(descriptor_offset);
    const AmdGpuKernelInfo *active_kernel =
        kernel == nullptr ? nullptr : kernel_resolver.active_kernel(*kernel);
    if (active_kernel == nullptr) {
      errors.emplace_back("ConSan MOI scalar plan could not resolve an active descriptor");
      return false;
    }
    const uint64_t active_descriptor_offset = active_kernel->descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    auto descriptor = read_kernel_descriptor(current_image, active_descriptor_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan MOI scalar-plan descriptor exceeds ELF bytes");
      return false;
    }
    if (!grow_descriptor_sgpr_allocation(*descriptor, required_count,
                                         result.program_inventory.arch())) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor SGPR allocation");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(active_descriptor_offset, *descriptor)) {
      errors.emplace_back("ConSan MOI could not patch the planned descriptor SGPR allocation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool apply_lds_descriptor_requirements(
    CodeObjectPatcher &patcher, const ConSanTransformArtifacts &result,
    const AmdGpuCodeObject &code_object, const MoiDescriptorLdsRequirements &requirements,
    const RuntimeCapabilities &capabilities, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  const MoiActiveKernelResolver kernel_resolver(result, code_object);
  for (const auto &[descriptor_offset, required_bytes] : requirements) {
    const ConSanKernelInfo *kernel = kernel_resolver.result_kernel(descriptor_offset);
    const AmdGpuKernelInfo *active_kernel =
        kernel == nullptr ? nullptr : kernel_resolver.active_kernel(*kernel);
    if (active_kernel == nullptr) {
      errors.emplace_back("ConSan MOI LDS plan could not resolve an active descriptor");
      return false;
    }
    const uint64_t active_descriptor_offset = active_kernel->descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    auto descriptor = read_kernel_descriptor(current_image, active_descriptor_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor exceeds ELF bytes");
      return false;
    }
    if (required_bytes > consan_moi_max_workgroup_lds_bytes(capabilities, arch) ||
        descriptor->group_segment_fixed_size > required_bytes) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor has an incompatible fixed-LDS size");
      return false;
    }
    descriptor->group_segment_fixed_size = required_bytes;
    if (!patcher.patch_kernel_descriptor(active_descriptor_offset, *descriptor)) {
      errors.emplace_back("ConSan MOI could not patch the planned fixed-LDS reservation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool apply_descriptor_requirements(std::vector<uint8_t> &image,
                                                 const ConSanTransformArtifacts &result,
                                                 const MoiDescriptorVgprRequirements &requirements,
                                                 rj_code_arch_t arch,
                                                 std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    auto descriptor = read_kernel_descriptor(image, descriptor_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan MOI resource-plan descriptor exceeds ELF bytes");
      return false;
    }
    const ConSanKernelInfo *kernel =
        result.program_inventory.find_kernel_by_descriptor(descriptor_offset);
    if (kernel == nullptr) {
      errors.emplace_back("ConSan MOI resource plan references an unknown kernel descriptor");
      return false;
    }
    if (!grow_descriptor_vgpr_allocation(
            *descriptor,
            {.required_ordinary_count = required_count,
             .maximum_ordinary_count = kMaxVgprs,
             .accumulator_bank_is_proven_empty = kernel->agpr_count == 0u},
            arch)) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor VGPR allocation");
      return false;
    }
    if (!write_kernel_descriptor(image, descriptor_offset, *descriptor))
      return false;
  }
  return true;
}

[[nodiscard]] bool apply_sgpr_descriptor_requirements(
    std::vector<uint8_t> &image, const ConSanTransformArtifacts &result,
    const MoiDescriptorSgprRequirements &requirements, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    auto descriptor = read_kernel_descriptor(image, descriptor_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan MOI scalar-plan descriptor exceeds ELF bytes");
      return false;
    }
    if (result.program_inventory.find_kernel_by_descriptor(descriptor_offset) == nullptr) {
      errors.emplace_back("ConSan MOI scalar plan references an unknown kernel descriptor");
      return false;
    }
    if (!grow_descriptor_sgpr_allocation(*descriptor, required_count,
                                         result.program_inventory.arch())) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor SGPR allocation");
      return false;
    }
    if (!write_kernel_descriptor(image, descriptor_offset, *descriptor))
      return false;
  }
  return true;
}

[[nodiscard]] bool apply_lds_descriptor_requirements(
    std::vector<uint8_t> &image, const ConSanTransformArtifacts &result,
    const MoiDescriptorLdsRequirements &requirements, const RuntimeCapabilities &capabilities,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_bytes] : requirements) {
    auto descriptor = read_kernel_descriptor(image, descriptor_offset);
    if (!descriptor ||
        result.program_inventory.find_kernel_by_descriptor(descriptor_offset) == nullptr) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor exceeds ELF bytes or has no owner");
      return false;
    }
    if (required_bytes > consan_moi_max_workgroup_lds_bytes(capabilities, arch) ||
        descriptor->group_segment_fixed_size > required_bytes) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor has an incompatible fixed-LDS size");
      return false;
    }
    descriptor->group_segment_fixed_size = required_bytes;
    if (!write_kernel_descriptor(image, descriptor_offset, *descriptor))
      return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_first_light_access_record_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, uint16_t scratch_vgpr, rj_code_arch_t arch,
    uint32_t record_index, uint32_t record_count, uint32_t logical_range_index,
    const ConSanMoiReportBufferLayout &layout, bool spill_overlaps_guest_operands,
    const VgprSpillSequence *spill, std::optional<uint32_t> private_epoch_offset,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset,
    uint32_t *guest_instruction_word_count) {
  const auto fail = [&](const char *message) -> std::optional<std::vector<uint32_t>> {
    errors.emplace_back(message);
    return std::nullopt;
  };
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return fail("ConSan MOI first-light probe has no target profile");
  const bool automatic_banked_capture = record_replay_uses_automatic_banked_capture(
      request, layout.record_replay_dispatch_token_capacity);
  const uint16_t base_scratch_count = automatic_banked_capture ? 10u : 6u;
  const bool materialize_flat_address = candidate_requires_flat_address_materialization(candidate);
  const bool materialize_direct_to_lds = candidate.is_direct_to_lds();
  const bool load_clobbers_address = moi_load_clobbers_address(candidate);
  const bool capture_high_bank_address =
      moi_access_requires_high_bank_address_capture(candidate, arch);
  const bool save_clobbered_address = load_clobbers_address || capture_high_bank_address;
  const bool reserve_two_address_replay_scratch =
      moi_guest_access_relocation_requires_adjusted_address(candidate, *target);
  const uint16_t address_scratch_count = static_cast<uint16_t>(
      (materialize_flat_address ? flat_access_address_scratch_count(candidate)
                                : (save_clobbered_address ? 1u : 0u)) +
      (materialize_direct_to_lds ? 1u : 0u) + (reserve_two_address_replay_scratch ? 1u : 0u));
  const uint16_t scratch_count = static_cast<uint16_t>(base_scratch_count + address_scratch_count);
  if (static_cast<uint32_t>(scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back(request.moi_dynamic_access_records
                            ? "ConSan MOI dynamic access-record probe needs six scratch VGPRs"
                        : automatic_banked_capture
                            ? "ConSan MOI automatic first-light probe needs ten scratch VGPRs"
                            : "ConSan MOI first-light probe needs six scratch VGPRs");
    return std::nullopt;
  }
  if (request.moi_dynamic_access_records && !point.moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (request.moi_dynamic_access_records &&
      (*point.moi_exec_save_sgpr > 100u || *point.moi_exec_save_sgpr % 2u != 0u)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..100");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (!spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, scratch_vgpr, scratch_count, errors))
    return std::nullopt;
  if (request.moi_dynamic_access_records && has_recent_saveexec(bytes, candidate)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe skipped a candidate immediately after "
        "s_*_saveexec");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(point.moi_owner_vgpr, scratch_vgpr, scratch_count,
                                            "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(point.moi_epoch_vgpr, scratch_vgpr, scratch_count,
                                            "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(point.moi_workgroup_key_vgpr, scratch_vgpr,
                                            scratch_count, "MOI workgroup key", errors) ||
      reject_optional_scratch_range_overlap(point.moi_record_replay_workgroup_vgprs.x, scratch_vgpr,
                                            scratch_count, "MOI Record/Replay workgroup x",
                                            errors) ||
      reject_optional_scratch_range_overlap(point.moi_record_replay_workgroup_vgprs.y, scratch_vgpr,
                                            scratch_count, "MOI Record/Replay workgroup y",
                                            errors) ||
      reject_optional_scratch_range_overlap(point.moi_record_replay_workgroup_vgprs.z, scratch_vgpr,
                                            scratch_count, "MOI Record/Replay workgroup z",
                                            errors) ||
      reject_optional_scratch_range_overlap(
          point.moi_record_replay_workgroup_vgprs.cluster_workgroup_id, scratch_vgpr, scratch_count,
          "MOI Record/Replay cluster workgroup ID", errors))
    return std::nullopt;
  const auto &access_ranges = candidate.ranges;
  if (access_ranges.empty()) {
    errors.emplace_back("ConSan MOI first-light probe requires a supported LDS access range");
    return std::nullopt;
  }
  if (materialize_flat_address && access_ranges.size() != 1u) {
    errors.emplace_back("ConSan MOI flat-address probe requires one normalized access range");
    return std::nullopt;
  }
  if (capture_high_bank_address && load_clobbers_address && candidate.is_native_two_range()) {
    errors.emplace_back("ConSan MOI first-light probe cannot yet preserve a clobbered high-bank "
                        "gfx1250 two-address LDS operand");
    return std::nullopt;
  }
  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!point.moi_owner_vgpr && !point.moi_persistent_sgprs.owner) {
    if (!owner_derivation) {
      errors.emplace_back("ConSan MOI first-light probe requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr =
        static_cast<uint16_t>(scratch_vgpr + (request.moi_dynamic_access_records ? 4u : 2u));
    const auto owner = build_moi_workitem_owner_derivation(*owner_derivation, value_vgpr, arch,
                                                           "first-light probe", errors);
    if (!owner)
      return std::nullopt;
    const auto owner_mask = instrumentation::build_v_and_b32_literal(
        value_vgpr, consan_moi_exact_shadow::max_owner, value_vgpr, arch);
    if (!owner_mask) {
      errors.emplace_back("ConSan MOI first-light probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
    derived_owner_words.insert(derived_owner_words.end(), owner_mask->begin(), owner_mask->end());
  }
  const auto persistent_workgroup_sources =
      record_replay_persistent_workgroup_sources(request.moi_engine, point);
  if (!persistent_workgroup_sources) {
    errors.emplace_back(
        "ConSan MOI Record/Replay access requires one exact entry-captured workgroup tuple");
    return std::nullopt;
  }
  const ConSanMoiWorkgroupSources workgroup_sources = *persistent_workgroup_sources;

  std::vector<uint32_t> words;
  words.reserve(candidate.size() / sizeof(uint32_t) + 1u + 7u * 12u + 9u + 10u + 20u + 24u +
                (point.moi_owner_vgpr || derived_owner_vgpr ? 9u : 0u) +
                (point.moi_epoch_vgpr ? 9u : 0u) + derived_owner_words.size());
  InstructionSequence sequence(words);
  const auto restore_exec_label = sequence.make_label();
  // A DS load may overwrite the same VGPR that supplied its address. Preserve
  // that effective address before executing the displaced guest instruction;
  // otherwise RecordReplay publishes the loaded payload as the LDS offset.
  if (materialize_direct_to_lds) {
    if (!point.moi_exec_save_sgpr) {
      errors.emplace_back(
          "ConSan MOI first-light direct-to-LDS probe requires an EXEC-save SGPR pair");
      return std::nullopt;
    }
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    if (!append_materialize_direct_to_lds_address(words, candidate, materialized_address_vgpr,
                                                  *point.moi_exec_save_sgpr, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not materialize a direct-to-LDS "
                          "destination");
      return std::nullopt;
    }
    lds_byte_offset_vgpr = materialized_address_vgpr;
  } else if (materialize_flat_address && request.moi_dynamic_access_records) {
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    words.push_back(build_v_mov_b32_e32(materialized_address_vgpr,
                                        vector_source_vgpr(*candidate.lowering.form->address_vgpr),
                                        arch));
    if (!candidate_uses_scalar_vector_flat_address(candidate)) {
      words.push_back(build_v_mov_b32_e32(
          static_cast<uint16_t>(materialized_address_vgpr + 1u),
          vector_source_vgpr(static_cast<uint16_t>(*candidate.lowering.form->address_vgpr + 1u)),
          arch));
    }
    lds_byte_offset_vgpr = materialized_address_vgpr;
  } else if (save_clobbered_address) {
    const uint16_t saved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    if (!capture_high_bank_address) {
      words.push_back(build_v_mov_b32_e32(saved_lds_byte_offset_vgpr,
                                          vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  if (request.moi_dynamic_access_records)
    words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
  const auto append_guest_access = [&] {
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    const uint16_t guest_address_vgpr =
        capture_high_bank_address ? *candidate.lowering.form->address_vgpr : *lds_byte_offset_vgpr;
    const auto guest_words = build_moi_relocated_guest_access_words(
        {.image = bytes,
         .candidate = &candidate,
         .target = target,
         .replay_address_vgpr = guest_address_vgpr,
         .adjusted_address_vgpr =
             reserve_two_address_replay_scratch
                 ? std::optional<uint16_t>(static_cast<uint16_t>(scratch_vgpr + scratch_count - 1u))
                 : std::nullopt},
        errors);
    if (!guest_words)
      return false;
    if (guest_instruction_word_count)
      *guest_instruction_word_count = static_cast<uint32_t>(guest_words->size());
    words.insert(words.end(), guest_words->begin(), guest_words->end());
    return true;
  };
  if (request.moi_dynamic_access_records) {
    if (!append_guest_access())
      return std::nullopt;
    if (!append_moi_lds_wait(words, arch))
      return fail("ConSan MOI dynamic access-record probe could not encode its LDS wait");
  }

  const uint64_t base = *bound_resources.moi_report_buffer_address;
  const uint32_t access_record_capacity = layout.access_record_capacity;
  const uint32_t access_dispatch_bank_count = layout.record_replay_access_dispatch_bank_count;
  const uint32_t access_owner_bank_count = layout.record_replay_access_owner_bank_count;
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  auto append_effective_range_offset = [&](const ConSanAccessRange &range,
                                           uint16_t value_vgpr) -> std::optional<uint16_t> {
    if (candidate.lowering_offset(range) == 0)
      return *lds_byte_offset_vgpr;
    if (!append_compute_effective_lds_byte_offset(words, value_vgpr, *lds_byte_offset_vgpr,
                                                  candidate.lowering_offset(range), arch)) {
      return std::nullopt;
    }
    return value_vgpr;
  };

  if (request.moi_dynamic_access_records) {
    const uint16_t slot_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
    const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
    const uint64_t dynamic_record_base = base + layout.access_records_offset;

    for (const ConSanAccessRange &range : access_ranges) {
      if (!append_save_moi_special_state(words, moi_special_state_sgprs(request, point), arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not save VCC/SCC");
        return std::nullopt;
      }
      if (materialize_flat_address) {
        if (!append_materialize_flat_access_address(words, candidate, *lds_byte_offset_vgpr,
                                                    *lds_byte_offset_vgpr, arch)) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not materialize FLAT address");
          return std::nullopt;
        }
      }

      if (!append_atomic_fetch_add_one_u32(
              words, base + offsetof(ConSanMoiReportHeader, access_record_count), slot_vgpr,
              scratch_vgpr, arch)) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not reserve a record slot");
        return std::nullopt;
      }

      const auto mov_capacity =
          instrumentation::build_v_mov_b32_literal(value_vgpr, access_record_capacity, arch);
      const auto slot_in_capacity =
          instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
      const auto narrow_exec =
          instrumentation::build_s_and_saveexec_b64(*point.moi_exec_save_sgpr, kRdna4VccLo, arch);
      if (!mov_capacity || !slot_in_capacity || !narrow_exec) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode capacity guard");
        return std::nullopt;
      }
      words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
      words.push_back(*slot_in_capacity);
      words.push_back(*narrow_exec);

      if (private_epoch_offset) {
        const auto load_epoch =
            instrumentation::build_private_load_b32(value_vgpr, *private_epoch_offset, arch);
        const auto wait_epoch = instrumentation::build_s_wait_private_load0(arch);
        if (!load_epoch || !wait_epoch) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not load private epoch state");
          return std::nullopt;
        }
        words.insert(words.end(), load_epoch->begin(), load_epoch->end());
        words.push_back(*wait_epoch);
        if (!append_dynamic_record_store_u32_vgpr(words, kAccessRecordLayout,
                                                  dynamic_record_base +
                                                      offsetof(ConSanMoiAccessRecord, epoch),
                                                  value_vgpr, slot_vgpr, scratch_vgpr, arch)) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not store private epoch state");
          return std::nullopt;
        }
      } else if (point.moi_persistent_sgprs.epoch) {
        words.push_back(build_v_mov_b32_e32(value_vgpr, *point.moi_persistent_sgprs.epoch, arch));
        if (!append_dynamic_record_store_u32_vgpr(words, kAccessRecordLayout,
                                                  dynamic_record_base +
                                                      offsetof(ConSanMoiAccessRecord, epoch),
                                                  value_vgpr, slot_vgpr, scratch_vgpr, arch)) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not store scalar epoch state");
          return std::nullopt;
        }
      }

      if ((derived_owner_vgpr && !append_dynamic_record_store_u32_vgpr(
                                     words, kAccessRecordLayout,
                                     dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
                                     *derived_owner_vgpr, slot_vgpr, scratch_vgpr, arch)) ||
          (point.moi_persistent_sgprs.owner &&
           !append_dynamic_record_store_u32_scalar_src(
               words, kAccessRecordLayout,
               dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *point.moi_persistent_sgprs.owner, slot_vgpr, scratch_vgpr, arch)) ||
          !append_dynamic_record_store_moi_report_dispatch_id_pair(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, generation), point,
              bound_resources, slot_vgpr, scratch_vgpr, arch,
              ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
          !append_dynamic_record_event_index_store(
              words, kAccessRecordLayout, base + offsetof(ConSanMoiReportHeader, event_counter),
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, event_index), slot_vgpr,
              scratch_vgpr, arch) ||
          !append_dynamic_record_store_workgroup_source(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_x),
              workgroup_sources.x, slot_vgpr, scratch_vgpr, arch) ||
          !append_dynamic_record_store_workgroup_source(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_y),
              workgroup_sources.y, slot_vgpr, scratch_vgpr, arch) ||
          !append_dynamic_record_store_workgroup_source(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_z),
              workgroup_sources.z, slot_vgpr, scratch_vgpr, arch) ||
          !append_dynamic_record_store_u32_scalar_src(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, lane_mask),
              *point.moi_exec_save_sgpr, slot_vgpr, scratch_vgpr, arch) ||
          !append_dynamic_record_store_u32_scalar_src(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
              static_cast<uint16_t>(*point.moi_exec_save_sgpr + 1u), slot_vgpr, scratch_vgpr,
              arch) ||
          (point.moi_owner_vgpr &&
           !append_dynamic_record_store_u32_vgpr(
               words, kAccessRecordLayout,
               dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *point.moi_owner_vgpr, slot_vgpr, scratch_vgpr, arch)) ||
          (point.moi_epoch_vgpr && !append_dynamic_record_store_u32_vgpr(
                                       words, kAccessRecordLayout,
                                       dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch),
                                       *point.moi_epoch_vgpr, slot_vgpr, scratch_vgpr, arch)) ||
          !append_dynamic_record_store_u32_literal(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, instruction_offset),
              static_cast<uint32_t>(candidate.anchor()), slot_vgpr, scratch_vgpr, arch) ||
          !append_dynamic_record_store_u32_literal(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, access_kind),
              static_cast<uint32_t>(kind), slot_vgpr, scratch_vgpr, arch)) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode record stores");
        return std::nullopt;
      }

      const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
          append_effective_range_offset(range, value_vgpr);
      if (!effective_lds_byte_offset_vgpr) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode LDS byte offset");
        return std::nullopt;
      }
      if (!append_dynamic_record_store_u32_vgpr(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
              *effective_lds_byte_offset_vgpr, slot_vgpr, scratch_vgpr, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode range fields");
        return std::nullopt;
      }
      const auto start_cell_shift = instrumentation::build_v_lshrrev_b32(
          value_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
          *effective_lds_byte_offset_vgpr, arch);
      if (!start_cell_shift) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode start cell");
        return std::nullopt;
      }
      words.push_back(*start_cell_shift);
      const ConSanMoiLdsCellRange static_range =
          consan_moi_lds_cell_range_for_bytes(candidate.lowering_offset(range), range.byte_width);
      if (!append_dynamic_record_store_u32_vgpr(words, kAccessRecordLayout,
                                                dynamic_record_base +
                                                    offsetof(ConSanMoiAccessRecord, start_cell),
                                                value_vgpr, slot_vgpr, scratch_vgpr, arch) ||
          !append_dynamic_record_store_u32_literal(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_count),
              range.byte_width, slot_vgpr, scratch_vgpr, arch) ||
          !append_dynamic_record_store_u32_literal(
              words, kAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, cell_count),
              static_range.cell_count, slot_vgpr, scratch_vgpr, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode range fields");
        return std::nullopt;
      }
      const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
      const auto restore_exec =
          instrumentation::build_s_mov_b64(kRdna4ExecLo, *point.moi_exec_save_sgpr, arch);
      if (!wait_store || !restore_exec) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not drain stores or restore EXEC");
        return std::nullopt;
      }
      // A terminal guest access can return directly to s_endpgm. Drain the
      // injected report publication while the scratch values and narrowed
      // EXEC are still intact, so wave termination cannot race those stores.
      words.push_back(*wait_store);
      words.push_back(*restore_exec);
      if (!append_restore_moi_special_state(words, moi_special_state_sgprs(request, point), arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not restore VCC/SCC");
        return std::nullopt;
      }
    }
    return words;
  }

  const uint16_t record_address_vgpr = scratch_vgpr;
  const uint16_t record_value_vgpr = static_cast<uint16_t>(record_address_vgpr + 2u);
  const uint16_t record_value_high_vgpr = static_cast<uint16_t>(record_address_vgpr + 3u);
  const uint16_t record_compare_vgpr = static_cast<uint16_t>(record_address_vgpr + 4u);
  const uint16_t record_compare_high_vgpr = static_cast<uint16_t>(record_address_vgpr + 5u);
  const uint16_t dispatch_bank_vgpr = static_cast<uint16_t>(record_address_vgpr + 6u);
  const uint16_t owner_bank_vgpr = static_cast<uint16_t>(record_address_vgpr + 7u);
  const uint16_t bank_probe_count_vgpr = static_cast<uint16_t>(record_address_vgpr + 8u);
  const uint16_t identity_claim_hash_vgpr = static_cast<uint16_t>(record_address_vgpr + 9u);
  const uint16_t original_exec_sgpr = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 8u);
  const uint16_t address_key_sgpr = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 10u);
  const uint16_t address_group_exec_sgpr = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 12u);
  const std::optional<uint16_t> spilled_lds_byte_offset_vgpr =
      !materialize_flat_address && !save_clobbered_address && spill != nullptr &&
              spill_overlaps_guest_operands && *lds_byte_offset_vgpr >= spill->vgpr_base &&
              *lds_byte_offset_vgpr < static_cast<uint32_t>(spill->vgpr_base) + spill->vgpr_count
          ? lds_byte_offset_vgpr
          : std::nullopt;
  if (spilled_lds_byte_offset_vgpr)
    lds_byte_offset_vgpr = record_compare_high_vgpr;
  ConSanMoiRecordEmitter record(words, record_address_vgpr, record_value_vgpr, arch);
  const uint64_t access_bank_count =
      static_cast<uint64_t>(access_dispatch_bank_count) * access_owner_bank_count;
  if (access_dispatch_bank_count == 0u ||
      (access_dispatch_bank_count & (access_dispatch_bank_count - 1u)) != 0u ||
      access_owner_bank_count == 0u ||
      (access_owner_bank_count & (access_owner_bank_count - 1u)) != 0u ||
      access_bank_count > std::numeric_limits<uint32_t>::max() ||
      (automatic_banked_capture &&
       (layout.record_replay_logical_access_range_count == 0u ||
        static_cast<uint64_t>(layout.record_replay_logical_access_range_count) * access_bank_count *
                layout.record_replay_address_group_headroom >
            access_record_capacity ||
        layout.record_replay_address_group_headroom == 0u ||
        layout.record_replay_address_group_headroom >
            kConSanMoiRecordReplayMaximumAddressGroupsPerWave ||
        (layout.record_replay_address_group_headroom &
         (layout.record_replay_address_group_headroom - 1u)) != 0u ||
        (access_record_capacity & (access_record_capacity - 1u)) != 0u ||
        layout.record_replay_dispatch_token_capacity >
            kConSanMoiRecordReplayMaximumDispatchTokenCount ||
        (layout.record_replay_dispatch_token_capacity &
         (layout.record_replay_dispatch_token_capacity - 1u)) != 0u)) ||
      (!automatic_banked_capture &&
       (access_dispatch_bank_count != 1u || access_owner_bank_count != 1u))) {
    errors.emplace_back("ConSan MOI first-light probe has an invalid identity-table layout");
    return std::nullopt;
  }
  const auto append_claim_token = [&](uint16_t low_vgpr, uint16_t high_vgpr,
                                      uint16_t temporary_vgpr) {
    if (!append_moi_report_dispatch_id_pair(words, point, bound_resources, low_vgpr, high_vgpr,
                                            arch,
                                            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture))
      return false;
    const auto low_xor_literal = instrumentation::build_v_mov_b32_literal(
        temporary_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask), arch);
    const auto low_xor = instrumentation::build_v_xor_b32(
        low_vgpr, vector_source_vgpr(temporary_vgpr), low_vgpr, arch);
    const auto high_xor_literal = instrumentation::build_v_mov_b32_literal(
        temporary_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u),
        arch);
    const auto high_xor = instrumentation::build_v_xor_b32(
        high_vgpr, vector_source_vgpr(temporary_vgpr), high_vgpr, arch);
    InstructionSequence append(words);
    return append.emit_all(low_xor_literal, low_xor, high_xor_literal, high_xor);
  };
  const auto append_owner_id = [&](uint16_t destination_vgpr) {
    if (point.moi_owner_vgpr) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, vector_source_vgpr(*point.moi_owner_vgpr), arch));
      return true;
    }
    if (point.moi_persistent_sgprs.owner) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, *point.moi_persistent_sgprs.owner, arch));
      return true;
    }
    if (!derived_owner_vgpr)
      return false;
    words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
    if (*derived_owner_vgpr != destination_vgpr) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, vector_source_vgpr(*derived_owner_vgpr), arch));
    }
    return true;
  };
  const auto append_access_claim_token = [&](uint16_t low_vgpr, uint16_t high_vgpr,
                                             uint16_t temporary_vgpr, uint32_t site_token) {
    if (!append_claim_token(low_vgpr, high_vgpr, temporary_vgpr))
      return false;
    const auto combine_identity = instrumentation::build_v_xor_b32(
        low_vgpr, vector_source_vgpr(low_vgpr), identity_claim_hash_vgpr, arch);
    const auto materialize_site =
        instrumentation::build_v_mov_b32_literal(temporary_vgpr, site_token, arch);
    const auto combine_site = instrumentation::build_v_xor_b32(
        high_vgpr, vector_source_vgpr(high_vgpr), temporary_vgpr, arch);
    if (!combine_identity || !materialize_site || !combine_site)
      return false;
    words.push_back(*combine_identity);
    words.insert(words.end(), materialize_site->begin(), materialize_site->end());
    words.push_back(*combine_site);
    return true;
  };
  const auto materialize_banked_record_address = [&](uint64_t first_address) {
    if (!record.materialize_address(first_address))
      return false;
    if (!automatic_banked_capture)
      return true;
    const auto scale_bank = instrumentation::build_v_mul_lo_u32_literal(
        record_value_vgpr, record_value_high_vgpr, sizeof(ConSanMoiAccessRecord), owner_bank_vgpr,
        arch);
    const auto add_bank =
        instrumentation::build_v_add_u64_vgpr_offset(record_address_vgpr, record_value_vgpr, arch);
    InstructionSequence append(words);
    return append.emit_all(scale_bank, add_bank);
  };
  const auto reload_spilled_lds_byte_offset = [&] {
    if (!spilled_lds_byte_offset_vgpr)
      return true;
    if (spill == nullptr) {
      errors.emplace_back("ConSan MOI first-light probe has no spill plan for its LDS address");
      return false;
    }
    const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
        words, *spill, record_compare_high_vgpr, *spilled_lds_byte_offset_vgpr, arch);
    if (reload == consan_detail::MoiSpilledVgprReloadResult::Appended)
      return true;
    errors.emplace_back("ConSan MOI first-light probe could not recover its spilled LDS address: " +
                        std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
    return false;
  };
  if (!point.moi_exec_save_sgpr ||
      !append_save_moi_special_state(words, moi_special_state_sgprs(request, point), arch)) {
    errors.emplace_back("ConSan MOI first-light probe could not save EXEC/VCC/SCC");
    return std::nullopt;
  }
  if (materialize_flat_address) {
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    if (!append_materialize_flat_access_address(words, candidate, *lds_byte_offset_vgpr,
                                                materialized_address_vgpr, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not materialize FLAT address");
      return std::nullopt;
    }
    lds_byte_offset_vgpr = materialized_address_vgpr;
  }
  std::optional<size_t> address_group_loop;
  if (automatic_banked_capture) {
    const auto save_original_exec =
        instrumentation::build_s_mov_b64(original_exec_sgpr, kRdna4ExecLo, arch);
    if (!save_original_exec) {
      errors.emplace_back("ConSan MOI first-light probe could not retain its incoming EXEC");
      return std::nullopt;
    }
    words.push_back(*save_original_exec);

    // Saturation is a monotonic report-wide latch. Read it once before the
    // distinct-address loop; a group that saturates during this probe still
    // exits through the in-loop saturation paths below.
    if (!record.materialize_address(base) ||
        !record.load(offsetof(ConSanMoiReportHeader, flags), record_value_vgpr)) {
      errors.emplace_back("ConSan MOI first-light probe could not read report saturation");
      return std::nullopt;
    }
    const auto saturation_read_wait = instrumentation::build_s_wait_global_load0(arch);
    const auto select_saturation = instrumentation::build_v_and_b32_literal(
        record_compare_vgpr, kConSanMoiReportFlagRecordReplayBankSaturated, record_value_vgpr,
        arch);
    const auto already_saturated = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_compare_vgpr, arch);
    if (!saturation_read_wait || !select_saturation || !already_saturated) {
      errors.emplace_back("ConSan MOI first-light probe could not test report saturation");
      return std::nullopt;
    }
    words.push_back(*saturation_read_wait);
    words.insert(words.end(), select_saturation->begin(), select_saturation->end());
    words.push_back(*already_saturated);
    if (!sequence.emit_branch(restore_exec_label, InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not branch around a saturated report");
    }

    address_group_loop = words.size();
    if (!reload_spilled_lds_byte_offset()) {
      errors.emplace_back(
          "ConSan MOI first-light probe could not recover its address-group source");
      return std::nullopt;
    }
    const auto read_address =
        instrumentation::build_v_readfirstlane_b32(address_key_sgpr, *lds_byte_offset_vgpr, arch);
    const auto select_address =
        instrumentation::build_v_cmp_eq_u32_vcc(address_key_sgpr, *lds_byte_offset_vgpr, arch);
    const auto narrow_group =
        instrumentation::build_s_and_saveexec_b64(*point.moi_exec_save_sgpr, kRdna4VccLo, arch);
    const auto save_address_group =
        instrumentation::build_s_mov_b64(address_group_exec_sgpr, kRdna4ExecLo, arch);
    const auto lane_rank_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
        record_value_vgpr, address_group_exec_sgpr, scalar_positive_inline_u32(0), arch);
    const auto lane_rank_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
        record_value_vgpr, static_cast<uint16_t>(address_group_exec_sgpr + 1u),
        vector_source_vgpr(record_value_vgpr), arch);
    const auto first_in_group = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(0), record_value_vgpr, arch);
    const auto narrow_representative =
        instrumentation::build_s_and_saveexec_b64(address_group_exec_sgpr, kRdna4VccLo, arch);
    if (!read_address || !select_address || !narrow_group || !save_address_group || !lane_rank_lo ||
        !lane_rank_hi || !first_in_group || !narrow_representative) {
      errors.emplace_back(
          "ConSan MOI first-light probe could not select one exact LDS address-group owner");
      return std::nullopt;
    }
    words.push_back(*read_address);
    words.push_back(*select_address);
    words.push_back(*narrow_group);
    words.push_back(*save_address_group);
    words.insert(words.end(), lane_rank_lo->begin(), lane_rank_lo->end());
    words.insert(words.end(), lane_rank_hi->begin(), lane_rank_hi->end());
    words.push_back(*first_in_group);
    // Keep the full group mask for replay provenance while one elected lane
    // performs the bounded table transaction.
    words.push_back(*narrow_representative);
  } else {
    const auto save_incoming_exec =
        instrumentation::build_s_mov_b64(*point.moi_exec_save_sgpr, kRdna4ExecLo, arch);
    const auto lane_rank_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
        record_value_vgpr, *point.moi_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
    const auto lane_rank_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
        record_value_vgpr, static_cast<uint16_t>(*point.moi_exec_save_sgpr + 1u),
        vector_source_vgpr(record_value_vgpr), arch);
    const auto first_active = instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0),
                                                                      record_value_vgpr, arch);
    const auto narrow_first =
        instrumentation::build_s_and_saveexec_b64(*point.moi_exec_save_sgpr, kRdna4VccLo, arch);
    if (!save_incoming_exec || !lane_rank_lo || !lane_rank_hi || !first_active || !narrow_first) {
      errors.emplace_back("ConSan MOI first-light probe could not elect a representative lane");
      return std::nullopt;
    }
    words.push_back(*save_incoming_exec);
    words.insert(words.end(), lane_rank_lo->begin(), lane_rank_lo->end());
    words.insert(words.end(), lane_rank_hi->begin(), lane_rank_hi->end());
    words.push_back(*first_active);
    words.push_back(*narrow_first);
  }
  if (automatic_banked_capture) {
    // The bounded bank hashes the complete persistent tuple. Descriptor SGPRs
    // and launch TTMPs are entry inputs, not probe-lifetime state, so falling
    // back to either here would recreate cross-workgroup LDS aliasing.
    if (!workgroup_sources.x.has_value() || !workgroup_sources.y.has_value() ||
        !workgroup_sources.z.has_value()) {
      errors.emplace_back(
          "ConSan MOI automatic first-light probe requires a persistent exact workgroup tuple");
      return std::nullopt;
    }

    const auto dispatch_probe_label = sequence.make_label();
    const auto dispatch_occupied_label = sequence.make_label();
    const auto dispatch_retry_label = sequence.make_label();
    const auto dispatch_bank_ready_label = sequence.make_label();
    const auto dispatch_saturated_label = sequence.make_label();
    const auto dispatch_token_ready_label = sequence.make_label();

    if (!append_claim_token(record_value_vgpr, record_value_high_vgpr, record_compare_vgpr)) {
      errors.emplace_back("ConSan MOI first-light probe could not form its dispatch claim");
      return std::nullopt;
    }
    const auto token_low_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_vgpr, arch);
    const auto token_high_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_high_vgpr, arch);
    const auto hash_dispatch = instrumentation::build_v_xor_b32(
        dispatch_bank_vgpr, vector_source_vgpr(record_value_vgpr), record_value_high_vgpr, arch);
    const auto bound_dispatch = instrumentation::build_v_and_b32_literal(
        dispatch_bank_vgpr, layout.record_replay_dispatch_token_capacity - 1u, dispatch_bank_vgpr,
        arch);
    if (!token_low_nonzero || !token_high_nonzero || !hash_dispatch || !bound_dispatch) {
      errors.emplace_back("ConSan MOI first-light probe could not hash its dispatch claim");
      return std::nullopt;
    }
    words.push_back(*token_low_nonzero);
    if (!sequence.emit_branch(dispatch_token_ready_label,
                              InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not validate its dispatch token");
    }
    words.push_back(*token_high_nonzero);
    if (!sequence.emit_branch(dispatch_token_ready_label,
                              InstructionSequence::BranchKind::VccNonzero) ||
        !sequence.emit_branch(dispatch_saturated_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(dispatch_token_ready_label)) {
      return fail("ConSan MOI first-light dispatch-token path is invalid");
    }
    words.push_back(*hash_dispatch);
    words.insert(words.end(), bound_dispatch->begin(), bound_dispatch->end());
    words.push_back(
        build_v_mov_b32_e32(bank_probe_count_vgpr, scalar_positive_inline_u32(0), arch));

    if (!sequence.bind(dispatch_probe_label) ||
        !record.materialize_address(base + layout.record_replay_dispatch_tokens_offset)) {
      errors.emplace_back("ConSan MOI first-light probe could not address its dispatch table");
      return std::nullopt;
    }
    const auto scale_dispatch_slot = instrumentation::build_v_mul_lo_u32_literal(
        record_compare_vgpr, record_compare_high_vgpr, sizeof(uint64_t), dispatch_bank_vgpr, arch);
    const auto add_dispatch_slot = instrumentation::build_v_add_u64_vgpr_offset(
        record_address_vgpr, record_compare_vgpr, arch);
    if (!scale_dispatch_slot || !add_dispatch_slot) {
      errors.emplace_back("ConSan MOI first-light probe could not select a dispatch slot");
      return std::nullopt;
    }
    words.insert(words.end(), scale_dispatch_slot->begin(), scale_dispatch_slot->end());
    words.insert(words.end(), add_dispatch_slot->begin(), add_dispatch_slot->end());
    words.push_back(build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch));
    words.push_back(
        build_v_mov_b32_e32(record_compare_high_vgpr, scalar_positive_inline_u32(0), arch));
    const auto claim_dispatch = instrumentation::build_flat_atomic_cmpswap_b64(
        record_address_vgpr, record_value_vgpr, record_value_vgpr,
        /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    if (!claim_dispatch) {
      errors.emplace_back("ConSan MOI first-light probe could not claim a dispatch slot");
      return std::nullopt;
    }
    words.insert(words.end(), claim_dispatch->begin(), claim_dispatch->end());
    if (!append_moi_global_atomic_wait(words, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not wait for a dispatch slot claim");
      return std::nullopt;
    }
    const auto prior_dispatch_low_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_vgpr, arch);
    const auto prior_dispatch_high_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_high_vgpr, arch);
    if (!prior_dispatch_low_nonzero || !prior_dispatch_high_nonzero) {
      errors.emplace_back("ConSan MOI first-light probe could not inspect a dispatch slot");
      return std::nullopt;
    }
    words.push_back(*prior_dispatch_low_nonzero);
    if (!sequence.emit_branch(dispatch_occupied_label,
                              InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not inspect its dispatch claim");
    }
    words.push_back(*prior_dispatch_high_nonzero);
    if (!sequence.emit_branch(dispatch_occupied_label,
                              InstructionSequence::BranchKind::VccNonzero) ||
        !append_atomic_fetch_add_one_u32(
            words, base + offsetof(ConSanMoiReportHeader, record_replay_dispatch_token_count),
            record_compare_vgpr, record_address_vgpr, arch) ||
        !sequence.emit_branch(dispatch_bank_ready_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(dispatch_occupied_label)) {
      errors.emplace_back("ConSan MOI first-light dispatch publication is invalid");
      return std::nullopt;
    }

    const auto prior_dispatch_low_xor_literal = instrumentation::build_v_mov_b32_literal(
        record_compare_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask), arch);
    const auto decode_prior_dispatch_low = instrumentation::build_v_xor_b32(
        record_value_vgpr, vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
    if (!prior_dispatch_low_xor_literal || !decode_prior_dispatch_low ||
        !sequence.emit(*prior_dispatch_low_xor_literal) ||
        !sequence.emit(*decode_prior_dispatch_low) ||
        !append_moi_report_dispatch_id_word(words, point, bound_resources, record_compare_vgpr,
                                            /*high_word=*/false, arch,
                                            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
      errors.emplace_back("ConSan MOI first-light probe could not compare a dispatch slot");
      return std::nullopt;
    }
    const auto dispatch_low_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
    if (!dispatch_low_mismatch) {
      return fail("ConSan MOI first-light probe could not compare a dispatch low word");
    }
    words.push_back(*dispatch_low_mismatch);
    if (!sequence.emit_branch(dispatch_retry_label, InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not branch to its dispatch retry");
    }

    const auto prior_dispatch_high_xor_literal = instrumentation::build_v_mov_b32_literal(
        record_compare_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u),
        arch);
    const auto decode_prior_dispatch_high = instrumentation::build_v_xor_b32(
        record_value_high_vgpr, vector_source_vgpr(record_compare_vgpr), record_value_high_vgpr,
        arch);
    if (!prior_dispatch_high_xor_literal || !decode_prior_dispatch_high ||
        !sequence.emit(*prior_dispatch_high_xor_literal) ||
        !sequence.emit(*decode_prior_dispatch_high) ||
        !append_moi_report_dispatch_id_word(words, point, bound_resources, record_compare_vgpr,
                                            /*high_word=*/true, arch,
                                            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
      errors.emplace_back("ConSan MOI first-light probe could not compare a dispatch slot");
      return std::nullopt;
    }
    const auto dispatch_high_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(record_compare_vgpr), record_value_high_vgpr, arch);
    if (!dispatch_high_mismatch) {
      return fail("ConSan MOI first-light probe could not compare a dispatch high word");
    }
    words.push_back(*dispatch_high_mismatch);
    if (!sequence.emit_branch(dispatch_retry_label, InstructionSequence::BranchKind::VccNonzero) ||
        !sequence.emit_branch(dispatch_bank_ready_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(dispatch_retry_label)) {
      return fail("ConSan MOI first-light dispatch-retry path is invalid");
    }

    const auto increment_dispatch_probe = instrumentation::build_v_add_u32_literal(
        bank_probe_count_vgpr, record_compare_vgpr, 1u, bank_probe_count_vgpr, arch);
    const auto materialize_dispatch_capacity = instrumentation::build_v_mov_b32_literal(
        record_compare_vgpr,
        std::min(layout.record_replay_dispatch_token_capacity, kConSanMoiRecordReplayProbeLimit),
        arch);
    const auto dispatch_probe_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
        vector_source_vgpr(record_compare_vgpr), bank_probe_count_vgpr, arch);
    // Mirror consan_moi_record_replay_advance_probe(): the incremented probe
    // count produces a triangular walk through the power-of-two directory.
    const auto increment_dispatch_bank = instrumentation::build_v_add_u32(
        dispatch_bank_vgpr, vector_source_vgpr(dispatch_bank_vgpr), bank_probe_count_vgpr, arch);
    const auto wrap_dispatch_bank = instrumentation::build_v_and_b32_literal(
        dispatch_bank_vgpr, layout.record_replay_dispatch_token_capacity - 1u, dispatch_bank_vgpr,
        arch);
    if (!increment_dispatch_probe || !materialize_dispatch_capacity || !dispatch_probe_in_range ||
        !increment_dispatch_bank || !wrap_dispatch_bank) {
      errors.emplace_back("ConSan MOI first-light probe could not advance its dispatch probe");
      return std::nullopt;
    }
    words.insert(words.end(), increment_dispatch_probe->begin(), increment_dispatch_probe->end());
    words.insert(words.end(), materialize_dispatch_capacity->begin(),
                 materialize_dispatch_capacity->end());
    words.push_back(*dispatch_probe_in_range);
    if (!sequence.emit_branch(dispatch_saturated_label, InstructionSequence::BranchKind::VccZero)) {
      return fail("ConSan MOI first-light probe could not bound its dispatch retry");
    }
    words.insert(words.end(), increment_dispatch_bank->begin(), increment_dispatch_bank->end());
    words.insert(words.end(), wrap_dispatch_bank->begin(), wrap_dispatch_bank->end());
    if (!append_claim_token(record_value_vgpr, record_value_high_vgpr, record_compare_vgpr) ||
        !sequence.emit_branch(dispatch_probe_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(dispatch_saturated_label) ||
        !append_atomic_or_u32_literal(words, base + offsetof(ConSanMoiReportHeader, flags),
                                      kConSanMoiReportFlagRecordReplayBankSaturated |
                                          kConSanMoiReportFlagRecordReplayDispatchBankSaturated,
                                      record_address_vgpr, arch) ||
        !sequence.emit_branch(restore_exec_label, InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(dispatch_bank_ready_label)) {
      errors.emplace_back("ConSan MOI first-light dispatch-directory path is invalid");
      return std::nullopt;
    }
  }

  const auto append_owner_bank_start = [&](uint32_t site_token) {
    if (!append_owner_id(owner_bank_vgpr))
      return false;
    const std::array<ConSanMoiWorkgroupSource, 3> sources = {
        workgroup_sources.x, workgroup_sources.y, workgroup_sources.z};
    for (const ConSanMoiWorkgroupSource &source : sources) {
      const auto mix = instrumentation::build_v_mul_lo_u32_literal(
          owner_bank_vgpr, record_value_high_vgpr, kConSanMoiRecordReplayIdentityHashMultiplier,
          owner_bank_vgpr, arch);
      if (!mix)
        return false;
      words.insert(words.end(), mix->begin(), mix->end());
      if (source.has_value()) {
        if (!consan_detail::append_workgroup_source_value(words, source, record_value_vgpr, arch)) {
          return false;
        }
      } else {
        words.push_back(
            build_v_mov_b32_e32(record_value_vgpr, scalar_positive_inline_u32(0), arch));
      }
      const auto combine = instrumentation::build_v_xor_b32(
          owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr), record_value_vgpr, arch);
      if (!combine)
        return false;
      words.push_back(*combine);
    }
    const auto mix_dispatch = instrumentation::build_v_mul_lo_u32_literal(
        owner_bank_vgpr, record_value_high_vgpr, kConSanMoiRecordReplayIdentityHashMultiplier,
        owner_bank_vgpr, arch);
    const auto combine_dispatch = instrumentation::build_v_xor_b32(
        owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr), dispatch_bank_vgpr, arch);
    const auto mix_address = instrumentation::build_v_mul_lo_u32_literal(
        owner_bank_vgpr, record_value_high_vgpr, kConSanMoiRecordReplayIdentityHashMultiplier,
        owner_bank_vgpr, arch);
    const auto combine_address =
        instrumentation::build_v_xor_b32(owner_bank_vgpr, address_key_sgpr, owner_bank_vgpr, arch);
    const auto mix_site = instrumentation::build_v_mul_lo_u32_literal(
        owner_bank_vgpr, record_value_high_vgpr, kConSanMoiRecordReplayIdentityHashMultiplier,
        owner_bank_vgpr, arch);
    const auto materialize_site =
        instrumentation::build_v_mov_b32_literal(record_value_vgpr, site_token, arch);
    const auto combine_site = instrumentation::build_v_xor_b32(
        owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr), record_value_vgpr, arch);
    const auto bound = instrumentation::build_v_and_b32_literal(
        owner_bank_vgpr, access_record_capacity - 1u, owner_bank_vgpr, arch);
    if (!mix_dispatch || !combine_dispatch || !mix_address || !combine_address || !mix_site ||
        !materialize_site || !combine_site || !bound)
      return false;
    words.insert(words.end(), mix_dispatch->begin(), mix_dispatch->end());
    words.push_back(*combine_dispatch);
    words.insert(words.end(), mix_address->begin(), mix_address->end());
    words.push_back(*combine_address);
    // Preserve the unbounded owner/workgroup/dispatch/address hash for the
    // access publication token. The bounded table slot additionally mixes the
    // static site before it advances independently while probing.
    words.push_back(
        build_v_mov_b32_e32(identity_claim_hash_vgpr, vector_source_vgpr(owner_bank_vgpr), arch));
    words.insert(words.end(), mix_site->begin(), mix_site->end());
    words.insert(words.end(), materialize_site->begin(), materialize_site->end());
    words.push_back(*combine_site);
    words.insert(words.end(), bound->begin(), bound->end());
    words.push_back(
        build_v_mov_b32_e32(bank_probe_count_vgpr, scalar_positive_inline_u32(0), arch));
    return true;
  };

  // Each automatic publication represents one exact address group. Direct
  // caller-owned layouts retain their historical one-record capacity and let
  // the per-slot atomic claim elect the first lane that executes the site.
  for (size_t range_index = 0; range_index < access_ranges.size(); ++range_index) {
    const auto publication_done_label = sequence.make_label();
    const auto token_ready_label = sequence.make_label();
    const auto occupied_label = sequence.make_label();
    const auto owner_probe_label = sequence.make_label();
    const auto owner_retry_label = sequence.make_label();
    const auto saturation_label = sequence.make_label();
    const auto owner_saturation_label = sequence.make_label();
    const auto publication_observe_label = sequence.make_label();
    const auto publication_incomplete_label = sequence.make_label();
    const ConSanAccessRange &range = access_ranges[range_index];
    const uint32_t site_token =
        automatic_banked_capture ? logical_range_index + static_cast<uint32_t>(range_index) : 0u;
    const uint64_t access_record_base =
        automatic_banked_capture ? base + layout.access_records_offset
                                 : base + layout.access_records_offset +
                                       (static_cast<uint64_t>(record_index) + range_index) *
                                           sizeof(ConSanMoiAccessRecord);

    if (automatic_banked_capture) {
      if (!append_owner_bank_start(site_token) || !sequence.bind(owner_probe_label)) {
        errors.emplace_back(
            "ConSan MOI first-light probe could not initialize its access-identity probe");
        return std::nullopt;
      }
    }

    // Claim one bounded slot with a reversible encoding of the full hardware
    // dispatch ID. The report-wide table keeps the dispatch bank stable across
    // all sites; this per-site probe resolves distinct workgroup/wave owners
    // within that bank. access_kind is committed atomically only after every
    // payload store has drained.
    if (!materialize_banked_record_address(access_record_base +
                                           offsetof(ConSanMoiAccessRecord, claim_token)) ||
        !(automatic_banked_capture
              ? append_access_claim_token(record_value_vgpr, record_value_high_vgpr,
                                          record_compare_vgpr, site_token)
              : append_claim_token(record_value_vgpr, record_value_high_vgpr,
                                   record_compare_vgpr))) {
      errors.emplace_back("ConSan MOI first-light probe could not materialize its bank claim");
      return std::nullopt;
    }
    const auto token_low_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_vgpr, arch);
    const auto token_high_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_high_vgpr, arch);
    if (!token_low_nonzero || !token_high_nonzero) {
      errors.emplace_back("ConSan MOI first-light probe could not validate its bank claim");
      return std::nullopt;
    }
    words.push_back(*token_low_nonzero);
    if (!sequence.emit_branch(token_ready_label, InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not validate its access token");
    }
    words.push_back(*token_high_nonzero);
    if (!sequence.emit_branch(token_ready_label, InstructionSequence::BranchKind::VccNonzero) ||
        !sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(token_ready_label)) {
      return fail("ConSan MOI first-light access-token path is invalid");
    }
    words.push_back(build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch));
    words.push_back(
        build_v_mov_b32_e32(record_compare_high_vgpr, scalar_positive_inline_u32(0), arch));
    const auto claim = instrumentation::build_flat_atomic_cmpswap_b64(
        record_address_vgpr, record_value_vgpr, record_value_vgpr,
        /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    if (!claim) {
      errors.emplace_back("ConSan MOI first-light probe could not encode its publication claim");
      return std::nullopt;
    }
    words.insert(words.end(), claim->begin(), claim->end());
    if (!append_moi_global_atomic_wait(words, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not wait for its publication claim");
      return std::nullopt;
    }
    const auto prior_low_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_vgpr, arch);
    const auto prior_high_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_high_vgpr, arch);
    if (!prior_low_nonzero || !prior_high_nonzero) {
      errors.emplace_back("ConSan MOI first-light probe could not test its publication claim");
      return std::nullopt;
    }
    words.push_back(*prior_low_nonzero);
    if (!sequence.emit_branch(occupied_label, InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not inspect its access claim low word");
    }
    words.push_back(*prior_high_nonzero);
    if (!sequence.emit_branch(occupied_label, InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not inspect its access claim high word");
    }
    const uint32_t visible_record_count =
        automatic_banked_capture ? access_record_capacity : record_count;
    if (!append_store_u32_literal(words,
                                  base + offsetof(ConSanMoiReportHeader, access_record_count),
                                  visible_record_count, scratch_vgpr, arch) ||
        !append_atomic_fetch_add_one_u32(words,
                                         base + offsetof(ConSanMoiReportHeader, event_counter),
                                         record_compare_vgpr, record_address_vgpr, arch) ||
        !materialize_banked_record_address(access_record_base)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode record stores");
      return std::nullopt;
    }
    // Recover an overlapping guest address only after the final bank address
    // is materialized, so the recorded LDS range cannot accidentally become a
    // transient bank-selection value.
    if (!reload_spilled_lds_byte_offset()) {
      errors.emplace_back("ConSan MOI first-light probe could not recover its spilled LDS address");
      return std::nullopt;
    }
    if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, event_index), record_compare_vgpr) ||
        !append_store_moi_report_dispatch_id_pair(
            record, point, bound_resources, offsetof(ConSanMoiAccessRecord, generation), arch,
            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
        !record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_x),
                                workgroup_sources.x) ||
        !record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_y),
                                workgroup_sources.y) ||
        !record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_z),
                                workgroup_sources.z) ||
        !record.store_sgpr(offsetof(ConSanMoiAccessRecord, lane_mask),
                           automatic_banked_capture ? address_group_exec_sgpr
                                                    : *point.moi_exec_save_sgpr) ||
        !record.store_sgpr(offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
                           automatic_banked_capture
                               ? static_cast<uint16_t>(address_group_exec_sgpr + 1u)
                               : static_cast<uint16_t>(*point.moi_exec_save_sgpr + 1u)) ||
        (point.moi_owner_vgpr &&
         !record.store_vgpr(offsetof(ConSanMoiAccessRecord, wave_id), *point.moi_owner_vgpr)) ||
        (point.moi_persistent_sgprs.owner &&
         !record.store_sgpr(offsetof(ConSanMoiAccessRecord, wave_id),
                            *point.moi_persistent_sgprs.owner)) ||
        (point.moi_epoch_vgpr &&
         !record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), *point.moi_epoch_vgpr)) ||
        !record.store_literal(offsetof(ConSanMoiAccessRecord, instruction_offset),
                              static_cast<uint32_t>(candidate.anchor())) ||
        !record.store_literal(offsetof(ConSanMoiAccessRecord, site_token), site_token) ||
        !record.store_literal(
            offsetof(ConSanMoiAccessRecord, flags),
            automatic_banked_capture ? kConSanMoiAccessRecordFlagExactAddressGroupMask : 0u)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode record stores");
      return std::nullopt;
    }
    if (derived_owner_vgpr) {
      words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
      if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, wave_id), *derived_owner_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not encode derived owner");
        return std::nullopt;
      }
    }
    if (private_epoch_offset) {
      const auto load_epoch =
          instrumentation::build_private_load_b32(record_value_vgpr, *private_epoch_offset, arch);
      const auto wait_epoch = instrumentation::build_s_wait_private_load0(arch);
      if (!load_epoch || !wait_epoch) {
        errors.emplace_back("ConSan MOI first-light probe could not load private epoch state");
        return std::nullopt;
      }
      words.insert(words.end(), load_epoch->begin(), load_epoch->end());
      words.push_back(*wait_epoch);
      if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), record_value_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not store private epoch state");
        return std::nullopt;
      }
    } else if (point.moi_persistent_sgprs.epoch) {
      words.push_back(
          build_v_mov_b32_e32(record_value_vgpr, *point.moi_persistent_sgprs.epoch, arch));
      if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), record_value_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not store scalar epoch state");
        return std::nullopt;
      }
    }

    const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
        append_effective_range_offset(range, record_value_vgpr);
    if (!effective_lds_byte_offset_vgpr) {
      errors.emplace_back("ConSan MOI first-light probe could not encode LDS byte offset");
      return std::nullopt;
    }
    const ConSanMoiLdsCellRange static_range =
        consan_moi_lds_cell_range_for_bytes(candidate.lowering_offset(range), range.byte_width);
    if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                           *effective_lds_byte_offset_vgpr)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode range offset");
      return std::nullopt;
    }
    const auto start_cell = instrumentation::build_v_lshrrev_b32(
        record_value_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
        *effective_lds_byte_offset_vgpr, arch);
    if (!start_cell) {
      errors.emplace_back("ConSan MOI first-light probe could not encode start cell");
      return std::nullopt;
    }
    words.push_back(*start_cell);
    if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, start_cell), record_value_vgpr) ||
        !record.store_literal(offsetof(ConSanMoiAccessRecord, lds_byte_count), range.byte_width) ||
        !record.store_literal(offsetof(ConSanMoiAccessRecord, cell_count),
                              static_range.cell_count)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode range fields");
      return std::nullopt;
    }
    const auto payload_wait = instrumentation::build_s_wait_global_store0(arch);
    const auto commit_address = instrumentation::build_v_add_u64_signed_i24(
        record_address_vgpr, offsetof(ConSanMoiAccessRecord, access_kind), arch);
    if (!payload_wait || !commit_address) {
      errors.emplace_back("ConSan MOI first-light probe could not drain record stores");
      return std::nullopt;
    }
    words.push_back(*payload_wait);
    words.insert(words.end(), commit_address->begin(), commit_address->end());
    const auto commit_value = instrumentation::build_v_mov_b32_literal(
        record_value_vgpr, static_cast<uint32_t>(kind), arch);
    const auto commit_expected = instrumentation::build_v_mov_b32_literal(
        record_value_high_vgpr, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty), arch);
    const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
        record_address_vgpr, record_value_vgpr, record_value_vgpr,
        /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    if (!commit_value || !commit_expected || !commit) {
      errors.emplace_back("ConSan MOI first-light probe could not commit its publication");
      return std::nullopt;
    }
    words.insert(words.end(), commit_value->begin(), commit_value->end());
    words.insert(words.end(), commit_expected->begin(), commit_expected->end());
    words.insert(words.end(), commit->begin(), commit->end());
    if (!append_moi_global_atomic_wait(words, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not wait for its publication commit");
      return std::nullopt;
    }
    const auto commit_collision = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty)),
        record_value_vgpr, arch);
    if (!commit_collision) {
      errors.emplace_back("ConSan MOI first-light probe could not validate its publication commit");
      return std::nullopt;
    }
    words.push_back(*commit_collision);
    if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero) ||
        !sequence.emit_branch(publication_done_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(occupied_label)) {
      return fail("ConSan MOI first-light publication-claim path is invalid");
    }

    if (automatic_banked_capture) {
      // Preserve the returned token in the now-dead address pair, recompute
      // this execution identity's fingerprint, and bypass unrelated
      // in-flight claims without waiting for their publishers.
      words.push_back(
          build_v_mov_b32_e32(record_address_vgpr, vector_source_vgpr(record_value_vgpr), arch));
      words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(record_address_vgpr + 1u),
                                          vector_source_vgpr(record_value_high_vgpr), arch));
      if (!append_access_claim_token(record_compare_vgpr, record_compare_high_vgpr,
                                     record_value_vgpr, site_token)) {
        errors.emplace_back("ConSan MOI first-light probe could not rebuild its identity claim");
        return std::nullopt;
      }
      const auto low_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), record_address_vgpr, arch);
      const auto high_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_high_vgpr),
          static_cast<uint16_t>(record_address_vgpr + 1u), arch);
      if (!low_mismatch || !high_mismatch) {
        errors.emplace_back("ConSan MOI first-light probe could not compare its identity claim");
        return std::nullopt;
      }
      words.push_back(*low_mismatch);
      if (!sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero)) {
        return fail("ConSan MOI first-light probe could not retry an access identity");
      }
      words.push_back(*high_mismatch);
      if (!sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero) ||
          !materialize_banked_record_address(access_record_base +
                                             offsetof(ConSanMoiAccessRecord, access_kind))) {
        return fail("ConSan MOI first-light access-identity retry path is invalid");
      }
    } else {
      // Direct capture retains the historical reversible dispatch token.
      const auto low_xor_literal = instrumentation::build_v_mov_b32_literal(
          record_compare_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask),
          arch);
      const auto low_xor = instrumentation::build_v_xor_b32(
          record_value_vgpr, vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
      if (!low_xor_literal || !low_xor) {
        errors.emplace_back("ConSan MOI first-light probe could not compare its dispatch claim");
        return std::nullopt;
      }
      words.insert(words.end(), low_xor_literal->begin(), low_xor_literal->end());
      words.push_back(*low_xor);
      if (!append_moi_report_dispatch_id_word(words, point, bound_resources, record_compare_vgpr,
                                              /*high_word=*/false, arch,
                                              ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
        return fail("ConSan MOI first-light probe could not materialize a dispatch low word");
      }
      const auto low_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
      if (!low_mismatch) {
        return fail("ConSan MOI first-light probe could not compare a direct dispatch low word");
      }
      words.push_back(*low_mismatch);
      if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero)) {
        return fail("ConSan MOI first-light probe could not reject a dispatch low mismatch");
      }

      const auto high_xor_literal = instrumentation::build_v_mov_b32_literal(
          record_compare_vgpr,
          static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u), arch);
      const auto high_xor = instrumentation::build_v_xor_b32(
          record_value_high_vgpr, vector_source_vgpr(record_compare_vgpr), record_value_high_vgpr,
          arch);
      if (!high_xor_literal || !high_xor) {
        errors.emplace_back("ConSan MOI first-light probe could not compare its dispatch claim");
        return std::nullopt;
      }
      words.insert(words.end(), high_xor_literal->begin(), high_xor_literal->end());
      words.push_back(*high_xor);
      if (!append_moi_report_dispatch_id_word(words, point, bound_resources, record_compare_vgpr,
                                              /*high_word=*/true, arch,
                                              ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
        return fail("ConSan MOI first-light probe could not materialize a dispatch high word");
      }
      const auto high_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), record_value_high_vgpr, arch);
      const auto access_kind_address = instrumentation::build_v_add_u64_signed_i24(
          record_address_vgpr, offsetof(ConSanMoiAccessRecord, access_kind), arch);
      if (!high_mismatch || !access_kind_address) {
        errors.emplace_back("ConSan MOI first-light probe could not compare its dispatch claim");
        return std::nullopt;
      }
      words.push_back(*high_mismatch);
      if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero)) {
        return fail("ConSan MOI first-light probe could not reject a dispatch high mismatch");
      }
      words.insert(words.end(), access_kind_address->begin(), access_kind_address->end());
    }

    // Use a no-op atomic compare-and-swap as an acquire-style read of the
    // access_kind commit. One lane publishes for each
    // dispatch/workgroup/wave/site/address-group identity, and a wave cannot
    // publish that same first-light identity concurrently with itself.
    // Therefore an incomplete automatic claim with the same compact token
    // belongs to a distinct identity whose fingerprint collided; probe another
    // slot instead of waiting for a wave that may not be scheduled while this
    // wave is resident. A committed record is qualified against the complete
    // identity below. The legacy direct layout has nowhere else to probe and
    // retains its fail-closed result.
    if (!sequence.bind(publication_observe_label)) {
      return fail("ConSan MOI first-light probe could not bind its publication observer");
    }
    words.push_back(build_v_mov_b32_e32(record_value_vgpr, scalar_positive_inline_u32(0), arch));
    words.push_back(
        build_v_mov_b32_e32(record_value_high_vgpr, scalar_positive_inline_u32(0), arch));
    const auto read_commit = instrumentation::build_flat_atomic_cmpswap_b32(
        record_address_vgpr, record_value_vgpr, record_value_vgpr,
        /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    const auto publication_incomplete = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty)),
        record_value_vgpr, arch);
    const auto publication_kind_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(kind)), record_value_vgpr, arch);
    if (!read_commit || !publication_incomplete || !publication_kind_mismatch) {
      errors.emplace_back("ConSan MOI first-light probe could not observe its publication commit");
      return std::nullopt;
    }
    words.insert(words.end(), read_commit->begin(), read_commit->end());
    if (!append_moi_global_atomic_wait(words, arch)) {
      errors.emplace_back(
          "ConSan MOI first-light probe could not wait for its publication observation");
      return std::nullopt;
    }
    words.push_back(*publication_incomplete);
    if (!sequence.emit_branch(automatic_banked_capture ? owner_retry_label
                                                       : publication_incomplete_label,
                              InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not handle incomplete publication");
    }
    words.push_back(*publication_kind_mismatch);
    const auto record_base_address = instrumentation::build_v_add_u64_signed_i24(
        record_address_vgpr, -static_cast<int32_t>(offsetof(ConSanMoiAccessRecord, access_kind)),
        arch);
    if (!record_base_address ||
        !sequence.emit_branch(automatic_banked_capture ? owner_retry_label : saturation_label,
                              InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not reject a publication-kind mismatch");
    }
    words.insert(words.end(), record_base_address->begin(), record_base_address->end());

    if (automatic_banked_capture) {
      const auto reject_dispatch_mismatch = [&](uint32_t offset, bool high_word) {
        if (!record.load(offset, record_value_vgpr) ||
            !append_moi_report_dispatch_id_word(
                words, point, bound_resources, record_compare_vgpr, high_word, arch,
                ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
          return false;
        }
        const auto mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
            vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
        if (!mismatch)
          return false;
        words.push_back(*mismatch);
        return sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero);
      };
      if (!reject_dispatch_mismatch(offsetof(ConSanMoiAccessRecord, generation),
                                    /*high_word=*/false) ||
          !reject_dispatch_mismatch(offsetof(ConSanMoiAccessRecord, generation) + sizeof(uint32_t),
                                    /*high_word=*/true)) {
        errors.emplace_back("ConSan MOI first-light probe could not qualify its retained dispatch");
        return std::nullopt;
      }
    }

    if (!record.load(offsetof(ConSanMoiAccessRecord, site_token), record_value_vgpr)) {
      return fail("ConSan MOI first-light probe could not load its retained site");
    }
    const auto expected_site =
        instrumentation::build_v_mov_b32_literal(record_compare_vgpr, site_token, arch);
    const auto site_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
    if (!expected_site || !site_mismatch) {
      errors.emplace_back("ConSan MOI first-light probe could not qualify its retained site");
      return std::nullopt;
    }
    words.insert(words.end(), expected_site->begin(), expected_site->end());
    words.push_back(*site_mismatch);
    if (!sequence.emit_branch(automatic_banked_capture ? owner_retry_label : saturation_label,
                              InstructionSequence::BranchKind::VccNonzero)) {
      return fail("ConSan MOI first-light probe could not reject a retained-site mismatch");
    }

    const auto reject_workgroup_mismatch = [&](uint32_t offset,
                                               const ConSanMoiWorkgroupSource &source) {
      if (!record.load(offset, record_value_vgpr))
        return false;
      if (source.has_value()) {
        if (!consan_detail::append_workgroup_source_value(words, source, record_compare_vgpr, arch))
          return false;
      } else {
        words.push_back(
            build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch));
      }
      const auto mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
      if (!mismatch)
        return false;
      words.push_back(*mismatch);
      return sequence.emit_branch(automatic_banked_capture ? owner_retry_label : saturation_label,
                                  InstructionSequence::BranchKind::VccNonzero);
    };
    if (!reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_x),
                                   workgroup_sources.x) ||
        !reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_y),
                                   workgroup_sources.y) ||
        !reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_z),
                                   workgroup_sources.z)) {
      errors.emplace_back("ConSan MOI first-light probe could not qualify its retained workgroup");
      return std::nullopt;
    }

    // Direct first-light capture intentionally coalesces waves within one
    // dispatch/workgroup. The automatic table has spare slots and retains
    // wave identity explicitly so replay can cover every execution owner.
    if (automatic_banked_capture) {
      if (!record.load(offsetof(ConSanMoiAccessRecord, wave_id), record_value_vgpr) ||
          !append_owner_id(record_compare_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not qualify its retained owner");
        return std::nullopt;
      }
      const auto owner_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
      if (!owner_mismatch) {
        errors.emplace_back("ConSan MOI first-light probe could not compare its retained owner");
        return std::nullopt;
      }
      words.push_back(*owner_mismatch);
      if (!sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero)) {
        errors.emplace_back("ConSan MOI first-light probe could not qualify its retained owner");
        return std::nullopt;
      }

      if (!record.load(offsetof(ConSanMoiAccessRecord, lds_byte_offset), record_value_vgpr)) {
        errors.emplace_back(
            "ConSan MOI first-light probe could not load its retained address group");
        return std::nullopt;
      }
      words.push_back(build_v_mov_b32_e32(record_compare_vgpr, address_key_sgpr, arch));
      if (candidate.lowering_offset(range) != 0u) {
        const auto materialize_static_offset = instrumentation::build_v_mov_b32_literal(
            record_compare_high_vgpr, candidate.lowering_offset(range), arch);
        const auto add_static_offset = instrumentation::build_v_add_u32(
            record_compare_vgpr, vector_source_vgpr(record_compare_vgpr), record_compare_high_vgpr,
            arch);
        if (!materialize_static_offset || !add_static_offset) {
          errors.emplace_back(
              "ConSan MOI first-light probe could not qualify its retained address group");
          return std::nullopt;
        }
        words.insert(words.end(), materialize_static_offset->begin(),
                     materialize_static_offset->end());
        words.insert(words.end(), add_static_offset->begin(), add_static_offset->end());
      }
      const auto address_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
      if (!address_mismatch) {
        errors.emplace_back(
            "ConSan MOI first-light probe could not compare its retained address group");
        return std::nullopt;
      }
      words.push_back(*address_mismatch);
      if (!sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero)) {
        errors.emplace_back(
            "ConSan MOI first-light probe could not qualify its retained address group");
        return std::nullopt;
      }
    }
    if (!sequence.emit_branch(publication_done_label,
                              InstructionSequence::BranchKind::Unconditional)) {
      errors.emplace_back("ConSan MOI first-light probe could not qualify its retained identity");
      return std::nullopt;
    }

    if (automatic_banked_capture) {
      if (!sequence.bind(owner_retry_label)) {
        return fail("ConSan MOI first-light probe could not bind its access-identity retry");
      }
      const auto increment_owner_probe = instrumentation::build_v_add_u32_literal(
          bank_probe_count_vgpr, record_compare_vgpr, 1u, bank_probe_count_vgpr, arch);
      const auto materialize_identity_capacity = instrumentation::build_v_mov_b32_literal(
          record_compare_vgpr, std::min(access_record_capacity, kConSanMoiRecordReplayProbeLimit),
          arch);
      const auto owner_probe_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), bank_probe_count_vgpr, arch);
      // Mirror consan_moi_record_replay_advance_probe(): triangular probing is
      // a permutation of every power-of-two table and avoids the primary
      // clustering of a linear walk at higher load.
      const auto increment_owner_bank = instrumentation::build_v_add_u32(
          owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr), bank_probe_count_vgpr, arch);
      const auto wrap_owner_bank = instrumentation::build_v_and_b32_literal(
          owner_bank_vgpr, access_record_capacity - 1u, owner_bank_vgpr, arch);
      if (!increment_owner_probe || !materialize_identity_capacity || !owner_probe_in_range ||
          !increment_owner_bank || !wrap_owner_bank) {
        errors.emplace_back(
            "ConSan MOI first-light probe could not advance its access-identity probe");
        return std::nullopt;
      }
      words.insert(words.end(), increment_owner_probe->begin(), increment_owner_probe->end());
      words.insert(words.end(), materialize_identity_capacity->begin(),
                   materialize_identity_capacity->end());
      words.push_back(*owner_probe_in_range);
      if (!sequence.emit_branch(owner_saturation_label, InstructionSequence::BranchKind::VccZero)) {
        return fail("ConSan MOI first-light probe could not bound its access-identity retry");
      }
      words.insert(words.end(), increment_owner_bank->begin(), increment_owner_bank->end());
      words.insert(words.end(), wrap_owner_bank->begin(), wrap_owner_bank->end());
      if (!sequence.emit_branch(owner_probe_label,
                                InstructionSequence::BranchKind::Unconditional) ||
          !sequence.bind(owner_saturation_label) ||
          !append_atomic_or_u32_literal(words, base + offsetof(ConSanMoiReportHeader, flags),
                                        kConSanMoiReportFlagRecordReplayBankSaturated |
                                            kConSanMoiReportFlagRecordReplayOwnerBankSaturated,
                                        record_address_vgpr, arch) ||
          !sequence.emit_branch(publication_done_label,
                                InstructionSequence::BranchKind::Unconditional)) {
        errors.emplace_back("ConSan MOI first-light access-identity path is invalid");
        return std::nullopt;
      }
    } else {
      if (!sequence.bind(publication_incomplete_label) ||
          !append_atomic_or_u32_literal(words, base + offsetof(ConSanMoiReportHeader, flags),
                                        kConSanMoiReportFlagRecordReplayBankSaturated |
                                            kConSanMoiReportFlagRecordReplayPublicationIncomplete,
                                        record_address_vgpr, arch) ||
          !sequence.emit_branch(publication_done_label,
                                InstructionSequence::BranchKind::Unconditional)) {
        errors.emplace_back("ConSan MOI first-light direct publication path is invalid");
        return std::nullopt;
      }
    }

    if (!sequence.bind(saturation_label) ||
        !append_atomic_or_u32_literal(words, base + offsetof(ConSanMoiReportHeader, flags),
                                      kConSanMoiReportFlagRecordReplayBankSaturated,
                                      record_address_vgpr, arch) ||
        !sequence.bind(publication_done_label)) {
      errors.emplace_back("ConSan MOI first-light publication fast path is invalid");
      return std::nullopt;
    }
  }

  if (automatic_banked_capture) {
    if (!address_group_loop) {
      errors.emplace_back("ConSan MOI first-light probe lost its address-group loop");
      return std::nullopt;
    }
    const auto remove_group = instrumentation::build_s_xor_b64(
        *point.moi_exec_save_sgpr, *point.moi_exec_save_sgpr, address_group_exec_sgpr, arch);
    const auto select_remaining =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, *point.moi_exec_save_sgpr, arch);
    if (!remove_group || !select_remaining) {
      errors.emplace_back("ConSan MOI first-light probe could not advance its address-group loop");
      return std::nullopt;
    }
    words.push_back(*remove_group);
    // Keep the scalar-mask dependency explicit before installing the next
    // EXEC. This matches the proven Inline Shadow traversal and prevents the
    // final divergent address group from being skipped on live hardware.
    words.push_back(build_s_nop(0, arch));
    words.push_back(build_s_nop(0, arch));
    words.push_back(*select_remaining);
    const int64_t loop_delta =
        static_cast<int64_t>(*address_group_loop) - static_cast<int64_t>(words.size() + 1u);
    if (loop_delta < std::numeric_limits<int16_t>::min() ||
        loop_delta > std::numeric_limits<int16_t>::max()) {
      errors.emplace_back("ConSan MOI first-light address-group loop is out of branch reach");
      return std::nullopt;
    }
    const auto continue_loop =
        instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(loop_delta), arch);
    if (!continue_loop) {
      errors.emplace_back(
          "ConSan MOI first-light probe could not branch to its next address group");
      return std::nullopt;
    }
    words.push_back(*continue_loop);
  }

  const uint16_t restore_exec_sgpr =
      automatic_banked_capture ? original_exec_sgpr : *point.moi_exec_save_sgpr;
  const auto restore_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, restore_exec_sgpr, arch);
  if (!restore_exec) {
    errors.emplace_back("ConSan MOI first-light probe could not restore EXEC");
    return std::nullopt;
  }
  if (!sequence.bind(restore_exec_label)) {
    return fail("ConSan MOI first-light probe could not bind its EXEC restore");
  }
  words.push_back(*restore_exec);
  if (!append_restore_moi_special_state(words, moi_special_state_sgprs(request, point), arch)) {
    errors.emplace_back("ConSan MOI first-light probe could not restore VCC/SCC");
    return std::nullopt;
  }

  // Static first-light records do not consume the guest result. Publish them
  // before a load so that the record scratch window may overlap its destination
  // VGPRs. When the load also overwrites its address VGPR, the saved address
  // above remains outside that window until the displaced instruction executes.
  if (!request.moi_dynamic_access_records && !append_guest_access())
    return std::nullopt;

  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI first-light local branch is out of reach");
    return std::nullopt;
  }

  return words;
}

[[nodiscard]] bool append_inline_shadow_owner_field(
    std::vector<uint32_t> &words, const ConSanRequest &request,
    const ConSanMoiOperatingPoint &point, uint16_t low_vgpr, uint16_t tmp_vgpr,
    uint16_t owner_backup_vgpr, rj_code_arch_t arch,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    std::vector<std::string> &errors) {
  if (point.moi_owner_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *point.moi_owner_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (point.moi_persistent_sgprs.owner) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *point.moi_persistent_sgprs.owner, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (!point.automatic_moi_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no persistent owner representation");
    return false;
  }

  switch (request.moi_owner_source) {
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
    if (!point.moi_owner_sgpr) {
      errors.emplace_back("ConSan MOI private owner hw_id source requires "
                          "RJ_CONSAN_MOI_OWNER_SGPR");
      return false;
    }
    if (point.automatic_moi_owner_sgpr) {
      const auto save = instrumentation::build_v_writelane_b32(owner_backup_vgpr,
                                                               *point.moi_owner_sgpr, 0u, arch);
      if (!save) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not save its borrowed owner scalar");
        return false;
      }
      words.insert(words.end(), save->begin(), save->end());
    }
    const ConSanTargetProfile *target = consan_target_profile(arch);
    const consan_detail::MoiResidentWaveOwnerRequest owner_request{
        .destination_sgpr = *point.moi_owner_sgpr,
        .one_based = true,
    };
    if (target == nullptr ||
        !consan_detail::append_moi_resident_wave_owner(words, owner_request, *target)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not encode its resident-wave owner");
      return false;
    }
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *point.moi_owner_sgpr, arch));
    if (point.automatic_moi_owner_sgpr) {
      const auto restore =
          instrumentation::build_v_readlane_b32(*point.moi_owner_sgpr, owner_backup_vgpr, 0u, arch);
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
                                                    const ConSanMoiOperatingPoint &point,
                                                    std::optional<uint32_t> private_epoch_offset,
                                                    uint16_t low_vgpr, uint16_t tmp_vgpr,
                                                    rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  if (point.moi_epoch_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *point.moi_epoch_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (point.moi_persistent_sgprs.epoch) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *point.moi_persistent_sgprs.epoch, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (!point.automatic_moi_private_epoch || !private_epoch_offset) {
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

[[nodiscard]] bool
append_inline_workgroup_key(std::vector<uint32_t> &words, const ConSanMoiWorkgroupSources &sources,
                            const MoiWorkgroupKeyRegisterPlan &registers, uint16_t key_vgpr,
                            uint16_t coordinate_vgpr, uint16_t value_vgpr,
                            uint16_t original_exec_save_offset, rj_code_arch_t arch);

[[nodiscard]] bool append_inline_acquired_token_slot_address(
    std::vector<uint32_t> &words, uint64_t table_base, uint32_t table_capacity,
    uint16_t workgroup_key_vgpr, uint16_t consumer_owner_vgpr, uint16_t producer_owner_vgpr,
    uint16_t consumer_epoch_vgpr, uint16_t slot_address_vgpr, uint16_t hash_vgpr,
    uint16_t temporary_vgpr, bool release_sequence, rj_code_arch_t arch);

[[nodiscard]] bool append_inline_atomic_slot_address(std::vector<uint32_t> &words,
                                                     uint64_t table_base, uint32_t table_capacity,
                                                     uint16_t atomic_address_vgpr,
                                                     uint16_t scratch_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool
append_inline_atomic_snapshot_address(std::vector<uint32_t> &words, uint64_t table_base,
                                      uint32_t table_capacity, uint16_t atomic_address_vgpr,
                                      uint16_t snapshot_address_vgpr, uint16_t hash_vgpr,
                                      uint16_t temporary_vgpr, rj_code_arch_t arch);

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
