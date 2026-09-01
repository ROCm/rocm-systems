// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/consan/consan_target_lds_ops.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

bool consan_detail::range_overlaps(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                   uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return static_cast<uint32_t>(lhs_base) < rhs_end && static_cast<uint32_t>(rhs_base) < lhs_end;
}

bool consan_detail::reject_optional_scratch_range_overlap(std::optional<uint16_t> value,
                                                          uint16_t scratch_vgpr,
                                                          uint16_t scratch_count,
                                                          std::string_view value_name,
                                                          std::vector<std::string> &errors) {
  const bool overlaps = value && scratch_count != 0u && *value >= scratch_vgpr &&
                        *value < static_cast<uint16_t>(scratch_vgpr + scratch_count);
  if (!overlaps)
    return false;
  errors.emplace_back(std::string("ConSan MOI scratch VGPRs overlap ") + std::string(value_name) +
                      " VGPR");
  return true;
}

bool consan_detail::reject_atomic_candidate_scratch_overlap(const ConSanAtomicLoweringForm &form,
                                                            uint16_t scratch_vgpr,
                                                            uint16_t scratch_vgpr_count,
                                                            std::vector<std::string> &errors) {
  if (range_overlaps(form.address_vgpr, form.address_vgpr_count, scratch_vgpr,
                     scratch_vgpr_count)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic address VGPRs");
    return true;
  }
  if (range_overlaps(form.data_vgpr, form.data_register_count, scratch_vgpr, scratch_vgpr_count)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic data VGPR");
    return true;
  }
  if (form.destination_vgpr && form.destination_register_count != 0u &&
      range_overlaps(*form.destination_vgpr, form.destination_register_count, scratch_vgpr,
                     scratch_vgpr_count)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic destination VGPR");
    return true;
  }
  return false;
}

bool consan_detail::has_recent_saveexec(std::span<const uint8_t> bytes,
                                        const ConSanMoiCandidate &candidate) {
  constexpr uint32_t kSop1PrefixMask = 0xFF800000u;
  constexpr uint32_t kSop1OpMask = 0x0000FF00u;
  const auto is_saveexec = [&](uint32_t word) {
    const uint32_t op = (word & kSop1OpMask) >> 8u;
    return (word & kSop1PrefixMask) == (kSop1EncodingPrefix << 23u) && op >= 0x20u && op <= 0x33u;
  };
  constexpr uint64_t kLookbackDwords = 3u;
  for (uint64_t dword = 1u; dword <= kLookbackDwords; ++dword) {
    const uint64_t byte_distance = dword * sizeof(uint32_t);
    if (candidate.file_offset < byte_distance)
      break;
    const uint64_t offset = candidate.file_offset - byte_distance;
    if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
      continue;
    uint32_t word = 0u;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if (is_saveexec(word))
      return true;
  }
  return false;
}

std::optional<uint16_t> ConSanMoiWorkgroupSource::operand() const {
  if (!has_value() || private_offset)
    return std::nullopt;
  return scalar_src ? *scalar_src : vector_source_vgpr(*vector_src);
}

bool consan_detail::append_moi_resident_wave_owner(std::vector<uint32_t> &words,
                                                   const MoiResidentWaveOwnerRequest &request,
                                                   const ConSanTargetProfile &target) {
  const ConSanResidentWaveIdentityEncoding &encoding = target.resident_wave_identity;
  const auto hwreg = build_hwreg_imm(encoding.hwreg_id, encoding.bit_offset, encoding.bit_width);
  const auto read =
      hwreg ? instrumentation::build_s_getreg_b32(request.destination_sgpr, *hwreg, target.arch)
            : std::nullopt;
  const auto delay = instrumentation::build_salu_dependency_delay(target.arch);
  const auto wait = instrumentation::build_salu_to_valu_dependency_wait(target.arch);
  if (!read || !delay || !wait)
    return false;

  words.push_back(*read);
  words.push_back(*delay);
  if (request.one_based) {
    words.push_back(build_s_add_u32(request.destination_sgpr, request.destination_sgpr,
                                    scalar_positive_inline_u32(1), target.arch));
    words.push_back(*delay);
  }
  words.push_back(*wait);
  return true;
}

bool consan_detail::moi_guest_access_relocation_requires_adjusted_address(
    const ConSanMoiCandidate &candidate, const ConSanTargetProfile &target) {
  return target.requires_split_two_address_lds_relocation && candidate.is_native_two_range() &&
         candidate.encoded_offset_scale_bytes() > 8u;
}

std::optional<std::vector<uint32_t>> consan_detail::build_moi_relocated_guest_access_words(
    const MoiGuestAccessRelocationRequest &request, std::vector<std::string> &errors) {
  if (request.candidate == nullptr || request.target == nullptr) {
    errors.emplace_back("ConSan MOI guest relocation requires a candidate and target profile");
    return std::nullopt;
  }
  const ConSanMoiCandidate &candidate = *request.candidate;
  const ConSanTargetProfile &target = *request.target;
  const auto copy_original = [&]() -> std::optional<std::vector<uint32_t>> {
    if (candidate.size() == 0u || candidate.size() % sizeof(uint32_t) != 0u ||
        candidate.file_offset > request.image.size() ||
        candidate.size() > request.image.size() - candidate.file_offset) {
      errors.emplace_back("ConSan MOI relocated guest access exceeds the code object");
      return std::nullopt;
    }
    std::vector<uint32_t> words(candidate.size() / sizeof(uint32_t));
    std::memcpy(words.data(), request.image.data() + candidate.file_offset, candidate.size());
    return words;
  };

  if (!target.requires_split_two_address_lds_relocation || !candidate.is_native_two_range())
    return copy_original();

  const auto &ranges = candidate.ranges;
  const uint16_t element_dwords = static_cast<uint16_t>(candidate.width_bits() / 32u);
  if (ranges.size() != 2u ||
      (candidate.kind != ConSanLdsAccessKind::Read &&
       candidate.kind != ConSanLdsAccessKind::Write) ||
      (element_dwords != 1u && element_dwords != 2u) || request.replay_address_vgpr > 255u) {
    errors.emplace_back("ConSan MOI could not normalize a split two-address guest access");
    return std::nullopt;
  }

  const bool load = candidate.kind == ConSanLdsAccessKind::Read;
  const std::optional<uint16_t> first_data_vgpr =
      load ? candidate.operands.destination_vgpr : candidate.operands.data_vgpr;
  const std::optional<uint16_t> second_data_vgpr =
      load && first_data_vgpr
          ? std::optional<uint16_t>(static_cast<uint16_t>(*first_data_vgpr + element_dwords))
          : candidate.operands.second_data_vgpr;
  if (!first_data_vgpr || !second_data_vgpr || *first_data_vgpr > 255u ||
      *second_data_vgpr > 255u ||
      (element_dwords == 2u && (*first_data_vgpr > 254u || *second_data_vgpr > 254u))) {
    errors.emplace_back("ConSan MOI split two-address guest access has invalid data operands");
    return std::nullopt;
  }

  const auto words = consan_build_split_two_address_lds_pair(
      {.first_byte_offset = candidate.lowering_offset(ranges[0]),
       .second_byte_offset = candidate.lowering_offset(ranges[1]),
       .element_dwords = element_dwords,
       .address_vgpr = request.replay_address_vgpr,
       .first_data_vgpr = *first_data_vgpr,
       .second_data_vgpr = *second_data_vgpr,
       .adjusted_address_vgpr = request.adjusted_address_vgpr,
       .load = load},
      target.arch);
  if (!words) {
    errors.emplace_back("ConSan MOI could not split a two-address guest access");
    return std::nullopt;
  }
  return words;
}

bool consan_detail::append_moi_relocated_guest_access(
    std::vector<uint32_t> &words, std::span<const uint8_t> image,
    const ConSanMoiCandidate &candidate, const ConSanTargetProfile *target,
    uint16_t replay_address_vgpr, std::optional<uint16_t> adjusted_address_vgpr,
    std::vector<std::string> &errors, uint32_t *guest_instruction_word_count) {
  auto guest_words = build_moi_relocated_guest_access_words(
      {image, &candidate, target, replay_address_vgpr, adjusted_address_vgpr}, errors);
  if (!guest_words)
    return false;
  if (guest_instruction_word_count)
    *guest_instruction_word_count = static_cast<uint32_t>(guest_words->size());
  words.insert(words.end(), guest_words->begin(), guest_words->end());
  return true;
}

bool consan_detail::append_save_moi_special_state(std::vector<uint32_t> &words,
                                                  const MoiSpecialStateSgprs &registers,
                                                  const ConSanTargetProfile &target) {
  const auto save_scc =
      instrumentation::build_s_cselect_b32(registers.scc_save_sgpr, scalar_positive_inline_u32(1),
                                           scalar_positive_inline_u32(0), target.arch);
  const auto save_vcc = instrumentation::build_s_mov_b64(
      registers.vcc_save_sgpr, scalar_operand_vcc_lo(target.arch), target.arch);
  if (!save_scc || !save_vcc)
    return false;
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  return true;
}

bool consan_detail::append_restore_moi_special_state(std::vector<uint32_t> &words,
                                                     const MoiSpecialStateSgprs &registers,
                                                     const ConSanTargetProfile &target) {
  const auto restore_vcc = instrumentation::build_s_mov_b64(scalar_operand_vcc_lo(target.arch),
                                                            registers.vcc_save_sgpr, target.arch);
  const auto restore_scc = instrumentation::build_s_cmp_lg_u32(
      registers.scc_save_sgpr, scalar_positive_inline_u32(0), target.arch);
  if (!restore_vcc || !restore_scc)
    return false;
  words.push_back(*restore_vcc);
  words.push_back(*restore_scc);
  return true;
}

bool consan_detail::append_moi_device_cache_refresh(std::vector<uint32_t> &words,
                                                    const ConSanTargetProfile &target) {
  std::optional<std::array<uint32_t, 2>> invalidate;
  switch (target.encoding_family) {
  case ConSanEncodingFamily::Gfx9Cdna3:
    invalidate = build_cdna3_buffer_inv_sc1(target.arch);
    break;
  case ConSanEncodingFamily::Gfx9Cdna4:
    invalidate = build_cdna4_buffer_inv_sc1(target.arch);
    break;
  case ConSanEncodingFamily::Gfx11:
  case ConSanEncodingFamily::Gfx12:
    return true;
  }
  if (!invalidate)
    return false;
  words.insert(words.end(), invalidate->begin(), invalidate->end());
  return true;
}

bool consan_detail::append_moi_global_atomic_completion(std::vector<uint32_t> &words,
                                                        const ConSanTargetProfile &target) {
  const auto load = instrumentation::build_s_wait_global_load0(target.arch);
  const bool needs_separate_store_wait =
      target.encoding_family != ConSanEncodingFamily::Gfx9Cdna3 &&
      target.encoding_family != ConSanEncodingFamily::Gfx9Cdna4;
  const auto store = needs_separate_store_wait
                         ? instrumentation::build_s_wait_global_store0(target.arch)
                         : std::optional<uint32_t>{};
  if (!load || (needs_separate_store_wait && !store))
    return false;
  words.push_back(*load);
  if (store)
    words.push_back(*store);
  return true;
}

bool consan_detail::append_moi_atomic_counter_increment(
    std::vector<uint32_t> &words, const MoiAtomicCounterIncrementRequest &request,
    const ConSanTargetProfile &target) {
  if (request.address_vgpr > 254u || request.result_vgpr > 255u ||
      request.result_vgpr == request.address_vgpr ||
      request.result_vgpr == static_cast<uint16_t>(request.address_vgpr + 1u)) {
    return false;
  }
  const auto address_lo = instrumentation::build_v_mov_b32_literal(
      request.address_vgpr, static_cast<uint32_t>(request.counter_address), target.arch);
  const auto address_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(request.address_vgpr + 1u),
      static_cast<uint32_t>(request.counter_address >> 32u), target.arch);
  const auto one = instrumentation::build_v_mov_b32_literal(request.result_vgpr, 1u, target.arch);
  const auto atomic = instrumentation::build_flat_atomic_add_u32(
      request.address_vgpr, request.result_vgpr, request.result_vgpr,
      /*return_old_value=*/true, /*scope=*/2u, target.arch);
  if (!address_lo || !address_hi || !one || !atomic)
    return false;

  std::vector<uint32_t> emitted;
  emitted.insert(emitted.end(), address_lo->begin(), address_lo->end());
  emitted.insert(emitted.end(), address_hi->begin(), address_hi->end());
  emitted.insert(emitted.end(), one->begin(), one->end());
  emitted.insert(emitted.end(), atomic->begin(), atomic->end());
  if (!append_moi_global_atomic_completion(emitted, target))
    return false;
  words.insert(words.end(), emitted.begin(), emitted.end());
  return true;
}

bool consan_detail::append_moi_workitem_owner_derivation(
    std::vector<uint32_t> &words, const MoiWorkitemOwnerDerivationRequest &request,
    const ConSanTargetProfile &target) {
  if (!request.is_well_formed())
    return false;

  std::vector<uint32_t> emitted;
  uint16_t owner_source_vgpr = 0u;
  if (request.plan.entry_workitem_x_private_offset) {
    const auto owner_load = instrumentation::build_private_load_b32(
        request.result_vgpr, *request.plan.entry_workitem_x_private_offset, target.arch);
    const auto owner_wait = instrumentation::build_s_wait_private_load0(target.arch);
    if (!owner_load || !owner_wait)
      return false;
    emitted.insert(emitted.end(), owner_load->begin(), owner_load->end());
    emitted.push_back(*owner_wait);
    owner_source_vgpr = request.result_vgpr;
  }

  // The AMDGPU kernel ABI supplies workitem-id-x in v0 at entry. A planned
  // private source captures that same value before guest code can repurpose it.
  const auto owner_init = instrumentation::build_v_lshrrev_b32(
      request.result_vgpr, scalar_positive_inline_u32(request.plan.wave_size_shift),
      owner_source_vgpr, target.arch);
  if (!owner_init)
    return false;
  emitted.push_back(*owner_init);
  words.insert(words.end(), emitted.begin(), emitted.end());
  return true;
}

bool consan_detail::append_moi_indexed_address(std::vector<uint32_t> &words,
                                               const MoiIndexedAddressRequest &request,
                                               const ConSanTargetProfile &target) {
  if (request.stride_bytes == 0u || request.address_vgpr > 254u || request.index_vgpr > 255u ||
      request.index_vgpr == request.address_vgpr ||
      request.index_vgpr == static_cast<uint16_t>(request.address_vgpr + 1u)) {
    return false;
  }

  const size_t original_size = words.size();
  const auto reject = [&]() {
    words.resize(original_size);
    return false;
  };
  const uint32_t highest_bit = std::bit_width(request.stride_bytes) - 1u;
  const auto scale_highest = instrumentation::build_v_lshlrev_b32(
      request.address_vgpr, scalar_positive_inline_u32(highest_bit), request.index_vgpr,
      target.arch);
  if (!scale_highest)
    return reject();
  words.push_back(*scale_highest);

  uint32_t remaining_bits = request.stride_bytes & ~(uint32_t{1} << highest_bit);
  while (remaining_bits != 0u) {
    const uint32_t bit = std::bit_width(remaining_bits) - 1u;
    const auto scale_term = instrumentation::build_v_lshlrev_b32(
        static_cast<uint16_t>(request.address_vgpr + 1u), scalar_positive_inline_u32(bit),
        request.index_vgpr, target.arch);
    const auto add_term = instrumentation::build_v_add_u32(
        request.address_vgpr, vector_source_vgpr(request.address_vgpr),
        static_cast<uint16_t>(request.address_vgpr + 1u), target.arch);
    if (!scale_term || !add_term)
      return reject();
    words.push_back(*scale_term);
    words.insert(words.end(), add_term->begin(), add_term->end());
    remaining_bits &= ~(uint32_t{1} << bit);
  }

  words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(request.address_vgpr + 1u),
                                      scalar_positive_inline_u32(0), target.arch));
  const auto add_base = instrumentation::build_v_add_u64_literal(
      request.address_vgpr, request.table_address, target.arch);
  if (!add_base)
    return reject();
  words.insert(words.end(), add_base->begin(), add_base->end());
  return true;
}

std::optional<consan_detail::ScalarOwnerContextResolution>
consan_detail::resolve_scalar_owner_contexts(bool planning_state_valid,
                                             std::span<const ScalarOwnerContextSummary> contexts,
                                             std::span<const uint64_t> owners) {
  if (!planning_state_valid || owners.empty())
    return std::nullopt;

  ScalarOwnerContextResolution resolution;
  resolution.context_indices.reserve(owners.size());
  for (uint64_t descriptor_offset : owners) {
    const auto context = std::ranges::find(contexts, descriptor_offset,
                                           &ScalarOwnerContextSummary::descriptor_file_offset);
    if (context == contexts.end() || !context->descriptor_valid) {
      return std::nullopt;
    }
    resolution.context_indices.push_back(
        static_cast<size_t>(std::distance(contexts.begin(), context)));
    resolution.tail_floor =
        std::max<uint32_t>(resolution.tail_floor, scalar_owner_tail_floor(*context));
  }
  return resolution;
}

uint16_t consan_detail::scalar_owner_tail_floor(const ScalarOwnerContextSummary &context) {
  uint16_t floor = context.max_referenced_sgpr_count;
  if (context.has_indirect_sgpr_access || !context.sgpr_reference_coverage_complete)
    floor = std::max(floor, context.current_sgpr_count);
  return floor;
}

std::optional<uint16_t> moi_dynamic_stack_frame_save_sgpr_offset(ConSanMoiEngine engine) {
  return consan_moi_impl::moi_mode_operations(engine).dynamic_stack_frame_save_sgpr_offset;
}

MoiExecSaveRequirement
resolve_moi_exec_save_requirement(const ConSanRequest &request,
                                  const BoundRuntimeResources &resources,
                                  const ConSanMoiOperatingPoint &operating_point,
                                  const consan_moi_impl::MoiObjectModeSemantics &mode_semantics) {
  return MoiExecSaveRequirement{
      .engine = request.moi_engine,
      .has_report_buffer = resources.moi_report_buffer_address.has_value(),
      .track_atomics = request.moi_track_atomics,
      .automatic_banked_record_capture =
          consan_moi_detail::record_replay_uses_automatic_banked_capture(request, resources),
      .runtime_sample_stride = request.moi_runtime_sample_stride,
      .scalar_spill = operating_point.has_compact_moi_scalar_spill(),
      .dynamic_stack_spill = operating_point.moi_dynamic_stack_spill,
      .inline_access_present = mode_semantics.inline_access_present,
      .dense_record_barrier_router = mode_semantics.dense_barrier_router,
  };
}

bool moi_initializes_owner_epoch(const ConSanRequest &request,
                                 const ConSanMoiOperatingPoint &operating_point) {
  return operating_point.moi_initialize_owner_epoch.value_or(request.moi_init_owner_epoch);
}

uint16_t moi_exec_save_sgpr_count(const MoiExecSaveRequirement &requirement, rj_code_arch_t arch) {
  consan_moi_impl::MoiExecSaveTargetFacts target_facts;
  if (const ConSanTargetProfile *target = consan_target_profile(arch))
    target_facts.direct_call_form = target->direct_call_form;
  const auto &operations = consan_moi_impl::moi_mode_operations(requirement.engine);
  return operations.exec_save_sgpr_count(requirement, target_facts);
}

namespace {

using consan_detail::append_moi_workitem_owner_derivation;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::MoiSpecialStateSgprs;

[[nodiscard]] std::optional<uint16_t>
scalar_owner_cdna_physical_vcc_base(uint32_t decoded_sgpr_count) {
  constexpr uint32_t kSgprGranularity = 8u;
  constexpr uint32_t kCdnaAllocationTailAfterOrdinarySgprs = 6u;
  const uint32_t encoded_allocation =
      ((decoded_sgpr_count + kSgprGranularity - 1u) / kSgprGranularity) * kSgprGranularity;
  if (encoded_allocation < kCdnaAllocationTailAfterOrdinarySgprs)
    return std::nullopt;
  const uint32_t base = encoded_allocation - kCdnaAllocationTailAfterOrdinarySgprs;
  if (base > std::numeric_limits<uint16_t>::max())
    return std::nullopt;
  return static_cast<uint16_t>(base);
}

[[nodiscard]] bool scalar_owner_ranges_conflict_with_physical_vcc(
    std::span<const consan_detail::ScalarOwnerSgprRange> ranges, uint32_t current_sgpr_count) {
  const auto original_vcc = scalar_owner_cdna_physical_vcc_base(current_sgpr_count);
  if (!original_vcc || ranges.empty())
    return true;
  uint32_t required_count = current_sgpr_count;
  for (const consan_detail::ScalarOwnerSgprRange &range : ranges)
    required_count = std::max<uint32_t>(required_count, range.base + range.width);
  const auto grown_vcc = scalar_owner_cdna_physical_vcc_base(required_count);
  if (!grown_vcc)
    return true;
  return std::ranges::any_of(ranges, [&](const consan_detail::ScalarOwnerSgprRange &range) {
    const auto overlaps = [&](uint16_t other_base) {
      return range.base < static_cast<uint32_t>(other_base) + 2u &&
             other_base < static_cast<uint32_t>(range.base) + range.width;
    };
    return range.width == 0u || overlaps(*original_vcc) || overlaps(*grown_vcc) ||
           static_cast<uint32_t>(range.base) + range.width > *grown_vcc;
  });
}

} // namespace

bool consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(
    std::span<const ScalarOwnerContextSummary> contexts,
    std::span<const ScalarOwnerSgprRange> ranges) {
  if (contexts.empty() || ranges.empty())
    return true;
  return std::ranges::any_of(contexts, [&](const ScalarOwnerContextSummary &context) {
    return !context.descriptor_valid ||
           scalar_owner_ranges_conflict_with_physical_vcc(ranges, context.current_sgpr_count);
  });
}

bool consan_detail::scalar_owner_contexts_admit_reserved_window(
    std::span<const ScalarOwnerContextSummary> contexts, uint16_t base, uint16_t width,
    bool protect_physical_vcc) {
  if (contexts.empty() || width == 0u)
    return false;
  const std::array ranges{ScalarOwnerSgprRange{base, width}};
  return std::ranges::all_of(contexts,
                             [&](const ScalarOwnerContextSummary &context) {
                               return context.descriptor_valid &&
                                      scalar_owner_tail_floor(context) <= base;
                             }) &&
         (!protect_physical_vcc ||
          !scalar_owner_contexts_conflict_with_physical_vcc(contexts, ranges));
}

bool consan_detail::validate_scalar_state_temporaries(
    const ConSanMoiOperatingPoint &point, const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs,
    std::string_view consumer, std::vector<std::string> &errors) {
  if (!point.moi_persistent_sgprs.complete() ||
      (owner_epoch_vgprs.owner && owner_epoch_vgprs.epoch)) {
    return true;
  }
  errors.emplace_back("ConSan MOI " + std::string(consumer) +
                      " has no scalar-state VGPR temporaries (owner=" +
                      std::string(owner_epoch_vgprs.owner ? "set" : "unset") +
                      ", epoch=" + std::string(owner_epoch_vgprs.epoch ? "set" : "unset") + ")");
  return false;
}

const char *consan_detail::moi_spilled_vgpr_reload_result_name(MoiSpilledVgprReloadResult result) {
  switch (result) {
  case MoiSpilledVgprReloadResult::Appended:
    return "appended";
  case MoiSpilledVgprReloadResult::SourceOutsideWindow:
    return "source_outside_window";
  case MoiSpilledVgprReloadResult::IncompleteSlotMetadata:
    return "incomplete_slot_metadata";
  case MoiSpilledVgprReloadResult::UnsupportedEncoding:
    return "unsupported_encoding";
  }
  return "unknown";
}

consan_detail::MoiSpilledVgprReloadResult
consan_detail::append_reload_moi_spilled_vgpr(std::vector<uint32_t> &words,
                                              const VgprSpillSequence &spill, uint16_t destination,
                                              uint16_t source, rj_code_arch_t arch) {
  const uint32_t spill_end = static_cast<uint32_t>(spill.vgpr_base) + spill.vgpr_count;
  if (source < spill.vgpr_base || source >= spill_end)
    return MoiSpilledVgprReloadResult::SourceOutsideWindow;
  if (!spill.has_complete_slot_metadata())
    return MoiSpilledVgprReloadResult::IncompleteSlotMetadata;

  const uint32_t slot_offset = spill.slot_offsets[source - spill.vgpr_base];
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  InstructionSequence sequence(words);
  bool encoded = false;
  if (spill.uses_dynamic_stack_frame) {
    encoded = sequence.emit_all(build_dynamic_stack_vgpr_load(
                                    destination, spill.dynamic_frame_base_sgpr, slot_offset, arch),
                                wait);
  } else {
    encoded = sequence.emit_all(
        instrumentation::build_private_load_b32(destination, slot_offset, arch), wait);
  }
  if (!encoded)
    return MoiSpilledVgprReloadResult::UnsupportedEncoding;

  return MoiSpilledVgprReloadResult::Appended;
}

bool consan_detail::append_workgroup_source_value(std::vector<uint32_t> &words,
                                                  const ConSanMoiWorkgroupSource &source,
                                                  uint16_t value_vgpr, rj_code_arch_t arch) {
  if (!source.is_well_formed() || !source.has_value())
    return false;
  if (source.private_offset) {
    const auto load =
        instrumentation::build_private_load_b32(value_vgpr, *source.private_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    InstructionSequence sequence(words);
    if (!sequence.emit_all(load, wait))
      return false;
  } else {
    const auto operand = source.operand();
    if (!operand)
      return false;
    words.push_back(build_v_mov_b32_e32(value_vgpr, *operand, arch));
  }
  if (source.mask_low_16) {
    const auto shift_left = instrumentation::build_v_lshlrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    const auto shift_right = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit_all(shift_left, shift_right))
      return false;
  }
  if (source.shift_right_16) {
    const auto shift = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit(shift))
      return false;
  }
  return true;
}

} // namespace rocjitsu
