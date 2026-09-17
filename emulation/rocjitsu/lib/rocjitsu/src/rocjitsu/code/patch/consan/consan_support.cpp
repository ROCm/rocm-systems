// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_instrumentation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/kernel_scope.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_identity_contracts.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_lds_ops.h"
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

namespace rocjitsu::consan {

bool detail::range_overlaps(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                            uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return static_cast<uint32_t>(lhs_base) < rhs_end && static_cast<uint32_t>(rhs_base) < lhs_end;
}

bool detail::reject_optional_scratch_range_overlap(std::optional<uint16_t> value,
                                                   uint16_t scratch_vgpr, uint16_t scratch_count,
                                                   std::string_view value_name,
                                                   std::vector<std::string> &errors) {
  const bool overlaps = value && scratch_count != 0u && *value >= scratch_vgpr &&
                        *value < static_cast<uint16_t>(scratch_vgpr + scratch_count);
  if (!overlaps)
    return false;
  errors.emplace_back(std::string("ConSan scratch VGPRs overlap ") + std::string(value_name) +
                      " VGPR");
  return true;
}

std::optional<uint16_t> WorkgroupSource::operand() const {
  if (!has_value() || private_offset)
    return std::nullopt;
  return scalar_src ? *scalar_src : vector_source_vgpr(*vector_src);
}

bool detail::append_resident_wave_owner(std::vector<uint32_t> &words,
                                        const ResidentWaveOwnerRequest &request,
                                        const TargetProfile &target) {
  const ResidentWaveIdentityEncoding &encoding = target.resident_wave_identity;
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

  words.push_back(*wait);
  return true;
}

bool detail::guest_access_relocation_requires_adjusted_address(const Candidate &candidate,
                                                               const TargetProfile &target) {
  return target.requires_split_two_address_lds_relocation && candidate.is_native_two_range() &&
         candidate.encoded_offset_scale_bytes() > 8u;
}

std::optional<std::vector<uint32_t>>
detail::build_relocated_guest_access_words(const GuestAccessRelocationRequest &request,
                                           std::vector<std::string> &errors) {
  if (request.candidate == nullptr || request.target == nullptr) {
    errors.emplace_back("ConSan guest relocation requires a candidate and target profile");
    return std::nullopt;
  }
  const Candidate &candidate = *request.candidate;
  const TargetProfile &target = *request.target;
  const auto copy_original = [&]() -> std::optional<std::vector<uint32_t>> {
    if (candidate.size() == 0u || candidate.size() % sizeof(uint32_t) != 0u ||
        candidate.site().decoded_file_offset() > request.image.size() ||
        candidate.size() > request.image.size() - candidate.site().decoded_file_offset()) {
      errors.emplace_back("ConSan relocated guest access exceeds the code object");
      return std::nullopt;
    }
    std::vector<uint32_t> words(candidate.size() / sizeof(uint32_t));
    std::memcpy(words.data(), request.image.data() + candidate.site().decoded_file_offset(),
                candidate.size());
    return words;
  };

  // Preserve the combined instruction when a load overwrites its own address.
  // Callers snapshot that address before the guest runs so evidence emission
  // can still use it afterward. Replaying the guest from the snapshot would
  // unnecessarily split the combined load, changing its target dependency and
  // completion semantics. The original instruction remains safe here because
  // its address operand is intact until the combined operation is issued.
  if (!target.requires_split_two_address_lds_relocation || !candidate.is_native_two_range() ||
      detail::load_clobbers_address(candidate))
    return copy_original();

  const auto &ranges = candidate.site().ranges;
  const uint16_t element_dwords = static_cast<uint16_t>(candidate.width_bits() / 32u);
  if (ranges.size() != 2u ||
      (candidate.site().kind != LdsAccessKind::Read &&
       candidate.site().kind != LdsAccessKind::Write) ||
      (element_dwords != 1u && element_dwords != 2u) || request.replay_address_vgpr > 255u) {
    errors.emplace_back("ConSan could not normalize a split two-address guest access");
    return std::nullopt;
  }

  const bool load = candidate.site().kind == LdsAccessKind::Read;
  const std::optional<uint16_t> first_data_vgpr =
      load ? candidate.site().operands.destination_vgpr : candidate.site().operands.data_vgpr;
  const std::optional<uint16_t> second_data_vgpr =
      load && first_data_vgpr
          ? std::optional<uint16_t>(static_cast<uint16_t>(*first_data_vgpr + element_dwords))
          : candidate.site().operands.second_data_vgpr;
  if (!first_data_vgpr || !second_data_vgpr || *first_data_vgpr > 255u ||
      *second_data_vgpr > 255u ||
      (element_dwords == 2u && (*first_data_vgpr > 254u || *second_data_vgpr > 254u))) {
    errors.emplace_back("ConSan split two-address guest access has invalid data operands");
    return std::nullopt;
  }

  const auto words =
      build_split_two_address_lds_pair({.first_byte_offset = candidate.lowering_offset(ranges[0]),
                                        .second_byte_offset = candidate.lowering_offset(ranges[1]),
                                        .element_dwords = element_dwords,
                                        .address_vgpr = request.replay_address_vgpr,
                                        .first_data_vgpr = *first_data_vgpr,
                                        .second_data_vgpr = *second_data_vgpr,
                                        .adjusted_address_vgpr = request.adjusted_address_vgpr,
                                        .load = load},
                                       target.arch);
  if (!words) {
    errors.emplace_back("ConSan could not split a two-address guest access");
    return std::nullopt;
  }
  return words;
}

bool detail::append_relocated_guest_access(std::vector<uint32_t> &words,
                                           std::span<const uint8_t> image,
                                           const Candidate &candidate, const TargetProfile *target,
                                           uint16_t replay_address_vgpr,
                                           std::optional<uint16_t> adjusted_address_vgpr,
                                           std::vector<std::string> &errors,
                                           uint32_t *guest_instruction_word_count) {
  auto guest_words = build_relocated_guest_access_words(
      {image, &candidate, target, replay_address_vgpr, adjusted_address_vgpr}, errors);
  if (!guest_words)
    return false;
  if (guest_instruction_word_count)
    *guest_instruction_word_count = static_cast<uint32_t>(guest_words->size());
  words.insert(words.end(), guest_words->begin(), guest_words->end());
  return true;
}

bool detail::append_save_special_state(std::vector<uint32_t> &words,
                                       const SpecialStateSgprs &registers,
                                       const TargetProfile &target) {
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

bool detail::append_restore_special_state(std::vector<uint32_t> &words,
                                          const SpecialStateSgprs &registers,
                                          const TargetProfile &target) {
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

bool detail::append_device_cache_refresh(std::vector<uint32_t> &words,
                                         const TargetProfile &target) {
  std::optional<std::array<uint32_t, 2>> invalidate;
  switch (target.device_cache_refresh) {
  case DeviceCacheRefreshForm::Cdna3BufferInvSc1:
    invalidate = build_cdna3_buffer_inv_sc1(target.arch);
    break;
  case DeviceCacheRefreshForm::Cdna4BufferInvSc1:
    invalidate = build_cdna4_buffer_inv_sc1(target.arch);
    break;
  case DeviceCacheRefreshForm::None:
    return true;
  }
  if (!invalidate)
    return false;
  words.insert(words.end(), invalidate->begin(), invalidate->end());
  return true;
}

bool detail::append_global_atomic_completion(std::vector<uint32_t> &words,
                                             const TargetProfile &target) {
  const auto load = instrumentation::build_s_wait_global_load0(target.arch);
  const auto store = instrumentation::build_s_wait_global_store0(target.arch);
  InstructionSequence sequence(words);
  sequence.append(load);
  if (load && store && *store != *load)
    sequence.append(store);
  else
    sequence.require(store.has_value());
  return sequence.finish();
}

bool detail::append_atomic_counter_increment(std::vector<uint32_t> &words,
                                             const AtomicCounterIncrementRequest &request,
                                             const TargetProfile &target) {
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
  InstructionSequence sequence(words);
  sequence.append(address_lo, address_hi, one, atomic)
      .require(append_global_atomic_completion(words, target));
  return sequence.finish();
}

bool detail::append_workitem_owner_derivation(std::vector<uint32_t> &words,
                                              const WorkitemOwnerDerivationRequest &request,
                                              const TargetProfile &target) {
  if (!request.is_well_formed())
    return false;

  InstructionSequence sequence(words);
  uint16_t owner_source_vgpr = 0u;
  if (request.plan.entry_workitem_x_private_offset) {
    const auto owner_load = instrumentation::build_private_load_b32(
        request.result_vgpr, *request.plan.entry_workitem_x_private_offset, target.arch);
    const auto owner_wait = instrumentation::build_s_wait_private_load0(target.arch);
    sequence.append(owner_load, owner_wait);
    owner_source_vgpr = request.result_vgpr;
  }

  // The AMDGPU kernel ABI supplies workitem-id-x in v0 at entry. A planned
  // private source captures that same value before guest code can repurpose it.
  const auto owner_init = instrumentation::build_v_lshrrev_b32(
      request.result_vgpr, scalar_positive_inline_u32(request.plan.wave_size_shift),
      owner_source_vgpr, target.arch);
  sequence.append(owner_init);
  return sequence.finish();
}

bool detail::append_indexed_address(std::vector<uint32_t> &words,
                                    const IndexedAddressRequest &request,
                                    const TargetProfile &target) {
  if (request.stride_bytes == 0u || request.address_vgpr > 254u || request.index_vgpr > 255u ||
      request.index_vgpr == request.address_vgpr ||
      request.index_vgpr == static_cast<uint16_t>(request.address_vgpr + 1u)) {
    return false;
  }

  InstructionSequence sequence(words);
  const uint32_t highest_bit = std::bit_width(request.stride_bytes) - 1u;
  const auto scale_highest = instrumentation::build_v_lshlrev_b32(
      request.address_vgpr, scalar_positive_inline_u32(highest_bit), request.index_vgpr,
      target.arch);
  sequence.append(scale_highest);

  uint32_t remaining_bits = request.stride_bytes & ~(uint32_t{1} << highest_bit);
  while (remaining_bits != 0u) {
    const uint32_t bit = std::bit_width(remaining_bits) - 1u;
    const auto scale_term = instrumentation::build_v_lshlrev_b32(
        static_cast<uint16_t>(request.address_vgpr + 1u), scalar_positive_inline_u32(bit),
        request.index_vgpr, target.arch);
    const auto add_term = instrumentation::build_v_add_u32(
        request.address_vgpr, vector_source_vgpr(request.address_vgpr),
        static_cast<uint16_t>(request.address_vgpr + 1u), target.arch);
    sequence.append(scale_term, add_term);
    remaining_bits &= ~(uint32_t{1} << bit);
  }

  sequence.append(build_v_mov_b32_e32(static_cast<uint16_t>(request.address_vgpr + 1u),
                                      scalar_positive_inline_u32(0), target.arch));
  const auto add_base = instrumentation::build_v_add_u64_literal(
      request.address_vgpr, request.table_address, target.arch);
  sequence.append(add_base);
  return sequence.finish();
}

std::optional<detail::ScalarOwnerContextResolution>
detail::resolve_scalar_owner_contexts(bool planning_state_valid,
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

uint16_t detail::scalar_owner_tail_floor(const ScalarOwnerContextSummary &context) {
  uint16_t floor = context.max_referenced_sgpr_count;
  if (context.has_indirect_sgpr_access || !context.sgpr_reference_coverage_complete)
    floor = std::max(floor, context.current_sgpr_count);
  return floor;
}

ExecSaveRequirement resolve_exec_save_requirement(const Request &request,
                                                  const BoundRuntimeResources &resources,
                                                  const OperatingPoint &operating_point) {
  return ExecSaveRequirement{
      .has_report_buffer = resources.report_buffer_address.has_value(),
      .track_atomics = request.track_atomics,
      .scalar_spill = operating_point.has_compact_scalar_spill(),
      .dynamic_stack_spill = operating_point.dynamic_stack_spill,
  };
}

uint16_t exec_save_sgpr_count(const ExecSaveRequirement &requirement, rj_code_arch_t arch) {
  if (!requirement.has_report_buffer)
    return 0u;
  if (requirement.scalar_spill)
    return 8u;
  if (requirement.dynamic_stack_spill)
    return 9u;
  const TargetProfile *target = target_profile(arch);
  if (target && target->direct_call_form == DirectCallForm::SCallI64)
    return 8u;
  return requirement.track_atomics ? 8u : 7u;
}

namespace {

using detail::append_workitem_owner_derivation;
using detail::guest_access_relocation_requires_adjusted_address;
using detail::SpecialStateSgprs;

constexpr auto kSpilledVgprReloadResults = make_enum_vocabulary(
    "unknown", enum_entry(detail::SpilledVgprReloadResult::Appended, "appended"),
    enum_entry(detail::SpilledVgprReloadResult::SourceOutsideWindow, "source_outside_window"),
    enum_entry(detail::SpilledVgprReloadResult::IncompleteSlotMetadata, "incomplete_slot_metadata"),
    enum_entry(detail::SpilledVgprReloadResult::UnsupportedEncoding, "unsupported_encoding"));

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

[[nodiscard]] bool
scalar_owner_ranges_conflict_with_physical_vcc(std::span<const detail::ScalarOwnerSgprRange> ranges,
                                               uint32_t current_sgpr_count,
                                               uint32_t required_sgpr_count_floor) {
  const auto original_vcc = scalar_owner_cdna_physical_vcc_base(current_sgpr_count);
  if (!original_vcc || ranges.empty())
    return true;
  uint32_t required_count = std::max(current_sgpr_count, required_sgpr_count_floor);
  for (const detail::ScalarOwnerSgprRange &range : ranges)
    required_count = std::max<uint32_t>(required_count, range.base + range.width);
  const auto grown_vcc = scalar_owner_cdna_physical_vcc_base(required_count);
  if (!grown_vcc)
    return true;
  return std::ranges::any_of(ranges, [&](const detail::ScalarOwnerSgprRange &range) {
    const auto overlaps = [&](uint16_t other_base) {
      return range.base < static_cast<uint32_t>(other_base) + 2u &&
             other_base < static_cast<uint32_t>(range.base) + range.width;
    };
    return range.width == 0u || overlaps(*original_vcc) || overlaps(*grown_vcc) ||
           static_cast<uint32_t>(range.base) + range.width > *grown_vcc;
  });
}

} // namespace

bool detail::scalar_owner_contexts_conflict_with_physical_vcc(
    std::span<const ScalarOwnerContextSummary> contexts,
    std::span<const ScalarOwnerSgprRange> ranges, uint32_t required_sgpr_count_floor) {
  if (contexts.empty() || ranges.empty())
    return true;
  return std::ranges::any_of(contexts, [&](const ScalarOwnerContextSummary &context) {
    return !context.descriptor_valid ||
           scalar_owner_ranges_conflict_with_physical_vcc(ranges, context.current_sgpr_count,
                                                          required_sgpr_count_floor);
  });
}

bool detail::scalar_owner_contexts_admit_reserved_window(
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

bool detail::validate_scalar_state_temporaries(const PersistentSgprState &persistent_sgprs,
                                               const OwnerEpochVgprSources &owner_epoch_vgprs,
                                               std::string_view consumer,
                                               std::vector<std::string> &errors) {
  if (!persistent_sgprs.complete() || (owner_epoch_vgprs.owner && owner_epoch_vgprs.epoch)) {
    return true;
  }
  errors.emplace_back("ConSan " + std::string(consumer) +
                      " has no scalar-state VGPR temporaries (owner=" +
                      std::string(owner_epoch_vgprs.owner ? "set" : "unset") +
                      ", epoch=" + std::string(owner_epoch_vgprs.epoch ? "set" : "unset") + ")");
  return false;
}

const char *detail::spilled_vgpr_reload_result_name(SpilledVgprReloadResult result) {
  return kSpilledVgprReloadResults.name(result).data();
}

detail::SpilledVgprReloadResult
detail::append_reload_spilled_vgpr(std::vector<uint32_t> &words, const VgprSpillSequence &spill,
                                   uint16_t destination, uint16_t source, rj_code_arch_t arch) {
  const uint32_t spill_end = static_cast<uint32_t>(spill.vgpr_base) + spill.vgpr_count;
  if (source < spill.vgpr_base || source >= spill_end)
    return SpilledVgprReloadResult::SourceOutsideWindow;
  if (!spill.has_complete_slot_metadata())
    return SpilledVgprReloadResult::IncompleteSlotMetadata;

  const uint32_t slot_offset = spill.slot_offsets[source - spill.vgpr_base];
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  InstructionSequence sequence(words);
  if (spill.uses_dynamic_stack_frame) {
    sequence.append(build_dynamic_stack_vgpr_load(destination, spill.dynamic_frame_base_sgpr,
                                                  slot_offset, arch),
                    wait);
  } else {
    sequence.append(instrumentation::build_private_load_b32(destination, slot_offset, arch), wait);
  }
  if (!sequence.finish())
    return SpilledVgprReloadResult::UnsupportedEncoding;

  return SpilledVgprReloadResult::Appended;
}

bool detail::append_workgroup_source_value(std::vector<uint32_t> &words,
                                           const WorkgroupSource &source, uint16_t value_vgpr,
                                           rj_code_arch_t arch) {
  if (!source.is_well_formed() || !source.has_value())
    return false;
  InstructionSequence sequence(words);
  if (source.private_offset) {
    const auto load =
        instrumentation::build_private_load_b32(value_vgpr, *source.private_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    sequence.append(load, wait);
  } else {
    const auto operand = source.operand();
    if (!operand)
      return false;
    sequence.append(build_v_mov_b32_e32(value_vgpr, *operand, arch));
  }
  if (source.right_shift != 0u) {
    const auto shift = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(source.right_shift), value_vgpr, arch);
    sequence.append(shift);
  }
  if (source.low_bit_count != 0u && source.low_bit_count < 32u) {
    const uint16_t mask_shift = scalar_positive_inline_u32(32u - source.low_bit_count);
    const auto shift_left =
        instrumentation::build_v_lshlrev_b32(value_vgpr, mask_shift, value_vgpr, arch);
    const auto shift_right =
        instrumentation::build_v_lshrrev_b32(value_vgpr, mask_shift, value_vgpr, arch);
    sequence.append(shift_left, shift_right);
  }
  return sequence.finish();
}

} // namespace rocjitsu::consan
