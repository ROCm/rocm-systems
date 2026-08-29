// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_supercollider_support.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace {

[[nodiscard]] bool is_forbidden_flat_scratch_vgpr_run(const ConSanAccessInventorySite &access,
                                                      uint16_t candidate, uint16_t required_vgprs) {
  if (!access.lowering.form)
    return true;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  if (form.address_vgpr &&
      vgpr_ranges_overlap(candidate, required_vgprs, *form.address_vgpr, form.address_vgpr_count))
    return true;
  if (form.destination_vgpr &&
      vgpr_ranges_overlap(candidate, required_vgprs, *form.destination_vgpr,
                          form.destination_register_count))
    return true;
  return form.data_vgpr &&
         vgpr_ranges_overlap(candidate, required_vgprs, *form.data_vgpr, form.data_register_count);
}

} // namespace

bool is_instrumentable_group_flat_hint(ConSanFlatAddressSpaceHint hint,
                                       ConSanFlatProvenanceMode mode) {
  return hint == ConSanFlatAddressSpaceHint::Group ||
         (mode == ConSanFlatProvenanceMode::Likely &&
          hint == ConSanFlatAddressSpaceHint::MaybeGroup);
}

std::optional<uint16_t> flat_dword_count(const ConSanAccessInventorySite &access) {
  if (!access.lowering.form || access.lowering.form->data_register_count == 0u)
    return std::nullopt;
  return access.lowering.form->data_register_count;
}

bool flat_scratch_tuple_base_is_valid(const ConSanAccessInventorySite &access, uint16_t candidate) {
  return !access.lowering.form || access.lowering.form->data_register_alignment <= 1u ||
         candidate % access.lowering.form->data_register_alignment == 0u;
}

uint16_t flat_scratch_search_start(const ConSanAccessInventorySite &access) {
  if (!access.lowering.form)
    return 0u;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  std::optional<uint32_t> first_after_operands;
  auto note_range = [&first_after_operands](std::optional<uint16_t> base, uint16_t count) {
    if (!base)
      return;
    const uint32_t end = static_cast<uint32_t>(*base) + count;
    if (!first_after_operands || end > *first_after_operands)
      first_after_operands = end;
  };
  note_range(form.address_vgpr, form.address_vgpr_count);
  note_range(form.destination_vgpr, form.destination_register_count);
  note_range(form.data_vgpr, form.data_register_count);
  if (!first_after_operands)
    return 0;
  return *first_after_operands <= 255 ? static_cast<uint16_t>(*first_after_operands) : 256;
}

std::optional<uint16_t>
choose_flat_scratch_vgpr(const ConSanAccessInventorySite &access, const ConSanOptions &options,
                         const Instruction *inst, const LivenessAnalysis *liveness,
                         std::optional<uint16_t> min_auto_scratch_vgpr,
                         std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs) {
  if (options.scratch_vgpr) {
    if (static_cast<uint32_t>(*options.scratch_vgpr) + required_vgprs <= 256 &&
        flat_scratch_tuple_base_is_valid(access, *options.scratch_vgpr) &&
        !is_forbidden_flat_scratch_vgpr_run(access, *options.scratch_vgpr, required_vgprs))
      return *options.scratch_vgpr;
    return std::nullopt;
  }
  if (inst == nullptr || liveness == nullptr)
    return std::nullopt;

  uint16_t search_start =
      std::max(flat_scratch_search_start(access), min_auto_scratch_vgpr.value_or(0));
  while (search_start < REGISTER_SET_MAX_VGPRS) {
    auto candidate = liveness->find_free_run(inst, required_vgprs, search_start);
    if (!candidate)
      return std::nullopt;
    const uint32_t candidate_end = static_cast<uint32_t>(*candidate) + required_vgprs;
    if (max_auto_scratch_vgpr && candidate_end > *max_auto_scratch_vgpr)
      return std::nullopt;
    if (candidate_end <= REGISTER_SET_MAX_VGPRS &&
        flat_scratch_tuple_base_is_valid(access, *candidate) &&
        !is_forbidden_flat_scratch_vgpr_run(access, *candidate, required_vgprs))
      return candidate;
    if (*candidate == UINT16_MAX)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*candidate + 1u);
  }
  return std::nullopt;
}

std::optional<uint16_t> choose_flat_spill_scratch_vgpr(const ConSanAccessInventorySite &access,
                                                       uint16_t allocation_count,
                                                       uint16_t required_vgprs) {
  for (uint32_t candidate = 0; candidate + required_vgprs <= allocation_count &&
                               candidate + required_vgprs <= REGISTER_SET_MAX_VGPRS;
       ++candidate) {
    if (flat_scratch_tuple_base_is_valid(access, static_cast<uint16_t>(candidate)) &&
        !is_forbidden_flat_scratch_vgpr_run(access, static_cast<uint16_t>(candidate),
                                            required_vgprs)) {
      return static_cast<uint16_t>(candidate);
    }
  }
  return std::nullopt;
}

std::optional<uint16_t> flat_check_trap_compare_vgpr(const ConSanAccessInventorySite &access) {
  if (access.kind == ConSanLdsAccessKind::Read)
    return access.operands.destination_vgpr;
  if (access.kind == ConSanLdsAccessKind::Write)
    return access.operands.data_vgpr;
  return std::nullopt;
}

std::optional<uint16_t> check_trap_compare_vgpr(const ConSanAccessInventorySite &access,
                                                uint16_t chunk_index, rj_code_arch_t arch,
                                                uint16_t gfx1250_vgpr_msb_mode) {
  if (access.kind == ConSanLdsAccessKind::Read) {
    const uint32_t dst_bank = consan_arch_has_selectable_vgpr_bank(arch)
                                  ? static_cast<uint32_t>((gfx1250_vgpr_msb_mode >> 6u) & 0x3u)
                                  : 0u;
    const uint32_t physical_vgpr =
        dst_bank * 256u + access.operands.destination_vgpr.value_or(0u) + chunk_index;
    const uint32_t register_limit = consan_arch_has_selectable_vgpr_bank(arch) ? 1024u : 256u;
    return access.operands.destination_vgpr && physical_vgpr < register_limit
               ? std::optional<uint16_t>(static_cast<uint16_t>(physical_vgpr))
               : std::nullopt;
  }
  if (access.kind == ConSanLdsAccessKind::Write) {
    const bool two_address =
        access.lowering.form &&
        access.lowering.form->kind == ConSanAccessLoweringFormKind::NativeTwoRange;
    const uint16_t dwords_per_address =
        static_cast<uint16_t>(two_address ? access.lowering.form->element_width_bits / 32u
                                          : access.decoded_width_bits / 32u);
    const std::optional<uint16_t> base = two_address && chunk_index >= dwords_per_address
                                             ? access.operands.second_data_vgpr
                                             : access.operands.data_vgpr;
    const uint16_t relative_index = two_address && chunk_index >= dwords_per_address
                                        ? static_cast<uint16_t>(chunk_index - dwords_per_address)
                                        : chunk_index;
    return base && static_cast<uint32_t>(*base) + relative_index < 256
               ? std::optional<uint16_t>(static_cast<uint16_t>(*base + relative_index))
               : std::nullopt;
  }
  return std::nullopt;
}

} // namespace rocjitsu
