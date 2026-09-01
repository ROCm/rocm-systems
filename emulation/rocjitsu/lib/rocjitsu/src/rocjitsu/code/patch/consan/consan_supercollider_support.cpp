// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_supercollider_support.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {

bool is_instrumentable_group_flat_hint(ConSanFlatAddressSpaceHint hint,
                                       ConSanFlatProvenanceMode mode) {
  return hint == ConSanFlatAddressSpaceHint::Group ||
         (mode == ConSanFlatProvenanceMode::Likely &&
          hint == ConSanFlatAddressSpaceHint::MaybeGroup);
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
                                                uint16_t selectable_vgpr_bank_mode) {
  if (access.kind == ConSanLdsAccessKind::Read) {
    const uint32_t dst_bank = consan_arch_has_selectable_vgpr_bank(arch)
                                  ? static_cast<uint32_t>((selectable_vgpr_bank_mode >> 6u) & 0x3u)
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
