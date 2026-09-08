// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/modes/supercollider/consan_supercollider_support.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace {

[[nodiscard]] uint32_t selected_vgpr_bank(rj_code_arch_t arch, uint16_t packed_mode,
                                          amdgpu::VgprMsbRole role) {
  if (!consan_arch_has_selectable_vgpr_bank(arch))
    return 0u;
  return amdgpu::vgpr_msb_bank_for_role(static_cast<uint8_t>(packed_mode), role).value_or(0u);
}

} // namespace

std::optional<uint16_t> flat_check_trap_compare_vgpr(const ConSanProgramSite &access) {
  if (access.kind == ConSanLdsAccessKind::Read)
    return access.operands.destination_vgpr;
  if (access.kind == ConSanLdsAccessKind::Write)
    return access.operands.data_vgpr;
  return std::nullopt;
}

std::optional<uint16_t> check_trap_compare_vgpr(const ConSanProgramSite &access,
                                                uint16_t chunk_index, rj_code_arch_t arch,
                                                uint16_t selectable_vgpr_bank_mode) {
  if (access.kind == ConSanLdsAccessKind::Read) {
    const uint32_t dst_bank =
        selected_vgpr_bank(arch, selectable_vgpr_bank_mode, amdgpu::VgprMsbRole::Dst);
    const uint32_t physical_vgpr =
        dst_bank * 256u + access.operands.destination_vgpr.value_or(0u) + chunk_index;
    const uint32_t register_limit = consan_arch_has_selectable_vgpr_bank(arch) ? 1024u : 256u;
    return access.operands.destination_vgpr && physical_vgpr < register_limit
               ? std::optional<uint16_t>(static_cast<uint16_t>(physical_vgpr))
               : std::nullopt;
  }
  if (access.kind == ConSanLdsAccessKind::Write) {
    const uint32_t source_bank =
        selected_vgpr_bank(arch, selectable_vgpr_bank_mode, amdgpu::VgprMsbRole::Src1);
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
    const uint32_t physical_vgpr = source_bank * 256u + base.value_or(0u) + relative_index;
    const uint32_t register_limit = consan_arch_has_selectable_vgpr_bank(arch) ? 1024u : 256u;
    return base && physical_vgpr < register_limit
               ? std::optional<uint16_t>(static_cast<uint16_t>(physical_vgpr))
               : std::nullopt;
  }
  return std::nullopt;
}

} // namespace rocjitsu
