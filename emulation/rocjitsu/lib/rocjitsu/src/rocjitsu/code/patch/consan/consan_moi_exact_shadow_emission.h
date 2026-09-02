// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Target-neutral packed-field operations shared by Sampled and InlineShadow.
[[nodiscard]] bool append_add_shifted_vgpr_field(std::vector<uint32_t> &words,
                                                 uint16_t destination_vgpr, uint16_t field_vgpr,
                                                 uint16_t shift, uint32_t mask,
                                                 uint16_t temporary_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool append_add_literal_field(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                            uint32_t value, uint16_t temporary_vgpr,
                                            rj_code_arch_t arch);

[[nodiscard]] bool append_extract_exact_shadow_field(std::vector<uint32_t> &words,
                                                     uint16_t destination_vgpr,
                                                     uint16_t packed_vgpr, uint16_t shift,
                                                     uint32_t mask, rj_code_arch_t arch);

[[nodiscard]] bool
append_extract_exact_shadow_generation(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                       uint16_t packed_low_vgpr, uint16_t packed_high_vgpr,
                                       uint16_t temporary_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool append_add_exact_shadow_generation(std::vector<uint32_t> &words,
                                                      uint16_t packed_low_vgpr,
                                                      uint16_t packed_high_vgpr,
                                                      uint16_t generation_vgpr,
                                                      uint16_t temporary_vgpr, rj_code_arch_t arch);

/// Exact owner representation selected for one InlineShadow body. At most one
/// vector or persistent-scalar source is present; otherwise private derivation
/// consumes the resolved semantic owner source and resident-wave scalar.
struct MoiInlineShadowOwnerFieldPlan {
  std::optional<uint16_t> owner_vgpr;
  std::optional<uint16_t> persistent_owner_sgpr;
  bool automatic_private_epoch = false;
  ConSanMoiOwnerSource private_owner_source = ConSanMoiOwnerSource::Automatic;
  std::optional<uint16_t> resident_wave_owner_sgpr;
  bool borrowed_resident_wave_owner_sgpr = false;
};

/// Exact epoch representation selected for one InlineShadow body.
struct MoiInlineShadowEpochFieldPlan {
  std::optional<uint16_t> epoch_vgpr;
  std::optional<uint16_t> persistent_epoch_sgpr;
  bool automatic_private_epoch = false;
};

[[nodiscard]] bool append_inline_shadow_owner_field(
    std::vector<uint32_t> &words, const MoiInlineShadowOwnerFieldPlan &plan, uint16_t low_vgpr,
    uint16_t temporary_vgpr, uint16_t owner_backup_vgpr, rj_code_arch_t arch,
    const std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    std::vector<std::string> &errors);

[[nodiscard]] bool append_inline_shadow_epoch_field(std::vector<uint32_t> &words,
                                                    const MoiInlineShadowEpochFieldPlan &plan,
                                                    std::optional<uint32_t> private_epoch_offset,
                                                    uint16_t low_vgpr, uint16_t temporary_vgpr,
                                                    rj_code_arch_t arch,
                                                    std::vector<std::string> &errors);

[[nodiscard]] consan_detail::MoiWorkgroupKeyRegisterPlan
moi_workgroup_key_register_plan(const ConSanMoiOperatingPoint &point);

[[nodiscard]] bool
append_inline_workgroup_key(std::vector<uint32_t> &words, const ConSanMoiWorkgroupSources &sources,
                            const consan_detail::MoiWorkgroupKeyRegisterPlan &registers,
                            uint16_t key_vgpr, uint16_t coordinate_vgpr, uint16_t value_vgpr,
                            uint16_t original_exec_save_offset, rj_code_arch_t arch);

[[nodiscard]] bool append_inline_acquired_token_slot_address(
    std::vector<uint32_t> &words, uint64_t table_base, uint32_t table_capacity,
    uint16_t workgroup_key_vgpr, uint16_t consumer_owner_vgpr, uint16_t producer_owner_vgpr,
    uint16_t consumer_epoch_vgpr, uint16_t slot_address_vgpr, uint16_t hash_vgpr,
    uint16_t temporary_vgpr, bool release_sequence, rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_impl
