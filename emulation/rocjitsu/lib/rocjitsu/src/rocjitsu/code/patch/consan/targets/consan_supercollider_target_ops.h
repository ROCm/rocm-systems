// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_target_ops.h
/// @brief Exact target operations consumed by SuperCollider lowering.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::consan {

struct SuperColliderTwoAddressLdsByteOffsets {
  uint32_t first = 0;
  uint32_t second = 0;
};

/// Target encoding of one direct-to-LDS write expanded into an ordinary
/// global load, an explicit LDS write, and an LDS readback. SuperCollider
/// retains the loaded payload as the comparison reference.
struct SuperColliderDirectToLdsTransfer {
  std::array<uint32_t, 2> global_load;
  std::array<uint32_t, 2> lds_write;
  std::array<uint32_t, 2> lds_readback;
};

[[nodiscard]] std::optional<uint32_t>
supercollider_build_guest_flat_completion_wait(rj_code_arch_t arch);

[[nodiscard]] std::optional<std::vector<uint32_t>>
supercollider_build_delay_words(const TargetProfile &target, const Request &request,
                                uint16_t temporary_sgpr, std::vector<std::string> &errors,
                                std::string_view context);

[[nodiscard]] std::optional<uint32_t>
supercollider_build_v_cmp_ne_u32(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch);

[[nodiscard]] std::optional<uint32_t>
supercollider_build_v_cmp_ne_u16(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch);

[[nodiscard]] std::optional<uint32_t>
supercollider_build_ds_load_word0(const AccessLoweringForm &form, uint32_t original_word0,
                                  rj_code_arch_t arch);

[[nodiscard]] uint32_t supercollider_build_ds_load_word1(uint16_t addr_vgpr, uint16_t dst_vgpr);

[[nodiscard]] std::optional<SuperColliderDirectToLdsTransfer>
supercollider_build_direct_to_lds_transfer(std::array<uint32_t, 2> original_words,
                                           uint32_t width_bits, uint16_t address_vgpr,
                                           uint16_t payload_vgpr, uint16_t readback_vgpr,
                                           rj_code_arch_t arch);

[[nodiscard]] std::optional<SuperColliderTwoAddressLdsByteOffsets>
supercollider_two_address_lds_byte_offsets(const AccessLoweringForm &form, uint32_t original_word0);

[[nodiscard]] std::optional<std::array<uint32_t, 2>>
supercollider_build_cdna_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr,
                                          const TargetProfile &target);

[[nodiscard]] std::optional<std::vector<uint32_t>>
supercollider_build_split_single_address_lds_pair(SuperColliderTwoAddressLdsByteOffsets offsets,
                                                  uint16_t element_dwords, uint16_t address_vgpr,
                                                  uint16_t first_data_vgpr,
                                                  uint16_t second_data_vgpr,
                                                  uint16_t adjusted_address_vgpr, bool load,
                                                  const TargetProfile &target);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst,
                                rj_code_arch_t arch);

} // namespace rocjitsu::consan
