// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_target_ops_internal.h
/// @brief Family/member-owned recipes behind SuperCollider target operations.

#pragma once

#include "rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops.h"

namespace rocjitsu::consan_sc_target_detail {

[[nodiscard]] std::optional<uint32_t>
build_cdna3_cdna4_ds_load_word0(const ConSanAccessLoweringForm &form, uint32_t original_word0);

[[nodiscard]] std::optional<ConSanScDirectToLdsTransfer> build_cdna3_cdna4_direct_to_lds_transfer(
    std::array<uint32_t, 2> original_words, uint32_t width_bits, uint16_t address_vgpr,
    uint16_t payload_vgpr, uint16_t readback_vgpr, rj_code_arch_t arch);

[[nodiscard]] std::optional<uint32_t>
build_rdna3_rdna4_cdna5_ds_load_word0(const ConSanAccessLoweringForm &form,
                                      uint32_t original_word0);

[[nodiscard]] std::optional<std::array<uint32_t, 2>>
build_cdna3_cdna4_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_cdna3_cdna4_rdna3_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_rdna4_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_cdna5_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_cdna3_cdna4_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                       uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_rdna3_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_rdna4_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_cdna5_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst);

} // namespace rocjitsu::consan_sc_target_detail
