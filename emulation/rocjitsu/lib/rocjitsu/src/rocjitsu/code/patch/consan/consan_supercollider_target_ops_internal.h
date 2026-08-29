// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_target_ops_internal.h
/// @brief Family/member-owned recipes behind SuperCollider target operations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops.h"

namespace rocjitsu::consan_sc_target_detail {

[[nodiscard]] std::optional<uint32_t> build_gfx9_ds_load_word0(const ConSanAccessLoweringForm &form,
                                                               uint32_t original_word0);

[[nodiscard]] std::optional<uint32_t>
build_gfx11_gfx12_ds_load_word0(const ConSanAccessLoweringForm &form, uint32_t original_word0);

[[nodiscard]] std::optional<std::array<uint32_t, 2>>
build_gfx9_cdna_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr,
                                 ConSanEncodingFamily family);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_classic_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_rdna4_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_gfx1250_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_gfx9_cdna_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                     uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_rdna3_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_rdna4_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst);

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                   uint16_t vdst);

} // namespace rocjitsu::consan_sc_target_detail
