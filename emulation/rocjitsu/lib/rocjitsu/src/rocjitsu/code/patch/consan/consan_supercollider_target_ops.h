// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_target_ops.h
/// @brief Exact target operations consumed by SuperCollider lowering.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

struct ConSanScTwoAddressLdsByteOffsets {
  uint32_t first = 0;
  uint32_t second = 0;
};

[[nodiscard]] std::optional<uint32_t> consan_sc_build_wait_dscnt(uint16_t count,
                                                                 rj_code_arch_t arch);

[[nodiscard]] std::optional<uint32_t>
consan_sc_delay_instruction_word_count(const ConSanOptions &options,
                                       std::vector<std::string> &errors, std::string_view context);

[[nodiscard]] bool consan_sc_append_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                                const ConSanOptions &options,
                                                std::vector<std::string> &errors,
                                                std::string_view context);

[[nodiscard]] std::optional<uint32_t> consan_sc_build_v_cmp_ne_u32(uint16_t src0, uint16_t vsrc1,
                                                                   rj_code_arch_t arch);

[[nodiscard]] std::optional<uint32_t> consan_sc_build_v_cmp_ne_u16(uint16_t src0, uint16_t vsrc1,
                                                                   rj_code_arch_t arch);

[[nodiscard]] std::optional<uint32_t>
consan_sc_build_ds_load_word0(const ConSanAccessLoweringForm &form, uint32_t original_word0,
                              rj_code_arch_t arch);

[[nodiscard]] uint32_t consan_sc_build_ds_load_word1(uint16_t addr_vgpr, uint16_t dst_vgpr);

[[nodiscard]] std::optional<ConSanScTwoAddressLdsByteOffsets>
consan_sc_two_address_lds_byte_offsets(const ConSanAccessLoweringForm &form,
                                       uint32_t original_word0);

} // namespace rocjitsu
