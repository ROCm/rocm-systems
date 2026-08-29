// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <vector>

namespace rocjitsu::consan_moi_detail {

[[nodiscard]] bool append_store_u32_literal(std::vector<uint32_t> &words, uint64_t address,
                                            uint32_t value, uint16_t scratch_vgpr,
                                            rj_code_arch_t arch);
[[nodiscard]] bool append_store_u32_vgpr(std::vector<uint32_t> &words, uint64_t address,
                                         uint16_t value_vgpr, uint16_t scratch_vgpr,
                                         rj_code_arch_t arch);
[[nodiscard]] bool append_store_u32_sgpr(std::vector<uint32_t> &words, uint64_t address,
                                         uint16_t value_sgpr, uint16_t value_vgpr,
                                         uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_store_u32_vgpr_at_offset(std::vector<uint32_t> &words,
                                                   uint16_t address_vgpr, uint32_t byte_offset,
                                                   uint16_t value_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_load_u32_vgpr_at_offset(std::vector<uint32_t> &words,
                                                  uint16_t address_vgpr, uint32_t byte_offset,
                                                  uint16_t destination_vgpr, rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_detail
