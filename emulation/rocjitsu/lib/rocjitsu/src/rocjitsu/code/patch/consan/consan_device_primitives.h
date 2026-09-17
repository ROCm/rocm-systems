// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"

#include <cstddef>
#include <vector>

namespace rocjitsu::consan::detail {

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch);
/// Select the lowest lane named by an existing EXEC mask. The current EXEC is
/// narrowed to that lane and its incoming value is retained in
/// `saved_exec_sgpr`.
[[nodiscard]] bool append_select_first_lane_in_exec_mask(std::vector<uint32_t> &words,
                                                         uint16_t lane_rank_vgpr,
                                                         uint16_t active_exec_sgpr,
                                                         uint16_t saved_exec_sgpr,
                                                         rj_code_arch_t arch);
} // namespace rocjitsu::consan::detail
