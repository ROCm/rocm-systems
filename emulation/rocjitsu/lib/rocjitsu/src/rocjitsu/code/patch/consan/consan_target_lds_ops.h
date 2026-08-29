// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_target_lds_ops.h
/// @brief Target-operation interface for normalized LDS access recipes.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu {

/// Target-neutral operands for splitting one two-address LDS operation into
/// two single-address operations after relocation changes its address base.
struct ConSanSplitTwoAddressLdsRequest {
  uint32_t first_byte_offset = 0;
  uint32_t second_byte_offset = 0;
  uint16_t element_dwords = 0;
  uint16_t address_vgpr = 0;
  uint16_t first_data_vgpr = 0;
  uint16_t second_data_vgpr = 0;
  std::optional<uint16_t> adjusted_address_vgpr;
  bool load = false;
};

/// Build the target's normalized split-access sequence, or no value when the
/// target has no such recipe or the operands cannot be represented.
[[nodiscard]] std::optional<std::vector<uint32_t>>
consan_build_split_two_address_lds_pair(const ConSanSplitTwoAddressLdsRequest &request,
                                        rj_code_arch_t arch);

} // namespace rocjitsu
