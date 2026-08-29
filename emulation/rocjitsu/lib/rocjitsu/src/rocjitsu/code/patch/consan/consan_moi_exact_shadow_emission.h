// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

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

} // namespace rocjitsu::consan_moi_impl
