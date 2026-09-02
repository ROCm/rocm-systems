// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::optional<uint16_t> moi_descriptor_owner_shift(std::span<const uint8_t> image,
                                                                 uint64_t descriptor_file_offset,
                                                                 rj_code_arch_t arch,
                                                                 std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
