// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_descriptor_growth.h
/// @brief Application of merged descriptor register extents.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class CodeObjectPatcher;

/// One kernel descriptor's requested one-past-the-end register extent.
/// Duplicate owners are merged by maximum before mutation.
struct ConSanDescriptorRegisterGrowth {
  uint64_t descriptor_file_offset = 0;
  uint16_t required_count = 0;
};

[[nodiscard]] std::unordered_map<uint64_t, uint16_t>
merge_consan_descriptor_register_growths(std::span<const ConSanDescriptorRegisterGrowth> growths);

[[nodiscard]] bool apply_consan_descriptor_sgpr_growths_to_patcher(
    CodeObjectPatcher &patcher, std::span<const ConSanDescriptorRegisterGrowth> growths,
    rj_code_arch_t arch, std::vector<std::string> &errors);

[[nodiscard]] bool apply_consan_descriptor_sgpr_growths_to_bytes(
    std::span<uint8_t> image, std::span<const ConSanDescriptorRegisterGrowth> growths,
    rj_code_arch_t arch, std::vector<std::string> &errors);

[[nodiscard]] bool apply_consan_descriptor_vgpr_growths_to_bytes(
    std::span<uint8_t> image, std::span<const ConSanDescriptorRegisterGrowth> growths,
    rj_code_arch_t arch, std::vector<std::string> &errors);

[[nodiscard]] bool apply_consan_descriptor_vgpr_growths_to_patcher(
    CodeObjectPatcher &patcher, std::span<const uint8_t> original_image,
    std::span<const ConSanDescriptorRegisterGrowth> growths, rj_code_arch_t arch,
    std::vector<std::string> &errors);

} // namespace rocjitsu
