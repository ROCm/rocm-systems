// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/trampoline_builder.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
class Decoder;
}

namespace rocjitsu::consan_moi_detail {

[[nodiscard]] std::optional<std::vector<uint32_t>>
decode_relocatable_entry_instruction(std::span<const uint8_t> text, uint64_t offset,
                                     rj_code_arch_t arch, std::vector<std::string> &errors,
                                     Decoder *reusable_decoder = nullptr);

[[nodiscard]] uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset,
                                         rj_code_arch_t arch);

void append_word_bytes(std::vector<uint8_t> &bytes, uint32_t word);
void append_words_bytes(std::vector<uint8_t> &bytes, std::span<const uint32_t> words);

[[nodiscard]] std::optional<DbiPatchPlacement>
plan_prebuilt_appended_cave(DbiPatchPlacementPlanner &planner, uint64_t anchor_offset,
                            uint32_t original_size, std::span<const uint32_t> cave_words,
                            std::vector<std::string> &errors, std::string_view probe_name);

} // namespace rocjitsu::consan_moi_detail
