// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_gfx1250_vgpr_bank_state.cpp
/// @brief gfx1250 selectable-VGPR-bank instruction state.

#include "rocjitsu/code/patch/consan/consan_vgpr_bank_state.h"

#include <cstring>

namespace rocjitsu {

std::optional<uint16_t> consan_selectable_vgpr_bank_mode_at(rj_code_arch_t arch,
                                                            std::span<const uint8_t> bytes,
                                                            uint64_t text_file_offset,
                                                            uint64_t container_entry_text_offset,
                                                            uint64_t site_file_offset) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || text_file_offset > bytes.size() ||
      container_entry_text_offset > bytes.size() - text_file_offset) {
    return std::nullopt;
  }
  const uint64_t container_file_offset = text_file_offset + container_entry_text_offset;
  if (container_file_offset > site_file_offset || site_file_offset > bytes.size())
    return std::nullopt;
  uint16_t mode = 0;
  for (uint64_t offset = container_file_offset; offset + sizeof(uint32_t) <= site_file_offset;
       offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if ((word & 0xFFFF0000u) == 0xBF860000u)
      // The low byte is the mode established by this instruction. The high
      // byte records the previous mode and must not become persistent state.
      mode = static_cast<uint16_t>(word & 0xFFu);
  }
  return mode;
}

} // namespace rocjitsu
