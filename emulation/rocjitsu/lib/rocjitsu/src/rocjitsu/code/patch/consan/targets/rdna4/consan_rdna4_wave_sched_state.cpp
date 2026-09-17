// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_rdna4_wave_sched_state.cpp
/// @brief RDNA4 guest wave-scheduling instruction state.

#include "rocjitsu/code/patch/consan/targets/consan_wave_sched_state.h"

#include "rocjitsu/code/builders/instruction_builder.h"

#include <cstring>

namespace rocjitsu::consan {

std::optional<uint16_t> wave_sched_mode_at(rj_code_arch_t arch, std::span<const uint8_t> bytes,
                                           uint64_t text_file_offset,
                                           uint64_t container_entry_text_offset,
                                           uint64_t site_file_offset) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || text_file_offset > bytes.size() ||
      container_entry_text_offset > bytes.size() - text_file_offset) {
    return std::nullopt;
  }
  const uint64_t container_file_offset = text_file_offset + container_entry_text_offset;
  if (container_file_offset > site_file_offset || site_file_offset > bytes.size())
    return std::nullopt;

  constexpr uint16_t kWaveSchedModeHwreg = 26u | ((2u - 1u) << 11u);
  constexpr uint32_t kSetreg = pack_sopk(/*op=*/19u, /*sdst=*/0u, kWaveSchedModeHwreg);
  uint16_t mode = 0u;
  for (uint64_t offset = container_file_offset; offset + 2u * sizeof(uint32_t) <= site_file_offset;
       offset += sizeof(uint32_t)) {
    uint32_t word = 0u;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if (word != kSetreg)
      continue;
    uint32_t literal = 0u;
    std::memcpy(&literal, bytes.data() + offset + sizeof(uint32_t), sizeof(literal));
    mode = static_cast<uint16_t>(literal & 0x3u);
  }
  return mode;
}

} // namespace rocjitsu::consan
