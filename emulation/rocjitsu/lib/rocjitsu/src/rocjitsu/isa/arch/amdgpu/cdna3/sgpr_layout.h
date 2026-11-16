// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>

namespace rocjitsu::cdna3 {

/// gfx942 encodes SGPR use in units of eight and allocates at least one 16-SGPR block.
constexpr uint32_t physical_sgpr_count(uint32_t requested) {
  return std::max(16u, (requested + 15u) & ~15u);
}

/// VCC (end-2) and FLAT_SCRATCH (end-6) follow the allocation up through s107.
constexpr uint32_t sgpr_alias_end(uint32_t requested) {
  return std::min(108u, physical_sgpr_count(requested));
}

/// Low words of the named pairs in the physical SGPR block ending at alias_end.
constexpr uint16_t vcc_base(uint32_t alias_end) { return static_cast<uint16_t>(alias_end - 2); }
constexpr uint16_t flat_scratch_base(uint32_t alias_end) {
  return static_cast<uint16_t>(alias_end - 6);
}

/// Whether either dword in an SGPR pair aliases VCC or FLAT_SCRATCH in this allocation.
constexpr bool sgpr_pair_overlaps_vcc_or_flat_scratch(uint32_t requested, uint16_t pair_base) {
  const uint32_t end = sgpr_alias_end(requested);
  const auto overlaps = [pair_base](uint32_t alias_base) {
    return pair_base < alias_base + 2 && uint32_t{pair_base} + 2 > alias_base;
  };
  return overlaps(vcc_base(end)) || overlaps(flat_scratch_base(end));
}

} // namespace rocjitsu::cdna3
