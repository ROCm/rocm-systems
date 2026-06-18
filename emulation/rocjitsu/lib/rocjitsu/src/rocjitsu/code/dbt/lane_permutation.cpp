// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/translation_rule.h"

#include <cassert>

namespace rocjitsu {

LanePermutation compute_lane_permutation(const LaneLayout &guest, const LaneLayout &host) {
  assert(guest.m == host.m && guest.n == host.n);
  assert(guest.wave_size == host.wave_size);

  const uint8_t num_rows = guest.m;
  const uint8_t rows_per_group = guest.dst_vgprs;
  const uint8_t lanes_per_group = guest.n; // columns = lanes within each row group

  uint8_t range_start = 255, range_end = 0;
  uint8_t xor_lane_mask = 0;

  for (uint8_t row = 0; row < num_rows; row += rows_per_group) {
    uint8_t g_lane = lane_for_row(guest, row);
    uint8_t h_lane = lane_for_row(host, row);
    if (g_lane != h_lane) {
      uint8_t xor_val = g_lane ^ h_lane;
      if (xor_lane_mask == 0)
        xor_lane_mask = xor_val;
      assert(xor_val == xor_lane_mask && "non-uniform XOR across row groups");
      range_start = std::min(range_start, std::min(g_lane, h_lane));
      range_end =
          std::max(range_end, static_cast<uint8_t>(std::max(g_lane, h_lane) + lanes_per_group));
    }
  }

  if (xor_lane_mask == 0)
    return {};

  return {static_cast<uint32_t>(xor_lane_mask) * 4, range_start, range_end};
}

} // namespace rocjitsu
