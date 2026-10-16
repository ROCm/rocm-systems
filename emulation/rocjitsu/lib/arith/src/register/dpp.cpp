// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DPP8/DPP16 source routing, encoding validation, and effective EXEC masks.
// Keeps this shared policy independent of instruction-specific register layouts.

// Inactive-source behavior follows rocjitsu's shared DPP execution plans.

#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Fill a routed 32-lane source snapshot using DPP8 selectors and inactive-source rules.
void dpp8_source(uint32_t exec_mask, uint64_t mode, uint32_t *out, const uint32_t *a) {
  uint32_t selectors = uint32_t(mode >> RJ_ARITH_DPP8_SELECT_SHIFT);
  bool fi = (mode & RJ_ARITH_DPP_FI) != 0;
  for (int lane = 0; lane < 32; ++lane) {
    int src = (lane & ~7u) | ((selectors >> (3 * (lane & 7))) & 7);
    out[lane] = (fi || ((exec_mask >> src) & 1)) ? a[src] : 0;
  }
}

// Return whether the high instruction-mode bits encode a supported DPP operation.
bool valid_dpp(uint64_t mode) {
  uint64_t high = mode & ~0xffffffffULL;
  if (!high) {
    return true;
  }
  if (mode & RJ_ARITH_DPP8) {
    return !(high & ~(RJ_ARITH_DPP8 | RJ_ARITH_DPP_FI | RJ_ARITH_DPP8_SELECT_MASK));
  }
  if (!(mode & RJ_ARITH_DPP16) ||
      (high & ~(RJ_ARITH_DPP16 | RJ_ARITH_DPP_FI | RJ_ARITH_DPP_BOUND_CTRL |
                RJ_ARITH_DPP_CTRL_MASK | RJ_ARITH_DPP_ROW_MASK | RJ_ARITH_DPP_BANK_MASK))) {
    return false;
  }
  unsigned ctrl = unsigned((mode & RJ_ARITH_DPP_CTRL_MASK) >> RJ_ARITH_DPP_CTRL_SHIFT);
  return ctrl <= 0xff || (ctrl >= 0x101 && ctrl <= 0x10f) || (ctrl >= 0x111 && ctrl <= 0x11f) ||
         (ctrl >= 0x121 && ctrl <= 0x12f) || ctrl == 0x140 || ctrl == 0x141 ||
         (ctrl >= 0x150 && ctrl <= 0x16f);
}

// Fill a DPP16 source snapshot and return the effective destination EXEC mask.
uint32_t dpp16_source(uint32_t exec_mask, uint64_t mode, uint32_t *out, const uint32_t *a) {
  unsigned ctrl = unsigned((mode & RJ_ARITH_DPP_CTRL_MASK) >> RJ_ARITH_DPP_CTRL_SHIFT);
  unsigned valid_row = 0xffff;
  if (ctrl >= 0x101 && ctrl <= 0x10f) {
    valid_row >>= ctrl & 15;
  }
  if (ctrl >= 0x111 && ctrl <= 0x11f) {
    valid_row = (valid_row << (ctrl & 15)) & 0xffff;
  }
  bool fi = (mode & RJ_ARITH_DPP_FI) != 0;
  uint32_t readable = 0;
  {
    for (int lane = 0; lane < 32; ++lane) {
      int sub = lane & 15, index = sub;
      if (ctrl <= 0xff) {
        index = (sub & ~3u) | ((ctrl >> (2 * (sub & 3))) & 3);
      } else if (ctrl <= 0x10f) {
        index = (sub + (ctrl & 15)) & 15;
      } else if (ctrl <= 0x12f) {
        index = (sub - (ctrl & 15)) & 15;
      } else if (ctrl == 0x140) {
        index = sub ^ 15;
      } else if (ctrl == 0x141) {
        index = sub ^ 7;
      } else if (ctrl <= 0x15f) {
        index = ctrl & 15;
      } else {
        index = sub ^ (ctrl & 15);
      }
      int src = (lane & 16) | index;
      bool read = ((valid_row >> sub) & 1) && (fi || ((exec_mask >> src) & 1));
      out[lane] = read ? a[src] : 0;
      readable |= uint32_t(read) << lane;
    }
  }
  unsigned rows = unsigned((mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT);
  unsigned banks = unsigned((mode & RJ_ARITH_DPP_BANK_MASK) >> RJ_ARITH_DPP_BANK_SHIFT);
  uint32_t bank_lanes = (banks & 1) * 15 | ((banks >> 1) & 1) * 0xf0 | ((banks >> 2) & 1) * 0xf00 |
                        ((banks >> 3) & 1) * 0xf000;
  uint32_t destinations = (bank_lanes * (rows & 1)) | ((bank_lanes * ((rows >> 1) & 1)) << 16);
  return exec_mask & destinations & (mode & RJ_ARITH_DPP_BOUND_CTRL ? UINT32_MAX : readable);
}

} // namespace rocjitsu::arith
