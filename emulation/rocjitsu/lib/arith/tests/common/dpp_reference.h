// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Test-side DPP8/DPP16 routing and destination participation rules.

#pragma once

#include "common/dpp16_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// DPP8 reverse with FI off/on; DPP16 row shift with all FI/BOUND pairs;
// and row XOR with restricted row/bank enables.
inline const uint64_t dpp_modes[] = {
    0x539770100000000ULL, 0x539770300000000ULL, 0x1ff010400000000ULL, 0x1ff010c00000000ULL,
    0x1ff010600000000ULL, 0x1ff010e00000000ULL, 0xa3670c00000000ULL};

// Return whether this destination lane writes, and set its routed source index; -1 denotes a
// zero source.
template <typename Mask>
inline bool dpp_source(uint64_t mode, Mask exec_mask, int lane, int &source) {
  if (mode & RJ_ARITH_DPP8) {
    unsigned selectors = unsigned(mode >> RJ_ARITH_DPP8_SELECT_SHIFT);
    source = int(lane / 8) * 8 + int((selectors >> (3 * (lane % 8))) & 7);
    if (!(mode & RJ_ARITH_DPP_FI) && !(exec_mask & (1ULL << source))) {
      source = -1;
    }
    return (exec_mask & (1ULL << lane)) != 0;
  }
  return dpp16_reference(unsigned((mode & RJ_ARITH_DPP_CTRL_MASK) >> RJ_ARITH_DPP_CTRL_SHIFT),
                         mode & RJ_ARITH_DPP_FI, mode & RJ_ARITH_DPP_BOUND_CTRL,
                         unsigned((mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT),
                         unsigned((mode & RJ_ARITH_DPP_BANK_MASK) >> RJ_ARITH_DPP_BANK_SHIFT),
                         exec_mask, lane, source);
}

} // namespace rocjitsu::arith::test
