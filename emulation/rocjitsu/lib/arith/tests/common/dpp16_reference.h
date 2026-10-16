// SPDX-License-Identifier: MIT

// Test-side DPP16 control encoding and lane-routing reference.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"
#include <stdint.h>

namespace rocjitsu::arith::test {

// Return whether ctrl is one of the DPP16 encodings exercised by the reference.
inline bool dpp16_control(unsigned ctrl) {
  return ctrl < 256 || (ctrl > 0x100 && ctrl < 0x110) || (ctrl > 0x110 && ctrl < 0x120) ||
         (ctrl > 0x120 && ctrl < 0x130) || ctrl == 0x140 || ctrl == 0x141 ||
         (ctrl >= 0x150 && ctrl < 0x170);
}

// Encode DPP16 routing, inactive-source policy, and row/bank masks.
inline uint64_t dpp16_mode(unsigned ctrl, unsigned fi, unsigned bound, int rows, unsigned banks) {
  return RJ_ARITH_DPP16 | (fi ? RJ_ARITH_DPP_FI : 0) | (bound ? RJ_ARITH_DPP_BOUND_CTRL : 0) |
         (uint64_t(ctrl) << RJ_ARITH_DPP_CTRL_SHIFT) | (uint64_t(rows) << RJ_ARITH_DPP_ROW_SHIFT) |
         (uint64_t(banks) << RJ_ARITH_DPP_BANK_SHIFT);
}

// Returns whether the destination is written; source is -1 for a zero input.
template <typename Mask>
inline bool dpp16_reference(unsigned ctrl, bool fi, bool bound, int rows, unsigned banks,
                            Mask exec_mask, int lane, int &source) {
  int sub = int(lane % 16), selected = sub;
  if (ctrl < 256) {
    selected = (sub / 4) * 4 + int((ctrl >> (2 * (sub % 4))) & 3);
  } else if (ctrl < 0x110) {
    selected = sub + int(ctrl - 0x100);
  } else if (ctrl < 0x120) {
    selected = sub - int(ctrl - 0x110);
  } else if (ctrl < 0x130) {
    selected = (sub - int(ctrl - 0x120) + 16) % 16;
  } else if (ctrl == 0x140) {
    selected = 15 - sub;
  } else if (ctrl == 0x141) {
    selected = (sub / 8) * 8 + 7 - sub % 8;
  } else if (ctrl < 0x160) {
    selected = int(ctrl - 0x150);
  } else {
    selected = sub ^ int(ctrl - 0x160);
  }
  source = selected < 0 || selected >= 16 ? -1 : int(lane / 16) * 16 + selected;
  if (source >= 0 && !fi && !(exec_mask & (1ULL << source))) {
    source = -1;
  }
  return (exec_mask & (1ULL << lane)) && (rows & (1u << (lane / 16))) &&
         (banks & (1u << ((lane % 16) / 4))) && (source >= 0 || bound);
}

} // namespace rocjitsu::arith::test
