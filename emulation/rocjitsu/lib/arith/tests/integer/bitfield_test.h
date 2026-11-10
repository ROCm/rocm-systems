// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bitfield extraction/insertion and word alignment.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/word_test.h"
#include "integer/bitfield_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check bitfield operations with routing and aliases for the supplied entry point.
template <int Lanes, int Operands, typename Call>
void check_bitfield(Call call, Bitfield operation) {
  auto input = make_integer_inputs<Lanes>();
  if (operation == Bitfield::Perm) {
    // Cover every byte/sign selector; arbitrary selectors mostly select 0xff.
    for (int lane = 0; lane < 16; ++lane) {
      input.v[0][lane] = 0x807f0180U;
      input.v[2][lane] = 0x7f808001U;
      input.v[4][lane] = 0x03020100U + uint32_t(lane % 4) * 0x04040404U;
    }
  }
  check_integer_word<Lanes, Operands, true>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return bitfield_reference(operation, a, b, c); },
      RJ_ARITH_ALU_CLAMP, false, input);
}

} // namespace rocjitsu::arith::test
