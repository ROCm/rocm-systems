// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wave-width types and representative EXEC masks for family test runners.

#pragma once

#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

template <int Lanes> using WaveMask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;

// A few masks distinguish full/empty waves, isolated extreme lanes, and holes.
template <int Lanes> WaveMask<Lanes> exec_mask_for_case(int scenario) {
  using Mask = WaveMask<Lanes>;
  switch (scenario % 5) {
  case 1:
    return 0;
  case 2:
    return Mask(1) | (Mask(1) << (Lanes - 1));
  case 3:
    return Mask(0xaaaaaaaaaaaaaaaaULL);
  case 4:
    return Mask(0x00ff00ff00ff00ffULL);
  default:
    return ~Mask(0);
  }
}

} // namespace rocjitsu::arith::test
