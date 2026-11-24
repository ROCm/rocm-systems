// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/risc_v/hart_state.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::risc_v::detail {

// The signed and unsigned W/L conversions round before testing the destination
// range. Every binary32 input is exactly representable in binary64.
template <typename Int>
void convert_fp_to_integer(HartState &hart, int rd, double value, unsigned rm) {
  if (rm == 7) {
    rm = hart.read_csr(kCsrFrm);
  }

  double rounded;
  switch (rm) {
  case 0:   // RNE
  case 4: { // RMM
    // modf splits exactly regardless of the host rounding mode. A value with a
    // fractional part is small enough that adding one to its integer part is exact.
    const double fraction = std::abs(std::modf(value, &rounded));
    if (fraction > 0.5 || (fraction == 0.5 && (rm == 4 || std::fmod(rounded, 2.0) != 0))) {
      rounded += std::copysign(1.0, value);
    }
    break;
  }
  case 2: // RDN
    rounded = std::floor(value);
    break;
  case 3: // RUP
    rounded = std::ceil(value);
    break;
  default: // RTZ; retain truncation for reserved rounding modes.
    rounded = std::trunc(value);
    break;
  }

  using Limits = std::numeric_limits<Int>;
  // max()+1 is an exact power of two even when max() would round up in double.
  const double upper = std::ldexp(1.0, Limits::digits);
  constexpr uint8_t invalid = 0x10, inexact = 0x01;
  Int result;
  if (std::isnan(rounded) || rounded >= upper) {
    result = Limits::max();
    hart.fcsr |= invalid;
  } else if (rounded < static_cast<double>(Limits::lowest())) {
    result = Limits::lowest();
    hart.fcsr |= invalid;
  } else {
    result = static_cast<Int>(rounded);
    if (rounded != value) {
      hart.fcsr |= inexact;
    }
  }

  // Both W and WU sign-extend the low word to XLEN.
  if constexpr (sizeof(Int) == sizeof(int32_t)) {
    hart.write_xreg(rd, sext32(static_cast<int32_t>(result)));
  } else {
    hart.write_xreg(rd, static_cast<int64_t>(result));
  }
}

} // namespace rocjitsu::risc_v::detail
