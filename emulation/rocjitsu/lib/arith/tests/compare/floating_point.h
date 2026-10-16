// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Raw floating-point classification, decoding, and comparison input generation.

#pragma once

#include "common/floating_point.h"
#include "common/integer.h"
#include "common/registers.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <limits>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Return the positive-infinity encoding for the selected floating-point width.
template <int Width> constexpr uint64_t compare_infinity() {
  return Width == 16 ? 0x7c00ULL : Width == 32 ? 0x7f800000ULL : 0x7ff0000000000000ULL;
}

// Return the NaN quiet-bit mask for the selected floating-point width.
template <int Width> constexpr uint64_t compare_quiet_bit() {
  return Width == 16 ? 0x200ULL : Width == 32 ? 0x400000ULL : 0x8000000000000ULL;
}

// Ten IEEE classes in ISA class-mask order. Used as deliberate cases among
// pseudorandom encodings; negative/positive zero and signaling/quiet NaN differ.
template <int Width> uint64_t compare_class_value(int index) {
  constexpr uint64_t sign = 1ULL << (Width - 1);
  constexpr uint64_t inf = compare_infinity<Width>();
  const uint64_t values[] = {inf | 1,    inf | compare_quiet_bit<Width>() | 1,
                             sign | inf, sign | (inf >> 1),
                             sign | 1,   sign,
                             0,          1,
                             inf >> 1,   inf};
  return values[index];
}

// Raw pseudorandom words cover full significands. Deliberate pairs cover each
// class, equal operands, and signed-zero equality in both halves of FP16.
template <int Width, int Lanes> Registers<4, Lanes> make_compare_float_inputs() {
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    for (int half = 0; half < (Width == 16 ? 2 : 1); ++half) {
      if (lane < 20) {
        write_compare_operand(input.v, lane, Width, compare_class_value<Width>((lane + half) % 10),
                              half);
        write_compare_operand(input.v + 2, lane, Width,
                              compare_class_value<Width>((lane / 2 + half) % 10), half);
      } else if (lane % 4 == 0) {
        write_compare_operand(input.v + 2, lane, Width, compare_operand(input.v, lane, Width, half),
                              half);
      }
    }
  }
  write_compare_operand(input.v, 20, Width, 0);
  write_compare_operand(input.v + 2, 20, Width, 1ULL << (Width - 1));
  return input;
}

// Exact IEEE ordering needs no rounded arithmetic. This computes the same
// relations as the old suite's raw-bit oracle and rocjitsu shared/comparison.h:
// flush first, signed zeros equal, ordered predicates false for NaNs. Convert
// NaNs to a quiet host NaN, leaving signaling classification to status tests.
template <int Width>
double compare_float_value(uint64_t bits, uint64_t mode, int operand, bool flush) {
  constexpr uint64_t sign = 1ULL << (Width - 1), inf = compare_infinity<Width>();
  if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
    bits &= ~sign;
  }
  if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
    bits ^= sign;
  }
  if (flush && !(bits & inf)) {
    bits &= sign;
  }
  if ((bits & (sign - 1)) > inf) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if constexpr (Width == 16) {
    return f16_value(uint16_t(bits));
  } else if constexpr (Width == 32) {
    return bit_cast<float>(uint32_t(bits));
  } else {
    return bit_cast<double>(bits);
  }
}

} // namespace rocjitsu::arith::test
