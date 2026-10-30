// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local FP64 register decoding, rounding, and output modifier helpers.
#pragma once

#include "common/internal.h"
#include "fp/fp64.h"
#include "rj_arith/rj_arith.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Identify FP64 selections, which preserve selected subnormal outputs.
constexpr bool is_selection(Fp64 op) {
  return op == Fp64::MinLegacy || op == Fp64::MaxLegacy || op == Fp64::MinNum ||
         op == Fp64::MaxNum || op == Fp64::Minimum || op == Fp64::Maximum;
}

// Return the number of source operands required by the selected FP64 operation.
constexpr int fp64_sources(Fp64 op) {
  switch (op) {
  case Fp64::Add:
  case Fp64::Mul:
  case Fp64::MinLegacy:
  case Fp64::MaxLegacy:
  case Fp64::MinNum:
  case Fp64::MaxNum:
  case Fp64::Minimum:
  case Fp64::Maximum:
    return 2;
  case Fp64::Fma:
    return 3;
  default:
    return 1;
  }
}

// Flush FP64 subnormals to zero of the same sign without touching host FP state.
inline double fp64_flush_denorm(double value) {
  uint64_t raw = double_bits(value);
  return (raw & 0x7fffffffffffffffULL) < 0x0010000000000000ULL
             ? as_double(raw & 0x8000000000000000ULL)
             : value;
}

// Round to an integral FP64 value, ties to even, preserving signed zero and
// quieting NaNs. Adapted from the rocjitsu-derived FP32 rndne helper.
inline double fp64_rndne(double value) {
  uint64_t raw = double_bits(value), magnitude = raw & 0x7fffffffffffffffULL;
  uint64_t sign = raw & 0x8000000000000000ULL;
  if (magnitude >= 0x7ff0000000000000ULL) {
    return as_double(raw | (magnitude > 0x7ff0000000000000ULL ? 0x0008000000000000ULL : 0));
  }
  int exponent = int(magnitude >> 52);
  if (exponent >= 1075) {
    return value;
  }
  if (exponent < 1023) {
    return as_double(sign | (magnitude > 0x3fe0000000000000ULL ? 0x3ff0000000000000ULL : 0));
  }
  uint64_t unit = 1ULL << (1075 - exponent);
  uint64_t fraction = magnitude & (unit - 1), rounded = magnitude & ~(unit - 1);
  if (fraction > unit / 2 || (fraction == unit / 2 && (rounded & unit))) {
    rounded += unit;
  }
  return as_double(sign | rounded);
}

// Each FP64 lane is split into low/high words in two independent VGPR buffers.
inline double fp64_input(const uint32_t *const *v, int lane, uint32_t mode) {
  uint64_t bits = v[0][lane] | (uint64_t(v[1][lane]) << 32);
  if (mode & RJ_ARITH_ALU_ABS_A) {
    bits &= 0x7fffffffffffffffULL;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits ^= 0x8000000000000000ULL;
  }
  return as_double(bits);
}

// Apply FP64 output modifiers, including OMOD zero/denormal rules.
inline double fp64_output(double value, uint32_t mode) {
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    uint64_t raw = double_bits(value), magnitude = raw & 0x7fffffffffffffffULL;
    if (magnitude < 0x0010000000000000ULL) {
      value = 0;
    } else if (omod == 3 && magnitude < 0x0020000000000000ULL) {
      value = as_double(raw & 0x8000000000000000ULL);
    }
  }
  switch ((mode >> 6) & 3) {
  case 1:
    value *= 2;
    break;
  case 2:
    value *= 4;
    break;
  case 3:
    value *= 0.5;
    break;
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    value = !(value > 0) ? 0 : value > 1 ? 1 : value;
  }
  return value;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
