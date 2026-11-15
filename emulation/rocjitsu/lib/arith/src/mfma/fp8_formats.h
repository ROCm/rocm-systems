// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA MFMA FP8 widening: gfx942 FNUZ versus gfx950 OCP encodings.
#pragma once
#include "common/float_formats.h"
#include "common/internal.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Widen one source byte, preserving the legacy architecture-specific special values.
template <bool Bf8, bool Fnuz> inline float mfma_decode_fp8(uint8_t bits) {
  if constexpr (!Fnuz) {
    return fp8_to_float<Bf8>(bits);
  }
  if (bits == 0x80) {
    return as_float(0xffc00000U);
  }
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 16 : 8;
  uint32_t sign = uint32_t(bits & 0x80) << 24;
  uint32_t exponent = (bits & 0x7f) >> fraction_bits;
  uint32_t fraction = bits & ((1 << fraction_bits) - 1);
  if (!exponent) {
    float magnitude = float(fraction) * (Bf8 ? 0x1p-17f : 0x1p-10f);
    return as_float(as_bits(magnitude) | sign);
  }
  return as_float(sign | ((exponent + 127 - bias) << 23) | (fraction << (23 - fraction_bits)));
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
