// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file trig_preop.h
/// @brief V_TRIG_PREOP_F64: a scaled 53-bit segment of 2/pi for argument reduction.
///
/// The lookup returns the segment truncated toward zero in every MODE rounding
/// mode. Its result is never NaN, infinity or negative; it is subnormal only
/// for sources with an exponent of at least 1968. Callers apply the
/// output-denormal policy, then OMOD and CLAMP, to the returned encoding.
///
/// gfx1201 and gfx1100 captures of every captured encoding (plain, ABS, NEG,
/// CLAMP, OMOD) under every MODE setting probed match these rules bit for bit.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

#include <bit>
#include <cstdint>
#include <iterator>

namespace rocjitsu::amdgpu::trig_preop {

/// @brief The fraction bits of 2/pi, most significant first.
/// @details ISA discrepancy: the ISA reads 1201 fraction bits, but gfx1201
/// and gfx1100 read bits 1184 to 1200 as zero; their table ends after these
/// 37 words.
inline constexpr uint32_t kTwoOverPi[] = {
    0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041, 0xfe5163ab, 0xdebbc561,
    0xb7246e3a, 0x424dd2e0, 0x06492eea, 0x09d1921c, 0xfe1deb1c, 0xb129a73e, 0xe88235f5, 0x2ebb4484,
    0xe99c7026, 0xb45f7e41, 0x3991d639, 0x835339f4, 0x9c845f8b, 0xbdf9283b, 0x1ff897ff, 0xde05980f,
    0xef2f118b, 0x5a0a6d1f, 0x6d367ecf, 0x27cb09b7, 0x4f463f66, 0x9e5fea2d, 0x7527bac7, 0xebe5f17b,
    0x3d0739f7, 0x8a5292ea, 0x6bfb5fb1, 0x1f8d5d08, 0x56033046};

namespace detail {

constexpr uint64_t table_word(uint32_t index) {
  return index < std::size(kTwoOverPi) ? kTwoOverPi[index] : 0;
}

/// @brief The 53 table bits starting at fraction bit `first`.
constexpr uint64_t segment(uint32_t first) {
  const uint32_t word = first / 32;
  const uint32_t offset = first % 32;
  uint64_t window = table_word(word) << 32 | table_word(word + 1);
  if (offset != 0)
    window = window << offset | table_word(word + 2) >> (32 - offset);
  return window >> (64 - 53);
}

/// @brief Encode segment * 2^scale as F64, truncating toward zero.
/// @details The value is always below 1.0, so it never overflows.
constexpr uint64_t encode_rtz(uint64_t segment, int scale) {
  using F64 = fp_format::F64;
  if (segment == 0)
    return 0;
  const int width = std::bit_width(segment);
  const int exponent = width - 1 + scale;
  if (exponent >= -1022) {
    const uint64_t mantissa = segment << (53 - width);
    return uint64_t(exponent + 1023) << F64::kMantissaBits | (mantissa & (F64::kMinNormal - 1));
  }
  // Subnormal: count units of 2^-1074 and drop the bits below them.
  const int drop = -1074 - scale;
  if (drop >= 64)
    return 0;
  return drop >= 0 ? segment >> drop : segment << -drop;
}

} // namespace detail

/// @brief Look up the 2/pi segment selected by src1[4:0], scaled for src0's exponent.
/// @details ABS and NEG change only src0's sign, which the lookup ignores, and
/// a subnormal src0 has exponent 0, so input flushing cannot change the result.
constexpr uint64_t lookup(uint64_t src0, uint32_t src1) {
  using F64 = fp_format::F64;
  const uint32_t exponent = uint32_t(src0 >> F64::kMantissaBits) & F64::kExponentMax;
  uint32_t shift = (src1 & 31u) * 53;
  if (exponent > 1077)
    shift += exponent - 1077;
  int scale = -53 - int(shift);
  // Large sources are scaled up by 2^128 to keep the segment's precision.
  if (exponent >= 1968)
    scale += 128;
  return detail::encode_rtz(detail::segment(shift), scale);
}

} // namespace rocjitsu::amdgpu::trig_preop
