// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 packed 16-bit operand selection with halfword selectors.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Byte indices select either half independently within every original VGPR word.
inline __m512i packed_integer_selection(bool low_high, bool high_low) {
  uint32_t low = low_high ? 0x0302 : 0x0100;
  uint32_t high = high_low ? 0x0100 : 0x0302;
  return _mm512_add_epi8(_mm512_set1_epi32(int(low | (high << 16))),
                         _mm512_setr_epi32(0, 0x04040404, 0x08080808, 0x0c0c0c0c, 0, 0x04040404,
                                           0x08080808, 0x0c0c0c0c, 0, 0x04040404, 0x08080808,
                                           0x0c0c0c0c, 0, 0x04040404, 0x08080808, 0x0c0c0c0c));
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
