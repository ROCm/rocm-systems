// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed integer dot products with signedness and saturation.

#include "common/integer.h"
#include "common/word_test.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Extract signed/unsigned factors and accumulate in a wide integer before clamp or wrap.
template <int Bits, bool Unsigned>
uint32_t integer_dot(uint32_t a, uint32_t b, uint32_t c, uint32_t mode) {
  // Select the source halves before interpreting their packed integer elements.
  a = ((a >> (mode & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0)) & 0xffffU) |
      ((a >> (mode & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16)) << 16);
  b = ((b >> (mode & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0)) & 0xffffU) |
      ((b >> (mode & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16)) << 16);
  int64_t sum = Unsigned ? int64_t(c) : extend_bits(c, 32, true);
  for (int shift = 0; shift < 32; shift += Bits) {
    sum += extend_bits((a >> shift) & ((1U << Bits) - 1), Bits, mode & RJ_ARITH_DOT_SIGNED_A) *
           extend_bits((b >> shift) & ((1U << Bits) - 1), Bits, mode & RJ_ARITH_DOT_SIGNED_B);
  }
  if (mode & RJ_ARITH_DOT_CLAMP) {
    sum = std::clamp(sum, Unsigned ? int64_t(0) : int64_t(INT32_MIN),
                     Unsigned ? int64_t(UINT32_MAX) : int64_t(INT32_MAX));
  }
  return uint32_t(sum);
}

// Check packed integer dot arithmetic and signedness/saturation controls for the supplied entry
// point.
template <int Lanes, int Bits, bool Unsigned, typename Call> void check_integer_dot(Call call) {
  auto input = make_integer_inputs<Lanes>();
  // Carry and signed saturation, including negative products with signed A.
  for (int lane = 0; lane < 16; ++lane) {
    input.v[0][lane] = Bits == 8 ? 0x80808080U : 0x88888888U;
    input.v[2][lane] = Bits == 8 ? 0x7f7f7f7fU : 0x77777777U;
    input.v[4][lane] = lane % 3 == 0 ? 0xfffffff0U : lane % 3 == 1 ? 0x7ffffff0U : 0x80000010U;
  }
  check_integer_word<Lanes>(
      call,
      [](uint32_t a, uint32_t b, uint32_t c, uint32_t, uint32_t mode, int) {
        return integer_dot<Bits, Unsigned>(a, b, c, mode);
      },
      RJ_ARITH_DOT_LO_A_HIGH | RJ_ARITH_DOT_LO_B_HIGH | RJ_ARITH_DOT_HI_A_LOW |
          RJ_ARITH_DOT_HI_B_LOW | RJ_ARITH_DOT_CLAMP |
          (Unsigned ? 0 : RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B),
      input);
}

} // namespace

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot4I32Iu8W32, Arithmetic) { check_integer_dot<32, 8, false>(rj_arith_v_dot4_i32_iu8_w32); }

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot4U32U8W32, Arithmetic) { check_integer_dot<32, 8, true>(rj_arith_v_dot4_u32_u8_w32); }

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot8I32Iu4W32, Arithmetic) { check_integer_dot<32, 4, false>(rj_arith_v_dot8_i32_iu4_w32); }

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot8U32U4W32, Arithmetic) { check_integer_dot<32, 4, true>(rj_arith_v_dot8_u32_u4_w32); }

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot4I32Iu8W64, Arithmetic) { check_integer_dot<64, 8, false>(rj_arith_v_dot4_i32_iu8_w64); }

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot4U32U8W64, Arithmetic) { check_integer_dot<64, 8, true>(rj_arith_v_dot4_u32_u8_w64); }

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot8I32Iu4W64, Arithmetic) { check_integer_dot<64, 4, false>(rj_arith_v_dot8_i32_iu4_w64); }

// Check packed integer dot arithmetic and signedness/saturation controls.
TEST(Dot8U32U4W64, Arithmetic) { check_integer_dot<64, 4, true>(rj_arith_v_dot8_u32_u4_w64); }
