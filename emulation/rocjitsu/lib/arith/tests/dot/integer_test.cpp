// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed integer dot products with signedness and saturation.

#include "common/integer.h"
#include "common/word_test.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Extract signed/unsigned factors and accumulate in a wide integer before clamp or wrap.
template <int Bits, bool Unsigned>
uint32_t integer_dot(uint32_t a, uint32_t b, uint32_t c, uint32_t instruction_flags) {
  // Select the source halves before interpreting their packed integer elements.
  a = ((a >> (instruction_flags & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0)) & 0xffffU) |
      ((a >> (instruction_flags & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16)) << 16);
  b = ((b >> (instruction_flags & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0)) & 0xffffU) |
      ((b >> (instruction_flags & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16)) << 16);
  int64_t sum = Unsigned ? int64_t(c) : extend_bits(c, 32, true);
  for (int shift = 0; shift < 32; shift += Bits) {
    sum += extend_bits((a >> shift) & ((1U << Bits) - 1), Bits,
                       instruction_flags & RJ_ARITH_DOT_SIGNED_A) *
           extend_bits((b >> shift) & ((1U << Bits) - 1), Bits,
                       instruction_flags & RJ_ARITH_DOT_SIGNED_B);
  }
  if (instruction_flags & RJ_ARITH_DOT_CLAMP) {
    sum = std::clamp(sum, Unsigned ? int64_t(0) : int64_t(INT32_MIN),
                     Unsigned ? int64_t(UINT32_MAX) : int64_t(INT32_MAX));
  }
  return uint32_t(sum);
}

// Check packed integer dot arithmetic and signedness/saturation controls for the supplied entry
// point.
template <int Lanes, int Bits, bool Unsigned, bool FixedSigned = false, bool Accumulate = false,
          bool IgnoreClamp = false, typename Call>
void check_integer_dot(Call call) {
  auto input = make_integer_inputs<Lanes>();
  // Carry and signed saturation, including negative products with signed A.
  for (int lane = 0; lane < 16; ++lane) {
    input.v[0][lane] = Bits == 16 ? 0x80008000U : Bits == 8 ? 0x80808080U : 0x88888888U;
    input.v[2][lane] = Bits == 16 ? 0x7fff7fffU : Bits == 8 ? 0x7f7f7f7fU : 0x77777777U;
    input.v[4][lane] = lane % 3 == 0 ? 0xfffffff0U : lane % 3 == 1 ? 0x7ffffff0U : 0x80000010U;
  }
  if constexpr (Bits == 16) {
    // Two signed minimum products make +2^31, but adding INT32_MIN cancels to zero.
    // Unsigned maximum products need 34 bits before final saturation.
    input.v[0][0] = input.v[2][0] = 0x80008000U;
    input.v[4][0] = 0x80000000U;
    input.v[0][1] = input.v[2][1] = input.v[4][1] = 0xffffffffU;
  }
  check_integer_word<Lanes, 3, false>(
      call,
      [](uint32_t a, uint32_t b, uint32_t c, uint32_t old_d, uint32_t instruction_flags, int) {
        if constexpr (FixedSigned) {
          instruction_flags |= RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B;
        }
        if constexpr (IgnoreClamp) {
          instruction_flags &= ~RJ_ARITH_DOT_CLAMP;
        }
        return integer_dot<Bits, Unsigned>(a, b, Accumulate ? old_d : c, instruction_flags);
      },
      Accumulate
          ? 0
          : RJ_ARITH_DOT_LO_A_HIGH | RJ_ARITH_DOT_LO_B_HIGH | RJ_ARITH_DOT_HI_A_LOW |
                RJ_ARITH_DOT_HI_B_LOW | RJ_ARITH_DOT_CLAMP |
                (Unsigned || FixedSigned ? 0 : RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B),
      false, input);
  if constexpr (!Unsigned && !FixedSigned && !IgnoreClamp && Bits < 16) {
    // gfx1100/gfx1201 review probes: cancellation occurs before final saturation.
    // Both product orders retain C, even when an intermediate sum exceeds its range.
    Registers<4, Lanes> r;
    r.v[0][0] = Bits == 8 ? 0x00007f7fU : 0x00000077U;
    r.v[1][0] = Bits == 8 ? 0x0000817fU : 0x00000097U;
    r.v[2][0] = 0x7ffffff0U;
    r.v[0][1] = 0x00007f7fU;
    r.v[1][1] = 0x00007f81U;
    r.v[2][1] = 0x80000010U;
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : semantics_modes()) {
        ASSERT_EQ(call(cpu | semantics, Bits == 8 ? 3 : 1,
                       RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B | RJ_ARITH_DOT_CLAMP, r.v + 3,
                       r.v, r.v + 1, r.v + 2),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[3][0], 0x7ffffff0U);
        if constexpr (Bits == 8) {
          EXPECT_EQ(r.v[3][1], 0x80000010U);
        }
      }
    }
  }
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

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot2I32I16W32, Arithmetic) {
  check_integer_dot<32, 16, false, true, false, false>(rj_arith_v_dot2_i32_i16_w32);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot2U32U16W32, Arithmetic) {
  check_integer_dot<32, 16, true, false, false, false>(rj_arith_v_dot2_u32_u16_w32);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4I32I8W32, Arithmetic) {
  check_integer_dot<32, 8, false, true, false, false>(rj_arith_v_dot4_i32_i8_w32);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot8I32I4W32, Arithmetic) {
  check_integer_dot<32, 4, false, true, false, false>(rj_arith_v_dot8_i32_i4_w32);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot2CI32I16W32, Arithmetic) {
  check_integer_dot<32, 16, false, true, true, false>(
      [](uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return rj_arith_v_dot2c_i32_i16_w32(flags, exec_mask, mode, d, a, b);
      });
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4CI32I8W32, Arithmetic) {
  check_integer_dot<32, 8, false, true, true, false>(
      [](uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return rj_arith_v_dot4c_i32_i8_w32(flags, exec_mask, mode, d, a, b);
      });
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot8CI32I4W32, Arithmetic) {
  check_integer_dot<32, 4, false, true, true, false>(
      [](uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return rj_arith_v_dot8c_i32_i4_w32(flags, exec_mask, mode, d, a, b);
      });
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4I32Iu8W32Gfx1250, Arithmetic) {
  check_integer_dot<32, 8, false, false, false, true>(rj_arith_v_dot4_i32_iu8_w32_gfx1250);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4U32U8W32Gfx1250, Arithmetic) {
  check_integer_dot<32, 8, true, false, false, true>(rj_arith_v_dot4_u32_u8_w32_gfx1250);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot2I32I16W64, Arithmetic) {
  check_integer_dot<64, 16, false, true, false, false>(rj_arith_v_dot2_i32_i16_w64);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot2U32U16W64, Arithmetic) {
  check_integer_dot<64, 16, true, false, false, false>(rj_arith_v_dot2_u32_u16_w64);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4I32I8W64, Arithmetic) {
  check_integer_dot<64, 8, false, true, false, false>(rj_arith_v_dot4_i32_i8_w64);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot8I32I4W64, Arithmetic) {
  check_integer_dot<64, 4, false, true, false, false>(rj_arith_v_dot8_i32_i4_w64);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot2CI32I16W64, Arithmetic) {
  check_integer_dot<64, 16, false, true, true, false>(
      [](uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return rj_arith_v_dot2c_i32_i16_w64(flags, exec_mask, mode, d, a, b);
      });
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4CI32I8W64, Arithmetic) {
  check_integer_dot<64, 8, false, true, true, false>(
      [](uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return rj_arith_v_dot4c_i32_i8_w64(flags, exec_mask, mode, d, a, b);
      });
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot8CI32I4W64, Arithmetic) {
  check_integer_dot<64, 4, false, true, true, false>(
      [](uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return rj_arith_v_dot8c_i32_i4_w64(flags, exec_mask, mode, d, a, b);
      });
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4I32Iu8W64Gfx1250, Arithmetic) {
  check_integer_dot<64, 8, false, false, false, true>(rj_arith_v_dot4_i32_iu8_w64_gfx1250);
}

// Check arithmetic, controls, aliases and inactive lanes for this instruction form.
TEST(Dot4U32U8W64Gfx1250, Arithmetic) {
  check_integer_dot<64, 8, true, false, false, true>(rj_arith_v_dot4_u32_u8_w64_gfx1250);
}
