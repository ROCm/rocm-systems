// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for integer narrowing/extension and byte/halfword packing.

#include "conversion/helpers.h"
#include "conversion/integer_conversion_reference.h"
#include "conversion/pack_reference.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class IntegerConversion {
  SignedWiden,
  UnsignedWiden,
  SignedPack,
  UnsignedPack,
  SaturateBytes
};

// Check integer narrowing/extension with modifiers and aliases for the supplied entry point.
template <int Lanes, int Operands, typename Call>
void check_integer(Call call, IntegerConversion operation) {
  int op = int(operation);
  auto input = conversion_inputs<Lanes>(16);
  // Signed limits and saturation boundaries for both packed sources.
  const uint32_t edges[] = {0xffffffffU, 0x80000000U, 0x7fffffffU, 32767, 32768,
                            65535,       65536,       255,         256};
  for (int reg : {0, 2}) {
    for (int lane = 0; lane < 9; ++lane) {
      input.v[reg][lane] = edges[(lane + reg) % 9];
    }
  }
  uint64_t modifiers = op < 2    ? RJ_ARITH_ALU_HIGH_A
                       : op == 4 ? RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP
                                 : 0;
  check_conversion<Lanes, Operands, 1, 1, true>(
      call, input,
      [=](uint64_t a, uint32_t b, uint32_t, uint32_t d, uint64_t, uint32_t mode) {
        return op == 4 ? pack_reference(bool(mode & RJ_ARITH_ALU_HIGH_D), uint32_t(a), b, d)
                       : integer_conversion_reference(op, uint32_t(a), b, mode);
      },
      conversion_bits, modifiers, true);
}

// Check byte/halfword packing with modifiers and aliases for the supplied entry point.
template <int Lanes, bool Gfx12, typename Call> void check_pack(Call call) {
  check_conversion<Lanes, 2, 1, 1, true>(
      call, conversion_inputs<Lanes>(16),
      [](uint64_t a, uint32_t b, uint32_t, uint32_t d, uint64_t flags, uint32_t mode) {
        if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
          for (int shift : {0, 16}) {
            if (((a >> shift) & 0x7fffU) < 0x400U) {
              a &= ~(0x7fffULL << shift);
            }
            if (((b >> shift) & 0x7fffU) < 0x400U) {
              b &= ~(0x7fffU << shift);
            }
          }
        }
        int variant = 2 + bool(mode & RJ_ARITH_ALU_ABS_A) + 2 * bool(mode & RJ_ARITH_ALU_ABS_B) +
                      4 * bool(mode & RJ_ARITH_ALU_NEG_A) + 8 * bool(mode & RJ_ARITH_ALU_NEG_B) +
                      16 * bool(mode & RJ_ARITH_ALU_HIGH_A) + 32 * bool(mode & RJ_ARITH_ALU_HIGH_B);
        return pack_reference(variant, uint32_t(a), b, d, Gfx12 || (flags & RJ_ARITH_FP_IEEE_MODE));
      },
      conversion_bits,
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B |
          RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B,
      Gfx12,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionSatPkU8I16W32, Arithmetic) {
  check_integer<32, 1>(rj_arith_v_sat_pk_u8_i16_w32, IntegerConversion::SaturateBytes);
}

// Check byte/halfword packing with modifiers and aliases.
TEST(ConversionPackB32F16W32Gfx11, Arithmetic) {
  check_pack<32, false>(rj_arith_v_pack_b32_f16_w32);
}

// Check byte/halfword packing with modifiers and aliases.
TEST(ConversionPackB32F16W32Gfx12, Arithmetic) {
  check_pack<32, true>(rj_arith_v_pack_b32_f16_w32_gfx12);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionSatPkU8I16W64, Arithmetic) {
  check_integer<64, 1>(rj_arith_v_sat_pk_u8_i16_w64, IntegerConversion::SaturateBytes);
}

// Check byte/halfword packing with modifiers and aliases.
TEST(ConversionPackB32F16W64Gfx11, Arithmetic) {
  check_pack<64, false>(rj_arith_v_pack_b32_f16_w64);
}

// Check byte/halfword packing with modifiers and aliases.
TEST(ConversionPackB32F16W64Gfx12, Arithmetic) {
  check_pack<64, true>(rj_arith_v_pack_b32_f16_w64_gfx12);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtI32I16W32, Arithmetic) {
  check_integer<32, 1>(rj_arith_v_cvt_i32_i16_w32, IntegerConversion::SignedWiden);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtU32U16W32, Arithmetic) {
  check_integer<32, 1>(rj_arith_v_cvt_u32_u16_w32, IntegerConversion::UnsignedWiden);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtPkI16I32W32, Arithmetic) {
  check_integer<32, 2>(rj_arith_v_cvt_pk_i16_i32_w32, IntegerConversion::SignedPack);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtPkU16U32W32, Arithmetic) {
  check_integer<32, 2>(rj_arith_v_cvt_pk_u16_u32_w32, IntegerConversion::UnsignedPack);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtI32I16W64, Arithmetic) {
  check_integer<64, 1>(rj_arith_v_cvt_i32_i16_w64, IntegerConversion::SignedWiden);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtU32U16W64, Arithmetic) {
  check_integer<64, 1>(rj_arith_v_cvt_u32_u16_w64, IntegerConversion::UnsignedWiden);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtPkI16I32W64, Arithmetic) {
  check_integer<64, 2>(rj_arith_v_cvt_pk_i16_i32_w64, IntegerConversion::SignedPack);
}

// Check integer narrowing/extension with modifiers and aliases.
TEST(ConversionCvtPkU16U32W64, Arithmetic) {
  check_integer<64, 2>(rj_arith_v_cvt_pk_u16_u32_w64, IntegerConversion::UnsignedPack);
}
