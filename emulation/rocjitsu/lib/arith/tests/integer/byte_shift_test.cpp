// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Saturating byte packing after signed shifts: masks, destination halves and aliases.
#include "common/word_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <algorithm>
#include <gtest/gtest.h>

namespace {
// Compare the ISA's signed shift and byte saturation with every available backend.
template <bool Signed, int Lanes, typename Call> void check(Call call) {
  rocjitsu::arith::test::check_integer_word<Lanes, 3>(
      call,
      [](uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t mode, int) {
        auto byte = [c](uint32_t input) {
          int64_t value = input <= INT32_MAX ? int64_t(input) : int64_t(input) - (1LL << 32);
          // Division rounded down independently models the signed arithmetic shift.
          int64_t divisor = 1LL << (c & 31);
          value = value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
          return uint32_t(std::clamp(value, Signed ? int64_t(-128) : int64_t(0),
                                     Signed ? int64_t(127) : int64_t(255))) &
                 255;
        };
        int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        return (d & ~(0xffffU << shift)) | ((byte(a) | (byte(b) << 8)) << shift);
      },
      RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP);
}
} // namespace

TEST(IntegerAshrPkU8I32W32, Arithmetic) { check<false, 32>(rj_arith_v_ashr_pk_u8_i32_w32); }

TEST(IntegerAshrPkI8I32W32, Arithmetic) { check<true, 32>(rj_arith_v_ashr_pk_i8_i32_w32); }

TEST(IntegerAshrPkU8I32W64, Arithmetic) { check<false, 64>(rj_arith_v_ashr_pk_u8_i32_w64); }

TEST(IntegerAshrPkI8I32W64, Arithmetic) { check<true, 64>(rj_arith_v_ashr_pk_i8_i32_w64); }
