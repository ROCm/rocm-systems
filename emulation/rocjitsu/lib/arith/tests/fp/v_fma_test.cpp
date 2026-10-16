// SPDX-License-Identifier: MIT

// gfx12 FP16 FMA: deterministic fractional inputs, exceptional values,
// EXEC, whole-register aliases, preserved halfwords, and errors.

#include "common/contracts.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using RegBank = rocjitsu::arith::test::Registers<5, 32>;
} // namespace

// Check low-half arithmetic and that all storage outside active results is preserved.
TEST(FpFmaF16W32Gfx12, Arithmetic) {
  using namespace rocjitsu::arith::test;
  RegBank input;
  std::minstd_rand random;
  for (int reg = 0; reg < 5; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = (random_f16_pair(random) & 0xffff0000U) | random_f16_bits(random);
    }
  }
  for (int reg = 0; reg < 3; ++reg) {
    input.v[reg][reg * 2] = kF16QuietNaNBits;
    input.v[reg][reg * 2 + 1] = kF16PositiveInfinityBits;
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        return rj_arith_v_fma_f16_w32_gfx12(flags, 0xffffffffU, mode, nullptr, nullptr, nullptr,
                                            nullptr, nullptr);
      },
      0);
  for (uint64_t semantics : semantics_modes()) {
    for (uint32_t exec_mask : {0U, 0xffffffffU, 0xaaaaaaaaU, 0x80000000U}) {
      for (int aliasing_offset : {0, 1, 2, 3}) {
        RegBank r = input;
        uint32_t status = 0x80000000U;
        ASSERT_EQ(rj_arith_v_fma_f16_w32_gfx12(semantics, exec_mask, 0, r.v + aliasing_offset, r.v,
                                               r.v + 1, r.v + 2, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(status, 0x80000000U);
        r.expect_preserved(input, aliasing_offset, 1);
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t actual = r.v[aliasing_offset][lane];
          if (!((exec_mask >> lane) & 1)) {
            EXPECT_EQ(actual, input.v[aliasing_offset][lane]);
            continue;
          }
          EXPECT_EQ(actual >> 16, input.v[aliasing_offset][lane] >> 16);
          // A computed reference is sufficient for loose arithmetic.
          double value = std::fma(half_value(uint16_t(input.v[0][lane])),
                                  half_value(uint16_t(input.v[1][lane])),
                                  half_value(uint16_t(input.v[2][lane])));
          float expected = float(half_value(half_bits(value)));
          expect_f32(bit_cast<uint32_t>(float(half_value(uint16_t(actual)))), expected, semantics,
                     /*atol=*/1e-5, /*rtol=*/5e-4);
        }
      }
    }
  }
  EXPECT_EQ(rj_arith_v_fma_f16_w32_gfx12(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, 0,
                                         nullptr, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
}
