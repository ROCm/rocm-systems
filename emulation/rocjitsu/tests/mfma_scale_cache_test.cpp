// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simd_correctness/mma_exact_test_support.h"

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t SRC_A = 0, SRC_B = 16, ACC = 32, DST = 64;
constexpr uint32_t SCALE_A = 96, SCALE_B = 97;

struct ScaleFixture : ExactFixture {
  ScaleFixture() : ExactFixture(ROCJITSU_CODE_ARCH_CDNA4, 64) {}
};

void execute(ScaleFixture &fx, uint32_t dim, uint32_t k, uint32_t scale_a, uint32_t scale_b,
             uint32_t a_byte, uint32_t b_byte, uint32_t const_acc = amdgpu::ACC_FROM_VGPR) {
  amdgpu::exec_f32_scaled_mixed(*fx.cu, dim, dim, k, 1, 4, 4, fx.vbase + DST, fx.vbase + SRC_A,
                                fx.vbase + SRC_B, fx.vbase + ACC, amdgpu::extract_fp4,
                                amdgpu::extract_fp4, const_acc, fx.vbase, scale_a, scale_b, a_byte,
                                b_byte);
}

void seed_inputs(ScaleFixture &fx) {
  fx.seed(SRC_A, 4, Fmt::RAW4, Mode::RandomInt, 51);
  fx.seed(SRC_B, 4, Fmt::RAW4, Mode::RandomInt, 83);
  fx.seed(ACC, 16, Fmt::F32, Mode::RandomInt, 97);
}

void expect_cached_matches_scalar(uint32_t dim, uint32_t k,
                                  const std::function<void(ScaleFixture &)> &seed,
                                  const std::function<void(ScaleFixture &)> &run) {
  ForceScalarGuard guard;
  ScaleFixture scalar;
  ScaleFixture cached;
  ASSERT_NE(scalar.wf, nullptr);
  ASSERT_NE(cached.wf, nullptr);
  seed(scalar);
  seed(cached);
  util::set_force_scalar_for_testing(true);
  run(scalar);
  util::set_force_scalar_for_testing(false);
  run(cached);
  EXPECT_EQ(scalar.snapshot(DST, dim * dim / 64), cached.snapshot(DST, dim * dim / 64))
      << "dim=" << dim << " K=" << k;
}

TEST(MfmaScaleCache, SelectedBytesAndLaneScalesHaveExpectedValues) {
  ForceScalarGuard guard;
  util::set_force_scalar_for_testing(false);
  for (uint32_t dim : {16u, 32u}) {
    const uint32_t k = 2048 / dim;
    for (uint32_t byte = 0; byte < 4; ++byte) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " byte=" << byte);
      ScaleFixture fx;
      ASSERT_NE(fx.wf, nullptr);
      for (uint32_t lane = 0; lane < 64; ++lane) {
        for (uint32_t reg = 0; reg < 4; ++reg) {
          fx.cu->write_vgpr(fx.vbase + SRC_A + reg, lane, 0x22222222u);
          fx.cu->write_vgpr(fx.vbase + SRC_B + reg, lane, 0x22222222u);
        }
        const uint32_t a = 125 + (lane % 5);
        const uint32_t b = 126 + ((lane / 3) % 3);
        const uint32_t a_word = (0xffffffffu & ~(0xffu << (byte * 8))) | (a << (byte * 8));
        const uint32_t b_byte = 3 - byte;
        const uint32_t b_word = (0xffffffffu & ~(0xffu << (b_byte * 8))) | (b << (b_byte * 8));
        fx.cu->write_vgpr(fx.vbase + SCALE_A, lane, a_word);
        fx.cu->write_vgpr(fx.vbase + SCALE_B, lane, b_word);
      }
      execute(fx, dim, k, 256 + SCALE_A, 256 + SCALE_B, byte, 3 - byte,
              std::bit_cast<uint32_t>(1.0f));
      for (uint32_t row = 0; row < dim; ++row)
        for (uint32_t col = 0; col < dim; ++col) {
          float expected = 1.0f;
          for (uint32_t block = 0; block < k / 32; ++block) {
            const int a = 125 + ((dim * block + row) % 5);
            const int b = 126 + (((dim * block + col) / 3) % 3);
            expected += std::ldexp(32.0f, a + b - 254);
          }
          const auto out = amdgpu::output_loc_32(dim, dim, row, col, 0);
          EXPECT_EQ(fx.cu->read_vgpr(fx.vbase + DST + out.reg, out.lane),
                    std::bit_cast<uint32_t>(expected))
              << "row=" << row << " col=" << col;
        }
    }
  }
}

TEST(MfmaScaleCache, ExponentBoundariesAndNaNMatchScalar) {
  constexpr std::array<uint8_t, 12> exponents = {0,   1,   62,  100, 125, 126,
                                                 127, 128, 130, 180, 253, 254};
  for (uint32_t dim : {16u, 32u}) {
    const uint32_t k = 2048 / dim;
    for (bool use_nan : {false, true})
      for (uint32_t byte = 0; byte < 4; ++byte) {
        SCOPED_TRACE(testing::Message() << "dim=" << dim << " byte=" << byte << " NaN=" << use_nan);
        auto seed = [=](ScaleFixture &fx) {
          seed_inputs(fx);
          for (uint32_t lane = 0; lane < 64; ++lane) {
            uint32_t a = 0, b = 0;
            for (uint32_t part = 0; part < 4; ++part) {
              a |= uint32_t(exponents[(lane + part) % exponents.size()]) << (part * 8);
              b |= uint32_t(exponents[(lane * 5 + part) % exponents.size()]) << (part * 8);
            }
            if (use_nan && lane == 7)
              a |= 0xffu << (byte * 8);
            if (use_nan && lane == 45)
              b |= 0xffu << ((3 - byte) * 8);
            fx.cu->write_vgpr(fx.vbase + SCALE_A, lane, a);
            fx.cu->write_vgpr(fx.vbase + SCALE_B, lane, b);
          }
        };
        expect_cached_matches_scalar(dim, k, seed, [=](ScaleFixture &fx) {
          execute(fx, dim, k, 256 + SCALE_A, 256 + SCALE_B, byte, 3 - byte);
        });
      }
  }
}

TEST(MfmaScaleCache, InlineSelectorsMatchScalar) {
  for (uint32_t selector = 240; selector <= 248; ++selector)
    expect_cached_matches_scalar(16, 128, seed_inputs, [=](ScaleFixture &fx) {
      execute(fx, 16, 128, selector, 488 - selector, 0, 3);
    });
}

TEST(MfmaScaleCache, DestinationAndInputOverlapMatchScalar) {
  for (uint32_t scale_a : {SRC_A, ACC, DST})
    for (uint32_t scale_b : {SRC_B, DST + 1}) {
      auto seed = [=](ScaleFixture &fx) {
        seed_inputs(fx);
        for (uint32_t lane = 0; lane < 64; ++lane) {
          fx.cu->write_vgpr(fx.vbase + scale_a, lane, 0x7f7e807du);
          fx.cu->write_vgpr(fx.vbase + scale_b, lane, 0x7d807e7fu);
        }
      };
      expect_cached_matches_scalar(16, 128, seed, [=](ScaleFixture &fx) {
        execute(fx, 16, 128, 256 + scale_a, 256 + scale_b, 1, 2);
      });
    }
}

} // namespace
