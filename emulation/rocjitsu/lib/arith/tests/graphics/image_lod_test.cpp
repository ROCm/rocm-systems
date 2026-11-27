// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected inherited footprint goldens, quad writes and CPU/semantics dispatch.
#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include <algorithm>
#include <bit>
#include <gtest/gtest.h>

namespace {
TEST(ImageGetLod, Arithmetic) {
  struct Case {
    rj_arith_image_gradients gradients;
    uint32_t extents[3], dimensions;
    float raw;
  };
  // Existing gfx1100/gfx1201 footprint captures distinguish log-segment rounding
  // and the volume's nested norms. Zero gradients distinguish gfx10 raw-LOD policy.
  const Case cases[] = {
      {{{0x1.955d52p-5, 0x1.f7ce2ap-5, 0}, {0x1.f82884p-5, -0x1.312724p-4, 0}, {}, 0},
       {37, 19, 1},
       2,
       0x1.91p+0f},
      {{{0x1.019dp-8, 0, 0}, {0, 0, 0}, {}, 0}, {256, 64, 1}, 2, 0x1p-7f},
      {{{-0x1.5bf1f8p-6, 0x1.208a3ep-6, -0x1.e9ea3ep-5}, {0, 0, 0}, {}, 0},
       {37, 19, 11},
       3,
       0x1.cp-4f},
      {{{}, {}, {}, 0}, {1, 1, 1}, 2, 0}};
  for (const auto &test : cases) {
    for (bool gfx10 : {false, true}) {
      for (int lanes : {32, 64}) {
        for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
          for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
            SCOPED_TRACE(testing::Message() << test.dimensions << ":" << gfx10 << ":" << lanes
                                            << ":" << semantics << ":" << cpu);
            rj_arith_image_gradients inputs[16]{};
            for (int i = 0; i < 16; ++i) {
              inputs[i] = test.gradients;
            }
            const rj_arith_image_lod_controls control{
                {test.extents[0], test.extents[1], test.extents[2]},
                test.dimensions,
                0,
                0,
                0,
                12,
                0,
                0,
                12,
                2,
                0,
                0};
            uint32_t output[2][64];
            for (auto &row : output) {
              std::fill_n(row, 64, 0xdeadbeef);
            }
            uint32_t *d[] = {output[0], output[1]};
            const uint64_t exec_mask = (1ULL << (lanes - 1)) | 2;
            const auto w32 = gfx10 ? rj_arith_image_get_lod_w32_gfx10 : rj_arith_image_get_lod_w32;
            const auto w64 = gfx10 ? rj_arith_image_get_lod_w64_gfx10 : rj_arith_image_get_lod_w64;
            const int status =
                lanes == 32 ? w32(cpu | semantics, uint32_t(exec_mask), 0, d, inputs, &control)
                            : w64(cpu | semantics, exec_mask, 0, d, inputs, &control);
            const bool supported = semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
                                   semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
            ASSERT_EQ(status, supported ? RJ_ARITH_SUCCESS : RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            for (int i = 0; i < 2; ++i) {
              for (int lane = 0; lane < lanes; ++lane) {
                float expected = test.raw;
                if (i == 1 && gfx10 && test.raw == 0) {
                  expected = -32;
                }
                EXPECT_EQ(output[i][lane], supported && ((exec_mask >> lane) & 1)
                                               ? std::bit_cast<uint32_t>(expected)
                                               : 0xdeadbeef);
              }
            }
          }
        }
      }
    }
  }
}

TEST(ImageGetLod, DistinctQuads) {
  rj_arith_image_gradients inputs[8]{};
  // Four successive powers of two give exact LODs 0, 1, 2 and 3.
  for (int quad = 0; quad < 4; ++quad) {
    inputs[quad].dx[0] = float(1 << quad);
  }
  const rj_arith_image_lod_controls control{{1, 1, 1}, 2, 0, 0, 0, 12, 0, 0, 12, 2, 0, 0};
  constexpr uint32_t exec_mask = 0x8421; // One active lane at a different position in each quad.
  for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
    for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
      SCOPED_TRACE(testing::Message() << semantics << ":" << cpu);
      uint32_t output[2][32];
      for (auto &row : output) {
        std::fill_n(row, 32, 0xdeadbeef);
      }
      uint32_t *d[] = {output[0], output[1]};
      const bool supported = semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
                             semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
      ASSERT_EQ(rj_arith_image_get_lod_w32(cpu | semantics, exec_mask, 0, d, inputs, &control),
                supported ? RJ_ARITH_SUCCESS : RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
      for (const auto &row : output) {
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(row[lane], supported && ((exec_mask >> lane) & 1)
                                   ? std::bit_cast<uint32_t>(float(lane / 4))
                                   : 0xdeadbeef);
        }
      }
    }
  }
}

} // namespace
