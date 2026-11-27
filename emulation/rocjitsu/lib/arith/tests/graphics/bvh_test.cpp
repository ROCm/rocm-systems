// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited triangle capture plus box ordering, fetch masking and overlapping outputs.
#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include <algorithm>
#include <bit>
#include <gtest/gtest.h>
#include <iterator>

namespace {
TEST(BvhIntersection, Arithmetic) {
  // Existing W7900 projected-barycentric case, including nontrivial RTZ steps.
  constexpr uint32_t vertices[] = {0x40c88e04, 0xc0b760b3, 0xbfc5c580, 0xc0a232a0, 0x3f285210,
                                   0x40a6ef68, 0xc0fd5513, 0x3ff1a668, 0x402993f4};
  constexpr uint32_t origin[] = {0xc0c9b3cb, 0x40651057, 0x3d2c07a0};
  constexpr uint32_t direction[] = {0x40b0161c, 0xc0a534f0, 0x3f8313e8};
  constexpr uint32_t golden[] = {0xc57b32f7, 0xc55f49c1, 0xc45f49c0, 0xc45f49c7};
  for (bool boxes : {false, true}) {
    for (int lanes : {32, 64}) {
      for (bool alias : {false, true}) {
        for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
          for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
            SCOPED_TRACE(testing::Message() << boxes << ":" << lanes << ":" << alias << ":"
                                            << semantics << ":" << cpu);
            uint32_t nodes[64 * 32]{}, output[4][64], before[4][64];
            rj_arith_bvh_ray rays[64]{};
            rj_arith_bvh_controls controls{0, 0, 1, uint32_t(!boxes), 0};
            uint32_t *d[4];
            for (int i = 0; i < 4; ++i) {
              d[i] = alias ? nodes + i * 64 : output[i];
            }
            for (int lane : {1, lanes - 1}) {
              if (boxes) {
                rays[lane].node_type = 5;
                rays[lane].extent = 10;
                for (int axis = 0; axis < 3; ++axis) {
                  rays[lane].origin[axis] = -1;
                  rays[lane].inverse[axis] = 1;
                }
                for (int child = 0; child < 4; ++child) {
                  nodes[lane * 32 + child] = uint32_t(8 * (child + 1) + 5);
                  for (int axis = 0; axis < 3; ++axis) {
                    nodes[lane * 32 + 4 + 6 * child + axis] = std::bit_cast<uint32_t>(float(child));
                    nodes[lane * 32 + 7 + 6 * child + axis] =
                        std::bit_cast<uint32_t>(float(child + 1));
                  }
                }
              } else {
                std::copy(std::begin(vertices), std::end(vertices), nodes + lane * 32);
                nodes[lane * 32 + 15] = 9;
                for (int axis = 0; axis < 3; ++axis) {
                  rays[lane].origin[axis] = std::bit_cast<float>(origin[axis]);
                  rays[lane].direction[axis] = std::bit_cast<float>(direction[axis]);
                }
              }
            }
            if (!alias) {
              for (auto &row : output) {
                std::fill_n(row, 64, 0xdeadbeef);
              }
            }
            for (int i = 0; i < 4; ++i) {
              std::copy_n(d[i], 64, before[i]);
            }
            const uint64_t exec_mask = (1ULL << (lanes - 1)) | 6;
            const uint64_t fetch_mask = exec_mask & ~4ULL;
            int status =
                lanes == 32
                    ? rj_arith_image_bvh_intersect_ray_w32(cpu | semantics, uint32_t(exec_mask), 0,
                                                           d, nodes, 128, uint32_t(fetch_mask),
                                                           rays, &controls)
                    : rj_arith_image_bvh_intersect_ray_w64(cpu | semantics, exec_mask, 0, d, nodes,
                                                           128, fetch_mask, rays, &controls);
            const bool supported = semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
                                   semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
            ASSERT_EQ(status, supported ? RJ_ARITH_SUCCESS : RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            for (int i = 0; i < 4; ++i) {
              for (int lane = 0; lane < lanes; ++lane) {
                uint32_t expected = before[i][lane];
                if (supported && ((exec_mask >> lane) & 1)) {
                  expected = lane == 2 ? 0xffffffff : boxes ? uint32_t(8 * (i + 1) + 5) : golden[i];
                }
                EXPECT_EQ(d[i][lane], expected);
              }
            }
          }
        }
      }
    }
  }
}

TEST(BvhIntersection, NaNVertices) {
  // Two NaN coordinates expose host-dependent sign and payload propagation.
  uint32_t node[16]{0xffc00001, 0x7f800001, 0xbfc5c580, 0xc0a232a0, 0x3f285210,
                    0x40a6ef68, 0xc0fd5513, 0x3ff1a668, 0x402993f4};
  node[15] = 9; // Return the weights of vertices 1 and 2.
  rj_arith_bvh_ray rays[32]{};
  constexpr uint32_t origin[] = {0xc0c9b3cb, 0x40651057, 0x3d2c07a0};
  constexpr uint32_t direction[] = {0x40b0161c, 0xc0a534f0, 0x3f8313e8};
  for (int axis = 0; axis < 3; ++axis) {
    rays[0].origin[axis] = std::bit_cast<float>(origin[axis]);
    rays[0].direction[axis] = std::bit_cast<float>(direction[axis]);
  }
  const rj_arith_bvh_controls controls{0, 0, 1, 1, 0};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_LOOSE}) {
      SCOPED_TRACE(testing::Message() << cpu << ":" << semantics);
      uint32_t output[4][32]{};
      uint32_t *d[] = {output[0], output[1], output[2], output[3]};
      ASSERT_EQ(rj_arith_image_bvh_intersect_ray_w32(cpu | semantics, 1, 0, d, node, sizeof(node),
                                                     1, rays, &controls),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(output[0][0], 0x7f800000u);
      for (int word = 1; word < 4; ++word) {
        EXPECT_EQ(output[word][0], 0xffc00000u);
      }
    }
  }
}

} // namespace
