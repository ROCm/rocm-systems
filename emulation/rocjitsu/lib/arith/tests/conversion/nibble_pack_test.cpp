// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Nibble saturation covers byte boundaries, both destination halves and aliasing.
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <random>
namespace {
// Use deterministic words plus explicit byte-clamp boundaries on every CPU path.
template <int Lanes, bool Signed, typename Function> void check(Function function) {
  rocjitsu::arith::test::Registers<2, Lanes> input;
  std::minstd_rand rng;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = (uint32_t(rng()) >> 15) | ((uint32_t(rng()) >> 15) << 16);
    input.v[1][lane] = 0xdeadbeefU;
  }
  input.v[0][0] = 0x0807f8f7U;
  input.v[0][1] = 0xff100f00U;
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int destination : {0, 1}) {
        for (uint64_t routing : {uint64_t{0}, rocjitsu::arith::test::dpp_modes[0],
                                 rocjitsu::arith::test::dpp_modes[6]}) {
          for (int shift : {0, 16}) {
            auto actual = input;
            uint64_t exec_mask = 0xaaaaaaaaaaaaaaabULL;
            uint64_t mode = routing | (shift ? RJ_ARITH_ALU_HIGH_D : 0) | RJ_ARITH_ALU_CLAMP;
            ASSERT_EQ(function(cpu | semantics, exec_mask, mode, actual.v + destination, actual.v),
                      RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < Lanes; ++lane) {
              int source = lane;
              bool active = routing
                                ? rocjitsu::arith::test::dpp_source(mode, exec_mask, lane, source)
                                : bool((exec_mask >> lane) & 1);
              uint32_t raw = source < 0 ? 0 : input.v[0][source];
              uint32_t packed = 0;
              for (int byte = 0; byte < 4; ++byte) {
                int value = (raw >> (8 * byte)) & 255;
                if (Signed && value >= 128) {
                  value -= 256;
                }
                int low = Signed ? -8 : 0, high = Signed ? 7 : 15;
                value = value < low ? low : value > high ? high : value;
                packed |= (uint32_t(value) & 15) << (4 * byte);
              }
              uint32_t expected = input.v[destination][lane];
              if (active) {
                expected = (expected & ~(0xffffU << shift)) | (packed << shift);
              }
              EXPECT_EQ(actual.v[destination][lane], expected);
            }
            actual.expect_preserved(input, destination, 1);
          }
        }
      }
    }
  }
}
} // namespace
TEST(NibblePack, SignedWave32) { check<32, true>(rj_arith_v_sat_pk4_i4_i8_w32); }
TEST(NibblePack, SignedWave64) { check<64, true>(rj_arith_v_sat_pk4_i4_i8_w64); }
TEST(NibblePack, UnsignedWave32) { check<32, false>(rj_arith_v_sat_pk4_u4_u8_w32); }
TEST(NibblePack, UnsignedWave64) { check<64, false>(rj_arith_v_sat_pk4_u4_u8_w64); }
