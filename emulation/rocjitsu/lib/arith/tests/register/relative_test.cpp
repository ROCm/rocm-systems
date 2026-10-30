// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for M0-relative VGPR reads and writes.

#include "register/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {
constexpr uint32_t offsets[] = {0, 1, 2, 5, 6, 255, 256, 0x00020001, 0x03ff03ff, 0xffffffff};
enum class Relative { Source, Destination, Both, Split };

// Check each M0 offset layout, including out-of-bank reads and ignored writes.
template <int Lanes, typename Call> void check_relative_move(Call call, Relative operation) {
  rocjitsu::arith::test::check_register<Lanes, true>(
      [=](auto flags, auto exec_mask, auto mode, auto &registers, int destination, int scenario) {
        return call(flags, exec_mask, mode, registers.v, registers.v, 6, 6, destination, 0,
                    offsets[scenario % 10]);
      },
      [=](auto &expected, const auto &input, auto exec_mask, auto mode, int destination,
          int scenario) {
        uint32_t m0 = offsets[scenario % 10];
        uint64_t a_offset = operation == Relative::Destination ? 0
                            : operation == Relative::Split     ? m0 & 0x3ff
                                                               : m0;
        uint64_t d_offset = operation == Relative::Source  ? 0
                            : operation == Relative::Split ? (m0 >> 16) & 0x3ff
                                                           : m0;
        if (d_offset > 255 || uint64_t(destination) + d_offset >= 6) {
          return;
        }
        destination += int(d_offset);
        int source_register = a_offset < 6 ? int(a_offset) : 0;
        for (int lane = 0; lane < Lanes; ++lane) {
          int source = lane;
          if (!((mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(mode, exec_mask, lane, source)
                    : bool((exec_mask >> lane) & 1))) {
            continue;
          }
          expected.v[destination][lane] =
              rocjitsu::arith::test::moved_word(source < 0 ? 0 : input.v[source_register][source],
                                                input.v[destination][lane], mode, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check swaps separately: an out-of-bank source suppresses both writes.
template <int Lanes, typename Call> void check_relative_swap(Call call) {
  rocjitsu::arith::test::check_register<Lanes, false>(
      [=](auto flags, auto exec_mask, auto mode, auto &registers, int destination, int scenario) {
        return call(flags, exec_mask, mode, registers.v, registers.v, 6, 6, destination, 0,
                    offsets[scenario % 10]);
      },
      [](auto &expected, const auto &input, auto exec_mask, auto mode, int destination,
         int scenario) {
        uint32_t m0 = offsets[scenario % 10];
        uint64_t a_offset = (m0 & 0x3ff);
        uint64_t d_offset = (m0 >> 16) & 0x3ff;
        if (d_offset > 255 || uint64_t(destination) + d_offset >= 6) {
          return;
        }
        destination += int(d_offset);
        if (a_offset >= 6) {
          return;
        }
        int source_register = int(a_offset);
        for (int lane = 0; lane < Lanes; ++lane) {
          int source = lane;
          if (!((mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(mode, exec_mask, lane, source)
                    : bool((exec_mask >> lane) & 1))) {
            continue;
          }
          expected.v[destination][lane] = input.v[source_register][lane];
          expected.v[source_register][lane] = input.v[destination][lane];
        }
      },
      0);
}
} // namespace

TEST(v_movrels_b32_w32, Arithmetic) {
  check_relative_move<32>(rj_arith_v_movrels_b32_w32, Relative::Source);
}

TEST(v_movreld_b32_w32, Arithmetic) {
  check_relative_move<32>(rj_arith_v_movreld_b32_w32, Relative::Destination);
}

TEST(v_movrelsd_b32_w32, Arithmetic) {
  check_relative_move<32>(rj_arith_v_movrelsd_b32_w32, Relative::Both);
}

TEST(v_movrelsd_2_b32_w32, Arithmetic) {
  check_relative_move<32>(rj_arith_v_movrelsd_2_b32_w32, Relative::Split);
}

TEST(v_swaprel_b32_w32, Arithmetic) { check_relative_swap<32>(rj_arith_v_swaprel_b32_w32); }

TEST(v_movrels_b32_w64, Arithmetic) {
  check_relative_move<64>(rj_arith_v_movrels_b32_w64, Relative::Source);
}

TEST(v_movreld_b32_w64, Arithmetic) {
  check_relative_move<64>(rj_arith_v_movreld_b32_w64, Relative::Destination);
}

TEST(v_movrelsd_b32_w64, Arithmetic) {
  check_relative_move<64>(rj_arith_v_movrelsd_b32_w64, Relative::Both);
}

TEST(v_movrelsd_2_b32_w64, Arithmetic) {
  check_relative_move<64>(rj_arith_v_movrelsd_2_b32_w64, Relative::Split);
}

TEST(v_swaprel_b32_w64, Arithmetic) { check_relative_swap<64>(rj_arith_v_swaprel_b32_w64); }
