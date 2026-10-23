// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for lane permutations across rows and half-waves.

#include "register/helpers.h"

namespace {

// Check immediate/per-lane selection within a row or from its adjacent row.
template <int Lanes, bool Variable, typename Call>
void check_permlane16(Call call, bool cross_row) {
  rocjitsu::arith::test::check_register<Lanes>(
      [=](auto flags, auto exec_mask, auto mode, auto &registers, int destination, int) {
        if constexpr (Variable) {
          return call(flags, exec_mask, mode, registers.v + destination, registers.v,
                      registers.v + 1);
        } else {
          return call(flags, exec_mask, mode, registers.v + destination, registers.v, 0x75310246U,
                      0xfedcba98U);
        }
      },
      [=](auto &expected, const auto &input, auto exec_mask, auto mode, int destination, int) {
        for (int lane = 0; lane < Lanes; ++lane) {
          if (!((exec_mask >> lane) & 1)) {
            continue;
          }
          int index =
              Variable
                  ? int(input.v[1][lane] & 15)
                  : int(((lane % 16 < 8 ? 0x75310246U : 0xfedcba98U) >> (4 * (lane % 8))) & 15);
          int source = ((lane / 16) ^ int(cross_row)) * 16 + index;
          if (((exec_mask >> source) & 1) || (mode & RJ_ARITH_PERMLANE_FI)) {
            expected.v[destination][lane] = input.v[0][source];
          } else if (mode & RJ_ARITH_PERMLANE_BOUND_CTRL) {
            expected.v[destination][lane] = 0;
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

} // namespace

TEST(v_permlane16_b32_w32, Arithmetic) {
  check_permlane16<32, false>(rj_arith_v_permlane16_b32_w32, /*cross_row=*/false);
}

TEST(v_permlane16_var_b32_w32, Arithmetic) {
  check_permlane16<32, true>(rj_arith_v_permlane16_var_b32_w32, /*cross_row=*/false);
}

TEST(v_permlanex16_b32_w32, Arithmetic) {
  check_permlane16<32, false>(rj_arith_v_permlanex16_b32_w32, /*cross_row=*/true);
}

TEST(v_permlanex16_var_b32_w32, Arithmetic) {
  check_permlane16<32, true>(rj_arith_v_permlanex16_var_b32_w32, /*cross_row=*/true);
}

TEST(v_permlane64_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto, auto m, auto &r, int d, int) {
        return rj_arith_v_permlane64_b32_w32(f, m, r.v + d, r.v);
      },
      [](auto &, const auto &, auto, auto, int, int) {
        // Wave32 is an unconditional no-op.
      });
}

TEST(v_permlane16_b32_w64, Arithmetic) {
  check_permlane16<64, false>(rj_arith_v_permlane16_b32_w64, /*cross_row=*/false);
}

TEST(v_permlane16_var_b32_w64, Arithmetic) {
  check_permlane16<64, true>(rj_arith_v_permlane16_var_b32_w64, /*cross_row=*/false);
}

TEST(v_permlanex16_b32_w64, Arithmetic) {
  check_permlane16<64, false>(rj_arith_v_permlanex16_b32_w64, /*cross_row=*/true);
}

TEST(v_permlanex16_var_b32_w64, Arithmetic) {
  check_permlane16<64, true>(rj_arith_v_permlanex16_var_b32_w64, /*cross_row=*/true);
}

TEST(v_permlane64_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlane64_b32_w64(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto, int d, int) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = in.v[0][lane ^ 32];
          }
        }
      });
}
