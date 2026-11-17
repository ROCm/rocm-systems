// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for lane permutations across rows and half-waves.

#include "register/helpers.h"
#include "rj_arith/detail/rj_arith_register.h"

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

TEST(RegisterVPermlane16B32W32, Arithmetic) {
  check_permlane16<32, false>(rj_arith_v_permlane16_b32_w32, /*cross_row=*/false);
}

TEST(RegisterVPermlane16VarB32W32, Arithmetic) {
  check_permlane16<32, true>(rj_arith_v_permlane16_var_b32_w32, /*cross_row=*/false);
}

TEST(RegisterVPermlanex16B32W32, Arithmetic) {
  check_permlane16<32, false>(rj_arith_v_permlanex16_b32_w32, /*cross_row=*/true);
}

TEST(RegisterVPermlanex16VarB32W32, Arithmetic) {
  check_permlane16<32, true>(rj_arith_v_permlanex16_var_b32_w32, /*cross_row=*/true);
}

TEST(RegisterVPermlane64B32W32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto, auto m, auto &r, int d, int) {
        return rj_arith_v_permlane64_b32_w32(f, m, r.v + d, r.v);
      },
      [](auto &, const auto &, auto, auto, int, int) {
        // Wave32 is an unconditional no-op.
      });
}

TEST(RegisterVPermlane16B32W64, Arithmetic) {
  check_permlane16<64, false>(rj_arith_v_permlane16_b32_w64, /*cross_row=*/false);
}

TEST(RegisterVPermlane16VarB32W64, Arithmetic) {
  check_permlane16<64, true>(rj_arith_v_permlane16_var_b32_w64, /*cross_row=*/false);
}

TEST(RegisterVPermlanex16B32W64, Arithmetic) {
  check_permlane16<64, false>(rj_arith_v_permlanex16_b32_w64, /*cross_row=*/true);
}

TEST(RegisterVPermlanex16VarB32W64, Arithmetic) {
  check_permlane16<64, true>(rj_arith_v_permlanex16_var_b32_w64, /*cross_row=*/true);
}

TEST(RegisterVPermlane64B32W64, Arithmetic) {
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

namespace {
enum class GroupOp { Bcast, Up, Down, Xor };

// Deliberately use division/modulo in the reference to check the SIMD index
// arithmetic. Include invalid widths and selectors that wrap uint32 addition.
template <int Lanes, GroupOp Op, typename Call> void check_group_permutation(Call call) {
  constexpr uint32_t widths[] = {1, 2, 8, 16, 32, 64, 128, 0, 3, 0x80000000U};
  constexpr uint32_t selectors[] = {0, 1, 7, 31, 32, 63, 64, 0xffffffffU};
  rocjitsu::arith::test::check_register<Lanes>(
      [&](auto f, auto e, auto m, auto &r, int d, int scenario) {
        const uint32_t *a = r.v[0];
        return call(f, e, m, r.v + d, &a, selectors[scenario % 8], widths[scenario % 10]);
      },
      [&](auto &out, const auto &in, auto e, auto, int d, int scenario) {
        uint32_t width = widths[scenario % 10], selector = selectors[scenario % 8];
        if (!width || width > Lanes || (width & (width - 1))) {
          width = Lanes;
        }
        for (int lane = 0; lane < Lanes; ++lane) {
          if (!((e >> lane) & 1)) {
            continue;
          }
          uint32_t base = (uint32_t(lane) / width) * width;
          uint32_t offset = uint32_t(lane) % width, source = lane;
          if constexpr (Op == GroupOp::Bcast) {
            source = base + selector % width;
          } else if constexpr (Op == GroupOp::Up) {
            source = selector <= offset ? uint32_t(lane) - selector : uint32_t(lane);
          } else if constexpr (Op == GroupOp::Down) {
            if (uint64_t(offset) + selector < width) {
              source = lane + selector;
            }
          } else {
            uint32_t candidate = uint32_t(lane) ^ selector;
            if (selector < Lanes && candidate < base + width) {
              source = candidate;
            }
          }
          out.v[d][lane] = in.v[0][source];
        }
      });
}
} // namespace

TEST(RegisterGroupBcastW32, Arithmetic) {
  check_group_permutation<32, GroupOp::Bcast>(rj_arith_v_permlane_bcast_b32_w32);
}

TEST(RegisterGroupBcastW64, Arithmetic) {
  check_group_permutation<64, GroupOp::Bcast>(rj_arith_v_permlane_bcast_b32_w64);
}

TEST(RegisterGroupUpW32, Arithmetic) {
  check_group_permutation<32, GroupOp::Up>(rj_arith_v_permlane_up_b32_w32);
}

TEST(RegisterGroupUpW64, Arithmetic) {
  check_group_permutation<64, GroupOp::Up>(rj_arith_v_permlane_up_b32_w64);
}

TEST(RegisterGroupDownW32, Arithmetic) {
  check_group_permutation<32, GroupOp::Down>(rj_arith_v_permlane_down_b32_w32);
}

TEST(RegisterGroupDownW64, Arithmetic) {
  check_group_permutation<64, GroupOp::Down>(rj_arith_v_permlane_down_b32_w64);
}

TEST(RegisterGroupXorW32, Arithmetic) {
  check_group_permutation<32, GroupOp::Xor>(rj_arith_v_permlane_xor_b32_w32);
}

TEST(RegisterGroupXorW64, Arithmetic) {
  check_group_permutation<64, GroupOp::Xor>(rj_arith_v_permlane_xor_b32_w64);
}

namespace {
// Group widths include values larger than a wave; indices retain all six low bits.
template <int Lanes, typename Call> void check_index_generation(Call call) {
  constexpr uint32_t widths[] = {1, 2, 8, 16, 32, 64, 128};
  rocjitsu::arith::test::check_register<Lanes>(
      [&](auto flags, auto exec_mask, auto mode, auto &registers, int destination, int scenario) {
        return call(flags, exec_mask, mode, registers.v + destination, registers.v,
                    widths[scenario % 7]);
      },
      [&](auto &expected, const auto &input, auto exec_mask, auto, int destination, int scenario) {
        for (int lane = 0; lane < Lanes; ++lane) {
          if ((exec_mask >> lane) & 1) {
            uint32_t width = widths[scenario % 7];
            expected.v[destination][lane] =
                ((uint32_t(lane) / width) * width + (input.v[0][lane] % 64) % width) * 4;
          }
        }
      });
}
} // namespace
TEST(RegisterPermlaneIndexGeneration, Wave32) {
  check_index_generation<32>(rj_arith_v_permlane_idx_gen_b32_w32);
}
TEST(RegisterPermlaneIndexGeneration, Wave64) {
  check_index_generation<64>(rj_arith_v_permlane_idx_gen_b32_w64);
}
