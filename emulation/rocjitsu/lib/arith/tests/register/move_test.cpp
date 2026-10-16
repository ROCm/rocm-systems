// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for whole-word and selected-halfword register moves.

#include "register/helpers.h"

// Check register routing and preservation of unselected destination data.
TEST(v_mov_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_mov_b32_w32(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                   ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                   : bool((e >> lane) & 1))) {
            out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[0][source],
                                                               in.v[d][lane], m, false);
          }
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_cndmask_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_cndmask_b32_w32(f, e, m, r.v + d, r.v, r.v + 1,
                                          rocjitsu::arith::test::exec_mask_for_case<32>(s + 2));
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        auto condition = rocjitsu::arith::test::exec_mask_for_case<32>(s + 2);
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          bool b = (condition >> lane) & 1;
          uint32_t word = b ? in.v[1][lane] : source < 0 ? 0 : in.v[0][source];
          uint64_t select = m & (RJ_ARITH_ALU_HIGH_D);
          if (m & (b ? RJ_ARITH_ALU_NEG_B : RJ_ARITH_ALU_NEG_A)) {
            select |= RJ_ARITH_ALU_NEG_A;
          }
          if (m & (b ? RJ_ARITH_ALU_ABS_B : RJ_ARITH_ALU_ABS_A)) {
            select |= RJ_ARITH_ALU_ABS_A;
          }
          if (m & (b ? RJ_ARITH_ALU_HIGH_B : RJ_ARITH_ALU_HIGH_A)) {
            select |= RJ_ARITH_ALU_HIGH_A;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(word, in.v[d][lane], select, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
          RJ_ARITH_ALU_CLAMP);
}

// Check register routing and preservation of unselected destination data.
TEST(v_mov_b16_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_mov_b16_w32(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                   ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                   : bool((e >> lane) & 1))) {
            out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[0][source],
                                                               in.v[d][lane], m, true);
          }
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D |
          RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP);
}

// Check register routing and preservation of unselected destination data.
TEST(v_cndmask_b16_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_cndmask_b16_w32(f, e, m, r.v + d, r.v, r.v + 1,
                                          rocjitsu::arith::test::exec_mask_for_case<32>(s + 2));
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        auto condition = rocjitsu::arith::test::exec_mask_for_case<32>(s + 2);
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          bool b = (condition >> lane) & 1;
          uint32_t word = b ? in.v[1][lane] : source < 0 ? 0 : in.v[0][source];
          uint64_t select = m & (RJ_ARITH_ALU_HIGH_D);
          if (m & (b ? RJ_ARITH_ALU_NEG_B : RJ_ARITH_ALU_NEG_A)) {
            select |= RJ_ARITH_ALU_NEG_A;
          }
          if (m & (b ? RJ_ARITH_ALU_ABS_B : RJ_ARITH_ALU_ABS_A)) {
            select |= RJ_ARITH_ALU_ABS_A;
          }
          if (m & (b ? RJ_ARITH_ALU_HIGH_B : RJ_ARITH_ALU_HIGH_A)) {
            select |= RJ_ARITH_ALU_HIGH_A;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(word, in.v[d][lane], select, true);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
          RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D);
}

// Check register routing and preservation of unselected destination data.
TEST(v_mov_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_mov_b32_w64(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                   ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                   : bool((e >> lane) & 1))) {
            out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[0][source],
                                                               in.v[d][lane], m, false);
          }
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_cndmask_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_cndmask_b32_w64(f, e, m, r.v + d, r.v, r.v + 1,
                                          rocjitsu::arith::test::exec_mask_for_case<64>(s + 2));
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        auto condition = rocjitsu::arith::test::exec_mask_for_case<64>(s + 2);
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          bool b = (condition >> lane) & 1;
          uint32_t word = b ? in.v[1][lane] : source < 0 ? 0 : in.v[0][source];
          uint64_t select = m & (RJ_ARITH_ALU_HIGH_D);
          if (m & (b ? RJ_ARITH_ALU_NEG_B : RJ_ARITH_ALU_NEG_A)) {
            select |= RJ_ARITH_ALU_NEG_A;
          }
          if (m & (b ? RJ_ARITH_ALU_ABS_B : RJ_ARITH_ALU_ABS_A)) {
            select |= RJ_ARITH_ALU_ABS_A;
          }
          if (m & (b ? RJ_ARITH_ALU_HIGH_B : RJ_ARITH_ALU_HIGH_A)) {
            select |= RJ_ARITH_ALU_HIGH_A;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(word, in.v[d][lane], select, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
          RJ_ARITH_ALU_CLAMP);
}

// Check register routing and preservation of unselected destination data.
TEST(v_mov_b16_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_mov_b16_w64(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                   ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                   : bool((e >> lane) & 1))) {
            out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[0][source],
                                                               in.v[d][lane], m, true);
          }
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D |
          RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP);
}

// Check register routing and preservation of unselected destination data.
TEST(v_cndmask_b16_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_cndmask_b16_w64(f, e, m, r.v + d, r.v, r.v + 1,
                                          rocjitsu::arith::test::exec_mask_for_case<64>(s + 2));
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        auto condition = rocjitsu::arith::test::exec_mask_for_case<64>(s + 2);
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          bool b = (condition >> lane) & 1;
          uint32_t word = b ? in.v[1][lane] : source < 0 ? 0 : in.v[0][source];
          uint64_t select = m & (RJ_ARITH_ALU_HIGH_D);
          if (m & (b ? RJ_ARITH_ALU_NEG_B : RJ_ARITH_ALU_NEG_A)) {
            select |= RJ_ARITH_ALU_NEG_A;
          }
          if (m & (b ? RJ_ARITH_ALU_ABS_B : RJ_ARITH_ALU_ABS_A)) {
            select |= RJ_ARITH_ALU_ABS_A;
          }
          if (m & (b ? RJ_ARITH_ALU_HIGH_B : RJ_ARITH_ALU_HIGH_A)) {
            select |= RJ_ARITH_ALU_HIGH_A;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(word, in.v[d][lane], select, true);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
          RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D);
}
