// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for whole-word, halfword, and relative-register swaps.

#include "register/helpers.h"
#include "rj_arith/detail/rj_arith_register.h"

// Check register routing and preservation of unselected destination data.
TEST(RegisterVSwapB32W32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_swap_b32_w32(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        (void)m;
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = in.v[0][lane];
            out.v[0][lane] = in.v[d][lane];
          }
        }
      },
      0);
}

// Check register routing and preservation of unselected destination data.
TEST(RegisterVSwapB16W32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_swap_b16_w32(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        (void)m;
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            int sa = m & RJ_ARITH_ALU_HIGH_A ? 16 : 0, sd = m & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
            uint32_t a = (in.v[0][lane] >> sa) & 0xffffU, b = (in.v[d][lane] >> sd) & 0xffffU;
            out.v[d][lane] = (out.v[d][lane] & ~(0xffffU << sd)) | (a << sd);
            out.v[0][lane] = (out.v[0][lane] & ~(0xffffU << sa)) | (b << sa);
          }
        }
      },
      RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
}

// Check register routing and preservation of unselected destination data.
TEST(RegisterVSwapB32W64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_swap_b32_w64(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        (void)m;
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = in.v[0][lane];
            out.v[0][lane] = in.v[d][lane];
          }
        }
      },
      0);
}

// Check register routing and preservation of unselected destination data.
TEST(RegisterVSwapB16W64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_swap_b16_w64(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        (void)m;
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            int sa = m & RJ_ARITH_ALU_HIGH_A ? 16 : 0, sd = m & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
            uint32_t a = (in.v[0][lane] >> sa) & 0xffffU, b = (in.v[d][lane] >> sd) & 0xffffU;
            out.v[d][lane] = (out.v[d][lane] & ~(0xffffU << sd)) | (a << sd);
            out.v[0][lane] = (out.v[0][lane] & ~(0xffffU << sa)) | (b << sa);
          }
        }
      },
      RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
}

namespace {
// Compare adjacent-row exchange with a snapshot, including aliased VGPRs.
template <int Lanes, int Stride, typename Call> void check_row_swap(Call call) {
  rocjitsu::arith::test::check_register<Lanes>(
      [call](auto f, auto e, auto m, auto &r, int d, int) { return call(f, e, m, r.v + d, r.v); },
      [](auto &out, const auto &in, auto e, auto, int d, int) {
        for (int lane = 0; lane < Lanes; ++lane) {
          if (Lanes <= Stride || !((e >> lane) & 1)) {
            continue;
          }
          if (lane & Stride) {
            out.v[d][lane] = in.v[0][lane ^ Stride];
          } else {
            out.v[0][lane] = in.v[d][lane ^ Stride];
          }
        }
      });
}
} // namespace

TEST(RegisterPermlane16SwapW32, Arithmetic) {
  check_row_swap<32, 16>(rj_arith_v_permlane16_swap_b32_w32);
}

TEST(RegisterPermlane16SwapW64, Arithmetic) {
  check_row_swap<64, 16>(rj_arith_v_permlane16_swap_b32_w64);
}

TEST(RegisterPermlane32SwapW32, Arithmetic) {
  check_row_swap<32, 32>(rj_arith_v_permlane32_swap_b32_w32);
}

TEST(RegisterPermlane32SwapW64, Arithmetic) {
  check_row_swap<64, 32>(rj_arith_v_permlane32_swap_b32_w64);
}
