// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for M0-relative VGPR reads and writes.

#include "register/helpers.h"

namespace {
static const uint32_t offsets[] = {0, 1, 2, 5, 6, 255, 256, 0x00020001, 0x03ff03ff, 0xffffffff};
} // namespace

// Check register routing and preservation of unselected destination data.
TEST(v_movrels_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movrels_b32_w32(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = m0;
        uint64_t d_offset = 0;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_movreld_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movreld_b32_w32(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = 0;
        uint64_t d_offset = m0;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_movrelsd_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movrelsd_b32_w32(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = m0;
        uint64_t d_offset = m0;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_movrelsd_2_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movrelsd_2_b32_w32(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = (m0 & 0x3ff);
        uint64_t d_offset = (m0 >> 16) & 0x3ff;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_swaprel_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32, false>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_swaprel_b32_w32(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = (m0 & 0x3ff);
        uint64_t d_offset = (m0 >> 16) & 0x3ff;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          return;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = in.v[a][lane];
          out.v[a][lane] = in.v[d][lane];
        }
      },
      0);
}

// Check register routing and preservation of unselected destination data.
TEST(v_movrels_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movrels_b32_w64(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = m0;
        uint64_t d_offset = 0;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_movreld_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movreld_b32_w64(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = 0;
        uint64_t d_offset = m0;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_movrelsd_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movrelsd_b32_w64(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = m0;
        uint64_t d_offset = m0;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_movrelsd_2_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, true>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_movrelsd_2_b32_w64(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = (m0 & 0x3ff);
        uint64_t d_offset = (m0 >> 16) & 0x3ff;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          a = 0;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = rocjitsu::arith::test::moved_word(source < 0 ? 0 : in.v[a][source],
                                                             in.v[d][lane], m, false);
        }
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
}

// Check register routing and preservation of unselected destination data.
TEST(v_swaprel_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64, false>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_v_swaprel_b32_w64(f, e, m, r.v, r.v, 6, 6, d, 0, offsets[s % 10]);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int s) {
        uint32_t m0 = offsets[s % 10];
        uint64_t a_offset = (m0 & 0x3ff);
        uint64_t d_offset = (m0 >> 16) & 0x3ff;
        if (d_offset > 255 || uint64_t(d) + d_offset >= 6) {
          return;
        }
        d += int(d_offset);
        int a = 0;
        if (a_offset > 255 || a_offset >= 6) {
          return;
        } else {
          a = int(a_offset);
        }
        for (int lane = 0; lane < 64; ++lane) {
          int source = lane;
          if (!((m & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                    ? rocjitsu::arith::test::dpp_source(m, e, lane, source)
                    : bool((e >> lane) & 1))) {
            continue;
          }
          out.v[d][lane] = in.v[a][lane];
          out.v[a][lane] = in.v[d][lane];
        }
      },
      0);
}
