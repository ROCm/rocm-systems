// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for lane permutations across rows and half-waves.

#include "register/helpers.h"

// Check register routing and preservation of unselected destination data.
TEST(v_permlane16_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlane16_b32_w32(f, e, m, r.v + d, r.v, 0x75310246U, 0xfedcba98U);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(((lane % 16 < 8 ? 0x75310246U : 0xfedcba98U) >> (4 * (lane % 8))) & 15);
            int source = ((lane / 16) ^ 0) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlane16_var_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlane16_var_b32_w32(f, e, m, r.v + d, r.v, r.v + 1);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(in.v[1][lane] & 15);
            int source = ((lane / 16) ^ 0) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlanex16_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlanex16_b32_w32(f, e, m, r.v + d, r.v, 0x75310246U, 0xfedcba98U);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(((lane % 16 < 8 ? 0x75310246U : 0xfedcba98U) >> (4 * (lane % 8))) & 15);
            int source = ((lane / 16) ^ 1) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlanex16_var_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlanex16_var_b32_w32(f, e, m, r.v + d, r.v, r.v + 1);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(in.v[1][lane] & 15);
            int source = ((lane / 16) ^ 1) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlane64_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        (void)e;
        return rj_arith_v_permlane64_b32_w32(f, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto, int d, int) {
        (void)out;
        (void)in;
        (void)e;
        (void)d;
        // Wave32 is an unconditional no-op.
      });
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlane16_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlane16_b32_w64(f, e, m, r.v + d, r.v, 0x75310246U, 0xfedcba98U);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(((lane % 16 < 8 ? 0x75310246U : 0xfedcba98U) >> (4 * (lane % 8))) & 15);
            int source = ((lane / 16) ^ 0) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlane16_var_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlane16_var_b32_w64(f, e, m, r.v + d, r.v, r.v + 1);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(in.v[1][lane] & 15);
            int source = ((lane / 16) ^ 0) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlanex16_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlanex16_b32_w64(f, e, m, r.v + d, r.v, 0x75310246U, 0xfedcba98U);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(((lane % 16 < 8 ? 0x75310246U : 0xfedcba98U) >> (4 * (lane % 8))) & 15);
            int source = ((lane / 16) ^ 1) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlanex16_var_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        return rj_arith_v_permlanex16_var_b32_w64(f, e, m, r.v + d, r.v, r.v + 1);
      },
      [](auto &out, const auto &in, auto e, auto m, int d, int) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            int index = int(in.v[1][lane] & 15);
            int source = ((lane / 16) ^ 1) * 16 + index;
            if (((e >> source) & 1) || (m & RJ_ARITH_PERMLANE_FI)) {
              out.v[d][lane] = in.v[0][source];
            } else if (m & RJ_ARITH_PERMLANE_BOUND_CTRL) {
              out.v[d][lane] = 0;
            }
          }
        }
      },
      RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL);
}

// Check register routing and preservation of unselected destination data.
TEST(v_permlane64_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int) {
        (void)e;
        return rj_arith_v_permlane64_b32_w64(f, e, m, r.v + d, r.v);
      },
      [](auto &out, const auto &in, auto e, auto, int d, int) {
        (void)out;
        (void)in;
        (void)e;
        (void)d;
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = in.v[0][lane ^ 32];
          }
        }
      });
}
