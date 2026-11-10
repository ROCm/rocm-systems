// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar word/byte packing and quad-mask expansion/compression.

#include "scalar/integer_helpers.h"

// Check integer results and SCC against the supplied scalar reference.
TEST(s_cselect_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t cc, uint32_t *) {
        return rj_arith_s_cselect_b32(f, m, d, a, b, cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = (cc & 1) ? a : b;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_quadmask_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_quadmask_b32(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = 0;
        for (int quad = 0; quad < 8; ++quad) {
          if ((a >> (4 * quad)) & 15) {
            result |= 1ULL << quad;
          }
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_wqm_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_wqm_b32(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint64_t result = 0;
        for (int quad = 0; quad < 8; ++quad) {
          if ((a >> (4 * quad)) & 15) {
            result |= 15ULL << (4 * quad);
          }
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_cselect_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t b, uint32_t cc, uint32_t *) {
        return rj_arith_s_cselect_b64(f, m, d, a, b, cc);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t b = uint64_t(raw_b);
        uint64_t result = (cc & 1) ? a : b;
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_quadmask_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_quadmask_b64(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t result = 0;
        for (int quad = 0; quad < 16; ++quad) {
          if ((a >> (4 * quad)) & 15) {
            result |= 1ULL << quad;
          }
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_wqm_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_wqm_b64(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t result = 0;
        for (int quad = 0; quad < 16; ++quad) {
          if ((a >> (4 * quad)) & 15) {
            result |= 15ULL << (4 * quad);
          }
        }
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_pack_ll_b32_b16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_pack_ll_b32_b16(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = ((a >> 0) & 0xffffU) | (((b >> 0) & 0xffffU) << 16);
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_pack_lh_b32_b16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_pack_lh_b32_b16(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = ((a >> 0) & 0xffffU) | (((b >> 16) & 0xffffU) << 16);
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_pack_hl_b32_b16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_pack_hl_b32_b16(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = ((a >> 16) & 0xffffU) | (((b >> 0) & 0xffffU) << 16);
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_pack_hh_b32_b16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t b, uint32_t, uint32_t *) {
        return rj_arith_s_pack_hh_b32_b16(f, m, d, a, b);
      },
      [](uint64_t raw_a, uint64_t raw_b, uint32_t cc) {
        uint32_t a = uint32_t(raw_a);
        uint32_t b = uint32_t(raw_b);
        uint64_t result = ((a >> 16) & 0xffffU) | (((b >> 16) & 0xffffU) << 16);
        return std::make_pair(result, cc);
      },
      false);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bitreplicate_b64_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint64_t>(
      [](uint64_t f, uint64_t m, uint64_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_bitreplicate_b64_b32(f, m, d, a);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t a = uint64_t(raw_a);
        uint64_t result = 0;
        for (int bit = 0; bit < 32; ++bit) {
          if ((a >> bit) & 1) {
            result |= 3ULL << (2 * bit);
          }
        }
        return std::make_pair(result, cc);
      },
      false);
}
