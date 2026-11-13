// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar bit counts, reversal, and significant-bit searches.

#include "rj_arith/detail/rj_arith_scalar.h"
#include "scalar/integer_helpers.h"

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bcnt0_i32_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bcnt0_i32_b32(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t result = 32 - rocjitsu::arith::test::count_word(raw_a, 32);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bcnt1_i32_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bcnt1_i32_b32(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t result = rocjitsu::arith::test::count_word(raw_a, 32);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_ctz_i32_b32, Arithmetic) {
  for (auto operation : {rj_arith_s_ctz_i32_b32, rj_arith_s_ff1_i32_b32}) {
    rocjitsu::arith::test::check_scalar_integer<uint32_t>(
        [operation](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t,
                    uint32_t *) { return operation(f, m, d, a); },
        [](uint64_t raw_a, uint64_t, uint32_t cc) {
          uint64_t result = 0;
          uint64_t value = raw_a & 0xffffffffULL;

          result = UINT32_MAX;
          for (int bit = 0; bit < 32; ++bit) {
            if ((value >> bit) & 1) {
              result = bit;
              break;
            }
          }
          return std::make_pair(result, cc);
        },
        false);
  }
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_clz_i32_u32, Arithmetic) {
  for (auto operation : {rj_arith_s_clz_i32_u32, rj_arith_s_flbit_i32_b32}) {
    rocjitsu::arith::test::check_scalar_integer<uint32_t>(
        [operation](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t,
                    uint32_t *) { return operation(f, m, d, a); },
        [](uint64_t raw_a, uint64_t, uint32_t cc) {
          uint64_t result = 0;
          uint64_t value = raw_a & 0xffffffffULL;

          result = UINT32_MAX;
          for (int bit = 0; bit < 32; ++bit) {
            if ((value >> (31 - bit)) & 1) {
              result = bit;
              break;
            }
          }
          return std::make_pair(result, cc);
        },
        false);
  }
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_cls_i32, Arithmetic) {
  for (auto operation : {rj_arith_s_cls_i32, rj_arith_s_flbit_i32}) {
    rocjitsu::arith::test::check_scalar_integer<uint32_t>(
        [operation](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t,
                    uint32_t *) { return operation(f, m, d, a); },
        [](uint64_t raw_a, uint64_t, uint32_t cc) {
          uint64_t result = 0;
          uint64_t value = raw_a & 0xffffffffULL;
          if (value & (1ULL << 31)) {
            value ^= 0xffffffffULL;
          }
          result = UINT32_MAX;
          for (int bit = 0; bit < 32; ++bit) {
            if ((value >> (31 - bit)) & 1) {
              result = bit;
              break;
            }
          }
          return std::make_pair(result, cc);
        },
        false);
  }
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bcnt0_i32_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bcnt0_i32_b64(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t result = 64 - rocjitsu::arith::test::count_word(raw_a, 64);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_bcnt1_i32_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *out_cc) {
        return rj_arith_s_bcnt1_i32_b64(f, m, d, a, out_cc);
      },
      [](uint64_t raw_a, uint64_t, uint32_t cc) {
        uint64_t result = rocjitsu::arith::test::count_word(raw_a, 64);
        cc = result != 0;
        return std::make_pair(result, cc);
      },
      true);
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_ctz_i32_b64, Arithmetic) {
  for (auto operation : {rj_arith_s_ctz_i32_b64, rj_arith_s_ff1_i32_b64}) {
    rocjitsu::arith::test::check_scalar_integer<uint32_t>(
        [operation](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t,
                    uint32_t *) { return operation(f, m, d, a); },
        [](uint64_t raw_a, uint64_t, uint32_t cc) {
          uint64_t result = 0;
          uint64_t value = raw_a & UINT64_MAX;

          result = UINT32_MAX;
          for (int bit = 0; bit < 64; ++bit) {
            if ((value >> bit) & 1) {
              result = bit;
              break;
            }
          }
          return std::make_pair(result, cc);
        },
        false);
  }
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_clz_i32_u64, Arithmetic) {
  for (auto operation : {rj_arith_s_clz_i32_u64, rj_arith_s_flbit_i32_b64}) {
    rocjitsu::arith::test::check_scalar_integer<uint32_t>(
        [operation](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t,
                    uint32_t *) { return operation(f, m, d, a); },
        [](uint64_t raw_a, uint64_t, uint32_t cc) {
          uint64_t result = 0;
          uint64_t value = raw_a & UINT64_MAX;

          result = UINT32_MAX;
          for (int bit = 0; bit < 64; ++bit) {
            if ((value >> (63 - bit)) & 1) {
              result = bit;
              break;
            }
          }
          return std::make_pair(result, cc);
        },
        false);
  }
}

// Check integer results and SCC against the supplied scalar reference.
TEST(s_cls_i32_i64, Arithmetic) {
  for (auto operation : {rj_arith_s_cls_i32_i64, rj_arith_s_flbit_i32_i64}) {
    rocjitsu::arith::test::check_scalar_integer<uint32_t>(
        [operation](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t,
                    uint32_t *) { return operation(f, m, d, a); },
        [](uint64_t raw_a, uint64_t, uint32_t cc) {
          uint64_t result = 0;
          uint64_t value = raw_a & UINT64_MAX;
          if (value & (1ULL << 63)) {
            value ^= UINT64_MAX;
          }
          result = UINT32_MAX;
          for (int bit = 0; bit < 64; ++bit) {
            if ((value >> (63 - bit)) & 1) {
              result = bit;
              break;
            }
          }
          return std::make_pair(result, cc);
        },
        false);
  }
}

// Compare first-zero searches against a bit-by-bit reference, including all ones.
TEST(s_ff0_i32_b32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_ff0_i32_b32(f, m, d, a);
      },
      [](uint64_t a, uint64_t, uint32_t cc) {
        uint32_t result = UINT32_MAX;
        for (int bit = 0; bit < 32; ++bit) {
          if (!((a >> bit) & 1)) {
            result = bit;
            break;
          }
        }
        return std::make_pair(result, cc);
      },
      false);
}

// Compare first-zero searches against a bit-by-bit reference, including all ones.
TEST(s_ff0_i32_b64, Arithmetic) {
  rocjitsu::arith::test::check_scalar_integer<uint32_t>(
      [](uint64_t f, uint64_t m, uint32_t *d, uint64_t a, uint64_t, uint32_t, uint32_t *) {
        return rj_arith_s_ff0_i32_b64(f, m, d, a);
      },
      [](uint64_t a, uint64_t, uint32_t cc) {
        uint32_t result = UINT32_MAX;
        for (int bit = 0; bit < 64; ++bit) {
          if (!((a >> bit) & 1)) {
            result = bit;
            break;
          }
        }
        return std::make_pair(result, cc);
      },
      false);
}
