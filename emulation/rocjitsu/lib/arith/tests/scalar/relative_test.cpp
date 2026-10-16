// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for M0-relative scalar-register addressing and bounds.

#include "scalar/integer_helpers.h"

#include <array>

namespace {

// SGPR/VCC and trap temporaries are separate index regions. Out-of-range
// reads redirect to SGPR0; writes are discarded. Addition never wraps.
inline bool relative_range(uint32_t base, uint32_t offset, int count, int words) {
  uint64_t end = uint64_t(base) + offset + words;
  return end <= uint64_t(count) && end <= uint64_t(base < 108 ? 108 : base < 124 ? 124 : 0);
}

// Check relative scalar-register moves and invalid bounds for the supplied entry point.
template <int Words, bool Read, class Call> void check_relative(Call call) {
  std::array<uint32_t, 128> input;
  std::minstd_rand random;
  for (auto &v : input) {
    v = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int count : {0, 1, 107, 108, 124, 128}) {
        for (uint32_t base : {0U, 106U, 108U, 122U, 124U, UINT32_MAX}) {
          for (uint32_t offset : {0U, 1U, 2U, 16U, 256U, UINT32_MAX}) {
            auto actual = input, expected = input;
            uint64_t result = 0xdeadbeefU;
            uint64_t value = uint64_t(input[0]) | (uint64_t(input[1]) << 32);
            bool valid = relative_range(base, offset, count, Words);
            if constexpr (Read) {
              int index = valid ? int(base + offset) : 0;
              value = count ? input[index] : 0;
              if (Words == 2 && count > 1) {
                value |= uint64_t(input[index + 1]) << 32;
              }
            } else if (valid) {
              expected[base + offset] = uint32_t(value);
              if (Words == 2) {
                expected[base + offset + 1] = uint32_t(value >> 32);
              }
            }
            ASSERT_EQ(call(cpu | semantics, 0, actual.data(), &result, count, base, offset),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(actual, expected);
            if constexpr (Read) {
              EXPECT_EQ(result, value);
            }
          }
        }
      }
    }
  }
  uint64_t result = 123;
  auto actual = input;
  rocjitsu::arith::test::expect_invalid_flags(
      [&](auto f, auto m) {
        int error = call(f, m, actual.data(), &result, 128, 0, 0);
        EXPECT_EQ(actual, input);
        EXPECT_EQ(result, 123U);
        return error;
      },
      0);
}
} // namespace

// Check relative scalar-register moves and invalid bounds.
TEST(s_movreld_b32, Arithmetic) {
  check_relative<1, false>(
      [](auto f, auto m, auto data, auto result, auto count, auto base, auto offset) {
        (void)result;
        uint64_t value = uint64_t(data[0]) | (uint64_t(data[1]) << 32);
        return rj_arith_s_movreld_b32(f, m, data, uint32_t(value), count, base, offset);
      });
}

// Check relative scalar-register moves and invalid bounds.
TEST(s_movrels_b32, Arithmetic) {
  check_relative<1, true>(
      [](auto f, auto m, auto data, auto result, auto count, auto base, auto offset) {
        uint32_t output = uint32_t(*result);
        int error = rj_arith_s_movrels_b32(f, m, &output, data, count, base, offset);
        if (!error) {
          *result = output;
        }
        return error;
      });
}

// Check relative scalar-register moves and invalid bounds.
TEST(s_movreld_b64, Arithmetic) {
  check_relative<2, false>(
      [](auto f, auto m, auto data, auto result, auto count, auto base, auto offset) {
        (void)result;
        uint64_t value = uint64_t(data[0]) | (uint64_t(data[1]) << 32);
        return rj_arith_s_movreld_b64(f, m, data, uint64_t(value), count, base, offset);
      });
}

// Check relative scalar-register moves and invalid bounds.
TEST(s_movrels_b64, Arithmetic) {
  check_relative<2, true>(
      [](auto f, auto m, auto data, auto result, auto count, auto base, auto offset) {
        uint64_t output = uint64_t(*result);
        int error = rj_arith_s_movrels_b64(f, m, &output, data, count, base, offset);
        if (!error) {
          *result = output;
        }
        return error;
      });
}

// Check M0-relative scalar-register addressing and bounds.
TEST(s_movrelsd_2_b32, Arithmetic) {
  std::array<uint32_t, 128> input;
  std::minstd_rand random;
  for (auto &v : input) {
    v = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint32_t offset : {0U, 1U, 2U, 16U, 256U, 1023U}) {
        for (uint32_t base : {0U, 106U, 108U, 122U, 124U, UINT32_MAX}) {
          auto actual = input, expected = input;
          uint32_t destination = offset / 2;
          int source = relative_range(base, offset, 128, 1) ? int(base + offset) : 0;
          if (relative_range(base, destination, 128, 1)) {
            expected[base + destination] = input[source];
          }
          ASSERT_EQ(rj_arith_s_movrelsd_2_b32(cpu | semantics, 0, actual.data(), actual.data(), 128,
                                              128, base, base,
                                              0xfc00fc00U | offset | (destination << 16)),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(actual, expected);
        }
      }
    }
  }
  auto actual = input;
  rocjitsu::arith::test::expect_invalid_flags(
      [&](auto f, auto m) {
        int error =
            rj_arith_s_movrelsd_2_b32(f, m, actual.data(), actual.data(), 128, 128, 0, 0, 0);
        EXPECT_EQ(actual, input);
        return error;
      },
      0);
}
