// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar integer fixtures, reference bit operations, and SCC checks.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/integer.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <utility>

namespace rocjitsu::arith::test {

// Inputs combine high engine bits with boundary values. No implementation or
// host signed-overflow behavior is used to compute the ISA integer results.
inline uint64_t scalar_input(std::minstd_rand &random, int sample) {
  static const uint64_t edges[] = {0,
                                   1,
                                   0xffffffffffffffffULL,
                                   0x8000000080000000ULL,
                                   0x7fffffff7fffffffULL,
                                   0xffffffff00000000ULL,
                                   0xaaaaaaaa55555555ULL,
                                   0x0000000100000000ULL};
  return sample < 8 ? edges[sample] : (uint64_t(random_word(random)) << 32) | random_word(random);
}

// Interpret the low width bits as a signed value without an out-of-range signed cast.
inline int64_t signed_word(uint64_t a, int width) {
  if (width == 64) {
    return a <= 0x7fffffffffffffffULL ? int64_t(a) : -1 - int64_t(~a);
  }
  return extend_bits(uint32_t(a) & ((1ULL << width) - 1), width, true);
}

// Reverse the low width bits.
inline uint64_t reverse_word(uint64_t a, int width) {
  uint64_t out = 0;
  for (int bit = 0; bit < width; ++bit) {
    out |= ((a >> bit) & 1) << (width - 1 - bit);
  }
  return out;
}

// Count set bits in the low width bits.
inline int count_word(uint64_t a, int width) {
  int count = 0;
  for (int bit = 0; bit < width; ++bit) {
    count += int((a >> bit) & 1);
  }
  return count;
}

// Extract the control-selected field and optionally sign-extend it.
inline uint64_t field_word(uint64_t a, uint32_t control, int width, bool sign) {
  int offset = control % width, count = (control >> 16) & 127;
  if (!count) {
    return 0;
  }
  uint64_t out = 0;
  for (int bit = 0; bit < width; ++bit) {
    int source = offset + bit;
    bool set = source < width ? ((a >> source) & 1) : sign && ((a >> (width - 1)) & 1);
    if (bit >= count) {
      set = sign && ((out >> (count - 1)) & 1);
    }
    out |= uint64_t(set) << bit;
  }
  return out;
}

// One narrow family helper covers scalar value/SCC outputs, by-value input
// aliasing, guards and error preservation. SCC is a required instruction result,
// independent of the optional FP-status semantics used by floating operations.
template <class Word, class Call, class Reference>
void check_scalar_integer(Call call, Reference reference, bool writes_scc = false) {
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      std::minstd_rand random;
      for (int sample = 0; sample < 128; ++sample) {
        uint64_t a = scalar_input(random, sample), b = scalar_input(random, (sample * 3 + 1) % 128);
        uint32_t cc = 0xa5000000U | uint32_t(sample & 1);
        Word storage[] = {Word(123), Word(a), Word(456)};
        auto expected = reference(a, b, cc);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << sample);
        ASSERT_EQ(call(cpu | semantics, 0, storage + 1, a, b, cc, &cc), RJ_ARITH_SUCCESS);
        EXPECT_EQ(storage[1], Word(expected.first));
        EXPECT_EQ(cc, expected.second);
        EXPECT_EQ(storage[0], Word(123));
        EXPECT_EQ(storage[2], Word(456));
        if constexpr (sizeof(Word) == 4) {
          if (writes_scc) {
            storage[1] = Word(a);
            uint32_t initial = 0xa5000000U | uint32_t(sample & 1);
            ASSERT_EQ(call(cpu | semantics, 0, storage + 1, a, b, initial, storage + 1),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(storage[1], expected.second);
          }
        }
      }
    }
  }
  Word d = 123;
  uint32_t cc = 456;
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        int error = call(flags, mode, &d, 0, 0, 0, &cc);
        EXPECT_EQ(d, Word(123));
        EXPECT_EQ(cc, 456U);
        return error;
      },
      0);
}

} // namespace rocjitsu::arith::test
