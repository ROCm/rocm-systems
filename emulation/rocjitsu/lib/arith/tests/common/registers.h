// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guarded VGPR storage with overlapping operand views and preservation checks.

#pragma once

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Padded VGPR storage with guards before and after each register's lane data.
// A single register bank lets operand pointer ranges overlap, so tests can
// exercise input/output aliasing by choosing offsets into v rather than using
// separate allocations for each operand.
// Reverse physical register order and offset the data by one word to exercise
// APIs accepting independent, potentially unaligned VGPR pointers.
// Copies own their storage and pointers; no pointers refer back to the source.
template <int RegisterCount, int LaneCount> class Registers {
public:
  static_assert(RegisterCount > 0 && LaneCount > 0);

  uint32_t words[RegisterCount][LaneCount + 2];
  uint32_t *v[RegisterCount];

  // Initialize guarded storage and make each VGPR pointer refer to its own lane data.
  Registers() {
    for (int reg = 0; reg < RegisterCount; ++reg) {
      std::fill_n(words[reg], LaneCount + 2, 0xdeadbeefU);
      // Reverse and offset the pointers to catch accidental assumptions of contiguous or aligned
      // VGPR storage.
      v[reg] = words[RegisterCount - 1 - reg] + 1;
    }
  }

  // Copy register contents while keeping pointers bound to this object's storage.
  Registers(const Registers &other) : Registers() { *this = other; }

  // Copy all data and guards, preserving this object's register pointers.
  Registers &operator=(const Registers &other) {
    if (this != &other) {
      std::memcpy(words, other.words, sizeof(words));
    }
    return *this;
  }

  // Check all guards and all registers outside the writable range. A zero
  // register count checks that the entire storage is unchanged.
  void expect_preserved(const Registers &before, int first_written, int written_count) const {
    ASSERT_GE(first_written, 0);
    ASSERT_GE(written_count, 0);
    ASSERT_LE(first_written, RegisterCount);
    ASSERT_LE(written_count, RegisterCount - first_written);
    for (int reg = 0; reg < RegisterCount; ++reg) {
      for (int word = 0; word < LaneCount + 2; ++word) {
        if (reg < first_written || reg >= first_written + written_count || word == 0 ||
            word == LaneCount + 1) {
          EXPECT_EQ(words[RegisterCount - 1 - reg][word],
                    before.words[RegisterCount - 1 - reg][word])
              << "register=" << reg << ", word=" << word;
        }
      }
    }
  }
};

} // namespace rocjitsu::arith::test
