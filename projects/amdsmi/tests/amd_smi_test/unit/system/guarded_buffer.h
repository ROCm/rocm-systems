// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// GuardedBuffer<N> — a fixed-size character buffer flanked by 8-byte canaries.
// An off-by-one write past either end of a stack buffer usually lands in
// padding and goes unnoticed; here it corrupts a recognizable constant that
// CanariesIntact() checks for, so the bounds tests fail on it even in builds
// without a sanitizer.
//
// The canaries are unsigned char arrays rather than uint64_t so that nothing in
// the struct has an alignment above 1. A 64-bit canary would pad buf[N] out to
// a multiple of 8, and an overflow landing in that padding would be invisible.

#ifndef AMDSMI_TESTS_UNIT_SYSTEM_GUARDED_BUFFER_H_
#define AMDSMI_TESTS_UNIT_SYSTEM_GUARDED_BUFFER_H_

#include <array>
#include <cstddef>

namespace amdsmi_test {

template <size_t N>
struct GuardedBuffer {
  static constexpr std::array<unsigned char, 8> kPre = {0xDE, 0xAD, 0xBE, 0xEF,
                                                        0xDE, 0xAD, 0xBE, 0xEF};
  static constexpr std::array<unsigned char, 8> kPost = {0xCA, 0xFE, 0xBA, 0xBE,
                                                         0xCA, 0xFE, 0xBA, 0xBE};

  std::array<unsigned char, 8> pre_canary = kPre;
  char buf[N] = {0};
  std::array<unsigned char, 8> post_canary = kPost;

  bool CanariesIntact() const {
    static_assert(offsetof(GuardedBuffer, post_canary) == offsetof(GuardedBuffer, buf) + N,
                  "padding after buf would hide an overflow from post_canary");
    return pre_canary == kPre && post_canary == kPost;
  }
};

}  // namespace amdsmi_test

#endif  // AMDSMI_TESTS_UNIT_SYSTEM_GUARDED_BUFFER_H_
