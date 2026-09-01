// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// GuardedBuffer<N> — a fixed-size character buffer flanked by 64-bit canaries.
// An off-by-one write past either end of a stack buffer usually lands in
// padding and goes unnoticed; here it corrupts a recognizable constant that
// CanariesIntact() checks for, so the bounds tests fail on it even in builds
// without a sanitizer.

#ifndef AMDSMI_TESTS_UNIT_SYSTEM_GUARDED_BUFFER_H_
#define AMDSMI_TESTS_UNIT_SYSTEM_GUARDED_BUFFER_H_

#include <cstddef>
#include <cstdint>

namespace amdsmi_test {

template <size_t N>
struct GuardedBuffer {
  uint64_t pre_canary = 0xDEADBEEFDEADBEEFULL;
  char buf[N] = {0};
  uint64_t post_canary = 0xCAFEBABECAFEBABEULL;

  bool CanariesIntact() const {
    return pre_canary == 0xDEADBEEFDEADBEEFULL && post_canary == 0xCAFEBABECAFEBABEULL;
  }
};

}  // namespace amdsmi_test

#endif  // AMDSMI_TESTS_UNIT_SYSTEM_GUARDED_BUFFER_H_
