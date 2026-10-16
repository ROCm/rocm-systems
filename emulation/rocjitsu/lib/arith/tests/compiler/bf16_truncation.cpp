// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone GCC reproducer for swmmac16.cpp's noinline workaround.
// g++ -std=c++20 -O2 bf16_truncation.cpp && ./a.out
// GCC 15: -O2 returns nonzero (mask lost), -O1 succeeds. Making narrow
// [[gnu::noinline]] also fixes -O2. Not a CTest: it intentionally exposes the compiler bug.

#include <bit>
#include <cmath>
#include <cstdio>
#include <stdint.h>

// Truncate to BF16 precision, returning a widened FP32 value.
inline float narrow(float x) {
  return std::bit_cast<float>(std::bit_cast<uint32_t>(x) & 0xffff0000U);
}

// Accumulate four groups, truncating the accumulator after every group.
[[gnu::noinline]] float accumulate(float a, float b, float c) {
  for (int stage = 0; stage < 4; ++stage) {
    for (int k = 0; k < 4; ++k) {
      c = std::fma(a, b, c);
    }
    c = narrow(c);
  }
  return c;
}

// Print the result and fail if the final truncation was lost.
int main() {
  float value = accumulate(0.1f, 0.1f, 0.01f);
  std::printf("%a\n", double(value));
  return (std::bit_cast<uint32_t>(value) & 0xffffU) != 0;
}
