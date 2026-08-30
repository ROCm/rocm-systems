// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_test_util.h
/// @brief Shared generators and skip guard for the MFMA/WMMA SIMD test
/// suites and benchmarks.

#pragma once

#include "util/simd.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>

namespace mma_test {

// Wavefront sizes for the two MMA families (differ by design, so they are named
// rather than a single shared WF_SIZE).
constexpr uint32_t MFMA_WF_SIZE = 64; // CDNA MFMA is wave64.
constexpr uint32_t WMMA_WF_SIZE = 32; // gfx1250 / RDNA WMMA is wave32.

// Register-file dimensions and iteration count shared verbatim by the MFMA and
// WMMA SIMD benchmarks.
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr int BENCH_ITERATIONS = 4000;

constexpr std::array<uint32_t, 4>
make_cdna4_mfma_scale_words(uint32_t op, uint32_t abid, uint32_t scale_a, uint32_t scale_b,
                            uint32_t a_byte = 0, uint32_t b_byte = 0, uint32_t cbsz = 4,
                            uint32_t blgp = 4, uint32_t vdst = 64, uint32_t src0 = 256,
                            uint32_t src1 = 272, uint32_t src2 = 288) {
  const uint32_t op_sel = (a_byte & 1u) | ((b_byte & 1u) << 1);
  const uint32_t op_sel_hi = ((a_byte >> 1) & 1u) | (((b_byte >> 1) & 1u) << 1);
  return {0xD3AC0000u | (op_sel << 11),
          (scale_a & 0x1FFu) | ((scale_b & 0x1FFu) << 9) | (op_sel_hi << 27),
          (vdst & 0xFFu) | ((cbsz & 0x7u) << 8) | ((abid & 0xFu) << 11) | ((op & 0x7Fu) << 16) |
              (423u << 23),
          (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18) |
              ((blgp & 0x7u) << 29)};
}

// Small finite generator: values in roughly [-1, 1], deterministic per call.
struct SmallGen {
  std::mt19937 rng;
  std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
  explicit SmallGen(uint32_t seed) : rng(seed) {}
  float operator()() { return dist(rng); }
};

struct SmallI8Gen {
  std::mt19937 rng;
  std::uniform_int_distribution<int> dist{-8, 7};
  explicit SmallI8Gen(uint32_t seed) : rng(seed) {}
  int operator()() { return dist(rng); }
};

} // namespace mma_test

#define SKIP_IF_NO_SIMD()                                                                          \
  if constexpr (!util::has_stdx_simd) {                                                            \
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";                    \
  } else if (util::native<float>::size() <= 1) {                                                   \
    GTEST_SKIP() << "host native_simd width is 1 — no SIMD fast path";                             \
  }
