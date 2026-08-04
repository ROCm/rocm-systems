// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_test_util.h
/// @brief Shared generators and skip guard for the MFMA/WMMA SIMD test
/// suites and benchmarks.

#pragma once

#include "util/simd.h"

#include <gtest/gtest.h>

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
