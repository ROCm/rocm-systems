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
