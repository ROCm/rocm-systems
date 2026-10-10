// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

struct RdnaWmmaFloatMatrixCase {
  bool gfx12;
  uint32_t width;
  std::array<uint32_t, 2> code;
  uint32_t mode;
  uint32_t seed;
  bool finite;
  uint32_t expected_words;
  std::array<uint32_t, 256> expected;
  // Bound finite BF16 factor exponents. Packed shapes also select BF16 C/D
  // boundary words repeated across both halves to keep accumulation observable.
  bool bounded_bf16 = false;
};

struct RdnaWmmaFloatOperandCase {
  static constexpr uint32_t kUniformFactors = UINT32_MAX;
  bool gfx12;
  uint32_t width;
  std::array<uint32_t, 2> code;
  uint32_t mode;
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  uint32_t index;
  // Complete captured destination word, including any preserved packed half.
  uint32_t expected;
  // First VGPR to replace; kUniformFactors keeps the uniform A/B inputs.
  uint32_t factor_base = kUniformFactors;
  // Each word repeats across sixteen consecutive lanes, in register-major order:
  // 128 lane words cover four VGPRs at wave32 or two at wave64.
  std::array<uint32_t, 8> factor_groups{};
};
