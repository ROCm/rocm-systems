// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Public FP64 MFMA dispatch for retained observation, layout and aliasing fixtures.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_dense_matrix_exec.h"
#include "rocjitsu/isa/isa_family.h"
#include "util/simd_test_hooks.h"
#include <bit>
#include <optional>

#include <gtest/gtest.h>

namespace rocjitsu::test {
// Execute either real MFMA shape, widening this fixture's FP32 inline C value.
inline void execute_mfma_f64_shape(amdgpu::ComputeUnitCore &cu, uint32_t m, uint32_t n, uint32_t k,
                                   uint32_t batch, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
                                   uint32_t constant, uint32_t neg = 0, uint32_t cbsz = 0) {
  ASSERT_EQ(m, n);
  ASSERT_EQ(k, 4U);
  auto *wave = cu.wf(0);
  ASSERT_NE(wave, nullptr);
  const bool cdna4 = cu.arch() == rocjitsu::IsaFamily::Cdna4;
  uint64_t flags = cu.arithmetic_flags();
  if (util::force_scalar()) {
    flags &= ~RJ_ARITH_CPU_MASK;
  }
  const uint64_t controls = cdna4 ? (uint64_t(neg) | (uint64_t(cbsz) << 3)) : uint64_t(neg);
  // Pass nullopt or the value directly, avoiding GCC 13's optional-copy false positive.
  const auto execute = [&](auto inline_c) {
    if (m == 16) {
      ASSERT_EQ(batch, 1U);
      amdgpu::arith::exec_dense_fp_matrix<2, 8, 64, 8, false, true>(
          *wave,
          cdna4 ? rj_arith_v_mfma_f64_16x16x4_f64_w64_gfx950 : rj_arith_v_mfma_f64_16x16x4_f64_w64,
          flags, controls, d, a, b, c, inline_c);
    } else {
      ASSERT_EQ(m, 4U);
      ASSERT_EQ(batch, 4U);
      amdgpu::arith::exec_dense_fp_matrix<2, 2, 64, 2, false, true>(
          *wave,
          cdna4 ? rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950
                : rj_arith_v_mfma_f64_4x4x4_4b_f64_w64,
          flags, controls, d, a, b, c, inline_c);
    }
  };
  if (constant == amdgpu::ACC_FROM_VGPR) {
    execute(std::nullopt);
  } else {
    execute(std::bit_cast<uint64_t>(double(std::bit_cast<float>(constant))));
  }
}
} // namespace rocjitsu::test
