// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Public FP8 and mixed-format MFMA dispatch for retained register-observation fixtures.
#pragma once
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_dense_matrix_exec.h"
#include "util/simd_test_hooks.h"
#include <gtest/gtest.h>

namespace rocjitsu::test {
// Execute one fixed FP8/BF8 MFMA shape, retaining explicit format interpretation.
template <int M, int N, int K, bool AFp8, bool BFp8, bool Fnuz = false>
void execute_mfma_f8_shape(amdgpu::ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b,
                           uint32_t c, uint32_t constant, uint32_t cbsz, uint32_t abid,
                           uint32_t blgp) {
  static_assert(M == N && ((M == 16 && K == 32) || (M == 32 && K == 16)));
  auto *wave = cu.wf(0);
  ASSERT_NE(wave, nullptr);
  using Entry = decltype(&rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64);
  Entry entry = nullptr;
  if constexpr (AFp8 == true && BFp8 == true) {
    if constexpr (M == 16) {
      entry = Fnuz ? rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64
                   : rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950;
    } else if constexpr (M == 32) {
      entry = Fnuz ? rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64
                   : rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64_gfx950;
    }
  } else if constexpr (AFp8 == true && BFp8 == false) {
    if constexpr (M == 16) {
      entry = Fnuz ? rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64
                   : rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64_gfx950;
    } else if constexpr (M == 32) {
      entry = Fnuz ? rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64
                   : rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64_gfx950;
    }
  } else if constexpr (AFp8 == false && BFp8 == true) {
    if constexpr (M == 16) {
      entry = Fnuz ? rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64
                   : rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64_gfx950;
    } else if constexpr (M == 32) {
      entry = Fnuz ? rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64
                   : rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64_gfx950;
    }
  } else if constexpr (AFp8 == false && BFp8 == false) {
    if constexpr (M == 16) {
      entry = Fnuz ? rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64
                   : rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950;
    } else if constexpr (M == 32) {
      entry = Fnuz ? rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64
                   : rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64_gfx950;
    }
  }
  uint64_t flags = cu.arithmetic_flags();
  if (util::force_scalar()) {
    flags &= ~RJ_ARITH_CPU_MASK;
  }
  const uint64_t a_lanes = amdgpu::mfma_permuted_lane_mask(
      [&](uint32_t lane) { return amdgpu::permute_a_lane(lane, cbsz, abid); });
  const uint64_t b_lanes = amdgpu::mfma_permuted_lane_mask(
      [&](uint32_t lane) { return amdgpu::permute_b_lane(lane, blgp); });
  amdgpu::arith::exec_dense_fp_matrix<M * K / 256, M * M / 64, 64>(
      *wave, entry, flags, cbsz | (uint64_t(abid) << 3) | (uint64_t(blgp) << 7), d, a, b, c,
      constant == amdgpu::ACC_FROM_VGPR ? std::nullopt : std::optional<uint64_t>(constant), a_lanes,
      b_lanes);
}

// Forward a mixed-format instruction with the actual source widths to the VM bridge.
template <int M>
void execute_mfma_mixed_shape(amdgpu::ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b,
                              uint32_t c, uint32_t constant, uint32_t a_format, uint32_t b_format) {
  static_assert(M == 16 || M == 32);
  ASSERT_LT(a_format, 5U);
  ASSERT_LT(b_format, 5U);
  constexpr unsigned bits[] = {8, 8, 6, 6, 4};
  auto *wave = cu.wf(0);
  ASSERT_NE(wave, nullptr);
  auto entry =
      M == 16 ? rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64 : rj_arith_v_mfma_f32_32x32x64_f8f6f4_w64;
  uint64_t flags = cu.arithmetic_flags();
  if (util::force_scalar()) {
    flags &= ~RJ_ARITH_CPU_MASK;
  }
  amdgpu::arith::exec_dense_fp_matrix<8, M * M / 64, 64>(
      *wave, entry, flags, a_format | (uint64_t(b_format) << RJ_ARITH_MATRIX_FORMAT_B_SHIFT), d, a,
      b, c, constant == amdgpu::ACC_FROM_VGPR ? std::nullopt : std::optional<uint64_t>(constant),
      ~uint64_t{0}, ~uint64_t{0}, bits[a_format], bits[b_format]);
}
} // namespace rocjitsu::test
