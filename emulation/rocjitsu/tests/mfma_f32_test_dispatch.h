// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Retained FP32 MFMA observer/overlap fixtures exercise rj-arith after executor removal.
#pragma once
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_dense_matrix_exec.h"
#include "rocjitsu/isa/isa_family.h"
#include "util/simd_test_hooks.h"
#include <gtest/gtest.h>

namespace rocjitsu::test {
// Execute a real FP32 MFMA shape on a single-wave fixture, retaining lane observations.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t Batch>
void execute_mfma_f32_shape(amdgpu::ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b,
                            uint32_t c, uint32_t constant, uint32_t cbsz, uint32_t abid,
                            uint32_t blgp) {
  static_assert(M == N);
  auto *wave = cu.wf(0);
  ASSERT_NE(wave, nullptr);
  using Entry = decltype(&rj_arith_v_mfma_f32_16x16x4_f32_w64);
  Entry entry = nullptr;
  if constexpr (M == 32 && K == 1 && Batch == 2) {
    entry = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx90a
            : cu.arch() == rocjitsu::IsaFamily::Cdna1
                ? rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx908
                : rj_arith_v_mfma_f32_32x32x1_2b_f32_w64;
  } else if constexpr (M == 16 && K == 1 && Batch == 4) {
    entry = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx90a
            : cu.arch() == rocjitsu::IsaFamily::Cdna1
                ? rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx908
                : rj_arith_v_mfma_f32_16x16x1_4b_f32_w64;
  } else if constexpr (M == 4 && K == 1 && Batch == 16) {
    entry = cu.arch() == rocjitsu::IsaFamily::Cdna2   ? rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx90a
            : cu.arch() == rocjitsu::IsaFamily::Cdna1 ? rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx908
                                                      : rj_arith_v_mfma_f32_4x4x1_16b_f32_w64;
  } else if constexpr (M == 32 && K == 2 && Batch == 1) {
    entry = cu.arch() == rocjitsu::IsaFamily::Cdna2   ? rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx90a
            : cu.arch() == rocjitsu::IsaFamily::Cdna1 ? rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx908
                                                      : rj_arith_v_mfma_f32_32x32x2_f32_w64;
  } else if constexpr (M == 16 && K == 4 && Batch == 1) {
    entry = cu.arch() == rocjitsu::IsaFamily::Cdna2   ? rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx90a
            : cu.arch() == rocjitsu::IsaFamily::Cdna1 ? rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx908
                                                      : rj_arith_v_mfma_f32_16x16x4_f32_w64;
  }
  ASSERT_NE(entry, nullptr);
  uint64_t flags = cu.arithmetic_flags();
  if (util::force_scalar()) {
    flags &= ~RJ_ARITH_CPU_MASK;
  }
  if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
    if (!(wave->fp_denorm_mode_f32() & 1)) {
      flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
    }
    if (!(wave->fp_denorm_mode_f32() & 2)) {
      flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
    }
  }
  const unsigned block = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? M : 0;
  uint64_t a_lanes = amdgpu::mfma_permuted_lane_mask(
      [&](uint32_t lane) { return amdgpu::permute_a_lane(lane, cbsz, abid, block); });
  uint64_t b_lanes = amdgpu::mfma_permuted_lane_mask(
      [&](uint32_t lane) { return amdgpu::permute_b_lane(lane, blgp); });
  amdgpu::arith::exec_dense_fp_matrix<M * K * Batch / 64, M * M * Batch / 64, 64>(
      *wave, entry, flags, cbsz | (uint64_t(abid) << 3) | (uint64_t(blgp) << 7), d, a, b, c,
      constant == amdgpu::ACC_FROM_VGPR ? std::nullopt : std::optional<uint64_t>(constant), a_lanes,
      b_lanes);
}
} // namespace rocjitsu::test
