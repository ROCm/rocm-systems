// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Retained MFMA snapshot/observer fixtures now exercise the public integer arithmetic.
#pragma once
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_dense_matrix_exec.h"
#include "rocjitsu/isa/isa_family.h"
#include "util/simd_test_hooks.h"
#include <gtest/gtest.h>

namespace rocjitsu::test {
namespace mfma_detail {
using Entry = decltype(&rj_arith_v_mfma_i32_16x16x32_i8_w64);

// Run one legal register shape on a single-wave fixture, preserving observed lane sets.
template <int M, int K, int Batch>
void execute(amdgpu::ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
             uint32_t constant, uint32_t cbsz, uint32_t abid, uint32_t blgp, bool clamp,
             Entry entry) {
  auto *wave = cu.wf(0);
  ASSERT_NE(wave, nullptr);
  uint64_t flags = cu.arithmetic_flags() | (clamp ? RJ_ARITH_FP16_OVFL : 0);
  if (util::force_scalar()) {
    flags &= ~RJ_ARITH_CPU_MASK;
  }
  const unsigned block = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? M : 0;
  const uint64_t a_lanes = amdgpu::mfma_permuted_lane_mask(
      [&](uint32_t lane) { return amdgpu::permute_a_lane(lane, cbsz, abid, block); });
  const uint64_t b_lanes = amdgpu::mfma_permuted_lane_mask(
      [&](uint32_t lane) { return amdgpu::permute_b_lane(lane, blgp); });
  amdgpu::arith::exec_dense_fp_matrix<M * K * Batch / 256, M * M * Batch / 64, 64>(
      *wave, entry, flags, cbsz | (uint64_t(abid) << 3) | (uint64_t(blgp) << 7), d, a, b, c,
      constant == amdgpu::ACC_FROM_VGPR ? std::nullopt : std::optional<uint64_t>(constant), a_lanes,
      b_lanes);
}
} // namespace mfma_detail

// Keep existing observation/race fixtures while removing their retired arithmetic helper.
inline void execute_mfma_i8(amdgpu::ComputeUnitCore &cu, uint32_t m, uint32_t n, uint32_t k,
                            uint32_t batch, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
                            uint32_t constant = amdgpu::ACC_FROM_VGPR, uint32_t cbsz = 0,
                            uint32_t abid = 0, uint32_t blgp = 0, bool clamp = false) {
  ASSERT_EQ(m, n);
  if (m == 32 && k == 4 && batch == 2) {
    auto entry =
        cu.arch() == rocjitsu::IsaFamily::Cdna2   ? rj_arith_v_mfma_i32_32x32x4_2b_i8_w64_gfx90a
        : cu.arch() == rocjitsu::IsaFamily::Cdna1 ? rj_arith_v_mfma_i32_32x32x4_2b_i8_w64_gfx908
                                                  : rj_arith_v_mfma_i32_32x32x4_2b_i8_w64;
    return mfma_detail::execute<32, 4, 2>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp, entry);
  }
  if (m == 16 && k == 4 && batch == 4) {
    auto entry =
        cu.arch() == rocjitsu::IsaFamily::Cdna2   ? rj_arith_v_mfma_i32_16x16x4_4b_i8_w64_gfx90a
        : cu.arch() == rocjitsu::IsaFamily::Cdna1 ? rj_arith_v_mfma_i32_16x16x4_4b_i8_w64_gfx908
                                                  : rj_arith_v_mfma_i32_16x16x4_4b_i8_w64;
    return mfma_detail::execute<16, 4, 4>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp, entry);
  }
  if (m == 4 && k == 4 && batch == 16) {
    auto entry =
        cu.arch() == rocjitsu::IsaFamily::Cdna2   ? rj_arith_v_mfma_i32_4x4x4_16b_i8_w64_gfx90a
        : cu.arch() == rocjitsu::IsaFamily::Cdna1 ? rj_arith_v_mfma_i32_4x4x4_16b_i8_w64_gfx908
                                                  : rj_arith_v_mfma_i32_4x4x4_16b_i8_w64;
    return mfma_detail::execute<4, 4, 16>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp, entry);
  }
  if (m == 32 && k == 8 && batch == 1) {
    auto entry = cu.arch() == rocjitsu::IsaFamily::Cdna2
                     ? rj_arith_v_mfma_i32_32x32x8_i8_w64_gfx90a
                     : rj_arith_v_mfma_i32_32x32x8_i8_w64_gfx908;
    return mfma_detail::execute<32, 8, 1>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp, entry);
  }
  if (m == 16 && k == 16 && batch == 1) {
    auto entry = cu.arch() == rocjitsu::IsaFamily::Cdna2
                     ? rj_arith_v_mfma_i32_16x16x16_i8_w64_gfx90a
                     : rj_arith_v_mfma_i32_16x16x16_i8_w64_gfx908;
    return mfma_detail::execute<16, 16, 1>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp,
                                           entry);
  }
  if (m == 32 && k == 16 && batch == 1) {
    auto entry = rj_arith_v_mfma_i32_32x32x16_i8_w64;
    return mfma_detail::execute<32, 16, 1>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp,
                                           entry);
  }
  if (m == 16 && k == 32 && batch == 1) {
    auto entry = rj_arith_v_mfma_i32_16x16x32_i8_w64;
    return mfma_detail::execute<16, 32, 1>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp,
                                           entry);
  }
  if (m == 16 && k == 64 && batch == 1) {
    auto entry = rj_arith_v_mfma_i32_16x16x64_i8_w64;
    return mfma_detail::execute<16, 64, 1>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp,
                                           entry);
  }
  if (m == 32 && k == 32 && batch == 1) {
    auto entry = rj_arith_v_mfma_i32_32x32x32_i8_w64;
    return mfma_detail::execute<32, 32, 1>(cu, d, a, b, c, constant, cbsz, abid, blgp, clamp,
                                           entry);
  }
  FAIL() << "Unsupported integer MFMA test shape";
}

// The former specialized fixtures use the same arithmetic without a second numerical body.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t Batch = 1>
void execute_mfma_i8_shape(amdgpu::ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b,
                           uint32_t c, uint32_t constant) {
  execute_mfma_i8(cu, M, N, K, Batch, d, a, b, c, constant);
}
} // namespace rocjitsu::test
