// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_SWMMAC_K64_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_SWMMAC_K64_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"

namespace rocjitsu::amdgpu {

RJ_NOINLINE inline void exec_swmmac_16x16x64_16bit_fast_body(
    auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base, uint32_t index_base,
    SwmmacK64Input input_format, SwmmacK64Accumulator accumulator_format,
    SwmmacK64Result result_format, uint32_t const_acc, bool fp16_ovfl) {
  constexpr uint32_t SPEC_M = 16;
  constexpr uint32_t SPEC_N = 16;
  constexpr uint32_t SPEC_K = 64;
  constexpr uint32_t COMPRESSED_K = SPEC_K / 2;
  constexpr uint32_t WF = WMMA_WAVE32;
  constexpr uint32_t IN_BITS = 16;
  constexpr uint32_t INDEX_ENTRIES = 16;

  const uint32_t acc_bits =
      accumulator_format == SwmmacK64Accumulator::F32 ? uint32_t{32} : uint32_t{16};
  auto reads = read_mixed_matrix_fast_path_regions(
      cu, s0, s1, acc_base, static_cast<uint64_t>(SPEC_M) * COMPRESSED_K, IN_BITS,
      static_cast<uint64_t>(SPEC_N) * SPEC_K, IN_BITS, static_cast<uint64_t>(SPEC_M) * SPEC_N,
      acc_bits, const_acc, WF);
  RegisterAccess regs(cu);
  auto index_read = regs.read_vgpr_region(index_base, /*reg_count=*/1, mfma_full_lane_mask(WF));

  // Snapshot every source before acquiring the destination write view.  The
  // instruction permits D to alias A, B, C, or the sparse index register.
  alignas(64) float a_physical[SPEC_M * COMPRESSED_K];
  alignas(64) float b_physical[SPEC_N * SPEC_K];
  alignas(64) uint32_t c_words[SPEC_M * SPEC_N] = {};
  alignas(64) uint32_t index_words[WF] = {};
  if (input_format == SwmmacK64Input::F16) {
    convert_f16_matrix_region(reads.a, a_physical, SPEC_M * COMPRESSED_K);
    convert_f16_matrix_region(reads.b, b_physical, SPEC_N * SPEC_K);
  } else {
    convert_bf16_matrix_region(reads.a, a_physical, SPEC_M * COMPRESSED_K);
    convert_bf16_matrix_region(reads.b, b_physical, SPEC_N * SPEC_K);
  }
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, c_words);
  index_read.copy_to({index_words, WF});

  alignas(64) float a_matrix[SPEC_M * COMPRESSED_K];
  alignas(64) float b_matrix[SPEC_K * SPEC_N];
  alignas(64) float c_matrix[SPEC_M * SPEC_N];
  alignas(64) uint32_t dense_k[SPEC_M * COMPRESSED_K];

  for (uint32_t row = 0; row < SPEC_M; ++row) {
    for (uint32_t ck = 0; ck < COMPRESSED_K; ++ck) {
      const auto a_loc = swmmac_a_input_loc(WF, SPEC_M, SPEC_K, row, ck, IN_BITS);
      a_matrix[row * COMPRESSED_K + ck] =
          a_physical[(a_loc.vgpr_offset * WF + a_loc.lane) * 2 + a_loc.sub_element];
      const auto index_loc = swmmac_index_loc(WF, SPEC_M, SPEC_K, IN_BITS, row, ck, INDEX_ENTRIES);
      dense_k[row * COMPRESSED_K + ck] =
          swmmac_dense_k(index_words[index_loc.lane], ck, index_loc.local_compressed_k);
    }
  }
  for (uint32_t k = 0; k < SPEC_K; ++k) {
    for (uint32_t col = 0; col < SPEC_N; ++col) {
      const auto b_loc = swmmac_b_input_loc(WF, SPEC_N, SPEC_K, col, k, IN_BITS);
      b_matrix[k * SPEC_N + col] =
          b_physical[(b_loc.vgpr_offset * WF + b_loc.lane) * 2 + b_loc.sub_element];
    }
  }

  auto initial_acc = [&](uint32_t row, uint32_t col) {
    if (const_acc != ACC_FROM_VGPR) {
      return std::bit_cast<float>(const_acc);
    }
    if (accumulator_format == SwmmacK64Accumulator::F32) {
      const auto out = wmma_output_loc_32(SPEC_M, SPEC_N, row, col);
      return std::bit_cast<float>(c_words[out.reg * WF + out.lane]);
    }
    const auto out = wmma_output_loc_16(SPEC_M, SPEC_N, row, col);
    const uint32_t raw = c_words[out.reg * WF + out.lane];
    const uint16_t packed = static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFFu);
    return accumulator_format == SwmmacK64Accumulator::F16 ? util::f16_to_f32(packed)
                                                           : util::bf16_to_f32(packed);
  };
  for (uint32_t row = 0; row < SPEC_M; ++row)
    for (uint32_t col = 0; col < SPEC_N; ++col)
      c_matrix[row * SPEC_N + col] = initial_acc(row, col);

  // Preserve compressed-K accumulation order while processing all 16 output
  // columns in parallel.  On an AVX-512 build native<float> is one ZMM.
  for (uint32_t row = 0; row < SPEC_M; ++row) {
    util::native<float> c_row;
    c_row.copy_from(&c_matrix[row * SPEC_N], util::stdx::vector_aligned);
    for (uint32_t ck = 0; ck < COMPRESSED_K; ++ck) {
      util::native<float> a_broadcast(a_matrix[row * COMPRESSED_K + ck]);
      util::native<float> b_row;
      b_row.copy_from(&b_matrix[dense_k[row * COMPRESSED_K + ck] * SPEC_N],
                      util::stdx::vector_aligned);
      c_row = util::native_fma(a_broadcast, b_row, c_row);
    }
    const uint64_t nan_lanes = util::simd_mask_to_bits(util::stdx::isnan(c_row));
    c_row.copy_to(&c_matrix[row * SPEC_N], util::stdx::vector_aligned);
    // Packed and scalar host FMAs can choose different NaN payload sources.
    // Recompute only exceptional lanes with the deterministic scalar helper.
    uint64_t pending_nan_lanes = nan_lanes;
    while (pending_nan_lanes != 0) {
      const uint32_t col = static_cast<uint32_t>(std::countr_zero(pending_nan_lanes));
      pending_nan_lanes &= pending_nan_lanes - 1;
      float acc = initial_acc(row, col);
      for (uint32_t ck = 0; ck < COMPRESSED_K; ++ck) {
        const uint32_t k = dense_k[row * COMPRESSED_K + ck];
        acc = matrix_fma(a_matrix[row * COMPRESSED_K + ck], b_matrix[k * SPEC_N + col], acc);
      }
      c_matrix[row * SPEC_N + col] = acc;
    }
  }

  const uint32_t result_bits = result_format == SwmmacK64Result::F32 ? uint32_t{32} : uint32_t{16};
  auto writes = write_wmma_output_region(cu, dst, SPEC_M, SPEC_N, result_bits, WF);
  if (result_format == SwmmacK64Result::F32) {
    for (uint32_t row = 0; row < SPEC_M; ++row)
      for (uint32_t col = 0; col < SPEC_N; ++col) {
        const auto out = wmma_output_loc_32(SPEC_M, SPEC_N, row, col);
        writes.set_linear_word(out.reg * WF + out.lane,
                               std::bit_cast<uint32_t>(c_matrix[row * SPEC_N + col]));
      }
  } else {
    constexpr uint32_t PACKED_REGS = (SPEC_M * SPEC_N / WF) / 2;
    alignas(64) uint32_t packed_words[PACKED_REGS * WF] = {};
    for (uint32_t row = 0; row < SPEC_M; ++row)
      for (uint32_t col = 0; col < SPEC_N; ++col) {
        const auto out = wmma_output_loc_16(SPEC_M, SPEC_N, row, col);
        // CDNA5 SWMMAC packed results use fixed round-to-nearest-even.
        const uint16_t value = result_format == SwmmacK64Result::F16
                                   ? wmma_round_f16(c_matrix[row * SPEC_N + col], fp16_ovfl)
                                   : wmma_round_bf16_rne(c_matrix[row * SPEC_N + col], fp16_ovfl);
        packed_words[out.reg * WF + out.lane] |= static_cast<uint32_t>(value)
                                                 << (out.sub_element * 16);
      }
    for (uint32_t reg = 0; reg < PACKED_REGS; ++reg)
      for (uint32_t lane = 0; lane < WF; ++lane)
        writes.set_linear_word(reg * WF + lane, packed_words[reg * WF + lane]);
  }
}

[[gnu::always_inline]] inline bool
try_exec_swmmac_16x16x64_16bit(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits,
                               uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                               uint32_t index_base, uint32_t index_entries, uint32_t index_key,
                               SwmmacK64Input input_format, SwmmacK64Accumulator accumulator_format,
                               SwmmacK64Result result_format, uint32_t const_acc = ACC_FROM_VGPR,
                               uint32_t wave_size = WMMA_WAVE32, bool fp16_ovfl = false) {
  if (util::force_scalar() || util::native<float>::size() != 16 || M != 16 || N != 16 || K != 64 ||
      in_bits != 16 || index_entries != 16 || index_key != 0 || wave_size != WMMA_WAVE32 ||
      cu.wf_size() != WMMA_WAVE32)
    return false;

  exec_swmmac_16x16x64_16bit_fast_body(cu, dst, s0, s1, acc_base, index_base, input_format,
                                       accumulator_format, result_format, const_acc, fp16_ovfl);
  return true;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_SWMMAC_K64_H_
