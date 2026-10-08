// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_SMFMAC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_SMFMAC_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"

namespace rocjitsu {
namespace amdgpu {

/// AVX-512 sparse F32 matrix kernel. Snapshot every operand region before
/// publishing D: tied accumulators and overlapping A/B/index registers then
/// have the same read-before-write behavior as the scalar helpers. Selected K
/// positions are resolved once per row; the FMA loop retains q,s order.
template <SmfmacLayout Layout, uint32_t M, uint32_t N, uint32_t K, typename ExtractA,
          typename ExtractB>
RJ_NOINLINE inline void smfmac_avx512_fast_body(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                                uint32_t idx_base) {
  static_assert((M == 16 && N == 16) || (M == 32 && N == 32));
  static_assert(K % 4 == 0);
  constexpr uint32_t compressed_k = K / 2;
  constexpr uint32_t a_regs = (M * compressed_k * smfmac_input_bits<ExtractA> + 2047) / 2048;
  constexpr uint32_t b_regs = (N * K * smfmac_input_bits<ExtractB> + 2047) / 2048;
  constexpr uint32_t d_regs = M * N / 64;
  static_assert(((a_regs + b_regs + 1 + d_regs) * 64 + M * compressed_k + K * N + M * N) *
                            sizeof(uint32_t) +
                        M * compressed_k <=
                    48 * 1024,
                "SMFMAC AVX-512 staging buffers exceed the 48 KiB stack budget");
  constexpr uint64_t full_lanes = ~uint64_t{0};
  RegisterAccess regs(cu);
  auto a_read = regs.read_vgpr_region(s0, a_regs, full_lanes);
  auto b_read = regs.read_vgpr_region(s1, b_regs, full_lanes);
  auto index_read = regs.read_vgpr_region(idx_base, 1, full_lanes);
  auto d_read = regs.read_vgpr_region(dst, d_regs, full_lanes);
  alignas(64) uint32_t a_words[a_regs * 64];
  alignas(64) uint32_t b_words[b_regs * 64];
  alignas(64) uint32_t index_words[64];
  alignas(64) uint32_t d_words[d_regs * 64];
  copy_matrix_region_words(a_read, a_words);
  copy_matrix_region_words(b_read, b_words);
  copy_matrix_region_words(index_read, index_words);
  copy_matrix_region_words(d_read, d_words);

  alignas(64) float a_values[M * compressed_k];
  alignas(64) float b_values[K * N];
  alignas(64) float c_values[M * N];
  uint8_t selected_k[M * compressed_k];
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      const auto out = output_loc_32(M, N, row, col, 0);
      c_values[row * N + col] = std::bit_cast<float>(d_words[out.reg * 64 + out.lane]);
    }
    for (uint32_t q = 0; q < K / 4; ++q) {
      const auto first = smfmac_sparse_loc<Layout, M, K>(row, q, 0);
      const uint32_t field = (index_words[first.lane] >> (4 * first.nibble)) & 0xFu;
      for (uint32_t s = 0; s < 2; ++s) {
        const auto loc = smfmac_sparse_loc<Layout, M, K>(row, q, s);
        const uint32_t t = row * compressed_k + 2 * q + s;
        a_values[t] = smfmac_decode_snapshot<ExtractA>(a_words, loc.element, loc.lane);
        selected_k[t] = static_cast<uint8_t>(4 * q + ((field >> (2 * s)) & 3u));
      }
    }
  }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      const auto loc = smfmac_dense_loc<Layout, N>(k, col);
      b_values[k * N + col] = smfmac_decode_snapshot<ExtractB>(b_words, loc.element, loc.lane);
    }

  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t c0 = 0; c0 < N; c0 += 16) {
      util::native<float> c_row;
      c_row.copy_from(&c_values[row * N + c0], util::stdx::vector_aligned);
      for (uint32_t t = 0; t < compressed_k; ++t) {
        util::native<float> b_row;
        b_row.copy_from(&b_values[selected_k[row * compressed_k + t] * N + c0],
                        util::stdx::vector_aligned);
        c_row =
            util::native_fma(util::native<float>(a_values[row * compressed_k + t]), b_row, c_row);
      }
      c_row.copy_to(&c_values[row * N + c0], util::stdx::vector_aligned);
    }

  // Packed host FMAs can select a different NaN payload than scalar FMA.
  // Replay only exceptional outputs through the shared matrix NaN policy.
  auto writes = regs.write_vgpr_region(dst, d_regs, full_lanes);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      float result = c_values[row * N + col];
      const auto out = output_loc_32(M, N, row, col, 0);
      if (std::isnan(result)) {
        result = std::bit_cast<float>(d_words[out.reg * 64 + out.lane]);
        for (uint32_t t = 0; t < compressed_k; ++t)
          result = matrix_fma(a_values[row * compressed_k + t],
                              b_values[selected_k[row * compressed_k + t] * N + col], result);
      }
      writes.set_linear_word(out.reg * 64 + out.lane, std::bit_cast<uint32_t>(result));
    }
}

/// Keep eligibility checks outside the stack-heavy implementation so
/// forced-scalar and non-AVX-512 executions avoid its scratch frame.
template <SmfmacLayout Layout, uint32_t M, uint32_t N, uint32_t K, typename ExtractA,
          typename ExtractB>
[[gnu::always_inline]] inline bool smfmac_try_avx512(auto &cu, uint32_t dst, uint32_t s0,
                                                     uint32_t s1, uint32_t idx_base, ExtractA,
                                                     ExtractB) {
  if (util::force_scalar() || cu.wf_size() != 64 || util::native<float>::size() != 16)
    return false;

  smfmac_avx512_fast_body<Layout, M, N, K, ExtractA, ExtractB>(cu, dst, s0, s1, idx_base);
  return true;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_SMFMAC_H_
