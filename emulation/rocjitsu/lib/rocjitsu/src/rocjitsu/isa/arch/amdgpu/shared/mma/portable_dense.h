// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_DENSE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_DENSE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/arguments.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/portable_matmul.h"

namespace rocjitsu::amdgpu::mma_backend {

inline bool try_exec_f32_mixed_simd(auto &cu, MatrixShape shape, uint32_t B, uint32_t s2,
                                    uint32_t const_acc, uint32_t wf, auto &stage_operands,
                                    auto &results) {
  const auto [M, N, K] = shape;

  // Pad the column (N) leading dimension up to a SIMD-width multiple so
  // every matmul row starts W-aligned: the inner loop then uses aligned
  // loads/stores for any N, and the staging buffers live on the stack
  // (no per-call heap allocation). MAX_* bound every real MFMA shape;
  // anything larger (or a forced-scalar run) falls back to the scalar path.
  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > MFMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > MFMA_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * stride > MFMA_SIMD_MAX_C) {
    return false;
  } else {
    // Zero-initialized as the uniform staging-buffer convention (see the
    // zero-init policy on wmma_simd_matmul). C is pre-seeded just below, so the
    // init is redundant here, but matching the WMMA paths keeps one rule.
    alignas(64) float Abuf[MFMA_SIMD_MAX_AB] = {};
    alignas(64) float Bbuf[MFMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) float Cbuf[MFMA_SIMD_MAX_C] = {};
    for (uint32_t b = 0; b < B; ++b) {
      stage_operands(b, stride, Abuf, Bbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          Cbuf[row * stride + col] =
              (const_acc != ACC_FROM_VGPR)
                  ? std::bit_cast<float>(const_acc)
                  : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
        }
      wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[row * stride + col])});
        }
    }
  }

  return true;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_wmma_f32_mixed_simd(auto &cu, MatrixShape shape, uint32_t a_bits, uint32_t b_bits,
                                  uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                                  uint32_t const_acc, uint32_t c_modifier, uint32_t wave_size,
                                  auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
    return false;
  } else {
    auto reads = read_mixed_matrix_fast_path_regions(cu, s0, s1, s2, static_cast<uint64_t>(M) * K,
                                                     a_bits, static_cast<uint64_t>(N) * K, b_bits,
                                                     static_cast<uint64_t>(M) * N,
                                                     /*acc_bits=*/32, const_acc, wave_size);
    alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
    alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        Cbuf[row * stride + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(reads.acc->lane(out.reg, out.lane)),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k)
        Abuf[row * K + k] =
            ea(reads.a, s0, gfx12_wmma_a_input_loc(wave_size, M, K, row, k, a_bits, b_bits));
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col)
        Bbuf[k * stride + col] =
            eb(reads.b, s1, gfx12_wmma_b_input_loc(wave_size, N, K, col, k, a_bits, b_bits));
    wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[row * stride + col])});
      }
  }

  return true;
}

template <typename ExtractA, typename ExtractB, typename ScaleAWord, typename ScaleBWord>
bool try_exec_wmma_f32_scaled_mixed_simd(
    auto &cu, MatrixShape shape, uint32_t a_bits, uint32_t b_bits, uint32_t s0, uint32_t s1,
    uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc, ScaleAWord scale_a_word,
    ScaleBWord scale_b_word, uint32_t matrix_a_scale, uint32_t matrix_b_scale,
    uint32_t matrix_a_scale_fmt, uint32_t matrix_b_scale_fmt, bool scale16, uint32_t c_modifier,
    uint32_t num_scale_blocks, auto &scale_for, auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
    return false;
  } else {
    auto reads = read_mixed_matrix_fast_path_regions(cu, s0, s1, s2, static_cast<uint64_t>(M) * K,
                                                     a_bits, static_cast<uint64_t>(N) * K, b_bits,
                                                     static_cast<uint64_t>(M) * N,
                                                     /*acc_bits=*/32, const_acc, /*wf_size=*/32);
    alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
    alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) float Cacc[WMMA_SIMD_MAX_C] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        Cacc[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(reads.acc->lane(out.reg, out.lane)),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_block_scaled_a_input_loc(M, K, row, k, a_bits);
        Abuf[row * K + k] = ea(reads.a, s0, al);
      }
    }
    for (uint32_t col = 0; col < N; ++col) {
      for (uint32_t k = 0; k < K; ++k) {
        auto bl = wmma_block_scaled_b_input_loc(N, K, col, k, b_bits);
        Bbuf[k * stride + col] = eb(reads.b, s1, bl);
      }
    }
    for (uint32_t row = 0; row < M; ++row) {
      const uint64_t a_scale_word =
          scale_a_word(wmma_a_scale_lane(M, K, row, matrix_a_scale, a_bits, b_bits));
      for (uint32_t block = 0; block < num_scale_blocks; ++block) {
        uint32_t col = 0;
        alignas(64) float block_sums[64];
        for (; col + W <= N; col += W) {
          util::native<float> block_sum(0.0f);
          for (uint32_t k = 0; k < K; ++k) {
            if (wmma_block_scale_byte(k, scale16) != block)
              continue;
            util::native<float> a(Abuf[row * K + k]);
            util::native<float> bv;
            bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
            block_sum = util::stdx::fma(a, bv, block_sum);
          }
          block_sum.copy_to(block_sums, util::stdx::vector_aligned);
          for (uint32_t j = 0; j < W; ++j) {
            float scaled = block_sums[j] * scale_for(a_scale_word, block, matrix_a_scale_fmt);
            const uint64_t b_scale_word = scale_b_word(wmma_scale_lane(col + j, matrix_b_scale));
            scaled *= scale_for(b_scale_word, block, matrix_b_scale_fmt);
            Cacc[row * N + col + j] += scaled;
          }
        }
        for (; col < N; ++col) {
          float block_sum = 0.0f;
          for (uint32_t k = 0; k < K; ++k) {
            if (wmma_block_scale_byte(k, scale16) == block)
              block_sum = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], block_sum);
          }
          block_sum *= scale_for(a_scale_word, block, matrix_a_scale_fmt);
          const uint64_t b_scale_word = scale_b_word(wmma_scale_lane(col, matrix_b_scale));
          block_sum *= scale_for(b_scale_word, block, matrix_b_scale_fmt);
          Cacc[row * N + col] += block_sum;
        }
      }
    }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cacc[row * N + col])});
      }
  }

  return true;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_swmmac_f32_mixed_simd(auto &cu, MatrixShape shape, uint32_t a_bits, uint32_t b_bits,
                                    uint32_t s0, uint32_t s1, ExtractA ea, ExtractB eb,
                                    uint32_t wave_size, uint32_t compressed_k, auto &dense_k_for,
                                    auto &initial_acc_for, auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
      static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
      stride > WMMA_SIMD_MAX_C) {
    return false;
  } else {
    alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
    alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col)
        Cbuf[col] = initial_acc_for(row, col);
      for (uint32_t ck = 0; ck < compressed_k; ++ck) {
        Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, a_bits));
        const uint32_t dense_k = dense_k_for(row, ck);
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[ck * stride + col] =
              eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, b_bits));
      }
      wmma_simd_matmul<float>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t col = 0; col < N; ++col) {
        if (std::isnan(Cbuf[col])) [[unlikely]] {
          Cbuf[col] = swmmac_replay_nan_buffers(Abuf, Bbuf, compressed_k, stride, col,
                                                initial_acc_for(row, col));
        }
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[col])});
      }
    }
  }

  return true;
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
bool try_exec_wmma_packed16_simd(auto &cu, MatrixShape shape, uint32_t in_bits, uint32_t s0,
                                 uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                                 ReadAcc read_acc, PackResult pack_result, uint32_t const_acc,
                                 uint32_t wave_size, auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
    return false;
  } else {
    alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
    alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
        Cbuf[row * stride + col] = (const_acc != ACC_FROM_VGPR)
                                       ? std::bit_cast<float>(const_acc)
                                       : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k)
        Abuf[row * K + k] = ea(cu, s0, gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits));
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col)
        Bbuf[k * stride + col] = eb(cu, s1, gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits));
    wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
        results.push_back(
            {out.reg, out.lane, out.sub_element, pack_result(Cbuf[row * stride + col])});
      }
  }

  return true;
}

template <typename ExtractA, typename ExtractB, typename PackResult>
bool try_exec_swmmac_packed16_simd(auto &cu, MatrixShape shape, uint32_t in_bits, uint32_t s0,
                                   uint32_t s1, ExtractA ea, ExtractB eb, PackResult pack_result,
                                   uint32_t wave_size, uint32_t compressed_k, auto &dense_k_for,
                                   auto &initial_acc_for, auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
      static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
      stride > WMMA_SIMD_MAX_C) {
    return false;
  } else {
    alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
    alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col)
        Cbuf[col] = initial_acc_for(row, col);
      for (uint32_t ck = 0; ck < compressed_k; ++ck) {
        Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits));
        const uint32_t dense_k = dense_k_for(row, ck);
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[ck * stride + col] =
              eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, in_bits));
      }
      wmma_simd_matmul<float>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t col = 0; col < N; ++col) {
        if (std::isnan(Cbuf[col])) [[unlikely]] {
          Cbuf[col] = swmmac_replay_nan_buffers(Abuf, Bbuf, compressed_k, stride, col,
                                                initial_acc_for(row, col));
        }
        auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
        results.push_back({out.reg, out.lane, out.sub_element, pack_result(Cbuf[col])});
      }
    }
  }

  return true;
}

template <typename ExtractA, typename ExtractB, typename ScaleBlock>
bool try_exec_f32_scaled_impl_simd(auto &cu, MatrixShape shape, uint32_t B, uint32_t a_bits,
                                   uint32_t b_bits, uint32_t s0, uint32_t s1, uint32_t s2,
                                   ExtractA ea, ExtractB eb, ScaleBlock scale_block_sum,
                                   uint32_t const_acc, uint32_t c_modifier, uint32_t wf,
                                   uint32_t num_blocks, auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t BLOCK_K = 32;

  // Column-padded, stack-allocated, aligned B loads. See exec_f32_mixed.
  // Cacc is touched scalar (per-output ldexp) so it keeps an N pitch.
  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > MFMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > MFMA_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * N > MFMA_SIMD_MAX_C) {
    return false;
  } else {
    auto reads = read_mixed_matrix_fast_path_regions(
        cu, s0, s1, s2, static_cast<uint64_t>(M) * K * B, a_bits, static_cast<uint64_t>(N) * K * B,
        b_bits, static_cast<uint64_t>(M) * N * B,
        /*acc_bits=*/32, const_acc, wf);
    // Zero-initialized staging buffers (uniform convention; see
    // wmma_simd_matmul). Cacc is pre-seeded below.
    alignas(64) float Abuf[MFMA_SIMD_MAX_AB] = {};
    alignas(64) float Bbuf[MFMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) float Cacc[MFMA_SIMD_MAX_C] = {};
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = mfma_scale_f8f6f4_input_loc(M, K, row, k, a_bits);
          Abuf[row * K + k] = ea(reads.a, s0, physicalize_loc(al, wf));
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = mfma_scale_f8f6f4_input_loc(N, K, col, k, b_bits);
          Bbuf[k * stride + col] = eb(reads.b, s1, physicalize_loc(bl, wf));
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          Cacc[row * N + col] =
              apply_wmma_c_modifier((const_acc != ACC_FROM_VGPR)
                                        ? std::bit_cast<float>(const_acc)
                                        : std::bit_cast<float>(reads.acc->lane(out.reg, out.lane)),
                                    c_modifier);
        }
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
          uint32_t k_start = blk * BLOCK_K;
          uint32_t k_end = std::min(k_start + BLOCK_K, K);
          uint32_t col = 0;
          alignas(64) float bs[64];
          for (; col + W <= N; col += W) {
            util::native<float> acc(0.0f);
            for (uint32_t k = k_start; k < k_end; ++k) {
              util::native<float> a(Abuf[row * K + k]);
              util::native<float> bv;
              bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
              acc = util::stdx::fma(a, bv, acc);
            }
            acc.copy_to(bs, util::stdx::vector_aligned);
            for (uint32_t j = 0; j < W; ++j)
              Cacc[row * N + col + j] += scale_block_sum(bs[j], row, col + j, b, blk);
          }
          for (; col < N; ++col) {
            float block_sum = 0.0f;
            for (uint32_t k = k_start; k < k_end; ++k)
              block_sum = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], block_sum);
            Cacc[row * N + col] += scale_block_sum(block_sum, row, col, b, blk);
          }
        }
      }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cacc[row * N + col])});
        }
    }
  }

  return true;
}

inline bool try_exec_i32_i8_simd(auto &cu, MatrixShape shape, uint32_t B, uint32_t s0, uint32_t s1,
                                 uint32_t s2, uint32_t const_acc, uint32_t cbsz, uint32_t abid,
                                 uint32_t blgp, uint32_t wf, auto &results) {
  const auto [M, N, K] = shape;
  // Column-padded, stack-allocated, aligned. See exec_f32_mixed.
  constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > MFMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > MFMA_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * stride > MFMA_SIMD_MAX_C) {
    return false;
  } else {
    // Zero-initialized staging buffers (uniform convention; see
    // wmma_simd_matmul). Cbuf is pre-seeded below.
    alignas(64) int32_t Abuf[MFMA_SIMD_MAX_AB] = {};
    alignas(64) int32_t Bbuf[MFMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) int32_t Cbuf[MFMA_SIMD_MAX_C] = {};
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          Cbuf[row * stride + col] =
              (const_acc != ACC_FROM_VGPR)
                  ? static_cast<int32_t>(const_acc)
                  : static_cast<int32_t>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, 8);
          if (cbsz != 0)
            al.lane = permute_a_lane(al.lane, cbsz, abid);
          Abuf[row * K + k] = extract_i8(cu, s0, physicalize_loc(al, wf));
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, 8);
          if (blgp != 0)
            bl.lane = permute_b_lane(bl.lane, blgp);
          Bbuf[k * stride + col] = extract_i8(cu, s1, physicalize_loc(bl, wf));
        }
      for (uint32_t row = 0; row < M; ++row) {
        uint32_t col = 0;
        for (; col + W <= N; col += W) {
          util::native<int32_t> c;
          c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<int32_t> a(Abuf[row * K + k]);
            util::native<int32_t> bv;
            bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
            c += a * bv;
          }
          c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
        }
        for (; col < N; ++col) {
          uint32_t acc = static_cast<uint32_t>(Cbuf[row * stride + col]);
          for (uint32_t k = 0; k < K; ++k)
            acc += static_cast<uint32_t>(Abuf[row * K + k] * Bbuf[k * stride + col]);
          Cbuf[row * stride + col] = static_cast<int32_t>(acc);
        }
      }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          results.push_back({out.reg, out.lane, static_cast<uint32_t>(Cbuf[row * stride + col])});
        }
    }
  }

  return true;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_wmma_i32_simd(auto &cu, MatrixShape shape, uint32_t in_bits, uint32_t s0, uint32_t s1,
                            uint32_t s2, ExtractA ea, ExtractB eb, bool clamp, uint32_t const_acc,
                            uint32_t wave_size, auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
    return false;
  } else {
    alignas(64) int32_t Abuf[WMMA_SIMD_MAX_AB] = {};
    alignas(64) int32_t Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) int32_t Cbuf[WMMA_SIMD_MAX_C] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k)
        Abuf[row * K + k] = ea(cu, s0, gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits));
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col)
        Bbuf[k * stride + col] = eb(cu, s1, gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits));
    wmma_simd_matmul<int32_t>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        int64_t acc = (const_acc != ACC_FROM_VGPR)
                          ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                          : static_cast<int64_t>(static_cast<int32_t>(
                                RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane)));
        acc += static_cast<int64_t>(Cbuf[row * stride + col]);
        results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
      }
  }

  return true;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_swmmac_i32_simd(auto &cu, MatrixShape shape, uint32_t in_bits, uint32_t s0,
                              uint32_t s1, uint32_t acc_base, ExtractA ea, ExtractB eb, bool clamp,
                              uint32_t const_acc, uint32_t wave_size, uint32_t compressed_k,
                              auto &dense_k_for, auto &results) {
  const auto [M, N, K] = shape;
  constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
      static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
      stride > WMMA_SIMD_MAX_C) {
    return false;
  } else {
    alignas(64) int32_t Abuf[WMMA_SIMD_MAX_AB] = {};
    alignas(64) int32_t Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
    alignas(64) int32_t Cbuf[WMMA_SIMD_MAX_C] = {};
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col)
        Cbuf[col] = 0;
      for (uint32_t ck = 0; ck < compressed_k; ++ck) {
        Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits));
        const uint32_t dense_k = dense_k_for(row, ck);
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[ck * stride + col] =
              eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, in_bits));
      }
      wmma_simd_matmul<int32_t>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        int64_t acc = (const_acc != ACC_FROM_VGPR)
                          ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                          : static_cast<int64_t>(static_cast<int32_t>(
                                RegisterAccess(cu).read_vgpr(acc_base + out.reg, out.lane)));
        acc += static_cast<int64_t>(Cbuf[col]);
        results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
      }
    }
  }

  return true;
}

inline bool try_exec_f64_simd(auto &cu, MatrixShape shape, uint32_t B, uint32_t s0, uint32_t s1,
                              uint32_t s2, uint32_t const_acc, auto &apply_neg, auto &results) {
  const auto [M, N, K] = shape;
  // Column-padded, stack-allocated, aligned. See exec_f32_mixed. MFMA_F64_SIMD_MAX_BSTRIDE
  // is half the f32 cap because native<double> packs half as many lanes.
  constexpr uint32_t W = static_cast<uint32_t>(util::native<double>::size());
  const uint32_t stride = ((N + W - 1) / W) * W;
  if (util::force_scalar() || static_cast<size_t>(M) * K > MFMA_SIMD_MAX_AB ||
      static_cast<size_t>(K) * stride > MFMA_F64_SIMD_MAX_BSTRIDE ||
      static_cast<size_t>(M) * stride > MFMA_SIMD_MAX_C) {
    return false;
  } else {
    // Zero-initialized staging buffers (uniform convention; see
    // wmma_simd_matmul). Cbuf is pre-seeded below.
    alignas(64) double Abuf[MFMA_SIMD_MAX_AB] = {};
    alignas(64) double Bbuf[MFMA_F64_SIMD_MAX_BSTRIDE] = {};
    alignas(64) double Cbuf[MFMA_SIMD_MAX_C] = {};
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_64(M, N, row, col, b);
          if (const_acc != ACC_FROM_VGPR) {
            Cbuf[row * stride + col] = static_cast<double>(std::bit_cast<float>(const_acc));
          } else {
            uint32_t lo = RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
            uint32_t hi = RegisterAccess(cu).read_vgpr(s2 + out.reg + 1, out.lane);
            Cbuf[row * stride + col] = std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
          }
          Cbuf[row * stride + col] = apply_neg(Cbuf[row * stride + col], 0x4u);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, 64);
          Abuf[row * K + k] = apply_neg(extract_f64(cu, s0, al), 0x1u);
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, 64);
          Bbuf[k * stride + col] = apply_neg(extract_f64(cu, s1, bl), 0x2u);
        }
      for (uint32_t row = 0; row < M; ++row) {
        uint32_t col = 0;
        for (; col + W <= N; col += W) {
          util::native<double> c;
          c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<double> a(Abuf[row * K + k]);
            util::native<double> bv;
            bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
            c = util::stdx::fma(a, bv, c);
          }
          c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
        }
        for (; col < N; ++col) {
          double acc = Cbuf[row * stride + col];
          for (uint32_t k = 0; k < K; ++k)
            acc = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], acc);
          Cbuf[row * stride + col] = acc;
        }
      }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_64(M, N, row, col, b);
          uint64_t bits = std::bit_cast<uint64_t>(Cbuf[row * stride + col]);
          results.push_back(
              {out.reg, out.lane, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)});
        }
    }
  }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K>
bool try_exec_wmma_f32_f32_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                     uint32_t const_acc = ACC_FROM_VGPR, uint32_t c_modifier = 0) {
  constexpr uint32_t in_bits = 32;

  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  if (util::force_scalar() || !mma_f32_native_width_supported(N, W)) {
    return false;
  }
  require_wmma_wave32(cu);
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32, const_acc, wf);
  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
  alignas(64) float A_buf[M * K];
  alignas(64) float B_buf[K * N];
  alignas(64) float C_buf[M * N];
  alignas(64) uint32_t A_words[M * K];
  alignas(64) uint32_t B_words[N * K];
  alignas(64) uint32_t C_words[M * N];
  copy_matrix_region_words(reads.a, A_words);
  copy_matrix_region_words(reads.b, B_words);
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      C_buf[row * N + col] = apply_wmma_c_modifier(
          (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]),
          c_modifier);
    }
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t k = 0; k < K; ++k) {
      auto al = wmma_input_loc(M, K, row, k, in_bits);
      A_buf[row * K + k] = std::bit_cast<float>(A_words[al.vgpr_offset * wf + al.lane]);
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_input_loc(N, K, col, k, in_bits);
      B_buf[k * N + col] = std::bit_cast<float>(B_words[bl.vgpr_offset * wf + bl.lane]);
    }
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t c0 = 0; c0 < N; c0 += W) {
      util::native<float> c_row;
      c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<float> a_bcast(A_buf[row * K + k]);
        util::native<float> b_row;
        b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
        c_row = util::stdx::fma(a_bcast, b_row, c_row);
      }
      c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
    }
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      writes.set_linear_word(out.reg * wf + out.lane,
                             std::bit_cast<uint32_t>(C_buf[row * N + col]));
    }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH>
bool try_exec_f32_mfma_f32_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                     uint32_t const_acc, uint32_t cbsz, uint32_t /*abid*/,
                                     uint32_t blgp) {
  constexpr uint32_t in_bits = 32;

  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  if (util::force_scalar() || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
      cu.wf_size() != 64) {
    return false;
  }
  const uint32_t wf = cu.wf_size();
  auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, BATCH, in_bits, const_acc, wf);
  auto writes = write_mfma_acc32_region(cu, dst, M, N, BATCH, wf);
  alignas(64) float A_buf[M * K];
  alignas(64) float B_buf[K * N];
  static_assert(M * N * BATCH * sizeof(float) <= 8 * 1024,
                "specialized MFMA result staging exceeds the stack budget");
  alignas(64) float C_buf[M * N * BATCH];
  alignas(64) uint32_t A_words[M * K * BATCH];
  alignas(64) uint32_t B_words[N * K * BATCH];
  alignas(64) uint32_t C_words[M * N * BATCH];
  copy_matrix_region_words(reads.a, A_words);
  copy_matrix_region_words(reads.b, B_words);
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  bool has_nan_or_inf = false;
  for (uint32_t b = 0; b < BATCH; ++b) {
    float *C_batch = &C_buf[b * M * N];
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        C_batch[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                     ? std::bit_cast<float>(const_acc)
                                     : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, BATCH, row, k, b, in_bits);
        A_buf[row * K + k] = std::bit_cast<float>(A_words[al.vgpr_offset * wf + al.lane]);
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, BATCH, col, k, b, in_bits);
        B_buf[k * N + col] = std::bit_cast<float>(B_words[bl.vgpr_offset * wf + bl.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
      }
  }
  // Publish only after every batch has consumed its inputs.
  for (uint32_t b = 0; b < BATCH; ++b)
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        float fv = C_buf[(b * M + row) * N + col];
        writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
        if (std::isnan(fv) || std::isinf(fv))
          has_nan_or_inf = true;
      }
  if (has_nan_or_inf) {
    util::Logger::vm([&](auto &os) {
      os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f32", dst,
                        s0, s1, s2, M, N, K);
    });
  }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
bool try_exec_f32_mfma_f16_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                     uint32_t const_acc, uint32_t cbsz, uint32_t /*abid*/,
                                     uint32_t blgp) {
  constexpr uint32_t B = BATCH, in_bits = 16;

  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  if (util::force_scalar() || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
      cu.wf_size() != 64) {
    return false;
  }
  const uint32_t wf = cu.wf_size();
  auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
  auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
  alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
  alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
  static_assert(M * N * B * sizeof(float) <= 8 * 1024,
                "specialized MFMA result staging exceeds the stack budget");
  alignas(64) float C_buf[M * N * B]; // C[batch][row][col]
  alignas(64) uint32_t C_words[M * N * B];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // A/B occupy M*K*B and N*K*B packed f16 over their VGPRs. Bulk-convert each
  // whole region to f32 once with F16C (one vector op per 16 halves) instead
  // of branchy per-element f16_to_f32, then the hoist is a pure f32
  // index-shuffle (f16 of word w lane l sub s -> flat (w*wf+l)*2+s).
  alignas(64) float A_f32[M * K * B];
  alignas(64) float B_f32[N * K * B];
  convert_f16_matrix_region(reads.a, A_f32, M * K * B);
  convert_f16_matrix_region(reads.b, B_f32, N * K * B);
  bool has_nan_or_inf = false;
  for (uint32_t b = 0; b < B; ++b) {
    float *C_batch = &C_buf[b * M * N];
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        C_batch[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                     ? std::bit_cast<float>(const_acc)
                                     : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, B, row, k, b, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, B, col, k, b, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    // Dense MxKxN matmul, W-lane native SIMD FMA over N (N/W chunks per row).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
      }
  }
  // Publish only after every batch has consumed its inputs. This preserves
  // instruction-level snapshot semantics when dst overlaps a later batch's
  // source or accumulator registers.
  for (uint32_t b = 0; b < B; ++b)
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        float fv = C_buf[(b * M + row) * N + col];
        writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
        if (std::isnan(fv) || std::isinf(fv))
          has_nan_or_inf = true;
      }
  if (has_nan_or_inf) {
    util::Logger::vm([&](auto &os) {
      os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f16", dst,
                        s0, s1, s2, M, N, K);
    });
  }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
bool try_exec_f32_mfma_bf16_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                      uint32_t const_acc, uint32_t cbsz, uint32_t /*abid*/,
                                      uint32_t blgp) {
  constexpr uint32_t B = BATCH, in_bits = 16;

  constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
  if (util::force_scalar() || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
      cu.wf_size() != 64) {
    return false;
  }
  const uint32_t wf = cu.wf_size();
  auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
  auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
  alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
  alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
  static_assert(M * N * B * sizeof(float) <= 8 * 1024,
                "specialized MFMA result staging exceeds the stack budget");
  alignas(64) float C_buf[M * N * B]; // C[batch][row][col]
  alignas(64) uint32_t C_words[M * N * B];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // Bulk-convert the packed bf16 regions to f32 once (zero-extend + shift),
  // then the hoist is a pure f32 index-shuffle.
  alignas(64) float A_f32[M * K * B];
  alignas(64) float B_f32[N * K * B];
  convert_bf16_matrix_region(reads.a, A_f32, M * K * B);
  convert_bf16_matrix_region(reads.b, B_f32, N * K * B);
  bool has_nan_or_inf = false;
  for (uint32_t b = 0; b < B; ++b) {
    float *C_batch = &C_buf[b * M * N];
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        C_batch[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                     ? std::bit_cast<float>(const_acc)
                                     : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, B, row, k, b, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, B, col, k, b, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    // Dense MxKxN matmul, W-lane native SIMD FMA over N (N/W chunks per row).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
      }
  }
  // Publish only after every batch has consumed its inputs.
  for (uint32_t b = 0; b < B; ++b)
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        float fv = C_buf[(b * M + row) * N + col];
        writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
        if (std::isnan(fv) || std::isinf(fv))
          has_nan_or_inf = true;
      }
  if (has_nan_or_inf) {
    util::Logger::vm([&](auto &os) {
      os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_bf16", dst,
                        s0, s1, s2, M, N, K);
    });
  }

  return true;
}

} // namespace rocjitsu::amdgpu::mma_backend

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_DENSE_H_
