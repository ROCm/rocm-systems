// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_DENSE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_DENSE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"

namespace rocjitsu::amdgpu::mma_backend {

inline bool try_exec_wmma_f32_16x16x32_f16_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                                uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                                                uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }
  require_wmma_wave32(cu);
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32, const_acc, wf);
  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
  alignas(64) float A_buf[M * K]; // A[row][k]
  alignas(64) float B_buf[K * N]; // B[k][col]
  alignas(64) float C_buf[M * N]; // C[row][col]
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // A and B each occupy 8 VGPRs x wf lanes = 2*8*wf packed f16 (16 f16/lane
  // for K=32). Bulk-convert the region to f32 once, then the hoist is a pure
  // f32 index-shuffle (f16 j of word w sub s -> flat (w*wf+lane)*2+s).
  constexpr uint32_t NUM_IN_REGS = 8;
  const uint32_t n_halves = 2 * NUM_IN_REGS * wf;
  alignas(64) float A_f32[2 * NUM_IN_REGS * 64];
  alignas(64) float B_f32[2 * NUM_IN_REGS * 64];
  convert_f16_matrix_region(reads.a, A_f32, n_halves);
  convert_f16_matrix_region(reads.b, B_f32, n_halves);
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
      A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_input_loc(N, K, col, k, in_bits);
      B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
    }
  // Dense 16x32 * 32x16 -> 16x16 matmul, 16-lane stdx FMA per row.
  for (uint32_t row = 0; row < M; ++row) {
    util::native<float> c_row;
    c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
    for (uint32_t k = 0; k < K; ++k) {
      util::native<float> a_bcast(A_buf[row * K + k]);
      util::native<float> b_row;
      b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
      c_row = util::stdx::fma(a_bcast, b_row, c_row);
    }
    c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
  }
  // Scatter directly back to VGPRs (no Result staging vector).
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      writes.set_linear_word(out.reg * wf + out.lane,
                             std::bit_cast<uint32_t>(C_buf[row * N + col]));
    }

  return true;
}

inline bool try_exec_wmma_f32_16x16x32_bf16_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                                 uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                                                 uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }
  require_wmma_wave32(cu);
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32, const_acc, wf);
  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
  alignas(64) float A_buf[M * K]; // A[row][k]
  alignas(64) float B_buf[K * N]; // B[k][col]
  alignas(64) float C_buf[M * N]; // C[row][col]
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // A and B each occupy 8 VGPRs x wf lanes = 2*8*wf packed bf16 (16 bf16/lane
  // for K=32). Bulk-convert the region to f32 once, then the hoist is a pure
  // f32 index-shuffle (bf16 j of word w sub s -> flat (w*wf+lane)*2+s).
  constexpr uint32_t NUM_IN_REGS = 8;
  const uint32_t n_halves = 2 * NUM_IN_REGS * wf;
  alignas(64) float A_f32[2 * NUM_IN_REGS * 64];
  alignas(64) float B_f32[2 * NUM_IN_REGS * 64];
  convert_bf16_matrix_region(reads.a, A_f32, n_halves);
  convert_bf16_matrix_region(reads.b, B_f32, n_halves);
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
      A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_input_loc(N, K, col, k, in_bits);
      B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
    }
  // Dense 16x32 * 32x16 -> 16x16 matmul, 16-lane stdx FMA per row.
  for (uint32_t row = 0; row < M; ++row) {
    util::native<float> c_row;
    c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
    for (uint32_t k = 0; k < K; ++k) {
      util::native<float> a_bcast(A_buf[row * K + k]);
      util::native<float> b_row;
      b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
      c_row = util::stdx::fma(a_bcast, b_row, c_row);
    }
    c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
  }
  // Scatter directly back to VGPRs (no Result staging vector).
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      writes.set_linear_word(out.reg * wf + out.lane,
                             std::bit_cast<uint32_t>(C_buf[row * N + col]));
    }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
bool try_exec_wmma_f32_f8_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                    uint32_t const_acc = ACC_FROM_VGPR, uint32_t c_modifier = 0) {
  constexpr uint32_t in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 WMMA assumes N is a multiple of the zmm width");

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }
  require_wmma_wave32(cu);
  constexpr uint32_t W = 16;
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32, const_acc, wf);
  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
  alignas(64) float A_buf[M * K]; // A[row][k]
  alignas(64) float B_buf[K * N]; // B[k][col]
  alignas(64) float C_buf[M * N]; // C[row][col]
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // Bulk-convert each whole packed f8 region to f32 once, then the hoist is a
  // pure f32 index-shuffle (byte of word w lane l sub s -> (w*wf+l)*4+s).
  alignas(64) float A_f32[M * K];
  alignas(64) float B_f32[N * K];
  convert_f8_matrix_region<A_FP8, FNUZ>(reads.a, A_f32, M * K);
  convert_f8_matrix_region<B_FP8, FNUZ>(reads.b, B_f32, N * K);
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
      auto al = wmma_a_input_loc(M, K, row, k, in_bits, in_bits);
      A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_b_input_loc(N, K, col, k, in_bits, in_bits);
      B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
    }
  // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
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
  // Scatter directly back to VGPRs (no Result staging vector).
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      writes.set_linear_word(out.reg * wf + out.lane,
                             std::bit_cast<uint32_t>(C_buf[row * N + col]));
    }

  return true;
}

inline bool try_exec_wmma_bf16f32_16x16x32_bf16_simd(auto &cu, uint32_t dst, uint32_t s0,
                                                     uint32_t s1, uint32_t s2,
                                                     uint32_t const_acc = ACC_FROM_VGPR,
                                                     uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }

  require_wmma_wave32(cu);
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32, const_acc, wf);
  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);

  alignas(64) float A_buf[M * K];
  alignas(64) float B_buf[K * N];
  alignas(64) float C_buf[M * N];
  alignas(64) float A_f32[M * K];
  alignas(64) float B_f32[K * N];
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  convert_bf16_matrix_region(reads.a, A_f32, M * K);
  convert_bf16_matrix_region(reads.b, B_f32, K * N);

  auto initial_acc = [&](uint32_t row, uint32_t col) {
    auto out = wmma_output_loc_32(M, N, row, col);
    return apply_wmma_c_modifier((const_acc != ACC_FROM_VGPR)
                                     ? std::bit_cast<float>(const_acc)
                                     : std::bit_cast<float>(C_words[out.reg * wf + out.lane]),
                                 c_modifier);
  };
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col)
      C_buf[row * N + col] = initial_acc(row, col);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t k = 0; k < K; ++k) {
      auto in = wmma_input_loc(M, K, row, k, in_bits);
      A_buf[row * K + k] = A_f32[(in.vgpr_offset * wf + in.lane) * 2 + in.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto in = wmma_input_loc(N, K, col, k, in_bits);
      B_buf[k * N + col] = B_f32[(in.vgpr_offset * wf + in.lane) * 2 + in.sub_element];
    }

  for (uint32_t row = 0; row < M; ++row) {
    util::native<float> c_row;
    c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
    for (uint32_t k = 0; k < K; ++k) {
      util::native<float> b_row;
      b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
      c_row = util::native_fma(util::native<float>(A_buf[row * K + k]), b_row, c_row);
    }
    const uint64_t nan_lanes = util::simd_mask_to_bits(util::stdx::isnan(c_row));
    c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
    // A packed host FMA may propagate a different NaN operand than scalar
    // FMA.  Recompute only exceptional lanes with the shared source-priority
    // helper; finite rows retain the full-width AVX-512 path.
    uint64_t pending_nan_lanes = nan_lanes;
    while (pending_nan_lanes != 0) {
      const uint32_t col = static_cast<uint32_t>(std::countr_zero(pending_nan_lanes));
      pending_nan_lanes &= pending_nan_lanes - 1;
      float acc = initial_acc(row, col);
      for (uint32_t k = 0; k < K; ++k)
        acc = matrix_fma(A_buf[row * K + k], B_buf[k * N + col], acc);
      C_buf[row * N + col] = acc;
    }
  }

  // wmma_output_loc_16 pairs adjacent rows in each destination register.
  // Pack two complete rows per vector: even rows become the low halves and
  // odd rows become the high halves.  This is the scalar BF16 truncation
  // contract (high 16 bits of each f32), not round-to-nearest-even.
  constexpr uint32_t DST_REGS = 4;
  alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32];
  for (uint32_t pair = 0; pair < M / 2; ++pair) {
    util::native<float> even_row;
    util::native<float> odd_row;
    even_row.copy_from(&C_buf[(2 * pair) * N], util::stdx::vector_aligned);
    odd_row.copy_from(&C_buf[(2 * pair + 1) * N], util::stdx::vector_aligned);
    auto packed = (std::bit_cast<util::native<uint32_t>>(even_row) >> 16) |
                  (std::bit_cast<util::native<uint32_t>>(odd_row) & 0xFFFF0000u);
    packed.copy_to(&words[(pair % DST_REGS) * WMMA_WAVE32 + (pair / DST_REGS) * N],
                   util::stdx::vector_aligned);
  }
  for (uint32_t reg = 0; reg < DST_REGS; ++reg)
    for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane)
      writes.set_linear_word(reg * wf + lane, words[reg * WMMA_WAVE32 + lane]);

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K>
bool try_exec_wmma_f16_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                 uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t in_bits = 16;
  static_assert(N % 16 == 0, "specialized f16 WMMA assumes N is a multiple of the zmm width");

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }
  require_wmma_wave32(cu);
  constexpr uint32_t W = 16;
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/16, const_acc, wf);
  auto writes = readwrite_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);
  alignas(64) float A_buf[M * K];
  alignas(64) float B_buf[K * N];
  alignas(64) float C_buf[M * N];
  alignas(64) float A_f32[M * K];
  alignas(64) float B_f32[N * K];
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  convert_f16_matrix_region(reads.a, A_f32, M * K);
  convert_f16_matrix_region(reads.b, B_f32, N * K);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_16(M, N, row, col);
      if (const_acc != ACC_FROM_VGPR) {
        C_buf[row * N + col] = std::bit_cast<float>(const_acc);
      } else {
        uint32_t raw = C_words[out.reg * wf + out.lane];
        C_buf[row * N + col] =
            util::f16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
      }
    }
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t k = 0; k < K; ++k) {
      auto al = wmma_input_loc(M, K, row, k, in_bits);
      A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_input_loc(N, K, col, k, in_bits);
      B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
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
  // Pack f32 results to f16, two per dst word (the WMMA 16-bit output map).
  constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
  alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
  alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_16(M, N, row, col);
      uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
      uint32_t shift = out.sub_element * 16;
      uint16_t v = util::f32_to_f16(C_buf[row * N + col]);
      words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
      masks[idx] |= 1u << out.sub_element;
    }
  for (uint32_t reg = 0; reg < DST_REGS; ++reg)
    for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
      uint32_t idx = reg * WMMA_WAVE32 + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = writes.linear_word(reg * wf + lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      writes.set_linear_word(reg * wf + lane, word);
    }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K>
bool try_exec_wmma_bf16_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                  uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t in_bits = 16;
  static_assert(N % 16 == 0, "specialized bf16 WMMA assumes N is a multiple of the zmm width");

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }
  require_wmma_wave32(cu);
  constexpr uint32_t W = 16;
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/16, const_acc, wf);
  auto writes = readwrite_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);
  alignas(64) float A_buf[M * K];
  alignas(64) float B_buf[K * N];
  alignas(64) float C_buf[M * N];
  alignas(64) float A_f32[M * K];
  alignas(64) float B_f32[N * K];
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  convert_bf16_matrix_region(reads.a, A_f32, M * K);
  convert_bf16_matrix_region(reads.b, B_f32, N * K);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_16(M, N, row, col);
      if (const_acc != ACC_FROM_VGPR) {
        C_buf[row * N + col] = std::bit_cast<float>(const_acc);
      } else {
        uint32_t raw = C_words[out.reg * wf + out.lane];
        C_buf[row * N + col] =
            util::bf16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
      }
    }
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t k = 0; k < K; ++k) {
      auto al = wmma_input_loc(M, K, row, k, in_bits);
      A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_input_loc(N, K, col, k, in_bits);
      B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
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
  // Pack f32 results to bf16 (truncation), two per dst word.
  constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
  alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
  alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_16(M, N, row, col);
      uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
      uint32_t shift = out.sub_element * 16;
      uint16_t v = util::f32_to_bf16(C_buf[row * N + col]);
      words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
      masks[idx] |= 1u << out.sub_element;
    }
  for (uint32_t reg = 0; reg < DST_REGS; ++reg)
    for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
      uint32_t idx = reg * WMMA_WAVE32 + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = writes.linear_word(reg * wf + lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      writes.set_linear_word(reg * wf + lane, word);
    }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
bool try_exec_wmma_f16_f8_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                    uint32_t const_acc = ACC_FROM_VGPR, bool fp16_ovfl = false) {
  constexpr uint32_t in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 WMMA assumes N is a multiple of the zmm width");

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }
  require_wmma_wave32(cu);
  constexpr uint32_t W = 16;
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/16, const_acc, wf);
  auto writes = readwrite_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);
  alignas(64) float A_buf[M * K];
  alignas(64) float B_buf[K * N];
  alignas(64) float C_buf[M * N];
  alignas(64) float A_f32[M * K];
  alignas(64) float B_f32[N * K];
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  convert_f8_matrix_region<A_FP8, FNUZ>(reads.a, A_f32, M * K);
  convert_f8_matrix_region<B_FP8, FNUZ>(reads.b, B_f32, N * K);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_16(M, N, row, col);
      if (const_acc != ACC_FROM_VGPR) {
        C_buf[row * N + col] = std::bit_cast<float>(const_acc);
      } else {
        uint32_t raw = C_words[out.reg * wf + out.lane];
        C_buf[row * N + col] =
            util::f16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
      }
    }
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t k = 0; k < K; ++k) {
      auto al = wmma_a_input_loc(M, K, row, k, in_bits, in_bits);
      A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_b_input_loc(N, K, col, k, in_bits, in_bits);
      B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
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
  // Pack f32 results to f16, two per dst word (the WMMA 16-bit output map).
  constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
  alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
  alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_16(M, N, row, col);
      uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
      uint32_t shift = out.sub_element * 16;
      uint16_t v = wmma_round_f16(C_buf[row * N + col], fp16_ovfl);
      words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
      masks[idx] |= 1u << out.sub_element;
    }
  for (uint32_t reg = 0; reg < DST_REGS; ++reg)
    for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
      uint32_t idx = reg * WMMA_WAVE32 + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = writes.linear_word(reg * wf + lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      writes.set_linear_word(reg * wf + lane, word);
    }

  return true;
}

inline bool try_exec_wmma_i32_16x16x64_iu8_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                                uint32_t s2, bool a_signed, bool b_signed,
                                                bool clamp, uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t M = 16, N = 16, K = 64, in_bits = 8;

  if (util::force_scalar() || util::native<float>::size() != 16) {
    return false;
  }
  require_wmma_wave32(cu);
  const uint32_t wf = cu.wf_size();
  auto reads =
      read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32, const_acc, wf);
  // Accumulate in unsigned 32-bit (wrap is well-defined; identical mod 2^32
  // to the intended signed wrap), sign-restore via int32 cast at pack time.
  alignas(64) uint32_t A_buf[M * K]; // A[row][k] (sign-/zero-extended bits)
  alignas(64) uint32_t B_buf[K * N]; // B[k][col] (sign-/zero-extended bits)
  alignas(64) uint32_t S_buf[M * N]; // sum-of-products, accumulator added at pack
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // Bulk-extend the packed byte regions to int32 once, then the hoist is a
  // pure i32 index-shuffle (byte j of word w lane l sub s -> (w*wf+l)*4+s).
  alignas(64) int32_t A_i32[M * K];
  alignas(64) int32_t B_i32[N * K];
  if (a_signed)
    convert_i8_matrix_region(reads.a, A_i32, M * K);
  else
    convert_u8_matrix_region(reads.a, A_i32, M * K);
  if (b_signed)
    convert_i8_matrix_region(reads.b, B_i32, N * K);
  else
    convert_u8_matrix_region(reads.b, B_i32, N * K);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t k = 0; k < K; ++k) {
      auto al = wmma_input_loc(M, K, row, k, in_bits);
      A_buf[row * K + k] = A_i32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = wmma_input_loc(N, K, col, k, in_bits);
      B_buf[k * N + col] = B_i32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
    }
  // Dense 16x64 * 64x16 -> 16x16 product-sum, 16-lane stdx u32 MAC per row
  // (unsigned to avoid signed-overflow UB; bits match the signed math).
  for (uint32_t row = 0; row < M; ++row) {
    util::native<uint32_t> s_row(0);
    for (uint32_t k = 0; k < K; ++k) {
      util::native<uint32_t> a_bcast(A_buf[row * K + k]);
      util::native<uint32_t> b_row;
      b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
      s_row += a_bcast * b_row;
    }
    s_row.copy_to(&S_buf[row * N], util::stdx::vector_aligned);
  }
  // Add the accumulator in 64-bit and pack (saturating when clamp is set),
  // scattering directly back to VGPRs.
  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      int64_t acc =
          (const_acc != ACC_FROM_VGPR)
              ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
              : static_cast<int64_t>(static_cast<int32_t>(C_words[out.reg * wf + out.lane]));
      acc += static_cast<int64_t>(static_cast<int32_t>(S_buf[row * N + col]));
      writes.set_linear_word(out.reg * wf + out.lane, pack_i32_acc(acc, clamp));
    }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
bool try_exec_f32_mfma_f8_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                    uint32_t const_acc, uint32_t cbsz, uint32_t /*abid*/,
                                    uint32_t blgp) {
  constexpr uint32_t B = 1, in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 MFMA assumes N is a multiple of the zmm width");

  if (util::force_scalar() || cbsz != 0 || blgp != 0 || util::native<float>::size() != 16 ||
      cu.wf_size() != 64) {
    return false;
  }
  constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
  const uint32_t wf = cu.wf_size();
  auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
  auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
  alignas(64) float A_buf[M * K]; // A[row][k]
  alignas(64) float B_buf[K * N]; // B[k][col]
  alignas(64) float C_buf[M * N]; // C[row][col]
  alignas(64) uint32_t C_words[M * N];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // Bulk-convert the packed f8 regions to f32 once through the LUTs, then the
  // hoist is a pure f32 index-shuffle (byte of word w lane l sub s ->
  // (w*wf+l)*4+s).
  alignas(64) float A_f32[M * K];
  alignas(64) float B_f32[N * K];
  convert_f8_matrix_region<A_FP8, FNUZ>(reads.a, A_f32, M * K);
  convert_f8_matrix_region<B_FP8, FNUZ>(reads.b, B_f32, N * K);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = output_loc_32(M, N, row, col, 0);
      C_buf[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                 ? std::bit_cast<float>(const_acc)
                                 : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
    }
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t k = 0; k < K; ++k) {
      auto al = input_loc(M, K, B, row, k, 0, in_bits);
      A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t col = 0; col < N; ++col) {
      auto bl = input_loc(N, K, B, col, k, 0, in_bits);
      B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
    }
  // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
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
  // Scatter directly back to VGPRs (no Result staging vector).
  bool has_nan_or_inf = false;
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      auto out = output_loc_32(M, N, row, col, 0);
      float fv = C_buf[row * N + col];
      writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
      if (std::isnan(fv) || std::isinf(fv))
        has_nan_or_inf = true;
    }
  if (has_nan_or_inf) {
    util::Logger::vm([&](auto &os) {
      os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f8", dst,
                        s0, s1, s2, M, N, K);
    });
  }

  return true;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
bool try_exec_i32_mfma_i8_spec_simd(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                    uint32_t const_acc) {
  constexpr uint32_t B = BATCH, in_bits = 8;
  static_assert(N % 16 == 0, "specialized i8 MFMA assumes N is a multiple of the zmm width");

  if (util::force_scalar() || util::native<float>::size() != 16 || cu.wf_size() != 64) {
    return false;
  }
  constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
  const uint32_t wf = cu.wf_size();
  auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
  auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
  // Accumulate in unsigned 32-bit (wrap is well-defined and identical mod
  // 2^32 to the intended signed wrap), so the SIMD path has no signed-
  // overflow UB.
  alignas(64) uint32_t A_buf[M * K]; // A[row][k] (sign-extended bits, one batch block)
  alignas(64) uint32_t B_buf[K * N]; // B[k][col] (sign-extended bits, one batch block)
  static_assert(M * N * B * sizeof(uint32_t) <= 8 * 1024,
                "specialized MFMA result staging exceeds the stack budget");
  alignas(64) uint32_t C_buf[M * N * B]; // C[batch][row][col]
  alignas(64) uint32_t C_words[M * N * B];
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, C_words);
  // Bulk sign-extend the packed i8 regions to int32 once, then the hoist is
  // a pure i32 index-shuffle (byte of word w lane l sub s -> (w*wf+l)*4+s).
  alignas(64) int32_t A_i32[M * K * B];
  alignas(64) int32_t B_i32[N * K * B];
  convert_i8_matrix_region(reads.a, A_i32, M * K * B);
  convert_i8_matrix_region(reads.b, B_i32, N * K * B);
  for (uint32_t b = 0; b < B; ++b) {
    uint32_t *C_batch = &C_buf[b * M * N];
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        C_batch[row * N + col] =
            (const_acc != ACC_FROM_VGPR) ? const_acc : C_words[out.reg * wf + out.lane];
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, B, row, k, b, in_bits);
        A_buf[row * K + k] = A_i32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, B, col, k, b, in_bits);
        B_buf[k * N + col] = B_i32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    // Dense MxKxN matmul, W-lane (zmm) stdx u32 MAC over N (N/W chunks per
    // row); unsigned wrap matches the scalar signed accumulation mod 2^32.
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<uint32_t> c_row;
        c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<uint32_t> a_bcast(A_buf[row * K + k]);
          util::native<uint32_t> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row += a_bcast * b_row;
        }
        c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
      }
  }
  // Publish only after every batch has consumed its inputs.
  for (uint32_t b = 0; b < B; ++b)
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, b);
        writes.set_linear_word(out.reg * wf + out.lane, C_buf[(b * M + row) * N + col]);
      }

  return true;
}

} // namespace rocjitsu::amdgpu::mma_backend

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_AVX512_DENSE_H_
