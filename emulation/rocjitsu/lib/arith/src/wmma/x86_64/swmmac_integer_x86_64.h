// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse integer WMMA using packed signed pairs and stage-wise saturation.
#pragma once

#include "common/compiler.h"
#include "wmma/layout.h"
#include "wmma/sparse_layout.h"
#include "wmma/x86_64/swmmac_vector.h"
#if defined(__AVX512F__)
#include "common/x86_64/alu_x86_64_v4.h"
#else
#include "common/x86_64/alu_x86_64_v3.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86 {
inline U32 madd_pairs(U32 a, U32 b) {
#if defined(__AVX512F__)
  return bit_cast<U32>(_mm512_madd_epi16(bit_cast<__m512i>(a), bit_cast<__m512i>(b)));
#else
  return bit_cast<U32>(_mm256_madd_epi16(bit_cast<__m256i>(a), bit_cast<__m256i>(b)));
#endif
}

inline U32 stage_add(U32 acc, U32 sum, bool clamp) {
  U32 next = acc + sum;
  if (!clamp) {
    return next;
  }
#if defined(__AVX512F__)
  return bit_cast<U32>(
      saturate_signed_sum(bit_cast<__m512i>(acc), bit_cast<__m512i>(sum), bit_cast<__m512i>(next)));
#else
  return bit_cast<U32>(
      saturate_signed_sum(bit_cast<__m256i>(acc), bit_cast<__m256i>(sum), bit_cast<__m256i>(next)));
#endif
}

template <int Bits, int K, int Wave>
inline void compute_integer(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b, const uint32_t *const *index) {
  constexpr int Parts = 16 / Width;
  const int key = SwmmacIndex<Bits, K, Wave>::shift(mode);
  const uint32_t sign_a = mode & RJ_ARITH_WMMA_SIGNED_A ? 1u << (Bits - 1) : 0;
  const uint32_t sign_b = mode & RJ_ARITH_WMMA_SIGNED_B ? 1u << (Bits - 1) : 0;
  constexpr unsigned mask = (1u << Bits) - 1;
  alignas(64) uint32_t left[K / 4][16];
  alignas(64) uint32_t select[K / 4][16];
  U32 right[K / 4][6 * Parts];
  for (int pair = 0; pair < K / 4; ++pair) {
    int ck = 2 * pair;
    int pos = swmmac4_position<K, Wave>(ck);
    SparseField field;
    int slot;
    if constexpr (Bits == 8) {
      field = swmmac_a_field<8, Wave>(0, ck);
      slot = ck % (Wave == 64 ? 4 : 8);
    } else {
      slot = swmmac4_index_slot<K, Wave>(ck);
      field = swmmac4_a_field<K, Wave>(0, ck);
    }
    for (int row = 0; row < 16; row += Width) {
      U32 raw = load(a[field.reg] + field.lane + row) >> field.shift;
      U32 l = ((raw & mask) ^ sign_a) - sign_a, h = (((raw >> Bits) & mask) ^ sign_a) - sign_a;
      U32 v = (l & 0xffffu) | (h << 16);
      std::memcpy(left[pair] + row, &v, sizeof(v));
      U32 nibbles = (load(index[0] + field.lane + row) >> (key + 2 * slot)) & 15u;
      U32 offsets = pair_offset<1>(nibbles) * Parts;
      std::memcpy(select[pair] + row, &offsets, sizeof(offsets));
    }
    for (int p = 0; p < Parts; ++p) {
      U32 four[4];
      RJ_ARITH_UNROLL(4)
      for (int i = 0; i < 4; ++i) {
        int k = 4 * (pos / 2) + i;
        SparseField bf;
        if constexpr (Bits == 8) {
          bf = swmmac_b_field<8, Wave>(0, k);
        } else {
          bf = swmmac4_b_field<K, Wave>(0, k);
        }
        four[i] = (((load(b[bf.reg] + bf.lane + p * Width) >> bf.shift) & mask) ^ sign_b) - sign_b;
      }
      int i = 0;
      for (int hi = 1; hi < 4; ++hi) {
        for (int lo = 0; lo < hi; ++lo) {
          U32 packed = (four[lo] & 0xffffu) | (four[hi] << 16);
          right[pair][i++ * Parts + p] = packed;
        }
      }
    }
  }
  U32 result[16][Parts];
  for (int row = 0; row < 16; ++row) {
    for (int p = 0; p < Parts; ++p) {
      U32 acc = load(d[row % 8] + 16 * (row / 8) + p * Width);
      for (int stage = 0; stage < 2; ++stage) {
        U32 sum{};
        for (int local = 0; local < K / 4; local += 2) {
          int pair = wmma_gfx12_integer_stage_k(stage, local) / 2;
          U32 l = U32{} + left[pair][row];
          sum += madd_pairs(l, right[pair][select[pair][row] + p]);
        }
        acc = stage_add(acc, sum, mode & RJ_ARITH_WMMA_CLAMP);
      }
      result[row][p] = acc;
    }
  }
  // A later 32-lane destination section wins overlapping stores.
  for (int reg = 0; reg < 8; ++reg) {
    for (int half = 0; half < 2; ++half) {
      std::memcpy(d[reg] + 16 * half, result[reg + 8 * half], 16 * sizeof(uint32_t));
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86
