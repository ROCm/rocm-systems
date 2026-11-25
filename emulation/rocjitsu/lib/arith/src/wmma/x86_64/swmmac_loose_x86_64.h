// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose sparse float WMMA over the packed architectural operands.
#pragma once

#include "common/compiler.h"
#include "common/x86_64/fp8.h"
#include "wmma/sparse_layout.h"
#include "wmma/x86_64/swmmac_vector.h"
#if defined(__AVX512F__)
#include "common/x86_64/half_x86_64_v4.h"
#else
#include "common/x86_64/half_x86_64_v3.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86 {
template <int Bits, bool Bfloat> inline F32 widen(U32 x) {
  if constexpr (Bits == 8) {
#if defined(__AVX512F__)
    return widen_fp8<Bfloat>(bit_cast<__m512i>(x));
#else
    return widen_fp8<Bfloat>(bit_cast<__m256i>(x));
#endif
  } else if constexpr (Bfloat) {
    return bit_cast<F32>(x << 16);
  } else {
    // Preserve signaling encodings and avoid raising invalid on an unused B
    // element. FP16 subnormals widen exactly as fraction * 2^-24.
    U32 m = x & 0x7fffu;
    U32 sign = (x & 0x8000u) << 16;
    U32 tiny = bit_cast<U32>(__builtin_convertvector(m & 1023u, F32) * 0x1p-24f);
    U32 normal = (m << 13) + (112u << 23);
    U32 special = (m << 13) | 0x7f800000u;
    return bit_cast<F32>(sign | (m < 1024u ? tiny : m >= 0x7c00u ? special : normal));
  }
}

inline F32 mul_add(F32 lhs, F32 rhs, F32 acc) {
#if defined(__AVX512F__)
  return _mm512_fmadd_ps(lhs, rhs, acc);
#else
  return _mm256_fmadd_ps(lhs, rhs, acc);
#endif
}

inline F32 broadcast(float x) {
#if defined(__AVX512F__)
  return _mm512_set1_ps(x);
#else
  return _mm256_set1_ps(x);
#endif
}

template <int Bits, bool BfA, bool BfB, int Wave, bool Packed>
inline void compute_float(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *index, bool saturate) {
  constexpr int Parts = 16 / Width;
  alignas(64) float left[16][16];
  alignas(64) uint32_t selected[16][16];
  F32 right[32][Parts];
  const int key = SwmmacIndex<Bits, 32, Wave>::shift(mode);
  for (int ck = 0; ck < 16; ++ck) {
    const auto field = swmmac_a_field<Bits, Wave>(0, ck);
    const int slot = swmmac_index_slot<Bits, Wave>(ck);
    unsigned flip =
        Bits == 16 && (mode & (ck % 2 ? RJ_ARITH_WMMA_NEG_HI_A : RJ_ARITH_WMMA_NEG_LO_A)) ? 0x8000u
                                                                                          : 0;
    for (int row = 0; row < 16; row += Width) {
      auto x = (load(a[field.reg] + field.lane + row) >> field.shift) ^ flip;
      F32 v = widen<Bits, BfA>(x);
      std::memcpy(left[ck] + row, &v, sizeof(v));
      U32 k = 4 * (ck / 2) + ((load(index[0] + field.lane + row) >> (key + 2 * slot)) & 3u);
      std::memcpy(selected[ck] + row, &k, sizeof(k));
    }
  }
  for (int k = 0; k < 32; ++k) {
    const auto field = swmmac_b_field<Bits, Wave>(0, k);
    for (int p = 0; p < Parts; ++p) {
      right[k][p] = widen<Bits, BfB>(load(b[field.reg] + field.lane + Width * p) >> field.shift);
    }
  }
  U32 result[16][Parts];
  constexpr int Rows = Bits == 16 && Width == 16 ? 8 : 4;
  for (int tile = 0; tile < 16; tile += Rows) {
    F32 acc[Rows][Parts];
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < Rows; ++r) {
      int row = tile + r, reg = row % 8;
      for (int p = 0; p < Parts; ++p) {
        U32 c = load(d[Packed ? reg / 2 : reg] + 16 * (row / 8) + Width * p);
        if constexpr (Packed) {
          c = bit_cast<U32>(widen<16, BfA>(c >> (16 * (reg % 2))));
        }
        if constexpr (Bits == 16) {
          if (mode & RJ_ARITH_WMMA_ABS_C) {
            c &= 0x7fffffffu;
          }
          if (mode & RJ_ARITH_WMMA_NEG_C) {
            c ^= 0x80000000u;
          }
        }
        acc[r][p] = bit_cast<F32>(c);
      }
    }
    for (int ck = 0; ck < 16; ++ck) {
      unsigned sign =
          (Bits == 16 && (mode & (ck % 2 ? RJ_ARITH_WMMA_NEG_HI_B : RJ_ARITH_WMMA_NEG_LO_B)))
              ? 0x80000000u
              : 0;
      RJ_ARITH_UNROLL(8)
      for (int r = 0; r < Rows; ++r) {
        F32 scalar = broadcast(left[ck][tile + r]);
        int k = selected[ck][tile + r];
        for (int p = 0; p < Parts; ++p) {
          F32 b = bit_cast<F32>(bit_cast<U32>(right[k][p]) ^ sign);
          F32 sum = mul_add(scalar, b, acc[r][p]);
          if constexpr (Packed) {
            if (ck % 4 == 3) {
              if constexpr (BfA) {
                sum = bit_cast<F32>(bit_cast<U32>(sum) & 0xffff0000u);
              } else {
                sum = widen<16, false>(bit_cast<U32>(half_narrow(sum, saturate)));
              }
            }
          }
          acc[r][p] = sum;
        }
      }
    }
    for (int r = 0; r < Rows; ++r) {
      for (int p = 0; p < Parts; ++p) {
        U32 v = bit_cast<U32>(acc[r][p]);
        if constexpr (Packed && BfA) {
          v >>= 16;
        } else if constexpr (Packed) {
          v = bit_cast<U32>(half_narrow(acc[r][p], saturate));
        }
        result[tile + r][p] = v;
      }
    }
  }
  // Commit complete destination sections in the public dispatcher's order.
  for (int reg = 0; reg < (Packed ? 4 : 8); ++reg) {
    for (int lane = 0; lane < 32; lane += Width) {
      int row = (Packed ? 2 * reg : reg) + 8 * (lane / 16);
      int part = (lane % 16) / Width;
      U32 v = result[row][part];
      if constexpr (Packed) {
        v |= result[row + 1][part] << 16;
      }
      std::memcpy(d[reg] + lane, &v, sizeof(v));
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86
