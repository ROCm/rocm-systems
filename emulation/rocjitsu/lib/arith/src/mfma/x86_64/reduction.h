// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MFMA row reductions shared by ISA-isolated input-format backends.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include "mfma/f32_model.h"
#include "mfma/layout.h"
#include <cstring>
#include <immintrin.h>

#if defined(RJ_ARITH_TARGET_X86_64_V4)
#include "mfma/x86_64/batch4.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Operate across output columns at the width selected for this translation unit.
#if defined(RJ_ARITH_TARGET_X86_64_V4)
using MfmaNative = __m512;
inline constexpr int kMfmaWidth = 16;
inline MfmaNative mfma_load(const float *p) { return _mm512_load_ps(p); }
inline void mfma_store(float *p, MfmaNative v) { _mm512_store_ps(p, v); }
inline MfmaNative mfma_broadcast(float v) { return _mm512_set1_ps(v); }
inline MfmaNative mfma_multiply_add(MfmaNative a, MfmaNative b, MfmaNative c) {
  return _mm512_fmadd_ps(a, b, c);
}
#else
using MfmaNative = __m256;
inline constexpr int kMfmaWidth = 8;
inline MfmaNative mfma_load(const float *p) { return _mm256_load_ps(p); }
inline void mfma_store(float *p, MfmaNative v) { _mm256_store_ps(p, v); }
inline MfmaNative mfma_broadcast(float v) { return _mm256_set1_ps(v); }
inline MfmaNative mfma_multiply_add(MfmaNative a, MfmaNative b, MfmaNative c) {
  return _mm256_fmadd_ps(a, b, c);
}
#endif

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// Load one selected element from sixteen consecutive wave lanes.
template <int PerWord> __m512 mfma_lane_values(const float *source, int index) {
  if constexpr (PerWord == 1) {
    return _mm512_load_ps(source + index);
  } else {
    static_assert(PerWord == 2);
    auto base = source + (index & ~1);
    auto even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    auto indices = _mm512_add_epi32(even, _mm512_set1_epi32(index & 1));
    return _mm512_permutex2var_ps(_mm512_load_ps(base), indices, _mm512_load_ps(base + 16));
  }
}

// Reduce four independent 4x4 matrices per vector, in guest K order.
template <int K, int PerWord, int ABlockLanes = 0>
void mfma_reduce_batch4(uint64_t controls, float (*result)[64], const float *left,
                        const float *right) {
  for (int lane = 0; lane < 64; lane += 16) {
    int first_a = mfma_a_lane(lane, controls, ABlockLanes), first_b = mfma_b_lane(lane, controls);
    auto groups = mfma_batch4_groups(lane, first_a, controls, ABlockLanes);
    __m512 acc[4];
    for (int row = 0; row < 4; ++row) {
      acc[row] = _mm512_load_ps(result[row] + lane);
    }
    RJ_ARITH_UNROLL(4)
    for (int k = 0; k < K; ++k) {
      int word = k / PerWord * 64 * PerWord + k % PerWord;
      auto a = mfma_lane_values<PerWord>(left, word + (first_a & ~15) * PerWord);
      auto b = mfma_lane_values<PerWord>(right, word + first_b * PerWord);
      RJ_ARITH_UNROLL(4)
      for (int row = 0; row < 4; ++row) {
        auto x = _mm512_permutexvar_ps(_mm512_add_epi32(groups, _mm512_set1_epi32(row)), a);
        acc[row] = _mm512_fmadd_ps(x, b, acc[row]);
      }
    }
    for (int row = 0; row < 4; ++row) {
      _mm512_store_ps(result[row] + lane, acc[row]);
    }
  }
}

// A changes lane within its broadcast block. B preserves the low four lane
// bits, so each group of 16 columns is one load (two for interleaved halves).
template <int M, int K, int Batch, int PerWord, int ABlockLanes = 0>
void mfma_reduce_columns(uint64_t controls, float (*result)[64], const float *left,
                         const float *right) {
  int cbsz = int(controls & 7);
  // gfx90a broadcasts whole matrix blocks; row offsets remain within that block.
  int row_mask = !ABlockLanes && cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
  for (int batch = 0; batch < Batch; ++batch) {
    const float *rows[K];
    __m512 columns[K][M / 16];
    for (int k = 0; k < K; ++k) {
      int i = mfma_input_index<M, K, Batch, PerWord>(0, k, batch);
      rows[k] = left + mfma_permute_input<PerWord, ABlockLanes>(i, controls, true);
      for (int part = 0; part < M / 16; ++part) {
        int j = mfma_permute_input<PerWord, ABlockLanes>(i + 16 * part * PerWord, controls, false);
        columns[k][part] = mfma_lane_values<PerWord>(right, j);
      }
    }
    for (int row = 0; row < M; ++row) {
      auto first = mfma_output<M, Batch>(row, 0, batch);
      float *out = result[first.reg] + first.lane;
      __m512 acc[M / 16];
      for (int part = 0; part < M / 16; ++part) {
        acc[part] = _mm512_load_ps(out + 16 * part);
      }
      RJ_ARITH_UNROLL(32)
      for (int k = 0; k < K; ++k) {
        auto x = _mm512_set1_ps(rows[k][(row & row_mask) * PerWord]);
        for (int part = 0; part < M / 16; ++part) {
          acc[part] = _mm512_fmadd_ps(x, columns[k][part], acc[part]);
        }
      }
      for (int part = 0; part < M / 16; ++part) {
        _mm512_store_ps(out + 16 * part, acc[part]);
      }
    }
  }
}
#endif

// Flush each FP32 subnormal to signed zero; normal, infinite and NaN bits pass through.
inline __m256i mfma_flush_f32x8(__m256i bits) {
  auto zero_exponent = _mm256_cmpeq_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000)),
                                          _mm256_setzero_si256());
  auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(int(0x80000000U)));
  return _mm256_blendv_epi8(bits, sign, zero_exponent);
}

// Flush one staged physical register, retaining its alignment and signed zeros.
inline void mfma_flush_register(float *values) {
  RJ_ARITH_UNROLL(8)
  for (int lane = 0; lane < 64; lane += kMfmaWidth) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    auto bits = _mm512_load_si512(values + lane);
    auto zero_exponent = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(bits, _mm512_set1_epi32(0x7f800000)), _mm512_setzero_si512());
    auto sign = _mm512_and_si512(bits, _mm512_set1_epi32(int(0x80000000U)));
    _mm512_store_si512(values + lane, _mm512_mask_mov_epi32(bits, zero_exponent, sign));
#else
    auto bits = _mm256_load_si256(reinterpret_cast<const __m256i *>(values + lane));
    _mm256_store_si256(reinterpret_cast<__m256i *>(values + lane), mfma_flush_f32x8(bits));
#endif
  }
}

// Reduce decoded packed sources with FMA, staging all outputs to preserve overlap.
template <int M, int K, int Batch, int PerWord, MfmaF32Model Model = MfmaF32Model::Default>
void mfma_reduce_simd(uint64_t controls, uint32_t *const *d, const float *left, const float *right,
                      const uint32_t *const *c, bool clamp_overflow = false,
                      uint64_t fp_flags = 0) {
  // Each matrix row is a contiguous slice of one physical accumulator. Stage it
  // in register order so source aliases remain valid through overflow checking.
  constexpr int ABlockLanes = Model == MfmaF32Model::Gfx90a ? M : 0;
  constexpr int registers = M * M * Batch / 64;
  alignas(64) float result[registers][64];
  for (int reg = 0; reg < registers; ++reg) {
    std::memcpy(result[reg], c[reg], sizeof(result[reg]));
    if constexpr (Model == MfmaF32Model::Gfx90a) {
      if (fp_flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        mfma_flush_register(result[reg]);
      }
    }
  }
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  if constexpr (M == 4 && Batch == 16 && (PerWord == 1 || PerWord == 2)) {
    mfma_reduce_batch4<K, PerWord, ABlockLanes>(controls, result, left, right);
  } else if constexpr (M >= 16 && (PerWord == 1 || PerWord == 2) &&
                       (ABlockLanes == 0 || Batch == 1 || PerWord == 2)) {
    // Small-K batched gfx90a FP32 does not amortize column staging; keep its general loop.
    mfma_reduce_columns<M, K, Batch, PerWord, ABlockLanes>(controls, result, left, right);
  } else
#endif
    for (int batch = 0; batch < Batch; ++batch) {
      alignas(64) float rows[M][K], columns[K][M];
      for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
          int index = mfma_input_index<M, K, Batch, PerWord>(i, k, batch);
          if (!controls) {
            rows[i][k] = left[index];
            columns[k][i] = right[index];
          } else {
            rows[i][k] = left[mfma_permute_input<PerWord, ABlockLanes>(index, controls, true)];
            columns[k][i] = right[mfma_permute_input<PerWord, ABlockLanes>(index, controls, false)];
          }
        }
      }
      for (int row = 0; row < M; ++row) {
        const auto first = mfma_output<M, Batch>(row, 0, batch);
        float *output = result[first.reg] + first.lane;
        if constexpr (M >= kMfmaWidth) {
          // Independent output columns share each A broadcast without serializing
          // their reduction chains. Each accumulator still visits K in order.
          // At most four vector accumulators: keep each at a constant register index.
          MfmaNative acc[M / kMfmaWidth];
          RJ_ARITH_UNROLL(4)
          for (int part = 0; part < M / kMfmaWidth; ++part) {
            acc[part] = mfma_load(output + part * kMfmaWidth);
          }
          RJ_ARITH_UNROLL(128)
          for (int k = 0; k < K; ++k) {
            auto x = mfma_broadcast(rows[row][k]);
            RJ_ARITH_UNROLL(4)
            for (int part = 0; part < M / kMfmaWidth; ++part) {
              acc[part] =
                  mfma_multiply_add(x, mfma_load(&columns[k][part * kMfmaWidth]), acc[part]);
            }
          }
          RJ_ARITH_UNROLL(4)
          for (int part = 0; part < M / kMfmaWidth; ++part) {
            mfma_store(output + part * kMfmaWidth, acc[part]);
          }
        } else {
          // Four-column batched forms fit one XMM vector rather than a full YMM/ZMM.
          __m128 acc = _mm_load_ps(output);
          RJ_ARITH_UNROLL(128)
          for (int k = 0; k < K; ++k) {
            acc = _mm_fmadd_ps(_mm_set1_ps(rows[row][k]), _mm_load_ps(columns[k]), acc);
          }
          _mm_store_ps(output, acc);
        }
      }
    }
  if constexpr (Model == MfmaF32Model::Default) {
    if (clamp_overflow) {
      for (int batch = 0; batch < Batch; ++batch) {
        for (int row = 0; row < M; ++row) {
          const auto first = mfma_output<M, Batch>(row, 0, batch);
          for (int col = 0; col < M; ++col) {
            result[first.reg][first.lane + col] = mfma_clamp_overflow<M, K, Batch, PerWord>(
                result[first.reg][first.lane + col], as_float(c[first.reg][first.lane + col]), left,
                right, row, col, batch, controls);
          }
        }
      }
    }
  }
  for (int reg = 0; reg < registers; ++reg) {
    if constexpr (Model == MfmaF32Model::Gfx90a) {
      if (fp_flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        mfma_flush_register(result[reg]);
      }
    }
    std::memcpy(d[reg], result[reg], sizeof(result[reg]));
  }
}

// Check whether writing each disjoint 4-VGPR output tile directly to d cannot overwrite any
// unread c VGPR. Assumes a/b have already been read/staged before any d stores.
template <int Registers>
inline bool mfma_direct_store_safe(uint32_t *const *d, const uint32_t *const *c) {
  constexpr uintptr_t last_byte = 64 * sizeof(uint32_t) - 1;
  uintptr_t d_min = reinterpret_cast<uintptr_t>(d[0]);
  uintptr_t d_max = d_min;
  uintptr_t c_min = reinterpret_cast<uintptr_t>(c[0]);
  uintptr_t c_max = c_min;
  bool all_eq = (d[0] == c[0]);
  bool strictly_inc = true;
  for (int i = 1; i < Registers; ++i) {
    const uintptr_t di = reinterpret_cast<uintptr_t>(d[i]);
    const uintptr_t ci = reinterpret_cast<uintptr_t>(c[i]);
    const uintptr_t c_prev = reinterpret_cast<uintptr_t>(c[i - 1]);
    if (di < d_min) {
      d_min = di;
    }
    if (di > d_max) {
      d_max = di;
    }
    if (ci < c_min) {
      c_min = ci;
    }
    if (ci > c_max) {
      c_max = ci;
    }
    all_eq &= (di == ci);
    strictly_inc &= (ci > c_prev) && (ci - c_prev > last_byte);
  }
  const bool d_before_c = (d_max < c_min) && (c_min - d_max > last_byte);
  const bool c_before_d = (c_max < d_min) && (d_min - c_max > last_byte);
  return d_before_c || c_before_d || (all_eq && strictly_inc);
}

// A cheaper, stricter form of mfma_direct_store_safe for reductions short enough that scanning
// 16 or 32 register pointers one at a time shows up in their time. Require non-overlapping,
// ascending VGPR ranges in both lists. D must be tied to C or separate from it.
template <int Registers>
inline bool mfma_ascending_store_safe(uint32_t *const *d, const uint32_t *const *c) {
  static_assert(sizeof(uintptr_t) == 8);
  constexpr uintptr_t last_byte = 64 * sizeof(uint32_t) - 1;
  const uintptr_t d_first = reinterpret_cast<uintptr_t>(d[0]);
  const uintptr_t d_last = reinterpret_cast<uintptr_t>(d[Registers - 1]);
  const uintptr_t c_first = reinterpret_cast<uintptr_t>(c[0]);
  const uintptr_t c_last = reinterpret_cast<uintptr_t>(c[Registers - 1]);
  const bool apart = d_last + last_byte < c_first || c_last + last_byte < d_first;
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  if constexpr (Registers >= 16) {
    const __m512i extent = _mm512_set1_epi64(last_byte);
    __mmask8 tied = 0xff, ascending = 0xff;
    for (int i = 0; i < Registers; i += 8) {
      // The final block steps back so each adjacent pair stays inside the lists.
      const int pair = i < Registers - 9 ? i : Registers - 9;
      tied &= _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(d + i), _mm512_loadu_si512(c + i));
      ascending &= _mm512_cmpgt_epu64_mask(_mm512_loadu_si512(d + pair + 1),
                                           _mm512_add_epi64(_mm512_loadu_si512(d + pair), extent));
      ascending &= _mm512_cmpgt_epu64_mask(_mm512_loadu_si512(c + pair + 1),
                                           _mm512_add_epi64(_mm512_loadu_si512(c + pair), extent));
    }
    return ascending == 0xff && (tied == 0xff || apart);
  }
#endif
  if constexpr (Registers < 8) {
    bool tied = d_first == c_first, ascending = true;
    for (int i = 1; i < Registers; ++i) {
      tied &= d[i] == c[i];
      ascending &=
          reinterpret_cast<uintptr_t>(d[i]) > reinterpret_cast<uintptr_t>(d[i - 1]) + last_byte;
      ascending &=
          reinterpret_cast<uintptr_t>(c[i]) > reinterpret_cast<uintptr_t>(c[i - 1]) + last_byte;
    }
    return ascending && (tied || apart);
  }
  auto pointers = [](const void *p) { return _mm256_loadu_si256(static_cast<const __m256i *>(p)); };
  // Bias unsigned addresses into signed order for AVX2's 64-bit comparison.
  const __m256i sign = _mm256_set1_epi64x(INT64_MIN);
  const __m256i biased_extent = _mm256_set1_epi64x(INT64_MIN + static_cast<int64_t>(last_byte));
  auto separated = [&](const void *before, const void *after) {
    return _mm256_cmpgt_epi64(_mm256_xor_si256(pointers(after), sign),
                              _mm256_add_epi64(pointers(before), biased_extent));
  };
  __m256i tied = _mm256_set1_epi64x(-1), ascending = tied;
  for (int i = 0; i < Registers; i += 4) {
    const int pair = i < Registers - 5 ? i : Registers - 5;
    tied = _mm256_and_si256(tied, _mm256_cmpeq_epi64(pointers(d + i), pointers(c + i)));
    ascending = _mm256_and_si256(ascending, separated(d + pair, d + pair + 1));
    ascending = _mm256_and_si256(ascending, separated(c + pair, c + pair + 1));
  }
  return _mm256_movemask_pd(_mm256_castsi256_pd(ascending)) == 0xf &&
         (_mm256_movemask_pd(_mm256_castsi256_pd(tied)) == 0xf || apart);
}

// Cold finite-input overflow clamp over deinterleaved [plane][64] decoded sources.
template <int M, int K, int Batch>
[[gnu::noinline]] void mfma_clamp_overflow_planes(float (*result)[64], const float (*left)[64],
                                                  const float (*right)[64],
                                                  const uint32_t *const *c, uint64_t controls) {
  constexpr int per_lane = K / (64 / (M * Batch));
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  if constexpr (M >= 16) {
    const __m512i magnitude_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512i infinity = _mm512_set1_epi32(0x7f800000);
    const __m512i one = _mm512_set1_epi32(1);
    for (int batch = 0; batch < Batch; ++batch) {
      for (int row = 0; row < M; ++row) {
        const MfmaWord first = mfma_output<M, Batch>(row, 0, batch);
        for (int col = 0; col < M; col += 16) {
          float *out = result[first.reg] + first.lane + col;
          __m512i bits = _mm512_load_si512(out);
          __m512i initial = _mm512_loadu_si512(c[first.reg] + first.lane + col);
          int clip = _mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, magnitude_mask), infinity) &
                     _mm512_cmplt_epi32_mask(_mm512_and_si512(initial, magnitude_mask), infinity);
          for (int k = 0; k < K && clip; ++k) {
            const int plane = k % per_lane;
            const int base = batch * M + (k / per_lane) * M * Batch;
            const int a_lane = mfma_a_lane(base + row, controls);
            const int b_lane = mfma_b_lane(base + col, controls);
            if (!__builtin_isfinite(left[plane][a_lane])) {
              clip = 0;
              break;
            }
            const __m512i b = _mm512_load_si512(right[plane] + b_lane);
            clip &= _mm512_cmplt_epi32_mask(_mm512_and_si512(b, magnitude_mask), infinity);
          }
          _mm512_mask_store_epi32(out, clip, _mm512_sub_epi32(bits, one));
        }
      }
    }
    return;
  }
#endif
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      const MfmaWord first = mfma_output<M, Batch>(row, 0, batch);
      for (int col = 0; col < M; ++col) {
        float val = result[first.reg][first.lane + col];
        float initial = as_float(c[first.reg][first.lane + col]);
        if (!__builtin_isinf(val) || !__builtin_isfinite(initial)) {
          continue;
        }
        bool all_finite = true;
        for (int k = 0; k < K; ++k) {
          int plane = k % per_lane;
          int base = batch * M + (k / per_lane) * M * Batch;
          int la = mfma_a_lane(base + row, controls);
          int lb = mfma_b_lane(base + col, controls);
          if (!__builtin_isfinite(left[plane][la]) || !__builtin_isfinite(right[plane][lb])) {
            all_finite = false;
            break;
          }
        }
        if (all_finite) {
          result[first.reg][first.lane + col] = bit_cast<float>(bit_cast<uint32_t>(val) - 1u);
        }
      }
    }
  }
}

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// Reduce four independent 4x4 matrices per vector across deinterleaved [plane][64] sources.
template <int K>
void mfma_reduce_planes_batch4(uint64_t controls, uint32_t *const *d, const float (*left)[64],
                               const float (*right)[64], const uint32_t *const *c,
                               bool clamp_overflow) {
  __m512 acc[4][4];
  for (int g = 0; g < 4; ++g) {
    for (int row = 0; row < 4; ++row) {
      acc[g][row] = _mm512_loadu_ps(reinterpret_cast<const float *>(c[row] + 16 * g));
    }
  }
  const __m512i row_offsets0 = _mm512_set1_epi32(0);
  const __m512i row_offsets1 = _mm512_set1_epi32(1);
  const __m512i row_offsets2 = _mm512_set1_epi32(2);
  const __m512i row_offsets3 = _mm512_set1_epi32(3);
  for (int g = 0; g < 4; ++g) {
    const int lane = 16 * g;
    const int first_a = mfma_a_lane(lane, controls);
    const int first_b = mfma_b_lane(lane, controls);
    const __m512i groups = mfma_batch4_groups(lane, first_a, controls);
    const __m512i idx0 = _mm512_add_epi32(groups, row_offsets0);
    const __m512i idx1 = _mm512_add_epi32(groups, row_offsets1);
    const __m512i idx2 = _mm512_add_epi32(groups, row_offsets2);
    const __m512i idx3 = _mm512_add_epi32(groups, row_offsets3);
    const int a_chunk = first_a & ~15;
    RJ_ARITH_UNROLL(4)
    for (int k = 0; k < K; ++k) {
      const __m512 a_vec = _mm512_load_ps(left[k] + a_chunk);
      const __m512 b_vec = _mm512_load_ps(right[k] + first_b);
      acc[g][0] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx0, a_vec), b_vec, acc[g][0]);
      acc[g][1] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx1, a_vec), b_vec, acc[g][1]);
      acc[g][2] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx2, a_vec), b_vec, acc[g][2]);
      acc[g][3] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx3, a_vec), b_vec, acc[g][3]);
    }
  }
  if (clamp_overflow) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512i inf_bits = _mm512_set1_epi32(0x7f800000);
    __mmask16 any_inf = 0;
    for (int g = 0; g < 4; ++g) {
      for (int row = 0; row < 4; ++row) {
        any_inf |= _mm512_cmpeq_epi32_mask(
            _mm512_and_si512(_mm512_castps_si512(acc[g][row]), abs_mask), inf_bits);
      }
    }
    if (__builtin_expect(any_inf != 0, 0)) {
      alignas(64) float result[4][64];
      for (int g = 0; g < 4; ++g) {
        for (int row = 0; row < 4; ++row) {
          _mm512_store_ps(result[row] + 16 * g, acc[g][row]);
        }
      }
      mfma_clamp_overflow_planes<4, K, 16>(result, left, right, c, controls);
      for (int row = 0; row < 4; ++row) {
        std::memcpy(d[row], result[row], sizeof(result[row]));
      }
      return;
    }
  }
  for (int row = 0; row < 4; ++row) {
    for (int g = 0; g < 4; ++g) {
      _mm512_storeu_ps(reinterpret_cast<float *>(d[row] + 16 * g), acc[g][row]);
    }
  }
}

// Reduce M=16, Batch=1 directly in 16 ZMM accumulators (zero result stack buffer).
template <int K>
void mfma_reduce_planes_16x16(uint64_t controls, uint32_t *const *d, const float (*left)[64],
                              const float (*right)[64], const uint32_t *const *c,
                              bool clamp_overflow) {
  constexpr int per_lane = K / 4;
  __m512 acc[16];
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    acc[row] = _mm512_loadu_ps(reinterpret_cast<const float *>(c[row % 4] + (row / 4) * 16));
  }
  if (__builtin_expect(controls == 0, 1)) {
    for (int g = 0; g < 4; ++g) {
      const int base = g * 16;
      RJ_ARITH_UNROLL(4)
      for (int local = 0; local < per_lane; ++local) {
        const float *a_ptr = left[local] + base;
        const __m512 b_vec = _mm512_load_ps(right[local] + base);
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row]), b_vec, acc[row]);
        }
      }
    }
  } else {
    const int cbsz = int(controls & 7);
    const int row_mask = cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
    for (int g = 0; g < 4; ++g) {
      const int base = g * 16;
      const int a_base = mfma_a_lane(base, controls);
      const int b_base = mfma_b_lane(base, controls);
      for (int local = 0; local < per_lane; ++local) {
        const float *a_ptr = left[local] + a_base;
        const __m512 b_vec = _mm512_load_ps(right[local] + b_base);
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row & row_mask]), b_vec, acc[row]);
        }
      }
    }
  }
  if (clamp_overflow) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512i inf_bits = _mm512_set1_epi32(0x7f800000);
    __mmask16 any_inf = 0;
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      any_inf |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(acc[row]), abs_mask),
                                         inf_bits);
    }
    if (__builtin_expect(any_inf != 0, 0)) {
      alignas(64) float result[4][64];
      for (int row = 0; row < 16; ++row) {
        _mm512_store_ps(result[row % 4] + (row / 4) * 16, acc[row]);
      }
      mfma_clamp_overflow_planes<16, K, 1>(result, left, right, c, controls);
      for (int reg = 0; reg < 4; ++reg) {
        std::memcpy(d[reg], result[reg], sizeof(result[reg]));
      }
      return;
    }
  }
  RJ_ARITH_UNROLL(16)
  for (int row = 0; row < 16; ++row) {
    _mm512_storeu_ps(reinterpret_cast<float *>(d[row % 4] + (row / 4) * 16), acc[row]);
  }
}

// Keep the four B planes for both column vectors live across all 32 rows.
inline void mfma_reduce_planes_32x32x4_2b(uint64_t controls, uint32_t *const *d,
                                          const float (*left)[64], const float (*right)[64],
                                          const uint32_t *const *c, bool clamp_overflow) {
  alignas(64) float result[32][64];
  const int cbsz = int(controls & 7);
  const int row_mask = cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
  for (int batch = 0; batch < 2; ++batch) {
    const int base = batch * 32;
    const int a_lane = mfma_a_lane(base, controls);
    const int b_lane_lo = mfma_b_lane(base, controls);
    const int b_lane_hi = mfma_b_lane(base + 16, controls);
    __m512 b_lo[4], b_hi[4];
    RJ_ARITH_UNROLL(4)
    for (int k = 0; k < 4; ++k) {
      b_lo[k] = _mm512_load_ps(right[k] + b_lane_lo);
      b_hi[k] = _mm512_load_ps(right[k] + b_lane_hi);
    }
    for (int row = 0; row < 32; ++row) {
      const MfmaWord out = mfma_output<32, 2>(row, 0, batch);
      const float *initial = reinterpret_cast<const float *>(c[out.reg] + out.lane);
      __m512 acc_lo = _mm512_loadu_ps(initial);
      __m512 acc_hi = _mm512_loadu_ps(initial + 16);
      RJ_ARITH_UNROLL(4)
      for (int k = 0; k < 4; ++k) {
        const __m512 x = _mm512_set1_ps(left[k][a_lane + (row & row_mask)]);
        acc_lo = _mm512_fmadd_ps(x, b_lo[k], acc_lo);
        acc_hi = _mm512_fmadd_ps(x, b_hi[k], acc_hi);
      }
      _mm512_store_ps(result[out.reg] + out.lane, acc_lo);
      _mm512_store_ps(result[out.reg] + out.lane + 16, acc_hi);
    }
  }
  if (clamp_overflow) {
    mfma_clamp_overflow_planes<32, 4, 2>(result, left, right, c, controls);
  }
  for (int reg = 0; reg < 32; ++reg) {
    std::memcpy(d[reg], result[reg], sizeof(result[reg]));
  }
}

// Reduce M>=16 general shapes (32x32xK, 16x16x4_4b) with 8/16-row blocking.
template <int M, int K, int Batch>
void mfma_reduce_planes_tiled_v4(uint64_t controls, uint32_t *const *d, const float (*left)[64],
                                 const float (*right)[64], const uint32_t *const *c,
                                 bool clamp_overflow) {
  constexpr int registers = M * M * Batch / 64;
  constexpr int per_lane = K / (64 / (M * Batch));
  constexpr int groups = 64 / (M * Batch);
  constexpr int row_tile = M == 16 ? 16 : 8;
  constexpr int col_vecs = M / 16;
  const bool direct = !clamp_overflow && mfma_ascending_store_safe<registers>(d, c);
  alignas(64) float result[registers][64];
  const int cbsz = int(controls & 7);
  const int row_mask = cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  const __m512i inf_bits = _mm512_set1_epi32(0x7f800000);
  __mmask16 any_inf = 0;
  for (int batch = 0; batch < Batch; ++batch) {
    for (int rb = 0; rb < M; rb += row_tile) {
      __m512 acc[row_tile][col_vecs];
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, Batch>(rb + r, 0, batch);
        for (int p = 0; p < col_vecs; ++p) {
          acc[r][p] =
              _mm512_loadu_ps(reinterpret_cast<const float *>(c[out.reg] + out.lane + 16 * p));
        }
      }
      if (__builtin_expect(controls == 0, 1)) {
        for (int g = 0; g < groups; ++g) {
          const int base = batch * M + g * M * Batch;
          RJ_ARITH_UNROLL(4)
          for (int local = 0; local < per_lane; ++local) {
            const float *a_ptr = left[local] + base + rb;
            __m512 b_vec[col_vecs];
            for (int p = 0; p < col_vecs; ++p) {
              b_vec[p] = _mm512_load_ps(right[local] + base + 16 * p);
            }
            RJ_ARITH_UNROLL(16)
            for (int r = 0; r < row_tile; ++r) {
              const __m512 x = _mm512_set1_ps(a_ptr[r]);
              for (int p = 0; p < col_vecs; ++p) {
                acc[r][p] = _mm512_fmadd_ps(x, b_vec[p], acc[r][p]);
              }
            }
          }
        }
      } else {
        for (int g = 0; g < groups; ++g) {
          const int base = batch * M + g * M * Batch;
          const int a_base = mfma_a_lane(base, controls);
          int b_base[col_vecs];
          for (int p = 0; p < col_vecs; ++p) {
            b_base[p] = mfma_b_lane(base + 16 * p, controls);
          }
          for (int local = 0; local < per_lane; ++local) {
            const float *a_ptr = left[local] + a_base;
            __m512 b_vec[col_vecs];
            for (int p = 0; p < col_vecs; ++p) {
              b_vec[p] = _mm512_load_ps(right[local] + b_base[p]);
            }
            RJ_ARITH_UNROLL(16)
            for (int r = 0; r < row_tile; ++r) {
              const __m512 x = _mm512_set1_ps(a_ptr[(rb + r) & row_mask]);
              for (int p = 0; p < col_vecs; ++p) {
                acc[r][p] = _mm512_fmadd_ps(x, b_vec[p], acc[r][p]);
              }
            }
          }
        }
      }
      if (__builtin_expect(direct, 1)) {
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < row_tile; ++r) {
          const MfmaWord out = mfma_output<M, Batch>(rb + r, 0, batch);
          for (int p = 0; p < col_vecs; ++p) {
            _mm512_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 16 * p), acc[r][p]);
          }
        }
      } else {
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < row_tile; ++r) {
          const MfmaWord out = mfma_output<M, Batch>(rb + r, 0, batch);
          for (int p = 0; p < col_vecs; ++p) {
            if (clamp_overflow) {
              any_inf |= _mm512_cmpeq_epi32_mask(
                  _mm512_and_si512(_mm512_castps_si512(acc[r][p]), abs_mask), inf_bits);
            }
            _mm512_store_ps(result[out.reg] + out.lane + 16 * p, acc[r][p]);
          }
        }
      }
    }
  }
  if (__builtin_expect(direct, 1)) {
    return;
  }
  if (__builtin_expect(clamp_overflow && any_inf != 0, 0)) {
    mfma_clamp_overflow_planes<M, K, Batch>(result, left, right, c, controls);
  }
  for (int reg = 0; reg < registers; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_ps(reinterpret_cast<float *>(d[reg] + lane),
                       _mm512_load_ps(result[reg] + lane));
    }
  }
}
#else
// AVX2 reduction for 4x4xK_16b: process two 4x4 batches (8 lanes) per YMM vector.
template <int K>
void mfma_reduce_planes_batch4_v3(uint64_t controls, uint32_t *const *d, const float (*left)[64],
                                  const float (*right)[64], const uint32_t *const *c,
                                  bool clamp_overflow) {
  alignas(32) float result[4][64];
  const __m256i abs_mask = _mm256_set1_epi32(0x7fffffff);
  const __m256i inf_bits = _mm256_set1_epi32(0x7f800000);
  __m256i any_inf = _mm256_setzero_si256();
  for (int lane = 0; lane < 64; lane += 8) {
    __m256 acc[4];
    for (int row = 0; row < 4; ++row) {
      acc[row] = _mm256_loadu_ps(reinterpret_cast<const float *>(c[row] + lane));
    }
    const int b_lane = mfma_b_lane(lane, controls);
    int a0[4], a1[4];
    for (int row = 0; row < 4; ++row) {
      a0[row] = mfma_a_lane(lane + row, controls);
      a1[row] = mfma_a_lane(lane + 4 + row, controls);
    }
    RJ_ARITH_UNROLL(4)
    for (int k = 0; k < K; ++k) {
      const __m256 b_vec = _mm256_load_ps(right[k] + b_lane);
      RJ_ARITH_UNROLL(4)
      for (int row = 0; row < 4; ++row) {
        const __m128 lo = _mm_broadcast_ss(left[k] + a0[row]);
        const __m128 hi = _mm_broadcast_ss(left[k] + a1[row]);
        const __m256 a_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
        acc[row] = _mm256_fmadd_ps(a_vec, b_vec, acc[row]);
      }
    }
    for (int row = 0; row < 4; ++row) {
      if (clamp_overflow) {
        any_inf = _mm256_or_si256(
            any_inf, _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(acc[row]), abs_mask),
                                        inf_bits));
      }
      _mm256_store_ps(result[row] + lane, acc[row]);
    }
  }
  if (__builtin_expect(clamp_overflow && _mm256_movemask_epi8(any_inf) != 0, 0)) {
    mfma_clamp_overflow_planes<4, K, 16>(result, left, right, c, controls);
  }
  for (int row = 0; row < 4; ++row) {
    std::memcpy(d[row], result[row], sizeof(result[row]));
  }
}

// AVX2 4x16 register-blocked reduction (8 YMM accumulators, zero spills) for M>=16.
template <int M, int K, int Batch>
void mfma_reduce_planes_tiled_v3(uint64_t controls, uint32_t *const *d, const float (*left)[64],
                                 const float (*right)[64], const uint32_t *const *c,
                                 bool clamp_overflow) {
  constexpr int registers = M * M * Batch / 64;
  constexpr int per_lane = K / (64 / (M * Batch));
  constexpr int groups = 64 / (M * Batch);
  const bool direct = !clamp_overflow && mfma_ascending_store_safe<registers>(d, c);
  alignas(32) float result[registers][64];
  const int cbsz = int(controls & 7);
  const int row_mask = cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
  const __m256i abs_mask = _mm256_set1_epi32(0x7fffffff);
  const __m256i inf_bits = _mm256_set1_epi32(0x7f800000);
  __m256i any_inf = _mm256_setzero_si256();
  for (int batch = 0; batch < Batch; ++batch) {
    for (int col_base = 0; col_base < M; col_base += 16) {
      for (int rb = 0; rb < M; rb += 4) {
        __m256 acc[4][2];
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          const MfmaWord out = mfma_output<M, Batch>(rb + r, col_base, batch);
          acc[r][0] = _mm256_loadu_ps(reinterpret_cast<const float *>(c[out.reg] + out.lane));
          acc[r][1] = _mm256_loadu_ps(reinterpret_cast<const float *>(c[out.reg] + out.lane + 8));
        }
        if (__builtin_expect(controls == 0, 1)) {
          for (int g = 0; g < groups; ++g) {
            const int base = batch * M + g * M * Batch;
            RJ_ARITH_UNROLL(4)
            for (int local = 0; local < per_lane; ++local) {
              const float *a_ptr = left[local] + base + rb;
              const __m256 b0 = _mm256_load_ps(right[local] + base + col_base);
              const __m256 b1 = _mm256_load_ps(right[local] + base + col_base + 8);
              RJ_ARITH_UNROLL(4)
              for (int r = 0; r < 4; ++r) {
                const __m256 x = _mm256_broadcast_ss(a_ptr + r);
                acc[r][0] = _mm256_fmadd_ps(x, b0, acc[r][0]);
                acc[r][1] = _mm256_fmadd_ps(x, b1, acc[r][1]);
              }
            }
          }
        } else {
          for (int g = 0; g < groups; ++g) {
            const int base = batch * M + g * M * Batch;
            const int a_base = mfma_a_lane(base, controls);
            const int b_base = mfma_b_lane(base + col_base, controls);
            for (int local = 0; local < per_lane; ++local) {
              const float *a_ptr = left[local] + a_base;
              const __m256 b0 = _mm256_load_ps(right[local] + b_base);
              const __m256 b1 = _mm256_load_ps(right[local] + b_base + 8);
              RJ_ARITH_UNROLL(4)
              for (int r = 0; r < 4; ++r) {
                const __m256 x = _mm256_broadcast_ss(a_ptr + ((rb + r) & row_mask));
                acc[r][0] = _mm256_fmadd_ps(x, b0, acc[r][0]);
                acc[r][1] = _mm256_fmadd_ps(x, b1, acc[r][1]);
              }
            }
          }
        }
        if (__builtin_expect(direct, 1)) {
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            const MfmaWord out = mfma_output<M, Batch>(rb + r, col_base, batch);
            _mm256_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane), acc[r][0]);
            _mm256_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 8), acc[r][1]);
          }
        } else {
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            const MfmaWord out = mfma_output<M, Batch>(rb + r, col_base, batch);
            if (clamp_overflow) {
              any_inf = _mm256_or_si256(
                  any_inf,
                  _mm256_or_si256(
                      _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(acc[r][0]), abs_mask),
                                         inf_bits),
                      _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(acc[r][1]), abs_mask),
                                         inf_bits)));
            }
            _mm256_store_ps(result[out.reg] + out.lane, acc[r][0]);
            _mm256_store_ps(result[out.reg] + out.lane + 8, acc[r][1]);
          }
        }
      }
    }
  }
  if (__builtin_expect(direct, 1)) {
    return;
  }
  if (__builtin_expect(clamp_overflow && _mm256_movemask_epi8(any_inf) != 0, 0)) {
    mfma_clamp_overflow_planes<M, K, Batch>(result, left, right, c, controls);
  }
  for (int reg = 0; reg < registers; ++reg) {
    std::memcpy(d[reg], result[reg], sizeof(result[reg]));
  }
}
#endif

// A small K needs only a few live B vectors. Reduce a contiguous column slice
// across the physical accumulator layout before moving to the next slice.
template <int K, int Batch>
void mfma_reduce_planes_32x32_small_k(uint64_t controls, uint32_t *const *d,
                                      const float (*left)[64], const float (*right)[64],
                                      const uint32_t *const *c, bool clamp_overflow) {
  constexpr int rows = 32;
  constexpr int regs_per_batch = 16;
  constexpr int groups = 64 / (rows * Batch);
  constexpr int per_lane = K / groups;
  constexpr int columns = K <= 2 ? rows : (kMfmaWidth < 16 ? 16 : kMfmaWidth);
  constexpr int chunks = columns / kMfmaWidth;
  alignas(64) float result[regs_per_batch * Batch][64];
  const int cbsz = static_cast<int>(controls & 7);
  const int row_mask = cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
  for (int batch = 0; batch < Batch; ++batch) {
    int a_lane[groups];
    for (int g = 0; g < groups; ++g) {
      a_lane[g] = mfma_a_lane(rows * (batch + g * Batch), controls);
    }
    for (int col = 0; col < rows; col += columns) {
      MfmaNative b_vec[K][chunks];
      for (int k = 0; k < K; ++k) {
        for (int part = 0; part < chunks; ++part) {
          const int lane = rows * (batch + (k / per_lane) * Batch) + col + kMfmaWidth * part;
          b_vec[k][part] = mfma_load(right[k % per_lane] + mfma_b_lane(lane, controls));
        }
      }
      for (int reg = 0; reg < regs_per_batch; ++reg) {
        const int out_reg = batch * regs_per_batch + reg;
        const int first_row = 8 * (reg / 4) + reg % 4;
        RJ_ARITH_UNROLL(2)
        for (int half = 0; half < 2; ++half) {
          const int row = (first_row + 4 * half) & row_mask;
          if constexpr (K <= 2) {
            MfmaNative a_vec[K];
            RJ_ARITH_UNROLL(4)
            for (int k = 0; k < K; ++k) {
              a_vec[k] = mfma_broadcast(left[k % per_lane][a_lane[k / per_lane] + row]);
            }
            RJ_ARITH_UNROLL(4)
            for (int part = 0; part < chunks; ++part) {
              const int offset = rows * half + col + kMfmaWidth * part;
              MfmaNative sum;
              std::memcpy(&sum, c[out_reg] + offset, sizeof(sum));
              RJ_ARITH_UNROLL(4)
              for (int k = 0; k < K; ++k) {
                sum = mfma_multiply_add(a_vec[k], b_vec[k][part], sum);
              }
              mfma_store(result[out_reg] + offset, sum);
            }
          } else {
            MfmaNative sum[chunks];
            RJ_ARITH_UNROLL(4)
            for (int part = 0; part < chunks; ++part) {
              const int offset = rows * half + col + kMfmaWidth * part;
              std::memcpy(&sum[part], c[out_reg] + offset, sizeof(sum[part]));
            }
            RJ_ARITH_UNROLL(4)
            for (int k = 0; k < K; ++k) {
              const MfmaNative a_k = mfma_broadcast(left[k % per_lane][a_lane[k / per_lane] + row]);
              RJ_ARITH_UNROLL(4)
              for (int part = 0; part < chunks; ++part) {
                sum[part] = mfma_multiply_add(a_k, b_vec[k][part], sum[part]);
              }
            }
            RJ_ARITH_UNROLL(4)
            for (int part = 0; part < chunks; ++part) {
              const int offset = rows * half + col + kMfmaWidth * part;
              mfma_store(result[out_reg] + offset, sum[part]);
            }
          }
        }
      }
    }
  }
  if (clamp_overflow) {
    mfma_clamp_overflow_planes<rows, K, Batch>(result, left, right, c, controls);
  }
  for (int reg = 0; reg < regs_per_batch * Batch; ++reg) {
    std::memcpy(d[reg], result[reg], sizeof(result[reg]));
  }
}

// Reduce deinterleaved [per_lane][64] decoded sources with FMA, preserving arbitrary overlap.
template <int M, int K, int Batch>
void mfma_reduce_planes(uint64_t controls, uint32_t *const *d, const float (*left)[64],
                        const float (*right)[64], const uint32_t *const *c,
                        bool clamp_overflow = false) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  if constexpr (M == 4 && Batch == 16) {
    mfma_reduce_planes_batch4<K>(controls, d, left, right, c, clamp_overflow);
  } else if constexpr (M == 32 && K == 1) {
    mfma_reduce_planes_32x32_small_k<K, Batch>(controls, d, left, right, c, clamp_overflow);
  } else if constexpr (M == 16 && Batch == 1) {
    mfma_reduce_planes_16x16<K>(controls, d, left, right, c, clamp_overflow);
  } else if constexpr (M == 32 && K == 4 && Batch == 2) {
    mfma_reduce_planes_32x32x4_2b(controls, d, left, right, c, clamp_overflow);
  } else {
    mfma_reduce_planes_tiled_v4<M, K, Batch>(controls, d, left, right, c, clamp_overflow);
  }
#else
  if constexpr (M == 4 && Batch == 16) {
    mfma_reduce_planes_batch4_v3<K>(controls, d, left, right, c, clamp_overflow);
  } else if constexpr (M == 32 && (K <= 2 || (K == 4 && Batch == 2))) {
    mfma_reduce_planes_32x32_small_k<K, Batch>(controls, d, left, right, c, clamp_overflow);
  } else {
    mfma_reduce_planes_tiled_v3<M, K, Batch>(controls, d, left, right, c, clamp_overflow);
  }
#endif
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
