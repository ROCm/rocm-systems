// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 gfx12 FP8/BF8 WMMA kernels using the Wave32 register layout.
// Uses direct vpshufb+F16C widening with fast-path finite E4M3FN decode and
// deferred bias rescaling, buffering results before stores for VGPR aliasing.

#include "common/compiler.h"
#include "common/simd.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_exact8_x86_64.h"
#include "wmma/x86_64/wmma_loose_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

template <bool Bf8, bool CheckNan>
[[gnu::always_inline]] inline void decode_pass(float (&tile)[16][16],
                                               const uint32_t *const *operand) {
  RJ_ARITH_UNROLL(4)
  for (int g = 0; g < 4; ++g) {
    const int k = 4 * g;
    const int reg = wmma_gfx12_packed_local(k) / 4;
    const int lane = 16 * (k / 8);
    const uint32_t *ptr = operand[reg] + lane;
    const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 8));
    __m256 a0, a1, a2, a3, b0, b1, b2, b3;
    wmma_decode_fp8_word_v3<Bf8, CheckNan>(w0, a0, a1, a2, a3);
    wmma_decode_fp8_word_v3<Bf8, CheckNan>(w1, b0, b1, b2, b3);
    _mm256_store_ps(tile[k], a0);
    _mm256_store_ps(tile[k + 1], a1);
    _mm256_store_ps(tile[k + 2], a2);
    _mm256_store_ps(tile[k + 3], a3);
    _mm256_store_ps(tile[k] + 8, b0);
    _mm256_store_ps(tile[k + 1] + 8, b1);
    _mm256_store_ps(tile[k + 2] + 8, b2);
    _mm256_store_ps(tile[k + 3] + 8, b3);
  }
}

// Decode 16x16 FP8/BF8 operand into K-major FP32.
// E5M2 (Bf8=true) shares the binary16 exponent bias and widens directly via F16C
// after byte-level NaN canonicalization. E4M3FN (Bf8=false) widens with a 2^-8
// scale factor (exact, including subnormals and NaNs) folded once at final C add;
// a fast L1 byte scan skips per-halfword NaN fixups on the common finite path
// without staging a chunks[8] spill buffer.
template <bool Bf8>
[[gnu::always_inline]] inline void decode(float (&tile)[16][16], const uint32_t *const *operand) {
  if constexpr (Bf8) {
    decode_pass<true, false>(tile, operand);
  } else {
    const __m256i byte_abs = _mm256_set1_epi8(0x7f);
    __m256i any_nan = _mm256_setzero_si256();
    for (int reg = 0; reg < 2; ++reg) {
      for (int lane = 0; lane < 32; lane += 8) {
        const __m256i w =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(operand[reg] + lane));
        const __m256i is_nan = _mm256_cmpeq_epi8(_mm256_and_si256(w, byte_abs), byte_abs);
        any_nan = _mm256_or_si256(any_nan, is_nan);
      }
    }
    if (_mm256_movemask_epi8(any_nan) == 0) {
      decode_pass<false, false>(tile, operand);
    } else {
      decode_pass<false, true>(tile, operand);
    }
  }
}

// Return true when direct stores to d[0..7] (32 lanes each) are guaranteed
// equivalent to buffered ascending stores. Recognizes both Wave32 contiguous
// VGPRs (d[0..7] increasing by >= 128B) and Wave64 Wave32Views (d[0..3]
// increasing by >= 256B with d[i+4] == d[i] + 32).
[[gnu::always_inline]] inline bool fp8_direct_store_safe(uint32_t *const *d,
                                                         const uint32_t *const *c) {
  constexpr uintptr_t kRegBytes = 32 * sizeof(uint32_t);
  const uintptr_t d0 = reinterpret_cast<uintptr_t>(d[0]);
  bool w32_ok = true;
  uintptr_t prev = d0;
  for (int i = 1; i < 8; ++i) {
    const uintptr_t di = reinterpret_cast<uintptr_t>(d[i]);
    if (di < prev || (di - prev) < kRegBytes) {
      w32_ok = false;
      break;
    }
    prev = di;
  }
  uintptr_t max_d = prev;
  if (!w32_ok) {
    uintptr_t p = d0;
    for (int i = 1; i < 4; ++i) {
      const uintptr_t di = reinterpret_cast<uintptr_t>(d[i]);
      if (di < p || (di - p) < 2 * kRegBytes) {
        return false;
      }
      p = di;
    }
    for (int i = 0; i < 4; ++i) {
      if (reinterpret_cast<uintptr_t>(d[i + 4]) != reinterpret_cast<uintptr_t>(d[i]) + kRegBytes) {
        return false;
      }
    }
    max_d = p + kRegBytes;
  }
  uintptr_t same_diff = 0;
  uintptr_t min_c = reinterpret_cast<uintptr_t>(c[0]);
  uintptr_t max_c = min_c;
  for (int i = 0; i < 8; ++i) {
    const uintptr_t ci = reinterpret_cast<uintptr_t>(c[i]);
    same_diff |= (reinterpret_cast<uintptr_t>(d[i]) ^ ci);
    if (ci < min_c) {
      min_c = ci;
    }
    if (ci > max_c) {
      max_c = ci;
    }
  }
  if (same_diff == 0) {
    return true;
  }
  const bool c_before_d = (max_c < d0) && ((d0 - max_c) >= kRegBytes);
  const bool d_before_c = (max_d < min_c) && ((min_c - max_d) >= kRegBytes);
  return c_before_d || d_before_c;
}

// Compute loose FP8/BF8 A*B+C into FP32 D in gfx12 Wave32 layout.
// D may overlap inputs; all source values are consumed before stores.
template <bool Bf8A, bool Bf8B, bool DirectStore>
[[gnu::noinline]] void wmma_impl(uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  alignas(32) float left[16][16], right[16][16];
  decode<Bf8A>(left, a);
  decode<Bf8B>(right, b);

  alignas(32) uint32_t result[DirectStore ? 1 : 8][DirectStore ? 1 : 32];
  constexpr int Rows = 4;
  constexpr float kScale = (!Bf8A && !Bf8B) ? 65536.0f : ((!Bf8A || !Bf8B) ? 256.0f : 1.0f);
  const __m256 scale_vec = _mm256_set1_ps(kScale);

#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (int base = 0; base < 16; base += Rows) {
    __m256 acc[Rows][2];
    if constexpr (Bf8A && Bf8B) {
      for (int r = 0; r < Rows; ++r) {
        const int row = base + r;
        const uint32_t *c_ptr = c[row % 8] + 16 * (row / 8);
        acc[r][0] =
            _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr)));
        acc[r][1] =
            _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr + 8)));
      }
      RJ_ARITH_UNROLL(4)
      for (int k = 0; k < 16; ++k) {
        const __m256 bl = _mm256_load_ps(right[k]);
        const __m256 bh = _mm256_load_ps(right[k] + 8);
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < Rows; ++r) {
          const __m256 x = _mm256_broadcast_ss(&left[k][base + r]);
          acc[r][0] = _mm256_fmadd_ps(x, bl, acc[r][0]);
          acc[r][1] = _mm256_fmadd_ps(x, bh, acc[r][1]);
        }
      }
    } else {
      {
        const __m256 bl = _mm256_load_ps(right[0]);
        const __m256 bh = _mm256_load_ps(right[0] + 8);
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < Rows; ++r) {
          const __m256 x = _mm256_broadcast_ss(&left[0][base + r]);
          acc[r][0] = _mm256_mul_ps(x, bl);
          acc[r][1] = _mm256_mul_ps(x, bh);
        }
      }
      RJ_ARITH_UNROLL(3)
      for (int k = 1; k < 4; ++k) {
        const __m256 bl = _mm256_load_ps(right[k]);
        const __m256 bh = _mm256_load_ps(right[k] + 8);
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < Rows; ++r) {
          const __m256 x = _mm256_broadcast_ss(&left[k][base + r]);
          acc[r][0] = _mm256_fmadd_ps(x, bl, acc[r][0]);
          acc[r][1] = _mm256_fmadd_ps(x, bh, acc[r][1]);
        }
      }
      RJ_ARITH_UNROLL(4)
      for (int k = 4; k < 16; ++k) {
        const __m256 bl = _mm256_load_ps(right[k]);
        const __m256 bh = _mm256_load_ps(right[k] + 8);
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < Rows; ++r) {
          const __m256 x = _mm256_broadcast_ss(&left[k][base + r]);
          acc[r][0] = _mm256_fmadd_ps(x, bl, acc[r][0]);
          acc[r][1] = _mm256_fmadd_ps(x, bh, acc[r][1]);
        }
      }
      for (int r = 0; r < Rows; ++r) {
        const int row = base + r;
        const uint32_t *c_ptr = c[row % 8] + 16 * (row / 8);
        const __m256 c0 =
            _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr)));
        const __m256 c1 =
            _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr + 8)));
        acc[r][0] = _mm256_fmadd_ps(acc[r][0], scale_vec, c0);
        acc[r][1] = _mm256_fmadd_ps(acc[r][1], scale_vec, c1);
      }
    }
    for (int r = 0; r < Rows; ++r) {
      const int row = base + r;
      uint32_t *dst =
          DirectStore ? (d[row % 8] + 16 * (row / 8)) : (result[row % 8] + 16 * (row / 8));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), _mm256_castps_si256(acc[r][0]));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 8), _mm256_castps_si256(acc[r][1]));
    }
  }

  if constexpr (!DirectStore) {
    for (int reg = 0; reg < 8; ++reg) {
      for (int lane = 0; lane < 32; lane += 8) {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(d[reg] + lane),
            _mm256_load_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
      }
    }
  }
}

template <bool Bf8A, bool Bf8B>
[[gnu::always_inline]] inline void wmma(uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c) {
  if (fp8_direct_store_safe(d, c)) {
    wmma_impl<Bf8A, Bf8B, true>(d, a, b, c);
  } else {
    wmma_impl<Bf8A, Bf8B, false>(d, a, b, c);
  }
}

} // namespace

// Compute loose gfx12 Wave32 A*B+C into FP32 D for the selected factor formats.
// D may overlap source registers.
template <bool Bf8A, bool Bf8B>
void fp8_wmma_x86_64_v3(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  wmma<Bf8A, Bf8B>(d, a, b, c);
}

template void fp8_wmma_x86_64_v3<false, false>(uint32_t *const *, const uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v3<false, true>(uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v3<true, false>(uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v3<true, true>(uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);

void fp8_wmma_exact_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  wmma_exact8::run(bf8_a, bf8_b, d, a, b, c);
}

} // namespace rocjitsu::arith
