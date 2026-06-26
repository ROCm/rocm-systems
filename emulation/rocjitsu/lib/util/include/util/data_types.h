// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DATA_TYPES_H_
#define UTIL_DATA_TYPES_H_

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Platform feature detection for the vectorized f16<->f32 block converters.
// x86 gets an F16C specialization; every other target (incl. ARM until a NEON
// path is added) uses the portable scalar fallback below.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define UTIL_ARCH_X86 1
#if defined(__AVX512F__) || defined(__F16C__)
#include <immintrin.h>
#define UTIL_HAS_X86_F16C 1
#endif
#endif

namespace util {

// ---- IEEE 754 half-precision (FP16) ----

inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(f);
}

namespace detail {

#if defined(UTIL_HAS_X86_F16C)
/// x86 F16C specialization: convert as many halves as the widest available
/// vector covers, advancing `i`. AVX-512 does 16/instr, AVX2 F16C does 8.
/// `_mm*_cvtph_ps` is IEEE-754 and bit-identical to f16_to_f32 for every
/// non-NaN input (verified exhaustively over all 65536 halves; NaN -> NaN with
/// a possibly different payload, which the SIMD execute paths tolerate).
inline void f16_to_f32_block_arch(const uint16_t *src, float *dst, size_t n, size_t &i) {
#if defined(__AVX512F__)
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(
        &dst[i], _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i))));
#endif
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(&dst[i],
                     _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
}
#else
/// Portable fallback (non-x86, or x86 without F16C). No vector advance; the
/// scalar loop in f16_to_f32_block does all the work.
/// TODO(arm): add an __aarch64__ NEON path here (vcvt_f32_f16 / fcvtl).
inline void f16_to_f32_block_arch(const uint16_t *, float *, size_t, size_t &) {}
#endif

} // namespace detail

/// @brief Convert `n` contiguous IEEE-754 half values to float.
///
/// Dispatches to a per-architecture vector backend (x86 F16C today) and falls
/// back to scalar f16_to_f32 for the remaining tail and on platforms without a
/// vector path. ~70x faster than the scalar bit-twiddling loop on AVX-512.
/// Hot path: MFMA f16 input gather, which converts 1024 halves per instruction.
inline void f16_to_f32_block(const uint16_t *src, float *dst, size_t n) {
  size_t i = 0;
  detail::f16_to_f32_block_arch(src, dst, n, i);
  for (; i < n; ++i)
    dst[i] = f16_to_f32(src[i]);
}
inline uint16_t f32_to_f16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 16) & 0x8000;
  uint32_t f_exp = (f >> 23) & 0xFF;
  uint32_t f_mant = f & 0x7FFFFF;

  if (f_exp == 0xFF) {
    if (f_mant)
      return static_cast<uint16_t>(sign | 0x7C00 | (f_mant >> 13) | 1);
    return static_cast<uint16_t>(sign | 0x7C00);
  }

  int32_t exp = static_cast<int32_t>(f_exp) - 127 + 15;

  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    uint32_t mant = f_mant | 0x800000;
    int shift = 14 - exp;
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    result += round_bit & (sticky | (result & 1));
    return static_cast<uint16_t>(sign | result);
  }

  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7C00);

  uint32_t round_bit = (f_mant >> 12) & 1;
  uint32_t sticky = (f_mant & 0xFFF) ? 1 : 0;
  uint32_t mant = f_mant >> 13;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x3FF) {
    mant = 0;
    exp += 1;
    if (exp >= 31)
      return static_cast<uint16_t>(sign | 0x7C00);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

inline uint16_t f32_to_f16_rtz(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 16) & 0x8000;
  uint32_t f_exp = (f >> 23) & 0xFF;
  uint32_t f_mant = f & 0x7FFFFF;

  if (f_exp == 0xFF) {
    if (f_mant)
      return static_cast<uint16_t>(sign | 0x7C00 | (f_mant >> 13) | 1);
    return static_cast<uint16_t>(sign | 0x7C00);
  }

  int32_t exp = static_cast<int32_t>(f_exp) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    uint32_t mant = f_mant | 0x800000;
    int shift = 14 - exp;
    return static_cast<uint16_t>(sign | (mant >> shift));
  }

  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7BFF);

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (f_mant >> 13));
}

// ---- BFloat16 (BF16) ----

inline float bf16_to_f32(uint16_t h) {
  uint32_t f = static_cast<uint32_t>(h) << 16;
  return std::bit_cast<float>(f);
}

inline uint16_t f32_to_bf16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  return static_cast<uint16_t>(f >> 16);
}

namespace detail {

#if defined(UTIL_HAS_X86_F16C)
/// x86 specialization: bf16->f32 is a zero-extend + 16-bit left shift, so it
/// needs no F16C, only the wide integer ops. AVX-512 does 16/instr, AVX2 8.
inline void bf16_to_f32_block_arch(const uint16_t *src, float *dst, size_t n, size_t &i) {
#if defined(__AVX512F__)
  for (; i + 16 <= n; i += 16) {
    __m512i w =
        _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i)));
    _mm512_storeu_ps(&dst[i], _mm512_castsi512_ps(_mm512_slli_epi32(w, 16)));
  }
#endif
#if defined(__AVX2__)
  for (; i + 8 <= n; i += 8) {
    __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i)));
    _mm256_storeu_ps(&dst[i], _mm256_castsi256_ps(_mm256_slli_epi32(w, 16)));
  }
#endif
}
#else
/// Portable fallback: no vector advance; the scalar shift loop in
/// bf16_to_f32_block does all the work (and trivially auto-vectorizes).
inline void bf16_to_f32_block_arch(const uint16_t *, float *, size_t, size_t &) {}
#endif

} // namespace detail

/// @brief Convert `n` contiguous BFloat16 values to float.
///
/// The bf16->f32 widening is exact (zero-extend into the low mantissa bits),
/// so the vector and scalar paths are bit-identical for every input including
/// NaN payloads. Hot path: MFMA/WMMA bf16 input gather.
inline void bf16_to_f32_block(const uint16_t *src, float *dst, size_t n) {
  size_t i = 0;
  detail::bf16_to_f32_block_arch(src, dst, n, i);
  for (; i < n; ++i)
    dst[i] = bf16_to_f32(src[i]);
}

namespace detail {

#if defined(UTIL_HAS_X86_F16C)
/// x86 specialization: i8->i32 / u8->i32 are plain sign-/zero-extends, no
/// F16C needed. AVX-512 does 16/instr, AVX2 8.
inline void i8_to_i32_block_arch(const int8_t *src, int32_t *dst, size_t n, size_t &i) {
#if defined(__AVX512F__)
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_si512(
        reinterpret_cast<__m512i *>(&dst[i]),
        _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
#endif
#if defined(__AVX2__)
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(&dst[i]),
        _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i))));
#endif
}
inline void u8_to_i32_block_arch(const uint8_t *src, int32_t *dst, size_t n, size_t &i) {
#if defined(__AVX512F__)
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_si512(
        reinterpret_cast<__m512i *>(&dst[i]),
        _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
#endif
#if defined(__AVX2__)
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(&dst[i]),
        _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i))));
#endif
}
#else
/// Portable fallback: no vector advance; the scalar extend loops below do all
/// the work (and trivially auto-vectorize).
inline void i8_to_i32_block_arch(const int8_t *, int32_t *, size_t, size_t &) {}
inline void u8_to_i32_block_arch(const uint8_t *, int32_t *, size_t, size_t &) {}
#endif

} // namespace detail

/// @brief Sign-extend `n` contiguous int8 values to int32.
///
/// Exact widening, so vector and scalar paths are bit-identical for every
/// input. Hot path: integer MFMA/WMMA i8 input gather.
inline void i8_to_i32_block(const int8_t *src, int32_t *dst, size_t n) {
  size_t i = 0;
  detail::i8_to_i32_block_arch(src, dst, n, i);
  for (; i < n; ++i)
    dst[i] = static_cast<int32_t>(src[i]);
}

/// @brief Zero-extend `n` contiguous uint8 values to int32 (unsigned iu8 WMMA).
inline void u8_to_i32_block(const uint8_t *src, int32_t *dst, size_t n) {
  size_t i = 0;
  detail::u8_to_i32_block_arch(src, dst, n, i);
  for (; i < n; ++i)
    dst[i] = static_cast<int32_t>(src[i]);
}

/// @brief Convert a float to 16-bit BFloat16 with round-to-nearest-even.
inline uint16_t f32_to_bf16_rne(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  if ((f & 0x7f800000u) != 0x7f800000u) {
    f += 0x7fffu + ((f >> 16) & 1u);
  } else if (f & 0xffffu) {
    f |= 0x10000u;
  }
  return static_cast<uint16_t>(f >> 16);
}

// ---- OCP-MX E8M0 unsigned exponent scale ----

inline float e8m0_to_f32(uint8_t code) {
  if (code == 0xffu)
    return std::numeric_limits<float>::quiet_NaN();
  if (code == 0u)
    return std::bit_cast<float>(0x00400000u);
  return std::bit_cast<float>(static_cast<uint32_t>(code) << 23);
}

// ---- FP8 E4M3 (OCP E4M3FN) — 1 sign, 4 exponent, 3 mantissa, bias=7 ----

inline float fp8_e4m3_to_f32(uint8_t v) {
  uint32_t sign = (v >> 7) & 1;
  uint32_t exp = (v >> 3) & 0xF;
  uint32_t mant = v & 0x7;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 15) {
    // OCP E4M3FN: NaN only when exp=15 AND mant=7 (0x7F/0xFF)
    if (mant == 7)
      return std::bit_cast<float>((sign << 31) | 0x7FC00000u);
    // exp=15, mant=0..6 → normal values (256..448)
    uint32_t f = (sign << 31) | ((15 + 127 - 7) << 23) | (mant << 20);
    return std::bit_cast<float>(f);
  }
  if (exp == 0) {
    float result = std::ldexp(static_cast<float>(mant), -9);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 7) << 23) | (mant << 20);
  return std::bit_cast<float>(f);
}

inline uint8_t f32_to_fp8_e4m3(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xFF) - 127 + 7;
  uint32_t mant = (f >> 20) & 0x7;
  if (exp <= 0)
    return static_cast<uint8_t>(sign);
  if (exp >= 15)
    return static_cast<uint8_t>(sign | 0x7E); // max normal
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}

inline uint8_t f32_to_fp8_e4m3_rne(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  if (std::isnan(val))
    return static_cast<uint8_t>(sign | 0x7F);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF)
    return static_cast<uint8_t>(sign | 0x7E);
  int32_t exp = f_exp - 127 + 7;
  if (exp <= 0) {
    if (exp < -3)
      return static_cast<uint8_t>(sign);
    uint32_t mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 8) {
      return static_cast<uint8_t>(sign | (1 << 3));
    }
    return static_cast<uint8_t>(sign | (result & 0x7));
  }
  if (exp > 15)
    return static_cast<uint8_t>(sign | 0x7E);
  uint32_t round_bit = (f_mant >> 19) & 1;
  uint32_t sticky = (f_mant & 0x7FFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 20) & 0x7;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x7) {
    mant = 0;
    exp += 1;
  }
  if (exp > 15 || (exp == 15 && mant >= 7))
    return static_cast<uint8_t>(sign | 0x7E);
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}

inline uint8_t f32_to_fp8_e4m3_sr(float val, uint32_t seed) {
  if (std::isnan(val))
    return static_cast<uint8_t>((std::bit_cast<uint32_t>(val) >> 24) & 0x80) | 0x7F;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF)
    return static_cast<uint8_t>(sign | 0x7E);
  int32_t exp = f_exp - 127 + 7;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t result = full_mant >> shift;
    uint32_t trunc_mask = (1u << shift) - 1;
    uint32_t trunc_bits = full_mant & trunc_mask;
    uint32_t random_add = seed >> (32 - shift);
    if ((trunc_bits + random_add) >= (1u << shift))
      result += 1;
    if (result >= 8)
      return static_cast<uint8_t>(sign | (1 << 3));
    return static_cast<uint8_t>(sign | (result & 0x7));
  }
  if (exp > 15)
    return static_cast<uint8_t>(sign | 0x7E);
  uint32_t trunc_bits = f_mant & 0xFFFFF;
  uint32_t random_add = seed >> 12;
  uint32_t mant = (f_mant >> 20) & 0x7;
  if ((trunc_bits + random_add) > 0xFFFFF) {
    mant += 1;
    if (mant > 0x7) {
      mant = 0;
      exp += 1;
    }
  }
  if (exp > 15 || (exp == 15 && mant >= 7))
    return static_cast<uint8_t>(sign | 0x7E);
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}

// ---- FP8 E5M3 (gfx1250 unsigned scale format) — 5 exponent, 3 mantissa, bias=15 ----

inline float fp8_e5m3_to_f32(uint8_t v) {
  uint32_t exp = (v >> 3) & 0x1F;
  uint32_t mant = v & 0x7;
  if (exp == 0 && mant == 0)
    return 0.0f;
  if (exp == 31 && mant == 7)
    return std::bit_cast<float>(0x7FC00000u);
  if (exp == 0)
    return std::ldexp(static_cast<float>(mant), -17);
  uint32_t f = ((exp + 127 - 15) << 23) | (mant << 20);
  return std::bit_cast<float>(f);
}

inline uint8_t f32_to_fp8_e5m3_rne(float val) {
  if (std::isnan(val) || std::isinf(val))
    return 0xFF;
  float mag = std::fabs(val);
  if (mag == 0.0f)
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(mag);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 15;
  if (exp <= 0) {
    if (exp < -3)
      return 0;
    uint32_t mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 24)
      return 0;
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 8)
      return 0x08;
    return static_cast<uint8_t>(result & 0x7);
  }
  if (exp > 31)
    return 0xFF;
  uint32_t round_bit = (f_mant >> 19) & 1;
  uint32_t sticky = (f_mant & 0x7FFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 20) & 0x7;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x7) {
    mant = 0;
    exp += 1;
  }
  if (exp > 31 || (exp == 31 && mant >= 7))
    return 0xFF;
  return static_cast<uint8_t>((static_cast<uint32_t>(exp) << 3) | mant);
}

inline uint8_t f32_to_fp8_e5m3_sr(float val, uint32_t seed) {
  if (std::isnan(val) || std::isinf(val))
    return 0xFF;
  float mag = std::fabs(val);
  if (mag == 0.0f)
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(mag);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 15;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 24)
      return 0;
    uint32_t result = full_mant >> shift;
    uint32_t trunc_mask = (1u << shift) - 1;
    uint32_t trunc_bits = full_mant & trunc_mask;
    uint32_t random_add = seed >> (32 - shift);
    if ((trunc_bits + random_add) >= (1u << shift))
      result += 1;
    if (result >= 8)
      return 0x08;
    return static_cast<uint8_t>(result & 0x7);
  }
  if (exp > 31)
    return 0xFF;
  uint32_t trunc_bits = f_mant & 0xFFFFF;
  uint32_t random_add = seed >> 12;
  uint32_t mant = (f_mant >> 20) & 0x7;
  if ((trunc_bits + random_add) > 0xFFFFF) {
    mant += 1;
    if (mant > 0x7) {
      mant = 0;
      exp += 1;
    }
  }
  if (exp > 31 || (exp == 31 && mant >= 7))
    return 0xFF;
  return static_cast<uint8_t>((static_cast<uint32_t>(exp) << 3) | mant);
}

// ---- BF8 E5M2 — 1 sign, 5 exponent, 2 mantissa, bias=15 ----

inline float bf8_e5m2_to_f32(uint8_t v) {
  uint32_t sign = (v >> 7) & 1;
  uint32_t exp = (v >> 2) & 0x1F;
  uint32_t mant = v & 0x3;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 31) {
    if (mant != 0)
      return std::bit_cast<float>((sign << 31) | 0x7FC00000u);
    return std::bit_cast<float>((sign << 31) | 0x7F800000u);
  }
  if (exp == 0) {
    float result = std::ldexp(static_cast<float>(mant), -16);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 21);
  return std::bit_cast<float>(f);
}

inline uint8_t f32_to_bf8_e5m2(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (f >> 21) & 0x3;
  if (exp <= 0)
    return static_cast<uint8_t>(sign);
  if (exp >= 31)
    return static_cast<uint8_t>(sign | 0x7C | (mant ? 0x1 : 0));
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}

inline uint8_t f32_to_bf8_e5m2_rne(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  if (std::isnan(val))
    return static_cast<uint8_t>(sign | 0x7F);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF)
    return static_cast<uint8_t>(sign | 0x7C);
  int32_t exp = f_exp - 127 + 15;
  if (exp <= 0) {
    if (exp < -2)
      return static_cast<uint8_t>(sign);
    uint32_t mant = f_mant | 0x800000;
    int shift = 22 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 4)
      return static_cast<uint8_t>(sign | (1 << 2));
    return static_cast<uint8_t>(sign | (result & 0x3));
  }
  if (exp >= 31)
    return static_cast<uint8_t>(sign | 0x7C);
  uint32_t round_bit = (f_mant >> 20) & 1;
  uint32_t sticky = (f_mant & 0xFFFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 21) & 0x3;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x3) {
    mant = 0;
    exp += 1;
    if (exp >= 31)
      return static_cast<uint8_t>(sign | 0x7C);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}

inline uint8_t f32_to_bf8_e5m2_sr(float val, uint32_t seed) {
  if (std::isnan(val))
    return static_cast<uint8_t>((std::bit_cast<uint32_t>(val) >> 24) & 0x80) | 0x7F;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  if (f_exp == 0xFF)
    return static_cast<uint8_t>(sign | 0x7C);
  int32_t exp = f_exp - 127 + 15;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 22 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t result = full_mant >> shift;
    uint32_t trunc_mask = (1u << shift) - 1;
    uint32_t trunc_bits = full_mant & trunc_mask;
    uint32_t random_add = seed >> (32 - shift);
    if ((trunc_bits + random_add) >= (1u << shift))
      result += 1;
    if (result >= 4)
      return static_cast<uint8_t>(sign | (1 << 2));
    return static_cast<uint8_t>(sign | (result & 0x3));
  }
  if (exp >= 31)
    return static_cast<uint8_t>(sign | 0x7C);
  uint32_t trunc_bits = f_mant & 0x1FFFFF;
  uint32_t random_add = seed >> 11;
  uint32_t mant = (f_mant >> 21) & 0x3;
  if ((trunc_bits + random_add) > 0x1FFFFF) {
    mant += 1;
    if (mant > 0x3) {
      mant = 0;
      exp += 1;
      if (exp >= 31)
        return static_cast<uint8_t>(sign | 0x7C);
    }
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}

namespace detail {

/// 256-entry fp8/bf8 -> f32 tables filled from the scalar converters, so every
/// entry (including the NaN payloads) is bit-exact with the per-element path
/// by construction.
inline const float *fp8_e4m3_lut() {
  static const auto lut = [] {
    std::array<float, 256> t{};
    for (uint32_t i = 0; i < 256; ++i)
      t[i] = fp8_e4m3_to_f32(static_cast<uint8_t>(i));
    return t;
  }();
  return lut.data();
}

inline const float *bf8_e5m2_lut() {
  static const auto lut = [] {
    std::array<float, 256> t{};
    for (uint32_t i = 0; i < 256; ++i)
      t[i] = bf8_e5m2_to_f32(static_cast<uint8_t>(i));
    return t;
  }();
  return lut.data();
}

} // namespace detail

/// @brief Convert `n` contiguous E4M3 FP8 bytes to float via the lookup table.
///
/// Bit-exact with fp8_e4m3_to_f32 for every code. The scalar gather loop is
/// enough to amortize the per-element converter cost on the MFMA/WMMA bulk
/// hoists (no vector path needed; works on every target).
inline void fp8_e4m3_to_f32_block(const uint8_t *src, float *dst, size_t n) {
  const float *lut = detail::fp8_e4m3_lut();
  for (size_t i = 0; i < n; ++i)
    dst[i] = lut[src[i]];
}

/// @brief Convert `n` contiguous E5M2 BF8 bytes to float via the lookup table.
inline void bf8_e5m2_to_f32_block(const uint8_t *src, float *dst, size_t n) {
  const float *lut = detail::bf8_e5m2_lut();
  for (size_t i = 0; i < n; ++i)
    dst[i] = lut[src[i]];
}

// ---- Generalized SR + OCP MX helpers (used by gfx1250 WMMA) ----
inline uint32_t f32_to_binary_float_sr(float val, uint32_t seed, uint32_t exp_bits,
                                       uint32_t mant_bits, int32_t bias, int32_t max_exp,
                                       int32_t min_exp, uint32_t nan_code, uint32_t inf_code) {
  const uint32_t bits = std::bit_cast<uint32_t>(val);
  const uint32_t sign_bit = bits >> 31;
  const uint32_t sign = sign_bit << (exp_bits + mant_bits);
  const uint32_t abs_bits = bits & 0x7FFFFFFFu;
  const uint32_t exp_field = (bits >> 23) & 0xFFu;
  uint32_t src_mant = bits & 0x7FFFFFu;

  if (abs_bits > 0x7F800000u)
    return sign | nan_code;
  if (abs_bits == 0x7F800000u)
    return sign | inf_code;
  if ((abs_bits & 0x7FFFFFFFu) == 0)
    return sign;

  const int32_t src_exp =
      static_cast<int32_t>((exp_field == 0 && src_mant != 0) ? 1u : exp_field) - 127;
  int32_t exp = src_exp;
  uint32_t mant = src_mant;
  bool subnorm = false;

  if (exp > max_exp)
    return sign | inf_code;

  if (exp < min_exp) {
    subnorm = true;
    exp = 0;
    const uint32_t diff = static_cast<uint32_t>(min_exp - src_exp);
    if (diff >= 32u) {
      mant = 0;
      src_mant = 0;
    } else {
      src_mant |= 1u << 23;
      src_mant >>= diff;
      mant = src_mant;
    }
  }

  const uint32_t sr_shift = (32u - 23u) + mant_bits;
  mant += seed >> sr_shift;
  if (mant >= (1u << 23))
    ++exp;
  mant >>= 23u - mant_bits;
  mant &= (1u << mant_bits) - 1u;

  if (exp > max_exp)
    return sign | inf_code;

  uint32_t biased_exp = static_cast<uint32_t>(exp);
  if (!subnorm)
    biased_exp = static_cast<uint32_t>(exp + bias);
  biased_exp &= (1u << exp_bits) - 1u;

  return sign | (biased_exp << mant_bits) | mant;
}

inline uint16_t f32_to_f16_sr(float val, uint32_t seed) {
  return static_cast<uint16_t>(
      f32_to_binary_float_sr(val, seed, 5, 10, 15, 15, -14, 0x7FFFu, 0x7C00u));
}

inline uint16_t f32_to_bf16_sr(float val, uint32_t seed) {
  return static_cast<uint16_t>(
      f32_to_binary_float_sr(val, seed, 8, 7, 127, 127, -126, 0x7FFFu, 0x7F80u));
}

inline float ocp_mx_to_f32(uint32_t raw, uint32_t exp_bits, uint32_t mant_bits, int32_t bias) {
  uint32_t sign = (raw >> (exp_bits + mant_bits)) & 1;
  uint32_t exp = (raw >> mant_bits) & ((1u << exp_bits) - 1u);
  uint32_t mant = raw & ((1u << mant_bits) - 1u);
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);

  float significand;
  int32_t exponent;
  if (exp == 0) {
    significand = static_cast<float>(mant) / static_cast<float>(1u << mant_bits);
    exponent = 1 - bias;
  } else {
    significand = 1.0f + static_cast<float>(mant) / static_cast<float>(1u << mant_bits);
    exponent = static_cast<int32_t>(exp) - bias;
  }

  float value = std::ldexp(significand, exponent);
  return sign ? -value : value;
}

inline uint8_t f32_to_ocp_mx_rne(float val, uint32_t exp_bits, uint32_t mant_bits, int32_t bias) {
  const uint32_t sign_bit = 1u << (exp_bits + mant_bits);
  const uint32_t positive_max = sign_bit - 1u;
  uint32_t bits = std::bit_cast<uint32_t>(val);
  uint8_t sign = static_cast<uint8_t>((bits >> 31) ? sign_bit : 0u);
  uint32_t abs_bits = bits & 0x7FFFFFFFu;
  if (abs_bits > 0x7F800000u)
    return static_cast<uint8_t>(sign | positive_max);
  if (abs_bits == 0x7F800000u)
    return static_cast<uint8_t>(sign | positive_max);

  float abs_val = std::bit_cast<float>(abs_bits);
  uint8_t best = 0;
  float best_diff = std::numeric_limits<float>::infinity();
  for (uint32_t code = 0; code <= positive_max; ++code) {
    float candidate = ocp_mx_to_f32(code, exp_bits, mant_bits, bias);
    float diff = std::fabs(abs_val - candidate);
    if (diff < best_diff || (diff == best_diff && ((code & 1u) == 0u) && ((best & 1u) != 0u))) {
      best = static_cast<uint8_t>(code);
      best_diff = diff;
    }
  }
  return static_cast<uint8_t>(sign | best);
}

// ---- FP4 E2M1 — 1 sign, 2 exponent, 1 mantissa, bias=1, no NaN/Inf, max=6.0 ----

inline float fp4_e2m1_to_f32(uint8_t v) {
  uint32_t sign = (v >> 3) & 1;
  uint32_t exp = (v >> 1) & 0x3;
  uint32_t mant = v & 0x1;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 0) {
    float result = 0.5f;
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 1) << 23) | (mant << 22);
  return std::bit_cast<float>(f);
}

inline uint8_t f32_to_fp4_e2m1_rne(float val) {
  if (std::isnan(val))
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 28) & 0x8;
  if (std::isinf(val))
    return static_cast<uint8_t>(sign | 0x7);
  float absval = std::fabs(val);
  if (absval > 6.0f)
    return static_cast<uint8_t>(sign | 0x7);
  if (absval < 0.25f)
    return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 23 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t round_bit = (full_mant >> (shift - 1)) & 1;
    uint32_t sticky = (full_mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = full_mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 2)
      return static_cast<uint8_t>(sign | (1 << 1));
    return static_cast<uint8_t>(sign | (result & 0x1));
  }
  if (exp > 3)
    return static_cast<uint8_t>(sign | 0x7);
  uint32_t round_bit = (f_mant >> 21) & 1;
  uint32_t sticky = (f_mant & 0x1FFFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 22) & 0x1;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 1) {
    mant = 0;
    exp += 1;
    if (exp > 3)
      return static_cast<uint8_t>(sign | 0x7);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 1) | mant);
}

inline uint8_t f32_to_fp4_e2m1_sr(float val, uint32_t seed) {
  if (std::isnan(val))
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 28) & 0x8;
  if (std::isinf(val) || std::fabs(val) > 6.0f)
    return static_cast<uint8_t>(sign | 0x7);
  if (std::fabs(val) < 0.25f)
    return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 23 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t result = full_mant >> shift;
    uint32_t trunc_mask = (1u << shift) - 1;
    uint32_t trunc_bits = full_mant & trunc_mask;
    uint32_t random_add = seed >> (32 - shift);
    if ((trunc_bits + random_add) >= (1u << shift))
      result += 1;
    if (result >= 2)
      return static_cast<uint8_t>(sign | (1 << 1));
    return static_cast<uint8_t>(sign | (result & 0x1));
  }
  if (exp > 3)
    return static_cast<uint8_t>(sign | 0x7);
  uint32_t trunc_bits = f_mant & 0x3FFFFF;
  uint32_t random_add = seed >> 10;
  uint32_t mant = (f_mant >> 22) & 0x1;
  if ((trunc_bits + random_add) > 0x3FFFFF) {
    mant += 1;
    if (mant > 1) {
      mant = 0;
      exp += 1;
      if (exp > 3)
        return static_cast<uint8_t>(sign | 0x7);
    }
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 1) | mant);
}

// ---- FP6 E2M3 — 1 sign, 2 exponent, 3 mantissa, bias=1, no NaN/Inf, max=7.5 ----

inline float fp6_e2m3_to_f32(uint8_t v) {
  uint32_t sign = (v >> 5) & 1;
  uint32_t exp = (v >> 3) & 0x3;
  uint32_t mant = v & 0x7;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 0) {
    float result = std::ldexp(static_cast<float>(mant), -3);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 1) << 23) | (mant << 20);
  return std::bit_cast<float>(f);
}

inline uint8_t f32_to_fp6_e2m3_rne(float val) {
  if (std::isnan(val))
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 7.5f)
    return static_cast<uint8_t>(sign | 0x1F);
  float absval = std::fabs(val);
  if (absval < std::ldexp(1.0f, -4))
    return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t round_bit = (full_mant >> (shift - 1)) & 1;
    uint32_t sticky = (full_mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = full_mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 8)
      return static_cast<uint8_t>(sign | (1 << 3));
    return static_cast<uint8_t>(sign | (result & 0x7));
  }
  if (exp > 3)
    return static_cast<uint8_t>(sign | 0x1F);
  uint32_t round_bit = (f_mant >> 19) & 1;
  uint32_t sticky = (f_mant & 0x7FFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 20) & 0x7;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x7) {
    mant = 0;
    exp += 1;
    if (exp > 3)
      return static_cast<uint8_t>(sign | 0x1F);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}

inline uint8_t f32_to_fp6_e2m3_sr(float val, uint32_t seed) {
  if (std::isnan(val))
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 7.5f)
    return static_cast<uint8_t>(sign | 0x1F);
  if (std::fabs(val) < std::ldexp(1.0f, -4))
    return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 1;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 21 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t result = full_mant >> shift;
    uint32_t trunc_mask = (1u << shift) - 1;
    uint32_t trunc_bits = full_mant & trunc_mask;
    uint32_t random_add = seed >> (32 - shift);
    if ((trunc_bits + random_add) >= (1u << shift))
      result += 1;
    if (result >= 8)
      return static_cast<uint8_t>(sign | (1 << 3));
    return static_cast<uint8_t>(sign | (result & 0x7));
  }
  if (exp > 3)
    return static_cast<uint8_t>(sign | 0x1F);
  uint32_t trunc_bits = f_mant & 0xFFFFF;
  uint32_t random_add = seed >> 12;
  uint32_t mant = (f_mant >> 20) & 0x7;
  if ((trunc_bits + random_add) > 0xFFFFF) {
    mant += 1;
    if (mant > 0x7) {
      mant = 0;
      exp += 1;
      if (exp > 3)
        return static_cast<uint8_t>(sign | 0x1F);
    }
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}

// ---- BF6 E3M2 — 1 sign, 3 exponent, 2 mantissa, bias=3, no NaN/Inf, max=28.0 ----

inline float bf6_e3m2_to_f32(uint8_t v) {
  uint32_t sign = (v >> 5) & 1;
  uint32_t exp = (v >> 2) & 0x7;
  uint32_t mant = v & 0x3;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 0) {
    float result = std::ldexp(static_cast<float>(mant), -4);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 3) << 23) | (mant << 21);
  return std::bit_cast<float>(f);
}

inline uint8_t f32_to_bf6_e3m2_rne(float val) {
  if (std::isnan(val))
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 28.0f)
    return static_cast<uint8_t>(sign | 0x1F);
  float absval = std::fabs(val);
  if (absval < std::ldexp(1.0f, -5))
    return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 3;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 22 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t round_bit = (full_mant >> (shift - 1)) & 1;
    uint32_t sticky = (full_mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = full_mant >> shift;
    result += round_bit & (sticky | (result & 1));
    if (result >= 4)
      return static_cast<uint8_t>(sign | (1 << 2));
    return static_cast<uint8_t>(sign | (result & 0x3));
  }
  if (exp > 7)
    return static_cast<uint8_t>(sign | 0x1F);
  uint32_t round_bit = (f_mant >> 20) & 1;
  uint32_t sticky = (f_mant & 0xFFFFF) ? 1 : 0;
  uint32_t mant = (f_mant >> 21) & 0x3;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x3) {
    mant = 0;
    exp += 1;
    if (exp > 7)
      return static_cast<uint8_t>(sign | 0x1F);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}

inline uint8_t f32_to_bf6_e3m2_sr(float val, uint32_t seed) {
  if (std::isnan(val))
    return 0;
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 26) & 0x20;
  if (std::isinf(val) || std::fabs(val) > 28.0f)
    return static_cast<uint8_t>(sign | 0x1F);
  if (std::fabs(val) < std::ldexp(1.0f, -5))
    return static_cast<uint8_t>(sign);
  int32_t f_exp = static_cast<int32_t>((f >> 23) & 0xFF);
  uint32_t f_mant = f & 0x7FFFFF;
  int32_t exp = f_exp - 127 + 3;
  if (exp <= 0) {
    uint32_t full_mant = f_mant | 0x800000;
    int shift = 22 - exp;
    if (shift > 24)
      return static_cast<uint8_t>(sign);
    uint32_t result = full_mant >> shift;
    uint32_t trunc_mask = (1u << shift) - 1;
    uint32_t trunc_bits = full_mant & trunc_mask;
    uint32_t random_add = seed >> (32 - shift);
    if ((trunc_bits + random_add) >= (1u << shift))
      result += 1;
    if (result >= 4)
      return static_cast<uint8_t>(sign | (1 << 2));
    return static_cast<uint8_t>(sign | (result & 0x3));
  }
  if (exp > 7)
    return static_cast<uint8_t>(sign | 0x1F);
  uint32_t trunc_bits = f_mant & 0x1FFFFF;
  uint32_t random_add = seed >> 11;
  uint32_t mant = (f_mant >> 21) & 0x3;
  if ((trunc_bits + random_add) > 0x1FFFFF) {
    mant += 1;
    if (mant > 0x3) {
      mant = 0;
      exp += 1;
      if (exp > 7)
        return static_cast<uint8_t>(sign | 0x1F);
    }
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}

// ---- 6-bit packing/unpacking (32×6-bit values ↔ 6 DWORDs, little-endian) ----

inline void pack_6bit(const uint8_t vals[32], uint32_t dwords[6]) {
  for (int d = 0; d < 6; ++d)
    dwords[d] = 0;
  for (int i = 0; i < 32; ++i) {
    uint64_t bit_offset = static_cast<uint64_t>(i) * 6;
    uint32_t dw_idx = static_cast<uint32_t>(bit_offset / 32);
    uint32_t bit_pos = static_cast<uint32_t>(bit_offset % 32);
    uint32_t val6 = vals[i] & 0x3F;
    dwords[dw_idx] |= val6 << bit_pos;
    if (bit_pos > 26)
      dwords[dw_idx + 1] |= val6 >> (32 - bit_pos);
  }
}

inline void unpack_6bit(const uint32_t dwords[6], uint8_t vals[32]) {
  for (int i = 0; i < 32; ++i) {
    uint64_t bit_offset = static_cast<uint64_t>(i) * 6;
    uint32_t dw_idx = static_cast<uint32_t>(bit_offset / 32);
    uint32_t bit_pos = static_cast<uint32_t>(bit_offset % 32);
    uint32_t val = (dwords[dw_idx] >> bit_pos) & 0x3F;
    if (bit_pos > 26)
      val |= (dwords[dw_idx + 1] << (32 - bit_pos)) & 0x3F;
    vals[i] = static_cast<uint8_t>(val);
  }
}

// ---- Stochastic rounding LFSR ----

inline uint32_t prng_advance(uint32_t seed) { return (seed << 1) ^ ((seed >> 31) ? 197u : 0u); }

} // namespace util

#endif // UTIL_DATA_TYPES_H_
