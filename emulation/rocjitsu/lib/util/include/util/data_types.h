// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DATA_TYPES_H_
#define UTIL_DATA_TYPES_H_

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

// IEEE 754 half-precision (FP16) conversion

/// @brief Convert a 16-bit IEEE 754 half-precision value to float.
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

/// @brief Convert a float to 16-bit IEEE 754 half-precision.
inline uint16_t f32_to_f16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 16) & 0x8000;
  uint32_t f_exp = (f >> 23) & 0xFF;
  uint32_t f_mant = f & 0x7FFFFF;

  // Inf or NaN.
  if (f_exp == 0xFF) {
    if (f_mant)
      return static_cast<uint16_t>(sign | 0x7C00 | (f_mant >> 13) |
                                   1);           // NaN, preserve payload MSBs
    return static_cast<uint16_t>(sign | 0x7C00); // Inf
  }

  // Rebias exponent from F32 (bias 127) to F16 (bias 15).
  int32_t exp = static_cast<int32_t>(f_exp) - 127 + 15;

  // F16 denormal or underflow.
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign); // too small, flush to zero
    // Denormal: shift mantissa right, add implicit 1-bit.
    uint32_t mant = f_mant | 0x800000;
    int shift = 14 - exp; // shift = 14 when exp=0, 24 when exp=-10
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    // Round-to-nearest-even.
    result += round_bit & (sticky | (result & 1));
    return static_cast<uint16_t>(sign | result);
  }

  // Overflow → Inf.
  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7C00);

  // Normal number: round-to-nearest-even on the 13 truncated mantissa bits.
  uint32_t round_bit = (f_mant >> 12) & 1;
  uint32_t sticky = (f_mant & 0xFFF) ? 1 : 0;
  uint32_t mant = f_mant >> 13;
  mant += round_bit & (sticky | (mant & 1));
  // Rounding may overflow mantissa into exponent.
  if (mant > 0x3FF) {
    mant = 0;
    exp += 1;
    if (exp >= 31)
      return static_cast<uint16_t>(sign | 0x7C00); // overflow to Inf
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

/// @brief Convert a float to 16-bit IEEE 754 half-precision, rounding toward zero.
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

// BFloat16 (BF16) conversion

/// @brief Convert a 16-bit BFloat16 value to float.
inline float bf16_to_f32(uint16_t h) {
  uint32_t f = static_cast<uint32_t>(h) << 16;
  return std::bit_cast<float>(f);
}

/// @brief Convert a float to 16-bit BFloat16 (truncation, no rounding).
inline uint16_t f32_to_bf16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  return static_cast<uint16_t>(f >> 16);
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

// FP8 (E4M3) conversion - 1 sign, 4 exponent, 3 mantissa bits

/// @brief Convert an 8-bit E4M3 FP8 value to float.
inline float fp8_e4m3_to_f32(uint8_t v) {
  uint32_t sign = (v >> 7) & 1;
  uint32_t exp = (v >> 3) & 0xF;
  uint32_t mant = v & 0x7;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 15) {
    // E4M3FN has a single NaN encoding when exponent and mantissa are all ones.
    if (mant == 7)
      return std::bit_cast<float>((sign << 31) | 0x7FC00000u);
  }
  if (exp == 0) {
    // Denormalized
    float result = std::ldexp(static_cast<float>(mant), -9);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 7) << 23) | (mant << 20);
  return std::bit_cast<float>(f);
}

/// @brief Convert a float to 8-bit E4M3 FP8 (truncation).
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

/// @brief Convert a float to 8-bit E4M3FN FP8 with round-to-nearest-even.
inline uint8_t f32_to_fp8_e4m3_rne(float val) {
  uint32_t bits = std::bit_cast<uint32_t>(val);
  uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  uint32_t abs_bits = bits & 0x7FFFFFFFu;
  if (abs_bits > 0x7F800000u)
    return static_cast<uint8_t>(sign | 0x7Fu);
  if (abs_bits == 0x7F800000u)
    return static_cast<uint8_t>(sign | 0x7Fu);

  float abs_val = std::bit_cast<float>(abs_bits);
  if (abs_val > 464.0f)
    return static_cast<uint8_t>(sign | 0x7Fu);

  uint8_t best = 0;
  float best_diff = std::numeric_limits<float>::infinity();
  for (uint8_t code = 0; code <= 0x7Eu; ++code) {
    float candidate = fp8_e4m3_to_f32(code);
    float diff = std::fabs(abs_val - candidate);
    if (diff < best_diff || (diff == best_diff && ((code & 1u) == 0u) && ((best & 1u) != 0u))) {
      best = code;
      best_diff = diff;
    }
  }
  return static_cast<uint8_t>(sign | best);
}

// BF8 (E5M2) conversion - 1 sign, 5 exponent, 2 mantissa bits

/// @brief Convert an 8-bit E5M2 BF8 value to float.
inline float bf8_e5m2_to_f32(uint8_t v) {
  uint32_t sign = (v >> 7) & 1;
  uint32_t exp = (v >> 2) & 0x1F;
  uint32_t mant = v & 0x3;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 31) {
    if (mant != 0)
      return std::bit_cast<float>((sign << 31) | 0x7FC00000u);
    return std::bit_cast<float>((sign << 31) | 0x7F800000u); // infinity
  }
  if (exp == 0) {
    float result = std::ldexp(static_cast<float>(mant), -16);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 21);
  return std::bit_cast<float>(f);
}

/// @brief Convert a float to 8-bit E5M2 BF8 (truncation).
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

/// @brief Convert a float to 8-bit E5M2 BF8 with round-to-nearest-even.
inline uint8_t f32_to_bf8_e5m2_rne(float val) {
  uint32_t bits = std::bit_cast<uint32_t>(val);
  uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  uint32_t abs_bits = bits & 0x7FFFFFFFu;
  if (abs_bits > 0x7F800000u)
    return static_cast<uint8_t>(sign | 0x7Fu);
  if (abs_bits == 0x7F800000u)
    return static_cast<uint8_t>(sign | 0x7Cu);

  float abs_val = std::bit_cast<float>(abs_bits);
  if (abs_val >= 61440.0f)
    return static_cast<uint8_t>(sign | 0x7Cu);

  uint8_t best = 0;
  float best_diff = std::numeric_limits<float>::infinity();
  for (uint8_t code = 0; code <= 0x7Bu; ++code) {
    float candidate = bf8_e5m2_to_f32(code);
    float diff = std::fabs(abs_val - candidate);
    if (diff < best_diff || (diff == best_diff && ((code & 1u) == 0u) && ((best & 1u) != 0u))) {
      best = code;
      best_diff = diff;
    }
  }
  return static_cast<uint8_t>(sign | best);
}

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

/// @brief Convert a float to 16-bit IEEE 754 half-precision with stochastic rounding.
inline uint16_t f32_to_f16_sr(float val, uint32_t seed) {
  return static_cast<uint16_t>(
      f32_to_binary_float_sr(val, seed, 5, 10, 15, 15, -14, 0x7FFFu, 0x7C00u));
}

/// @brief Convert a float to 16-bit BFloat16 with stochastic rounding.
inline uint16_t f32_to_bf16_sr(float val, uint32_t seed) {
  return static_cast<uint16_t>(
      f32_to_binary_float_sr(val, seed, 8, 7, 127, 127, -126, 0x7FFFu, 0x7F80u));
}

/// @brief Convert a float to 8-bit E4M3FN FP8 with stochastic rounding.
inline uint8_t f32_to_fp8_e4m3_sr(float val, uint32_t seed) {
  return static_cast<uint8_t>(f32_to_binary_float_sr(val, seed, 4, 3, 7, 8, -6, 0x7Fu, 0x7Fu));
}

/// @brief Convert a float to 8-bit E5M2 BF8 with stochastic rounding.
inline uint8_t f32_to_bf8_e5m2_sr(float val, uint32_t seed) {
  return static_cast<uint8_t>(f32_to_binary_float_sr(val, seed, 5, 2, 15, 15, -14, 0x7Fu, 0x7Cu));
}

// OCP MX low-precision formats used by gfx1250 WMMA matrix format fields.
// Encoding parameters match HIP's amd_hip_ocp_host.hpp fallback model.
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

/// @brief Convert a 4-bit OCP FP4 E2M1 value to float.
inline float fp4_e2m1_to_f32(uint8_t v) { return ocp_mx_to_f32(v & 0xFu, 2, 1, 1); }

/// @brief Convert a float to 4-bit OCP FP4 E2M1 with round-to-nearest-even.
inline uint8_t f32_to_fp4_e2m1_rne(float v) { return f32_to_ocp_mx_rne(v, 2, 1, 1); }

/// @brief Convert a float to 4-bit OCP FP4 E2M1 with stochastic rounding.
inline uint8_t f32_to_fp4_e2m1_sr(float v, uint32_t seed) {
  return static_cast<uint8_t>(f32_to_binary_float_sr(v, seed, 2, 1, 1, 2, 0, 0x7u, 0x7u));
}

/// @brief Convert a 6-bit OCP FP6 E2M3 value to float.
inline float fp6_e2m3_to_f32(uint8_t v) { return ocp_mx_to_f32(v & 0x3Fu, 2, 3, 1); }

/// @brief Convert a float to 6-bit OCP FP6 E2M3 with round-to-nearest-even.
inline uint8_t f32_to_fp6_e2m3_rne(float v) { return f32_to_ocp_mx_rne(v, 2, 3, 1); }

/// @brief Convert a float to 6-bit OCP FP6 E2M3 with stochastic rounding.
inline uint8_t f32_to_fp6_e2m3_sr(float v, uint32_t seed) {
  return static_cast<uint8_t>(f32_to_binary_float_sr(v, seed, 2, 3, 1, 2, 0, 0x3Fu, 0x3Fu));
}

/// @brief Convert a 6-bit OCP BF6 E3M2 value to float.
inline float bf6_e3m2_to_f32(uint8_t v) { return ocp_mx_to_f32(v & 0x3Fu, 3, 2, 3); }

/// @brief Convert a float to 6-bit OCP BF6 E3M2 with round-to-nearest-even.
inline uint8_t f32_to_bf6_e3m2_rne(float v) { return f32_to_ocp_mx_rne(v, 3, 2, 3); }

/// @brief Convert a float to 6-bit OCP BF6 E3M2 with stochastic rounding.
inline uint8_t f32_to_bf6_e3m2_sr(float v, uint32_t seed) {
  return static_cast<uint8_t>(f32_to_binary_float_sr(v, seed, 3, 2, 3, 4, -2, 0x3Fu, 0x3Fu));
}

} // namespace util

#endif // UTIL_DATA_TYPES_H_
