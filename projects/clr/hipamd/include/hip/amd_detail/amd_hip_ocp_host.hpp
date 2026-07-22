/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "amd_hip_ocp_types.h"
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <cmath>
#endif

namespace fcbx {
constexpr __hip_int8_t OCP_SCALE_EXP_NAN = -128;

enum class Encoding : size_t {
  E2M1 = 0,
  E2M3,
  E3M2,
  E4M3,
  E4M3Mx,
  E4M3Nanoo,
  E5M2,
  E5M2Mx,
  E5M2Nanoo,

  E5M10,  // FP16
  E8M7,   // BF16

  IEEE754,

  // Keep this one last
  NumEncodings,
};
enum fp16 : __hip_uint16_t {};
enum bf16 : __hip_uint16_t {};

// Map a source storage type to its Encoding as a true compile-time constant.
// Using a traits struct (rather than an IIFE assigned to `const auto`) guarantees
// the value is usable as a non-type template argument across all supported
// toolchains, including the comgr-bundled Clang used by HIPRTC on Windows where
// implicit-constexpr lambda rules differ from the Linux host toolchain.
template <typename T> struct EncodingOf;
template <> struct EncodingOf<float> {
  static constexpr Encoding value = Encoding::IEEE754;
};
template <> struct EncodingOf<__amd_fp16_storage_t> {
  static constexpr Encoding value = Encoding::E5M10;
};
template <> struct EncodingOf<__amd_bf16_storage_t> {
  static constexpr Encoding value = Encoding::E8M7;
};

struct Float {
  __hip_int32_t ExpBias;
  __hip_uint32_t ExpBits;
  __hip_uint32_t ExpMask;
  __hip_uint32_t ManBits;
  __hip_uint32_t ManMask;
  __hip_int32_t MaxExp;
  __hip_int32_t MinExp;
  bool MxScale;
  bool HasNaN;
  bool HasInf;
};

static const float ieee754_nan = __hip_internal::NumericLimits<float>::quiet_NaN();
static const float ieee754_inf = __hip_internal::NumericLimits<float>::infinity();

__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t U32(float f) {
  static_assert(sizeof(__hip_uint32_t) == sizeof(float), "");
  union {
    float f32;
    __hip_uint32_t ui32;
  } u{f};
  return u.ui32;
}

__OCP_FP_HOST_DEVICE_STATIC__ float F32(__hip_uint32_t u32) {
  static_assert(sizeof(__hip_uint32_t) == sizeof(float), "");
  union {
    __hip_uint32_t ui32;
    float f32;
  } u{u32};
  return u.f32;
}

constexpr __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t bitmask(__hip_uint32_t bits) {
  if (bits < 1) return 0;
  return ((__hip_uint32_t)1 << bits) - 1;
}

struct EncodingTable {
  Float __elements[(size_t)Encoding::NumEncodings];
#if __cplusplus >= 201402L
  constexpr Float& operator[](size_t i) { return __elements[i]; }
#else
  Float& operator[](size_t i) { return __elements[i]; }
#endif
  constexpr const Float& operator[](size_t i) const { return __elements[i]; }
};

constexpr EncodingTable init() {
  EncodingTable a{};

  a[(size_t)Encoding::E2M1] = {
      .ExpBias = 1,
      .ExpBits = 2,
      .ExpMask = bitmask(2),
      .ManBits = 1,
      .ManMask = bitmask(1),
      .MaxExp = 2,
      .MinExp = 0,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E2M3] = {
      .ExpBias = 1,
      .ExpBits = 2,
      .ExpMask = bitmask(2),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 2,
      .MinExp = 0,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E3M2] = {
      .ExpBias = 3,
      .ExpBits = 3,
      .ExpMask = bitmask(3),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 4,
      .MinExp = -2,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3] = {
      .ExpBias = 7,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 8,
      .MinExp = -6,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3Mx] = {
      .ExpBias = 7,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 8,
      .MinExp = -6,
      .MxScale = true,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3Nanoo] = {
      .ExpBias = 8,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 7,
      .MinExp = -7,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E5M2] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M2Mx] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = true,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M2Nanoo] = {
      .ExpBias = 16,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -15,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M10] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 10,
      .ManMask = bitmask(10),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E8M7] = {
      .ExpBias = 127,
      .ExpBits = 8,
      .ExpMask = bitmask(8),
      .ManBits = 7,
      .ManMask = bitmask(7),
      .MaxExp = 127,
      .MinExp = -126,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::IEEE754] = {
      .ExpBias = 127,
      .ExpBits = 8,
      .ExpMask = bitmask(8),
      .ManBits = 23,
      .ManMask = bitmask(23),
      .MaxExp = 127,
      .MinExp = -126,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  return a;
}

static constexpr auto encodings = init();

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t exponentbits(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  return (val >> enc.ManBits) & enc.ExpMask;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t mantissa(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  return val & enc.ManMask;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool issubnorm(__hip_uint32_t val) {
  switch (E) {
    default:
      return exponentbits<E, sat>(val) == 0 && mantissa<E, sat>(val) != 0;
  }

  __builtin_trap();
  // Unreachable
  return false;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_int32_t exponent(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  auto unbiased_exp = exponentbits<E, sat>(val);
  unbiased_exp = issubnorm<E, sat>(val) ? 1 : unbiased_exp;
  return (__hip_int32_t)unbiased_exp - enc.ExpBias;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t signbit(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  return (val >> (enc.ExpBits + enc.ManBits)) & 1;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t nan(__hip_uint32_t sign) {
  const auto& enc = encodings[(size_t)E];

  switch (E) {
    case Encoding::E2M1:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0b0111;
    case Encoding::E2M3:
    case Encoding::E3M2:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0b011111;
    case Encoding::E4M3:
    case Encoding::E4M3Mx:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0x7f;
    case Encoding::E5M2:
    case Encoding::E5M2Mx:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0x7e;
    case Encoding::E4M3Nanoo:
    case Encoding::E5M2Nanoo:
      return 0b10000000;
    case Encoding::E5M10:
    case Encoding::E8M7:
      return (sign << (enc.ExpBits + enc.ManBits)) | (enc.ExpMask << enc.ManBits) | enc.ManMask;
    case Encoding::IEEE754:
      return U32(sign ? __hip_internal::copysign(ieee754_nan, -1.0F) : ieee754_nan);
    default:
      __builtin_trap();
      return 0;
  }
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t zero(__hip_uint32_t sign) {
  const auto& enc = encodings[(size_t)E];

  switch (E) {
    case Encoding::E2M1:
    case Encoding::E2M3:
    case Encoding::E3M2:
    case Encoding::E4M3:
    case Encoding::E4M3Mx:
    case Encoding::E5M2:
    case Encoding::E5M2Mx:
    case Encoding::E5M10:
    case Encoding::E8M7:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0;
    case Encoding::E4M3Nanoo:
    case Encoding::E5M2Nanoo:
      return 0;
    case Encoding::IEEE754:
      return U32(sign ? __hip_internal::copysign(0.0F, -1.0F) : 0.0F);
    default:
      __builtin_trap();
      return 0;
  }
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t inf(__hip_uint32_t sign) {
  const auto& enc = encodings[(size_t)E];

  switch (E) {
    case Encoding::E2M1:
    case Encoding::E2M3:
    case Encoding::E3M2:
      return nan<E, sat>(sign);
    case Encoding::E4M3:
    case Encoding::E4M3Mx:
    case Encoding::E4M3Nanoo:
    case Encoding::E5M2Nanoo:
      if constexpr (sat) {
        sign <<= enc.ExpBits + enc.ManBits;
        return sign | 0b01111111;
      }

      return nan<E, sat>(sign);
    case Encoding::E5M2:
    case Encoding::E5M2Mx:
      sign <<= enc.ExpBits + enc.ManBits;
      if constexpr (sat) {
        return sign | 0b01111011;
      }

      return sign | 0b01111100;
    case Encoding::E5M10:
    case Encoding::E8M7:
      sign <<= enc.ExpBits + enc.ManBits;
      return sign | (enc.ExpMask << enc.ManBits) | 0;
    case Encoding::IEEE754:
      return U32(sign ? __hip_internal::copysign(ieee754_inf, -1.0F) : ieee754_inf);
    default:
      __builtin_trap();
      return 0;
  }
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool isnan(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  if (!enc.HasNaN) return false;

  if constexpr (E == Encoding::E4M3Mx || E == Encoding::E4M3 || E == Encoding::E4M3Nanoo ||
                E == Encoding::E5M2Nanoo)
    return nan<E, sat>(signbit<E, sat>(val)) == val;

  return exponentbits<E, sat>(val) == enc.ExpMask && mantissa<E, sat>(val) != 0;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool isinf(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  if (!enc.HasInf) return false;

  if constexpr (E == Encoding::E5M10 || E == Encoding::E8M7) {
    return exponentbits<E, sat>(val) == enc.ExpMask && mantissa<E, sat>(val) == 0;
  }

  return inf<E, sat>(signbit<E, sat>(val)) == val;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool iszero(__hip_uint32_t val) {
  return zero<E, sat>(signbit<E, sat>(val)) == val;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool inrange(__hip_uint32_t val) {
  return !(isnan<E, sat>(val) || isinf<E, sat>(val));
}

template <typename T> __OCP_FP_HOST_DEVICE_STATIC__ T makenan(Encoding E, __hip_uint32_t sign) {
  switch (E) {
    case Encoding::E5M10:
      return (T)nan<Encoding::E5M10, false>(sign);
    case Encoding::E8M7:
      return (T)nan<Encoding::E8M7, false>(sign);
    case Encoding::IEEE754:
      return (T)F32(nan<Encoding::IEEE754, false>(sign));
    default:
      __builtin_trap();
      // Unreachable
      return T();
  }
}

template <typename T> __OCP_FP_HOST_DEVICE_STATIC__ T makeinf(Encoding E, __hip_uint32_t sign) {
  switch (E) {
    case Encoding::E5M10:
      return (T)inf<Encoding::E5M10, false>(sign);
    case Encoding::E8M7:
      return (T)inf<Encoding::E8M7, false>(sign);
    case Encoding::IEEE754:
      return (T)F32(inf<Encoding::IEEE754, false>(sign));
    default:
      __builtin_trap();
      // Unreachable
      return T();
  }
}

template <typename T> __OCP_FP_HOST_DEVICE_STATIC__ T makezero(Encoding E, __hip_uint32_t sign) {
  switch (E) {
    case Encoding::E5M10:
      return (T)zero<Encoding::E5M10, false>(sign);
    case Encoding::E8M7:
      return (T)zero<Encoding::E8M7, false>(sign);
    case Encoding::IEEE754:
      return (T)F32(zero<Encoding::IEEE754, false>(sign));
    default:
      __builtin_trap();
      // Unreachable
      return T();
  }
}

template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ T to_float(__hip_uint32_t u32, __hip_int8_t scale_exp) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 && E != Encoding::E8M7, "");

  const auto& enc = encodings[(size_t)E];
  static_assert(__hip_internal::is_same<T, float>() ||
                    __hip_internal::is_same<T, __amd_fp16_storage_t>() ||
                    __hip_internal::is_same<T, __amd_bf16_storage_t>(),
                "to_float: unsupported destination type T");
  constexpr Encoding dstE = EncodingOf<T>::value;
  const auto& dstEnc = encodings[(size_t)dstE];

  if (isnan<E, sat>(u32) || (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return makenan<T>(dstE, signbit<E, sat>(u32));

  if (isinf<E, sat>(u32)) return makeinf<T>(dstE, signbit<E, sat>(u32));

  if (iszero<E, sat>(u32)) return makezero<T>(dstE, signbit<E, sat>(u32));

  auto dstMan = mantissa<E, sat>(u32) << (dstEnc.ManBits - enc.ManBits);
  auto dstExp = (__hip_uint32_t)(exponent<E, sat>(u32) + dstEnc.ExpBias);
  dstExp &= dstEnc.ExpMask;

  if (issubnorm<E, sat>(u32)) {
    auto leadbit = (__hip_uint32_t)1 << dstEnc.ManBits;
    while ((dstMan & leadbit) == 0) {
      dstMan <<= 1;
      dstExp -= 1;
    }

    dstMan &= dstEnc.ManMask;
  }

  auto sign = signbit<E, sat>(u32) << (dstEnc.ExpBits + dstEnc.ManBits);

  if (enc.MxScale) {
    __hip_int32_t exp = dstExp - dstEnc.ExpBias;
    __hip_int32_t tmp = exp + (__hip_int32_t)scale_exp;
    size_t diff = abs(tmp - dstEnc.MinExp);


    if (tmp < dstEnc.MinExp) {
      if (diff > dstEnc.ManBits + 1) return makezero<T>(dstE, signbit<E, sat>(u32));

      dstExp = 0;  // Subnormal
      dstMan |= (__hip_uint32_t)1 << dstEnc.ManBits;

      auto roundBitShift = diff - 1;
      auto roundBit = (dstMan & ((__hip_uint32_t)1 << roundBitShift)) != 0;
      auto stickyMask = ((__hip_uint32_t)1 << roundBitShift) - 1;
      auto stickyBits = dstMan & stickyMask;
      auto odd = (dstMan & ((__hip_uint32_t)1 << diff)) != 0;

      dstMan >>= diff;

      if ((roundBit && stickyBits != 0) || (roundBit && odd)) {
        ++dstMan;
        if ((dstMan & ((__hip_uint32_t)1 << dstEnc.ManBits)) != 0) ++dstExp;
      }

      dstMan &= dstEnc.ManMask;
    } else {
      dstExp = (__hip_uint32_t)(exp + scale_exp + dstEnc.ExpBias);

      // Overflow: return infinity (gfx950 HW behavior)
      if (dstExp >= dstEnc.ExpMask) return makeinf<T>(dstE, signbit<E, sat>(u32));

      dstExp &= dstEnc.ExpMask;
    }
  }

  auto dst = sign | (dstExp << dstEnc.ManBits) | dstMan;

  union {
    float f32;
    __amd_fp16_storage_t fp16[2];
    __amd_bf16_storage_t bf16[2];
    __hip_uint32_t u32;
  } u;
  u.u32 = dst;
  if constexpr (__hip_internal::is_same<T, float>())
    return u.f32;
  else if constexpr (__hip_internal::is_same<T, __amd_fp16_storage_t>())
    return u.fp16[0];
  else if constexpr (__hip_internal::is_same<T, __amd_bf16_storage_t>())
    return u.bf16[0];
  else
    __builtin_trap();
}

template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t from_float_sr(T f, __hip_uint32_t seed, __hip_int8_t scale_exp) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 && E != Encoding::E8M7, "");
  static_assert(sizeof(__amd_fp16_storage_t[2]) == sizeof(float), "");
  static_assert(sizeof(__amd_bf16_storage_t[2]) == sizeof(float), "");
  union {
    float f32;
    __amd_fp16_storage_t fp16[2];
    __amd_bf16_storage_t bf16[2];
    __hip_uint32_t u32;
  } u{0};

  if constexpr (__hip_internal::is_same<T, float>())
    u.f32 = f;
  else if constexpr (__hip_internal::is_same<T, __amd_fp16_storage_t>())
    u.fp16[0] = f;
  else if constexpr (__hip_internal::is_same<T, __amd_bf16_storage_t>())
    u.bf16[0] = f;
  else
    __builtin_trap();

  const auto& enc = encodings[(size_t)E];
  static_assert(__hip_internal::is_same<T, float>() ||
                    __hip_internal::is_same<T, __amd_fp16_storage_t>() ||
                    __hip_internal::is_same<T, __amd_bf16_storage_t>(),
                "from_float_sr: unsupported source type T");
  constexpr Encoding srcE = EncodingOf<T>::value;
  const auto& srcEnc = encodings[(size_t)srcE];

  auto srcU32 = u.u32;  // (srcE == Encoding::IEEE754) ? U32(f) : (__hip_uint32_t)f;
  auto signBit = signbit<srcE, false>(srcU32);
  auto sign = signBit << (enc.ExpBits + enc.ManBits);

  if (isnan<srcE, sat>(srcU32) || (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return nan<E, sat>(signBit);

  if (isinf<srcE, sat>(srcU32)) return inf<E, sat>(signBit);

  if (iszero<srcE, sat>(srcU32)) return zero<E, sat>(signBit);

  auto srcMan = mantissa<srcE, false>(srcU32);
  auto srcExp = exponent<srcE, false>(srcU32);
  if (enc.MxScale) {
    if (issubnorm<srcE, false>(srcU32)) {
      auto leadbit = (__hip_uint32_t)1 << srcEnc.ManBits;
      while ((srcMan & leadbit) == 0) {
        srcMan <<= 1;
        srcExp -= 1;
      }

      srcMan &= srcEnc.ManMask;
    }

    srcExp -= scale_exp;
  }

  auto exp = srcExp;
  auto man = srcMan;
  bool subnorm = false;

  if (exp > enc.MaxExp) {
    return inf<E, sat>(signBit);
  } else if (exp >= enc.MinExp) {
    man = srcMan;
  } else if (exp < enc.MinExp) {
    subnorm = true;
    exp = 0;

    auto diff = (__hip_uint32_t)(enc.MinExp - srcExp);
    if (diff >= 32) {
      man = 0;
      srcMan = 0;
    } else {
      srcMan |= (__hip_uint32_t)1 << srcEnc.ManBits;
      srcMan >>= diff;
    }

    man = srcMan;
  }

  // Align random value to be one past the kept mant bit
  size_t sr_shift = (32 - srcEnc.ManBits) + enc.ManBits;

  // For stochastic-rounding we add the aligned random value to the
  // mantissa and then truncate (RTZ).
  man += seed >> sr_shift;

  // Increment exponent when mantissa overflows due to rounding
  if (man >= (__hip_uint32_t)1 << srcEnc.ManBits) ++exp;
  man >>= (srcEnc.ManBits - enc.ManBits);
  man &= enc.ManMask;

  if (exp > enc.MaxExp) return inf<E, sat>(signBit);

  auto biasedExp = (__hip_uint32_t)exp;
  if (!subnorm) biasedExp = (__hip_uint32_t)(exp + enc.ExpBias);
  biasedExp &= enc.ExpMask;

  auto val = sign | biasedExp << enc.ManBits | man;
  if (inrange<E, sat>(val))
    return val;
  else if (man == 0 && exp == 0)
    return zero<E, sat>(signBit);
  else
    return inf<E, sat>(signBit);
}


template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t from_float(T f, __hip_int8_t scale_exp) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 && E != Encoding::E8M7, "");
  static_assert(sizeof(__amd_fp16_storage_t[2]) == sizeof(float), "");
  static_assert(sizeof(__amd_bf16_storage_t[2]) == sizeof(float), "");
  union {
    float f32;
    __amd_fp16_storage_t fp16[2];
    __amd_bf16_storage_t bf16[2];
    __hip_uint32_t u32;
  } u{0};

  if constexpr (__hip_internal::is_same<T, float>())
    u.f32 = f;
  else if constexpr (__hip_internal::is_same<T, __amd_fp16_storage_t>())
    u.fp16[0] = f;
  else if constexpr (__hip_internal::is_same<T, __amd_bf16_storage_t>())
    u.bf16[0] = f;
  else
    __builtin_trap();

  const auto& enc = encodings[(size_t)E];
  static_assert(__hip_internal::is_same<T, float>() ||
                    __hip_internal::is_same<T, __amd_fp16_storage_t>() ||
                    __hip_internal::is_same<T, __amd_bf16_storage_t>(),
                "from_float: unsupported source type T");
  constexpr Encoding srcE = EncodingOf<T>::value;
  const auto& srcEnc = encodings[(size_t)srcE];

  auto srcU32 = u.u32;  // (srcE == Encoding::IEEE754) ? U32(f) : (__hip_uint32_t)f;
  auto signBit = signbit<srcE, false>(srcU32);
  auto sign = signBit << (enc.ExpBits + enc.ManBits);

  if (isnan<srcE, sat>(srcU32) || (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return nan<E, sat>(signBit);

  if (isinf<srcE, sat>(srcU32)) return inf<E, sat>(signBit);

  if (iszero<srcE, sat>(srcU32)) return zero<E, sat>(signBit);

  auto srcMan = mantissa<srcE, false>(srcU32);
  auto srcExp = exponent<srcE, false>(srcU32);
  if (enc.MxScale) {
    if (issubnorm<srcE, false>(srcU32)) {
      auto leadbit = (__hip_uint32_t)1 << srcEnc.ManBits;
      while ((srcMan & leadbit) == 0) {
        srcMan <<= 1;
        srcExp -= 1;
      }

      srcMan &= srcEnc.ManMask;
    }

    srcExp -= scale_exp;
  }

  auto exp = srcExp;
  auto man = srcMan;
  __hip_uint32_t stickyBits = 0;
  bool subnorm = false;

  if (exp > enc.MaxExp) {
    return inf<E, sat>(signBit);
  } else if (exp >= enc.MinExp) {
    man >>= srcEnc.ManBits - enc.ManBits;
  } else if (exp < enc.MinExp) {
    subnorm = true;
    exp = 0;

    auto diff = (__hip_uint32_t)(enc.MinExp - srcExp);
    if (diff >= 32) {
      man = 0;
      srcMan = 0;
    } else {
      srcMan |= (__hip_uint32_t)1 << srcEnc.ManBits;
      stickyBits = srcMan & (((__hip_uint32_t)1 << diff) - (__hip_uint32_t)1);
      srcMan >>= diff;

      man = srcMan;
      man >>= srcEnc.ManBits - enc.ManBits;
      man &= enc.ManMask;
    }
  }

  auto roundBitShift = srcEnc.ManBits - (enc.ManBits + 1);
  auto roundBit = ((srcMan >> roundBitShift) & 1) != 0;
  stickyBits |= srcMan & (((__hip_uint32_t)1 << roundBitShift) - 1);
  auto odd = (man & 1) != 0;

  if ((roundBit && stickyBits != 0) || (roundBit && odd)) {
    ++man;
    if ((man & ((__hip_uint32_t)1 << enc.ManBits)) != 0) ++exp;
    man &= enc.ManMask;
  }

  if (exp > enc.MaxExp) return inf<E, sat>(signBit);

  auto biasedExp = (__hip_uint32_t)exp;
  if (!subnorm) biasedExp = (__hip_uint32_t)(exp + enc.ExpBias);
  biasedExp &= enc.ExpMask;

  auto val = sign | biasedExp << enc.ManBits | man;
  if (inrange<E, sat>(val))
    return val;
  else if (man == 0 && exp == 0)
    return zero<E, sat>(signBit);
  else
    return inf<E, sat>(signBit);
}

// Manual little-endian, LSB-first 6-bit packing.
__OCP_FP_HOST_DEVICE_STATIC__ void pack_fp6(__hip_uint8_t* buf, int idx, __hip_uint8_t val) {
  const unsigned bit = static_cast<unsigned>(idx) * 6u;
  const unsigned byte = bit >> 3;
  const unsigned off = bit & 7u;
  val &= 0x3f;
  buf[byte] = static_cast<__hip_uint8_t>((buf[byte] & ~(0x3f << off)) | (val << off));
  if (off > 2u)
    buf[byte + 1] = static_cast<__hip_uint8_t>((buf[byte + 1] & ~(0x3f >> (8u - off))) |
                                               (val >> (8u - off)));
}

__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint8_t unpack_fp6(const __hip_uint8_t* buf, int idx) {
  const unsigned bit = static_cast<unsigned>(idx) * 6u;
  const unsigned byte = bit >> 3;
  const unsigned off = bit & 7u;
  unsigned v = static_cast<unsigned>(buf[byte]) >> off;
  if (off > 2u) v |= static_cast<unsigned>(buf[byte + 1]) << (8u - off);
  return static_cast<__hip_uint8_t>(v & 0x3f);
}

// ------------
template <typename InType, typename OutType, typename float_base_t, Encoding in_encode,
          Encoding out_encode, bool sr = false>
__OCP_FP_HOST_DEVICE_STATIC__ OutType fp6_cvt_packedx16(InType in, __hip_int8_t scale = 0,
                                                        __hip_uint32_t seed = 0) {
  // This is tightly coupled with the definitions of the amd_ocp_types
  constexpr bool in_float = __hip_internal::is_same<InType, __amd_floatx16_storage_t>::value ||
      __hip_internal::is_same<InType, __amd_fp16x16_storage_t>::value ||
      __hip_internal::is_same<InType, __amd_bf16x16_storage_t>::value;
  using other_type = __hip_internal::conditional<in_float, OutType, InType>::type;

  union {
    other_type o;
    __hip_uint8_t bytes[sizeof(other_type)];
  } u;

  if constexpr (in_float) {
    for (int i = 0; i < static_cast<int>(sizeof(other_type)); ++i) {
       u.bytes[i] = 0;
    }
    for (int i = 0; i < 16; ++i) {
      __hip_uint8_t v;
      if constexpr (sr) {
        v = static_cast<__hip_uint8_t>(
            from_float_sr<float_base_t, out_encode, true>(in[i], seed, scale));
      } else {
        v = static_cast<__hip_uint8_t>(from_float<float_base_t, out_encode, true>(in[i], scale));
      }
      pack_fp6(u.bytes, i, v);
    }
    return u.o;
  } else {
    OutType ret;
    u.o = in;
    for (int i = 0; i < 16; ++i) {
      ret[i] = to_float<float_base_t, in_encode, true>(unpack_fp6(u.bytes, i), scale);
    }
    return ret;
  }
}
// ------------

template <typename InType, typename OutType, typename float_base_t, Encoding in_encode,
          Encoding out_encode, bool sr = false>
__OCP_FP_HOST_DEVICE_STATIC__ OutType fp6_cvt_packedx32(InType in, __hip_int8_t scale = 0,
                                                        __hip_uint32_t seed = 0) {
  // This is tightly coupled with the definitions of the amd_ocp_types
  constexpr bool in_float = __hip_internal::is_same<InType, __amd_floatx32_storage_t>::value ||
                            __hip_internal::is_same<InType, __amd_fp16x32_storage_t>::value ||
                            __hip_internal::is_same<InType, __amd_bf16x32_storage_t>::value;
  using other_type = __hip_internal::conditional<in_float, OutType, InType>::type;

  union {
    other_type o;
    __hip_uint8_t bytes[sizeof(other_type)];
  } u;

  if constexpr (in_float) {
    for (int i = 0; i < static_cast<int>(sizeof(other_type)); ++i) {
      u.bytes[i] = 0;
    }
    for (int i = 0; i < 32; ++i) {
      __hip_uint8_t v;
      if constexpr (sr) {
        v = static_cast<__hip_uint8_t>(
            from_float_sr<float_base_t, out_encode, true>(in[i], seed, scale));
      } else {
        v = static_cast<__hip_uint8_t>(from_float<float_base_t, out_encode, true>(in[i], scale));
      }
      pack_fp6(u.bytes, i, v);
    }
    return u.o;
  } else {
    OutType ret;
    u.o = in;
    for (int i = 0; i < 32; ++i) {
      ret[i] = to_float<float_base_t, in_encode, true>(unpack_fp6(u.bytes, i), scale);
    }
    return ret;
  }
}
}  // namespace fcbx
