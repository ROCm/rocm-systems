/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


#pragma once

#include "amd_hip_fp16.h"

#if __GNUC__ >= 13
#define HIP_GNUC_BF16_TYPE 1
#else
#define HIP_GNUC_BF16_TYPE 0
#endif  // GNUC >= 13

#if defined(__cplusplus)
#define ALIGN_IT(x) alignas(x)
#define MAYBE_UNUSED [[maybe_unused]]
#else
#define ALIGN_IT(x) __attribute__((aligned(x)))
#define MAYBE_UNUSED __attribute__((unused))
#endif

static_assert(sizeof(unsigned short) == 2);

typedef struct ALIGN_IT(2) {
  unsigned short x;
} __hip_bfloat16_raw;

typedef struct ALIGN_IT(4) {
  unsigned short x;
  unsigned short y;
} __hip_bfloat162_raw;

static inline __hip_bfloat16_raw __hip_float_to_bfloat_cvt(float in) {
#if HIP_GNUC_BF16_TYPE
  static_assert(sizeof(__bf16) == sizeof(__hip_bfloat16_raw));
  union {
    __bf16 bf;
    __hip_bfloat16_raw br;
  } u{static_cast<__bf16>(in)};
  return u.br;
#else
  static_assert(sizeof(float) == sizeof(unsigned int));
  union {
    float f32;
    unsigned int ui;
  } u{in};
  if (~u.ui & 0x7f800000) {
    u.ui += 0x7fff + ((u.ui >> 16) & 1);  // Round to nearest, round to even
  } else if (u.ui & 0xffff) {
    u.ui |= 0x10000;  // Preserve signaling NaN
  }
  return __hip_bfloat16_raw{static_cast<unsigned short>(u.ui >> 16)};
#endif
}

static inline float __hip_bfloat_to_float(__hip_bfloat16_raw br) {
#if HIP_GNUC_BF16_TYPE
  union {
    unsigned short us;
    __bf16 bf;
  } u{br.x};
  return u.bf;
#else
  union {
    unsigned int ui;
    float f32;
  } u = {static_cast<unsigned int>(br.x) << 16};
  return u.f32;
#endif
}

// Fwd decls
struct __hip_bfloat16;
MAYBE_UNUSED static __hip_bfloat16 __float2bfloat16(const float in);
MAYBE_UNUSED static float __bfloat162float(__hip_bfloat16 a);

#if defined(__cplusplus)
struct ALIGN_IT(2) __hip_bfloat16 {
 protected:
  unsigned short __x;

 public:
  __hip_bfloat16() = default;
  constexpr __hip_bfloat16(const __hip_bfloat16_raw& br) : __x(br.x) {}
  explicit __hip_bfloat16(const __half h) { __x = __float2bfloat16(__half2float(h)).__x; }
  __hip_bfloat16(const float f) { __x = __float2bfloat16(f).__x; }

  __hip_bfloat16& operator=(const __hip_bfloat16_raw& br) {
    __x = br.x;
    return *this;
  }
  __hip_bfloat16& operator=(const float f) {
    __x = __float2bfloat16(f).__x;
    return *this;
  }


  operator __hip_bfloat16_raw() const { return __hip_bfloat16_raw{__x}; }
  operator float() const { return __bfloat162float(__hip_bfloat16_raw{__x}); }
};

struct ALIGN_IT(4) __hip_bfloat162 {
  __hip_bfloat16 x;
  __hip_bfloat16 y;

 public:
  __hip_bfloat162() = default;
  __hip_bfloat162(__hip_bfloat162&& in) : x(in.x), y(in.y) {}
  constexpr __hip_bfloat162(const __hip_bfloat16& a, const __hip_bfloat16& b) : x(a), y(b) {}
  __hip_bfloat162(const __hip_bfloat162& in) : x(in.x), y(in.y) {}

  operator __hip_bfloat162_raw() const {
    return __hip_bfloat162_raw{__hip_bfloat16_raw(x).x, __hip_bfloat16_raw(y).x};
  }

  __hip_bfloat162& operator=(const __hip_bfloat162_raw& b2r) {
    x = __hip_bfloat16_raw{b2r.x};
    y = __hip_bfloat16_raw{b2r.y};
    return *this;
  }
};

MAYBE_UNUSED static inline __hip_bfloat16 __float2bfloat16(const float in) {
  return __hip_float_to_bfloat_cvt(in);
}

MAYBE_UNUSED static inline float __bfloat162float(__hip_bfloat16 a) {
  return __hip_bfloat_to_float(a);
}

MAYBE_UNUSED static inline __hip_bfloat16 __ushort_as_bfloat16(const unsigned short int a) {
  return __hip_bfloat16(__hip_bfloat16_raw{a});
}
#endif  // __cplusplus

#undef ALIGN_IT
#undef MAYBE_UNUSED
