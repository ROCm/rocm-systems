/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "amd_hip_mx_common.h"
#include "amd_hip_fp8.h"

#include "amd_hip_ocp_host.hpp"

#if defined(__HIPCC_RTC__)
#define __FP4_HOST_DEVICE__ __device__
#define __FP4_HOST_DEVICE_STATIC__ __FP4_HOST_DEVICE__ static
#else
#define __FP4_HOST_DEVICE__ __host__ __device__
#define __FP4_HOST_DEVICE_STATIC__ __FP4_HOST_DEVICE__ static inline
#endif  // __HIPCC_RTC__

typedef __hip_fp8_storage_t __hip_fp4_storage_t;
typedef __hip_fp8_storage_t __hip_fp4x2_storage_t;
typedef __hip_fp8x2_storage_t __hip_fp4x4_storage_t;

static_assert(sizeof(__hip_fp4_storage_t[4]) == sizeof(uint32_t), "");
static_assert(sizeof(__hip_fp4x2_storage_t[4]) == sizeof(uint32_t), "");
static_assert(sizeof(__hip_fp4x4_storage_t[2]) == sizeof(uint32_t), "");

enum __hip_fp4_interpretation_t {
  __HIP_E2M1 = 0,
};

// Note: Ignore rounding input on AMD GPUs for now. At the moment AMD GPUs do not support rounding
// modes, all the inputs are rounded to nearest or use an input to do stochastic rounding.
// We hide the rounding variable to not trigger the unused variable compiler warning.
__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_bfloat16raw_to_fp4(
    const __hip_bfloat16_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
        u.ui32, internal::hipbf162_to_bf16x2(__hip_bfloat162{x, 0}), 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  union {
    __hip_bfloat16_raw bhalf_raw;
    __bf16 bf16;
  } bhalf{x};
  __amd_bf16x8_storage_t bf16x8{bhalf.bf16, bhalf.bf16, bhalf.bf16, bhalf.bf16,
                                bhalf.bf16, bhalf.bf16, bhalf.bf16, bhalf.bf16};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16(bf16x8, 1.0f);
  return u.fp4[0];
#else
  u.ui32 = fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(
      internal::hipbf16_to_bf16(x), 0 /* scale */);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_bfloat16raw2_to_fp4x2(
    const __hip_bfloat162_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u.ui32, internal::hipbf162_to_bf16x2(x),
                                                       1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  auto bf16x2 = internal::hipbf162_to_bf16x2(x);
  __amd_bf16x8_storage_t bf16x8{bf16x2[0], bf16x2[1], bf16x2[0], bf16x2[1],
                                bf16x2[0], bf16x2[1], bf16x2[0], bf16x2[1]};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16(bf16x8, 1.0f);
  return u.fp4x2[0];
#else
  auto bf16x2 = internal::hipbf162_to_bf16x2(x);
  u.ui32 |=
      fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(bf16x2[1], 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |=
      fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(bf16x2[0], 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t
__hip_cvt_double_to_fp4(const double x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
                        const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, float(x), 0.0f, 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{float(x), float(x), float(x), float(x),
                                 float(x), float(x), float(x), float(x)};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4[0];
#else
  u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M1, true>(float(x), 0 /* scale */);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_double2_to_fp4x2(
    const double2 x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, float(x.x), float(x.y),
                                                      1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{float(x.x), float(x.y), float(x.x), float(x.y),
                                 float(x.x), float(x.y), float(x.x), float(x.y)};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4x2[0];
#else
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(float(x.y), 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(float(x.x), 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t
__hip_cvt_float_to_fp4(const float x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
                       const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, x, 0.0f, 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{x, x, x, x, x, x, x, x};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4[0];
#else
  u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x, 0 /*scale*/);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t
__hip_cvt_float2_to_fp4x2(const float2 x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
                          const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, x.x, x.y, 1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{x.x, x.y, x.x, x.y, x.x, x.y, x.x, x.y};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4x2[0];
#else
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x.y, 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x.x, 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __half_raw __hip_cvt_fp4_to_halfraw(
    const __hip_fp4_storage_t x, const __hip_fp4_interpretation_t /* fp4_interpretation */) {
  __half2_raw ret;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  ret.data =
      __amd_fp16x2_storage_t{__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_fp4)
                                 ? __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(x, 0, 0)
                                 : __amd_fp16x2_storage_t{}};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  static_assert(sizeof(__amd_fp16x2_storage_t[4]) == sizeof(__amd_fp16x8_storage_t));
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    __amd_fp16x8_storage_t fp16x8;
    __amd_fp16x2_storage_t fp16x2[4];
  } uhalf{};
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.fp4x2[0] = x | (x << 4);
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f16_fp4))
    uhalf.fp16x8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp4(u.ui32, 0x7F7Fu, 0);
  ret.data = uhalf.fp16x2[0];
#else
  using namespace fcbx;
  ret.data =
      __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x & 0xFu, 0),
                             to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x >> 4, 0)};
#endif
  return ret.x;
}

__FP4_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp4x2_to_halfraw2(
    const __hip_fp4x2_storage_t x, const __hip_fp4_interpretation_t /* fp4_interpretation */) {
  __half2_raw ret;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  ret.data =
      __amd_fp16x2_storage_t{__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_fp4)
                                 ? __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(x, 0, 0)
                                 : __amd_fp16x2_storage_t{}};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  static_assert(sizeof(__amd_fp16x2_storage_t[4]) == sizeof(__amd_fp16x8_storage_t));
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    __amd_fp16x8_storage_t fp16x8;
    __amd_fp16x2_storage_t fp16x2[4];
  } uhalf{};
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.fp4x2[0] = x;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f16_fp4))
    uhalf.fp16x8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp4(u.ui32, 0x7F7Fu, 0);
  ret.data = uhalf.fp16x2[0];
#else
  using namespace fcbx;
  ret.data =
      __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x & 0xFu, 0),
                             to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x >> 4, 0)};
#endif
  return ret;
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_halfraw_to_fp4(
    const __half_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(
        u.ui32, internal::half2_to_f16x2(__half2{x, 0}), 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  auto val = internal::half2_to_f16x2(__half2{x, x});
  __amd_fp16x8_storage_t fp16x8{val[0], val[1], val[0], val[1], val[0], val[1], val[0], val[1]};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16(fp16x8, 1.0f);
  return u.fp4[0] >> 4;
#else
  u.ui32 = fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M1, true>(
      internal::half_to_f16(x), 0 /* scale */);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_halfraw2_to_fp4x2(
    const __half2_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(u.ui32, internal::half2_to_f16x2(x),
                                                      1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  auto val = internal::half2_to_f16x2(x);
  __amd_fp16x8_storage_t fp16x8{val[0], val[1], val[0], val[1], val[0], val[1], val[0], val[1]};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16(fp16x8, 1.0f);
  return u.fp4x2[0];
#else
  auto fp16x2 = internal::half2_to_f16x2(x);
  u.ui32 |=
      fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M1, true>(fp16x2[1], 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |=
      fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M1, true>(fp16x2[0], 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

struct __hip_fp4_e2m1 {
  __hip_fp4_storage_t __x;

 public:
  __FP4_HOST_DEVICE__ __hip_fp4_e2m1() = default;

#if !defined(__HIP_NO_FP4_CONVERSIONS__)
  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const __half f)
      : __x(__hip_cvt_halfraw_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const __hip_bfloat16 f)
      : __x(__hip_cvt_bfloat16raw_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__
  explicit __hip_fp4_e2m1(const double f)
      : __x(__hip_cvt_double_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const float f)
      : __x(__hip_cvt_float_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const long long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const short int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned long long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned short int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}
#endif  // #if !defined(__HIP_NO_FP4_CONVERSIONS__)

#if !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
  __FP4_HOST_DEVICE__ operator __half_raw() const {
    return __hip_cvt_fp4_to_halfraw(__x, __HIP_E2M1);
  }

  __FP4_HOST_DEVICE__ operator __hip_bfloat16_raw() const {
    static_assert(sizeof(__hip_bfloat16_raw[2]) == sizeof(__amd_bf16x2_storage_t), "");
    union {
      __hip_bfloat16_raw bf16_raw[2];
      __amd_bf16x2_storage_t bf16x2;
    } u{};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4))
      u.bf16x2 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(__x, 1.0f /* scale */, 0);
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    static_assert(sizeof(__amd_bf16x2_storage_t[4]) == sizeof(__amd_bf16x8_storage_t));
    static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
    union {
      __amd_bf16x8_storage_t bf16x8;
      __amd_bf16x2_storage_t bf16x2[4];
    } bhalf{};
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u_in{0};
    u_in.fp4x2[0] = __x | __x << 4;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_bf16_fp4))
      bhalf.bf16x8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(u_in.ui32, 0x7F7Fu, 0);
    u.bf16x2[0] = bhalf.bf16x8[0];
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x & 0xFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return u.bf16_raw[0];
  }

  __FP4_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto ret = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                   ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x, 1.0f /* scale */, 0)
                   : __amd_floatx2_storage_t{};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.fp4x2[0] = __x | __x << 4;
    __amd_floatx8_storage_t ret{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      ret = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu, 0);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return ret[0];
  }

  __FP4_HOST_DEVICE__ operator double() const { return double(float(*this)); }
#endif  // !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
};

/* FP4x2 E2M1 */
struct __hip_fp4x2_e2m1 {
  __hip_fp4x2_storage_t __x;

  __FP4_HOST_DEVICE__ __hip_fp4x2_e2m1() = default;

#if !defined(__HIP_NO_FP4_CONVERSIONS__)
  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const __half2 f)
      : __x(__hip_cvt_halfraw2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const __hip_bfloat162 f)
      : __x(__hip_cvt_bfloat16raw2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const double2 f)
      : __x(__hip_cvt_double2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const float2 f)
      : __x(__hip_cvt_float2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

#endif  // #if !defined(__HIP_NO_FP4_CONVERSIONS__)

#if !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
  __FP4_HOST_DEVICE__ operator __half2_raw() const {
    return __hip_cvt_fp4x2_to_halfraw2(__x, __HIP_E2M1);
  }

  __FP4_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    static_assert(sizeof(__hip_bfloat162_raw) == sizeof(__amd_bf16x2_storage_t), "");
    union {
      __hip_bfloat162_raw bf162_raw;
      __amd_bf16x2_storage_t bf16x2;
    } u{};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4))
      u.bf16x2 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(__x, 1.0f /* scale */, 0);
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    static_assert(sizeof(__amd_bf16x2_storage_t[4]) == sizeof(__amd_bf16x8_storage_t));
    static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
    union {
      __amd_bf16x8_storage_t bf16x8;
      __amd_bf16x2_storage_t bf16x2[4];
    } bhalf{};
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u_in{0};
    u_in.fp4x2[0] = __x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_bf16_fp4))
      bhalf.bf16x8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(u_in.ui32, 0x7F7Fu, 0);
    u.bf16x2[0] = bhalf.bf16x8[0];
    u.bf16x2[1] = bhalf.bf16x8[1];
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x & 0xFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return u.bf162_raw;
  }

  __FP4_HOST_DEVICE__ operator float2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto fp32x2 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                      ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x, 1.0f /* scale */, 0)
                      : __amd_floatx2_storage_t{};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.fp4x2[0] = __x;

    // Even though it's x8, we call it x2 so that we have a common return line
    __amd_floatx8_storage_t fp32x2{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      fp32x2 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu, 0);
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                          to_float<float, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return float2(fp32x2[0], fp32x2[1]);
  }

  __FP4_HOST_DEVICE__ operator double2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto fp32x2 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                      ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x, 1.0f /* scale */, 0)
                      : __amd_floatx2_storage_t{};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.fp4x2[0] = __x;
    // Although it's x8 but we still call it x2 since we will use it in common code.
    __amd_floatx8_storage_t fp32x2{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      fp32x2 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu, 0);
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                          to_float<float, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return double2(fp32x2[0], fp32x2[1]);
  }
#endif  // !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
};

/* FP4x4 E2M1 */
struct __hip_fp4x4_e2m1 {
  __hip_fp4x4_storage_t __x;

  __FP4_HOST_DEVICE__ inline __hip_fp4x4_e2m1() = default;

#if !defined(__HIP_NO_FP4_CONVERSIONS__)
  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const __half2 low, const __half2 high)
      : __x(__hip_cvt_halfraw2_to_fp4x2(high, __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_halfraw2_to_fp4x2(low, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const __hip_bfloat162 low,
                                                       const __hip_bfloat162 high)
      : __x(__hip_cvt_bfloat16raw2_to_fp4x2(high, __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_bfloat16raw2_to_fp4x2(low, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const double4 f)
      : __x(__hip_cvt_double2_to_fp4x2(double2(f.z, f.w), __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_double2_to_fp4x2(double2(f.x, f.y), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const float4 f)
      : __x(__hip_cvt_float2_to_fp4x2(float2(f.z, f.w), __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_float2_to_fp4x2(float2(f.x, f.y), __HIP_E2M1, hipRoundNearest)) {}
#endif  // #if !defined(__HIP_NO_FP4_CONVERSIONS__)

#if !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
  __FP4_HOST_DEVICE__ operator float4() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto fp32x2_1 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                        ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x & 0xFFu, 1.0f /* scale */, 0)
                        : __amd_floatx2_storage_t{};
    auto fp32x2_2 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                        ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x >> 8, 1.0f /* scale */, 0)
                        : __amd_floatx2_storage_t{};
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __hip_fp4x4_storage_t fp4x4[2];
    } u{0};
    u.fp4x4[0] = __x;
    __amd_floatx8_storage_t fp32x8{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      fp32x8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu, 0);
    return float4{fp32x8[0], fp32x8[1], fp32x8[2], fp32x8[3]};
#else
    using namespace fcbx;
    auto fp32x2_1 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>((__x >> 4) & 0xFu, 0)};
    auto fp32x2_2 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>((__x >> 8) & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>(__x >> 12, 0)};
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
#endif
  }

  __FP4_HOST_DEVICE__ operator double4() const {
    auto fp32 = float4(*this);
    return double4{fp32.x, fp32.y, fp32.z, fp32.w};
  }
#endif  // !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
};
