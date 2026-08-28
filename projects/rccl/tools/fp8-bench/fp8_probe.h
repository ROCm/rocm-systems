/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef FP8_PROBE_H_
#define FP8_PROBE_H_

#include <stdint.h>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <hip/hip_fp8.h>

typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bhalf8_t __attribute__((ext_vector_type(8)));
typedef short shortx2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));

enum Fp8HwIntrinsic {
  HW_HIP_FP8_E4M3 = 0,
  HW_HIP_FP8_E5M2,
  HW_HIP_FP8_E4M3_FNUZ,
  HW_HIP_FP8_E5M2_FNUZ,

  HW_CVT_PK_F32_FP8,
  HW_CVT_PK_F32_BF8,
  HW_CVT_F32_FP8,
  HW_CVT_F32_BF8,

  HW_CVT_PK_FP8_F32,
  HW_CVT_PK_BF8_F32,
  HW_CVT_SR_FP8_F32,
  HW_CVT_SR_BF8_F32,

  HW_CVT_SCALEF32_PK_F16_FP8,
  HW_CVT_SCALEF32_PK_F16_BF8,
  HW_CVT_SCALEF32_PK_FP8_F16,
  HW_CVT_SCALEF32_PK_BF8_F16,

  HW_V_PK_ADD_F32,
  HW_V_PK_ADD_F16,
  HW_FMED3F,

  HW_FP8_E4M3_ADD_PK1,
  HW_FP8_E4M3_ADD_PK2,
  HW_FP8_E5M2_ADD_PK1,
  HW_FP8_E5M2_ADD_PK2,

  // gfx1250 / gfx12: direct f16 <-> fp8 (pk2 and scalar)
  HW_CVT_PK_F16_FP8,
  HW_CVT_PK_F16_BF8,
  HW_CVT_PK_FP8_F16,
  HW_CVT_PK_BF8_F16,
  HW_CVT_F16_FP8,
  HW_CVT_F16_BF8,
  HW_CVT_SR_FP8_F16,
  HW_CVT_SR_BF8_F16,

  // gfx1250 pk8 scalef32: wide vector -> fp8x8
  HW_CVT_SCALEF32_PK8_FP8_F16,
  HW_CVT_SCALEF32_PK8_BF8_F16,
  HW_CVT_SCALEF32_PK8_FP8_F32,
  HW_CVT_SCALEF32_PK8_BF8_F32,
  HW_CVT_SCALEF32_PK8_FP8_BF16,
  HW_CVT_SCALEF32_PK8_BF8_BF16,

  // gfx1250 pk8 scalef32 + stochastic rounding
  HW_CVT_SCALEF32_SR_PK8_FP8_F16,
  HW_CVT_SCALEF32_SR_PK8_BF8_F16,
  HW_CVT_SCALEF32_SR_PK8_FP8_F32,
  HW_CVT_SCALEF32_SR_PK8_BF8_F32,
  HW_CVT_SCALEF32_SR_PK8_FP8_BF16,
  HW_CVT_SCALEF32_SR_PK8_BF8_BF16,

  // gfx1250 pk8 scale upcast: fp8x8 -> wide (uint32 block scale)
  HW_CVT_SCALE_PK8_F16_FP8,
  HW_CVT_SCALE_PK8_F16_BF8,
  HW_CVT_SCALE_PK8_F32_FP8,
  HW_CVT_SCALE_PK8_F32_BF8,
  HW_CVT_SCALE_PK8_BF16_FP8,
  HW_CVT_SCALE_PK8_BF16_BF8,

  FP8_HW_NUM_INTRINSICS
};

struct Fp8HwTestResult {
  int ran;
  int ok;
};

struct Fp8ProbeResult {
  Fp8HwTestResult tests[FP8_HW_NUM_INTRINSICS];
};

__device__ inline void fp8SetTest(Fp8HwTestResult* r, int ok) {
  r->ran = 1;
  r->ok = ok ? 1 : 0;
}

__device__ inline void fp8SkipTest(Fp8HwTestResult* r) {
  r->ran = 0;
  r->ok = 0;
}

__device__ inline int fp8Near(float a, float b, float tol) {
  return (a > b - tol && a < b + tol) ? 1 : 0;
}

__device__ inline uint32_t fp8E4m3Bits(float f) {
  __hip_fp8_e4m3 v(f);
  return (uint32_t)v.__x;
}

__device__ inline uint32_t fp8E5m2Bits(float f) {
  __hip_fp8_e5m2 v(f);
  return (uint32_t)v.__x;
}

__device__ inline uint16_t fp8E4m3Pair(float a, float b) {
  return (uint16_t)(fp8E4m3Bits(a) | (fp8E4m3Bits(b) << 8));
}

__device__ inline uint16_t fp8E5m2Pair(float a, float b) {
  return (uint16_t)(fp8E5m2Bits(a) | (fp8E5m2Bits(b) << 8));
}

// Pack one fp8 byte into all 8 lanes of an fp8x8 register pair.
__device__ inline uint2_t fp8Pack8(uint8_t byte) {
  uint32_t word = (uint32_t)byte | ((uint32_t)byte << 8) | ((uint32_t)byte << 16) |
                  ((uint32_t)byte << 24);
  uint2_t u;
  u[0] = word;
  u[1] = word;
  return u;
}

__device__ inline uint2_t fp8Pack8E4m3(float f) {
  return fp8Pack8((uint8_t)fp8E4m3Bits(f));
}

__device__ inline uint2_t fp8Pack8E5m2(float f) {
  return fp8Pack8((uint8_t)fp8E5m2Bits(f));
}

__device__ inline half8_t fp8Half8Ones() {
  return half8_t{(_Float16)1.0f, (_Float16)1.0f, (_Float16)1.0f, (_Float16)1.0f,
                 (_Float16)1.0f, (_Float16)1.0f, (_Float16)1.0f, (_Float16)1.0f};
}

__device__ inline float8_t fp8Float8Ones() {
  return float8_t{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
}

__device__ inline uint16_t fp8Shortx2ToU16(shortx2_t v) {
  union {
    shortx2_t s;
    uint16_t u16;
  } u{v};
  return u.u16;
}

__device__ inline bhalf8_t fp8Bhalf8Ones() {
  return bhalf8_t{(__bf16)1.0f, (__bf16)1.0f, (__bf16)1.0f, (__bf16)1.0f,
                  (__bf16)1.0f, (__bf16)1.0f, (__bf16)1.0f, (__bf16)1.0f};
}

#if HIP_VERSION >= 60300000
__device__ void fp8TestHipE4m3(Fp8HwTestResult* r) {
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
  __hip_fp8_e4m3 a(1.0f);
  __hip_fp8_e4m3 b(2.0f);
  float fa = static_cast<float>(a);
  float fb = static_cast<float>(b);
  fp8SetTest(r, fp8Near(fa, 1.0f, 0.5f) && fp8Near(fb, 2.0f, 0.5f));
#else
  fp8SkipTest(r);
#endif
}
#else
__device__ void fp8TestHipE4m3(Fp8HwTestResult* r) { fp8SkipTest(r); }
#endif

#if HIP_VERSION >= 60300000
__device__ void fp8TestHipE5m2(Fp8HwTestResult* r) {
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
  __hip_fp8_e5m2 a(1.0f);
  float fa = static_cast<float>(a);
  fp8SetTest(r, fp8Near(fa, 1.0f, 0.5f));
#else
  fp8SkipTest(r);
#endif
}
#else
__device__ void fp8TestHipE5m2(Fp8HwTestResult* r) { fp8SkipTest(r); }
#endif

#if HIP_VERSION >= 60300000
__device__ void fp8TestHipE4m3Fnuz(Fp8HwTestResult* r) {
#if defined(__gfx942__)
  __hip_fp8_e4m3_fnuz a(1.0f);
  float fa = static_cast<float>(a);
  fp8SetTest(r, fp8Near(fa, 1.0f, 0.5f));
#else
  fp8SkipTest(r);
#endif
}
#else
__device__ void fp8TestHipE4m3Fnuz(Fp8HwTestResult* r) { fp8SkipTest(r); }
#endif

#if HIP_VERSION >= 60300000
__device__ void fp8TestHipE5m2Fnuz(Fp8HwTestResult* r) {
#if defined(__gfx942__)
  __hip_fp8_e5m2_fnuz a(1.0f);
  float fa = static_cast<float>(a);
  fp8SetTest(r, fp8Near(fa, 1.0f, 0.5f));
#else
  fp8SkipTest(r);
#endif
}
#else
__device__ void fp8TestHipE5m2Fnuz(Fp8HwTestResult* r) { fp8SkipTest(r); }
#endif

__device__ void fp8TestCvtPkF32Fp8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_fp8)) {
    fp8SkipTest(r);
    return;
  }
  float2_t v = __builtin_amdgcn_cvt_pk_f32_fp8(fp8E4m3Bits(1.0f), 0);
  fp8SetTest(r, fp8Near(v[0], 1.0f, 0.5f));
}

__device__ void fp8TestCvtPkF32Bf8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_bf8)) {
    fp8SkipTest(r);
    return;
  }
  float2_t v = __builtin_amdgcn_cvt_pk_f32_bf8(fp8E5m2Bits(1.0f), 0);
  fp8SetTest(r, fp8Near(v[0], 1.0f, 0.5f));
}

__device__ void fp8TestVCvtF32Fp8(Fp8HwTestResult* r) {
#if defined(__gfx942__) || defined(__gfx950__)
  float fval = 0.f;
  uint32_t i32val = fp8E4m3Bits(1.0f);
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
  fp8SetTest(r, fp8Near(fval, 1.0f, 0.5f));
#else
  fp8SkipTest(r);
#endif
}

__device__ void fp8TestVCvtF32Bf8(Fp8HwTestResult* r) {
#if defined(__gfx942__) || defined(__gfx950__)
  float fval = 0.f;
  uint32_t i32val = fp8E5m2Bits(1.0f);
  asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
  fp8SetTest(r, fp8Near(fval, 1.0f, 0.5f));
#else
  fp8SkipTest(r);
#endif
}

__device__ void fp8TestCvtPkFp8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint32_t ival = 0;
  ival = __builtin_amdgcn_cvt_pk_fp8_f32(1.0f, 1.0f, ival, false);
  fp8SetTest(r, ((uint8_t)(ival & 0xffu)) != 0);
}

__device__ void fp8TestCvtPkBf8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_bf8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint32_t ival = 0;
  ival = __builtin_amdgcn_cvt_pk_bf8_f32(1.0f, 1.0f, ival, false);
  fp8SetTest(r, ((uint8_t)(ival & 0xffu)) != 0);
}

__device__ void fp8TestCvtSrFp8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_sr_fp8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint32_t ival = 0;
  ival = __builtin_amdgcn_cvt_sr_fp8_f32(1.0f, 0u, ival, 0);
  fp8SetTest(r, ((uint8_t)(ival & 0xffu)) != 0);
}

__device__ void fp8TestCvtSrBf8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_sr_bf8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint32_t ival = 0;
  ival = __builtin_amdgcn_cvt_sr_bf8_f32(1.0f, 0u, ival, 0);
  fp8SetTest(r, ((uint8_t)(ival & 0xffu)) != 0);
}

__device__ void fp8TestScalef32PkF16Fp8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_fp8)) {
    fp8SkipTest(r);
    return;
  }
  half2_t v = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(fp8E4m3Bits(1.0f), 1.f, 0);
  fp8SetTest(r, fp8Near((float)v[0], 1.0f, 0.5f));
}

__device__ void fp8TestScalef32PkF16Bf8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_bf8)) {
    fp8SkipTest(r);
    return;
  }
  half2_t v = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(fp8E5m2Bits(1.0f), 1.f, 0);
  fp8SetTest(r, fp8Near((float)v[0], 1.0f, 0.5f));
}

__device__ void fp8TestScalef32PkFp8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp8_f16)) {
    fp8SkipTest(r);
    return;
  }
  half2_t h = {(_Float16)1.0f, (_Float16)1.0f};
  shortx2_t out = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(h, h, 1.f, 0);
  fp8SetTest(r, fp8Shortx2ToU16(out) != 0);
}

__device__ void fp8TestScalef32PkBf8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_bf8_f16)) {
    fp8SkipTest(r);
    return;
  }
  half2_t h = {(_Float16)1.0f, (_Float16)1.0f};
  shortx2_t out = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(h, h, 1.f, 0);
  fp8SetTest(r, fp8Shortx2ToU16(out) != 0);
}

__device__ void fp8TestCvtPkF16Fp8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f16_fp8)) {
    fp8SkipTest(r);
    return;
  }
  half2_t v = __builtin_amdgcn_cvt_pk_f16_fp8((short)fp8E4m3Pair(1.0f, 2.0f));
  fp8SetTest(r, fp8Near((float)v[0], 1.0f, 0.5f) && fp8Near((float)v[1], 2.0f, 0.5f));
}

__device__ void fp8TestCvtPkF16Bf8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f16_bf8)) {
    fp8SkipTest(r);
    return;
  }
  half2_t v = __builtin_amdgcn_cvt_pk_f16_bf8((short)fp8E5m2Pair(1.0f, 2.0f));
  fp8SetTest(r, fp8Near((float)v[0], 1.0f, 0.5f) && fp8Near((float)v[1], 2.0f, 0.5f));
}

__device__ void fp8TestCvtPkFp8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f16)) {
    fp8SkipTest(r);
    return;
  }
  half2_t h = {(_Float16)1.0f, (_Float16)2.0f};
  uint16_t out = (uint16_t)__builtin_amdgcn_cvt_pk_fp8_f16(h);
  half2_t check = __builtin_amdgcn_cvt_pk_f16_fp8((short)out);
  fp8SetTest(r, fp8Near((float)check[0], 1.0f, 0.5f) && fp8Near((float)check[1], 2.0f, 0.5f));
}

__device__ void fp8TestCvtPkBf8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_bf8_f16)) {
    fp8SkipTest(r);
    return;
  }
  half2_t h = {(_Float16)1.0f, (_Float16)2.0f};
  uint16_t out = (uint16_t)__builtin_amdgcn_cvt_pk_bf8_f16(h);
  half2_t check = __builtin_amdgcn_cvt_pk_f16_bf8((short)out);
  fp8SetTest(r, fp8Near((float)check[0], 1.0f, 0.5f) && fp8Near((float)check[1], 2.0f, 0.5f));
}

__device__ void fp8TestCvtF16Fp8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_f16_fp8)) {
    fp8SkipTest(r);
    return;
  }
  _Float16 h = __builtin_amdgcn_cvt_f16_fp8(fp8E4m3Bits(1.0f), 0);
  fp8SetTest(r, fp8Near((float)h, 1.0f, 0.5f));
}

__device__ void fp8TestCvtF16Bf8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_f16_bf8)) {
    fp8SkipTest(r);
    return;
  }
  _Float16 h = __builtin_amdgcn_cvt_f16_bf8(fp8E5m2Bits(1.0f), 0);
  fp8SetTest(r, fp8Near((float)h, 1.0f, 0.5f));
}

__device__ void fp8TestCvtSrFp8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_sr_fp8_f16)) {
    fp8SkipTest(r);
    return;
  }
  uint32_t ival = 0;
  ival = __builtin_amdgcn_cvt_sr_fp8_f16((_Float16)1.0f, 0u, ival, 0);
  fp8SetTest(r, ((uint8_t)(ival & 0xffu)) != 0);
}

__device__ void fp8TestCvtSrBf8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_sr_bf8_f16)) {
    fp8SkipTest(r);
    return;
  }
  uint32_t ival = 0;
  ival = __builtin_amdgcn_cvt_sr_bf8_f16((_Float16)1.0f, 0u, ival, 0);
  fp8SetTest(r, ((uint8_t)(ival & 0xffu)) != 0);
}

__device__ void fp8TestScalef32Pk8Fp8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp8_f16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f16(fp8Half8Ones(), 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32Pk8Bf8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_bf8_f16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_pk8_bf8_f16(fp8Half8Ones(), 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32Pk8Fp8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(fp8Float8Ones(), 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32Pk8Bf8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_bf8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32(fp8Float8Ones(), 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32Pk8Fp8Bf16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(fp8Bhalf8Ones(), 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32Pk8Bf8Bf16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_bf8_bf16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_pk8_bf8_bf16(fp8Bhalf8Ones(), 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32SrPk8Fp8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f16(fp8Half8Ones(), 0u, 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32SrPk8Bf8F16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f16(fp8Half8Ones(), 0u, 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32SrPk8Fp8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32(fp8Float8Ones(), 0u, 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32SrPk8Bf8F32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32(fp8Float8Ones(), 0u, 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32SrPk8Fp8Bf16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_bf16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_bf16(fp8Bhalf8Ones(), 0u, 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalef32SrPk8Bf8Bf16(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_bf16)) {
    fp8SkipTest(r);
    return;
  }
  uint2_t out = __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_bf16(fp8Bhalf8Ones(), 0u, 1.0f);
  fp8SetTest(r, out[0] != 0 || out[1] != 0);
}

__device__ void fp8TestScalePk8F16Fp8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f16_fp8)) {
    fp8SkipTest(r);
    return;
  }
  half8_t out = __builtin_amdgcn_cvt_scale_pk8_f16_fp8(fp8Pack8E4m3(1.0f), 127u, 0u);
  fp8SetTest(r, fp8Near((float)out[0], 1.0f, 0.5f));
}

__device__ void fp8TestScalePk8F16Bf8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f16_bf8)) {
    fp8SkipTest(r);
    return;
  }
  half8_t out = __builtin_amdgcn_cvt_scale_pk8_f16_bf8(fp8Pack8E5m2(1.0f), 127u, 0u);
  fp8SetTest(r, fp8Near((float)out[0], 1.0f, 0.5f));
}

__device__ void fp8TestScalePk8F32Fp8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp8)) {
    fp8SkipTest(r);
    return;
  }
  float8_t out = __builtin_amdgcn_cvt_scale_pk8_f32_fp8(fp8Pack8E4m3(1.0f), 127u, 0u);
  fp8SetTest(r, fp8Near(out[0], 1.0f, 0.5f));
}

__device__ void fp8TestScalePk8F32Bf8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_bf8)) {
    fp8SkipTest(r);
    return;
  }
  float8_t out = __builtin_amdgcn_cvt_scale_pk8_f32_bf8(fp8Pack8E5m2(1.0f), 127u, 0u);
  fp8SetTest(r, fp8Near(out[0], 1.0f, 0.5f));
}

__device__ void fp8TestScalePk8Bf16Fp8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_bf16_fp8)) {
    fp8SkipTest(r);
    return;
  }
  bhalf8_t out = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(fp8Pack8E4m3(1.0f), 127u, 0u);
  fp8SetTest(r, fp8Near((float)out[0], 1.0f, 0.5f));
}

__device__ void fp8TestScalePk8Bf16Bf8(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_bf16_bf8)) {
    fp8SkipTest(r);
    return;
  }
  bhalf8_t out = __builtin_amdgcn_cvt_scale_pk8_bf16_bf8(fp8Pack8E5m2(1.0f), 127u, 0u);
  fp8SetTest(r, fp8Near((float)out[0], 1.0f, 0.5f));
}

__device__ void fp8TestVPkAddF32(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_fp8)) {
    fp8SkipTest(r);
    return;
  }
  float2_t a = {1.0f, 2.0f};
  float2_t b = {3.0f, 4.0f};
  float2_t c;
  asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
  fp8SetTest(r, fp8Near(c[0], 4.0f, 0.01f) && fp8Near(c[1], 6.0f, 0.01f));
}

__device__ void fp8TestVPkAddF16(Fp8HwTestResult* r) {
  half2_t a = {(_Float16)1.0f, (_Float16)2.0f};
  half2_t b = {(_Float16)3.0f, (_Float16)4.0f};
  half2_t c;
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
  fp8SetTest(r, fp8Near((float)c[0], 4.0f, 0.01f) && fp8Near((float)c[1], 6.0f, 0.01f));
}

__device__ void fp8TestFmed3f(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f)) {
    fp8SkipTest(r);
    return;
  }
  float out = __builtin_amdgcn_fmed3f(300.f, 240.f, -240.f);
  fp8SetTest(r, fp8Near(out, 240.f, 0.01f));
}

__device__ void fp8TestE4m3AddPk1(Fp8HwTestResult* r) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_fp8) &&
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f32)) {
    float2_t va = __builtin_amdgcn_cvt_pk_f32_fp8(fp8E4m3Bits(1.0f), 0);
    float2_t vb = __builtin_amdgcn_cvt_pk_f32_fp8(fp8E4m3Bits(2.0f), 0);
    float2_t vc;
    asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(vc) : "v"(va), "v"(vb));
    uint32_t ival = 0;
    uint8_t out = (uint8_t)__builtin_amdgcn_cvt_pk_fp8_f32(vc[0], vc[0], ival, false);
    float2_t check = __builtin_amdgcn_cvt_pk_f32_fp8((uint32_t)out, 0);
    fp8SetTest(r, fp8Near(check[0], 3.0f, 0.75f));
  } else if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f16_fp8) &&
             __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f16)) {
    half2_t va = __builtin_amdgcn_cvt_pk_f16_fp8((short)fp8E4m3Pair(1.0f, 1.0f));
    half2_t vb = __builtin_amdgcn_cvt_pk_f16_fp8((short)fp8E4m3Pair(2.0f, 2.0f));
    half2_t vc;
    asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(vc) : "v"(va), "v"(vb));
    uint16_t out = (uint16_t)__builtin_amdgcn_cvt_pk_fp8_f16(vc);
    half2_t check = __builtin_amdgcn_cvt_pk_f16_fp8((short)out);
    fp8SetTest(r, fp8Near((float)check[0], 3.0f, 0.75f));
  } else if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_fp8) &&
             __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp8_f16)) {
    half2_t va = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(fp8E4m3Bits(1.0f), 1.f, 0);
    half2_t vb = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(fp8E4m3Bits(2.0f), 1.f, 0);
    half2_t vc;
    asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(vc) : "v"(va), "v"(vb));
    shortx2_t out = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(vc, vc, 1.f, 0);
    half2_t check = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(fp8Shortx2ToU16(out), 1.f, 0);
    fp8SetTest(r, fp8Near((float)check[0], 3.0f, 0.75f));
  } else {
    fp8SkipTest(r);
  }
}

__device__ void fp8TestE4m3AddPk2(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_fp8) ||
      !__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint16_t pair = fp8E4m3Pair(1.0f, 2.0f);
  float2_t va = __builtin_amdgcn_cvt_pk_f32_fp8(pair, 0);
  float2_t vb = __builtin_amdgcn_cvt_pk_f32_fp8(pair, 0);
  float2_t vc;
  asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(vc) : "v"(va), "v"(vb));
  uint32_t ival = 0;
  uint16_t out = (uint16_t)__builtin_amdgcn_cvt_pk_fp8_f32(vc[0], vc[1], ival, false);
  float2_t check = __builtin_amdgcn_cvt_pk_f32_fp8((uint32_t)out, 0);
  fp8SetTest(r, fp8Near(check[0], 2.0f, 0.75f) && fp8Near(check[1], 4.0f, 0.75f));
}

__device__ void fp8TestE5m2AddPk1(Fp8HwTestResult* r) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_bf8) &&
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_bf8_f32)) {
    float2_t va = __builtin_amdgcn_cvt_pk_f32_bf8(fp8E5m2Bits(1.0f), 0);
    float2_t vb = __builtin_amdgcn_cvt_pk_f32_bf8(fp8E5m2Bits(2.0f), 0);
    float2_t vc;
    asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(vc) : "v"(va), "v"(vb));
    uint32_t ival = 0;
    uint8_t out = (uint8_t)__builtin_amdgcn_cvt_pk_bf8_f32(vc[0], vc[0], ival, false);
    float2_t check = __builtin_amdgcn_cvt_pk_f32_bf8((uint32_t)out, 0);
    fp8SetTest(r, fp8Near(check[0], 3.0f, 0.75f));
  } else if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_bf8) &&
             __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_bf8_f16)) {
    half2_t va = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(fp8E5m2Bits(1.0f), 1.f, 0);
    half2_t vb = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(fp8E5m2Bits(2.0f), 1.f, 0);
    half2_t vc;
    asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(vc) : "v"(va), "v"(vb));
    shortx2_t out = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(vc, vc, 1.f, 0);
    half2_t check = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(fp8Shortx2ToU16(out), 1.f, 0);
    fp8SetTest(r, fp8Near((float)check[0], 3.0f, 0.75f));
  } else {
    fp8SkipTest(r);
  }
}

__device__ void fp8TestE5m2AddPk2(Fp8HwTestResult* r) {
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_bf8) ||
      !__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_bf8_f32)) {
    fp8SkipTest(r);
    return;
  }
  uint16_t pair = fp8E5m2Pair(1.0f, 2.0f);
  float2_t va = __builtin_amdgcn_cvt_pk_f32_bf8(pair, 0);
  float2_t vb = __builtin_amdgcn_cvt_pk_f32_bf8(pair, 0);
  float2_t vc;
  asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(vc) : "v"(va), "v"(vb));
  uint32_t ival = 0;
  uint16_t out = (uint16_t)__builtin_amdgcn_cvt_pk_bf8_f32(vc[0], vc[1], ival, false);
  float2_t check = __builtin_amdgcn_cvt_pk_f32_bf8((uint32_t)out, 0);
  fp8SetTest(r, fp8Near(check[0], 2.0f, 0.75f) && fp8Near(check[1], 4.0f, 0.75f));
}

__global__ void fp8ProbeKernel(Fp8ProbeResult* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  Fp8ProbeResult r{};
  fp8TestHipE4m3(&r.tests[HW_HIP_FP8_E4M3]);
  fp8TestHipE5m2(&r.tests[HW_HIP_FP8_E5M2]);
  fp8TestHipE4m3Fnuz(&r.tests[HW_HIP_FP8_E4M3_FNUZ]);
  fp8TestHipE5m2Fnuz(&r.tests[HW_HIP_FP8_E5M2_FNUZ]);

  fp8TestCvtPkF32Fp8(&r.tests[HW_CVT_PK_F32_FP8]);
  fp8TestCvtPkF32Bf8(&r.tests[HW_CVT_PK_F32_BF8]);
  fp8TestVCvtF32Fp8(&r.tests[HW_CVT_F32_FP8]);
  fp8TestVCvtF32Bf8(&r.tests[HW_CVT_F32_BF8]);

  fp8TestCvtPkFp8F32(&r.tests[HW_CVT_PK_FP8_F32]);
  fp8TestCvtPkBf8F32(&r.tests[HW_CVT_PK_BF8_F32]);
  fp8TestCvtSrFp8F32(&r.tests[HW_CVT_SR_FP8_F32]);
  fp8TestCvtSrBf8F32(&r.tests[HW_CVT_SR_BF8_F32]);

  fp8TestScalef32PkF16Fp8(&r.tests[HW_CVT_SCALEF32_PK_F16_FP8]);
  fp8TestScalef32PkF16Bf8(&r.tests[HW_CVT_SCALEF32_PK_F16_BF8]);
  fp8TestScalef32PkFp8F16(&r.tests[HW_CVT_SCALEF32_PK_FP8_F16]);
  fp8TestScalef32PkBf8F16(&r.tests[HW_CVT_SCALEF32_PK_BF8_F16]);

  fp8TestVPkAddF32(&r.tests[HW_V_PK_ADD_F32]);
  fp8TestVPkAddF16(&r.tests[HW_V_PK_ADD_F16]);
  fp8TestFmed3f(&r.tests[HW_FMED3F]);

  fp8TestE4m3AddPk1(&r.tests[HW_FP8_E4M3_ADD_PK1]);
  fp8TestE4m3AddPk2(&r.tests[HW_FP8_E4M3_ADD_PK2]);
  fp8TestE5m2AddPk1(&r.tests[HW_FP8_E5M2_ADD_PK1]);
  fp8TestE5m2AddPk2(&r.tests[HW_FP8_E5M2_ADD_PK2]);

  fp8TestCvtPkF16Fp8(&r.tests[HW_CVT_PK_F16_FP8]);
  fp8TestCvtPkF16Bf8(&r.tests[HW_CVT_PK_F16_BF8]);
  fp8TestCvtPkFp8F16(&r.tests[HW_CVT_PK_FP8_F16]);
  fp8TestCvtPkBf8F16(&r.tests[HW_CVT_PK_BF8_F16]);
  fp8TestCvtF16Fp8(&r.tests[HW_CVT_F16_FP8]);
  fp8TestCvtF16Bf8(&r.tests[HW_CVT_F16_BF8]);
  fp8TestCvtSrFp8F16(&r.tests[HW_CVT_SR_FP8_F16]);
  fp8TestCvtSrBf8F16(&r.tests[HW_CVT_SR_BF8_F16]);

  fp8TestScalef32Pk8Fp8F16(&r.tests[HW_CVT_SCALEF32_PK8_FP8_F16]);
  fp8TestScalef32Pk8Bf8F16(&r.tests[HW_CVT_SCALEF32_PK8_BF8_F16]);
  fp8TestScalef32Pk8Fp8F32(&r.tests[HW_CVT_SCALEF32_PK8_FP8_F32]);
  fp8TestScalef32Pk8Bf8F32(&r.tests[HW_CVT_SCALEF32_PK8_BF8_F32]);
  fp8TestScalef32Pk8Fp8Bf16(&r.tests[HW_CVT_SCALEF32_PK8_FP8_BF16]);
  fp8TestScalef32Pk8Bf8Bf16(&r.tests[HW_CVT_SCALEF32_PK8_BF8_BF16]);

  fp8TestScalef32SrPk8Fp8F16(&r.tests[HW_CVT_SCALEF32_SR_PK8_FP8_F16]);
  fp8TestScalef32SrPk8Bf8F16(&r.tests[HW_CVT_SCALEF32_SR_PK8_BF8_F16]);
  fp8TestScalef32SrPk8Fp8F32(&r.tests[HW_CVT_SCALEF32_SR_PK8_FP8_F32]);
  fp8TestScalef32SrPk8Bf8F32(&r.tests[HW_CVT_SCALEF32_SR_PK8_BF8_F32]);
  fp8TestScalef32SrPk8Fp8Bf16(&r.tests[HW_CVT_SCALEF32_SR_PK8_FP8_BF16]);
  fp8TestScalef32SrPk8Bf8Bf16(&r.tests[HW_CVT_SCALEF32_SR_PK8_BF8_BF16]);

  fp8TestScalePk8F16Fp8(&r.tests[HW_CVT_SCALE_PK8_F16_FP8]);
  fp8TestScalePk8F16Bf8(&r.tests[HW_CVT_SCALE_PK8_F16_BF8]);
  fp8TestScalePk8F32Fp8(&r.tests[HW_CVT_SCALE_PK8_F32_FP8]);
  fp8TestScalePk8F32Bf8(&r.tests[HW_CVT_SCALE_PK8_F32_BF8]);
  fp8TestScalePk8Bf16Fp8(&r.tests[HW_CVT_SCALE_PK8_BF16_FP8]);
  fp8TestScalePk8Bf16Bf8(&r.tests[HW_CVT_SCALE_PK8_BF16_BF8]);

  *out = r;
}

#endif  // FP8_PROBE_H_
