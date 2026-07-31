/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCBLAS_FLOAT8_H
#define ROCBLAS_FLOAT8_H

#include <stdint.h>
#include <hip/hip_version.h>

typedef uint16_t fp8x2_storage_t;

// Inf handling on the narrow-float downconvert.
//
// The reduce operations produce an Inf result only when one of their operands is
// Inf; an out-of-range result computed from finite operands saturates to
// +/-max_finite instead. That is the default (0) here, and it matches what the
// hip_fp8 constructors do in __HIP_SATFINITE mode, where the clamp is skipped
// for inputs whose exponent field is all ones.
//
// Setting this to 1 also folds +/-Inf onto +/-max_finite, the way AMD's
// MODE.FP16_OVFL=1 and NVIDIA's .satfinite conversions do. It is kept
// switchable so the two conventions can be compared on hardware; note that it
// makes an Inf operand indistinguishable from an overflow. Finite operands whose
// result is in range are unaffected either way.
//
// Whichever setting is in force applies to both the 1-wide and 2-wide reduce
// paths, so a reduction result never depends on how the data happened to be
// aligned.
#ifndef RCCL_NARROW_FP_SATURATE_INF
#define RCCL_NARROW_FP_SATURATE_INF 0
#endif

// Largest finite magnitude of the fp8/bf8 encodings in force for this
// translation unit. This mirrors the branch below that picks the types: gfx942
// and the software-emulated fallback use the fnuz variants, where e4m3 stops at
// 240, while the OCP variants used by the host pass and by gfx950 and gfx12xx
// reach 448. e5m2 is 57344 in every case.
#if HIP_VERSION >= 60300000 &&                                                                     \
    (!__HIP_DEVICE_COMPILE__ || defined(__gfx950__) || defined(__gfx1200__) ||                      \
     defined(__gfx1201__) || defined(__gfx1250__) || defined(__gfx1251__))
#define RCCL_FP8_MAX_FINITE 448.0f
#else
#define RCCL_FP8_MAX_FINITE 240.0f
#endif
#define RCCL_BF8_MAX_FINITE 57344.0f

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// Clamp into the destination's range, leaving Inf and NaN alone. Used by the
// 1-wide reduce specializations and, in its two-lane form further down, by the
// packed helpers. Every narrow-float result the reduce ops produce goes through
// here, so it is the single place where the range of the destination encoding is
// enforced.
//
// fminf/fmaxf return the non-NaN operand, so a NaN would come back clamped
// rather than propagated; Inf would be folded onto maxFinite. Both are excluded
// explicitly by the finite test. gfx950 lowers the whole thing to three VALU ops
// per lane: v_med3_f32 for the clamp, then v_cmp_nlg_f32 against Inf with the abs
// modifier and a v_cndmask_b32 to put the excluded values back.
//
// The RCCL_NARROW_FP_SATURATE_INF case needs no such guard: elementwise_minimum
// and elementwise_maximum are the IEEE-754-2019 minimum/maximum operations,
// which propagate NaN, so NaN survives the clamp on its own while Inf saturates.
// gfx950 lowers that to v_maximum3_f32 plus v_minimum3_f32 and gfx1250 to
// v_maximum_f32 plus v_minimum_f32, two VALU ops per lane; gfx942 has no
// NaN-propagating min/max and lowers it to a compare and select.
inline __host__ __device__ float rccl_narrow_sat_f32(float v, float maxFinite) {
#if RCCL_NARROW_FP_SATURATE_INF
  return __builtin_elementwise_minimum(__builtin_elementwise_maximum(v, -maxFinite), maxFinite);
#else
  bool finite = (v == v) && (v < __builtin_huge_valf()) && (v > -__builtin_huge_valf());
  return finite ? __builtin_fminf(__builtin_fmaxf(v, -maxFinite), maxFinite) : v;
#endif
}
#endif

#if __cplusplus < 201103L || (!defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__))
/*! \brief Struct to represent a 8 bit floating-point number. */

typedef struct {
  uint8_t data;
} rccl_float8;

typedef struct {
  uint8_t data;
} rccl_bfloat8;

// __cplusplus < 201103L || (!defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__))
#elif HIP_VERSION >= 60300000 && (!__HIP_DEVICE_COMPILE__ || defined(__gfx942__) || defined(__gfx950__) || \
                                  defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))

#include <hip/hip_fp8.h>

#if __HIP_DEVICE_COMPILE__ && (defined(__gfx942__))
typedef __hip_fp8_e4m3_fnuz rccl_float8;
typedef __hip_fp8_e5m2_fnuz rccl_bfloat8;
#else
typedef __hip_fp8_e4m3 rccl_float8;
typedef __hip_fp8_e5m2 rccl_bfloat8;
#endif

typedef _Float16 half_t;
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

typedef short shortx2_t __attribute__((ext_vector_type(2)));
typedef short __attribute__((ext_vector_type(2))) __amd_shortx2_storage_t;
typedef float float2_t __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------------------
// fp8/bf8 reduce helpers, 1-wide and 2-wide: Sum, Prod, MinMax and PreMulSum's
// premul.
//
// Every 2-wide helper is bit-identical to the per-element expression its 1-wide
// counterpart uses, so a reduction result never depends on how the data happened
// to be aligned. Two properties are needed for that, and neither comes for free:
//
//  1. An intermediate that rounds to the same result the exact one would. f32
//     always does. f16 does for Sum: e4m3 sums stay inside it outright, reaching
//     at most 448+448 = 896, and e5m2 sums round the same way there even though
//     114688 overflows it, for the reason rccl_add_pack2_bf8_f16 gives. Prod
//     needs f32 -- e4m3 alone reaches 200704 -- as does PreMulSum's premul, whose
//     scalar is arbitrary. So Sum takes the cheap f16 converts where the target
//     has them, and everything else goes through f32.
//  2. Clamping before the pack. The packs round rather than saturate, so they
//     send out-of-range values to Inf (bf8) or NaN (e4m3, which has no Inf) and
//     the clamp has to run first. MinMax is the exception: its result is always
//     one of its inputs and so is already representable, and it uses the
//     unclamped pack.
//
// gfx1250 (MI4xx) can instead saturate in hardware via MODE.FP16_OVFL (bit 23).
// That is not used here because it is wave-level state that also redirects
// f32->f16/bf16 converts and v_pk_*_bf16 overflow from Inf to saturation, which
// reaches well beyond fp8. Note also that the 8-wide v_cvt_scalef32_pk8_*
// converts ignore FP16_OVFL entirely and always need an explicit clamp.
// ---------------------------------------------------------------------------

#if __HIP_DEVICE_COMPILE__ &&                                                                     \
    (defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1200__) ||                         \
     defined(__gfx1201__) || defined(__gfx1250__) || defined(__gfx1251__))
#define RCCL_FP8_PACKED_CVT 1
#endif

#ifdef RCCL_FP8_PACKED_CVT
inline __device__ float2_t rccl_narrow_sat2_f32(float2_t v, float maxFinite) {
#if RCCL_NARROW_FP_SATURATE_INF
  float2_t hi = {maxFinite, maxFinite};
  float2_t lo = {-maxFinite, -maxFinite};
  return __builtin_elementwise_minimum(__builtin_elementwise_maximum(v, lo), hi);
#else
  v[0] = rccl_narrow_sat_f32(v[0], maxFinite);
  v[1] = rccl_narrow_sat_f32(v[1], maxFinite);
  return v;
#endif
}

inline __device__ float2_t rccl_unpack2_f32_fp8(fp8x2_storage_t x) {
  return __builtin_amdgcn_cvt_pk_f32_fp8((int)x, false);
}

inline __device__ float2_t rccl_unpack2_f32_bf8(fp8x2_storage_t x) {
  return __builtin_amdgcn_cvt_pk_f32_bf8((int)x, false);
}

inline __device__ fp8x2_storage_t rccl_pack2_fp8_f32(float2_t v) {
  return (fp8x2_storage_t)(__builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], 0, false) & 0xFFFFu);
}

inline __device__ fp8x2_storage_t rccl_pack2_bf8_f32(float2_t v) {
  return (fp8x2_storage_t)(__builtin_amdgcn_cvt_pk_bf8_f32(v[0], v[1], 0, false) & 0xFFFFu);
}

inline __device__ fp8x2_storage_t rccl_sat_pack2_fp8_f32(float2_t v) {
  return rccl_pack2_fp8_f32(rccl_narrow_sat2_f32(v, RCCL_FP8_MAX_FINITE));
}

inline __device__ fp8x2_storage_t rccl_sat_pack2_bf8_f32(float2_t v) {
  return rccl_pack2_bf8_f32(rccl_narrow_sat2_f32(v, RCCL_BF8_MAX_FINITE));
}
#endif

// The f16 route for Sum. Only gfx950 has the f16 forms of the fp8 converts.
#if __HIP_DEVICE_COMPILE__ && defined(__gfx950__)
#define RCCL_FP8_F16_CVT 1
#endif

#ifdef RCCL_FP8_F16_CVT
inline __device__ half2_t rccl_unpack2_f16_fp8(fp8x2_storage_t x) {
  return __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(x, 1.f, 0);
}

inline __device__ half2_t rccl_unpack2_f16_bf8(fp8x2_storage_t x) {
  return __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(x, 1.f, 0);
}

// Unlike rccl_narrow_sat_f32 this needs no finite test, because Inf cannot reach
// it: e4m3 has no Inf encoding, so both operands are finite, and their sum cannot
// leave f16's range. NaN can arrive, and the IEEE-754-2019 minimum/maximum
// operations propagate it rather than returning the other operand. gfx950 lowers
// the pair to v_pk_maximum3_f16 and v_pk_minimum3_f16, two VALU ops for both
// lanes together.
inline __device__ fp8x2_storage_t rccl_clamp_pack2_fp8_f16(half2_t v) {
  const half2_t hi = {(half_t)RCCL_FP8_MAX_FINITE, (half_t)RCCL_FP8_MAX_FINITE};
  const half2_t lo = {(half_t)-RCCL_FP8_MAX_FINITE, (half_t)-RCCL_FP8_MAX_FINITE};
  v = __builtin_elementwise_minimum(__builtin_elementwise_maximum(v, lo), hi);
  union {
    shortx2_t i16x2;
    fp8x2_storage_t fp8x2;
  } u{0};
  u.i16x2 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(v, v, /* scale */ 1.f, 0);
  return u.fp8x2;
}

// e5m2 Sum in f16. e5m2 shares f16's exponent range and has fewer mantissa bits,
// so the unpack is exact, but unlike e4m3 the sum can leave f16: 57344+57344 is
// 114688 and f16 stops at 65504. An operand that is Inf produces the same Inf the
// overflow does, and the two have to end up in different places -- max_finite for
// the overflow, Inf for the Inf -- so the sum alone cannot decide it.
//
// The operands can. No finite e5m2 value exceeds max_finite, so raising the clamp
// bound to max(|x|, |y|, max_finite) leaves it at max_finite for finite operands
// and lifts it to Inf exactly when one is Inf, which is when the clamp has to stop
// clamping. A NaN operand makes the bound NaN, and the clamp then propagates that
// NaN, which is the wanted answer anyway since the sum is NaN too.
//
// gfx950 spends three VALU ops on the bound: the two abs become one v_and_b32
// each, and the three-way max is a single v_pk_maximum3_f16 against an inline
// constant. The clamp is the same two ops e4m3 uses.
//
// Adding in f16 rather than f32 does not change any result: for the sum of two
// e5m2 values to round differently in the two widths, the second operand would
// have to sit within f16's 11 bits of a rounding boundary of the first and still
// be an e5m2 value, and 3 significant bits cannot do that. The exhaustive
// NarrowFloatGolden sweep confirms it over all 65536 pairs.
inline __device__ fp8x2_storage_t rccl_add_pack2_bf8_f16(fp8x2_storage_t x,
                                                         fp8x2_storage_t y) {
  const half2_t maxFinite = {(half_t)RCCL_BF8_MAX_FINITE, (half_t)RCCL_BF8_MAX_FINITE};
  half2_t a = rccl_unpack2_f16_bf8(x);
  half2_t b = rccl_unpack2_f16_bf8(y);
#if RCCL_NARROW_FP_SATURATE_INF
  half2_t bound = maxFinite;
#else
  half2_t bound = __builtin_elementwise_maximum(
      __builtin_elementwise_maximum(__builtin_elementwise_abs(a), __builtin_elementwise_abs(b)),
      maxFinite);
#endif
  half2_t v = a + b;
  v = __builtin_elementwise_minimum(__builtin_elementwise_maximum(v, -bound), bound);
  union {
    shortx2_t i16x2;
    fp8x2_storage_t bfp8x2;
  } u{0};
  u.i16x2 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(v, v, /* scale */ 1.f, 0);
  return u.bfp8x2;
}
#endif

// Add. The 1-wide form is the 2-wide pack with its high lane discarded: the byte
// goes in as the low half of a pair whose high half the unpack reads as +0.0.
// Sharing the pack is what makes the two widths agree by construction rather than
// by a matching pair of expressions that could drift apart.
inline __device__ rccl_float8 hadd(rccl_float8 x, rccl_float8 y) {
#ifdef RCCL_FP8_PACKED_CVT
  // The pack returns the fp8 bytes in an int. Handing that to rccl_float8 would
  // select its numeric int constructor, which converts the bit pattern as a
  // value, so reinterpret it as raw storage instead.
  union {
    fp8x2_storage_t fp8x2;
    rccl_float8 fp8[2];
  } u{0};
#ifdef RCCL_FP8_F16_CVT
  u.fp8x2 = rccl_clamp_pack2_fp8_f16(rccl_unpack2_f16_fp8(x.__x) + rccl_unpack2_f16_fp8(y.__x));
#else
  u.fp8x2 = rccl_sat_pack2_fp8_f32(rccl_unpack2_f32_fp8(x.__x) + rccl_unpack2_f32_fp8(y.__x));
#endif
  return u.fp8[0];
#else
  return rccl_float8(float(x) + float(y));
#endif
}

inline __device__ rccl_bfloat8 hadd_b(rccl_bfloat8 x, rccl_bfloat8 y) {
#ifdef RCCL_FP8_PACKED_CVT
  // See hadd: reinterpret the packed bytes rather than letting the int
  // constructor convert them numerically.
  union {
    fp8x2_storage_t bfp8x2;
    rccl_bfloat8 bfp8[2];
  } u{0};
#ifdef RCCL_FP8_F16_CVT
  u.bfp8x2 = rccl_add_pack2_bf8_f16(x.__x, y.__x);
#else
  u.bfp8x2 = rccl_sat_pack2_bf8_f32(rccl_unpack2_f32_bf8(x.__x) + rccl_unpack2_f32_bf8(y.__x));
#endif
  return u.bfp8[0];
#else
  return rccl_bfloat8(float(x) + float(y));
#endif
}

// Packed add. Both lanes go through one v_pk_add_f16 and one clamped packed
// convert, on the targets that have the f16 converts; elsewhere the same in f32.
inline __device__ fp8x2_storage_t hadd2(fp8x2_storage_t x, fp8x2_storage_t y) {
#ifdef RCCL_FP8_F16_CVT
  return rccl_clamp_pack2_fp8_f16(rccl_unpack2_f16_fp8(x) + rccl_unpack2_f16_fp8(y));
#elif defined(RCCL_FP8_PACKED_CVT)
  return rccl_sat_pack2_fp8_f32(rccl_unpack2_f32_fp8(x) + rccl_unpack2_f32_fp8(y));
#else
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, v, w;
  u.fp8x2 = x;
  v.fp8x2 = y;
  w.fp8[0] = hadd(u.fp8[0], v.fp8[0]);
  w.fp8[1] = hadd(u.fp8[1], v.fp8[1]);
  return w.fp8x2;
#endif
}

inline __device__ fp8x2_storage_t hadd2_b(fp8x2_storage_t x, fp8x2_storage_t y) {
#ifdef RCCL_FP8_F16_CVT
  return rccl_add_pack2_bf8_f16(x, y);
#elif defined(RCCL_FP8_PACKED_CVT)
  return rccl_sat_pack2_bf8_f32(rccl_unpack2_f32_bf8(x) + rccl_unpack2_f32_bf8(y));
#else
  union {
    rccl_bfloat8 bfp8[2];
    fp8x2_storage_t bfp8x2;
  } u, v, w;
  u.bfp8x2 = x;
  v.bfp8x2 = y;
  w.bfp8[0] = hadd_b(u.bfp8[0], v.bfp8[0]);
  w.bfp8[1] = hadd_b(u.bfp8[1], v.bfp8[1]);
  return w.bfp8x2;
#endif
}

// Packed multiply. Both lanes go through one v_pk_mul_f32 and one packed
// saturating convert.
inline __device__ fp8x2_storage_t hmul2(fp8x2_storage_t x, fp8x2_storage_t y) {
#ifdef RCCL_FP8_PACKED_CVT
  return rccl_sat_pack2_fp8_f32(rccl_unpack2_f32_fp8(x) * rccl_unpack2_f32_fp8(y));
#else
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, v, w;
  u.fp8x2 = x;
  v.fp8x2 = y;
  w.fp8[0] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[0]) * float(v.fp8[0]), RCCL_FP8_MAX_FINITE));
  w.fp8[1] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[1]) * float(v.fp8[1]), RCCL_FP8_MAX_FINITE));
  return w.fp8x2;
#endif
}

inline __device__ fp8x2_storage_t hmul2_b(fp8x2_storage_t x, fp8x2_storage_t y) {
#ifdef RCCL_FP8_PACKED_CVT
  return rccl_sat_pack2_bf8_f32(rccl_unpack2_f32_bf8(x) * rccl_unpack2_f32_bf8(y));
#else
  union {
    rccl_bfloat8 bfp8[2];
    fp8x2_storage_t bfp8x2;
  } u, v, w;
  u.bfp8x2 = x;
  v.bfp8x2 = y;
  w.bfp8[0] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[0]) * float(v.bfp8[0]), RCCL_BF8_MAX_FINITE));
  w.bfp8[1] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[1]) * float(v.bfp8[1]), RCCL_BF8_MAX_FINITE));
  return w.bfp8x2;
#endif
}

// Packed multiply by a broadcast scalar, for PreMulSum's premul step.
inline __device__ fp8x2_storage_t hpremul2(fp8x2_storage_t x, float s) {
#ifdef RCCL_FP8_PACKED_CVT
  float2_t vs = {s, s};
  return rccl_sat_pack2_fp8_f32(rccl_unpack2_f32_fp8(x) * vs);
#else
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, w;
  u.fp8x2 = x;
  w.fp8[0] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[0]) * s, RCCL_FP8_MAX_FINITE));
  w.fp8[1] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[1]) * s, RCCL_FP8_MAX_FINITE));
  return w.fp8x2;
#endif
}

inline __device__ fp8x2_storage_t hpremul2_b(fp8x2_storage_t x, float s) {
#ifdef RCCL_FP8_PACKED_CVT
  float2_t vs = {s, s};
  return rccl_sat_pack2_bf8_f32(rccl_unpack2_f32_bf8(x) * vs);
#else
  union {
    rccl_bfloat8 bfp8[2];
    fp8x2_storage_t bfp8x2;
  } u, w;
  u.bfp8x2 = x;
  w.bfp8[0] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[0]) * s, RCCL_BF8_MAX_FINITE));
  w.bfp8[1] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[1]) * s, RCCL_BF8_MAX_FINITE));
  return w.bfp8x2;
#endif
}

// Packed min/max. The result is always one of the inputs, so it is already
// representable and the pack needs no clamp. fminf/fmaxf are used rather than a
// packed f16 min/max so that NaN and signed-zero tie-breaks are decided by the
// same operation the scalar path uses.
inline __device__ fp8x2_storage_t hminmax2(fp8x2_storage_t x, fp8x2_storage_t y, bool isMin) {
#ifdef RCCL_FP8_PACKED_CVT
  float2_t a = rccl_unpack2_f32_fp8(x);
  float2_t b = rccl_unpack2_f32_fp8(y);
  float2_t r = {isMin ? fminf(a[0], b[0]) : fmaxf(a[0], b[0]),
                isMin ? fminf(a[1], b[1]) : fmaxf(a[1], b[1])};
  return rccl_pack2_fp8_f32(r);
#else
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, v, w;
  u.fp8x2 = x;
  v.fp8x2 = y;
  w.fp8[0] = rccl_float8(isMin ? fminf(float(u.fp8[0]), float(v.fp8[0]))
                               : fmaxf(float(u.fp8[0]), float(v.fp8[0])));
  w.fp8[1] = rccl_float8(isMin ? fminf(float(u.fp8[1]), float(v.fp8[1]))
                               : fmaxf(float(u.fp8[1]), float(v.fp8[1])));
  return w.fp8x2;
#endif
}

inline __device__ fp8x2_storage_t hminmax2_b(fp8x2_storage_t x, fp8x2_storage_t y, bool isMin) {
#ifdef RCCL_FP8_PACKED_CVT
  float2_t a = rccl_unpack2_f32_bf8(x);
  float2_t b = rccl_unpack2_f32_bf8(y);
  float2_t r = {isMin ? fminf(a[0], b[0]) : fmaxf(a[0], b[0]),
                isMin ? fminf(a[1], b[1]) : fmaxf(a[1], b[1])};
  return rccl_pack2_bf8_f32(r);
#else
  union {
    rccl_bfloat8 bfp8[2];
    fp8x2_storage_t bfp8x2;
  } u, v, w;
  u.bfp8x2 = x;
  v.bfp8x2 = y;
  w.bfp8[0] = rccl_bfloat8(isMin ? fminf(float(u.bfp8[0]), float(v.bfp8[0]))
                                 : fmaxf(float(u.bfp8[0]), float(v.bfp8[0])));
  w.bfp8[1] = rccl_bfloat8(isMin ? fminf(float(u.bfp8[1]), float(v.bfp8[1]))
                                 : fmaxf(float(u.bfp8[1]), float(v.bfp8[1])));
  return w.bfp8x2;
#endif
}

inline std::ostream& operator<<(std::ostream& os, const rccl_float8& f8) {
  return os << float(f8);
}

inline std::ostream& operator<<(std::ostream& os, const rccl_bfloat8& bf8) {
  return os << float(bf8);
}

inline __host__ __device__ float operator*(rccl_float8 a, rccl_float8 b) {
  return float(a) * float(b);
}

inline __host__ __device__ float operator*(rccl_bfloat8 a, rccl_bfloat8 b) {
  return float(a) * float(b);
}

inline __host__ __device__ float operator*(rccl_float8 a, float b) {
  return float(a) * float(b);
}

inline __host__ __device__ float operator*(rccl_bfloat8 a, float b) {
  return float(a) * float(b);
}

// For older versions of ROCm that do not include hip_fp8.h,
// we provide a local version of the header file as a fallback.
#else

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// We are clipping in down conversion by default
#define rccl_float8_downcast_clipping 1

namespace rocblas_hip_f8_impl {
__host__ inline int clz(uint32_t x) {
  return __builtin_clz(x);
}
__device__ inline int clz(uint32_t x) {
  return __clz(x);
}

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0) {
  constexpr bool is_half = std::is_same<T, _Float16>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  static_assert(wm + we == 7, "wm+we==7");
  static_assert(is_half || is_float, "Only half and float can be cast to f8");

  const int mfmt = (sizeof(T) == 4) ? 23 : 10;
  uint32_t x;
  if (sizeof(T) == 4) x = reinterpret_cast<uint32_t&>(_x);
  else x = reinterpret_cast<uint16_t&>(_x);

  uint32_t head, mantissa;
  int exponent, bias;
  uint32_t sign;

  if (sizeof(T) == 4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign = head >> 31;
    bias = 127;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head >> 10) & 0x1F;
    sign = head >> 15;
    bias = 15;
  }

  uint32_t signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

        // Deal with inf and NaNs
  if (negative_zero_nan) {
    if (sizeof(T) == 4) {
      if ((x & 0x7F800000) == 0x7F800000) return 0x80;
    } else {
                // if(__hisinf(x) || __hisnan(x))
      if ((x & 0x7C00) == 0x7C00) return 0x80;
    }
  } else {
    if (sizeof(T) == 4) {
      if ((x & 0x7F800000) == 0x7F800000) return signed_inf + (mantissa != 0 ? 1 : 0);
    } else {
      if ((x & 0x7C00) == 0x7C00) return signed_inf + (mantissa != 0 ? 1 : 0);
    }
  }
  if (x == 0) return 0;

        // First need to check if it is normal or denorm as there is a difference of implict 1
        // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
        // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
        // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
        // exponent and mantissa again

        // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
  const int f8_bias = (1 << (we - 1)) - 1 + (negative_zero_nan ? 1 : 0);
  const int f8_denormal_act_exponent = 1 - f8_bias; // actual exponent of f8 denormal
        // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
        // f8_exponent is the converted f8 exponent with bias encoding
        // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
        // the difference needs to be adjusted and mantissa shifted
  int act_exponent, f8_exponent, exponent_diff;

  if (exponent == 0) { // fp32/fp16 is in denormal.
            /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16 here.
In this case, f8 is usually in denormal. But there could be exceptions.
fp16 denormal has exponent bias 15 while bf8 with NANOO has exponent bias 16.
It means that there are some numbers in fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15.
fp16 numbers where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal.
In this case, the fp16 mantissa should be shift left by 1  */
    act_exponent = exponent - bias + 1;
    exponent_diff = f8_denormal_act_exponent - act_exponent; // actual exponent is exponent-bias+1 as it is denormal
  } else { // fp32/fp16 is normal with implicit 1
    act_exponent = exponent - bias;
    if (act_exponent <= f8_denormal_act_exponent) {
                /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
actual exponent is -7, it is actually larger due to the implict 1,
Therefore it needs to be adjust to -6 and mantissa shift right by 1.
So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
      exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else { // both fp32/fp16 and f8 are in normal range
      exponent_diff = 0; // exponent_diff=0 does not mean there is no difference for this case,
                // act_exponent could be larger. Just that it does not need shift mantissa
    }
    mantissa += (1 << mfmt); // Add the implicit 1 into mantissa
  }

  bool midpoint = (mantissa & ((1 << (mfmt - wm + exponent_diff)) - 1)) == (1 << (mfmt - wm + exponent_diff - 1));
        /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we shift right
as shift right could rip off some residual part and make something not midpoint look like midpoint.
For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than midpoint,
but after shift right by 4 bits, it would look like midpoint.
*/

  if (exponent_diff > 0) mantissa >>= exponent_diff;
  else if (exponent_diff == -1) mantissa <<= -exponent_diff;
  bool implicit_one = mantissa & (1 << mfmt);
        // if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
  f8_exponent = (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

        // Now we have the exponent and mantissa adjusted
  uint32_t drop_mask = (1 << (mfmt - wm)) - 1;
  bool odd = mantissa & (1 << (mfmt - wm)); // if the least significant bit that is not truncated is 1
  mantissa += (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) & drop_mask;

        // Now we deal with overflow
  if (f8_exponent == 0) {
    if ((1 << mfmt) & mantissa) {
      f8_exponent = 1; // denormal overflow to become normal, promote exponent
    }
  } else {
    if ((1 << (mfmt + 1)) & mantissa) {
      mantissa >>= 1;
      f8_exponent++;
    }
  }

  mantissa >>= (mfmt - wm);

        // above range: quantize to maximum possible float of the same sign
  const int max_exp = (1 << we) - (negative_zero_nan ? 1 : 2);
  if (f8_exponent > max_exp) {
    if (clip) {
      mantissa = (1 << wm) - 1;
      f8_exponent = max_exp;
    } else {
      return signed_inf;
    }
  }

  if (f8_exponent == 0 && mantissa == 0) return negative_zero_nan ? 0 : (sign << 7);
  mantissa &= (1 << wm) - 1;
  return (sign << 7) | (f8_exponent << wm) | mantissa;
}

template <int wm, int we, typename T, bool negative_zero_nan>
HIP_HOST_DEVICE T cast_from_f8(uint8_t x) {
  constexpr bool is_half = std::is_same<T, _Float16>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  static_assert(is_half || is_float, "only half and float are supported");

  constexpr int weo = is_half ? 5 : 8;
  constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

  T fInf, fNegInf, fNaN, fNeg0;
  if (is_half) {
    const uint16_t ihInf = 0x7C00;
    const uint16_t ihNegInf = 0xFC00;
    const uint16_t ihNaN = 0x7C01;
    const uint16_t ihNeg0 = 0x8000;
    fInf = reinterpret_cast<const _Float16&>(ihInf);
    fNegInf = reinterpret_cast<const _Float16&>(ihNegInf);
    fNaN = reinterpret_cast<const _Float16&>(ihNaN);
    fNeg0 = reinterpret_cast<const _Float16&>(ihNeg0);
  } else if (is_float) {
    const uint32_t ifInf = 0x7F800000;
    const uint32_t ifNegInf = 0xFF800000;
    const uint32_t ifNaN = 0x7F800001;
    const uint32_t ifNeg0 = 0x80000000;
    fInf = reinterpret_cast<const float&>(ifInf);
    fNegInf = reinterpret_cast<const float&>(ifNegInf);
    fNaN = reinterpret_cast<const float&>(ifNaN);
    fNeg0 = reinterpret_cast<const float&>(ifNeg0);
  }

  if (x == 0) return 0;

  uint32_t sign = x >> 7;
  uint32_t mantissa = x & ((1 << wm) - 1);
  int exponent = (x & 0x7F) >> wm;
  if (negative_zero_nan) {
    if (x == 0x80) return fNaN;
  } else {
    if (x == 0x80) return fNeg0;
    if (exponent == ((1 << we) - 1)) return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
  }
  typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type retval;
  if (we == 5 && is_half && !negative_zero_nan) {
    retval = x << 8;
    return reinterpret_cast<const T&>(retval);
  }

  const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

        // subnormal input
  if (exponent == 0) {
            // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + clz(mantissa) - (32 - wm);
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1 << wm) - 1);
  }
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

        // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if (exponent <= 0) {
    mantissa |= 1 << wmo;
    mantissa >>= 1 - exponent;
    exponent = 0;
  }

  if (sizeof(T) == 2) retval = (sign << 15) | (exponent << 10) | mantissa;
  else retval = (sign << 31) | (exponent << 23) | mantissa;
  return reinterpret_cast<const T&>(retval);
}
} // namespace rocblas_hip_f8_impl

static __device__ bool rocblas_hip_f8_bias_mode_bit_device = true;
static bool rocblas_hip_f8_bias_mode_bit_host = true;

struct rccl_float8 {
  uint8_t data;
  enum class rocblas_hip_f8_rounding_mode {
    standard,
    stochastic
  };

    // default constructor
  HIP_HOST_DEVICE rccl_float8() = default;

  constexpr inline HIP_HOST_DEVICE rccl_float8(const rccl_float8& a) : data(a.data) {}

#if defined(__gfx942__) || defined(__gfx950__)
    // device specific optimized F8 down-conversion code

  template <bool stochastic_rounding = false>
  static HIP_DEVICE uint8_t cast_to_f8_from_f32(float v, uint32_t rng = 0) {
    uint8_t i8data;
    union {
      float fval;
      uint32_t i32val;
      uint8_t i8val[4]; // NOTE: not endian independent
    } val;

    uint32_t ival = 0;
    val.fval = v;

#ifdef rccl_float8_downcast_clipping
    if ((val.i32val & 0x7F800000) != 0x7F800000) /// propagate NAN/INF, no clipping
      val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
#endif
    if (stochastic_rounding) {
      ival = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
      val.i32val = ival;
      i8data = val.i8val[0]; // little endian
    } else // RNE CVT
    {
      ival = __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false); // false -> WORD0
      val.i32val = ival;
      i8data = val.i8val[0];
    }
    return i8data;
  }

#endif // __gfx942__

    // constructor from float
#if defined(__gfx942__) || defined(__gfx950__)

    // NOTE: ON-DEVICE... always optimal bias
  explicit HIP_DEVICE rccl_float8(float v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard,
                                  uint32_t rng = 0) {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
    if (rm == rocblas_hip_f8_rounding_mode::stochastic) data = cast_to_f8_from_f32<true>(v, rng);
    else data = cast_to_f8_from_f32<false>(v);
  }

    // Host only implementation using s/w simulation
  explicit HIP_HOST
#else
    // both Host and DEVICE for non-gfx942 using s/w simulation
  explicit HIP_HOST_DEVICE
#endif
  rccl_float8(float v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard, uint32_t rng = 0) {
#ifdef rccl_float8_downcast_clipping
    data = rocblas_hip_f8_impl::cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
      v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#else // rccl_float8_downcast_clipping
    data = rocblas_hip_f8_impl::cast_to_f8<3, 4, float, true /*negative_zero_nan*/, false /*clip*/>(
      v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#endif // rccl_float8_downcast_clipping
  }

    // Constructor from half
  explicit HIP_HOST_DEVICE rccl_float8(
    _Float16 v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard, uint32_t rng = 0)
    : rccl_float8((float)v, rm, rng) {}
    // constructor from int
  explicit HIP_HOST_DEVICE rccl_float8(int v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard,
                                       uint32_t rng = 0)
    : rccl_float8((float)v, rm, rng) {}
    // constructor from double
  explicit HIP_HOST_DEVICE rccl_float8(
    double v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard, uint32_t rng = 0)
    : rccl_float8((float)v, rm, rng) {}

    // convert to float
#if defined(__gfx942__) || defined(__gfx950__)
    // upcast using device specific intrinsic
  explicit inline HIP_DEVICE operator float() const {
    float fval;
    uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
    asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

    return fval;
  }

  explicit inline HIP_HOST operator float() const
#else // non gfx942
  explicit inline HIP_HOST_DEVICE operator float() const
#endif
  {
    return rocblas_hip_f8_impl::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(data);
  }

    // convert to half
  explicit inline HIP_HOST_DEVICE operator _Float16() const {
    return _Float16(float(*this)); // convert to float, then convert to f16
  }

    // check for zero
  inline HIP_HOST_DEVICE bool is_zero() const {
    return data == 0x00;
  }

    // check for nan
  inline HIP_HOST_DEVICE bool is_nan() const {
    return data == 0x80;
  }

    // check for inf
  inline HIP_HOST_DEVICE bool is_inf() const {
    return data == 0x80;
  }

    // assignment overloading only from the same F8 types
  inline HIP_HOST_DEVICE rccl_float8& operator=(const rccl_float8& a) {
    data = a.data;
    return *this;
  }
};

struct rccl_bfloat8 {
  uint8_t data;
  enum class rocblas_hip_f8_rounding_mode {
    standard,
    stochastic
  };

    // default constructor
  HIP_HOST_DEVICE rccl_bfloat8() = default;

  constexpr inline HIP_HOST_DEVICE rccl_bfloat8(const rccl_bfloat8& a) : data(a.data) {}

#if defined(__gfx942__) || defined(__gfx950__)
    // device specific optimized F8 down-conversion code

  template <bool stochastic_rounding = false>
  static HIP_DEVICE uint8_t cast_to_bf8_from_f32(float v, uint32_t rng = 0) {
    uint8_t i8data;
    union {
      float fval;
      uint32_t i32val;
      uint8_t i8val[4]; // NOTE: not endian independent
    } val;

    uint32_t ival = 0;
    val.fval = v;

#ifdef rccl_float8_downcast_clipping
    if ((val.i32val & 0x7F800000) != 0x7F800000) // propagate NAN/INF, no clipping
      val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
#endif
    if (stochastic_rounding) {
      ival = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
      val.i32val = ival;
      i8data = val.i8val[0]; // little endian
    } else // RNE CVT
    {
      ival = __builtin_amdgcn_cvt_pk_bf8_f32(val.fval, val.fval, ival, false); // false -> WORD0
      val.i32val = ival;
      i8data = val.i8val[0];
    }
    return i8data;
  }

#endif // __gfx942__

    // constructor from float
#if defined(__gfx942__) || defined(__gfx950__)

    // NOTE: ON-DEVICE... always optimal bias
  explicit HIP_DEVICE rccl_bfloat8(float v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard,
                                   uint32_t rng = 0) {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
    if (rm == rocblas_hip_f8_rounding_mode::stochastic) data = cast_to_bf8_from_f32<true>(v, rng);
    else data = cast_to_bf8_from_f32<false>(v);
  }

    // Host only implementation using s/w simulation
  explicit HIP_HOST
#else
    // both Host and DEVICE for non-gfx942 using s/w simulation
  explicit HIP_HOST_DEVICE
#endif
  rccl_bfloat8(float v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard, uint32_t rng = 0) {
#ifdef rccl_float8_downcast_clipping
    data = rocblas_hip_f8_impl::cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
      v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#else
    data = rocblas_hip_f8_impl::cast_to_f8<2, 5, float, true /*negative_zero_nan*/, false /*clip*/>(
      v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#endif // rccl_float8_downcast_clipping
  }

    // Constructor from half
  explicit HIP_HOST_DEVICE rccl_bfloat8(
    _Float16 v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard, uint32_t rng = 0)
    : rccl_bfloat8((float)v, rm, rng) {}
    // constructor from int
  explicit HIP_HOST_DEVICE rccl_bfloat8(int v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard,
                                        uint32_t rng = 0)
    : rccl_bfloat8((float)v, rm, rng) {}
    // constructor from double
  explicit HIP_HOST_DEVICE rccl_bfloat8(
    double v, rocblas_hip_f8_rounding_mode rm = rocblas_hip_f8_rounding_mode::standard, uint32_t rng = 0)
    : rccl_bfloat8((float)v, rm, rng) {}

    // convert to float
#if defined(__gfx942__) || defined(__gfx950__)
    // upcast using device specific intrinsic
  explicit inline HIP_DEVICE operator float() const {
    float fval;
    uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
    asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

    return fval;
  }

  explicit inline HIP_HOST operator float() const
#else // non gfx942
  explicit inline HIP_HOST_DEVICE operator float() const
#endif
  {
    return rocblas_hip_f8_impl::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(data);
  }

    // convert to half
  explicit inline HIP_HOST_DEVICE operator _Float16() const {
    return _Float16(float(*this)); // convert to float, then convert to f16
  }

    // check for zero
  inline HIP_HOST_DEVICE bool is_zero() const {
    return data == 0x00;
  }

    // check for nan
  inline HIP_HOST_DEVICE bool is_nan() const {
    return data == 0x80;
  }

    // check for inf
  inline HIP_HOST_DEVICE bool is_inf() const {
    return data == 0x80;
  }

    // assignment overloading only from the same F8 types
  inline HIP_HOST_DEVICE rccl_bfloat8& operator=(const rccl_bfloat8& a) {
    data = a.data;
    return *this;
  }
};

namespace std {
inline rccl_float8 sin(rccl_float8 a) {
  return rccl_float8(sinf(float(a)));
}
inline rccl_float8 cos(rccl_float8 a) {
  return rccl_float8(cosf(float(a)));
}
inline rccl_bfloat8 sin(rccl_bfloat8 a) {
  return rccl_bfloat8(sinf(float(a)));
}
inline rccl_bfloat8 cos(rccl_bfloat8 a) {
  return rccl_bfloat8(cosf(float(a)));
}
HIP_HOST_DEVICE constexpr rccl_float8 real(const rccl_float8& a) {
  return a;
}
HIP_HOST_DEVICE constexpr rccl_bfloat8 real(const rccl_bfloat8& a) {
  return a;
}
} // namespace std

inline __device__ rccl_float8 hadd(rccl_float8 x, rccl_float8 y) {
  return rccl_float8(float(x) + float(y));
}

inline __device__ fp8x2_storage_t hadd2(fp8x2_storage_t x, fp8x2_storage_t y) {
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, v, w;
  u.fp8x2 = x;
  v.fp8x2 = y;
  w.fp8[0] = hadd(u.fp8[0], v.fp8[0]);
  w.fp8[1] = hadd(u.fp8[1], v.fp8[1]);

  return w.fp8x2;
}

inline __device__ rccl_bfloat8 hadd_b(rccl_bfloat8 x, rccl_bfloat8 y) {
  return rccl_bfloat8(float(x) + float(y));
}

inline __device__ fp8x2_storage_t hadd2_b(fp8x2_storage_t x, fp8x2_storage_t y) {
  union {
    rccl_bfloat8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, v, w;
  u.fp8x2 = x;
  v.fp8x2 = y;
  w.fp8[0] = hadd_b(u.fp8[0], v.fp8[0]);
  w.fp8[1] = hadd_b(u.fp8[1], v.fp8[1]);

  return w.fp8x2;
}

// Per-element forms of the packed reduce helpers, for builds without the fp8
// conversion instructions. The rccl_float8/rccl_bfloat8 constructors saturate,
// so these define the semantics the packed paths above have to reproduce.
inline __device__ fp8x2_storage_t hmul2(fp8x2_storage_t x, fp8x2_storage_t y) {
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, v, w;
  u.fp8x2 = x;
  v.fp8x2 = y;
  w.fp8[0] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[0]) * float(v.fp8[0]), RCCL_FP8_MAX_FINITE));
  w.fp8[1] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[1]) * float(v.fp8[1]), RCCL_FP8_MAX_FINITE));

  return w.fp8x2;
}

inline __device__ fp8x2_storage_t hmul2_b(fp8x2_storage_t x, fp8x2_storage_t y) {
  union {
    rccl_bfloat8 bfp8[2];
    fp8x2_storage_t bfp8x2;
  } u, v, w;
  u.bfp8x2 = x;
  v.bfp8x2 = y;
  w.bfp8[0] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[0]) * float(v.bfp8[0]), RCCL_BF8_MAX_FINITE));
  w.bfp8[1] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[1]) * float(v.bfp8[1]), RCCL_BF8_MAX_FINITE));

  return w.bfp8x2;
}

inline __device__ fp8x2_storage_t hpremul2(fp8x2_storage_t x, float s) {
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, w;
  u.fp8x2 = x;
  w.fp8[0] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[0]) * s, RCCL_FP8_MAX_FINITE));
  w.fp8[1] = rccl_float8(rccl_narrow_sat_f32(float(u.fp8[1]) * s, RCCL_FP8_MAX_FINITE));

  return w.fp8x2;
}

inline __device__ fp8x2_storage_t hpremul2_b(fp8x2_storage_t x, float s) {
  union {
    rccl_bfloat8 bfp8[2];
    fp8x2_storage_t bfp8x2;
  } u, w;
  u.bfp8x2 = x;
  w.bfp8[0] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[0]) * s, RCCL_BF8_MAX_FINITE));
  w.bfp8[1] = rccl_bfloat8(rccl_narrow_sat_f32(float(u.bfp8[1]) * s, RCCL_BF8_MAX_FINITE));

  return w.bfp8x2;
}

inline __device__ fp8x2_storage_t hminmax2(fp8x2_storage_t x, fp8x2_storage_t y, bool isMin) {
  union {
    rccl_float8 fp8[2];
    fp8x2_storage_t fp8x2;
  } u, v, w;
  u.fp8x2 = x;
  v.fp8x2 = y;
  w.fp8[0] = rccl_float8(isMin ? fminf(float(u.fp8[0]), float(v.fp8[0]))
                               : fmaxf(float(u.fp8[0]), float(v.fp8[0])));
  w.fp8[1] = rccl_float8(isMin ? fminf(float(u.fp8[1]), float(v.fp8[1]))
                               : fmaxf(float(u.fp8[1]), float(v.fp8[1])));

  return w.fp8x2;
}

inline __device__ fp8x2_storage_t hminmax2_b(fp8x2_storage_t x, fp8x2_storage_t y, bool isMin) {
  union {
    rccl_bfloat8 bfp8[2];
    fp8x2_storage_t bfp8x2;
  } u, v, w;
  u.bfp8x2 = x;
  v.bfp8x2 = y;
  w.bfp8[0] = rccl_bfloat8(isMin ? fminf(float(u.bfp8[0]), float(v.bfp8[0]))
                                 : fmaxf(float(u.bfp8[0]), float(v.bfp8[0])));
  w.bfp8[1] = rccl_bfloat8(isMin ? fminf(float(u.bfp8[1]), float(v.bfp8[1]))
                                 : fmaxf(float(u.bfp8[1]), float(v.bfp8[1])));

  return w.bfp8x2;
}

// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, const rccl_float8& f8) {
  return os << float(f8);
}

inline std::ostream& operator<<(std::ostream& os, const rccl_bfloat8& bf8) {
  return os << float(bf8);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator+(const float fa, rccl_float8 b) {
  return (fa + float(b));
}

inline __host__ __device__ float operator+(const float fa, rccl_bfloat8 b) {
  return (fa + float(b));
}

inline __host__ __device__ float operator+(rccl_float8 a, const float fb) {
  return (float(a) + fb);
}

inline __host__ __device__ float operator+(rccl_bfloat8 a, const float fb) {
  return (float(a) + fb);
}

inline __host__ __device__ float operator+(rccl_float8 a, rccl_bfloat8 b) {
  return (float(a) + float(b));
}

inline __host__ __device__ float operator+(rccl_bfloat8 a, rccl_float8 b) {
  return (float(a) + float(b));
}

inline __host__ __device__ rccl_float8 operator+(rccl_float8 a, rccl_float8 b) {
  return rccl_float8(float(a) + float(b));
}

inline __host__ __device__ rccl_bfloat8 operator+(rccl_bfloat8 a, rccl_bfloat8 b) {
  return rccl_bfloat8(float(a) + float(b));
}

inline __host__ __device__ rccl_float8& operator+=(rccl_float8& a, rccl_float8 b) {
  return a = rccl_float8(float(a) + float(b));
}

inline __host__ __device__ rccl_bfloat8& operator+=(rccl_bfloat8& a, rccl_bfloat8 b) {
  return a = rccl_bfloat8(float(a) + float(b));
}

// overloading multiplication, always returns float,
inline __host__ __device__ float operator*(rccl_float8 a, rccl_float8 b) {
  return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, rccl_float8 b) {
  return (a * float(b));
}

inline __host__ __device__ float operator*(rccl_float8 a, float b) {
  return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, rccl_float8 b) {
  return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, rccl_float8 b) {
  return ((float)a * float(b));
}

inline __host__ __device__ float operator*(rccl_bfloat8 a, rccl_bfloat8 b) {
  return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, rccl_bfloat8 b) {
  return (a * float(b));
}

inline __host__ __device__ float operator*(rccl_bfloat8 a, float b) {
  return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, rccl_bfloat8 b) {
  return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, rccl_bfloat8 b) {
  return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator*(rccl_float8 a, rccl_bfloat8 b) {
  return float(a) * float(b);
}

inline __host__ __device__ float operator*(rccl_bfloat8 a, rccl_float8 b) {
  return float(a) * float(b);
}

// all - operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator-(const float fa, rccl_float8 b) {
  return (fa - float(b));
}

inline __host__ __device__ float operator-(const float fa, rccl_bfloat8 b) {
  return (fa - float(b));
}

inline __host__ __device__ float operator-(rccl_float8 a, const float fb) {
  return (float(a) - fb);
}

inline __host__ __device__ float operator-(rccl_bfloat8 a, const float fb) {
  return (float(a) - fb);
}

inline __host__ __device__ float operator-(rccl_float8 a, rccl_bfloat8 b) {
  return (float(a) - float(b));
}

inline __host__ __device__ float operator-(rccl_bfloat8 a, rccl_float8 b) {
  return (float(a) - float(b));
}

inline __host__ __device__ rccl_float8 operator-(rccl_float8 a, rccl_float8 b) {
  return rccl_float8(float(a) - float(b));
}

inline __host__ __device__ rccl_bfloat8 operator-(rccl_bfloat8 a, rccl_bfloat8 b) {
  return rccl_bfloat8(float(a) - float(b));
}

inline __host__ __device__ rccl_float8& operator-=(rccl_float8& a, rccl_float8 b) {
  return a = rccl_float8(float(a) - float(b));
}

inline __host__ __device__ rccl_bfloat8& operator-=(rccl_bfloat8& a, rccl_bfloat8 b) {
  return a = rccl_bfloat8(float(a) - float(b));
}

// overloading division, always returns float,
inline __host__ __device__ float operator/(rccl_float8 a, rccl_float8 b) {
  return float(a) / float(b);
}

inline __host__ __device__ float operator/(float a, rccl_float8 b) {
  return (a / float(b));
}

inline __host__ __device__ float operator/(rccl_float8 a, float b) {
  return (float(a) / b);
}

inline __host__ __device__ float operator/(int32_t a, rccl_float8 b) {
  return ((float)a / float(b));
}

inline __host__ __device__ float operator/(double a, rccl_float8 b) {
  return ((float)a / float(b));
}

inline __host__ __device__ float operator/(rccl_bfloat8 a, rccl_bfloat8 b) {
  return float(a) / float(b);
}

inline __host__ __device__ float operator/(float a, rccl_bfloat8 b) {
  return (a / float(b));
}

inline __host__ __device__ float operator/(rccl_bfloat8 a, float b) {
  return (float(a) / b);
}

inline __host__ __device__ float operator/(int32_t a, rccl_bfloat8 b) {
  return ((float)a / float(b));
}

inline __host__ __device__ float operator/(double a, rccl_bfloat8 b) {
  return ((float)a / float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator/(rccl_float8 a, rccl_bfloat8 b) {
  return float(a) / float(b);
}

inline __host__ __device__ float operator/(rccl_bfloat8 a, rccl_float8 b) {
  return float(a) / float(b);
}

// overloading for compare
inline __host__ __device__ bool operator==(rccl_float8 a, rccl_float8 b) {
  return (a.data == b.data);
}

inline __host__ __device__ bool operator==(rccl_bfloat8 a, rccl_bfloat8 b) {
  return (a.data == b.data);
}

inline __host__ __device__ bool operator!=(rccl_float8 a, rccl_float8 b) {
  return (a.data != b.data);
}

inline __host__ __device__ bool operator!=(rccl_bfloat8 a, rccl_bfloat8 b) {
  return (a.data != b.data);
}

// ================ Explicit downcasting to support different rounding (RNE, SR) ===============
// NOTE: we going to remove all assignment operator overloading from other types and enforce
// this explicit_downcast function to make any roudning behavior default
// We have to explicitly call this function with SR flag

template <typename T, typename Ta, bool stochastic_rounding,
          typename std::enable_if<std::is_same<T, Ta>{}, int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng = 0) {
    // same type, no conversion
  return a;
}

// Use h/w intrinsic and optimized version when __gfx942__
template <
  typename T, typename Ta, bool stochastic_rounding,
  typename std::enable_if<
    (!(std::is_same<T, Ta>{}) && (std::is_same<T, rccl_float8>{} || std::is_same<T, rccl_bfloat8>{})), int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng) {
#if defined(__gfx942__) || defined(__gfx950__)
    // NOTE: we are directly calling cast_to_f8_from_f32 instead of constructor to optimize away one runtime branch
  T val;
  if (std::is_same<T, rccl_float8>::value)
    val.data = rccl_float8::cast_to_f8_from_f32<stochastic_rounding>(float(a), rng);
  else val.data = rccl_bfloat8::cast_to_bf8_from_f32<stochastic_rounding>(float(a), rng);
  return val;
#else // non gfx942
  return T(
    float(a),
    stochastic_rounding ? T::rocblas_hip_f8_rounding_mode::stochastic : T::rocblas_hip_f8_rounding_mode::standard, rng);
#endif // __gfx942__
}

// NOTE NOTE: The above code is good if we don't consider HIP-GEMM code and only consider the quantization
// However, if we need HIP-GEMM for fall-back, we would need explicit_cast handles Tacc=f32 to To=f16/bf16 conversion
template <
  typename T, typename Ta, bool stochastic_rounding,
  typename std::enable_if<
    (!(std::is_same<T, Ta>{}) && !(std::is_same<T, rccl_float8>{} || std::is_same<T, rccl_bfloat8>{})), int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng) {
    // the return type is not a F8 types, no SR for those types
    // not sure if we have direct conversion, so converting to float first
    // no effect if the input type is float
  return T(float(a));
}

// =================================================================================================

#endif

#endif // ROCBLAS_FLOAT8_H
