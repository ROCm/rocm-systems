/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#if defined(__gfx950__)
#define HIP_ENABLE_GFX950_OCP_BUILTINS 1
#else
#define HIP_ENABLE_GFX950_OCP_BUILTINS 0
#endif
#if defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__gfx1251__)
#define HIP_ENABLE_GFX1250_OCP_BUILTINS 1
#else
#define HIP_ENABLE_GFX1250_OCP_BUILTINS 0
#endif
#if !defined(__gfx950__) && !defined(__gfx1250__) && !defined(__gfx1250_strict__) && !defined(__gfx1251__)
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 1
#else
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 0
#endif

#if !defined(__HIPCC_RTC__)
#include "amd_hip_ocp_types.h"
#include "amd_hip_fp16.h"
#include "amd_hip_bf16.h"
#endif

enum hipRoundMode {
  hipRoundNearest = 0,
  hipRoundZero = 1,
  hipRoundPosInf = 2,
  hipRoundMinInf = 3,
};

#if (defined(__clang__) && (__clang_major__ > 17)) || (defined(__GNUC__) && (__GNUC__ >= 13))

namespace internal {
__host__ __device__ static inline __amd_fp16_storage_t half_to_f16(const __half val) {
  const __half_raw tmp = val;
  return __builtin_bit_cast(__amd_fp16_storage_t, tmp);
}

__host__ __device__ static inline __amd_fp16x2_storage_t half2_to_f16x2(const __half2 val) {
  const __half2_raw tmp = val;
  return __builtin_bit_cast(__amd_fp16x2_storage_t, tmp);
}

__host__ __device__ static inline __amd_bf16_storage_t hipbf16_to_bf16(const __hip_bfloat16 val) {
  const __hip_bfloat16_raw tmp = val;
  return __builtin_bit_cast(__amd_bf16_storage_t, tmp);
}

__host__ __device__ static inline __amd_bf16x2_storage_t hipbf162_to_bf16x2(
    const __hip_bfloat162_raw val) {
  return __builtin_bit_cast(__amd_bf16x2_storage_t, val);
}

__host__ __device__ static inline __amd_bf16x2_storage_t hipbf162_to_bf16x2(const __hip_bfloat162 val) {
  return hipbf162_to_bf16x2(static_cast<__hip_bfloat162_raw>(val));
}

__host__ __device__ static inline __half_raw f16_to_halfraw(const __amd_fp16_storage_t val) {
  return __builtin_bit_cast(__half_raw, val);
}

__host__ __device__ static inline __half2_raw f16x2_to_halfraw2(const __amd_fp16x2_storage_t val) {
  return __builtin_bit_cast(__half2_raw, val);
}

}  // namespace internal

#endif
