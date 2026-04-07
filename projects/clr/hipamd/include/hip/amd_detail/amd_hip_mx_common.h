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
#if !defined(__gfx950__)
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 1
#else
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 0
#endif

#include "amd_hip_ocp_types.h"
#include "amd_hip_fp16.h"
#include "amd_hip_bf16.h"

enum hipRoundMode {
  hipRoundNearest = 0,
  hipRoundZero = 1,
  hipRoundPosInf = 2,
  hipRoundMinInf = 3,
};

#if defined(__clang__) && defined(__HIP__)

namespace internal {
__host__ __device__ static inline __amd_fp16_storage_t half_to_f16(const __half val) {
  __half_raw tmp = val;
  return tmp.data;
}

__host__ __device__ static inline __amd_fp16x2_storage_t half2_to_f16x2(const __half2 val) {
  __half2_raw tmp = val;
  return tmp.data;
}

__host__ __device__ static inline __amd_bf16_storage_t hipbf16_to_bf16(const __hip_bfloat16 val) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(__amd_bf16_storage_t));
  union {
    __hip_bfloat16 hip_bf16;
    __amd_bf16_storage_t bf16;
  } u{val};
  return u.bf16;
}

__host__ __device__ static inline __amd_bf16x2_storage_t hipbf162_to_bf16x2(const __hip_bfloat162 val) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(__amd_bf16x2_storage_t));
  union {
    __hip_bfloat162 hip_bf16;
    __amd_bf16x2_storage_t bf16;
  } u{val};
  return u.bf16;
}

}  // namespace internal

#endif  // defined(__clang__) && defined(__HIP__)
