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
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
#define HIP_ENABLE_GFX1250_OCP_BUILTINS 1
#else
#define HIP_ENABLE_GFX1250_OCP_BUILTINS 0
#endif
#if defined(__gfx1250__)
#define HIP_ENABLE_GFX1250_BLOCK16_BUILTINS 1
#else
// gfx1250-strict does not have the block16 convert ops, so the fp4/fp6 ->
// f16/bf16/f32 unpack builtins are unavailable there even though the rest of
// the gfx1250 OCP builtins are supported.
#define HIP_ENABLE_GFX1250_BLOCK16_BUILTINS 0
#endif
#if defined(__gfx1250__)
#define HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS 1
#else
// The pk8 scaled unpack converts (fp4, fp8, bf8) are tracked separately from the
// pk16 ones (fp6, bf6). LLVM currently requires block16-cvt-scale-insts for both,
// and gfx1250-strict lacks that feature, so they are unavailable there today.
// LCOMPILER-2841 reports that the pk8 opcodes do exist on MI450-A0 and that only
// their block16 OP_SEL values are B0-only, which would make this gate unnecessary.
// Keeping it as its own predicate means that decision can be reverted by editing
// this one macro rather than the call sites, and without disturbing the pk16
// gating, which is correct either way.
#define HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS 0
#endif
#if !defined(__gfx950__) && !defined(__gfx1250__) && !defined(__gfx1250_strict__)
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
