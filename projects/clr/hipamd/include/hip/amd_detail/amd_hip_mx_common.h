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
// The pk16 fp6/bf6 unpack converts. LCOMPILER-2841 established that all six
// execute on MI450-A0; what A0 lacks is the block16 half of the scale_sel
// range, not the opcodes. llvm/llvm-project#227475 moves them off
// block16-cvt-scale-insts onto gfx1250-insts, which gfx1250-strict has, and
// validates scale_sel instead: 0-3 on strict, 0-7 with block16. Every HIP
// entry point here passes scale_sel 0, so both targets are in range.
//
// This requires a compiler carrying that change. Without it, gfx1250-strict
// fails to compile with "needs target feature block16-cvt-scale-insts" rather
// than silently taking a slower path, so the dependency is enforced by the
// build.
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
#define HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS 1
#else
#define HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS 0
#endif
// The pk8 scaled unpack converts (fp4, fp8, bf8) are tracked separately from the
// pk16 ones (fp6, bf6). LCOMPILER-2841 established that the nine pk8 opcodes do
// exist on MI450-A0 at identical encodings: only their block16 OP_SEL values
// (scale_sel 4-7) are B0-only, and every HIP entry point here passes scale_sel 0,
// which is block32 and valid on A0. llvm/llvm-project#227426 moves them off
// block16-cvt-scale-insts onto gfx1250-insts, which gfx1250-strict has, so this
// gate covers both targets.
//
// This requires a compiler carrying that change. Without it, gfx1250-strict
// fails to compile with "needs target feature block16-cvt-scale-insts" rather
// than silently taking a slower path, so the dependency is enforced by the
// build. The pk16 gate below covers gfx1250-strict for the same reason:
// llvm/llvm-project#227475 moves those six opcodes off block16-cvt-scale-insts
// as well, and validates scale_sel in their place.
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
#define HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS 1
#else
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
