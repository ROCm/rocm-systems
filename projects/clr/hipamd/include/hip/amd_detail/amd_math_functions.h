/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "hip_fp16_math_fwd.h"
#include "amd_hip_vector_types.h"
#include "math_fwd.h"

#include <hip/amd_detail/host_defines.h>

#include <algorithm>
// assert.h is only for the host version of assert.
// The device version of assert is implemented in hip/amd_detail/hip_runtime.h.
// Users should include hip_runtime.h for the device version of assert.
#if !__HIP_DEVICE_COMPILE__
#include <assert.h>
#endif
#include <limits.h>
#include <limits>
#include <stdint.h>
#endif  // !defined(__HIPCC_RTC__)

#pragma push_macro("__DEVICE__")
#pragma push_macro("__RETURN_TYPE")

#define __DEVICE__ static __device__
#define __RETURN_TYPE bool

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__DEVICE__
inline int amd_mixed_dot(short2 a, short2 b, int c, bool saturate) {
  return __ockl_sdot2(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline uint amd_mixed_dot(ushort2 a, ushort2 b, uint c, bool saturate) {
  return __ockl_udot2(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline int amd_mixed_dot(char4 a, char4 b, int c, bool saturate) {
  return __ockl_sdot4(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate) {
  return __ockl_udot4(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline int amd_mixed_dot(int a, int b, int c, bool saturate) {
  return __ockl_sdot8(a, b, c, saturate);
}
__DEVICE__
inline uint amd_mixed_dot(uint a, uint b, uint c, bool saturate) {
  return __ockl_udot8(a, b, c, saturate);
}
#endif

#pragma pop_macro("__DEVICE__")
#pragma pop_macro("__RETURN_TYPE")
// For backward compatibility.
// There are HIP applications e.g. TensorFlow, expecting __HIP_ARCH_* macros
// defined after including math_functions.h.
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h>
#endif
