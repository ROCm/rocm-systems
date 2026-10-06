/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#include "amd_hip_vector_types.h"  // For Native_vec_
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__device__ __attribute__((const)) int __ockl_sdot2(HIP_vector_base<short, 2>::Native_vec_,
                                                   HIP_vector_base<short, 2>::Native_vec_, int,
                                                   bool);

__device__ __attribute__((const)) unsigned int __ockl_udot2(
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    HIP_vector_base<unsigned short, 2>::Native_vec_, unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot4(HIP_vector_base<char, 4>::Native_vec_,
                                                   HIP_vector_base<char, 4>::Native_vec_, int,
                                                   bool);

__device__ __attribute__((const)) unsigned int __ockl_udot4(
    HIP_vector_base<unsigned char, 4>::Native_vec_, HIP_vector_base<unsigned char, 4>::Native_vec_,
    unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot8(int, int, int, bool);

__device__ __attribute__((const)) unsigned int __ockl_udot8(unsigned int, unsigned int,
                                                            unsigned int, bool);
#endif

// BEGIN FLOAT
__device__ __attribute__((const)) float __ocml_ceil_f32(float);
__device__ __attribute__((const)) __device__ float __ocml_copysign_f32(float, float);
__device__ float __ocml_cos_f32(float);
__device__ __attribute__((pure)) float __ocml_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_exp_f32(float);
__device__ __attribute__((const)) float __ocml_fabs_f32(float);
__device__ __attribute__((const)) float __ocml_floor_f32(float);
__device__ __attribute__((const)) float __ocml_fma_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fmax_f32(float, float);
__device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
__device__ float __ocml_frexp_f32(float, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) int __ocml_isfinite_f32(float);
__device__ __attribute__((const)) int __ocml_isinf_f32(float);
__device__ __attribute__((const)) int __ocml_isnan_f32(float);
__device__ __attribute__((const)) float __ocml_ldexp_f32(float, int);
__device__ __attribute__((pure)) float __ocml_log10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log10_f32(float);
__device__ __attribute__((pure)) float __ocml_log2_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log2_f32(float);
__device__ __attribute__((const)) float __ocml_logb_f32(float);
__device__ __attribute__((pure)) float __ocml_log_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log_f32(float);
__device__ __attribute__((const)) float __ocml_nearbyint_f32(float);
__device__ __attribute__((const)) float __ocml_rint_f32(float);
__device__ __attribute__((const)) float __ocml_round_f32(float);
__device__ __attribute__((pure)) float __ocml_rsqrt_f32(float);
__device__ __attribute__((const)) float __ocml_scalb_f32(float, float);
__device__ __attribute__((const)) float __ocml_scalbn_f32(float, int);
__device__ __attribute__((const)) int __ocml_signbit_f32(float);
__device__ float __ocml_sin_f32(float);
__device__ __attribute__((const)) float __ocml_trunc_f32(float);
// END FLOAT

// BEGIN DOUBLE
__device__ __attribute__((const)) double __ocml_ceil_f64(double);
__device__ __attribute__((const)) double __ocml_copysign_f64(double, double);
__device__ __attribute__((const)) double __ocml_fabs_f64(double);
__device__ __attribute__((const)) double __ocml_floor_f64(double);
__device__ __attribute__((const)) double __ocml_fma_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fmax_f64(double, double);
__device__ __attribute__((const)) double __ocml_fmin_f64(double, double);
__device__ double __ocml_frexp_f64(double, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) int __ocml_isfinite_f64(double);
__device__ __attribute__((const)) int __ocml_isinf_f64(double);
__device__ __attribute__((const)) int __ocml_isnan_f64(double);
__device__ __attribute__((const)) double __ocml_ldexp_f64(double, int);
__device__ __attribute__((const)) double __ocml_logb_f64(double);
__device__ __attribute__((const)) double __ocml_nearbyint_f64(double);
__device__ __attribute__((const)) double __ocml_rint_f64(double);
__device__ __attribute__((const)) double __ocml_round_f64(double);
__device__ __attribute__((const)) double __ocml_scalb_f64(double, double);
__device__ __attribute__((const)) double __ocml_scalbn_f64(double, int);
__device__ __attribute__((const)) int __ocml_signbit_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_f64(double);
__device__ __attribute__((const)) double __ocml_trunc_f64(double);
// END DOUBLE

#if defined(__cplusplus)
}  // extern "C"
#endif
