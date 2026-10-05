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
__device__ float __ocml_cos_f32(float);
__device__ __attribute__((pure)) float __ocml_rsqrt_f32(float);
__device__ float __ocml_sin_f32(float);

// END FLOAT

#if defined(__cplusplus)
}  // extern "C"
#endif
