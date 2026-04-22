/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_LDG_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_LDG_H

#if __HIP_CLANG_ONLY__
#include "amd_hip_vector_types.h"
#include "host_defines.h"

__device__ inline static char __ldg(const char* ptr) { return *ptr; }

__device__ inline static char2 __ldg(const char2* ptr) { return *ptr; }

__device__ inline static char4 __ldg(const char4* ptr) { return *ptr; }

__device__ inline static signed char __ldg(const signed char* ptr) { return ptr[0]; }

__device__ inline static unsigned char __ldg(const unsigned char* ptr) { return ptr[0]; }


__device__ inline static short __ldg(const short* ptr) { return ptr[0]; }

__device__ inline static short2 __ldg(const short2* ptr) { return ptr[0]; }

__device__ inline static short4 __ldg(const short4* ptr) { return ptr[0]; }

__device__ inline static unsigned short __ldg(const unsigned short* ptr) { return ptr[0]; }


__device__ inline static int __ldg(const int* ptr) { return ptr[0]; }

__device__ inline static int2 __ldg(const int2* ptr) { return ptr[0]; }

__device__ inline static int4 __ldg(const int4* ptr) { return ptr[0]; }

__device__ inline static unsigned int __ldg(const unsigned int* ptr) { return ptr[0]; }


__device__ inline static long __ldg(const long* ptr) { return ptr[0]; }

__device__ inline static unsigned long __ldg(const unsigned long* ptr) { return ptr[0]; }


__device__ inline static long long __ldg(const long long* ptr) { return ptr[0]; }

__device__ inline static longlong2 __ldg(const longlong2* ptr) { return ptr[0]; }

__device__ inline static unsigned long long __ldg(const unsigned long long* ptr) { return ptr[0]; }


__device__ inline static uchar2 __ldg(const uchar2* ptr) { return ptr[0]; }

__device__ inline static uchar4 __ldg(const uchar4* ptr) { return ptr[0]; }


__device__ inline static ushort2 __ldg(const ushort2* ptr) { return ptr[0]; }


__device__ inline static uint2 __ldg(const uint2* ptr) { return ptr[0]; }

__device__ inline static uint4 __ldg(const uint4* ptr) { return ptr[0]; }


__device__ inline static ulonglong2 __ldg(const ulonglong2* ptr) { return ptr[0]; }


__device__ inline static float __ldg(const float* ptr) { return ptr[0]; }

__device__ inline static float2 __ldg(const float2* ptr) { return ptr[0]; }

__device__ inline static float4 __ldg(const float4* ptr) { return ptr[0]; }


__device__ inline static double __ldg(const double* ptr) { return ptr[0]; }

__device__ inline static double2 __ldg(const double2* ptr) { return ptr[0]; }

#endif  // __HIP_CLANG_ONLY__

#endif  // HIP_LDG_H
