/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_FP16_H
#define HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_FP16_H

#define HIPRT_ONE_BF16 CUDART_ONE_BF16
#define HIPRT_ZERO_BF16 CUDART_ZERO_BF16
#define HIPRT_INF_BF16 CUDART_INF_BF16
#define HIPRT_MAX_NORMAL_BF16 CUDART_MAX_NORMAL_BF16
#define HIPRT_MIN_DENORM_BF16 CUDART_MIN_DENORM_BF16
#define HIPRT_NAN_BF16 CUDART_NAN_BF16
#define HIPRT_NEG_ZERO_BF16 CUDART_NEG_ZERO_BF16

#include <cuda_bf16.h>

typedef struct __nv_bfloat16 __hip_bfloat16;
typedef struct __nv_bfloat162 __hip_bfloat162;

#endif  // HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_FP16_H
