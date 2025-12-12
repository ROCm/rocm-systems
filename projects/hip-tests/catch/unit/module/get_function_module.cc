/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>

extern "C" {
__global__ void GlobalKernel() {}

__device__ void DeviceKernel() {}
}