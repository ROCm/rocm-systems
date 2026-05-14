/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
__managed__ int x = 10;

extern "C" __global__ void GPU_func() { x++; }
