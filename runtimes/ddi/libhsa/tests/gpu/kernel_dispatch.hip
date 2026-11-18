// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
extern "C" __global__ void rocddi_increment(unsigned *out) { out[0] += 1; }
