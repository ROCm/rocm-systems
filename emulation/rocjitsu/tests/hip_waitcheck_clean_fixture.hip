// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

extern "C" __global__ void waitcheck_clean_kernel() {
  asm volatile("s_nop 0" ::: "memory");
}
